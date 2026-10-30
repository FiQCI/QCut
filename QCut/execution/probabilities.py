"""The distribution a cut experiment reconstructs over a chosen set of qubits."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import QuasiDistribution

from QCut.cutlocation import SingleQubitCutLocation
from QCut.execution.distribution import Mode, SeparableDistribution
from QCut.execution.postprocess import (
    _bit_layout,
    _outcome_rows,
    _process_results,
)
from QCut.execution.qcutresult import RawResult

#: How many bitstrings are shown without being asked for: the default of the dict views
#: and the length of a ``repr`` too wide to print whole. A distribution over more than a
#: few qubits has more entries than anyone reads, and the leading ones are what a reader
#: is normally after. Pass ``top=None`` to a view for the whole distribution.
DEFAULT_TOP = 10


def _checked_top(top: int | None) -> int | None:
    """Check the ``top`` argument the dict views take.

    Args:
        top (int | None): how many bitstrings to report, or None for all of them.

    Returns:
        int | None: ``top`` unchanged.

    Raises:
        ValueError: ``top`` is neither None nor positive.
    """
    if top is not None and top < 1:
        raise ValueError(f"top must be at least one, or None for all, got {top}")
    return top


class QuasiProbabilities(Mapping):
    """A reconstructed distribution over bitstrings.

    A mapping from bitstring to quasi-probability, so it indexes, iterates and plots
    like a dict. The values are held as a product over the subcircuits,
    which is what lets :meth:`top`, :meth:`probability_of` and :meth:`marginal` answer
    without ever building the ``2**k`` entries the mapping would have. Iterating,
    :meth:`values` and :meth:`items` do build them, once, and keep them.

    The three dict views, :meth:`quasi_probabilities`, :meth:`nearest_probabilities`
    and :meth:`counts`, report the :data:`DEFAULT_TOP` most likely bitstrings unless
    given a ``top`` of their own, since a distribution over more than a few qubits has
    more entries than is usually needed. ``top=None`` asks for all of them. Only
    :meth:`quasi_probabilities` gets out of building them all, since the other two
    project onto the nearest physical distribution first, and what that projection is
    depends on every value.

    Attributes:
        shots (int | None): what the experiment ran at, carried so :meth:`counts` has a
            scale to work with by default.
        qubits (list[int]): the qubits the distribution spans, in the order their bits
            are read back. The first is the last character of a key.
    """

    def __init__(self, data, shots: int | None = None):
        """Init from a mapping of bitstring to value, which must cover every bitstring.

        Args:
            data (Mapping[str, float]): the quasi-probabilities, keyed by bitstring.
            shots (int): what the experiment ran at, if it is known.
        """
        keys = list(data)
        width = len(keys[0]) if keys else 0
        values = np.zeros(1 << width)
        for key in keys:
            values[int(key, 2)] = data[key]
        self._values: np.ndarray | None = values
        self._distribution: SeparableDistribution | None = None
        self._width = width
        self.shots = shots
        self.qubits = list(range(width))

    @classmethod
    def from_distribution(
        cls,
        distribution: SeparableDistribution,
        shots: int | None = None,
        qubits: list[int] | None = None,
    ) -> QuasiProbabilities:
        """Wrap a separable distribution without expanding it.

        Args:
            distribution (SeparableDistribution): the reconstruction, as
                :func:`_separable_form` builds it.
            shots (int): what the experiment ran at, if it is known.
            qubits (list[int]): the qubits it spans, in the order their bits are read
                back. Defaults to naming the bits themselves.

        Returns:
            QuasiProbabilities: the distribution, nothing materialised.
        """
        self = cls.__new__(cls)
        self._values = None
        self._distribution = distribution
        self._width = distribution.width
        self.shots = shots
        self.qubits = list(range(distribution.width) if qubits is None else qubits)
        return self

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        shots: int | None = None,
        qubits: list[int] | None = None,
    ) -> QuasiProbabilities:
        """Wrap values already worked out, indexed by outcome.

        Args:
            values (np.ndarray): ``2**k`` quasi-probabilities, entry ``x`` being the
                bitstring ``format(x, f"0{k}b")``.
            shots (int): what the experiment ran at, if it is known.
            qubits (list[int]): the qubits it spans, in the order their bits are read
                back. Defaults to naming the bits themselves.

        Returns:
            QuasiProbabilities: the distribution.
        """
        self = cls.__new__(cls)
        self._values = np.asarray(values, dtype=float)
        self._distribution = None
        self._width = int(self._values.size).bit_length() - 1
        self.shots = shots
        self.qubits = list(range(self._width) if qubits is None else qubits)
        return self

    @property
    def width(self) -> int:
        """How many qubits the distribution spans."""
        return self._width

    def probabilities(self) -> np.ndarray:
        """Every value at once.

        This is the one accessor that holds ``2**width`` values. It is built on the
        first call and kept, so asking twice costs once.

        Returns:
            np.ndarray: ``2**width`` quasi-probabilities, entry ``x`` being the
            bitstring ``format(x, f"0{width}b")``.
        """
        if self._values is None:
            # One of the two is always set, whichever way the object was built.
            assert self._distribution is not None
            self._values = self._distribution.array()
        return self._values

    def _key(self, outcome: int) -> str:
        return format(outcome, f"0{self._width}b")

    def __getitem__(self, key: str) -> float:
        """The quasi-probability of one bitstring."""
        try:
            outcome = int(key, 2)
        except (TypeError, ValueError):
            raise KeyError(key) from None
        if len(key) != self._width or not 0 <= outcome < (1 << self._width):
            raise KeyError(key)
        return float(self.probabilities()[outcome])

    def __iter__(self):
        """Every bitstring, in numerical order."""
        return (self._key(outcome) for outcome in range(1 << self._width))

    def __len__(self) -> int:
        """How many bitstrings the distribution covers, which is ``2**width``."""
        return 1 << self._width

    def __repr__(self) -> str:
        """Represent as string."""
        name = type(self).__name__
        if len(self) <= DEFAULT_TOP:
            return f"{name}({dict(self)!r})"
        entries = ", ".join(
            f"{key!r}: {value!r}" for key, value in self.top(DEFAULT_TOP).items()
        )
        return f"{name}({{{entries}, ...}}, {len(self)} bitstrings)"

    def probability_of(self, bitstring: str) -> float:
        """One bitstring's quasi-probability, without building the rest.

        Costs one pass over the experiment's groups per subcircuit, so it does not grow
        with how many qubits the distribution spans.

        Args:
            bitstring (str): the bitstring, first chosen qubit last.

        Returns:
            float: its quasi-probability, which may be negative.

        Raises:
            KeyError: the bitstring is not one of this distribution's.
        """
        if self._distribution is None or self._values is not None:
            return self[bitstring]
        outcome = int(bitstring, 2)
        if len(bitstring) != self._width:
            raise KeyError(bitstring)
        return self._distribution.value(outcome)

    def top(self, count: int) -> dict[str, float]:
        """The most likely bitstrings, without building the rest.

        Args:
            count (int): how many bitstrings to return. Capped at ``2**width``.

        Returns:
            dict[str, float]: the bitstrings and their quasi-probabilities, most likely
            first. Values may be negative and are not projected -- projection needs the
            whole distribution, so it is only offered by
            :meth:`nearest_probabilities`.

        Raises:
            ValueError: ``count`` is not positive.
        """
        if count < 1:
            raise ValueError(f"count must be at least one, got {count}")

        if self._values is not None:
            values = self._values
            take = min(count, values.size)
            order = np.argpartition(-values, take - 1)[:take]
            found = sorted(
                ((int(outcome), float(values[outcome])) for outcome in order),
                key=lambda pair: -pair[1],
            )
        else:
            assert self._distribution is not None
            found = self._distribution.top(count)
        return {self._key(outcome): value for outcome, value in found}

    def marginal(self, qubits: list[int]) -> QuasiProbabilities:
        """The distribution over a subset of the qubits, summed over the rest.

        Args:
            qubits (list[int]): the qubits to keep, from :attr:`qubits`, in the order
                their bits should be read back.

        Returns:
            QuasiProbabilities: the marginal distribution.

        Raises:
            ValueError: a qubit is not one this distribution spans.
        """
        unknown = [qubit for qubit in qubits if qubit not in self.qubits]
        if unknown:
            raise ValueError(
                f"qubits {unknown} are not among the {self.qubits} this distribution "
                "spans"
            )
        bits = [self.qubits.index(qubit) for qubit in qubits]
        if self._distribution is not None:
            return QuasiProbabilities.from_distribution(
                self._distribution.marginal(bits), shots=self.shots, qubits=qubits
            )

        values = self.probabilities()
        reduced = np.zeros(1 << len(bits))
        for outcome in range(values.size):
            index = 0
            for position, bit in enumerate(bits):
                index |= ((outcome >> bit) & 1) << position
            reduced[index] += values[outcome]
        marginal = QuasiProbabilities(
            {format(o, f"0{len(bits)}b"): float(v) for o, v in enumerate(reduced)},
            shots=self.shots,
        )
        marginal.qubits = list(qubits)
        return marginal

    def quasi_probabilities(self, top: int | None = DEFAULT_TOP) -> dict[str, float]:
        """The values as reconstructed, negative ones included.

        Args:
            top (int): how many of the most likely bitstrings to return, most likely
                first. ``None`` for the whole distribution, in numerical order. Capped
                at ``2**width``. Defaults to :data:`DEFAULT_TOP`.

        Returns:
            dict[str, float]: the quasi-probabilities, keyed by bitstring.

        Raises:
            ValueError: ``top`` is neither ``None`` nor positive.
        """
        top = _checked_top(top)
        if top is None:
            return dict(self)
        return self.top(top)

    def nearest_probabilities(self, top: int | None = DEFAULT_TOP) -> dict[str, float]:
        """The closest true distribution.

        The projection is over the whole distribution however few bitstrings are asked
        for, since the mass clipped off the negative entries has to go somewhere.
        So ``top`` sets how much is reported, not how much is computed.

        With ``top=None`` the bitstrings clipped to zero are kept, so the result covers
        the same keys as the quasi-probabilities it came from.

        Args:
            top (int): how many of the most likely bitstrings to return, most likely
                first. ``None`` for all of them, in numerical order. Defaults to
                :data:`DEFAULT_TOP`.

        Returns:
            dict[str, float]: the projected distribution, keyed by bitstring.

        Raises:
            ValueError: ``top`` is neither ``None`` nor positive.
        """
        top = _checked_top(top)
        width = self._width
        nearest = QuasiDistribution(
            self.quasi_probabilities(top=None)
        ).nearest_probability_distribution()
        projected = dict.fromkeys(self, 0.0)
        projected.update(
            {format(key, f"0{width}b"): value for key, value in nearest.items()}
        )
        if top is None:
            return projected
        leading = sorted(projected.items(), key=lambda pair: -pair[1])[:top]
        return dict(leading)

    def counts(
        self, shots: int | None = None, top: int | None = DEFAULT_TOP
    ) -> dict[str, float]:
        """The distribution scaled to a shot count.

        Args:
            shots (int): what to scale by. Defaults to the shots the experiment ran at.
            top (int): how many of the most likely bitstrings to return, most likely
                first. ``None`` for all of them. Defaults to :data:`DEFAULT_TOP`.

        Returns:
            dict[str, float]: the projected distribution scaled by ``shots``.

        Raises:
            ValueError: no shot count was carried and none was given, or ``top`` is
                neither ``None`` nor positive.
        """
        shots = self.shots if shots is None else shots
        if shots is None:
            raise ValueError(
                "these results carried no shot count, so pass shots to scale by"
            )
        return {
            key: value * shots
            for key, value in self.nearest_probabilities(top=top).items()
        }


def _validate_qubits(number_of_qubits: int, qubit_indices: list[int]) -> None:
    """Check the qubits a distribution was asked for against the circuit.

    Args:
        number_of_qubits (int): width of the uncut circuit.
        qubit_indices (list[int]): the qubits to measure.

    Raises:
        ValueError: the qubits are empty, repeated, or outside the circuit.
    """
    if not qubit_indices:
        raise ValueError("qubits must name at least one qubit")
    if len(set(qubit_indices)) != len(qubit_indices):
        raise ValueError(f"qubits must not repeat, got {qubit_indices}")
    outside = [q for q in qubit_indices if not 0 <= q < number_of_qubits]
    if outside:
        raise ValueError(
            f"qubits {outside} are outside the {number_of_qubits}-qubit circuit"
        )


def _all_z_paulis_for_subset(
    number_of_qubits: int, qubit_indices: list[int]
) -> SparsePauliOp:
    """Every Pauli Z string over ``qubit_indices``, identity excluded.

    Args:
        number_of_qubits (int): width of the uncut circuit, which the strings span.
        qubit_indices (list[int]): the qubits to measure, in the order their bits are
            read back.

    Returns:
        SparsePauliOp: the ``2**k - 1`` non-identity Z strings, ordered so that the
        ``i``-th carries Z on the qubits named by the set bits of ``i``.

    Raises:
        ValueError: the qubits are empty, repeated, or outside the circuit.
    """
    _validate_qubits(number_of_qubits, qubit_indices)

    paulis = []
    for i in range(2 ** len(qubit_indices)):
        pauli_str = ["I"] * number_of_qubits
        for j, qubit_index in enumerate(qubit_indices):
            ind = number_of_qubits - qubit_index - 1
            if (i >> j) & 1:
                pauli_str[ind] = "Z"
        paulis.append("".join(pauli_str))
    return SparsePauliOp(paulis[1:])


#: One group's measured outcomes for one subcircuit, and the weight each carried.
_SparseRows = tuple[np.ndarray, np.ndarray]


def _stacked(columns: list[_SparseRows], width: int) -> Mode:
    """One mode from the same subcircuit's rows in each group.

    The groups measured overlapping but not identical sets of outcomes, so the mode
    holds their union, with a zero wherever a group did not see one.

    Args:
        columns (list[_SparseRows]): per group, the outcomes it measured and their
            weights.
        width (int): how many bits the local outcome spans.

    Returns:
        Mode: the rows any group measured, one column per group.
    """
    present = [idx for idx, _values in columns if idx.size]
    if not present:
        return Mode.empty(width, len(columns))

    idx = np.unique(np.concatenate(present))
    val = np.zeros((idx.size, len(columns)))
    for group, (group_idx, values) in enumerate(columns):
        if group_idx.size:
            val[np.searchsorted(idx, group_idx), group] = values

    keep = np.flatnonzero(np.any(val != 0.0, axis=1))
    return Mode(idx[keep], val[keep], width)


def _separable_form(result: RawResult) -> SeparableDistribution | None:
    """The distribution the results describe, in the form it is actually held in.

    Args:
        result (RawResult): results of an experiment built with ``qubits``.

    Returns:
        SeparableDistribution: the distribution over ``result.experiment.qubits``, bit
        ``j`` of an outcome being the qubit ``qubits[j]``, or None when the subcircuits
        did not cover the same bits in every group and there is no one form to hold.
    """
    width, coefficients, per_group = _group_weights(result)

    if not per_group:
        return SeparableDistribution(
            width, [list(range(width))], [Mode.empty(width, 0)], np.zeros(0)
        )

    layout = sorted(per_group[0])
    if any(sorted(modes) != layout for modes in per_group[1:]):
        return None

    return SeparableDistribution(
        width,
        [list(held) for held in layout],
        [_stacked([modes[held] for modes in per_group], len(held)) for held in layout],
        np.array(coefficients),
    )


def _group_weights(
    result: RawResult,
) -> tuple[int, list[float], list[dict[tuple[int, ...], _SparseRows]]]:
    """Per group, the weight of each outcome of the bits each subcircuit covers.

    Args:
        result (RawResult): results of an experiment built with ``qubits``.

    Returns:
        tuple[int, list[float], list[dict[tuple[int, ...], _SparseRows]]]: the width,
        one coefficient per contributing group, and that group's measured outcomes and
        their weights, by the bits they hold.
    """
    experiment = result.experiment
    qubits = list(experiment.qubits)
    processed = _process_results(result.results, result._shots, experiment.qpd_bits)

    wire_cuts = len(
        [i for i in experiment.cut_locations if isinstance(i, SingleQubitCutLocation)]
    )
    parity = float(np.power(-1, wire_cuts + 1))

    coefficients: list[float] = []
    per_group: list[dict[tuple[int, ...], _SparseRows]] = []

    for experiment_run, coefficient in zip(processed, experiment.coefficients):
        # Every Z string shares one measurement setting, so there is only ever one.
        subcircuits = experiment_run[0].subcircuits[0]
        if any(len(sub) == 0 for sub in subcircuits):
            continue
        scale = parity * coefficient
        modes: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
        for sub, (held, offsets) in zip(
            subcircuits, _bit_layout(subcircuits, qubits, experiment.map_qubit)
        ):
            idx, values = _outcome_rows(sub, offsets)
            if held:
                modes[tuple(held)] = (idx, values)
            else:
                scale *= float(values.sum())
        coefficients.append(scale)
        per_group.append(modes)

    return len(qubits), coefficients, per_group


def _dense_values(result: RawResult) -> np.ndarray:
    """The distribution built group by group, for results with no one separable form.

    Args:
        result (RawResult): results of an experiment built with ``qubits``.

    Returns:
        np.ndarray: ``2**k`` quasi-probabilities, indexed by outcome.
    """
    width, coefficients, per_group = _group_weights(result)
    accumulated = np.zeros(1 << width)
    for coefficient, modes in zip(coefficients, per_group):
        block = SeparableDistribution(
            width,
            [list(held) for held in modes],
            [
                Mode(idx, values[:, None], len(held))
                for held, (idx, values) in modes.items()
            ],
            np.array([coefficient]),
        )
        accumulated += block.array() - block.constant
    return (1.0 - accumulated.sum()) / (1 << width) + accumulated


def estimate_probabilities(result: RawResult) -> QuasiProbabilities:
    """Reconstruct the distribution over the qubits the experiment measured.

    The circuits do not grow with the number of qubits asked for. Z observables all
    commute, so they share one measurement setting and the experiment is the size it
    would have been for a single observable. Neither does the reading. What comes back
    is held as a product over the subcircuits rather than as a table, so building it
    costs nothing in ``k`` -- see :func:`_separable_form`. Only asking for every
    bitstring at once does, and :meth:`QuasiProbabilities.top`,
    :meth:`QuasiProbabilities.probability_of` and
    :meth:`QuasiProbabilities.marginal` answer without doing that.

    The bitstrings are written with the first of the chosen qubits last, so a subset
    given as ``[2, 0]`` reads qubit 2 as the rightmost character.

    Args:
        result (RawResult): results of an experiment built with ``qubits``.

    Returns:
        QuasiProbabilities: the distribution, which also offers the projected
        distribution and a scaling to counts.

    Raises:
        ValueError: the experiment was built with observables rather than ``qubits``, so
            it does not carry the full set this needs.
    """
    if not result.experiment.can_reconstruct_probabilities:
        raise ValueError(
            "Cannot reconstruct probabilities for this experiment. Pass ``qubits`` to "
            "the ``get_experiment_circuits`` function to enable this feature."
        )

    qubits = list(result.experiment.qubits)
    distribution = _separable_form(result)
    if distribution is None:
        return QuasiProbabilities.from_array(
            _dense_values(result), shots=result.shots, qubits=qubits
        )
    return QuasiProbabilities.from_distribution(
        distribution, shots=result.shots, qubits=qubits
    )
