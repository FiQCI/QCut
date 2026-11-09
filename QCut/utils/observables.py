"""What the caller asked to measure, turned into the Pauli terms an experiment runs.

Observables are taken the way qiskit's estimator takes them: a label, a ``Pauli``, a
``SparsePauliOp``, a ``SparseObservable`` on the qiskit versions that coerce one, a
``{label: coefficient}`` mapping, or any nested sequence of those. An array of
observables comes back as an array of expectation values of the same shape, so a single
observable gives a single number and a list of ``n`` gives ``n``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp

# TODO(3.0): drop the fallback if qiskit 1.1 has the alias too; 1.2 does.
try:  # qiskit's own name for "anything an estimator takes as observables"
    from qiskit.primitives.containers.observables_array import (
        ObservablesArrayLike as ObservablesLike,
    )
except ImportError:  # pragma: no cover, older qiskit does not export the alias
    ObservablesLike = Any

# TODO(3.0): import directly once the qiskit floor is 1.4 or newer.
try:
    from qiskit.quantum_info import SparseObservable
except ImportError:  # pragma: no cover, qiskit below 1.4 has no such class
    SparseObservable = None


# TODO(3.0): always True from qiskit 2.1. Remove with _check_sparse_observables and
# the skipifs in tests/test_observables.py.
def _coerces_sparse_observables() -> bool:
    """Whether this qiskit turns a ``SparseObservable`` into observables of its own."""
    if SparseObservable is None:
        return False
    try:
        ObservablesArray.coerce(SparseObservable.from_label("I"))
    except Exception:  # noqa: BLE001 - any refusal is a refusal
        return False
    return True


#: Whether ``SparseObservable`` may be passed as an observable here. Its own qiskit
#: version decides, and QCut supports versions older than the one that enabled it.
SUPPORTS_SPARSE_OBSERVABLE = _coerces_sparse_observables()


# Every projector a ``SparseObservable`` can carry, as the Pauli it is built from and
# the sign that Pauli takes. ``|0><0| = (I + Z)/2``, ``|1><1| = (I - Z)/2``, and so on
PROJECTORS: dict[str, tuple[str, float]] = {
    "0": ("Z", 1.0),
    "1": ("Z", -1.0),
    "+": ("X", 1.0),
    "-": ("X", -1.0),
    "r": ("Y", 1.0),
    "l": ("Y", -1.0),
}

# The letters a Pauli term is written with.
PAULI_LETTERS = frozenset("IXYZ")

# Most Pauli terms one observable may expand into. A projector on ``k`` qubits is a sum
# of ``2**k`` Paulis, and while they all share a single measurement setting
# reading them back does grow with their number. Past this
# the question being asked is a distribution, and ``qubits`` answers that directly.
MAX_PROJECTOR_TERMS = 2**16


def _flatten(nested, depth: int) -> list[dict[str, float]]:
    """The elements of a ``depth``-deep nested list, in row-major order.

    ``ObservablesArray`` has no ``__len__`` and does not offer ``ravel`` in every qiskit
    QCut supports, but ``tolist`` is nested exactly as deep as the array's shape.
    """
    if depth == 0:
        return [nested]
    return [element for item in nested for element in _flatten(item, depth - 1)]


def _expand(label: str, coefficient: float) -> list[tuple[str, float]]:
    """Write one observable term as Pauli terms.

    A Pauli term is already one. A term carrying projectors is a product of them, and
    each projector is ``(I ± P)/2`` for its own Pauli ``P``, so the product multiplies
    out into one Pauli term per way of choosing ``I`` or ``P`` at each projector.

    Args:
        label (str): the term, over the alphabet qiskit accepts (``IXYZ+-rl01``).
        coefficient (float): what the observable puts on this term.

    Returns:
        list[tuple[str, float]]: the Pauli terms and their coefficients.

    Raises:
        ValueError: the label uses a letter that is neither a Pauli nor a projector, or
            it carries more projectors than :data:`MAX_PROJECTOR_TERMS` allows.
    """
    unknown = sorted(set(label) - PAULI_LETTERS - set(PROJECTORS))
    if unknown:
        raise ValueError(
            f"observable term '{label}' uses {unknown}, which is neither a Pauli "
            "(I, X, Y, Z) nor a projector (0, 1, +, -, r, l)."
        )

    projectors = [index for index, letter in enumerate(label) if letter in PROJECTORS]
    if not projectors:
        return [(label, coefficient)]

    if 1 << len(projectors) > MAX_PROJECTOR_TERMS:
        raise ValueError(
            f"observable term '{label}' projects onto {len(projectors)} qubits, which "
            f"is a sum of {1 << len(projectors)} Pauli terms. This is more than the "
            f"{MAX_PROJECTOR_TERMS} QCut will expand. Asking for a distribution over "
            "that many qubits is what the ``qubits`` argument is for. It reconstructs "
            "them all at once at no extra costs."
        )

    share = coefficient / (1 << len(projectors))
    terms = []
    for choice in range(1 << len(projectors)):
        letters = list(label)
        sign = 1.0
        for bit, index in enumerate(projectors):
            pauli, parity = PROJECTORS[label[index]]
            if (choice >> bit) & 1:
                letters[index] = pauli
                sign *= parity
            else:
                letters[index] = "I"
        terms.append(("".join(letters), sign * share))
    return terms


@dataclass(frozen=True)
class ObservableSpec:
    """The observables asked for, and the Pauli terms that answer them.

    Attributes:
        array (ObservablesArray): what the caller asked for, in qiskit's own container.
            Its shape is the shape of the returned expectation values.
        terms (SparsePauliOp): the distinct Pauli terms the circuits measure, each once.
        weights (np.ndarray | None): ``(number of observables, number of terms)``, the
            coefficient each observable puts on each term. None stands for the identity,
            one observable per term in the order the terms are held, which is what
            :func:`per_term_observables` asks for.
    """

    array: ObservablesArray
    terms: SparsePauliOp
    weights: np.ndarray | None

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the observables, and so of the expectation values."""
        return tuple(self.array.shape)

    @property
    def num_terms(self) -> int:
        """How many distinct Pauli terms the experiment estimates."""
        return len(self.terms)

    def combine(self, term_values) -> np.ndarray:
        """The observables' expectation values, from the terms' own.

        Args:
            term_values: one expectation value per entry of :attr:`terms`.

        Returns:
            np.ndarray: one expectation value per observable, shaped like
            :attr:`shape`, so a single observable gives a zero-dimensional array.

        Raises:
            ValueError: the values given are not one per term.
        """
        values = np.asarray(term_values, dtype=float)
        if values.shape != (self.num_terms,):
            raise ValueError(
                f"expected {self.num_terms} term value(s), got shape {values.shape}"
            )
        if self.weights is None:
            return values.reshape(self.shape)
        return (self.weights @ values).reshape(self.shape)


def per_term_observables(op: SparsePauliOp) -> ObservableSpec:
    """One observable per Pauli of ``op``, in the order they are given.

    Args:
        op (SparsePauliOp): the Paulis to estimate, each on its own. Its coefficients
            are not read.

    Returns:
        ObservableSpec: the terms and the weights that put them back together.
    """
    return ObservableSpec(
        ObservablesArray.coerce(op.paulis), SparsePauliOp(op.paulis), None
    )


# TODO(3.0): not needed from qiskit 2.1, see _coerces_sparse_observables.
def _check_sparse_observables(observables) -> None:
    """Refuse a ``SparseObservable`` on a qiskit that will not take one.

    Nested sequences are walked, since an array of observables may hold one anywhere.

    Args:
        observables: whatever the caller passed.

    Raises:
        ValueError: one was given and this qiskit does not coerce it.
    """
    if SparseObservable is None or SUPPORTS_SPARSE_OBSERVABLE:
        return
    if isinstance(observables, SparseObservable):
        raise ValueError(
            "this qiskit does not take a SparseObservable as an observable; it began "
            "coercing them in 2.1. Pass a SparsePauliOp, a label, or a "
            "{label: coefficient} mapping instead, or upgrade qiskit."
        )
    if isinstance(observables, (list, tuple)):
        for item in observables:
            _check_sparse_observables(item)


def coerce_observables(
    observables: ObservablesLike | ObservableSpec,  # type: ignore
    num_qubits: int | None = None,  # type: ignore
) -> ObservableSpec:
    """Take observables the way qiskit's estimator does, and find the terms behind them.

    Args:
        observables: a label, a ``Pauli``, a ``SparsePauliOp``, a ``SparseObservable``,
            a ``{label: coefficient}`` mapping, an ``ObservablesArray``, or any nested
            sequence of those. An :class:`ObservableSpec` is passed through, so an
            experiment can be rebuilt from one without redoing this.
        num_qubits (int): width of the uncut circuit. Every observable must span all of
            it.

    Returns:
        ObservableSpec: the observables, the distinct Pauli terms they are made of, and
        the weight each observable puts on each term.

    Raises:
        ValueError: an observable does not span the circuit, or there is nothing to
            measure at all.
    """
    if isinstance(observables, ObservableSpec):
        _check_width(observables.terms.paulis.to_labels(), num_qubits)
        return observables

    if isinstance(observables, ObservablesArray):
        array = observables
    else:
        _check_sparse_observables(observables)
        array = ObservablesArray.coerce(observables)

    flat = _flatten(array.tolist(), len(array.shape))
    _check_width([label for element in flat for label in element], num_qubits)

    position_of: dict[str, int] = {}
    rows: list[dict[int, float]] = []
    for element in flat:
        row: dict[int, float] = {}
        for label, coefficient in element.items():
            for term, weight in _expand(label, float(np.real(coefficient))):
                position = position_of.setdefault(term, len(position_of))
                row[position] = row.get(position, 0.0) + weight
        rows.append(row)

    if not position_of:
        raise ValueError(
            "there is nothing to measure. The observables given carry no terms."
        )

    weights = np.zeros((len(rows), len(position_of)))
    for index, row in enumerate(rows):
        for position, coefficient in row.items():
            weights[index, position] = coefficient

    return ObservableSpec(array, SparsePauliOp(list(position_of)), weights)


def _check_width(labels, num_qubits: int | None) -> None:
    """Check that every observable spans the whole uncut circuit.

    Args:
        labels: the term labels to check.
        num_qubits (int): the width they must have, or None to skip the check.

    Raises:
        ValueError: one of them has a different width.
    """
    if num_qubits is None:
        return
    for label in labels:
        if len(label) != num_qubits:
            raise ValueError(
                f"observable '{label}' spans {len(label)} qubits, but every observable "
                "must match the number of qubits in the original uncut circuit "
                f"({num_qubits})."
            )


def _combine_pauli_ops(op: SparsePauliOp) -> list[dict[int, str]]:  # noqa: C901
    """Combine Pauli operators that have no conflicting non-identity components.

    Args:
        op (SparsePauliOp): The SparsePauliOp to analyze.

    Returns:
        list[dict[int, str]]: A list of combined measurement settings, where each dict
                              maps qubit indices to Pauli basis measurements.
    """

    pauli_strings = [pauli.to_label()[::-1] for pauli in op.paulis]

    combined_settings = []
    used = [False] * len(pauli_strings)

    for i, pauli_string in enumerate(pauli_strings):
        if used[i]:
            continue

        # Start a new combined setting with the current Pauli string
        combined = {}
        for qubit_index, pauli in enumerate(pauli_string):
            if pauli != "I":
                combined[qubit_index] = pauli

        used[i] = True

        # Try to combine with remaining Pauli strings
        for j in range(i + 1, len(pauli_strings)):
            if used[j]:
                continue

            # Check if pauli_strings[j] can be combined with current combined setting
            can_combine = True
            for qubit_index, pauli in enumerate(pauli_strings[j]):
                if pauli != "I":
                    if qubit_index in combined and combined[qubit_index] != pauli:
                        can_combine = False
                        break

            # If compatible, add to combined setting
            if can_combine:
                for qubit_index, pauli in enumerate(pauli_strings[j]):
                    if pauli != "I":
                        combined[qubit_index] = pauli
                used[j] = True

        combined_settings.append(combined)

    return combined_settings


def _get_observable_circuit_index(pauli, combined: list[dict[int, str]]):
    """Find which measurement setting covers the non-identity letters of `pauli`,
    and return the indices of the qubits involved."""
    label = pauli
    non_identity = {i: p for i, p in enumerate(label) if p.to_label() != "I"}

    for idx, setting in enumerate(combined):
        # All non-identity qubits must be measured in the matching basis
        if all(setting.get(q) == p.to_label() for q, p in non_identity.items()):
            return {"circuit_index": idx, "obs_indices": list(non_identity.keys())}

    return {"circuit_index": None, "obs_indices": []}


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
