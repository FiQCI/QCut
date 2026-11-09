"""Class for nicely representing a cut circuit/experiment. Also implements
some of the same functionality as the qiskit QuantumCircuit class for
a group of circuits."""

from __future__ import annotations

import copy
from typing import Iterable, Literal, overload

from qiskit import QuantumCircuit
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp

from QCut.model.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.options import CutOptions, resolve
from QCut.utils.observables import (
    ObservablesLike,
    ObservableSpec,
    _all_z_paulis_for_subset,
    coerce_observables,
    per_term_observables,
)


# TODO(3.0): use assign_parameters(strict=False) directly once the floor is qiskit 2.1.
def _assign_present(circuit: QuantumCircuit, parameters: dict) -> QuantumCircuit:
    """Bind the parameters ``circuit`` has and ignore the rest.

    Some qiskit releases (1.2, 2.0) still raise for an unknown name even with
    ``strict=False``.
    """
    present = {
        key: value
        for key, value in parameters.items()
        if not isinstance(key, str) or circuit.has_parameter(key)
    }
    return circuit.assign_parameters(present, strict=False)


class CutCircuit:
    """Class for representing a cut circuit. Contains the subcircuits, cut locations,
    and mapping of qubits. Also contains some of the same functionality as the qiskit
    QuantumCircuit class for a group of circuits."""

    def __init__(
        self,
        subcircuits: list[QuantumCircuit],
        cut_locations: list[CutLocation | SingleQubitCutLocation],
        map_qubit: dict[int, int],
        uncut_num_qubits: int,
        backend=None,
        options: CutOptions | None = None,
        final_measurements: tuple[tuple[int, int], ...] = (),
        uncut_num_clbits: int = 0,
    ) -> None:
        """Init.

        ``final_measurements`` and ``uncut_num_clbits`` remember what the uncut circuit
        measured at its end, since splitting strips those measurements: one
        ``(qubit, clbit)`` pair per measurement, as indices into the uncut circuit, and
        how many classical bits it had.
        """

        self.subcircuits = subcircuits
        self.cut_locations = cut_locations
        self.map_qubit = map_qubit
        self.uncut_num_qubits = uncut_num_qubits
        self.backend = backend
        self.options = resolve(options)
        self.final_measurements = tuple(final_measurements)
        self.uncut_num_clbits = uncut_num_clbits
        self._gamma: tuple[float, float] | None = None

    def _costs(self) -> tuple[float, float]:
        """Both overheads, computed once. Closed form, so no circuits are built."""
        if self._gamma is None:
            from QCut.qpd.qpd_operations import plan_cost

            # Everything the decompositions can do, so the gap to gamma shows what a
            # backend's topology or a switched-off option costs. consolidate is not
            # forced: it decides which gates exist, and by now that has happened.
            best = self.options.replace(
                joint_rotation_cuts=True, wire_cut_communication="always"
            )
            self._gamma = (
                plan_cost(self, self.options),
                plan_cost(self, best, respect_backend=False),
            )
        return self._gamma

    def _replace(self, **changes) -> CutCircuit:
        """A copy with ``changes`` applied, its cost worked out again when asked for."""
        new = copy.copy(self)
        new.__dict__.update(changes)
        new._gamma = None
        return new

    @property
    def gamma(self) -> float:
        """Sampling overhead of this split, as it will actually be run."""
        return self._costs()[0]

    @property
    def optimal_gamma(self) -> float:
        """The least these same cuts could cost with every decomposition available."""
        return self._costs()[1]

    @overload
    def assign_parameters(
        self, parameters: dict, inplace: Literal[False] = ...
    ) -> CutCircuit: ...

    @overload
    def assign_parameters(self, parameters: dict, inplace: Literal[True]) -> None: ...

    def assign_parameters(
        self, parameters: dict, inplace: bool = False
    ) -> CutCircuit | None:
        """Assign parameters to the circuits. Same as qiskit
        QuantumCircuit.assign_parameters.

        Parameters on a cut gate are bound too. Those live on the cut location rather
        than in any subcircuit, but a generated QPD needs them numeric.

        Every subcircuit is given all of ``parameters``; any it does not contain are
        ignored.
        """
        bound_locations = [
            location.assign_parameters(parameters)
            if isinstance(location, CutLocation)
            else location
            for location in self.cut_locations
        ]

        bound = [_assign_present(circuit, parameters) for circuit in self.subcircuits]
        if inplace:
            self.subcircuits[:] = bound
            self.cut_locations = bound_locations
            return None
        return self._replace(subcircuits=bound, cut_locations=bound_locations)

    @property
    def num_qubits(self):
        """Number of qubits per subcircuit."""
        return [i.num_qubits for i in self.subcircuits]

    @property
    def num_subcircuits(self):
        """Total number of circuits."""
        return len(self.subcircuits)


class CutExperiment:
    def __init__(
        self,
        experiment_circuits: list[list[dict[int, QuantumCircuit]]],
        cut_locations: list[CutLocation | SingleQubitCutLocation],
        map_qubit: dict[int, int],
        coefficients: Iterable[float],
        can_reconstruct_probabilities: bool = False,
        observables: ObservablesLike | ObservableSpec | None = None,
        qubits: list[int] | None = None,
        backend=None,
        options: CutOptions | None = None,
        num_draws: int | None = None,
        plan=None,
        qpd_bits: dict[tuple[int, int, int], tuple[int, int]] | None = None,
        gamma: float | None = None,
        optimal_gamma: float | None = None,
        uncut_num_qubits: int | None = None,
    ) -> None:
        """Init.

        ``gamma`` and ``optimal_gamma`` come from the :class:`CutCircuit` these circuits
        were built from, which is the only thing that can work them out: they are read
        off the subcircuits, and an experiment does not keep those.

        ``num_draws`` records how many samples were drawn when the decomposition was
        sampled rather than enumerated. It is informational. The estimator does not need
        it, because the sampled coefficients already carry their multiplicity.

        ``plan`` is a :class:`QCut.qpd_locc.CommunicationPlan` when any wire cut
        exchanges its measured outcome, and None otherwise. Those experiments run in two
        phases, so execution needs to know which bits carry the outcome.

        ``qpd_bits`` says, per circuit, how many qpd measurement bits it writes and how
        many were dropped for going unwritten. Post-processing needs both: the first to
        find those bits without relying on how a backend reports its registers, and the
        second to restore the sign the dropped ones carried.

        ``uncut_num_qubits`` is the width of the circuit before it was cut, which the
        observables of a ``qubits`` experiment span. Only that width is kept, since the
        observables themselves are built on demand -- see :attr:`observables`.

        ``observables`` is taken the way qiskit's estimator takes them, and is kept as
        an :class:`QCut.utils.observables.ObservableSpec`.
        """

        self.experiments = experiment_circuits
        self.backend = backend
        self.cut_locations = cut_locations
        self.map_qubit = map_qubit
        self.coefficients = coefficients
        self._observables = (
            None if observables is None else coerce_observables(observables)
        )
        self.qubits = qubits
        self._uncut_num_qubits = uncut_num_qubits
        self.options = resolve(options)
        self._num_draws = num_draws
        self.plan = plan
        self.qpd_bits = qpd_bits or {}
        self._gamma = gamma
        self._optimal_gamma = optimal_gamma
        self._can_reconstruct_probabilities = can_reconstruct_probabilities

    @property
    def observable_spec(self) -> ObservableSpec | None:
        """The observables and their terms together, with the weights between them.

        An experiment built from ``qubits`` is built for every Pauli Z over them, and
        there are ``2**k`` of those. They all share one measurement setting, so none of
        them shapes a circuit, and reconstructing the distribution does not go through
        them either, see :func:`QCut.post_process.probabilities._separable_form`. So
        they are built here, on demand, for whoever does want them: estimating them one
        at a time with :func:`QCut.estimate_expectation_values` still works.

        Raises:
            ValueError: the experiment was given ``qubits`` but not the width they span,
                so the observables cannot be reconstructed.
        """
        if self._observables is None and self.qubits is not None:
            if self._uncut_num_qubits is None:
                raise ValueError(
                    "this experiment carries qubits but not the width of the circuit "
                    "they came from, so its observables cannot be built. Pass "
                    "uncut_num_qubits, or the observables themselves."
                )
            self._observables = per_term_observables(
                _all_z_paulis_for_subset(self._uncut_num_qubits, self.qubits)
            )
        return self._observables

    @property
    def observables(self) -> ObservablesArray | None:
        """The observables this experiment estimates, as qiskit's own container.

        Its shape is the shape of the expectation values that come back, so a single
        observable is a zero-dimensional array and a list of ``n`` is ``(n,)``. Note
        that this is not what the circuits measure. See :attr:`observable_terms`.
        """
        spec = self.observable_spec
        return None if spec is None else spec.array

    @property
    def observable_terms(self) -> SparsePauliOp | None:
        """The distinct Pauli terms the circuits measure, each appearing once.

        An observable is a weighted sum of these, and separate observables share them,
        so there are usually fewer terms than observables asked for.
        """
        spec = self.observable_spec
        return None if spec is None else spec.terms

    def assign_parameters(
        self, parameters: dict, inplace=False
    ) -> CutExperiment | None:
        """Assign parameters to the circuits. Same as qiskit
        QuantumCircuit.assign_parameters.

        Every circuit is given all of ``parameters``; any it does not contain are
        ignored.
        """
        if inplace:
            for group in self.experiments:
                for circuits in group:
                    for ind, circuit in circuits.items():
                        circuits[ind] = _assign_present(circuit, parameters)
            return None
        return self._replace(
            experiments=[
                [
                    {ind: _assign_present(c, parameters) for ind, c in circuits.items()}
                    for circuits in group
                ]
                for group in self.experiments
            ]
        )

    def _replace(self, **changes) -> CutExperiment:
        """A copy with ``changes`` applied."""
        new = copy.copy(self)
        new.__dict__.update(changes)
        return new

    @property
    def num_qubits(self):
        """Number of qubits per subcircuit."""
        return [i.num_qubits for i in self.experiments[0][0].values()]

    @property
    def num_circuits(self):
        """Total number of circuits."""
        return sum(
            len(subcircuits) * len(subcircuits[0]) for subcircuits in self.experiments
        )

    @property
    def group_size(self):
        """Number of circuits in a group, which is the same for every group."""
        return self.num_circuits // self.num_groups

    @property
    def num_groups(self):
        """Number of circuit groups."""
        return len(self.experiments)

    @property
    def num_draws(self):
        """How many samples were drawn, or the group count if fully enumerated."""
        if self._num_draws is None:
            return self.num_groups
        return self._num_draws

    @property
    def communicates(self):
        """Whether any wire cut exchanges its outcome between the partitions."""
        return self.plan is not None

    @property
    def sampled(self):
        """Whether the decomposition was sampled rather than fully enumerated."""
        return self._num_draws is not None

    @property
    def seed(self) -> int | None:
        """The seed the decomposition was sampled with, ``None`` if it was enumerated.

        Pass it as ``CutOptions(seed=...)`` to build the same experiment again.
        """
        return self.options.seed if self.sampled else None

    @property
    def num_obs_groups(self):
        return len(self.experiments[0])

    @property
    def gamma(self) -> float | None:
        """Sampling overhead of the decomposition these circuits came from.

        None only for an experiment built by hand rather than by
        :func:`QCut.get_experiment_circuits`, which has no split to read it off.
        """
        return self._gamma

    @property
    def optimal_gamma(self) -> float | None:
        """The least these cuts could have cost with every decomposition available."""
        return self._optimal_gamma

    @property
    def can_reconstruct_probabilities(self) -> bool:
        """Whether these circuits carry the observables a distribution needs.

        True only when the experiment was built from ``qubits`` rather than from
        observables of the caller's own, since reconstructing a distribution needs every
        Pauli Z over those qubits and nothing less will do.
        """
        return self._can_reconstruct_probabilities
