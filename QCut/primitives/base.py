"""What the cut primitives share: checking their input, building and submitting the
experiments, and working out what a run will cost."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from QCut.errors.qcuterror import QCutError
from QCut.execution.estimate import RunEstimate, _named, estimate_run
from QCut.execution.submit import submit_experiments
from QCut.job.handle import QCutJob
from QCut.model.cutcircuit import CutCircuit, CutExperiment
from QCut.transpilation.transpile_cut_circuits import has_topology


@dataclass(frozen=True, repr=False)
class CutRunEstimate(RunEstimate):
    """What a primitive's run will cost, over every circuit and for each one.

    The totals and :attr:`breakdown` cover the whole run, numbered in submission order.
    Indexing gives the :class:`~QCut.RunEstimate` of one input circuit, in the order
    the circuits were given, and ``len`` counts the circuits.
    """

    inputs: tuple[RunEstimate, ...] = ()

    def __getitem__(self, index: int) -> RunEstimate:
        """The estimate for one input circuit."""
        return self.inputs[index]

    def __len__(self) -> int:
        """How many circuits the run takes."""
        return len(self.inputs)


class _CutResult(Sequence):
    """What a primitive's run gives back: one entry per input circuit."""

    def __init__(self, entries: list) -> None:
        self._entries = list(entries)

    def __getitem__(self, index):
        """The entry of one input circuit."""
        return self._entries[index]

    def __len__(self) -> int:
        """How many circuits the run took."""
        return len(self._entries)


def _as_list(circuits: CutCircuit | list[CutCircuit]) -> list[CutCircuit]:
    """The cut circuits as a list, whether one was given or several.

    Raises:
        QCutError: something other than a cut circuit was given.
        ValueError: the list is empty.
    """
    listed = [circuits] if isinstance(circuits, CutCircuit) else list(circuits)
    if not listed:
        raise ValueError("no circuits were given")
    for index, circuit in enumerate(listed):
        if not isinstance(circuit, CutCircuit):
            raise QCutError(
                f"circuit {index} is a {type(circuit).__name__}, but the primitives "
                "take CutCircuits. Split a circuit with marked cuts with "
                "QCut.get_locations_and_subcircuits(), or find cuts with "
                "QCut.find_cuts(), and for a device with a coupling map transpile the "
                "result with QCut.transpile_subcircuits()."
            )
    return listed


def _pair(circuits, values, count: int, name: str) -> list:
    """One entry of ``values`` per circuit, broadcast the way FiQCI's primitives do it.

    One circuit takes ``values`` exactly as given. With a list of circuits, a list or
    tuple gives one entry per circuit, and anything else is used for every circuit.

    Raises:
        ValueError: the list does not have one entry per circuit.
    """
    if isinstance(circuits, CutCircuit):
        return [values]
    if isinstance(values, (list, tuple)):
        if len(values) != count:
            raise ValueError(
                f"{count} circuit(s) were given but {len(values)} {name} entries. "
                "With a list of circuits, give one entry per circuit, or a single one "
                "for all of them."
            )
        return list(values)
    return [values] * count


def _unbound_parameters(cut_circuit: CutCircuit) -> list[str]:
    """Names of the parameters left unbound, in the subcircuits or on a cut gate."""
    names = {param.name for sub in cut_circuit.subcircuits for param in sub.parameters}
    for location in cut_circuit.cut_locations:
        gate = getattr(location, "gate", None)
        for value in gate.params if gate is not None else ():
            names.update(param.name for param in getattr(value, "parameters", ()))
    return sorted(names)


class _CutPrimitive:
    """The backend a primitive runs on, and the steps every run takes: check each cut
    circuit, build its experiment, then submit them all."""

    def __init__(self, backend) -> None:
        if backend is None:
            raise QCutError("a backend or sampler to run the circuits on is needed")
        self._backend = backend

    @property
    def backend(self):
        """The backend or sampler the circuits run on."""
        return self._backend

    def _check(self, cut_circuit: CutCircuit, index: int) -> None:
        """Refuse a cut circuit that is not ready for this backend.

        The primitives never transpile. A backend with a coupling map needs the
        subcircuits transpiled for it, and for no other backend. One without, such as
        an ideal simulator or a sampler, takes them as they are.

        Raises:
            QCutError: the backend has a coupling map, and the cut circuit was not
                transpiled for it.
        """
        if not has_topology(self._backend):
            return
        if cut_circuit.backend is None:
            raise QCutError(
                f"circuit {index} is not transpiled, and "
                f"{type(self._backend).__name__} has a coupling map. Transpile it "
                "first with QCut.transpile_subcircuits(cut_circuit, backend)."
            )
        if cut_circuit.backend is not self._backend:
            raise QCutError(
                f"circuit {index} was transpiled for a different backend than the one "
                "this primitive runs on. Transpile it for this one with "
                "QCut.transpile_subcircuits(cut_circuit, backend)."
            )

    def _bind(
        self, cut_circuit: CutCircuit, parameter_values: dict | None, index: int
    ) -> CutCircuit:
        """Bind ``parameter_values`` into a cut circuit and check nothing is left.

        Raises:
            QCutError: a parameter is still unbound.
        """
        if parameter_values is not None:
            cut_circuit = cut_circuit.assign_parameters(parameter_values)
        unbound = _unbound_parameters(cut_circuit)
        if unbound:
            raise QCutError(
                f"circuit {index} has unbound parameters {unbound}. Give their values "
                "in parameter_values."
            )
        return cut_circuit

    def _bound(self, circuits, parameter_values) -> list[CutCircuit]:
        """Check every input circuit, then bind its parameter values.

        Raises:
            QCutError: a circuit is not ready for this backend, or has a parameter
                left unbound.
            ValueError: there is not one entry of ``parameter_values`` per circuit.
        """
        listed = _as_list(circuits)
        values = _pair(circuits, parameter_values, len(listed), "parameter_values")
        bound = []
        for index, (cut_circuit, entry) in enumerate(zip(listed, values)):
            self._check(cut_circuit, index)
            bound.append(self._bind(cut_circuit, entry, index))
        return bound

    def _submit(
        self,
        experiments: list[CutExperiment],
        shots: int,
        max_batch_size: int,
        run_options: dict,
    ) -> list[QCutJob]:
        """Submit every experiment, each in batches of its own.

        Every experiment is built before this is called, so a circuit that cannot be
        run fails the run before any job has been queued.

        TODO: batch circuits across experiments. Each one goes out in batches of its
        own, so many small cut circuits submit many part-filled jobs where shared
        batches would fill them. :meth:`_estimate` sums per-experiment costs and would
        have to follow.
        """
        return [
            submit_experiments(
                experiment,
                shots=shots,
                backend=self._backend,
                max_batch_size=max_batch_size,
                run_options=run_options,
            )
            for experiment in experiments
        ]

    def _estimate(
        self, experiments: list[CutExperiment], shots: int, max_batch_size: int
    ) -> CutRunEstimate:
        """Combine what each experiment will cost into one estimate for the run.

        Each experiment is submitted in batches of its own, so the run's cost is the
        sum of theirs, and this is exact wherever :func:`QCut.estimate_run` is.
        """
        inputs = tuple(
            estimate_run(
                experiment,
                shots=shots,
                backend=self._backend,
                max_batch_size=max_batch_size,
            )
            for experiment in experiments
        )
        jobs = [job for estimate in inputs for job in estimate.breakdown.values()]
        return CutRunEstimate(
            _named(jobs),
            exact=all(estimate.exact for estimate in inputs),
            inputs=inputs,
        )
