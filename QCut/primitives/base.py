"""What the cut primitives share: checking their input, building and submitting the
experiments, and working out what a run will cost."""

from __future__ import annotations

from dataclasses import dataclass

from QCut.errors.qcuterror import QCutError
from QCut.execution.estimate import RunEstimate, _named, estimate_run
from QCut.execution.experiments import get_experiment_circuits
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

    def _experiment(
        self,
        cut_circuit: CutCircuit,
        observables=None,
        qubits: list[int] | None = None,
    ) -> CutExperiment:
        """Build one cut circuit's experiment, with the options it carries."""
        return get_experiment_circuits(cut_circuit, observables, qubits)

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
