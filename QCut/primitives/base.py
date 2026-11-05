"""What the cut primitives share: checking circuits, building and submitting their
experiments, and working out what a run will cost."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp

from QCut.errors.qcuterror import QCutError
from QCut.execution.estimate import RunEstimate, _named, estimate_run
from QCut.execution.experiments import get_experiment_circuits
from QCut.execution.submit import submit_experiments
from QCut.job.handle import QCutJob
from QCut.model.cutcircuit import CutExperiment
from QCut.options import CutOptions
from QCut.pre_process.circuit_preparation import get_locations_and_subcircuits
from QCut.transpilation.transpile_cut_circuits import (
    has_topology,
    transpile_subcircuits,
)


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


def _as_list(circuits: QuantumCircuit | list[QuantumCircuit]) -> list[QuantumCircuit]:
    """The circuits as a list, whether one was given or several."""
    if isinstance(circuits, QuantumCircuit):
        return [circuits]
    circuits = list(circuits)
    if not circuits:
        raise ValueError("no circuits were given")
    for index, circuit in enumerate(circuits):
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                f"circuit {index} is a {type(circuit).__name__}, not a QuantumCircuit"
            )
    return circuits


def _has_cut_markers(circuit: QuantumCircuit) -> bool:
    """Whether the circuit carries a cut, found the way circuit preparation finds it."""
    prepared = circuit.decompose(["CutGate"])
    return any("Cut" in instruction.operation.name for instruction in prepared.data)


def _check_circuit(circuit: QuantumCircuit, index: int) -> None:
    """Refuse what the primitives cannot run yet, before anything is submitted.

    Raises:
        QCutError: the circuit has no cut markers, or measures or branches on
            classical bits anywhere but at its end.
    """
    if not _has_cut_markers(circuit):
        raise QCutError(
            f"circuit {index} has no cut markers. The primitives only run circuits "
            "that are already cut: place cuts with QCut.cut() and QCut.cutGate(), or "
            "find them with QCut.find_cuts()."
        )
    remaining = circuit.remove_final_measurements(inplace=False)
    for instruction in remaining.data:
        if instruction.clbits or isinstance(instruction.operation, ControlFlowOp):
            raise QCutError(
                f"circuit {index} measures or branches on classical bits before its "
                f"end ('{instruction.operation.name}'). Mid-circuit measurements and "
                "classical control are not supported by the primitives."
            )


class _CutPrimitive:
    """The backend and options a primitive was made with, and the steps every run
    takes: build each circuit's experiment, then submit them all."""

    def __init__(self, backend, options: CutOptions | None = None) -> None:
        if backend is None:
            raise QCutError("a backend or sampler to run the circuits on is needed")
        self._backend = backend
        self._options = options

    @property
    def backend(self):
        """The backend or sampler the circuits run on."""
        return self._backend

    @property
    def options(self) -> CutOptions | None:
        """The cutting options every run uses unless it is given its own."""
        return self._options

    def _experiment(
        self,
        circuit: QuantumCircuit,
        options: CutOptions | None,
        observables=None,
        qubits: list[int] | None = None,
    ) -> CutExperiment:
        """Cut one circuit, fit it to the backend, and build its experiment.

        The same steps :func:`QCut.run_cut_circuit` takes, short of running anything.
        """
        cut_circuit = get_locations_and_subcircuits(
            circuit, options=options if options is not None else self._options
        )
        if has_topology(self._backend):
            cut_circuit = transpile_subcircuits(
                cut_circuit, self._backend, optimization_level=3
            )
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
        cut fails the run before any job has been queued.
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
