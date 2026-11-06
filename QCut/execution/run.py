"""
Running the circuit knitting sequence end to end, waiting for the result.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from QCut.errors.qcuterror import QCutError
from QCut.execution.dispatch import DEFAULT_SHOTS, SHOT_SPREAD
from QCut.execution.estimate import JobEstimate, RunEstimate, estimate_run
from QCut.execution.experiments import _check_measurement_spec, get_experiment_circuits
from QCut.execution.submit import submit_experiments
from QCut.execution.waves import MEASURE_SHARE
from QCut.job.cutresult import RawResult
from QCut.model.cutcircuit import CutCircuit, CutExperiment
from QCut.options import CutOptions
from QCut.post_process.postprocess import estimate_expectation_values
from QCut.post_process.probabilities import (
    QuasiProbabilities,
    estimate_probabilities,
)
from QCut.pre_process.circuit_preparation import get_locations_and_subcircuits
from QCut.transpilation.transpile_cut_circuits import (
    has_topology,
    transpile_subcircuits,
)
from QCut.utils.deprecation import deprecated
from QCut.utils.observables import ObservablesLike

# The old QCut.execution.circuit_knitting path resolves to this module, so the public
# names that used to live there are re-exported here too.
__all__ = [
    "run",
    "run_cut_circuit",
    "run_experiments",
    "get_experiment_circuits",
    "submit_experiments",
    "estimate_run",
    "JobEstimate",
    "RunEstimate",
    "DEFAULT_SHOTS",
    "SHOT_SPREAD",
    "MEASURE_SHARE",
]


@deprecated(
    "run_experiments() is planned to change in QCut 3.0: it will return a QCutJob "
    "without waiting, as submit_experiments() does now, and the job's result() will "
    "give expectation values or a distribution rather than a RawResult. The RawResult "
    "will stay available on the job."
)
def run_experiments(
    cut_experiment: CutExperiment,
    shots: int = DEFAULT_SHOTS,
    backend=None,
    max_batch_size: int = 100,
    run_options: dict | None = None,
) -> RawResult:
    """Run experiment circuits and wait for their results.

    The same as :func:`submit_experiments` followed by :meth:`QCutJob.result`. The
    counts come back as ``[group][observable][subcircuit]``, each key ``"xxx yy"``
    where ``xxx`` is the end-of-circuit measurements on the ``meas`` register and
    ``yy`` the QPD basis measurements on the ``qpd_meas`` register.

    Args:
        cut_experiment (CutExperiment): the experiment circuits to run
        shots (int): number of shots per circuit run (optional), as in
            :func:`submit_experiments`.
        backend: backend or V2 sampler used for running the circuits.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call, as in :func:`submit_experiments`.
        run_options (dict): passed on to every ``run`` call.

    Returns:
        RawResult:
            the raw counts, carrying the experiment they came from so that
            :func:`QCut.estimate_expectation_values` can be called on them alone

    """
    return submit_experiments(
        cut_experiment,
        shots=shots,
        backend=backend,
        max_batch_size=max_batch_size,
        run_options=run_options,
    ).result()


@deprecated(
    "run_cut_circuit() is planned to change in QCut 3.0: it will return a QCutJob "
    "without waiting, whose result() gives the expectation values or the "
    "distribution. The raw counts will stay available on the job."
)
def run_cut_circuit(
    cut_circuit: CutCircuit,
    observables: ObservablesLike | None = None,
    backend=None,
    max_batch_size: int = 100,
    options: CutOptions | None = None,
    shots: int = DEFAULT_SHOTS,
    qubits: list[int] | None = None,
    run_options: dict | None = None,
) -> np.ndarray | QuasiProbabilities:
    """After splitting the circuit run the rest of the circuit knitting sequence.

    Args:
        cut_circuit (CutCircuit): the split circuit, carrying its placeholder
            operations and cut locations
        observables: the observables to estimate, taken as qiskit's estimator takes
            them. See :func:`get_experiment_circuits`.
        backend: backend or V2 sampler to run the experiment circuits on. A device
            with a coupling map gets the subcircuits transpiled for it first. A
            sampler is not transpiled for, so it must take the circuits as built.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call (optional)
        options (CutOptions): configuration, overriding what the CutCircuit carries
            (optional)
        shots (int): number of shots per circuit run, as in :func:`submit_experiments`
            (optional)
        qubits (list[int]): the qubits to reconstruct a distribution over, instead of
            estimating observables (optional)
        run_options (dict): passed on to the backend or sampler, as in
            :func:`submit_experiments` (optional)

        One of observables or qubits must be provided.

    Returns:
        np.ndarray | QuasiProbabilities: one expectation value per observable, shaped
        like the observables given. A single observable comes back as a
        zero-dimensional array or, when given ``qubits``, the reconstructed
        distribution over them.

    """
    _check_measurement_spec(observables, qubits)

    if options is not None:
        cut_circuit.options = options

    if backend is None:
        raise QCutError("a backend or sampler to run the circuits on is needed")

    if has_topology(backend):
        cut_circuit = transpile_subcircuits(cut_circuit, backend, optimization_level=3)

    cut_experiment = get_experiment_circuits(cut_circuit, observables, qubits)

    results = submit_experiments(
        cut_experiment,
        shots=shots,
        backend=backend,
        max_batch_size=max_batch_size,
        run_options=run_options,
    ).result()

    if qubits is not None:
        return estimate_probabilities(results)

    return estimate_expectation_values(results)


@deprecated(
    "run() is planned to change in QCut 3.0: it will return a QCutJob without "
    "waiting, whose result() gives the expectation values or the distribution. The "
    "raw counts will stay available on the job."
)
def run(
    circuit: QuantumCircuit,
    observables: ObservablesLike | None = None,
    backend=None,
    max_batch_size: int = 100,
    options: CutOptions | None = None,
    shots: int = DEFAULT_SHOTS,
    qubits: list[int] | None = None,
    run_options: dict | None = None,
) -> np.ndarray | QuasiProbabilities:
    """Run the whole circuit knitting sequence with one function call.

    Args:
        circuit (QuantumCircuit): circuit with cut experiments
        observables: the observables to estimate, taken as qiskit's estimator takes
            them. See :func:`get_experiment_circuits`.
        backend: backend or V2 sampler to run the experiment circuits on. A device
            with a coupling map gets the subcircuits transpiled for it first. A
            sampler is not transpiled for, so it must take the circuits as built.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call (optional)
        options (CutOptions): configuration for the run (optional)
        shots (int): number of shots per circuit run, as in :func:`submit_experiments`
            (optional)
        qubits (list[int]): the qubits to reconstruct a distribution over, instead of
            estimating observables (optional)
        run_options (dict): passed on to the backend or sampler, as in
            :func:`submit_experiments` (optional)

        One of observables or qubits must be provided.

    Returns:
        np.ndarray | QuasiProbabilities: one expectation value per observable, shaped
        like the observables given.A single observable comes back as a
        zero-dimensional array or, when given ``qubits``, the reconstructed
        distribution over them.

    """
    _check_measurement_spec(observables, qubits)

    # circuit = circuit.copy()
    cut_circuit = get_locations_and_subcircuits(circuit, options=options)

    return run_cut_circuit(
        cut_circuit,
        observables,
        backend,
        max_batch_size,
        shots=shots,
        qubits=qubits,
        run_options=run_options,
    )
