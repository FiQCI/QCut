"""
Submitting an experiment and handing back a job without waiting for it.
"""

from __future__ import annotations

import logging

from qiskit import QuantumCircuit

from QCut.errors.qcuterror import QCutError
from QCut.execution.dispatch import (
    DEFAULT_SHOTS,
    _has_measurements,
    _seeded,
    _submit_batches,
)
from QCut.execution.waves import _submit_communicating
from QCut.job.cutresult import CircuitResult
from QCut.job.handle import QCutJob
from QCut.model.cutcircuit import CutExperiment
from QCut.utils.circuit import circuit_fingerprint
from QCut.utils.deprecation import deprecated

logger: logging.Logger = logging.getLogger(__name__)


#: One slot of an experiment: which group, which measurement setting, which subcircuit.
Slot = tuple[int, int, int]


def _runnable(cut_experiment) -> tuple[list, list]:
    """Split an experiment's slots into the ones worth submitting and the ones not.

    A QPD term that measures nothing on a subcircuit leaves a circuit with no
    measurements at all. There is nothing to run, and its counts are known: every shot
    reads zero. Those are filled in directly rather than submitted.

    Returns:
        The slots to submit, as ``(slot, circuit)``, and the empty ones, as
        ``(slot, num_clbits)``. Both in group, setting, subcircuit order.
    """
    runnable: list[tuple[Slot, QuantumCircuit]] = []
    empty: list[tuple[Slot, int]] = []
    for group_idx, circuit_group in enumerate(cut_experiment.experiments):
        for obs_idx, obs_group in enumerate(circuit_group):
            for sub_idx, subcircuit in obs_group.items():
                slot = (group_idx, obs_idx, sub_idx)
                if _has_measurements(subcircuit):
                    runnable.append((slot, subcircuit))
                else:
                    empty.append((slot, subcircuit.num_clbits))
    return runnable, empty


def _deduplicate(runnable: list, enabled: bool) -> list[tuple[list, QuantumCircuit]]:
    """Group slots filled by the same circuit, so it is only submitted once.

    A subcircuit's experiment circuit depends on the QPD terms of the cuts touching that
    subcircuit and on the measurement setting restricted to its own qubits, neither of
    which most of an experiment's groups change. The slot count grows exponentially in
    the number of pieces and the distinct count linearly, so almost all of a large run
    is the same handful of circuits submitted over and over.

    The key carries the subcircuit index as well as the fingerprint. The estimator takes
    one factor per subcircuit from each group and multiplies them, so two factors of a
    single product sharing a sample would give ``E[X**2]`` where ``E[X]**2`` is meant.
    Keying on the subcircuit makes that impossible, since a group holds each subcircuit
    once. Sharing between groups is a different matter: groups are summed rather than
    multiplied, so it leaves the estimate unbiased and moves only its variance.

    Two subcircuits of a symmetric circuit really can produce identical experiment
    circuits, so this is not a theoretical concern.

    Returns:
        ``(slots, circuit)`` per distinct circuit, in the order first seen, so two runs
        of one experiment submit the same circuits in the same order.
    """
    if not enabled:
        return [([slot], circuit) for slot, circuit in runnable]

    cache: dict = {}
    distinct: dict[tuple, tuple[list, QuantumCircuit]] = {}
    for slot, circuit in runnable:
        key = (slot[2], circuit_fingerprint(circuit, cache))
        hit = distinct.get(key)
        if hit is None:
            distinct[key] = ([slot], circuit)
        else:
            hit[0].append(slot)
    return list(distinct.values())


def _submission(cut_experiment) -> tuple[list[tuple[list, QuantumCircuit]], list]:
    """What a run of this experiment actually submits, and what it fills in itself."""
    runnable, empty = _runnable(cut_experiment)
    jobs = _deduplicate(runnable, cut_experiment.options.deduplicate)
    if len(jobs) < len(runnable):
        logger.info(
            f"{len(runnable)} experiment circuits hold {len(jobs)} distinct ones"
        )
    return jobs, empty


@deprecated(
    "submit_experiments() is planned to be removed in QCut 3.0. run_experiments() will "
    "return the QCutJob itself, and the job's result() will give expectation values "
    "or a distribution rather than raw counts. The raw counts will stay available on "
    "the job."
)
def submit_experiments(
    cut_experiment: CutExperiment,
    shots: int = DEFAULT_SHOTS,
    backend=None,
    max_batch_size: int = 100,
    run_options: dict | None = None,
) -> QCutJob:
    """Submit experiment circuits and return without waiting for their results.

    Every batch is submitted before this returns, and the returned job collects them
    when its :meth:`QCutJob.result` is asked for. An experiment whose wire cuts
    communicate runs in waves: the first is submitted before this returns, and the
    job submits the rest itself as each wave before them comes back.

    Args:
        cut_experiment (CutExperiment): the experiment circuits to run
        shots (int): number of shots per circuit run (optional). Communicating wire
            cuts spend it differently: the waves split a total of ``shots`` times the
            number of groups per subcircuit between them, in proportion to how often
            each label came up, so individual circuits run at very different counts and
            only that total is fixed. See :func:`_submit_communicating`.
        backend: backend or V2 sampler used for running the circuits.
            A sampler is run through its own interface and its results are read
            per register, so the circuits must already be in its target's basis.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call. Larger batches reduce per-job overhead on real hardware. A target that
            batches on its own account, as e.g. fiqci-ems does, is given this size
            too, so it does not split a batch QCut has already sized.
        run_options (dict): passed on to every ``run`` call. Use it for whatever the
            target takes per run, since QCut sets only ``shots`` itself.

    Returns:
        QCutJob: a handle on the submitted experiment, whose :meth:`QCutJob.result`
        returns what :func:`run_experiments` does.

    """
    if backend is None:
        raise QCutError("a backend or sampler to run the circuits on is needed")

    plan = cut_experiment.plan
    job = QCutJob(
        cut_experiment,
        backend,
        shots,
        max_batch_size,
        run_options,
        num_waves=1 if plan is None else plan.last_wave + 1,
    )

    if plan is not None:
        _submit_communicating(job)
        return job

    jobs, empty_locations = _submission(cut_experiment)

    num_batches = len(jobs) // max_batch_size + (1 if len(jobs) % max_batch_size else 0)
    logger.info(
        f"Running {len(jobs)} circuits on the backend {backend} with {shots} shots each"
    )
    logger.info(
        "Circuits will be split into "
        f"{num_batches}"
        f" batches of size {max_batch_size} for execution."
    )
    _submit_batches(
        job,
        0,
        [
            (
                [circuit for _slots, circuit in batch],
                [slots for slots, _circuit in batch],
                shots,
                1.0,
            )
            for batch in (
                jobs[start : start + max_batch_size]
                for start in range(0, len(jobs), max_batch_size)
            )
        ],
    )

    def collect():
        results = _seeded(cut_experiment)
        job._collect(results)
        for (group_idx, obs_idx, sub_idx), num_clbits in empty_locations:
            results[group_idx][obs_idx][sub_idx] = CircuitResult(
                {" " + "0" * num_clbits: shots}
            )
        return results

    job._start(collect, background=False)
    return job
