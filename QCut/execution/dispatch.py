"""
Submitting batches of experiment circuits to a backend or sampler.
"""

from __future__ import annotations

import inspect
import logging

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2

from QCut.job.cutresult import CircuitResult
from QCut.job.handle import QCutJob, _Batch, _job_id

logger: logging.Logger = logging.getLogger(__name__)


#: Widest ratio of requested shot counts allowed to share one job. Circuits in a batch
#: all run at the same number of shots, so a batch spanning a wide range starves the
#: circuits at its top end. Holding the ratio to two keeps the variance cost of that
#: within a few percent while still leaving only a handful of batches.
SHOT_SPREAD: float = 2.0


#: Shots per circuit when the caller does not say. Used by :func:`submit_experiments`
#: and every wrapper around it, so they cannot drift apart.
DEFAULT_SHOTS: int = 2**12


def _has_measurements(circuit: QuantumCircuit) -> bool:
    return "measure" in circuit.count_ops()


def _takes_max_batch_size(run) -> bool:
    """Whether this ``run`` batches on its own account, so it can be told the size."""
    try:
        return "max_batch_size" in inspect.signature(run).parameters
    except (TypeError, ValueError):
        return False


def _submit(backend, circuits, shots, max_batch_size, run_options):
    """Run one batch, whether ``backend`` is a backend, a sampler, or a wrapper.

    Both hand back a job rather than a result, so either way the caller can submit
    everything before collecting any of it.

    A wrapper that batches on its own account is given the size QCut used, since this
    batch is already sized both for the device and for the shots its circuits asked for,
    and splitting it again would only add jobs. Anything else in ``run_options`` is
    passed through untouched.
    """
    options = dict(run_options or {})
    if isinstance(backend, BaseSamplerV2):
        return backend.run([(circuit,) for circuit in circuits], shots=shots, **options)
    if "max_batch_size" not in options and _takes_max_batch_size(backend.run):
        options["max_batch_size"] = max_batch_size
    return backend.run(circuits, shots=shots, **options)


def _submit_batches(job: QCutJob, wave: int, planned) -> None:
    """Submit batches to the job's target, every one before any is collected.

    ``planned`` holds ``(circuits, targets, shots, scale)`` per batch. A target can
    refuse a batch after accepting the ones before it, and raising there would lose the
    jobs already queued. The refusal is kept on its batch instead, nothing after it is
    submitted, and the job raises it when its results are asked for.
    """
    refused = False
    for circuits, targets, shots, scale in planned:
        batch = _Batch(wave, targets, shots, scale)
        if not refused and not job._cancelled.is_set():
            try:
                batch.job = _submit(
                    job.backend,
                    circuits,
                    shots,
                    job.max_batch_size,
                    job._run_options,
                )
            except Exception as error:  # noqa: BLE001 - kept on the batch, raised later
                refused = True
                batch.error = error
                logger.warning(
                    f"The target refused a batch of {len(circuits)} circuits: {error}. "
                    "Nothing after it is submitted, and the job raises this when its "
                    "results are asked for."
                )
            else:
                logger.info(
                    f"Submitted {len(circuits)} circuits at {shots} shots each, job "
                    f"{_job_id(batch.job) or 'unknown'}"
                )
        job._add(batch)
        # Cancelled while this one was going out, so it missed the sweep.
        if job._cancelled.is_set():
            batch.cancel()


def _seeded(cut_experiment) -> list[list[dict[int, CircuitResult]]]:
    """An empty result per slot, ready for the batches to fill in.

    Seeded in subcircuit order, because the estimator reads a group's subcircuits back
    in the order these were filled, and both deduplicated circuits and the waves fill
    them out of order.
    """
    return [
        [{sub: CircuitResult({}) for sub in sorted(obs_group)} for obs_group in group]
        for group in cut_experiment.experiments
    ]


def _batches(runnable, max_batch_size):
    """Group jobs, already sorted by requested shots, into runs that can share a job.

    A batch closes when it is full or when the next circuit wants more than
    :data:`SHOT_SPREAD` times what the batch's smallest asked for. Both limits matter.
    Without the size limit a batch could exceed what the backend takes in one job, and
    without the spread limit a generous ``max_batch_size`` would put everything in one
    batch at one shot count, which is uniform allocation and throws away the whole point
    of splitting the shots in proportion to the labels.
    """
    batch = []
    for job in runnable:
        if batch and (
            len(batch) >= max_batch_size or job[2] > batch[0][2] * SHOT_SPREAD
        ):
            yield batch
            batch = []
        batch.append(job)
    if batch:
        yield batch


def _batch_shots(batch, cap) -> int:
    """The shots a batch runs at: the mean of what its own circuits asked for."""
    shots = max(1, round(sum(job[2] for job in batch) / len(batch)))
    return min(shots, cap) if cap is not None else shots


def _dispatch(job: QCutJob, wave: int, jobs, cap, results) -> None:
    """Submit one wave, and fill in the circuits with nothing to measure.

    Circuits in a wave no longer want the same number of shots, and only circuits asking
    for the same number can share a job. They are sorted and grouped by
    :func:`_batches`, and each batch runs at the mean of what its own circuits asked
    for. Sorting first is what keeps that mean close to every request in the batch.

    Each batch carries the factor that rescales its counts to the job's nominal shots
    when they are collected, so that everything downstream can keep dividing by the one
    number it was given.
    """
    job._wave = wave
    runnable = []
    for targets, circuit, wanted in jobs:
        if not _has_measurements(circuit):
            synthetic = {" " + "0" * circuit.num_clbits: job.shots}
            for group, obs, sub in targets:
                results[group][obs][sub] = CircuitResult(dict(synthetic))
            continue
        runnable.append((targets, circuit, wanted))

    runnable.sort(key=lambda entry: entry[2])

    planned = []
    for batch in _batches(runnable, job.max_batch_size):
        shots = _batch_shots(batch, cap)
        planned.append(
            (
                [circuit for _t, circuit, _w in batch],
                [targets for targets, _c, _w in batch],
                shots,
                job.shots / shots,
            )
        )
    _submit_batches(job, wave, planned)
