"""A handle on a submitted experiment, handed back before its results are in."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from qiskit.providers import JobStatus
from qiskit.providers.exceptions import JobTimeoutError

from QCut.errors.qcuterror import QCutError
from QCut.execution.qcutresult import CircuitResult, RawResult

logger: logging.Logger = logging.getLogger(__name__)

#: How long to wait before asking for a job's results a second time.
RESULT_RETRY_DELAY: float = 5.0

_FINAL = frozenset({JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED})

#: Most advanced first, which is the one an aggregate status reports while work remains.
_ACTIVE = (
    JobStatus.RUNNING,
    JobStatus.VALIDATING,
    JobStatus.QUEUED,
    JobStatus.INITIALIZING,
)


def _job_id(job) -> str | None:
    """A job's id, or ``None`` when it has none to give."""
    identifier = getattr(job, "job_id", None)
    try:
        identifier = identifier() if callable(identifier) else identifier
    except Exception:  # noqa: BLE001 - an id is for reporting, never worth failing over
        return None
    return str(identifier) if identifier is not None else None


def _job_ids(job) -> list[str | None]:
    """Every backend job id behind one ``run`` call.

    A target that batches on its own account, as fiqci-ems does, hands back a single
    handle over several backend jobs and lists them through ``job_ids``.
    """
    listed = getattr(job, "job_ids", None)
    if callable(listed):
        try:
            return [None if one is None else str(one) for one in listed()]
        except Exception:  # noqa: BLE001 - fall back to the one id the handle gives
            pass
    return [_job_id(job)]


def _status(job) -> JobStatus:
    """A job's status as a :class:`JobStatus`, whatever the target answered with."""
    status = getattr(job, "status", None)
    if status is None:
        return JobStatus.RUNNING
    status = status()
    if isinstance(status, JobStatus):
        return status
    # Some targets answer with the status' name rather than qiskit's enum.
    try:
        return JobStatus[str(getattr(status, "name", status)).upper()]
    except KeyError:
        return JobStatus.INITIALIZING


def _result(job):
    """A job's result, asked for twice if the results are not ready yet."""
    try:
        return job.result()
    except Exception as error:
        if "no results" not in str(error).lower():
            raise
        logger.info(f"Results not ready yet, asking again in {RESULT_RETRY_DELAY}s")
        time.sleep(RESULT_RETRY_DELAY)
        return job.result()


def _aggregate(statuses: list[JobStatus]) -> JobStatus:
    """Collapse per-batch statuses into one.

    A failure anywhere decides it, then a cancellation. Otherwise the experiment is done
    only once every batch is, and until then reports the most advanced state any batch
    has reached, so a run with one batch executing reads as running.
    """
    if JobStatus.ERROR in statuses:
        return JobStatus.ERROR
    if JobStatus.CANCELLED in statuses:
        return JobStatus.CANCELLED
    for active in _ACTIVE:
        if active in statuses:
            return active
    return JobStatus.DONE


class _Batch:
    """One ``run`` call: the job it returned and the slots its circuits fill.

    A batch the target refused holds the error instead of a job, and one skipped
    because an earlier batch was refused, or the experiment cancelled, holds neither.
    """

    def __init__(self, wave: int, targets: list, shots: int, scale: float) -> None:
        self.wave = wave
        self.targets = targets  # per circuit, the slots its counts go to
        self.shots = shots
        self.scale = scale
        self.job = None
        self.error: Exception | None = None
        self._result = None
        self._final: JobStatus | None = None
        self._lock = threading.Lock()

    def status(self) -> JobStatus:
        """The batch's status. A final one is kept, since asking can cost a request."""
        if self._final is not None:
            return self._final
        if self.job is None:
            return JobStatus.ERROR if self.error is not None else JobStatus.CANCELLED
        status = _status(self.job)
        if status in _FINAL:
            self._final = status
        return status

    def result(self):
        """The job's result, fetched once and kept."""
        if self.job is None:
            if self.error is not None:
                raise self.error
            raise QCutError(
                "this batch was never submitted, because the experiment was cancelled "
                "or a batch before it was refused"
            )
        with self._lock:
            if self._result is None:
                self._result = _result(self.job)
                self._final = JobStatus.DONE
        return self._result

    def scatter(self, results) -> None:
        """Put each circuit's result in every slot it fills."""
        result = self.result()
        for index, slots in enumerate(self.targets):
            # One result object for the circuit, shared by every slot it fills. It is
            # frozen and holds the job's result by reference, so there is nothing to
            # copy and nothing that could be mutated through one slot.
            circuit_result = CircuitResult(result, index, self.scale)
            for group, obs, sub in slots:
                results[group][obs][sub] = circuit_result

    def cancel(self) -> None:
        if self.job is None or self.status() in _FINAL:
            return
        try:
            self.job.cancel()
        except Exception as error:  # noqa: BLE001 - one job refusing must not stop the rest
            logger.warning(f"Could not cancel job {_job_id(self.job)}: {error}")


class PartialBatch(NamedTuple):
    """One ``run`` call of a submitted experiment, as it stands.

    Attributes:
        index: position of the batch in submission order.
        wave: the wave it belongs to. Zero unless the experiment's wire cuts
            communicate.
        job_id: the backend's id for it, ``None`` if it was never submitted.
        status: its current :class:`~qiskit.providers.JobStatus`.
        circuits: how many circuits it carries.
        shots: the shots each of them runs at.
        result: what the backend returned, once the batch is done, else ``None``.
    """

    index: int  # type: ignore[bad-override]  # shadows tuple.index
    wave: int
    job_id: str | None
    status: JobStatus
    circuits: int
    shots: int
    result: object | None


class QCutJob:
    """A submitted experiment, handed back before any of its results are in.

    Shaped like a qiskit job: :meth:`job_ids` and :meth:`status` can be read at once,
    and :meth:`result` waits for everything and returns the same :class:`RawResult`
    :func:`QCut.run_experiments` does. An experiment is usually several ``run`` calls,
    so the per-batch view is there too, in :meth:`statuses` and :meth:`partial_results`.

    An experiment whose wire cuts communicate runs in waves, and every wave after the
    first spends its shots on what the one before it measured. A thread of the job's
    own collects each wave and submits the next as soon as it can, so such a job comes
    back as quickly as any other and does not sit idle waiting to be polled. Its job
    ids and batches grow as the waves go out, and it reports itself running until the
    last one is in.

    Submission is not atomic: a target can refuse a batch after accepting the ones
    before it. Raising there would lose the jobs already queued, so the job is handed
    back regardless, reports ``ERROR``, and raises the refusal from :meth:`result`.
    """

    def __init__(
        self,
        experiment,
        backend,
        shots: int,
        max_batch_size: int,
        run_options: dict | None,
        num_waves: int = 1,
    ) -> None:
        self._experiment = experiment
        self._backend = backend
        self._shots = shots
        self._max_batch_size = max_batch_size
        self._run_options = dict(run_options or {})
        self._num_waves = num_waves
        self._wave = 0
        self._batches: list[_Batch] = []
        self._lock = threading.Lock()
        self._finishing = threading.Lock()
        self._cancelled = threading.Event()
        self._finish: Callable[[], list] | None = None
        self._thread: threading.Thread | None = None
        self._results: list | None = None
        self._error: Exception | None = None
        self._raw: RawResult | None = None

    # -- what was submitted ------------------------------------------------------- #

    @property
    def experiment(self):
        """The experiment that was submitted."""
        return self._experiment

    @property
    def backend(self):
        """The backend or sampler it was submitted to."""
        return self._backend

    @property
    def shots(self) -> int:
        """Shots per circuit, as given. What the estimator divides by."""
        return self._shots

    @property
    def max_batch_size(self) -> int:
        """The most circuits submitted in one ``run`` call."""
        return self._max_batch_size

    @property
    def run_options(self) -> dict:
        """What was passed on to every ``run`` call, as it was at submission."""
        return dict(self._run_options)

    @property
    def num_waves(self) -> int:
        """How many waves it runs in. One unless its wire cuts communicate."""
        return self._num_waves

    @property
    def wave(self) -> int:
        """The wave most recently submitted, counting from zero."""
        return self._wave

    @property
    def num_circuits(self) -> int:
        """How many circuits have been submitted so far."""
        return sum(len(b.targets) for b in self._snapshot() if b.job is not None)

    @property
    def total_shots(self) -> int:
        """How many shots have been submitted so far, over every circuit.

        Matches :func:`QCut.estimate_run` exactly for an experiment that runs in one
        go. One whose wire cuts communicate decides its later waves' shots from what
        the earlier ones measured, so this is what it actually spent.
        """
        return sum(
            len(b.targets) * b.shots for b in self._snapshot() if b.job is not None
        )

    # -- identity and status ------------------------------------------------------ #

    def job_id(self) -> str | None:
        """The first backend job's id, ``None`` if nothing was submitted."""
        ids = self.job_ids()
        return ids[0] if ids else None

    def job_ids(self) -> list[str | None]:
        """Every backend job's id, in submission order.

        A batch that was never submitted has ``None``. A target that splits a batch
        into several jobs of its own, as fiqci-ems does, contributes all of them.
        """
        ids: list[str | None] = []
        for batch in self._snapshot():
            ids.extend(_job_ids(batch.job) if batch.job is not None else [None])
        return ids

    def statuses(self) -> list[JobStatus]:
        """Each batch's current status, in submission order."""
        return [batch.status() for batch in self._snapshot()]

    def status(self) -> JobStatus:
        """One status for the whole experiment.

        ``DONE`` only once every wave has been submitted and collected and every batch
        is done. A batch failing, or the waves stopping on an error, makes it
        ``ERROR``.
        """
        if self._cancelled.is_set():
            return JobStatus.CANCELLED
        if self._error is not None:
            return JobStatus.ERROR
        status = _aggregate(self.statuses())
        if status == JobStatus.DONE and self._busy():
            return JobStatus.RUNNING
        return status

    def done(self) -> bool:
        """Whether the experiment finished successfully."""
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Whether any of it is running, or waves remain to submit."""
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """Whether the experiment was cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """Whether the experiment is done, failed or cancelled."""
        return self.status() in _FINAL

    def wait_for_final_state(
        self, timeout: float | None = None, wait: float = 5.0
    ) -> None:
        """Poll until the experiment reaches a final state.

        Args:
            timeout (float): seconds to wait before giving up, or ``None`` to wait as
                long as it takes.
            wait (float): seconds between polls.

        Raises:
            JobTimeoutError: the experiment was not finished within ``timeout``.
        """
        start = time.monotonic()
        while not self.in_final_state():
            elapsed = time.monotonic() - start
            if timeout is not None and elapsed >= timeout:
                raise JobTimeoutError(
                    f"the experiment was not finished after {elapsed:.1f}s"
                )
            time.sleep(wait if timeout is None else min(wait, timeout - elapsed))

    def cancel(self) -> None:
        """Cancel every backend job not yet finished, and submit no further waves."""
        self._cancelled.set()
        for batch in self._snapshot():
            batch.cancel()

    # -- results ------------------------------------------------------------------ #

    def partial_results(self) -> list[PartialBatch]:
        """Each batch as it stands, with its result once it is done.

        The slots of the experiment only fill in once every batch is in, so this is
        the place to look at what came back while the rest is still running.
        """
        snapshots = []
        for index, batch in enumerate(self._snapshot()):
            status = batch.status()
            result = None
            if status == JobStatus.DONE:
                try:
                    result = batch.result()
                except Exception:  # noqa: BLE001 - reported as not in yet
                    result = None
            snapshots.append(
                PartialBatch(
                    index,
                    batch.wave,
                    _job_id(batch.job) if batch.job is not None else None,
                    status,
                    len(batch.targets),
                    batch.shots,
                    result,
                )
            )
        return snapshots

    def result(self, timeout: float | None = None) -> RawResult:
        """Wait for every batch and return the raw results, computed once and kept.

        Args:
            timeout (float): seconds to wait before giving up, or ``None`` to wait as
                long as it takes.

        Returns:
            RawResult: the raw counts, carrying the experiment they came from, exactly
            as :func:`QCut.run_experiments` returns them.

        Raises:
            JobTimeoutError: the experiment was not finished within ``timeout``.
            QCutError: the experiment was cancelled.
            Exception: whatever a refused batch, a failed job or a failed wave raised.
        """
        if self._raw is not None:
            return self._raw
        if timeout is not None:
            # Collecting on a thread lets the wait be cut short without polling the
            # target, and whatever came in is kept for the next call.
            self._launch()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise JobTimeoutError(
                    f"the experiment was not finished after {timeout}s, in wave "
                    f"{self._wave + 1} of {self._num_waves}"
                )

        with self._finishing:
            if self._raw is not None:
                return self._raw
            if self._cancelled.is_set():
                raise QCutError("the experiment was cancelled") from self._error
            if self._error is not None:
                raise self._error
            results = self._results if self._thread is not None else None
            if results is None:
                assert self._finish is not None
                results = self._finish()
            self._raw = RawResult(results, self._shots, self._experiment)
        return self._raw

    def __repr__(self) -> str:
        """Represent without asking the backend anything."""
        batches = self._snapshot()
        submitted = sum(1 for batch in batches if batch.job is not None)
        line = f"<QCutJob {submitted} of {len(batches)} batch(es) submitted"
        if self._num_waves > 1:
            line += f", wave {self._wave + 1} of {self._num_waves}"
        return line + ">"

    # -- driven by the submission code -------------------------------------------- #

    def _snapshot(self) -> list[_Batch]:
        with self._lock:
            return list(self._batches)

    def _add(self, batch: _Batch) -> None:
        with self._lock:
            self._batches.append(batch)

    def _busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _collect(self, results, wave: int | None = None) -> None:
        """Wait for a wave's batches, or all of them, and put their results in place.

        A refused batch is raised before anything is waited for, so a run that cannot
        complete says so at once rather than after the batches before it come back.
        """
        batches = [b for b in self._snapshot() if wave is None or b.wave == wave]
        for batch in batches:
            if batch.job is None and batch.error is not None:
                raise batch.error
        for batch in batches:
            batch.scatter(results)

    def _start(self, finish: Callable[[], list], background: bool) -> None:
        """Hand over what turns the batches into results.

        In the background it runs at once, on a thread of its own, and has the
        results ready when they are asked for. Otherwise it runs on the first call
        to :meth:`result`, on that thread if the call has a timeout.
        """
        self._finish = finish
        if background:
            self._launch()

    def _launch(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._drive, name="qcut-collect", daemon=True
                )
                self._thread.start()

    def _drive(self) -> None:
        assert self._finish is not None
        try:
            self._results = self._finish()
        except Exception as error:  # noqa: BLE001 - kept for whoever asks for the result
            self._error = error
            if self._cancelled.is_set():
                logger.info("Experiment cancelled, no further waves submitted")
            else:
                logger.warning(
                    f"Experiment stopped in wave {self._wave + 1} of "
                    f"{self._num_waves}: {error}"
                )
