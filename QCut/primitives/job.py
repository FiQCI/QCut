"""A handle on everything one primitive run submitted, one experiment per circuit."""

from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

from qiskit.providers import JobStatus
from qiskit.providers.exceptions import JobTimeoutError

from QCut.job.cutresult import RawResult
from QCut.job.handle import _FINAL, PartialBatch, QCutJob, _aggregate

R = TypeVar("R")


class QCutPrimitiveJob(Generic[R]):
    """A primitive's run, handed back as soon as everything is submitted.

    Shaped like a qiskit job. Each input circuit is one experiment with a
    :class:`~QCut.QCutJob` of its own, in :attr:`jobs`, and this answers for all of
    them together: the ids and statuses cover every circuit, and :meth:`cancel`
    cancels every circuit.

    :meth:`result` waits for every circuit and builds the primitive's own result, which
    is what the estimator and the sampler differ in. :meth:`raw_results` gives the raw
    counts behind it.
    """

    def __init__(self, jobs: list[QCutJob]) -> None:
        self._jobs = list(jobs)
        self._result: R | None = None
        self._lock = threading.Lock()

    @property
    def jobs(self) -> list[QCutJob]:
        """The job of each input circuit, in the order the circuits were given."""
        return list(self._jobs)

    @property
    def experiments(self) -> list:
        """The experiment each input circuit was cut into."""
        return [job.experiment for job in self._jobs]

    @property
    def num_circuits(self) -> int:
        """How many experiment circuits have been submitted so far, over every input."""
        return sum(job.num_circuits for job in self._jobs)

    @property
    def total_shots(self) -> int:
        """How many shots have been submitted so far, over every input."""
        return sum(job.total_shots for job in self._jobs)

    # -- identity and status ------------------------------------------------------ #

    def job_id(self) -> str | None:
        """The first backend job's id, ``None`` if nothing was submitted."""
        ids = self.job_ids()
        return ids[0] if ids else None

    def job_ids(self) -> list[str | None]:
        """Every backend job's id, circuit by circuit, in submission order."""
        return [identifier for job in self._jobs for identifier in job.job_ids()]

    def statuses(self) -> list[JobStatus]:
        """Each input circuit's status, in the order the circuits were given."""
        return [job.status() for job in self._jobs]

    def status(self) -> JobStatus:
        """One status for the whole run.

        A circuit failing makes it ``ERROR``, then one being cancelled ``CANCELLED``.
        Otherwise it is ``DONE`` only once every circuit is.
        """
        return _aggregate(self.statuses())

    def done(self) -> bool:
        """Whether every circuit finished successfully."""
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Whether any circuit is running, or has waves left to submit."""
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """Whether the run was cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """Whether the run is done, failed or cancelled."""
        return self.status() in _FINAL

    def wait_for_final_state(
        self, timeout: float | None = None, wait: float = 5.0
    ) -> None:
        """Poll until the run reaches a final state.

        Args:
            timeout (float): seconds to wait before giving up, or ``None`` to wait as
                long as it takes.
            wait (float): seconds between polls.

        Raises:
            JobTimeoutError: the run was not finished within ``timeout``.
        """
        start = time.monotonic()
        while not self.in_final_state():
            elapsed = time.monotonic() - start
            if timeout is not None and elapsed >= timeout:
                raise JobTimeoutError(f"the run was not finished after {elapsed:.1f}s")
            time.sleep(wait if timeout is None else min(wait, timeout - elapsed))

    def cancel(self) -> None:
        """Cancel every circuit's backend jobs, and submit no further waves."""
        for job in self._jobs:
            job.cancel()

    # -- results ------------------------------------------------------------------ #

    def partial_results(self) -> list[list[PartialBatch]]:
        """Each input circuit's batches as they stand, with results once they are in."""
        return [job.partial_results() for job in self._jobs]

    def raw_results(self, timeout: float | None = None) -> list[RawResult]:
        """Wait for every circuit and return its raw counts.

        Args:
            timeout (float): seconds to wait for the whole run before giving up, or
                ``None`` to wait as long as it takes.

        Returns:
            list[RawResult]: one per input circuit, in the order they were given.

        Raises:
            JobTimeoutError: the run was not finished within ``timeout``.
            QCutError: the run was cancelled.
            Exception: whatever a refused batch, a failed job or a failed wave raised.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        raws = []
        for job in self._jobs:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            raws.append(job.result(remaining))
        return raws

    def result(self, timeout: float | None = None) -> R:
        """Wait for every circuit and return the primitive's result, built once.

        Args:
            timeout (float): seconds to wait for the whole run before giving up, or
                ``None`` to wait as long as it takes.

        Raises:
            JobTimeoutError: the run was not finished within ``timeout``.
            QCutError: the run was cancelled.
            Exception: whatever a refused batch, a failed job or a failed wave raised.
        """
        if self._result is not None:
            return self._result
        raws = self.raw_results(timeout)
        with self._lock:
            if self._result is None:
                self._result = self._build(raws)
        return self._result

    def _build(self, raws: list[RawResult]) -> R:
        """Turn every circuit's raw counts into the primitive's result."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Represent without asking the backend anything."""
        return f"<{type(self).__name__} over {len(self._jobs)} circuit(s)>"
