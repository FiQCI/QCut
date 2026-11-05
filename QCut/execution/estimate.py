"""
Working out what running an experiment will cost, without submitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from QCut.execution.dispatch import (
    DEFAULT_SHOTS,
    _batch_shots,
    _batches,
    _has_measurements,
)
from QCut.execution.submit import _submission
from QCut.execution.waves import _first_wave_jobs, _wave_budget
from QCut.model.cutcircuit import CutExperiment
from QCut.utils.backend import _backend_shot_cap


@dataclass(frozen=True)
class JobEstimate:
    """One ``run`` call: how many circuits it carries, at how many shots each."""

    circuits: int
    shots: int


@dataclass(frozen=True)
class RunEstimate:
    """What running an experiment will cost, before anything is submitted."""

    breakdown: dict[str, JobEstimate] = field(default_factory=dict)
    exact: bool = True

    @property
    def jobs(self) -> int:
        """How many ``run`` calls the experiment takes."""
        return len(self.breakdown)

    @property
    def circuits(self) -> int:
        """How many circuits are submitted, over every job."""
        return sum(job.circuits for job in self.breakdown.values())

    @property
    def shots(self) -> int:
        """How many shots the experiment spends in total."""
        return sum(job.circuits * job.shots for job in self.breakdown.values())

    def __str__(self) -> str:
        """Format string."""
        line = (
            f"Experiment will run {self.circuits} circuits in {self.jobs} jobs with a "
            f"total of {self.shots} shots."
        )
        if not self.exact:
            line += (
                " Past the first wave the shots each group gets depend on what the wave"
                " before it measured. The circuit count is the most it can submit, the"
                " job count the fewest it can take, since circuits wanting very"
                " different shot counts cannot share a job, and the shot total is right"
                " to rounding."
            )
        return f"{line} See the returned object for the breakdown."

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


def _named(jobs: list[JobEstimate]) -> dict[str, JobEstimate]:
    return {f"job{index}": job for index, job in enumerate(jobs, start=1)}


def _estimated_batches(jobs, max_batch_size, cap) -> list[JobEstimate]:
    """Batch a wave's jobs the way :func:`_dispatch` will, without running them."""
    runnable = [job for job in jobs if _has_measurements(job[1])]
    runnable.sort(key=lambda job: job[2])
    return [
        JobEstimate(len(batch), _batch_shots(batch, cap))
        for batch in _batches(runnable, max_batch_size)
    ]


def _projected_wave_jobs(cut_experiment, wave: int, shots: int):
    """A later wave's jobs as they would be if every group got an equal share."""
    plan = cut_experiment.plan
    return [
        ([(group, obs, sub)], circuit, shots)
        for group, obs_groups in enumerate(cut_experiment.experiments)
        for obs, obs_group in enumerate(obs_groups)
        for sub, circuit in obs_group.items()
        if plan.waves.get(sub, 0) == wave
    ]


def estimate_run(
    cut_experiment: CutExperiment,
    shots: int = DEFAULT_SHOTS,
    backend=None,
    max_batch_size: int = 100,
) -> RunEstimate:
    """Work out what :func:`submit_experiments` will cost, without submitting anything.

    Exact for an experiment that runs in one go. One whose wire cuts communicate runs in
    waves, and every wave after the first spends its shots in proportion to what the one
    before it measured, so only that first wave is known in advance. The rest is worked
    out as if every group got an equal share. That gives the most circuits a wave can
    submit, since a group whose label never came up submits none, and the fewest jobs it
    can take, since an uneven share puts circuits wanting very different shot counts in
    jobs of their own.

    Args:
        cut_experiment (CutExperiment): the experiment circuits that would be run.
        shots (int): shots per circuit, as passed to :func:`submit_experiments`.
        backend: the backend or sampler that would run it, read only for the most shots
            it takes in one job.
        max_batch_size (int): maximum number of circuits submitted per ``run`` call.

    Returns:
        RunEstimate: the circuits, jobs and shots the run takes, and a breakdown by job.

    """
    cap = _backend_shot_cap(backend) if backend is not None else None

    if cut_experiment.plan is None:
        # Counted through the same pair the run uses, so an estimate cannot drift from
        # what is submitted.
        jobs, _empty = _submission(cut_experiment)
        return RunEstimate(
            _named(
                [
                    JobEstimate(len(jobs[start : start + max_batch_size]), shots)
                    for start in range(0, len(jobs), max_batch_size)
                ]
            ),
            exact=True,
        )

    plan = cut_experiment.plan
    budget = _wave_budget(cut_experiment)

    estimates = _estimated_batches(
        _first_wave_jobs(cut_experiment, shots, budget.sharing, budget.first_scale),
        max_batch_size,
        cap,
    )
    for wave in range(1, plan.last_wave + 1):
        even = max(1, round(budget.later_total(shots) / budget.groups))
        estimates += _estimated_batches(
            _projected_wave_jobs(cut_experiment, wave, even), max_batch_size, cap
        )

    return RunEstimate(_named(estimates), exact=plan.last_wave == 0)
