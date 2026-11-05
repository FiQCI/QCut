"""
Running an experiment whose wire cuts communicate, in waves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from QCut.errors.qcuterror import QCutError
from QCut.execution.dispatch import _dispatch, _seeded
from QCut.job.cutresult import CircuitResult
from QCut.job.handle import QCutJob
from QCut.utils.backend import _backend_shot_cap

logger: logging.Logger = logging.getLogger(__name__)


#: Share of the shot budget spent on the measuring wave. A measuring circuit is shared
#: between every group that differs only in which label it answers, so a shot spent
#: there buys precision for all of them at once, while a preparing shot buys it for one
#: group alone. Splitting the budget evenly over the subcircuits therefore over-funds
#: the measuring side. Scans at one, two and three wires, over cuts into two and three
#: pieces, all put the best share near a sixth, independent of the block width and of
#: how many pieces the circuit was cut into.
MEASURE_SHARE: float = 1 / 6


@dataclass(frozen=True)
class _WaveBudget:
    """How a communicating run's shots are split between its waves.

    Attributes:
        sharing: the groups holding the same measuring circuit, keyed by what they
            share. Wave zero runs each such circuit once for all of them.
        first_scale: :data:`MEASURE_SHARE` as a factor on the shots of each measuring
            subcircuit.
        later_scale: the rest of the budget as a factor on the shots of each preparing
            subcircuit.
        groups: how many experiment groups the run has.
    """

    sharing: dict[tuple, list[int]]
    first_scale: float
    later_scale: float
    groups: int

    def later_total(self, shots: int) -> int:
        """The shots one later wave splits between the groups."""
        return round(shots * self.groups * self.later_scale)


def _wave_budget(cut_experiment) -> _WaveBudget:
    """Work out how a communicating run spends its shots, for running or estimating it.

    Budget-neutral reweighting of the waves. Every subcircuit would otherwise get
    shots * groups whatever wave it fell in, which hands the measuring wave a share of
    one over the number of pieces the circuit was cut into. Rewriting
    :data:`MEASURE_SHARE` as a per-subcircuit factor leaves the run's total untouched,
    so ``shots`` keeps meaning what it always did.

    The run and :func:`QCut.estimate_run` both read it from here, so an estimate cannot
    drift from what is submitted.
    """
    plan = cut_experiment.plan
    groups = len(cut_experiment.experiments)
    subcircuits = list(cut_experiment.experiments[0][0])
    measuring = sum(1 for sub in subcircuits if plan.waves.get(sub, 0) == 0)
    preparing = len(subcircuits) - measuring
    first_scale = MEASURE_SHARE * len(subcircuits) / measuring
    later_scale = (
        (1 - MEASURE_SHARE) * len(subcircuits) / preparing if preparing else 0.0
    )

    sharing: dict[tuple, list[int]] = {}
    for group in range(groups):
        sharing.setdefault(plan.keys[group], []).append(group)

    return _WaveBudget(sharing, first_scale, later_scale, groups)


def _apply_communication(cut_experiment, results) -> None:
    """Record which outcome each group takes from its measuring run.

    The selection is attached to the group's own result rather than applied to it, since
    groups sharing a measuring circuit each keep a different label out of the same
    counts.
    """
    plan = cut_experiment.plan
    for (group, obs, sub), entries in plan.label_clbits.items():
        leaf = results[group][obs][sub]
        results[group][obs][sub] = replace(
            leaf,
            label_filter=leaf.label_filter
            + tuple(
                (clbits, plan.labels[group][bundle], 2**bundle.size)
                for bundle, clbits in entries
            ),
        )


def _label_fraction(counts: dict, clbits, label) -> float:
    """Return how often a measuring run came out with this label."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    kept = sum(
        value
        for key, value in counts.items()
        if all(
            key.split(" ")[-1][len(key.split(" ")[-1]) - 1 - clbit] == str(wanted)
            for clbit, wanted in zip(clbits, label)
        )
    )
    return kept / total


def _label_weights(cut_experiment, results, wave: int) -> list[float]:
    """Return how likely each group's path was, using the waves already run.

    A bundle's outcome is only known once its preparing side has been reached, so at
    wave ``w`` the weight is the product over the bundles settled by then. That is just
    the information the protocol itself has at that point.
    """
    plan = cut_experiment.plan
    settled = {bundle for bundle, at in plan.bundle_waves.items() if at <= wave}
    seen: dict[tuple[int, object], float] = {}
    for (group, obs, sub), entries in plan.label_clbits.items():
        for bundle, clbits in entries:
            if bundle not in settled or (group, bundle) in seen:
                continue
            seen[(group, bundle)] = _label_fraction(
                results[group][obs][sub].raw_counts(),
                clbits,
                plan.labels[group][bundle],
            )
    weights = []
    for group in range(len(cut_experiment.experiments)):
        weight = 1.0
        for bundle in plan.labels[group]:
            if bundle in settled:
                weight *= seen.get((group, bundle), 0.0)
        weights.append(weight)
    return weights


def _allocate(weights: list[float], total: int) -> list[int]:
    """Split ``total`` shots between groups in proportion to ``weights``.

    Proportional allocation is what minimises the variance of the sum, and it is the
    whole point of communicating: a label that rarely comes up needs correspondingly few
    shots spent on the state it asks for. A group whose label did come up must still get
    at least one shot, or its contribution goes missing and the estimate is biased, so
    the split uses largest remainders on top of a floor of one.
    """
    live = [index for index, weight in enumerate(weights) if weight > 0]
    allocation = [0] * len(weights)
    if not live:
        return allocation
    spare = max(total - len(live), 0)
    scale = sum(weights[index] for index in live)
    exact = [spare * weights[index] / scale for index in live]
    for position, index in enumerate(live):
        allocation[index] = 1 + int(exact[position])
    remainder = spare - sum(int(value) for value in exact)
    order = sorted(range(len(live)), key=lambda p: -(exact[p] % 1))
    for position in order[: max(remainder, 0)]:
        allocation[live[position]] += 1
    return allocation


def _first_wave_jobs(cut_experiment, shots, sharing, scale=1.0):
    """Return the jobs for wave zero, one per distinct label-independent circuit.

    ``scale`` is :data:`MEASURE_SHARE` rewritten as a per-subcircuit factor, so that
    the measuring wave takes that share of the run rather than an even one.
    """
    plan = cut_experiment.plan
    experiments = cut_experiment.experiments
    jobs = []
    for members in sharing.values():
        budget = max(1, round(shots * len(members) * scale))
        for obs, obs_group in enumerate(experiments[members[0]]):
            for sub, circuit in obs_group.items():
                if plan.waves.get(sub, 0) != 0:
                    continue
                jobs.append(([(group, obs, sub) for group in members], circuit, budget))
    return jobs


def _later_wave_jobs(cut_experiment, wave, allocation, results):
    """Return one wave's jobs, blanking groups whose path never came up."""
    plan = cut_experiment.plan
    jobs = []
    for group, obs_groups in enumerate(cut_experiment.experiments):
        for obs, obs_group in enumerate(obs_groups):
            for sub, circuit in obs_group.items():
                if plan.waves.get(sub, 0) != wave:
                    continue
                if allocation[group] <= 0:
                    results[group][obs][sub] = CircuitResult({})
                else:
                    jobs.append(([(group, obs, sub)], circuit, allocation[group]))
    return jobs


def _submit_communicating(job: QCutJob) -> None:
    """Submit an experiment whose wire cuts exchange their measured outcome.

    The state one side prepares depends on what the other measured, so the circuits run
    in waves. Wave zero holds everything no label can affect, and its circuits are
    shared between the groups differing only in which label they answer, so they run
    once for the whole set. Each later wave has its shots split in proportion to how
    often the labels it depends on actually came up.

    The budget is not divided evenly between the waves. Sharing makes a measuring shot
    worth more than a preparing one, so wave zero takes :data:`MEASURE_SHARE` of the run
    and the later waves divide the rest. The total is unchanged either way.

    Wave zero is submitted here, so the job has ids to report as soon as it is handed
    back. Every later wave needs the one before it collected first, so those are left
    to a thread of the job's own, which submits each as soon as it can rather than when
    someone next polls.

    TODO: the waves after the first do not share circuits at all. ``_later_wave_jobs``
    gives every group a job of its own, where :func:`_deduplicate` would collapse the
    groups holding the same circuit as wave zero's ``plan.keys`` already does. It is not
    the same change as the plain path's, because each group carries its own shot
    allocation by then, so the shared job would have to run at their sum and be scaled
    back per group.
    """
    cut_experiment = job.experiment
    shots = job.shots
    plan = cut_experiment.plan
    results = _seeded(cut_experiment)
    cap = _backend_shot_cap(job.backend)
    budget = _wave_budget(cut_experiment)

    logger.info(
        f"Wave 1 of {plan.last_wave + 1}: {len(budget.sharing)} distinct measuring "
        f"run(s) shared across {len(cut_experiment.experiments)} group(s)."
    )
    _dispatch(
        job,
        0,
        _first_wave_jobs(cut_experiment, shots, budget.sharing, budget.first_scale),
        cap,
        results,
    )

    def drive():
        job._collect(results, 0)
        for wave in range(1, plan.last_wave + 1):
            if job._cancelled.is_set():
                raise QCutError(f"cancelled before wave {wave + 1}")
            weights = _label_weights(cut_experiment, results, wave)
            allocation = _allocate(weights, budget.later_total(shots))
            logger.info(
                f"Wave {wave + 1} of {plan.last_wave + 1}: preparing circuits for "
                f"{sum(1 for value in allocation if value > 0)} of "
                f"{len(cut_experiment.experiments)} group(s), shots split in "
                "proportion to how often each label came up."
            )
            _dispatch(
                job,
                wave,
                _later_wave_jobs(cut_experiment, wave, allocation, results),
                cap,
                results,
            )
            job._collect(results, wave)

        _apply_communication(cut_experiment, results)
        return results

    job._start(drive, background=True)
