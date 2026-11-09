"""Helper classes for storing results."""

from dataclasses import dataclass, field


def _select_label(counts: dict, clbits, label, scale: float) -> dict:
    """Keep the shots whose measured label matches, rescaled to the nominal total.

    A communicating wire cut's measured bits say which state the other side prepared,
    so only the shots that came out with this group's label belong to it. Scaling the
    survivors by the number of labels turns the surviving fraction into the estimate of
    that outcome's probability that the decomposition asks for.

    The qpd register is added before the end-of-circuit one, so it is the last field of
    a counts key, and within a field the highest classical bit comes first.
    """
    kept = {}
    for key, value in counts.items():
        bits = key.split(" ")[-1]
        if all(
            bits[len(bits) - 1 - clbit] == str(wanted)
            for clbit, wanted in zip(clbits, label)
        ):
            kept[key] = value * scale
    return kept


def _counts_from(result, index: int) -> dict[str, float]:
    """Counts for one circuit, whatever ran it."""
    if isinstance(result, dict):
        return dict(result)
    get_counts = getattr(result, "get_counts", None)
    if get_counts is not None:
        return dict(get_counts(index))
    return dict(result[index].join_data().get_counts())


@dataclass(frozen=True)
class CircuitResult:
    """One subcircuit's result within one group, before it is turned into counts.

    Holds what the backend or the sampler handed back rather than counts taken out of
    it, so running an experiment does no arithmetic of its own and the estimate can be
    recomputed from the same results.

    ``scale`` brings the circuit back to the nominal shot count, since a batch runs at
    whatever its own circuits asked for. ``label_filter`` is the communicating wire
    cuts' selection: several groups share one measuring circuit and each keeps only the
    shots carrying the label it answers, so the selection belongs to the group's own
    result rather than to the circuit they all read.
    """

    result: object
    index: int = 0
    scale: float = 1.0
    label_filter: tuple[tuple, ...] = field(default_factory=tuple)

    def raw_counts(self) -> dict[str, float]:
        """Counts as measured, brought to the nominal shot count."""
        counts = _counts_from(self.result, self.index)
        if self.scale == 1.0:
            return counts
        return {key: value * self.scale for key, value in counts.items()}

    def counts(self) -> dict[str, float]:
        """What this group takes from the circuit: rescaled, then label-selected."""
        counts = self.raw_counts()
        for clbits, label, scale in self.label_filter:
            counts = _select_label(counts, clbits, label, scale)
        return counts


# Class for storing results from single sub-circuit run
class SubResult:
    """Storage class for easier storage/access to the results of a subcircuit."""

    def __init__(self, measurements: list, count: int) -> None:
        """Init."""
        self.measurements = measurements  # measurement results
        self.count = count  # counts for this specific measurement

    def __str__(self) -> str:
        """Format string."""
        return f"{self.measurements}, {self.count}"

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


# Store total results of all sub-circuits for a single experiment run
class TotalResult:
    """Storage class for easier access to the results of a subcircuit group."""

    def __init__(self, *subcircuits: list[list[SubResult]]) -> None:
        """Init."""
        self.subcircuits = subcircuits

    def __str__(self) -> str:
        """Format string."""
        substr = ""
        for i in self.subcircuits:
            substr += f"{i}"
        return substr

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


class RawResult:
    """
    Simple wrapper class for raw experiment results. Stores the raw results, the shot
    count they were taken at, and the experiment they came from, so that
    :func:`QCut.estimate_expectation_values` can be called on the result alone rather
    than having the caller carry a second object around.
    """

    def __init__(
        self,
        results: list[list[dict[int, CircuitResult]]],
        shots: int,
        experiment=None,
    ):
        self._shots = shots
        self._experiment = experiment
        self.results = results

    @property
    def experiment(self):
        """The experiment these results came from, if it was recorded."""
        return self._experiment

    @property
    def shots(self) -> int:
        """The shot count the experiment was run at."""
        return self._shots

    def result(self) -> list[list[dict[int, CircuitResult]]]:
        """
        Get raw results for all experiments, as ``[group][observable][subcircuit]``.

        Each subcircuit holds a :class:`CircuitResult`, which is what the backend or
        sampler returned rather than counts taken out of it. Call
        :meth:`CircuitResult.counts` on one for its counts.
        """
        return self.results
