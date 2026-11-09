"""
The separable form a reconstructed distribution takes, and the queries on it.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator

import numpy as np


class Mode:
    r"""One subcircuit's weights, as the outcomes that carry one.

    A mode is indexed by the local outcomes of the bits its subcircuit holds, so it
    spans ``2**width`` rows in principle. Only outcomes that were actually measured
    carry a weight, and a group cannot produce more distinct outcomes than it had
    shots, so the number of rows that matter is bounded by the shot count rather than
    by ``2**width``. Holding them by index is what keeps a wide subcircuit affordable:
    the rows a search reads, and the arithmetic it does per row, then grow with the
    shots rather than with the width.

    Args:
        idx (np.ndarray): the local outcomes carrying a weight, ascending and distinct.
        val (np.ndarray): ``(len(idx), groups)``, their weights per group.
        width (int): how many bits the local outcome spans.

    Attributes:
        idx (np.ndarray): as given.
        val (np.ndarray): as given.
        width (int): as given.
    """

    __slots__ = ("idx", "val", "width")

    def __init__(self, idx: np.ndarray, val: np.ndarray, width: int) -> None:
        """Init."""
        self.idx = np.asarray(idx, dtype=np.intp)
        self.val = np.asarray(val, dtype=float)
        self.width = width

    @classmethod
    def from_dense(cls, mode: np.ndarray, width: int | None = None) -> Mode:
        """Keep the rows of a dense ``(2**width, groups)`` array that are not zero.

        Args:
            mode (np.ndarray): the dense mode.
            width (int): how many bits it is indexed by, inferred from its length when
                not given.

        Returns:
            Mode: the same mode, by index.
        """
        dense = np.asarray(mode, dtype=float)
        if width is None:
            width = int(dense.shape[0]).bit_length() - 1
        if dense.shape[1] == 0:
            return cls(np.empty(0, dtype=np.intp), dense[:0], width)
        keep = np.flatnonzero(np.any(dense != 0.0, axis=1))
        return cls(keep, dense[keep], width)

    @classmethod
    def empty(cls, width: int, groups: int) -> Mode:
        """A mode carrying no weight anywhere."""
        return cls(np.empty(0, dtype=np.intp), np.zeros((0, groups)), width)

    @property
    def groups(self) -> int:
        """How many groups the mode carries a weight for."""
        return self.val.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """The shape the mode would have written out."""
        return (1 << self.width, self.groups)

    def __len__(self) -> int:
        """How many local outcomes the mode spans, stored or not."""
        return 1 << self.width

    def __getitem__(self, key):
        """Read as if the mode were dense, so ``mode[local]`` and ``mode[local, g]``.

        Rows that are not stored read as zero, which is what they are.
        """
        if isinstance(key, tuple):
            local, group = key
            return self.row(int(local))[group]
        return self.row(int(key))

    def is_full(self) -> bool:
        """Whether every local outcome carries a weight."""
        return self.idx.size == (1 << self.width)

    def position(self, local: int) -> int | None:
        """Where a local outcome sits in :attr:`val`, or None when it has no row."""
        pos = int(np.searchsorted(self.idx, local))
        if pos < self.idx.size and self.idx[pos] == local:
            return pos
        return None

    def row(self, local: int) -> np.ndarray:
        """One local outcome's weights, zeros when it carries none."""
        pos = self.position(local)
        return np.zeros(self.groups) if pos is None else self.val[pos]

    def missing(self) -> Iterator[int]:
        """The local outcomes carrying no weight, ascending, read off the gaps."""
        previous = -1
        for current in self.idx:
            yield from range(previous + 1, int(current))
            previous = int(current)
        yield from range(previous + 1, 1 << self.width)

    def sum(self, axis: int = 0) -> np.ndarray:
        """The mode summed over its outcomes, which the absent rows do not affect."""
        if axis != 0:
            raise ValueError("a mode is only summed over its outcomes")
        return self.val.sum(axis=0)

    def extremes(self) -> np.ndarray:
        """Per group, the largest magnitude any one outcome contributes."""
        if self.val.size == 0:
            return np.zeros(self.groups)
        return np.abs(self.val).max(axis=0)

    def dense(self) -> np.ndarray:
        """The mode written out, ``(2**width, groups)``.

        Only for the callers that are exponential in the width anyway, such as
        :meth:`SeparableDistribution.array`.
        """
        out = np.zeros(self.shape)
        if self.idx.size:
            out[self.idx] = self.val
        return out


def _as_mode(mode, width: int) -> Mode:
    """Take either a dense array or a mode, and give back a mode."""
    return mode if isinstance(mode, Mode) else Mode.from_dense(mode, width)


class SeparableDistribution:
    r"""A reconstructed distribution, held as a sum of separable terms.

    Args:
        width (int): how many qubits the distribution spans, :math:`k`.
        bits (list[list[int]]): per mode, which bits of the outcome it carries, in
            ascending order. The modes partition ``range(width)``: every bit belongs to
            exactly one, which is what makes the terms separable.
        modes (list[Mode | np.ndarray]): per mode, the weight each group gave each of
            its local outcomes, where local bit ``t`` is the outcome bit ``bits[i][t]``.
            Weights carry the qpd signs. A dense ``(2**len(bits[i]), groups)``
            array is accepted and kept by index; see :class:`Mode`.
        coefficients (np.ndarray): one per group, carrying the quasiprobability
            coefficient and the shared wire cut parity.

    Attributes:
        constant (float): the :math:`(1 + \sum_y T[y]) / 2^k` above. Worked out from the
            modes rather than by summing the table, so it costs nothing and the
            distribution sums to one by construction.
    """

    def __init__(
        self,
        width: int,
        bits: list[list[int]],
        modes: list,
        coefficients: np.ndarray,
    ) -> None:
        """Init."""
        self.width = width
        self.bits = bits
        self.modes = [_as_mode(mode, len(held)) for mode, held in zip(modes, bits)]
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.constant = (1.0 + self.total()) / (1 << width)

    @property
    def groups(self) -> int:
        """How many groups contributed. A group with an empty result is not one."""
        return len(self.coefficients)

    def total(self) -> float:
        r"""The sum :math:`\sum_y T[y]`, which is separable like everything else."""
        if self.groups == 0:
            return 0.0
        totals = np.prod([mode.sum(axis=0) for mode in self.modes], axis=0)
        return float(self.coefficients @ totals)

    def _outcome(self, blocks: list[int]) -> int:
        """One local index per mode, assembled into the outcome they name together."""
        outcome = 0
        for bits, local in zip(self.bits, blocks):
            for position, bit in enumerate(bits):
                outcome |= ((local >> position) & 1) << bit
        return outcome

    def _blocks(self, outcome: int) -> list[int]:
        """The reverse: an outcome split into one local index per mode."""
        blocks = []
        for bits in self.bits:
            local = 0
            for position, bit in enumerate(bits):
                local |= ((outcome >> bit) & 1) << position
            blocks.append(local)
        return blocks

    def array(self) -> np.ndarray:
        """The whole distribution, indexed by outcome.

        Returns:
            np.ndarray: ``2**width`` quasi-probabilities, entry ``x`` being the
            probability of the outcome whose bit ``b`` is ``(x >> b) & 1``.
        """
        accumulated = np.zeros((2,) * self.width)
        term = np.empty((2,) * self.width)
        dense = [mode.dense() for mode in self.modes]
        for group in range(self.groups):
            term.fill(self.coefficients[group])
            for bits, mode in zip(self.bits, dense):
                shape = [1] * self.width
                for bit in bits:
                    shape[self.width - 1 - bit] = 2
                term *= mode[:, group].reshape(shape)
            accumulated += term
        return self.constant - accumulated.reshape(-1)[::-1]

    def value(self, outcome: int) -> float:
        """One outcome's quasi-probability, without building anything.

        Costs one pass over the groups per mode, so it does not grow with ``width``.

        Args:
            outcome (int): the outcome, bit ``b`` being ``(outcome >> b) & 1``.

        Returns:
            float: its quasi-probability, which may be negative.
        """
        if self.groups == 0:
            return self.constant
        weights = self.coefficients
        for mode, local in zip(self.modes, self._blocks(self._flip(outcome))):
            position = mode.position(local)
            if position is None:
                return self.constant
            weights = weights * mode.val[position]
        return self.constant - float(weights.sum())

    def _flip(self, outcome: int) -> int:
        """The complement of an outcome, which is the index into :math:`T`."""
        return outcome ^ ((1 << self.width) - 1)

    def _unmeasured(self) -> Iterator[int]:
        """Outcomes no group measured, which all have the value :attr:`constant`.

        Yields:
            int: such outcomes, in the flipped indexing the search works in.
        """
        seen: set[int] = set()
        for level, mode in enumerate(self.modes):
            if mode.is_full():
                continue
            others = [i for i in range(len(self.modes)) if i != level]
            span = 1
            for i in others:
                span *= 1 << self.modes[i].width
            for gap in mode.missing():
                for choice in range(span):
                    blocks = [0] * len(self.modes)
                    blocks[level] = gap
                    rest = choice
                    for i in others:
                        blocks[i] = rest % (1 << self.modes[i].width)
                        rest //= 1 << self.modes[i].width
                    # A single outcome can be unmeasured through more than one mode,
                    # so the levels overlap and have to be deduplicated.
                    outcome = self._outcome(blocks)
                    if outcome not in seen:
                        seen.add(outcome)
                        yield outcome

    def top(self, count: int) -> list[tuple[int, float]]:
        """The most likely outcomes, exactly, without building the whole table.

        Args:
            count (int): how many outcomes to return. Capped at ``2**width``.

        Returns:
            list[tuple[int, float]]: ``count`` pairs of ``(outcome,
            quasi-probability)``, most likely first. Values may be negative, and are not
            projected onto a physical distribution -- that needs the whole table, see
            :meth:`QCut.QuasiProbabilities.nearest_probabilities`.

        Raises:
            ValueError: ``count`` is not positive.
        """
        if count < 1:
            raise ValueError(f"count must be at least one, got {count}")
        count = min(count, 1 << self.width)

        found: list[tuple[int, float]] = []
        if self.groups > 0:
            search = _Search(self, count)
            search.descend(0, self.coefficients, [])
            for taken, outcome in enumerate(self._unmeasured()):
                if taken >= count:
                    break
                search.offer(outcome, 0.0)
            found = [
                (self._flip(outcome), self.constant - value)
                for value, outcome in ((-key, out) for key, out in search.heap)
            ]

        if len(found) < count:
            taken_outcomes = {outcome for outcome, _value in found}
            spare = (out for out in range(1 << self.width) if out not in taken_outcomes)
            found += [(next(spare), self.constant) for _ in range(count - len(found))]

        found.sort(key=lambda pair: -pair[1])
        return found

    def marginal(self, keep: list[int]) -> SeparableDistribution:
        """The distribution over a subset of the bits, summed over the rest.

        Args:
            keep (list[int]): the bits to keep, becoming bits ``0..len(keep)-1`` of the
                result in the order given.

        Returns:
            SeparableDistribution: the marginal, in the same separable form.

        Raises:
            ValueError: ``keep`` is empty, repeats a bit, or names one outside the
                distribution.
        """
        if not keep:
            raise ValueError("keep must name at least one bit")
        if len(set(keep)) != len(keep):
            raise ValueError(f"keep must not repeat, got {keep}")
        outside = [bit for bit in keep if not 0 <= bit < self.width]
        if outside:
            raise ValueError(f"bits {outside} are outside a {self.width}-bit outcome")

        position_of = {bit: position for position, bit in enumerate(keep)}
        new_bits, new_modes = [], []
        for bits, mode in zip(self.bits, self.modes):
            held = sorted(
                (position_of[bit], index)
                for index, bit in enumerate(bits)
                if bit in position_of
            )
            new_bits.append([position for position, _index in held])
            kept = [index for _position, index in held]
            new_modes.append(_summed_over(mode, kept))
        return SeparableDistribution(len(keep), new_bits, new_modes, self.coefficients)


def _summed_over(mode: Mode, held: list[int]) -> Mode:
    """A mode summed over the local bits it is not keeping.

    Args:
        mode (Mode): the mode to reduce.
        held (list[int]): the local bits to keep, in the order they become the local
            bits of the result.

    Returns:
        Mode: the reduced mode, over ``len(held)`` bits.
    """
    if held == list(range(mode.width)):
        return mode
    if mode.idx.size == 0:
        return Mode.empty(len(held), mode.groups)

    target = np.zeros(mode.idx.size, dtype=np.intp)
    for position, bit in enumerate(held):
        target |= ((mode.idx >> bit) & 1) << position

    idx, inverse = np.unique(target, return_inverse=True)
    summed = np.zeros((idx.size, mode.groups))
    np.add.at(summed, inverse, mode.val)

    keep = np.flatnonzero(np.any(summed != 0.0, axis=1))
    return Mode(idx[keep], summed[keep], len(held))


class _Search:
    """Branch and bound state for :meth:`SeparableDistribution.top`."""

    def __init__(self, distribution: SeparableDistribution, count: int) -> None:
        """Init."""
        self.distribution = distribution
        self.count = count
        #: The best ``count`` found so far, as a max-heap on the value by negation.
        self.heap: list[tuple[float, int]] = []

        # Per mode, per group, the largest magnitude any one outcome can contribute.
        extremes = [mode.extremes() for mode in distribution.modes]

        self.tail = [np.ones(distribution.groups) for _ in range(len(extremes) + 1)]
        for level in range(len(extremes) - 1, -1, -1):
            self.tail[level] = self.tail[level + 1] * extremes[level]

    @property
    def limit(self) -> float:
        """How small an entry has to be to still make the list."""
        return -self.heap[0][0] if len(self.heap) >= self.count else np.inf

    def offer(self, outcome: int, value: float) -> None:
        """Keep an entry if it beats the worst one held."""
        if len(self.heap) < self.count:
            heapq.heappush(self.heap, (-value, outcome))
        elif value < -self.heap[0][0]:
            heapq.heapreplace(self.heap, (-value, outcome))

    def leaf(self, weights: np.ndarray, blocks: list[int]) -> None:
        """Read the last mode exactly, every group at once."""
        mode = self.distribution.modes[-1]
        if mode.val.size == 0:
            return
        values = mode.val @ weights
        take = min(self.count, values.size)
        for position in np.argpartition(values, take - 1)[:take]:
            self.offer(
                self.distribution._outcome([*blocks, int(mode.idx[position])]),
                float(values[position]),
            )

    def descend(self, level: int, weights: np.ndarray, blocks: list[int]) -> None:
        """Choose this mode's outcome, best bound first, and stop when it cannot win."""
        if level == len(self.distribution.modes) - 1:
            self.leaf(weights, blocks)
            return
        mode = self.distribution.modes[level]
        if mode.val.size == 0:
            return
        bounds = np.abs(mode.val) @ (np.abs(weights) * self.tail[level + 1])
        for position in np.argsort(-bounds):
            beaten = len(self.heap) >= self.count and bounds[position] <= -self.limit
            if bounds[position] == 0.0 or beaten:
                break
            self.descend(
                level + 1,
                weights * mode.val[position],
                [*blocks, int(mode.idx[position])],
            )
