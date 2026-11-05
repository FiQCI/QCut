"""Group cut locations that share a single quasiprobability decomposition.

Cutting each location separately costs the product of their sampling overheads. Some
groups of cuts have a joint decomposition that costs strictly less, so a QPD term has to
be able to span several cut locations at once.

A :class:`Bundle` names the locations that share one decomposition. Its terms carry an
operation per *side* acting on as many qubits as the bundle holds, where side 0 is the
control-side placeholder of every cut and side 1 the target-side one. A bundle of one
is exactly the single-cut case, with one-qubit operations, and takes the original code
path.

The only joint decomposition here is :mod:`QCut.qpd_joint`, for parallel two-qubit
rotation gates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate

from QCut.model.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.options import CutOptions
from QCut.qpd.qpd_joint import (
    gamma_joint,
    gamma_separate,
    single_axis_frame,
    single_axis_theta,
)

logger: logging.Logger = logging.getLogger(__name__)

#: Placeholder side indices. Side 0 holds the ``_c_`` and ``Meas`` placeholders, side 1
#: the ``_t_`` and ``Init`` ones.
SIDE_0, SIDE_1 = 0, 1


@dataclass(frozen=True)
class Bundle:
    """Cut locations sharing one quasiprobability decomposition.

    ``layout[b][s]`` names the ``(cut, placeholder side)`` that bundle side ``b`` puts
    on its qubit ``s``. Each bundle side lives entirely in one subcircuit, which is what
    lets its operation act on several qubits at once. A cut whose placeholder sides run
    the other way round to its neighbours still joins the bundle, on the far side, since
    a single-axis gate is symmetric about its interaction axis.

    ``place[b]`` says whether side ``b``'s operation goes in where the first or the last
    of its placeholders sat. Which one works depends on the circuit, see
    :func:`_placement`.
    """

    cuts: tuple[int, ...]
    kind: str = "single"
    layout: tuple[tuple[tuple[int, int], ...], ...] = ()
    place: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        """How many cut locations the bundle covers."""
        return len(self.cuts)

    @property
    def anchor(self) -> int:
        """The cut that carries the bundle's operations once terms are flattened."""
        return self.cuts[0]

    @property
    def flipped(self) -> list[bool]:
        """Per cut, whether its two qubits are reversed relative to the bundle sides."""
        return [side == SIDE_1 for _, side in self.layout[0]]

    def bundle_side(self, cut: int, side: int) -> int | None:
        """Return which bundle side holds a placeholder, or None if it is not in one."""
        for index, members in enumerate(self.layout):
            if (cut, side) in members:
                return index
        return None


@dataclass(frozen=True)
class Placeholder:
    """Where one side of one cut ended up after the circuit was split."""

    subcircuit: int
    index: int
    qubit: int


def parse_placeholder(name: str) -> tuple[int, int] | None:
    """Return ``(cut_index, side)`` for a placeholder name, else None."""
    if "_c_" in name:
        side = SIDE_0
    elif "_t_" in name:
        side = SIDE_1
    elif name.startswith("Meas_"):
        side = SIDE_0
    elif name.startswith("Init_"):
        side = SIDE_1
    else:
        return None
    try:
        return int(name.split("_")[-1]), side
    except ValueError:  # pragma: no cover - defensive
        return None


def locate_placeholders(
    subcircuits: list[QuantumCircuit],
) -> dict[tuple[int, int], Placeholder]:
    """Map ``(cut index, side)`` to where that placeholder sits after the split."""
    found: dict[tuple[int, int], Placeholder] = {}
    for sub_index, subcircuit in enumerate(subcircuits):
        for index, instruction in enumerate(subcircuit.data):
            parsed = parse_placeholder(instruction.operation.name)
            if parsed is None:
                continue
            found[parsed] = Placeholder(
                sub_index, index, subcircuit.find_bit(instruction.qubits[0]).index
            )
    return found


def _placement(
    subcircuit: QuantumCircuit, placeholders: list[Placeholder]
) -> str | None:
    """Where one multi-qubit operation replacing ``placeholders`` can go, if anywhere.

    The block needs a single point in the subcircuit, so every placeholder has to be
    able to slide there without crossing anything on its own qubit. Operations on the
    group's other qubits are irrelevant, which is the point: the cuts are parallel, so
    each slot slides independently.

    Sliding the whole group to the earliest placeholder and to the latest are different
    questions and either can be the one that works, so both are tried. Returns
    ``"first"``, ``"last"``, or None if neither does.
    """
    if len(placeholders) < 2:
        return "first"
    qubits = [placeholder.qubit for placeholder in placeholders]
    if len(set(qubits)) != len(qubits):
        # Two cuts sharing a qubit are not parallel and the block cannot use it twice.
        return None
    own = {placeholder.index for placeholder in placeholders}
    for name, target in (("first", min(own)), ("last", max(own))):
        if all(
            _clear_between(subcircuit, placeholder, target, own)
            for placeholder in placeholders
        ):
            return name
    return None


def _clear_between(
    subcircuit: QuantumCircuit, placeholder: Placeholder, target: int, own: set[int]
) -> bool:
    """Whether a placeholder can slide to ``target`` without crossing its own wire."""
    for index in range(min(placeholder.index, target), max(placeholder.index, target)):
        if index in own:
            continue
        instruction = subcircuit.data[index]
        if placeholder.qubit in {
            subcircuit.find_bit(qubit).index for qubit in instruction.qubits
        }:
            return False
    return True


def _candidate_groups(
    cut_locations: list, placeholders: dict[tuple[int, int], Placeholder]
) -> dict[tuple[int, int], list[int]]:
    """Group gate cuts by which two subcircuits they join, ignoring which way round."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, location in enumerate(cut_locations):
        if not isinstance(location, CutLocation):
            continue
        side_0 = placeholders.get((index, SIDE_0))
        side_1 = placeholders.get((index, SIDE_1))
        if side_0 is None or side_1 is None:
            continue
        if side_0.subcircuit == side_1.subcircuit:  # pragma: no cover - defensive
            continue
        # A single-axis gate is symmetric about its interaction axis, so a cut whose
        # sides are the other way round joins the same bundle with its locals swapped.
        key = (
            (side_0.subcircuit, side_1.subcircuit)
            if side_0.subcircuit < side_1.subcircuit
            else (side_1.subcircuit, side_0.subcircuit)
        )
        groups.setdefault(key, []).append(index)
    return groups


def is_joint_eligible(location) -> bool:
    """Whether a cut location's gate has a single non-zero interaction coordinate."""
    if not isinstance(location, CutLocation):
        return False
    gate = joint_gate(location)
    return gate is not None and single_axis_frame(gate) is not None


def joint_gate(location: CutLocation):
    """Return the gate a joint decomposition would be built from, or None.

    The hand-written tables cover ``cz``, ``swap`` and ``iswap``. Only ``cz`` is
    single-axis, and it needs a gate object to derive its frame from, so it is
    reconstructed here when the marker did not carry one.
    """
    if location.gate is not None:
        return location.gate
    if location.gate_name == "cz":
        return CZGate()
    return None


def _wire_candidate_groups(
    cut_locations: list, placeholders: dict[tuple[int, int], Placeholder]
) -> dict[tuple[int, int], list[int]]:
    """Group wire cuts by the ordered pair of subcircuits they run between.

    Ordered, unlike the gate cut version, because a communicating wire cut sends its
    measured outcome one way only. Cuts pointing the other way between the same two
    subcircuits form their own bundle.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for index, location in enumerate(cut_locations):
        if not isinstance(location, SingleQubitCutLocation):
            continue
        measure = placeholders.get((index, SIDE_0))
        prepare = placeholders.get((index, SIDE_1))
        if measure is None or prepare is None:
            continue
        if measure.subcircuit == prepare.subcircuit:
            continue
        groups.setdefault((measure.subcircuit, prepare.subcircuit), []).append(index)
    return groups


def _would_cycle(edges: set[tuple[int, int]], new: tuple[int, int]) -> bool:
    """Whether adding ``new`` makes the subcircuit dependency graph cyclic.

    A communicating wire cut forces its measure side to run before its prepare side. A
    cycle of such constraints has no valid execution order, so one of the bundles
    involved has to fall back to the non-communicating decomposition.
    """
    source, target = new
    if source == target:
        return True
    reachable, frontier = {target}, [target]
    while frontier:
        node = frontier.pop()
        for a, b in edges:
            if a == node and b not in reachable:
                reachable.add(b)
                frontier.append(b)
    return source in reachable


def _group_placeholders(
    placeholders: dict[tuple[int, int], Placeholder],
    cuts: list[int],
    subcircuit: int,
) -> list[Placeholder]:
    """Return the group's placeholders that landed in one subcircuit."""
    return [
        placeholders[(cut, side)]
        for cut in cuts
        for side in (SIDE_0, SIDE_1)
        if placeholders[(cut, side)].subcircuit == subcircuit
    ]


#: Largest block the group search will attempt. Past this the term count makes the
#: decomposition impractical anyway, and it keeps the search from growing cubically on a
#: circuit with very many cuts.
MAX_GROUP: int = 8


def _widest_group(
    remaining: list[int],
    min_size: int,
    viable: Callable[[list[int]], tuple[str, str] | None],
) -> tuple[list[int], tuple[str, str]] | None:
    """The longest run of ``remaining`` that can be one bundle, or None if none can.

    Longest first, because growing a group one cut at a time misses real blocks: a gate
    coupling two of the block's qubits is in the way of a small group and harmlessly
    inside a larger one covering both.
    """
    for size in range(min(len(remaining), MAX_GROUP), min_size - 1, -1):
        for offset in range(len(remaining) - size + 1):
            candidate = remaining[offset : offset + size]
            place = viable(candidate)
            if place is not None:
                return candidate, place
    return None


def _form_groups(  # noqa: PLR0913
    eligible: list[int],
    placeholders: dict[tuple[int, int], Placeholder],
    subcircuits: list[QuantumCircuit],
    pair: tuple[int, int],
    min_size: int = 2,
    kind: str = "joint_rotation",
    fits: Callable[[Bundle], bool] | None = None,
) -> list[tuple[list[int], tuple[str, str]]]:
    """Split ``eligible`` into groups that one operation per side can replace.

    Runs are tried longest first rather than grown one cut at a time. Growing misses
    real blocks, because an operation coupling two of the block's qubits is in the way
    of a small group and harmlessly inside a larger one covering both of them.

    ``min_size`` is 2 for joint gate cutting, where a group of one is just the ordinary
    single-cut table, and 1 for communicating wire cuts when they are forced on.

    ``fits`` rejects a group the caller cannot use, and because the search already walks
    sizes downwards, rejecting a group is all it takes to fall back to smaller ones
    covering the same cuts. That is how a block too wide for a device's topology ends up
    as several narrow blocks rather than as no block at all.
    """

    def placements(group: list[int]) -> tuple[str, str] | None:
        found = []
        for sub in pair:
            where = _placement(
                subcircuits[sub], _group_placeholders(placeholders, group, sub)
            )
            if where is None:
                return None
            found.append(where)
        return (found[0], found[1])

    def viable(candidate: list[int]) -> tuple[str, str] | None:
        """Where this group's operations go, or None if it cannot be one bundle."""
        place = placements(candidate)
        if place is None:
            return None
        if fits is None:
            return place
        layout = _layout(candidate, placeholders, pair)
        bundle = Bundle(tuple(candidate), kind, layout, place)
        return place if fits(bundle) else None

    groups: list[tuple[list[int], tuple[str, str]]] = []
    remaining = list(eligible)
    while len(remaining) >= min_size:
        best = _widest_group(remaining, min_size, viable)
        if best is None:
            break
        groups.append(best)
        remaining = [cut for cut in remaining if cut not in best[0]]
    return groups


def _layout(
    group: list[int],
    placeholders: dict[tuple[int, int], Placeholder],
    pair: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Assign each cut's two placeholders to a bundle side.

    Bundle side 0 is the lower-numbered subcircuit, so a bundle side always sits wholly
    inside one subcircuit and its operation can span all of that side's qubits.
    """
    sides: list[list[tuple[int, int]]] = [[], []]
    for cut in group:
        for side in (SIDE_0, SIDE_1):
            which = 0 if placeholders[(cut, side)].subcircuit == pair[0] else 1
            sides[which].append((cut, side))
    return tuple(tuple(members) for members in sides)


def plan_bundles(
    cut_locations: list,
    subcircuits: list[QuantumCircuit],
    options: CutOptions,
    announce: bool = True,
    fits: Callable[[Bundle], bool] | None = None,
) -> list[Bundle]:
    """Decide which cut locations share a decomposition.

    Every cut ends up in exactly one bundle, and the bundles come back ordered by their
    first cut so the experiment tensor is reproducible.

    Args:
        cut_locations: the cuts to group.
        subcircuits: the split circuit, needed to tell which cuts are parallel and which
            subcircuits each cut joins.
        options: configuration. ``joint_rotation_cuts`` and ``wire_cut_communication``
            turn the two kinds of bundle off.
        announce: whether to log what was bundled. Off while costing a candidate plan,
            which would otherwise report a grouping that may not be the one used.
        fits: optional veto on a candidate bundle, used to keep planning inside what a
            backend can actually run. A vetoed group is retried in smaller pieces, so a
            block that is too wide becomes narrower blocks rather than none.

    Returns:
        One :class:`Bundle` per group, of kind ``"joint_rotation"`` for parallel
        rotation gates, ``"cc_wire"`` for wire cuts that exchange their measured
        outcome, and ``"single"`` for everything else.
    """
    placeholders = locate_placeholders(subcircuits)
    bundled: dict[int, Bundle] = {}

    if options.joint_rotation_cuts:
        for pair, members in _candidate_groups(cut_locations, placeholders).items():
            eligible = [
                index for index in members if is_joint_eligible(cut_locations[index])
            ]
            for group, place in _form_groups(
                eligible,
                placeholders,
                subcircuits,
                pair,
                kind="joint_rotation",
                fits=fits,
            ):
                _claim(bundled, group, "joint_rotation", placeholders, pair, place)

    minimum = options.min_communicating_block
    if minimum:
        _plan_wire_bundles(
            cut_locations, subcircuits, placeholders, bundled, minimum, fits
        )

    bundles: list[Bundle] = []
    seen: set[Bundle] = set()
    for index in range(len(cut_locations)):
        bundle = bundled.get(index, Bundle((index,), "single"))
        if bundle in seen:
            continue
        seen.add(bundle)
        bundles.append(bundle)

    if announce:
        _log_savings(bundles, cut_locations)
    return bundles


def _claim(
    bundled: dict[int, Bundle],
    group: list[int],
    kind: str,
    placeholders: dict[tuple[int, int], Placeholder],
    pair: tuple[int, int],
    place: tuple[str, str],
) -> None:
    """Record one bundle against every cut it covers."""
    bundle = Bundle(tuple(group), kind, _layout(group, placeholders, pair), place)
    for index in group:
        bundled[index] = bundle


def _plan_wire_bundles(
    cut_locations: list,
    subcircuits: list[QuantumCircuit],
    placeholders: dict[tuple[int, int], Placeholder],
    bundled: dict[int, Bundle],
    minimum: int,
    fits: Callable[[Bundle], bool] | None = None,
) -> None:
    """Group wire cuts that can exchange their measured outcome.

    Only blocks of at least ``minimum`` wires are formed. Communicating does not
    realise the whole of the gamma it advertises, because the protocol's per-shot
    feed-forward is emulated by post-selecting batched runs, so a narrow block can end
    up dearer than leaving it alone even though its gamma looks better.

    Bundles are considered in a fixed order and one is skipped when its direction would
    close a cycle in the execution order, since then no sequence of runs could satisfy
    it.
    """
    edges: set[tuple[int, int]] = set()
    candidates = _wire_candidate_groups(cut_locations, placeholders)
    for pair in sorted(candidates):
        if _would_cycle(edges, pair):
            logger.info(
                f"Wire cuts from subcircuit {pair[0]} to {pair[1]} keep the "
                "non-communicating decomposition, since running them in order would "
                "need a cycle."
            )
            continue
        groups = [
            entry
            for entry in _form_groups(
                candidates[pair],
                placeholders,
                subcircuits,
                pair,
                min_size=minimum,
                kind="cc_wire",
                fits=fits,
            )
            if len(entry[0]) >= minimum
        ]
        if not groups:
            continue
        edges.add(pair)
        for group, place in groups:
            _claim(bundled, group, "cc_wire", placeholders, pair, place)


def _log_savings(bundles: list[Bundle], cut_locations: list) -> None:
    """Report what bundling bought, since it changes both gamma and the group count."""
    from QCut.qpd.qpd_locc import gamma_local, gamma_locc

    joint = [bundle for bundle in bundles if bundle.kind == "joint_rotation"]
    if joint:
        together, apart = 1.0, 1.0
        for bundle in joint:
            thetas = [
                single_axis_theta(joint_gate(cut_locations[index]))
                for index in bundle.cuts
            ]
            together *= gamma_joint(thetas)
            apart *= gamma_separate(thetas)
        logger.info(
            "Bundled %d cut(s) into %d joint decomposition(s), gamma %.4f against %.4f "
            "cut separately.",
            sum(bundle.size for bundle in joint),
            len(joint),
            together,
            apart,
        )

    wire = [bundle for bundle in bundles if bundle.kind == "cc_wire"]
    if wire:
        together, apart = 1.0, 1.0
        for bundle in wire:
            together *= gamma_locc(bundle.size)
            apart *= gamma_local(bundle.size)
        logger.info(
            "Bundled %d wire cut(s) into %d communicating decomposition(s), gamma %.4f "
            "against %.4f without communication. These run in waves.",
            sum(bundle.size for bundle in wire),
            len(wire),
            together,
            apart,
        )


def communication_waves(
    bundles: list[Bundle],
    placeholders: dict[tuple[int, int], Placeholder],
    num_subcircuits: int,
) -> tuple[dict[int, int], dict[Bundle, int]]:
    """Order the subcircuits by the communicating cuts' dependencies.

    A communicating cut forces its measuring side to run before its preparing side, and
    those constraints chain. Cutting a circuit into A, B and C so that A feeds B and B
    feeds C takes three waves, because B's own measured outcome is what decides what C
    prepares. The wave of a subcircuit is the longest chain reaching it.

    Returns the wave per subcircuit and, per bundle, the wave in which its measured
    outcome becomes available, which is the wave of its preparing side.
    """
    edges = []
    for bundle in bundles:
        if bundle.kind != "cc_wire":
            continue
        cut = bundle.cuts[0]
        edges.append(
            (
                bundle,
                placeholders[(cut, SIDE_0)].subcircuit,
                placeholders[(cut, SIDE_1)].subcircuit,
            )
        )

    waves = dict.fromkeys(range(num_subcircuits), 0)
    # plan_bundles refuses any edge that would close a cycle, so this settles.
    for _ in range(num_subcircuits):
        changed = False
        for _bundle, measure, prepare in edges:
            if waves[prepare] < waves[measure] + 1:
                waves[prepare] = waves[measure] + 1
                changed = True
        if not changed:
            break

    return waves, {bundle: waves[prepare] for bundle, _m, prepare in edges}


def flatten_term(bundle: Bundle, term: dict) -> dict[int, dict]:
    """Spread one bundle term over its cut locations.

    The anchor keeps the operations and the whole coefficient, and the rest get a
    coefficient of one and no operations, meaning their placeholder is covered by the
    anchor's and should just be removed. Everything downstream that multiplies the
    per-cut coefficients together therefore keeps working unchanged.

    A bundle of one hands back the term itself rather than a copy, so single cuts behave
    exactly as they did before bundling existed.
    """
    if bundle.size == 1:
        return {bundle.anchor: term}
    spread = {bundle.anchor: dict(term)}
    for index in bundle.cuts[1:]:
        spread[index] = {"op_0": None, "op_1": None, "c": 1.0}
    return spread
