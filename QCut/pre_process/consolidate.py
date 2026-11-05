"""Merge runs of gates on the same qubit pair so the pair costs one cut.

If a circuit applies several two-qubit gates to the same pair with nothing else
touching those qubits in between, the whole run is itself one two-qubit unitary.
Cutting that unitary once is cheaper than cutting each gate, because the sampling
overhead of separate cuts multiplies. Two ``rzz(0.4)`` gates cost ``gamma = 3.16`` over
36 experiment groups apart, and ``gamma = 2.43`` over 6 as the single ``rzz(0.8)`` they
compose to.

QCut already requires that if one two-qubit gate on a pair is cut then every gate on
that pair is cut, otherwise the subcircuits cannot be separated. So merging is less an
optimisation of the user's choice than the cheapest way to honour one they already
made.

A cut marker is opaque, so it is unwrapped to the gate it carries before its run is
merged, and the result is marked again. Wire cut markers end a run rather than being
absorbed into it, since the cut has to stay where it was placed.

A run does not have to be contiguous. An instruction on disjoint qubits commutes with
the whole run and is simply skipped, and one that overlaps the pair is slid past
whenever it commutes with the members that would move across it. That second case is
the common one in an Ising or QAOA layer, where neighbouring ``rzz`` and ``cz`` gates
all commute, and without it a pair's gates are rarely adjacent enough to merge at all.
"""

from __future__ import annotations

import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import CZGate, SwapGate, UnitaryGate, iSwapGate
from qiskit.quantum_info import Operator

from QCut.qpd.qpd_gates import CutTwoQubitGate

logger: logging.Logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the qiskit version
    from qiskit.circuit.commutation_library import SessionCommutationChecker as _CHECKER
except ImportError:  # pragma: no cover
    _CHECKER = None
    logger.debug("no commutation library available; runs must be contiguous")

#: Never slid past, whatever the checker says. Cut markers pin a location the caller
#: chose, and the rest either carry classical data or are not unitary, so reordering
#: around them is not a question of commutation.
_UNSLIDEABLE: tuple[str, ...] = (
    "barrier",
    "measure",
    "Cut",
    "reset",
    "delay",
    "initialize",
)

# The per-gate markers carry no gate of their own, so the gate each one stands for has
# to be recovered before its run can be merged.
_MARKER_GATES: dict[str, Gate] = {
    "CutCZ": CZGate(),
    "CutSWAP": SwapGate(),
    "CutISWAP": iSwapGate(),
}


def marker_gate(operation) -> Gate | None:
    """Return the gate a two-qubit cut marker stands for, else None."""
    if operation.num_qubits != 2 or not operation.name.startswith("Cut"):
        return None
    if isinstance(operation, CutTwoQubitGate):
        return operation.gate
    return _MARKER_GATES.get(operation.name)


def _gate_of(operation) -> Gate:
    """The gate an instruction stands for, unwrapping a cut marker."""
    return marker_gate(operation) or operation


def _commutes(circuit: QuantumCircuit, member: int, operation, qubits) -> bool:
    """Whether ``operation`` on ``qubits`` commutes with the instruction at ``member``.

    Both sides are unwrapped first. The checker answers False for anything it cannot
    look inside, and a cut marker is an opaque custom instruction, so without unwrapping
    every marked pair would look non-commuting and no run would ever be extended.
    """
    if _CHECKER is None:
        return False
    other = circuit.data[member]
    try:
        return bool(
            _CHECKER.commute(
                _gate_of(operation),
                list(qubits),
                [],
                _gate_of(other.operation),
                [circuit.find_bit(q).index for q in other.qubits],
                [],
            )
        )
    except Exception:  # noqa: BLE001 - an unknown operation is simply not commuting
        return False


def _commuting_suffix(
    circuit: QuantumCircuit, instruction, qubits, current: list[int]
) -> list[int] | None:
    """The longest tail of ``current`` that ``instruction`` may be slid past, or None.

    Only members before the blocker move, and they move as a group, so the blocker has
    to commute with all of them. Trimming from the front rather than giving up matters:
    a run that absorbed a single-qubit gate early would otherwise be killed by a check
    that gate need not have been part of.
    """
    if _CHECKER is None or not current or instruction.clbits:
        return None
    if instruction.operation.name in _UNSLIDEABLE:
        return None
    suffix = list(current)
    while suffix:
        if all(
            _commutes(circuit, member, instruction.operation, qubits)
            for member in suffix
        ):
            return suffix
        suffix = suffix[1:]
    return None


def _cost(gate: Gate) -> tuple[float, int]:
    """Return ``(gamma, term count)`` for cutting ``gate``."""
    from QCut.qpd.qpd_generate import gamma, qpd_from_gate

    qpd = qpd_from_gate(gate)
    return gamma(qpd), len(qpd)


def _worth_merging(gates: list[Gate], merged: Gate) -> bool:
    """Whether cutting ``merged`` once beats cutting each of ``gates``.

    No case has been found where merging is worse, but gamma is a closed form so the
    check costs almost nothing. The term count is compared too, since a merged gate
    lands in a generic frame that can need more QPD rows than the gates it replaces.
    """
    separate_gamma, separate_terms = 1.0, 1
    for gate in gates:
        gate_gamma, gate_terms = _cost(gate)
        separate_gamma *= gate_gamma
        separate_terms *= gate_terms
    merged_gamma, merged_terms = _cost(merged)
    return merged_gamma <= separate_gamma + 1e-9 and merged_terms <= separate_terms


def _blocks_on_pair(
    circuit: QuantumCircuit, pair: tuple[int, int], claimed: set[int]
) -> list[tuple[list[int], bool]]:
    """Return the maximal runs of instruction indices acting only within ``pair``.

    Each run comes with a flag saying whether anything was slid past it, which decides
    where the merged gate goes.

    An instruction touching neither qubit of the pair is skipped, since it acts on
    disjoint qubits and so commutes with the whole run. An instruction touching one of
    the pair's qubits but reaching outside it is slid past if it commutes with the
    members that would cross it, and otherwise ends the run — as do measurements,
    barriers, wire cut markers, and anything another pair's run already claimed. That
    last case matters for a single-qubit gate on a qubit shared by two pairs, which
    would otherwise be absorbed into both runs and applied twice.
    """
    pair_set = set(pair)
    blocks: list[tuple[list[int], bool]] = []
    current: list[int] = []
    moved = False
    for index, instruction in enumerate(circuit.data):
        # Ordered, because the commutation check needs the gate's argument order: cx on
        # (1, 2) commutes with a Z-diagonal gate on qubit 1 and cx on (2, 1) does not.
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        touched = set(qubits)
        if not touched & pair_set:
            continue
        absorbable = (
            touched <= pair_set
            and not instruction.clbits
            and index not in claimed
            and instruction.operation.name not in ("barrier", "measure", "Cut")
        )
        if absorbable:
            current.append(index)
            continue
        suffix = _commuting_suffix(circuit, instruction, qubits, current)
        if suffix is not None:
            prefix = current[: len(current) - len(suffix)]
            if prefix:
                blocks.append((prefix, moved))
            current, moved = suffix, True
            continue
        if current:
            blocks.append((current, moved))
            current, moved = [], False
    if current:
        blocks.append((current, moved))
    return blocks


def _merge_block(
    circuit: QuantumCircuit, indices: list[int], pair: tuple[int, int]
) -> Gate:
    """Compose the instructions at ``indices`` into a single two-qubit gate."""
    local = {pair[0]: 0, pair[1]: 1}
    block = QuantumCircuit(2)
    for index in indices:
        instruction = circuit.data[index]
        gate = marker_gate(instruction.operation) or instruction.operation
        block.append(
            gate, [local[circuit.find_bit(q).index] for q in instruction.qubits]
        )
    return UnitaryGate(np.asarray(Operator(block).data), label="merged")


def _plan_merges(
    circuit: QuantumCircuit,
    pairs: dict[frozenset, tuple[int, int]],
    marked: set[frozenset],
) -> dict[int, tuple[Instruction, tuple[int, int]] | None]:
    """Decide which instructions to merge.

    Maps an instruction index either to the gate replacing it and the qubits it goes on,
    or to None meaning the instruction is absorbed and dropped.
    """
    replacements: dict[int, tuple[Instruction, tuple[int, int]] | None] = {}
    claimed: set[int] = set()
    for key, pair in pairs.items():
        for block, moved in _blocks_on_pair(circuit, pair, claimed):
            gates = [
                marker_gate(circuit.data[i].operation) or circuit.data[i].operation
                for i in block
            ]
            two_qubit = [gate for gate in gates if gate.num_qubits == 2]
            if len(two_qubit) < 2:
                # Absorbing single-qubit gates into a lone two-qubit gate cannot
                # change gamma, which depends only on the KAK coordinates, but it would
                # replace a named gate by a generic unitary and lose its QPD table.
                continue
            merged = _merge_block(circuit, block, pair)
            if not _worth_merging(two_qubit, merged):
                continue
            # The block may start with a single-qubit gate, so the merged gate goes
            # on the pair rather than on the first instruction's qubits. If anything was
            # slid past, the members before it have moved forward across it, so the
            # merged gate belongs at the end of the run rather than the start.
            anchor = block[-1] if moved else block[0]
            replacements[anchor] = (
                CutTwoQubitGate(merged) if key in marked else merged,
                pair,
            )
            for index in block:
                if index != anchor:
                    replacements[index] = None
            claimed.update(block)
    return replacements


def consolidate_two_qubit_blocks(
    circuit: QuantumCircuit, restrict_to: set[frozenset] | None = None
) -> QuantumCircuit:
    """Merge runs of gates acting on the same qubit pair into single unitaries.

    Args:
        circuit: circuit to rewrite. Two-qubit cut markers are unwrapped, merged and
            marked again. Wire cut markers end a run and are left in place.
        restrict_to: if given, only these qubit pairs are considered. Pass the marked
            pairs to leave gates alone that were never going to be cut.

    Returns:
        An equivalent circuit. Returns the input unchanged if nothing was worth merging.
    """
    pairs: dict[frozenset, tuple[int, int]] = {}
    for instruction in circuit.data:
        if instruction.operation.num_qubits != 2 or instruction.clbits:
            continue
        indices = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        pairs.setdefault(frozenset(indices), indices)

    if restrict_to is not None:
        pairs = {key: value for key, value in pairs.items() if key in restrict_to}

    marked = {
        frozenset(circuit.find_bit(q).index for q in instruction.qubits)
        for instruction in circuit.data
        if marker_gate(instruction.operation) is not None
    }

    replacements = _plan_merges(circuit, pairs, marked)

    if not replacements:
        return circuit

    out = circuit.copy_empty_like()
    for index, instruction in enumerate(circuit.data):
        if index not in replacements:
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
            continue
        entry = replacements[index]
        if entry is not None:
            gate, pair = entry
            out.append(gate, [out.qubits[i] for i in pair])
    logger.debug(
        "merged %d instruction(s) into %d gate(s)",
        len(replacements),
        sum(1 for gate in replacements.values() if gate is not None),
    )
    return out
