"""
A module for preparing the circuit for experiment generation and the proper
circuit knitting workflow, including finding cut locations,
inserting placeholder operations, and separating into subcircuits.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import cast

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from QCut.errors.qcuterror import QCutError
from QCut.model.cutcircuit import CutCircuit
from QCut.model.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.options import CutOptions, resolve
from QCut.pre_process.consolidate import consolidate_two_qubit_blocks, marker_gate
from QCut.utils.circuit import NonCommutingGate

logger: logging.Logger = logging.getLogger(__name__)


def _final_measurements(
    circuit: QuantumCircuit, stripped: QuantumCircuit | None = None
) -> tuple[tuple[int, int], ...]:
    """The measurements stripping would remove, as ``(qubit, clbit)`` indices.

    In circuit order. ``stripped`` is the circuit with its final measurements already
    removed, if the caller has it; it must be a copy of ``circuit``, so that the two
    share their bits.
    """
    if stripped is None:
        stripped = cast(
            QuantumCircuit, circuit.remove_final_measurements(inplace=False)
        )
    remaining = Counter(
        (instruction.qubits[0], instruction.clbits[0])
        for instruction in stripped.data
        if instruction.operation.name == "measure"
    )
    final = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        bits = (instruction.qubits[0], instruction.clbits[0])
        # A measurement left behind is not final. Any on the same bits come before
        # the final one, so they are the ones to pass over.
        if remaining[bits]:
            remaining[bits] -= 1
            continue
        final.append((circuit.find_bit(bits[0]).index, circuit.find_bit(bits[1]).index))
    return tuple(final)


def _remember_final_measurements(
    cut_circuit: CutCircuit,
    circuit: QuantumCircuit,
    stripped: QuantumCircuit | None = None,
) -> CutCircuit:
    """Record on ``cut_circuit`` what ``circuit`` measured at its end."""
    cut_circuit.final_measurements = _final_measurements(circuit, stripped)
    cut_circuit.uncut_num_clbits = circuit.num_clbits
    return cut_circuit


def _get_cut_locations(circuit):
    """Remove the cut markers from ``circuit`` and return where they stood.

    A location's index is its marker's position once the markers before it are gone.
    """
    cut_locations = []
    markers = []
    for position, instruction in enumerate(circuit.data):
        operation = instruction.operation
        if "Cut" not in operation.name:
            continue
        index = position - len(markers)
        markers.append(position)
        qubits = [circuit.find_bit(qubit).registers[0] for qubit in instruction.qubits]
        if len(qubits) == 1:
            cut_locations.append(SingleQubitCutLocation((qubits[0], index)))
        else:
            # CutTwoQubitGate markers carry the gate so a QPD can be generated from
            # it. The per-gate markers do not need to.
            cut_locations.append(
                CutLocation(
                    (qubits, index),
                    gate_name=operation.name.replace("Cut", "").lower(),
                    gate=getattr(operation, "gate", None),
                )
            )
    for position in reversed(markers):
        del circuit.data[position]
    return cut_locations


def _insert_cut_nodes(circuit, cut_locations):
    """Put each cut's two placeholders where its marker stood."""
    offset = 0
    for cut_index, location in enumerate(cut_locations):
        if isinstance(location, SingleQubitCutLocation):
            names, wires = ("Meas", "Init"), [location.qubits] * 2
        else:
            prefix = f"cut{location.gate_name.upper()}"
            names, wires = (f"{prefix}_c", f"{prefix}_t"), location.qubits
        for name, (register, bit) in zip(names, wires):
            circuit.data.insert(
                location.index + offset,
                CircuitInstruction(
                    NonCommutingGate(f"{name}_{cut_index}"), [Qubit(register, bit)]
                ),
            )
            offset += 1
    return circuit


def _move_to_new_wire(orig: QuantumCircuit) -> QuantumCircuit:
    """Copy ``orig``, moving each qubit onto a fresh wire after every ``Meas``."""
    new = QuantumCircuit(name=orig.name + "_rebuilt")
    for creg in orig.cregs:
        new.add_register(ClassicalRegister(creg.size, creg.name))
    for qreg in orig.qregs:
        new.add_register(QuantumRegister(0, qreg.name))
    qubit_map = {qubit: Qubit() for qubit in orig.qubits}
    new.add_bits(list(qubit_map.values()))

    offset = 0
    for instruction in orig.data:
        qubits = instruction.qubits
        new.append(
            instruction.operation, [qubit_map[q] for q in qubits], instruction.clbits
        )
        if instruction.operation.name.startswith("Meas"):
            # Later uses of the qubit go on a fresh wire, placed right after this one.
            fresh = Qubit()
            new.add_bits([fresh])
            new.qubits.remove(fresh)
            offset += 1
            new.qubits.insert(orig.find_bit(qubits[0]).index + offset, fresh)
            qubit_map[qubits[0]] = fresh

    return new


def _separate_subcircuits(circuit):
    """The circuit's disconnected parts, each as a circuit of its own."""
    parts = circuit_to_dag(circuit).separable_circuits(remove_idle_qubits=True)
    return [circ for circ in map(dag_to_circuit, parts) if circ.num_qubits]


def _add_cbits(subcircuits):
    for circ in subcircuits:
        clbits = 0
        clbits_qpd = 0
        for i in circ:
            name = i.operation.name
            if "Meas" in name:
                clbits_qpd += 1
            elif "cut" in name:
                clbits += 1
                clbits_qpd += 1

        # A register of no bits still counts as a register, and a backend that refuses
        # jobs carrying one it cannot see written to -- IQM's does -- rejects the whole
        # experiment over it. Since it holds no bits, leaving it out moves nothing.
        meas_size = circ.num_qubits - clbits_qpd + clbits
        if clbits_qpd:
            circ.add_register(ClassicalRegister(clbits_qpd, "qpd_meas"))
        if meas_size:
            circ.add_register(ClassicalRegister(meas_size, "meas"))

    return subcircuits


def _record_spent_qubits(subcircuits):
    """Note which of each subcircuit's qubits end in a cut measurement.

    That is what the observable register was sized against, so it is what the rest of
    them have to be counted by. The wire a placeholder sits on will not do: routing is
    free to reuse a wire once a cut has ended what was on it, and then two placeholders
    share one wire and a qubit still to be measured looks like it is already spoken for.
    """
    for circ in subcircuits:
        circ.metadata = dict(circ.metadata or {})
        circ.metadata["qcut_spent"] = [
            circ.find_bit(instruction.qubits[0]).index
            for instruction in circ.data
            if "Meas" in instruction.operation.name
        ]
    return subcircuits


def get_qubit_map(subcircuits: list[QuantumCircuit]):
    map_qubit = {}
    count = 0
    for i in reversed(subcircuits):
        for j in sorted(
            [elem for elem in i.data if "obs" in elem.operation.name],
            key=lambda x: i.find_bit(x.qubits[0]).index,
            reverse=True,
        ):
            map_qubit[int(j.operation.name.split("_")[1])] = count
            count += 1

    return map_qubit


def _consolidate_marked_pairs(circuit: QuantumCircuit) -> QuantumCircuit:
    """Merge runs of gates on qubit pairs that carry a cut marker.

    Pairs with no marker are left alone. Merging them would not change any cutting cost
    and would only replace named gates by generic unitaries.
    """
    marked = {
        frozenset(circuit.find_bit(q).index for q in instruction.qubits)
        for instruction in circuit.data
        if marker_gate(instruction.operation) is not None
    }
    if not marked:
        return circuit
    return consolidate_two_qubit_blocks(circuit, restrict_to=marked)


def _split(
    prepared: QuantumCircuit,
    max_qubits: list[int] | None,
    options: CutOptions,
) -> CutCircuit:
    """Turn one marked circuit into subcircuits with placeholder operations."""
    from QCut.QCutFind import construct_final_subcircuits

    working = prepared.copy()
    for qubit in range(working.num_qubits):
        obs_m = QuantumCircuit(1, name=f"obs_{qubit}")
        working.append(obs_m.to_instruction(), [qubit])

    cut_locations = _get_cut_locations(working)
    circuit1 = _insert_cut_nodes(working, cut_locations)
    circuit_new = _move_to_new_wire(circuit1.copy())
    subcircuits = _add_cbits(_separate_subcircuits(circuit_new))

    fixed_circs = []
    for subcircuit in subcircuits:
        rebuilt = QuantumCircuit(subcircuit.num_qubits)
        # By name, not by position: a register of no bits is not created at all, so
        # which index holds which is not fixed.
        for register in subcircuit.cregs:
            rebuilt.add_register(register)
        for instruction in subcircuit.data:
            qubits = [
                rebuilt.qubits[subcircuit.qubits.index(q)] for q in instruction.qubits
            ]
            rebuilt.append(CircuitInstruction(instruction.operation, qubits))
        fixed_circs.append(rebuilt)

    fixed_circs = construct_final_subcircuits(fixed_circs, max_qubits)
    if len(fixed_circs) <= 1:
        raise QCutError(
            "Invalid cuts. Check documentation to see how cuts should be placed."
        )

    return CutCircuit(
        _record_spent_qubits(fixed_circs),
        cut_locations,
        get_qubit_map(fixed_circs),
        uncut_num_qubits=working.num_qubits,
        options=options,
    )


def _cheaper_split(
    unmerged: QuantumCircuit,
    merged: QuantumCircuit,
    max_qubits: list[int] | None,
    options: CutOptions,
) -> CutCircuit:
    """Split both ways and keep whichever plan costs less.

    Merging a run of gates on one pair lowers that pair's cost but can raise the total,
    because a merged run of gates about different axes is no longer a single-axis
    rotation and so can no longer join a joint decomposition. Which way wins depends on
    the whole circuit, so both plans are costed outright.
    """
    from QCut.qpd.qpd_operations import plan_cost

    candidates = []
    first_error = None
    for label, prepared in (("merged", merged), ("unmerged", unmerged)):
        try:
            split = _split(prepared, max_qubits, options)
        except QCutError as error:  # noqa: PERF203
            logger.debug("the %s plan does not split: %s", label, error)
            first_error = first_error or error
            continue
        candidates.append((plan_cost(split, options), label, split))

    if not candidates:
        # Neither plan works, so report why rather than inventing a new message.
        raise first_error or QCutError("the circuit could not be split")

    candidates.sort(key=lambda candidate: candidate[0])
    cost, label, chosen = candidates[0]
    if len(candidates) > 1:
        logger.info(
            f"Consolidation is {'on' if label == 'merged' else 'off'} for this run, "
            f"since the {label} plan costs gamma {cost:.4f} against "
            f"{candidates[1][0]:.4f} for the {candidates[1][1]} one."
        )
    return chosen


def get_locations_and_subcircuits(
    circuit: QuantumCircuit,
    max_qubits: list[int] | None = None,
    options: CutOptions | None = None,
) -> CutCircuit:
    """Get cut locations and subcircuits with placeholder operations.

    Args:
        circuit (QuantumCircuit): circuit with cuts inserted
        max_qubits (list[int], optional):
            list of maximum qubits per subcircuit when using automatic cut
            finding. If None, no constraint is used. Defaults to None.
            In general it is not necessary to manually specify this parameter.
        options (CutOptions, optional): configuration for the run. Defaults to
            QCut.options.DEFAULT_OPTIONS.

    Returns:
        CutCircuit: the subcircuits, the cut locations, and the map of subcircuit qubit
        indices to original circuit qubit indices.

    """

    options = resolve(options)
    prepared = circuit.copy()
    prepared.remove_final_measurements()
    stripped = prepared
    prepared = prepared.decompose(["CutGate"])

    # Consolidation has to happen before the obs_i tags go on, since those touch every
    # qubit and would end every run. Cut locations are recorded afterwards, so a merged
    # run counts as one cut.
    merged = None
    if options.consolidate_mode != "never":
        merged = _consolidate_marked_pairs(prepared)
        if merged is prepared:
            merged = None  # nothing was worth merging, so there is nothing to compare

    if merged is None:
        cut_circuit = _split(prepared, max_qubits, options)
    elif options.consolidate_mode == "always":
        cut_circuit = _split(merged, max_qubits, options)
    else:
        cut_circuit = _cheaper_split(prepared, merged, max_qubits, options)

    locations = cut_circuit.cut_locations
    num_wire_cuts = len(
        [loc for loc in locations if isinstance(loc, SingleQubitCutLocation)]
    )
    logger.info(
        f"Found {len(locations)} cut locations"
        f"({num_wire_cuts})"
        f" wire cut(s) and {len(locations) - num_wire_cuts} gate cut(s))"
        f" and separated into {cut_circuit.num_subcircuits} subcircuits."
    )

    return _remember_final_measurements(cut_circuit, circuit, stripped)
