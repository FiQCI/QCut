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
    index = 0  # index of the current instruction in circuit_data
    circuit_data = circuit.data
    cut_locations = []

    # loop through circuit instructions
    # if operation is a Cut() instruction remove it and add registers and
    # offset index to cut_locations

    # rename variables to be more descriptive (namely qs)
    while index < len(circuit):
        op = circuit_data[index]
        if "Cut" in op.operation.name:
            # find qubits for Cut operation
            qubits = [circuit.find_bit(qubit).registers[0] for qubit in op.qubits]

            # remove the cut operation
            circuit_data.remove(op)

            # append to cut_locations
            if len(qubits) == 1:
                cut_locations.append(SingleQubitCutLocation((qubits[0], index)))
            else:
                gate_name = op.operation.name.replace("Cut", "").lower()
                # CutTwoQubitGate markers carry the gate so a QPD can be generated
                # from it. The per-gate markers do not need to.
                cut_locations.append(
                    CutLocation(
                        (qubits, index),
                        gate_name=gate_name,
                        gate=getattr(op.operation, "gate", None),
                    )
                )

            # adjust index to account for removed operation
            index -= 1
        index += 1

    return cut_locations


def _insert_cut_nodes(circuit, cut_locations):
    circuit_data = circuit.data
    cut_index = 0
    offset = 0
    for cut_location in cut_locations:
        measure_node = NonCommutingGate(f"Meas_{cut_index}")

        initialize_node = NonCommutingGate(f"Init_{cut_index}")

        prefix = (
            f"cut{cut_location.gate_name.upper()}"
            if isinstance(cut_location, CutLocation)
            else ""
        )
        cut_czc = NonCommutingGate(f"{prefix}_c_{cut_index}")

        cut_czt = NonCommutingGate(f"{prefix}_t_{cut_index}")

        cut_index += 1

        cur_ops = (
            (measure_node, initialize_node)
            if isinstance(cut_location, SingleQubitCutLocation)
            else (cut_czc, cut_czt)
        )

        if isinstance(cut_location, SingleQubitCutLocation):
            for ph_op in cur_ops:
                circuit_data.insert(
                    cut_location.index + offset,
                    CircuitInstruction(
                        operation=ph_op,
                        qubits=[Qubit(cut_location.qubits[0], cut_location.qubits[1])],
                    ),
                )

                offset += 1

        else:
            circuit_data.insert(
                cut_location.index + offset,
                CircuitInstruction(
                    operation=cur_ops[0],
                    qubits=[
                        Qubit(cut_location.qubits[0][0], cut_location.qubits[0][1])
                    ],
                ),
            )

            offset += 1

            circuit_data.insert(
                cut_location.index + offset,
                CircuitInstruction(
                    operation=cur_ops[1],
                    qubits=[
                        Qubit(cut_location.qubits[1][0], cut_location.qubits[1][1])
                    ],
                ),
            )

            offset += 1

    return circuit


def _move_to_new_wire(orig: QuantumCircuit) -> QuantumCircuit:
    # Create the new circuit and add registers
    offset = 0
    new = QuantumCircuit(name=orig.name + "_rebuilt")
    for creg in orig.cregs:
        new.add_register(ClassicalRegister(creg.size, creg.name))
    for qreg in orig.qregs:
        new.add_register(QuantumRegister(0, qreg.name))

    # Create fresh Qubit objects for each original wire
    # and record a mapping old_qubit -> new_qubit
    qubit_map = {}
    new_qubits = []
    for idx, q_old in enumerate(orig.qubits):
        q_new = Qubit()
        new_qubits.append(q_new)
        qubit_map[q_old] = q_new
    new.add_bits(new_qubits)

    # 3) Replay every instruction, splitting on Measure
    for instruction in orig.data:
        inst, qargs, cargs = (
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )
        # map every qarg via our current mapping
        mapped_qs = [qubit_map[q] for q in qargs]

        if inst.name.startswith("Meas"):
            # append this measurement on the current wire
            new.append(inst, mapped_qs, cargs)

            # now allocate a fresh wire for all future uses of qargs[0]
            q_fresh = Qubit()
            new.add_bits([q_fresh])
            new.qubits.remove(q_fresh)
            new.qubits.insert(orig.find_bit(qargs[0]).index + 1 + offset, q_fresh)
            offset += 1
            # update the mapping so q_old -> q_fresh going forward
            qubit_map[qargs[0]] = q_fresh

        else:
            # just copy the gate over to mapped_qs
            new.append(inst, mapped_qs, cargs)

    return new


def _separate_subcircuits(circuit):
    dag = circuit_to_dag(circuit)

    circs = dag.separable_circuits(remove_idle_qubits=True)

    new_circs = []
    for i in circs:
        circ = dag_to_circuit(i)
        if len(circ.qubits) == 0:
            continue
        new_circs.append(circ)

    return new_circs


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
    def filter_obs_i(qc_data):
        return [i for i in qc_data if "obs" in i.operation.name]

    def sort_func(obs):
        return int(obs.operation.name.split("_")[1])

    map_qubit = {}
    count = 0
    for ind, i in enumerate(reversed(subcircuits)):
        for j in sorted(
            filter_obs_i(i.data),
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

    if len(fixed_circs) <= 1:
        raise QCutError(
            "Invalid cuts. Check documentation to see how cuts should be placed."
        )

    if max_qubits and len(fixed_circs) != len(max_qubits):
        fixed_circs = construct_final_subcircuits(fixed_circs, max_qubits)

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
