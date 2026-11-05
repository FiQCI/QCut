"""
Move routing for a circuit that still carries cut placeholders.
"""

from __future__ import annotations

import logging

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from QCut.errors.qcuterror import QCutError

logger: logging.Logger = logging.getLogger(__name__)

#: Register the probe measurements are written to. Named so it cannot collide with a
#: register QCut put there itself.
PROBE_REGISTER: str = "qcut_move_probe"


def _positions(bits) -> dict:
    """Where each bit sits in the circuit's own bit list.

    Not ``find_bit``, which answers from the registers. On a MOVE-routed circuit the two
    disagree, and it is this one that a submitted circuit is read by.
    """
    return {bit: index for index, bit in enumerate(bits)}


def _barrier_labels(circuit: QuantumCircuit) -> dict[int, list[str | None]]:
    """Every barrier's label, grouped by the wire it sits on, in per-wire order."""
    labels: dict[int, list[str | None]] = {}
    wires = _positions(circuit.qubits)
    for instruction in circuit.data:
        if instruction.operation.name != "barrier":
            continue
        if len(instruction.qubits) != 1:
            raise QCutError(
                "move routing needs every barrier on a single qubit, so its label can "
                f"be put back on the right wire, but one spans "
                f"{len(instruction.qubits)}"
            )
        labels.setdefault(wires[instruction.qubits[0]], []).append(
            instruction.operation.label
        )
    return labels


def _with_probe(circuit: QuantumCircuit) -> QuantumCircuit:
    """A copy with one throwaway measurement per qubit, to pin where each ends up."""
    probe = circuit.copy()
    register = ClassicalRegister(circuit.num_qubits, PROBE_REGISTER)
    probe.add_register(register)
    probe.measure(range(circuit.num_qubits), register)
    return probe


def _probe_layout(routed: QuantumCircuit, num_logical: int) -> list[int]:
    """Which wire each qubit ended on, read off where its probe measurement sits."""
    wires: dict[int, int] = {}
    qubits = _positions(routed.qubits)
    clbits = _positions(routed.clbits)
    for instruction in routed.data:
        if instruction.operation.name != "measure":
            continue
        wires[clbits[instruction.clbits[0]]] = qubits[instruction.qubits[0]]

    if sorted(wires) != list(range(num_logical)):
        raise QCutError(
            f"move routing returned {len(wires)} probe measurement(s) for "
            f"{num_logical} qubit(s), so where each qubit ended up is unknown"
        )
    return [wires[index] for index in range(num_logical)]


def _restore(
    routed: QuantumCircuit,
    labels: dict[int, list[str | None]],
    wires: list[int],
    registers: list[tuple[str, int]],
) -> QuantumCircuit:
    """
    Drop the probe measurements, put the labels and registers back.

    Rebuilt onto one register in wire order. The routed circuit's own ``ancilla,
    resonators, q`` would number the wires differently from the order they are in.
    """
    pending = {wires[qubit]: list(marks) for qubit, marks in labels.items()}
    order = _positions(routed.qubits)

    out = QuantumCircuit(QuantumRegister(routed.num_qubits, "q"), name=routed.name)
    for name, size in registers:
        out.add_register(ClassicalRegister(size, name))

    for instruction in routed.data:
        operation = instruction.operation
        if operation.name == "measure":
            continue
        qubits = [out.qubits[order[qubit]] for qubit in instruction.qubits]
        if operation.name == "barrier":
            wire = order[instruction.qubits[0]]
            if not pending.get(wire):
                raise QCutError(
                    f"move routing left a barrier on wire {wire}, which held no "
                    "placeholder, so which is which cannot be told"
                )
            label = pending[wire].pop(0)
            if label is not None:
                operation = operation.to_mutable()
                operation.label = label
        out.append(operation, qubits, [])

    if any(pending.values()):
        missing = {wire: marks for wire, marks in pending.items() if marks}
        raise QCutError(f"move routing dropped placeholder(s) {missing}")

    # Marks the circuit as laid out on physical wires: without it a backend, and the
    # QASM exporter, take the wires for logical qubits and place the gates elsewhere.
    out._layout = routed.layout
    return out


def move_route(
    circuit: QuantumCircuit,
    backend,
    transpile_to_iqm,
    **options,
) -> QuantumCircuit:
    """Transpile with MOVE routing, keeping the placeholders, registers and layout.

    Args:
        circuit: a subcircuit whose placeholders are labelled barriers.
        backend: the resonator backend to route for.
        transpile_to_iqm: IQM's ``transpile_to_IQM``, passed in so this module does not
            import an optional dependency.
        **options: forwarded to the transpiler.

    Returns:
        The routed circuit, with labels and registers restored and ``qcut_layout`` in
        its metadata.

    Raises:
        QCutError: routing left something that cannot be put back.
    """
    labels = _barrier_labels(circuit)
    registers = [(register.name, register.size) for register in circuit.cregs]
    num_logical = circuit.num_qubits

    routed = transpile_to_iqm(
        _with_probe(circuit), backend, perform_move_routing=True, **options
    )

    wires = _probe_layout(routed, num_logical)
    restored = _restore(routed, labels, wires, registers)
    restored.metadata = dict(circuit.metadata or {})
    restored.metadata["qcut_layout"] = wires

    logger.info(
        "move routed onto wires %s, restoring %d placeholder(s) and %d register(s)",
        wires,
        sum(len(marks) for marks in labels.values()),
        len(registers),
    )
    return restored
