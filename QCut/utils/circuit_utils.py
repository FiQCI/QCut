"""
Utility functions for working with quantum circuits in the context of circuit
cutting and knitting.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Barrier, ControlFlowOp, Qubit
from qiskit.circuit.library import get_standard_gate_name_mapping


def _count_gates(circuit: QuantumCircuit) -> dict[Qubit, int]:
    """Count the number of gates acting on each qubit in a QuantumCircuit.

    Args:
        circuit (QuantumCircuit): The input quantum circuit.

    Returns:
        dict[Qubit, int]: A dictionary mapping each qubit to the number of gates
        acting on it.
    """
    gate_count = dict.fromkeys(circuit.qubits, 0)
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            gate_count[qubit] += 1

    return gate_count


def _fence_markers(circuit: QuantumCircuit, names: set[str]) -> QuantumCircuit:
    """Put a barrier either side of every placeholder, so transpiling cannot move it.

    The experiment builder walks a subcircuit's instructions in order and expects the
    placeholders where it left them. Transpiling is free to commute single-qubit gates
    past each other, and a placeholder looks like one, so without a fence the markers
    come back permuted and the walk misreads the circuit.

    Fencing is also the more honest instruction to give the transpiler. A placeholder
    stands for an operation that has not been chosen yet, so merging the gates on either
    side of it across the gap is not a valid simplification in the first place.
    """
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        fenced = instruction.operation.name in names
        if fenced:
            out.barrier(instruction.qubits)
        out.append(instruction.operation, instruction.qubits, instruction.clbits)
        if fenced:
            out.barrier(instruction.qubits)
    return out


def _drop_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove the fences again, once transpilation can no longer reorder anything."""
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name != "barrier":
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
    return out


def markers_to_barriers(circuit: QuantumCircuit, names: set[str]) -> QuantumCircuit:
    """Swap placeholder instructions for barriers carrying their name as a label.

    A placeholder is an opaque one-qubit instruction, so a transpiler that builds its
    own target has no way to be told about it and refuses to synthesise it. IQM's does
    exactly that. A barrier is a directive every transpiler passes through untouched,
    and it can carry a label, so the placeholder survives and comes back afterwards.

    Barriers also stop the surrounding gates being merged across the gap, which is the
    right instruction anyway: the placeholder stands for an operation not yet chosen.
    """
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name in names:
            out.append(Barrier(1, label=instruction.operation.name), instruction.qubits)
        else:
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
    return out


def barriers_to_markers(circuit: QuantumCircuit, names: set[str]) -> QuantumCircuit:
    """Turn the labelled barriers back into placeholders, once transpiling is done."""
    from QCut.cutting.circuit_preparation import NonCommutingGate

    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        label = instruction.operation.label
        if instruction.operation.name == "barrier" and label in names:
            out.append(NonCommutingGate(label), instruction.qubits)
        else:
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
    return out


def compact_qpd_register(circuit: QuantumCircuit) -> tuple[QuantumCircuit, int]:
    """Drop ``qpd_meas`` when this circuit writes nothing to it.

    The register is sized when the subcircuits are built, for the most measurements any
    QPD term on that subcircuit could need. A term often needs none -- the identity term
    measures nothing -- and a classical register that nothing writes to is not something
    every backend accepts: IQM's refuses the job outright.

    Only the all-or-nothing case is handled, because that is the one that occurs: a
    term either measures on this subcircuit or it does not. Renumbering a partly used
    register would also move the bits a communicating wire cut recorded as its label.

    What is lost is the sign those bits carried. An unwritten bit reads as zero and the
    estimator maps zero to -1, so the number dropped is returned and put back later.

    Returns:
        The circuit and how many bits were dropped.
    """
    register = next((reg for reg in circuit.cregs if reg.name == "qpd_meas"), None)
    if register is None or register.size == 0:
        return circuit, 0

    offset = circuit.find_bit(register[0]).index
    used = any(
        offset <= circuit.find_bit(clbit).index < offset + register.size
        for instruction in circuit.data
        for clbit in instruction.clbits
    )
    if used:
        return circuit, 0

    out = QuantumCircuit(*circuit.qregs, name=circuit.name)
    out.metadata = dict(circuit.metadata or {})
    out._layout = circuit.layout
    for other in circuit.cregs:
        if other.name != register.name:
            out.add_register(ClassicalRegister(other.size, other.name))
    for instruction in circuit.data:
        out.append(
            instruction.operation,
            instruction.qubits,
            [
                out.clbits[circuit.find_bit(clbit).index - register.size]
                for clbit in instruction.clbits
            ],
        )
    return out, register.size


def _record_layout(circuit: QuantumCircuit, num_logical: int) -> QuantumCircuit:
    """Note where the transpiler put each of a subcircuit's qubits. Move nothing.

    Transpiling lays a subcircuit out on physical qubits, chosen so that its two-qubit
    gates land on pairs the device actually couples. Renaming the wires afterwards, to
    put the subcircuit's own qubits back at 0, 1, 2 ... restores the order the estimator
    reads bits in, but throws that placement away: the gates end up on index pairs
    that mean nothing to the device, and a backend that checks -- IQM's does -- rejects
    the job for a gate on a locus it does not have.

    So the placement is left exactly as the transpiler made it, and the map from the
    subcircuit's own qubits to the wires holding them is recorded instead.
    :func:`QCut.circuit_knitting._finalize_subcircuit` measures through the map, which
    puts the bits in the order everything downstream expects without moving a gate.

    Wires the layout did not use, and wires routing borrowed, are simply not in the map,
    so nothing measures them.

    Args:
        circuit (QuantumCircuit): a transpiled circuit carrying a layout.
        num_logical (int): how many qubits it had before transpilation.

    Returns:
        QuantumCircuit: the same circuit, with ``qcut_layout`` in its metadata.
    """
    layout = circuit.layout
    if layout is None:
        return circuit
    circuit.metadata = dict(circuit.metadata or {})
    circuit.metadata["qcut_layout"] = list(layout.final_index_layout())[:num_logical]
    return circuit


def _remove_obsm(subcircuits: list[dict[int, QuantumCircuit]]):

    for obs_set in subcircuits:
        for ind, circ in obs_set.items():
            j = 0
            while j < len(circ.data):
                if "obs" in circ[j].operation.name:
                    circ.data.remove(circ[j])
                else:
                    j += 1


def _remove_obsm_2(subcircuits: list[QuantumCircuit]):

    for circ in subcircuits:
        j = 0
        while j < len(circ.data):
            if "obs" in circ[j].operation.name:
                circ.data.remove(circ[j])
            else:
                j += 1


@dataclass(frozen=True)
class MarkerSpan:
    """One bundle side's placeholders, and the single wider marker standing in for them.

    ``wires`` and the member names recorded alongside are both in bundle-qubit order, so
    expanding the marker again puts each cut's placeholder back on the wire whose
    operation it stands for.
    """

    name: str
    wires: tuple[int, ...]
    indices: tuple[int, ...]
    at: int


def fuse_markers(circuit: QuantumCircuit, spans, gate_for) -> QuantumCircuit:
    """Replace each span's one-qubit placeholders by one marker as wide as its block.

    A bundle's operations act on one qubit per cut at once, but its placeholders are one
    qubit each, so the transpiler sees unrelated single-qubit gates and has no reason to
    lay them out on wires the device couples. The block then goes in after transpilation
    on whatever wires they landed on, with nothing to route it.

    One marker of the block's real width says what is actually coming. Routing has to
    put it on a locus the device has, and because it is a single instruction nothing can
    be scheduled inside it: the cuts it covers stay simultaneous, which is what a block
    needs. :func:`split_markers` takes it apart again once transpiling is done.
    """
    covered: dict[int, MarkerSpan] = {}
    for span in spans:
        for index in span.indices:
            covered[index] = span

    out = circuit.copy_empty_like()
    for index, instruction in enumerate(circuit.data):
        span = covered.get(index)
        if span is None:
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
        elif index == span.at:
            out.append(gate_for(span), [out.qubits[wire] for wire in span.wires])
    return out


def split_markers(
    circuit: QuantumCircuit, members: dict[str, list[str]]
) -> QuantumCircuit:
    """Expand each block marker back into the placeholders it stood in for.

    They come back consecutive, on the wires the transpiler chose, in bundle-qubit
    order. Everything downstream reads placeholders one at a time, so this is what keeps
    the widening confined to transpilation.
    """
    from QCut.cutting.circuit_preparation import NonCommutingGate

    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in members:
            out.append(operation, instruction.qubits, instruction.clbits)
            continue
        for position, marker in enumerate(members[operation.name]):
            out.append(NonCommutingGate(marker), [instruction.qubits[position]])
    return out


@lru_cache(maxsize=1)
def _standard_gate_names() -> frozenset[str]:
    """Names qiskit defines itself, which therefore say what the operation does."""
    return frozenset(get_standard_gate_name_mapping())


def _param_key(param, cache: dict):
    """One instruction parameter, reduced to something hashable and comparable.

    Floats go in as they are. Equal QPD terms share one cached transpiled operation, so
    their parameters are bit-identical; rounding them would only ever merge two
    operations that are not the same one.
    """
    if isinstance(param, np.ndarray):
        return (param.shape, param.dtype.str, param.tobytes())
    if isinstance(param, QuantumCircuit):
        return circuit_fingerprint(param, cache)
    if isinstance(param, (int, float, complex, str)):
        return param
    # A ParameterExpression, or whatever else a gate chooses to carry. The type keeps
    # two different kinds apart and the string two values of one kind.
    return (type(param).__name__, str(param))


def _definition_key(operation, cache: dict):
    """What a non-standard operation does, since its name does not say.

    QPD operations are custom instructions built at expansion time, and their names
    collide across different definitions -- the local unitaries a generated KAK
    decomposition produces are the case that matters.
    """
    if isinstance(operation, ControlFlowOp):
        # The predicate is carried outside ``params``, so the blocks alone would make
        # ``c == 0`` and ``c == 1`` look like the same operation.
        predicate = getattr(operation, "condition", None)
        if predicate is None:
            predicate = getattr(operation, "target", None)
        return (
            str(predicate),
            tuple(circuit_fingerprint(block, cache) for block in operation.blocks),
        )
    if operation.name in _standard_gate_names():
        return None
    try:
        definition = operation.definition
    except Exception:  # noqa: BLE001
        # ``definition`` is a property that can synthesise. One that cannot be built
        # tells us nothing, and the operation's class and parameters still have to
        # match before anything is merged.
        return None
    return None if definition is None else circuit_fingerprint(definition, cache)


def _operation_key(operation, cache: dict):
    """A key for one operation, cached by identity.

    The same operation objects recur across every experiment group, so without the
    cache the definitions would be walked again for each. The object is stored next to
    its key because ``id()`` is only unique among live objects: holding a reference
    keeps this one alive, and the identity check catches a recycled address anyway.
    """
    hit = cache.get(id(operation))
    if hit is not None and hit[0] is operation:
        return hit[1]
    key = (
        operation.name,
        type(operation).__name__,
        operation.num_qubits,
        operation.num_clbits,
        tuple(_param_key(param, cache) for param in operation.params),
        _definition_key(operation, cache),
    )
    cache[id(operation)] = (operation, key)
    return key


def circuit_fingerprint(circuit: QuantumCircuit, cache: dict | None = None) -> tuple:
    """A hashable key, equal for circuits whose counts come out identically.

    Two circuits sharing a fingerprint are interchangeable as far as a run is
    concerned: they sample the same distribution and label it the same way. That is
    what lets a run submit one of them and hand the result to every slot the others
    would have filled.

    Register layout is part of the key rather than a detail, because it decides how a
    counts key is formatted -- one space-separated field per classical register -- and
    because a circuit that writes nothing to ``qpd_meas`` has had it dropped by
    :func:`compact_qpd_register`, which is a real difference in what comes back.

    The global phase, the name and the metadata are left out. Qiskit's own circuit
    equality ignores the last two; the phase is unobservable in counts, so excluding it
    only merges more. ``metadata`` carries the transpiler layout, which is consumed
    while the experiment circuits are being built and is vestigial by the time anything
    is submitted.

    Args:
        circuit (QuantumCircuit): the circuit to key.
        cache (dict): operation keys already computed, shared across a batch of
            circuits so the operations they have in common are walked once (optional).

    Returns:
        A tuple that can be hashed and compared.
    """
    if cache is None:
        cache = {}
    qubit_index = {bit: index for index, bit in enumerate(circuit.qubits)}
    clbit_index = {bit: index for index, bit in enumerate(circuit.clbits)}
    return (
        circuit.num_qubits,
        circuit.num_clbits,
        tuple((register.name, register.size) for register in circuit.qregs),
        tuple(
            (register.name, tuple(clbit_index[bit] for bit in register))
            for register in circuit.cregs
        ),
        tuple(
            (
                _operation_key(instruction.operation, cache),
                tuple(qubit_index[bit] for bit in instruction.qubits),
                tuple(clbit_index[bit] for bit in instruction.clbits),
            )
            for instruction in circuit.data
        ),
    )
