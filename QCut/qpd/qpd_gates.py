"""Helper gates for circuit knitting."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, QuantumRegister
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import BasisTranslator, UnrollCustomDefinitions


def _marker(name: str, num_qubits: int) -> Instruction:
    """An opaque instruction marking where a cut goes."""
    marker = QuantumCircuit(num_qubits, name=name).to_instruction(label=name)
    marker.definition = None
    return marker


cut_op = _marker("Cut", 1)
cutCZ_op = _marker("CutCZ", 2)
cutSWAP_op = _marker("CutSWAP", 2)
cutISWAP_op = _marker("CutISWAP", 2)


def cut() -> QuantumCircuit | Instruction:
    """Return a single qubit wire cut instruction."""
    return cut_op


def cutCZ() -> QuantumCircuit | Instruction:
    """Return a two qubit cutCZ gate instruction."""
    return cutCZ_op


def cutSWAP() -> QuantumCircuit | Instruction:
    """Return a two qubit cutSWAP gate instruction."""
    return cutSWAP_op


def cutISWAP() -> QuantumCircuit | Instruction:
    """Return a two qubit cutISWAP gate instruction."""
    return cutISWAP_op


QPD_DECOMPOSITION_SQ_BASIS: list[str] = ["u"]

QPD_GATE_REGISTRY: dict[str, Instruction] = {
    "cz": cutCZ_op,
    "swap": cutSWAP_op,
    "iswap": cutISWAP_op,
}


class CutTwoQubitGate(Instruction):
    """Opaque cut marker for an arbitrary two-qubit gate, carrying the gate itself.

    The markers above identify themselves by name alone, which is enough while every
    cuttable gate has a hand-written QPD. A generated QPD needs the gate's matrix, so
    this marker carries the gate to QPD-expansion time.

    The name is ``Cut<GATE>`` so that ``_get_cut_locations`` and ``_insert_cut_nodes``
    keep working unchanged. The wrapped gate's parameters are re-exposed as this
    instruction's own ``params``, without which Qiskit could not bind them.
    """

    def __init__(self, gate: Gate):
        """Wrap ``gate`` in a cut marker."""
        if gate.num_qubits != 2:
            raise ValueError(
                f"CutTwoQubitGate needs a two-qubit gate, got {gate.name} on "
                f"{gate.num_qubits} qubits."
            )
        super().__init__(
            name=f"Cut{gate.name.upper()}",
            num_qubits=2,
            num_clbits=0,
            params=list(gate.params),
        )
        self._base_gate = gate
        self._opaque = True
        self.definition = None

    @property
    def gate(self) -> Gate:
        """The wrapped gate, carrying the marker's current parameter values."""
        if not self.params:
            return self._base_gate
        bound = self._base_gate.copy()
        bound.params = list(self.params)
        return bound

    def __repr__(self) -> str:
        """Represent as string."""
        return self.name


class _ReplaceWithCutGates(TransformationPass):
    def __init__(self, registry: dict[str, Instruction]):
        self.registry = registry
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if node.op.name not in self.registry:
                continue
            cut_op = self.registry[node.op.name]
            mini_dag = DAGCircuit()
            qreg = QuantumRegister(node.op.num_qubits)
            mini_dag.add_qreg(qreg)
            mini_dag.apply_operation_back(cut_op, list(qreg))
            dag.substitute_node_with_dag(node, mini_dag)
        return dag


def cutGate(
    gate: Gate,
    control: int | list[int],
    target: int | list[int],
    single_qubit_basis: list[str] | None = None,
) -> dict:
    """Return a CutGate circuit decomposing the input gate with QPD cut markers.

    control and target together define the ordered physical qubit mapping for
    gate inputs 0..n-1: gate input i maps to physical qubit (controls + targets)[i].

    A two-qubit gate becomes a single :class:`CutTwoQubitGate` marker, so its QPD is
    generated from its matrix at expansion time. Larger gates are still transpiled into
    the ``{u, cz, swap, iswap}`` basis first, which turns them into several cuts.

    Args:
        gate (Gate): the gate to cut.
        control (int | list[int]): qubit, or qubits, the gate's first inputs map to.
        target (int | list[int]): qubit, or qubits, its remaining inputs map to.
        single_qubit_basis (list[str]): basis the single-qubit parts are translated
            into, if the default is not wanted (optional).

    Returns:
        dict: keyword arguments for ``QuantumCircuit.append``, so the call site reads
        ``circuit.append(**cutGate(...))``.
    """
    controls = [control] if isinstance(control, int) else list(control)
    targets = [target] if isinstance(target, int) else list(target)
    all_qargs = controls + targets

    if gate.num_qubits < 2:
        raise ValueError("Input gate must have at least 2 qubits.")
    if len(all_qargs) != gate.num_qubits:
        raise ValueError(
            f"Expected {gate.num_qubits} qubit arguments, got {len(all_qargs)}."
        )
    if len(set(all_qargs)) != len(all_qargs):
        raise ValueError("Qubit arguments must be unique.")
    if any(q < 0 for q in all_qargs):
        raise ValueError("Qubit arguments must be non-negative.")

    if gate.num_qubits == 2:
        return {"instruction": CutTwoQubitGate(gate), "qargs": all_qargs}

    qc = QuantumCircuit(gate.num_qubits, name=f"cut{gate.name.upper()}")
    qc.append(gate, list(range(gate.num_qubits)))

    sq_basis = (
        single_qubit_basis
        if single_qubit_basis is not None
        else QPD_DECOMPOSITION_SQ_BASIS
    )
    basis = sq_basis + list(QPD_GATE_REGISTRY.keys())
    tr = PassManager(
        [
            UnrollCustomDefinitions(SessionEquivalenceLibrary, basis_gates=basis),
            BasisTranslator(SessionEquivalenceLibrary, basis),
            _ReplaceWithCutGates(QPD_GATE_REGISTRY),
        ]
    ).run(qc)

    return {
        "instruction": tr.to_instruction(label="CutGate"),
        "qargs": all_qargs,
    }


def _seq(name: str, *gates: str) -> QuantumCircuit:
    """A one-qubit circuit applying ``gates`` in order, measuring into its one bit."""
    circuit = QuantumCircuit(1, int("measure" in gates), name=name)
    for gate in gates:
        if gate == "measure":
            circuit.measure(0, 0)
        else:
            getattr(circuit, gate)(0)
    return circuit


# measurements in different bases
xmeas = _seq("x-meas", "h", "measure", "h")
ymeas = _seq("y-meas", "sdg", "h", "measure", "h", "s")
idmeas = _seq("id-meas", "id")
zmeas = _seq("z-meas", "measure")

# initialisations
zero_init = _seq("0-init", "id")
one_init = _seq("1-init", "x")
plus_init = _seq("'+'-init", "h")
minus_init = _seq("'-'-init", "h", "z")
i_plus_init = _seq("'i+'-init", "h", "s")
i_minus_init = _seq("'i-'-init", "h", "z", "s")

# local operations of the generated decompositions
i_gate = _seq("I", "id")
x_gate = _seq("X", "x")
y_gate = _seq("Y", "y")
z_gate = _seq("Z", "z")
axyp = _seq("Axyp", "y", "s")
axym = _seq("Axym", "y", "s", "z")
ayzp = _seq("Ayzp", "z", "sx")
ayzm = _seq("Ayzm", "z", "sx", "x")
azxp = _seq("Azxp", "h")
azxm = _seq("Azxm", "h", "y")
sx_gate = _seq("SX", "sx")
sxdg_gate = _seq("SXdg", "sxdg")
s_gate = _seq("S", "s")
sdg_gate = _seq("Sdg", "sdg")
bxy = _seq("Bxy", "measure", "x")
byz = _seq("Byz", "h", "measure", "h", "y")
bzx = _seq("Bzx", "sx", "measure", "sxdg", "z")

ry_p = QuantumCircuit(1, name="RY+")
ry_p.ry(np.pi / 2, 0)

ry_m = QuantumCircuit(1, name="RY-")
ry_m.ry(-np.pi / 2, 0)
