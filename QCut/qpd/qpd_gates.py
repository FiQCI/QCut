"""Helper gates for circuit knitting."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, QuantumRegister
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import BasisTranslator, UnrollCustomDefinitions

# define the cut location marker
cut_op = QuantumCircuit(1, name="Cut")
cut_op = cut_op.to_instruction(label="Cut")
cut_op.definition = None


def cut() -> QuantumCircuit | Instruction:
    """Return a single qubit wire cut instruction."""
    return cut_op


cutCZ_op = QuantumCircuit(2, name="CutCZ")
cutCZ_op = cutCZ_op.to_instruction(label="CutCZ")
cutCZ_op.definition = None

cutSWAP_op = QuantumCircuit(2, name="CutSWAP")
cutSWAP_op = cutSWAP_op.to_instruction(label="CutSWAP")
cutSWAP_op.definition = None

cutISWAP_op = QuantumCircuit(2, name="CutISWAP")
cutISWAP_op = cutISWAP_op.to_instruction(label="CutISWAP")
cutISWAP_op.definition = None


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


# define measurements for different bases
xmeas = QuantumCircuit(1, 1, name="x-meas")
xmeas.h(0)
xmeas.measure(0, 0)
xmeas.h(0)

ymeas = QuantumCircuit(1, 1, name="y-meas")
ymeas.sdg(0)
ymeas.h(0)
ymeas.measure(0, 0)
ymeas.h(0)
ymeas.s(0)

idmeas = QuantumCircuit(1, name="id-meas")
idmeas.id(0)

zmeas = QuantumCircuit(1, 1, name="z-meas")
zmeas.measure(0, 0)

# define initialization operations
zero_init = QuantumCircuit(1, name="0-init")
zero_init.id(0)

one_init = QuantumCircuit(1, name="1-init")
one_init.x(0)

plus_init = QuantumCircuit(1, name="'+'-init")
plus_init.h(0)

minus_init = QuantumCircuit(1, name="'-'-init")
minus_init.h(0)
minus_init.z(0)

i_plus_init = QuantumCircuit(1, name="'i+'-init")
i_plus_init.h(0)
i_plus_init.s(0)

i_minus_init = QuantumCircuit(1, name="'i-'-init")
i_minus_init.h(0)
i_minus_init.z(0)
i_minus_init.s(0)

i_gate = QuantumCircuit(1, name="I")
i_gate.id(0)

x_gate = QuantumCircuit(1, name="X")
x_gate.x(0)

y_gate = QuantumCircuit(1, name="Y")
y_gate.y(0)

z_gate = QuantumCircuit(1, name="Z")
z_gate.z(0)

axyp = QuantumCircuit(1, name="Axyp")
axyp.y(0)
axyp.s(0)

axym = QuantumCircuit(1, name="Axym")
axym.y(0)
axym.s(0)
axym.z(0)

ayzp = QuantumCircuit(1, name="Ayzp")
ayzp.z(0)
ayzp.sx(0)

ayzm = QuantumCircuit(1, name="Ayzm")
ayzm.z(0)
ayzm.sx(0)
ayzm.x(0)

azxp = QuantumCircuit(1, name="Azxp")
azxp.h(0)

azxm = QuantumCircuit(1, name="Azxm")
azxm.h(0)
azxm.y(0)

sx_gate = QuantumCircuit(1, name="SX")
sx_gate.sx(0)

sxdg_gate = QuantumCircuit(1, name="SXdg")
sxdg_gate.sxdg(0)

ry_p = QuantumCircuit(1, name="RY+")
ry_p.ry(np.pi / 2, 0)

ry_m = QuantumCircuit(1, name="RY-")
ry_m.ry(-np.pi / 2, 0)

s_gate = QuantumCircuit(1, name="S")
s_gate.s(0)

sdg_gate = QuantumCircuit(1, name="Sdg")
sdg_gate.sdg(0)

bxy = QuantumCircuit(1, 1, name="Bxy")
bxy.measure(0, 0)
bxy.x(0)

byz = QuantumCircuit(1, 1, name="Byz")
byz.h(0)
byz.measure(0, 0)
byz.h(0)
byz.y(0)

bzx = QuantumCircuit(1, 1, name="Bzx")
bzx.sx(0)
bzx.measure(0, 0)
bzx.sxdg(0)
bzx.z(0)
