"""
Module for transforming measurement bases in quantum circuits
when generating experiment circuits, particularly for combining Pauli
operators into compatible measurement settings.
"""

from __future__ import annotations

from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import HGate, SdgGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass


def _apply_basis_change(
    mini_dag: DAGCircuit, basis_change: QuantumCircuit, qubit
) -> None:
    """Write out a basis change's own gates, rather than one instruction wrapping them.

    A wrapper hides them in its definition, under a name the device has never heard of.
    """
    for instruction in basis_change.data:
        mini_dag.apply_operation_back(instruction.operation, [qubit])
    mini_dag.global_phase += basis_change.global_phase


class _ModifyMeasurementBasis(TransformationPass):
    def __init__(
        self,
        measurement_settings: list[dict[int, str]],
        ops: dict[str, QuantumCircuit] | None = None,
    ):

        self.measurement_settings = measurement_settings
        self.ops = ops
        super().__init__()

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:

        no_obs = True
        cloned_dag = deepcopy(dag)
        for node in dag.op_nodes():
            if "obs" not in node.op.name:
                continue

            obs_ind = int(node.op.name.split("_")[-1])

            for setting in self.measurement_settings:
                if obs_ind not in setting:
                    # continue
                    dag.remove_op_node(node)
                    break

                ob = setting[obs_ind]
                no_obs = False

                mini_dag = DAGCircuit()
                register = QuantumRegister(1)
                mini_dag.add_qreg(register)

                if ob == "X":
                    if self.ops and "X-meas" in self.ops:
                        _apply_basis_change(mini_dag, self.ops["X-meas"], register[0])
                    else:
                        mini_dag.apply_operation_back(HGate(), [register[0]])
                elif ob == "Y":
                    if self.ops and "Y-meas" in self.ops:
                        _apply_basis_change(mini_dag, self.ops["Y-meas"], register[0])
                    else:
                        mini_dag.apply_operation_back(SdgGate(), [register[0]])
                        mini_dag.apply_operation_back(HGate(), [register[0]])

                dag.substitute_node_with_dag(node, mini_dag)

        if no_obs:
            return cloned_dag
        return dag


def _get_obs_subcircuits(
    subcircuits: list[QuantumCircuit],
    measurement_settings: list[dict[int, str]],
    ops: dict[str, QuantumCircuit] | None = None,
) -> list[dict[int, QuantumCircuit]]:
    pms = [
        PassManager([_ModifyMeasurementBasis([setting], ops)])
        for setting in measurement_settings
    ]
    obs_subcircuits = []
    for pm in pms:
        pm_circs = {}
        for ind, subcircuit in enumerate(subcircuits):
            modified_circuit = pm.run(subcircuit)
            if modified_circuit.num_qubits == 0:
                continue

            modified_circuit._layout = subcircuit.layout
            pm_circs[ind] = modified_circuit
        obs_subcircuits.append(pm_circs)
    return obs_subcircuits
