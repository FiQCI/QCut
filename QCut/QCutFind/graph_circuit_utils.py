"""
Utility functions for converting quantum circuits to graph representations and
vice versa, as well as functions for updating node and edge information in the graph.
"""

import math

import rustworkx as rx

from QCut.qpd.qpd_generate import gamma as qpd_gamma
from QCut.qpd.qpd_generate import qpd_from_gate
from QCut.qpd.qpd_operations import QPD_REGISTRY

#: METIS takes integer edge weights, so a weight is scaled before rounding. The scale
#: has to resolve the gap between cheap cuts, since log(1.05) is only 0.049.
WEIGHT_SCALE: int = 10000

#: Stands in for "this cut is not allowed in the current mode". Must stay far above the
#: total weight of any real cut set so METIS never prefers it.
DISALLOWED_WEIGHT: int = 100000000000

#: gamma of the identity-channel decomposition used for a wire cut.
WIRE_GAMMA: int = 4


def cut_weight(gamma: float) -> int:
    """Turn a sampling overhead into an edge weight METIS can minimise correctly.

    The overhead of a cut set is the *product* of its gammas, but METIS minimises the
    *sum* of the cut edge weights. Taking the logarithm turns one into the other, so
    ``log(gamma)`` is the weight that makes the partitioner optimise the real cost.
    Using gamma directly, as this used to, systematically undervalues cheap cuts. An
    ``rzz(0.3)`` at gamma 1.59 against a ``cz`` at gamma 3.00 reads as 1.9 times cheaper
    added up but is 2.4 times cheaper multiplied.

    A free cut at ``gamma = 1`` would weigh nothing, which is true of its shot cost but
    not of the circuits it still adds, so the weight floor is one.
    """
    return max(1, round(math.log(gamma) * WEIGHT_SCALE))


def weight_fn(edge_data):
    """
    Custom weight function for edges.
    """
    if len(edge_data) > 2:
        # your existing logic: tuple[3] is raw_weight
        return edge_data[3]  # Use the weight from the tuple
    # default weight
    return edge_data[1]


def _node_ids(qubit_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Two new node ids per two-qubit gate, one for each of its qubits in order.

    Ids reuse a qubit index while it is still free, and otherwise count up from the
    highest id handed out. Only the order of the ids matters, since METIS reads the
    nodes in that order, so the scheme is kept as it was.
    """
    used: set[int] = set()
    top = -1
    ids = []
    for q0, q1 in qubit_pairs:
        if q0 in used and q1 in used:
            a, b = top + 1, top + 2
        elif q0 in used:
            a, b = q1, max(top, q1) + 1
        elif q1 in used:
            a, b = q0, max(top, q0) + 1
        else:
            a, b = q0, q1
        used.update((a, b))
        top = max(top, a, b)
        ids.append((a, b))
    return ids


def get_gate_weight(gate_name, mode="gate", gate=None):
    """
    Get the weight of a gate based on its name and the specified mode
    (gate or wire or both).

    The weight is ``log(gamma)`` scaled by ``WEIGHT_SCALE``, see :func:`cut_weight`.

    Args:
        gate_name: The name of the gate.
        mode: The mode for calculating weight ("gate", "wire", or "both").
        gate: The gate itself, required for gates with no hand-written QPD.
    Returns:
        The weight of the gate.
    """
    if mode == "wire":
        return DISALLOWED_WEIGHT
    if gate_name in QPD_REGISTRY:
        gamma = sum(abs(term["c"]) for term in QPD_REGISTRY[gate_name])
    elif gate is not None:
        gamma = qpd_gamma(qpd_from_gate(gate))
    else:
        raise ValueError(
            f"Gate {gate_name} has no QPD in QPD_REGISTRY and no gate was supplied to "
            "generate one from."
        )
    return cut_weight(gamma)


def get_wire_weight(mode="wire"):
    """
    Get the weight of a wire based on the specified mode (gate or wire or both).
    Args:
        mode: The mode for calculating weight ("gate", "wire", or "both").
    Returns:
        The weight of the wire.
    """
    if mode == "gate":
        return DISALLOWED_WEIGHT
    return cut_weight(WIRE_GAMMA)


def circ_to_graph(circuit, mode="both"):
    """
    Convert a quantum circuit to a graph representation.

    Each two-qubit gate is an edge between two new nodes, one per qubit, and each
    qubit's consecutive nodes are joined by a wire edge.

    Args:
        circuit: A quantum circuit object.

    Returns:
        A graph representation of the circuit.
    """
    gates = [
        (index, inst.operation, [circuit.find_bit(q).index for q in inst.qubits])
        for index, inst in enumerate(circuit.data)
        if inst.operation.num_qubits == 2
    ]
    ids = _node_ids([tuple(qubits) for _, _, qubits in gates])

    G = rx.PyGraph()
    G.add_nodes_from(range(max(node for pair in ids for node in pair) + 1))

    nodes_on_qubit: dict[int, list[int]] = {}
    for (index, operation, qubits), new in zip(gates, ids):
        # The operation is carried so get_gate_weight can work out gamma for gates
        # with no hand-written QPD.
        weight = get_gate_weight(operation.name, mode, operation)
        G.add_edge(*new, (index, operation.name, qubits, weight))
        for qubit, node in zip(qubits, new):
            previous = nodes_on_qubit.setdefault(qubit, [])
            if previous:
                G.add_edge(node, previous[-1], (qubit, get_wire_weight(mode)))
            previous.append(node)

    # Remove nodes with degree 0 (not connected to anything)
    isolated_nodes = [node for node in G.node_indices() if G.degree(node) == 0]

    G.remove_nodes_from(isolated_nodes)
    G, nodes_on_qubit = final_graph(G, nodes_on_qubit)
    return G, nodes_on_qubit


def final_graph(graph, nodes_on_qubit):
    # Relabel nodes to ensure consecutive numbering
    # Since rustworkx doesn't have relabel_nodes, we need to create a new graph manually
    old_to_new = {
        old_idx: new_idx for new_idx, old_idx in enumerate(graph.node_indices())
    }

    # Create a new graph
    new_graph = rx.PyGraph()

    # Add nodes with consecutive indices (0, 1, 2, ...)
    for i in range(len(graph.node_indices())):
        new_graph.add_node(i)

    # Add edges with remapped indices
    for edge in graph.edge_list():
        old_source, old_target = edge
        new_source = old_to_new[old_source]
        new_target = old_to_new[old_target]
        edge_data = graph.get_edge_data(old_source, old_target)
        new_graph.add_edge(new_source, new_target, edge_data)

    # Update nodes_on_qubit dictionary with new node indices
    updated_nodes_on_qubit = {}
    for qubit, old_node_list in nodes_on_qubit.items():
        updated_nodes_on_qubit[qubit] = [
            old_to_new[old_node] for old_node in old_node_list if old_node in old_to_new
        ]

    return new_graph, updated_nodes_on_qubit
