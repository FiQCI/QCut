"""
The main cut finding workflow for QCut.
"""

import logging

import rustworkx as rx
from qiskit import transpile
from qiskit.circuit import CircuitInstruction

from QCut.errors.qcuterror import QCutError
from QCut.options import CutOptions, resolve
from QCut.pre_process.circuit_preparation import get_locations_and_subcircuits
from QCut.pre_process.consolidate import consolidate_two_qubit_blocks
from QCut.QCutFind.graph_circuit_utils import circ_to_graph
from QCut.QCutFind.metis import (
    BUDGET_UFACTORS,
    FREE_UFACTOR,
    k_way_metis_partition,
    qubit_node_weights,
)
from QCut.QCutFind.refine import refine_cuts
from QCut.qpd.qpd_gates import QPD_GATE_REGISTRY, CutTwoQubitGate
from QCut.qpd.qpd_gates import cut_op as cut

#: Basis the circuit is transpiled into before the interaction graph is built. The
#: two-qubit entries stop transpilation from breaking apart gates that could be cut
#: whole, which would cost one cut per cz rather than one cut at the gate's own gamma.
BASIS_GATES = [
    "r",
    "u",
    "cz",
    "swap",
    "iswap",
    "cx",
    "cy",
    "ch",
    "ecr",
    "dcx",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "crx",
    "cry",
    "crz",
    "cp",
    "xx_plus_yy",
    "xx_minus_yy",
]

logger: logging.Logger = logging.getLogger(__name__)


def extract_cuts(graph, labels):  # noqa: C901
    cut_edges = []

    for u, v in graph.edge_list():
        if labels[u] != labels[v]:
            cut_edges.append((u, v))

    cut_data_test = []

    for edge in cut_edges:
        cut_data_test.append(graph.get_edge_data(edge[0], edge[1]))
        # Collect all unique nodes involved in cut_edges
    cut_nodes = set()
    cut_data = []
    for u, v in cut_edges:
        # Get all edges connected to u and v (excluding the cut edge itself)
        if len(graph.get_edge_data(u, v)) > 2:
            cut_data.append((u, v, graph.get_edge_data(u, v)))
            continue
        connected_edges = []
        for nbr in graph.neighbors(u):
            if nbr != v:
                data = graph.get_edge_data(u, nbr)
                if len(data) > 2:
                    connected_edges.append((u, nbr, data))
        for nbr in graph.neighbors(v):
            if nbr != u:
                data = graph.get_edge_data(v, nbr)
                if len(data) > 2:
                    connected_edges.append((v, nbr, data))
        # Find the edge with the smallest data[1]
        if len(connected_edges) > 0:
            filtered_edges = [
                edge
                for edge in connected_edges
                if graph.get_edge_data(u, v)[0] in edge[2][2]
            ]
            min_edge = min(filtered_edges, key=lambda x: x[2])
            cut_data.append(min_edge)

        cut_nodes.add(u)
        cut_nodes.add(v)

    return cut_data, cut_data_test


def insert_or_append(circuit, index, instruction):
    """
    Insert an instruction at a specific index in the circuit.
    If the index is out of bounds, append the instruction.
    """
    if index < 0 or index >= len(circuit.data):
        circuit.data.append(instruction)
    else:
        circuit.data.insert(index, instruction)


def add_cuts_to_circuit(circuit, cut_data, cut_data_test):
    qctest = circuit.copy()
    offset = 0

    zipped_data = zip(cut_data, cut_data_test)
    zipped_data = sorted(
        zipped_data,
        key=lambda pair: (
            pair[0][2][0],
            -len(pair[1]) if hasattr(pair[1], "__len__") else 0,
        ),
    )

    for ind, (i, j) in enumerate(zipped_data):
        if len(j) > 2:
            # Remove the original operation at the specified index§
            qubits = list(
                filter(
                    lambda x: x is not None,
                    [
                        q if circuit.find_bit(q).index in j[2] else None
                        for q in qctest.qubits
                    ],
                )
            )
            target_index = i[2][0] + offset
            original = qctest.data.pop(target_index)

            gate_name = j[1]
            if gate_name in QPD_GATE_REGISTRY:
                cut_marker = QPD_GATE_REGISTRY[gate_name]
            else:
                # No dedicated marker, so wrap the gate we just removed and let its QPD
                # be generated from its matrix at expansion time.
                cut_marker = CutTwoQubitGate(original.operation)

            # Insert the cut operation
            insert_or_append(
                qctest,
                target_index,
                CircuitInstruction(cut_marker, qubits),
            )

        else:
            # Find the correct index for the cut operation
            ind = i[2][2][i[2][2].index(j[0])]
            target_index = i[2][0] + offset + 1

            # Insert the cut operation
            insert_or_append(
                qctest,
                target_index,
                CircuitInstruction(cut, [qctest.qubits[ind]]),
            )

            # Increment offset to account for the new instruction
            offset += 1

    return qctest


def _cheaper_find_cuts(circuit, options):
    """Find cuts both with and without consolidation and keep the cheaper plan.

    Merging runs of gates on a pair lowers each pair's cost but can raise the total,
    since a merged run of gates about different axes can no longer join a joint
    decomposition. Here it also changes what the partitioner sees, so the two plans may
    cut different gates and cannot be compared any other way.
    """
    from QCut.qpd.qpd_operations import plan_cost

    candidates = []
    first_error = None
    for label in ("always", "never"):
        forced = options.replace(consolidate=label)
        try:
            found = find_cuts(circuit.copy(), options=forced)
        except Exception as error:  # noqa: BLE001, PERF203
            logger.debug("the %s plan could not be cut: %s", label, error)
            first_error = first_error or error
            continue
        candidates.append((plan_cost(found, forced), label, found))

    if not candidates:
        # Neither plan works, so report why rather than inventing a new message.
        raise first_error or QCutError("no cutting plan could be found")

    # Stable sort, so a tie keeps "always" and therefore the plan with fewer cuts.
    candidates.sort(key=lambda candidate: candidate[0])
    cost, label, chosen = candidates[0]
    if len(candidates) > 1:
        logger.info(
            f"Consolidation is {'on' if label == 'always' else 'off'} for this run, "
            f"since it costs gamma {cost:.4f} against {candidates[1][0]:.4f} the other "
            "way."
        )
    # Hand back the options the caller passed, not the forced ones the search used, so
    # both consolidation paths behave the same. Which way it went is in the log.
    chosen.options = options
    return chosen


def find_cuts(  # noqa: C901
    circuit,
    options: CutOptions | None = None,
):
    """Partition a quantum circuit into subcircuits by inserting cut operations.

    Converts the input circuit into a graph representation and partitions it into the
    specified number of subcircuits using METIS partitioning. Optionally refines the
    partitioning to respect maximum qubit constraints per subcircuit. Cut operations
    are inserted at the partition boundaries, and the resulting subcircuits and their
    mappings are returned.

    Args:
        circuit (QuantumCircuit): The input quantum circuit to partition.
        options (CutOptions, optional): configuration for the run. Defaults to
            QCut.options.DEFAULT_OPTIONS.

    Returns:
        CutCircuit: A circuit with cut operations inserted, ready for decomposition into
        subcircuits.
    """
    options = resolve(options)

    num_partitions = options.num_partitions
    max_qubits = options.max_qubits
    cuts = options.finder_cut_mode

    if num_partitions is None:
        if max_qubits is None:
            raise ValueError("Either num_partitions or max_qubits must be specified.")
        num_partitions = len(max_qubits)
    elif max_qubits is not None and len(max_qubits) != num_partitions:
        raise ValueError(
            "If both num_partitions and max_qubits are specified, length of"
            "max_qubits must match num_partitions."
        )

    if (max_qubits is not None and len(max_qubits) < 2) or num_partitions < 2:
        raise ValueError("Number of partitions has to be at least 2")

    if num_partitions == 1:
        return circuit, [], []

    circuit = circuit.remove_final_measurements(inplace=False)

    circuit = transpile(circuit, optimization_level=0, basis_gates=BASIS_GATES)

    mode = options.consolidate_mode
    if mode == "auto" and consolidate_two_qubit_blocks(circuit) is circuit:
        mode = "never"  # nothing was worth merging, so there is nothing to compare
    if mode == "auto":
        return _cheaper_find_cuts(circuit, options)
    if mode == "always" or mode == "auto":
        # Before the graph is built, so a merged run shows up as one edge to cut rather
        # than several. Nothing is marked yet, so every pair is a candidate.
        circuit = consolidate_two_qubit_blocks(circuit)

    graph, nodes_on_qubit = circ_to_graph(circuit, mode=cuts)

    components = rx.connected_components(graph)
    if len(components) == num_partitions:
        labels = {}
        for comp_ind, comp in enumerate(components):
            for node in comp:
                labels[node] = comp_ind
        candidates = [(labels, [], [])]
    else:
        # METIS minimises the weighted edge cut, which is the true cost only while no
        # cuts share a decomposition, and it returns just the best partitioning it saw
        # by that measure. Asking for one partitioning per seed and costing each whole
        # plan afterwards compares them by what they actually cost. The seeds are fixed,
        # so the answer is reproducible; ``options.seed`` shifts them as a set.
        base = 0 if options.seed is None else int(options.seed)
        # A qubit budget has to be asked for, not repaired for. The graph's nodes are
        # wire segments, so METIS balancing node counts says nothing about how many
        # qubits a partition ends up holding, and the repair afterwards moves whole
        # qubits across and pays in cuts for each. Weighting the nodes so a partition's
        # weight is its qubit count, and naming the share each may hold, gets the
        # constraint met by the partitioner instead.
        if max_qubits is None:
            weights, targets = None, None
            ufactors = (FREE_UFACTOR,)
        else:
            weights = qubit_node_weights(graph, nodes_on_qubit)
            total = sum(max_qubits)
            targets = [allowance / total for allowance in max_qubits]
            ufactors = BUDGET_UFACTORS
        candidates = []
        for offset in range(max(1, options.finder_candidates)):
            labels = k_way_metis_partition(
                graph,
                num_partitions,
                seed=base + offset // len(ufactors),
                ncuts=1,
                node_weights=weights,
                targets=targets,
                ufactor=ufactors[offset % len(ufactors)],
            )
            candidates.append((labels, *extract_cuts(graph, labels)))

    from QCut.qpd.qpd_operations import plan_cost  # local: avoids an import cycle

    best = None
    for labels, cut_data, cut_data_test in candidates:
        if max_qubits is not None:
            cut_data, cut_data_test, labels = refine_cuts(
                cut_data,
                cut_data_test,
                labels,
                graph,
                max_qubits,
                nodes_on_qubit,
                circuit,
                cuts,
            )

        cut_circuit = add_cuts_to_circuit(circuit, cut_data, cut_data_test)

        if max_qubits is not None:
            final_cut_circuit = get_locations_and_subcircuits(
                cut_circuit, max_qubits=max_qubits, options=options
            )
        else:
            final_cut_circuit = get_locations_and_subcircuits(
                cut_circuit, options=options
            )

        # Costed after refinement, because refinement can add cuts of its own.
        cost = plan_cost(final_cut_circuit, options)
        if best is None or cost < best[0]:
            best = (
                cost,
                final_cut_circuit,
                cut_circuit,
                cut_data,
                cut_data_test,
                labels,
            )

    if best is None:  # pragma: no cover - there is always at least one candidate
        raise QCutError("no candidate partition could be costed")

    if len(candidates) > 1:
        logger.info(
            f"Costed {len(candidates)} candidate partition(s), keeping one at "
            f"gamma={best[0]:.6g}."
        )
    _, final_cut_circuit, cut_circuit, cut_data, cut_data_test, labels = best

    return final_cut_circuit
