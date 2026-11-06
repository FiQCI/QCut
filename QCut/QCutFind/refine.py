"""
Custom refinement algorithm for improving cut locations in a quantum circuit
partitioning.
"""

from itertools import groupby


def qubits_per_partition(nodes_on_qubit, labels, k):
    part_qubits = {}
    for i in range(k):
        part_qubits[i] = set()

    for qubit, nodes in nodes_on_qubit.items():
        for node in nodes:
            part = labels.get(node)
            if part is not None:
                if part not in part_qubits:
                    part_qubits[part] = set()
                part_qubits[part].add(qubit)

    return {p: len(qs) for p, qs in part_qubits.items()}


def extra_wire_cuts(circuit, max_qubits, cut_data_test):
    num_allowed_wire_cuts = sum(max_qubits) - circuit.num_qubits
    num_wire_cuts = len([x for x in cut_data_test if len(x) == 2])
    return max(0, num_wire_cuts - num_allowed_wire_cuts)


def revert_wire_cut_cost(graph, cut_data_test, cut_data):  # noqa: C901
    wirecuts = [x for x in zip(cut_data, cut_data_test) if len(x[1]) == 2]

    wirecuts.sort(key=lambda x: x[1][0])

    wirecuts_grouped = groupby(wirecuts, key=lambda x: x[1])

    wirecuts_grouped_sorted = []
    for key, group in wirecuts_grouped:
        wirecuts_grouped_sorted.append(list(group))
        wirecuts_grouped_sorted[-1].sort(key=lambda x: x[0][2][0], reverse=True)

    res = []
    for group in wirecuts_grouped_sorted:
        prev_cost = -1
        group = list(group)
        for wirecut in group:
            zero_edges = graph.in_edges(wirecut[0][0])

            leftLeftNode = (
                wirecut[0][0]
                if wirecut[1] in [x[2] for x in zero_edges]
                else wirecut[0][1]
            )

            # Get all neighbours of leftLeftNode
            leftleft_neighbours = list(graph.neighbors(leftLeftNode))

            # For each neighbour, get the indices of connected gates from out_edges
            neigh_gate_indices = {}
            for n in leftleft_neighbours:
                out_edges = graph.out_edges(n)
                gate_indices = [x[2][0] for x in out_edges if len(x[2]) > 2]
                if gate_indices:
                    neigh_gate_indices[n] = gate_indices

            # For each list in the dict, call max
            max_indices = {k: max(v) for k, v in neigh_gate_indices.items() if v}

            # Choose neighbour node index with highest max
            if max_indices:
                best_neigh = max(max_indices, key=lambda k: max_indices[k])
            else:
                best_neigh = None

            leftNode = best_neigh
            neighbours = [x for x in graph.neighbors(leftNode) if x != leftLeftNode]
            filtered = neighbours
            filtered.append(leftNode)
            filtered.sort()

            # Should never be reached
            if len(filtered) == 0:
                raise RuntimeError(
                    "No valid neighbours found for filtered list; cannot proceed."
                    "If you encounter this open an issue in github"
                )

            rightNode = leftNode

            connected_2q_gates = []
            nodes_to_explore = [rightNode]
            prev_leftNodes = [leftNode]

            for i in nodes_to_explore:
                neighbours = graph.neighbors(i)

                for n in neighbours:
                    if n in prev_leftNodes:
                        continue
                    edge = graph.get_edge_data(i, n)
                    if len(edge) == 2:
                        nodes_to_explore.append(n)
                        prev_leftNodes.append(i)
                    else:
                        if edge[0] > wirecut[0][2][0]:
                            connected_2q_gates.append((i, n, edge))

            cost = len(connected_2q_gates)
            if cost <= prev_cost:
                cost = prev_cost + 1
            prev_cost = cost
            res.append({"cost": cost, "data": wirecut, "connected": connected_2q_gates})

    return res


def revert_wire_cuts(
    graph, wirecuts, cut_data_test, cut_data, labels, nodes_on_qubit, num
):
    for wirecut in wirecuts[:num]:
        zipped = list(zip(cut_data, cut_data_test))
        ind = zipped.index(wirecut["data"])
        cut_data_test.pop(ind)
        cut_data.pop(ind)
        to_flip = nodes_on_qubit[wirecut["data"][1][0]]
        cut_index = to_flip.index(wirecut["data"][0][0])
        to_flip = to_flip[cut_index + 1 :]

        connected = [graph.out_edges(x) for x in to_flip]
        connected_flat = [
            item for sublist in connected for item in sublist if len(item[2]) > 2
        ]
        # Cuts made redundant here are not dropped here. The labels below are still
        # being rewritten, so whether a cut's ends have landed on the same side is not
        # settled yet. `refine_cuts` filters them out once, after the labels are final.
        for x in connected_flat:
            if x[2] not in cut_data_test:
                cut_data_test.append(x[2])
                cut_data.append(x)

        for i in to_flip:
            ind = (
                wirecut["data"][0][0]
                if wirecut["data"][0][0] in nodes_on_qubit[wirecut["data"][1][0]]
                else wirecut["data"][0][1]
            )
            labels[i] = labels[ind]


def give_receive_qubits(qubits_per_partition, max_qubits):
    copy_max_qubits = max_qubits.copy()
    res = {}
    for key, value in qubits_per_partition.items():
        closest = min(
            [(x, x - value, abs(x - value)) for x in copy_max_qubits],
            key=lambda y: y[2],
        )
        res[key] = {"receive": closest[1], "q": value, "max_q": closest[0]}
        copy_max_qubits.remove(closest[0])

    return res


def swap_qubits(  # noqa: C901
    graph, cut_data_test, cut_data, receivers, givers, labels, nodes_on_qubit
):
    """
    Multi-partition qubit reallocation.

    - receivers: list[ {partition_label: deficit} ]
    - givers:    list[ {partition_label: excess} ]
    - labels:    dict[node_id] -> partition_label
    - nodes_on_qubit: dict[qubit_index] -> list[node_id] (topological order)
    """
    from collections import Counter

    def is_gate_edge(e):
        return len(e[2]) > 2

    def edge_in_cuts_test(e):
        u, v, d = e
        return d in cut_data_test

    def add_cut(e):
        u, v, d = e
        if not edge_in_cuts_test(e):
            cut_data.append((u, v, d))
            cut_data_test.append(d)

    # Build mutable counters for givers/receivers
    recv = {list(d.keys())[0]: list(d.values())[0] for d in receivers}
    give = {list(d.keys())[0]: list(d.values())[0] for d in givers}

    # Determine owner label for each qubit (mode of node labels on that wire)
    qubit_owner = {}
    for q, nodes in nodes_on_qubit.items():
        if not nodes:
            continue

        cnt = Counter(labels[n] for n in nodes)
        owner, _ = cnt.most_common(1)[0]
        qubit_owner[q] = owner

    # Gather gate edges touching a set of nodes (both in and out, unique by
    # unordered endpoints)
    def gate_edges_touching_nodes(nodes):
        nodes_set = set(nodes)
        seen_pairs = set()
        res = []
        for n in nodes_set:
            # Combine in_edges/out_edges, filter by gate edges, deduplicate
            for e in graph.in_edges(n):
                if is_gate_edge(e):
                    u, v, d = e
                    key = (u, v) if u <= v else (v, u)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        res.append((u, v, d))
            for e in graph.out_edges(n):
                if is_gate_edge(e):
                    u, v, d = e
                    key = (u, v) if u <= v else (v, u)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        res.append((u, v, d))
        return res

    # Compute delta in number of gate cuts if we relabel all nodes on 'nodes'
    # to 'to_label'
    def delta_cuts_for_qubit(nodes, to_label):
        nodes_set = set(nodes)
        delta = 0
        for u, v, d in gate_edges_touching_nodes(nodes):
            # Only consider edges that touch the moved qubit
            u_on = u in nodes_set
            v_on = v in nodes_set
            if not (u_on or v_on):
                continue

            # Current labels
            lu = labels[u]
            lv = labels[v]
            before_cross = lu != lv

            # After move: only endpoints on the moved qubit change to 'to_label'
            au = to_label if u_on else lu
            av = to_label if v_on else lv
            after_cross = au != av

            if before_cross and not after_cross:
                delta -= 1  # healed one cut
            elif (not before_cross) and after_cross:
                delta += 1  # created one new cut
        return delta

    # Apply relabel and update cut_data/cut_data_test for gate edges touching 'nodes'
    def relabel_qubit(nodes, to_label):
        nodes_set = set(nodes)
        # Update labels first
        for n in nodes_set:
            labels[n] = to_label

        # Update gate cut set membership. A cut whose ends have just landed on the same
        # side is left in place rather than removed: a later relabelling can separate
        # them again, so only `refine_cuts` knows which are really redundant.
        for u, v, d in gate_edges_touching_nodes(nodes):
            if labels[u] == labels[v] and (d in cut_data_test):
                continue
            else:
                add_cut((u, v, d))

    # Main allocation loop across all givers
    # For determinism, iterate givers in sorted label order
    for giver_label in sorted(give.keys()):
        remaining = give[giver_label]
        if remaining <= 0:
            continue

        # Candidate qubits owned by this giver
        candidates = [q for q, owner in qubit_owner.items() if owner == giver_label]

        # Skip if no receivers
        def has_receivers():
            return any(v > 0 for v in recv.values())

        # Greedy: pick the best (qubit, receiver) by minimal delta_cuts
        picked_qubits = set()
        while remaining > 0 and has_receivers():
            best = None  # (delta, q, recv_label)
            for q in candidates:
                if q in picked_qubits:
                    continue
                nodes = nodes_on_qubit[q]
                # Evaluate against all receivers with deficit
                for rlabel, deficit in recv.items():
                    if deficit <= 0 or rlabel == giver_label:
                        continue
                    delta = delta_cuts_for_qubit(nodes, rlabel)
                    if best is None or (delta, q, rlabel) < best:
                        best = (delta, q, rlabel)
            if best is None:
                break  # nothing more to do

            _, qpick, rpick = best
            # Apply the change
            relabel_qubit(nodes_on_qubit[qpick], rpick)
            qubit_owner[qpick] = rpick
            picked_qubits.add(qpick)

            # Update counters
            recv[rpick] -= 1
            remaining -= 1

        give[giver_label] = remaining

    # Rebuild receivers list in-place to reflect updated deficits
    receivers[:] = [{k: v} for k, v in recv.items() if v > 0]


def refine_cuts(
    cut_data_in,
    cut_data_test_in,
    labels_in,
    graph,
    max_qubits,
    nodes_on_qubit,
    circuit,
    cuts,
):
    if len(cut_data_in) == 0 or len(cut_data_test_in) == 0:
        return cut_data_in, cut_data_test_in, labels_in
    cut_data_loc = cut_data_in.copy()
    cut_data_test_loc = cut_data_test_in.copy()
    labels_loc = labels_in.copy()
    if cuts == "gate":
        # remove all wirecuts that slipped through
        extra_wire_cuts_val = len([i for i in cut_data_test_loc if len(i) == 2])
    else:
        extra_wire_cuts_val = extra_wire_cuts(circuit, max_qubits, cut_data_test_loc)

    if extra_wire_cuts_val > 0:
        cost = revert_wire_cut_cost(graph, cut_data_test_loc, cut_data_loc)
        cost.sort(key=lambda x: x["cost"])
        revert_wire_cuts(
            graph,
            cost,
            cut_data_test_loc,
            cut_data_loc,
            labels_loc,
            nodes_on_qubit,
            extra_wire_cuts_val,
        )
    qubits_per_partition_val = qubits_per_partition(
        nodes_on_qubit, labels_loc, len(max_qubits)
    )
    res = give_receive_qubits(qubits_per_partition_val, max_qubits)
    receivers = [
        {key: value["receive"]} for key, value in res.items() if value["receive"] > 0
    ]
    givers = [
        {key: abs(value["receive"])}
        for key, value in res.items()
        if value["receive"] < 0
    ]
    swap_qubits(
        graph,
        cut_data_test_loc,
        cut_data_loc,
        receivers,
        givers,
        labels_loc,
        nodes_on_qubit,
    )

    # Drop gate cuts left behind by reverting and swapping, whose ends are now on the
    # same side. A wire cut's entry names a neighbouring edge, so skip those.
    kept = [
        (test, data)
        for test, data in zip(cut_data_test_loc, cut_data_loc)
        if len(test) == 2 or labels_loc.get(data[0]) != labels_loc.get(data[1])
    ]
    if not kept:
        return [], [], labels_loc

    kept.sort(key=lambda x: x[1][2][0])
    cut_data_test_loc, cut_data_loc = map(list, zip(*kept))

    return cut_data_loc, cut_data_test_loc, labels_loc
