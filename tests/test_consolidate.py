"""Tests for merging runs of gates on the same qubit pair before cutting."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CPhaseGate, CZGate, RYYGate, RZZGate
from qiskit.quantum_info import Operator, Pauli, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut, cutCZ, cutGate
from QCut.pre_process.consolidate import consolidate_two_qubit_blocks, marker_gate
from QCut.qpd.qpd_generate import gamma
from QCut.qpd.qpd_operations import qpd_for_location

TOLERANCE = 0.1

#: Shot budget for the end-to-end tests. These assert that a plan reconstructs the
#: right answer, and a plan that pairs the wrong operations is out by order one, so
#: they do not need shot-noise precision. Measured over six runs the worst error on
#: the heaviest fixture is 0.041, leaving the tolerance a factor of 2.5 clear.
SHOTS = 2**11


def _unwrap(circuit: QuantumCircuit) -> QuantumCircuit:
    """Replace cut markers by the gates they carry, so the unitary can be compared."""
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        gate = marker_gate(instruction.operation)
        out.append(
            gate if gate is not None else instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )
    return out


def _equivalent(before: QuantumCircuit, after: QuantumCircuit) -> bool:
    return Operator(_unwrap(before)).equiv(Operator(_unwrap(after)))


def test_two_gates_on_a_pair_become_one():
    circuit = QuantumCircuit(2)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.4, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 1}
    assert _equivalent(circuit, merged)


def test_marker_survives_the_merge():
    """A merged run on a marked pair must still be marked, or the cut is lost."""
    circuit = QuantumCircuit(2)
    circuit.append(**cutGate(RZZGate(0.4), 0, 1))
    circuit.rzz(0.4, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert list(merged.count_ops()) == ["CutUNITARY"]
    assert _equivalent(circuit, merged)


def test_lone_two_qubit_gate_keeps_its_name():
    """Absorbing single-qubit gates alone cannot help, and loses the named gate."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_single_qubit_gate_shared_by_two_pairs_is_not_applied_twice():
    """A gate on a shared qubit may only be absorbed into one of the two runs."""
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.4, 0, 1)
    circuit.h(0)
    circuit.rzz(0.4, 0, 2)
    circuit.rzz(0.4, 0, 2)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 2}
    assert _equivalent(circuit, merged)


def test_a_wire_cut_ends_a_run():
    """Merging across a wire cut would move the cut, so the run has to stop there."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.4, 0, 1)
    circuit.append(cut(), [0])
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_a_gate_reaching_outside_the_pair_ends_a_run():
    """A blocker that does not commute cannot be moved, so the run really does end.

    Which leg of the blocker lands on the pair is what decides it. ``cx(2, 1)`` puts its
    target on qubit 1 and does not commute with a Z-diagonal gate there, while
    ``cx(1, 2)`` puts its control there and does, so that one gets slid past instead.
    """
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.cx(2, 1)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_a_control_leg_on_the_pair_commutes_and_is_slid_past():
    """The companion to the above: a control leg is Z-diagonal, so the run survives."""
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.cx(1, 2)
    circuit.rzz(0.5, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 1, "cx": 1}
    assert _equivalent(circuit, merged)


def test_a_commuting_gate_reaching_outside_the_pair_is_slid_past():
    """The Ising case: neighbouring rzz gates all commute, so the run survives.

    Without this the gates on a pair are rarely adjacent enough to merge, since a layer
    of an Ising or QAOA circuit puts a neighbouring gate between every pair of them.
    """
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.3, 1, 2)
    circuit.rzz(0.5, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 1, "rzz": 1}
    assert _equivalent(circuit, merged)


def test_a_slid_run_puts_the_merged_gate_after_the_blocker():
    """Members before the blocker move across it, so the merge lands at the run's end.

    Anchoring at the start instead would reorder the blocker against gates that have
    not moved, which is why the position depends on whether anything was slid.
    """
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.3, 1, 2)
    circuit.rzz(0.5, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    names = [instruction.operation.name for instruction in merged.data]
    assert names == ["rzz", "unitary"]


def test_an_absorbed_single_qubit_gate_does_not_block_a_slide():
    """Only the members crossing the blocker have to commute with it.

    A run that greedily absorbed ``ry`` would otherwise be killed by a check that gate
    need not be part of: the ``ry`` stays where it is and only the ``rzz`` moves.
    """
    circuit = QuantumCircuit(3)
    circuit.ry(0.7, 1)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.3, 1, 2)
    circuit.rzz(0.5, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 1, "rzz": 1, "ry": 1}
    assert _equivalent(circuit, merged)


def test_a_cut_marker_is_unwrapped_before_the_commutation_check():
    """A marker is opaque, and the checker refuses anything it cannot look inside.

    Without unwrapping, every marked pair looks non-commuting and the whole exercise
    silently does nothing -- which is the case that matters, since consolidation only
    ever runs on marked pairs.
    """
    circuit = QuantumCircuit(4)
    circuit.append(**cutGate(RZZGate(0.5), 1, 2))
    circuit.rzz(0.3, 2, 3)
    circuit.append(**cutGate(RZZGate(0.6), 1, 2))
    merged = consolidate_two_qubit_blocks(circuit, restrict_to={frozenset({1, 2})})
    assert merged.count_ops() == {"CutUNITARY": 1, "rzz": 1}


def test_a_wire_cut_marker_is_never_slid_past():
    """A wire cut pins a location the caller chose, so it ends the run regardless."""
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.append(cut(), [1])
    circuit.rzz(0.5, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


@pytest.mark.parametrize("seed", range(24))
def test_sliding_keeps_the_circuit_equivalent(seed):
    """Randomised check that reordering around commuting gates preserves the unitary.

    Both orientations of ``cx`` appear on purpose. Only one of them commutes with a
    Z-diagonal neighbour, so a check that dropped the gate's argument order would pass
    on symmetric gates and quietly corrupt these.
    """
    rng = np.random.default_rng(1000 + seed)
    width = int(rng.integers(3, 6))
    circuit = QuantumCircuit(width)
    for qubit in range(width):
        circuit.ry(float(rng.uniform(0, np.pi)), qubit)
    for _ in range(int(rng.integers(3, 7))):
        first = int(rng.integers(width - 1))
        draw = rng.random()
        if draw < 0.5:
            circuit.rzz(float(rng.uniform(0.2, 1.2)), first, first + 1)
        elif draw < 0.7:
            circuit.cx(first, first + 1)
        elif draw < 0.9:
            circuit.cx(first + 1, first)
        else:
            circuit.cz(first, first + 1)
        if rng.random() < 0.4:
            circuit.rz(float(rng.uniform(0, np.pi)), int(rng.integers(width)))
    assert _equivalent(circuit, consolidate_two_qubit_blocks(circuit))


def test_a_disjoint_gate_does_not_end_a_run():
    """Gates on other qubits commute with the run, so they must not break it."""
    circuit = QuantumCircuit(4)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.9, 2, 3)
    circuit.rzz(0.4, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 1, "rzz": 1}
    assert _equivalent(circuit, merged)


def test_measurement_ends_a_run():
    circuit = QuantumCircuit(2, 1)
    circuit.rzz(0.4, 0, 1)
    circuit.measure(0, 0)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_restrict_to_leaves_other_pairs_alone():
    circuit = QuantumCircuit(2)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit, restrict_to=set()) is circuit


def test_two_cz_cuts_collapse_to_a_free_cut():
    """CZ twice is the identity, so the merged cut costs gamma = 1."""
    circuit = QuantumCircuit(2)
    circuit.append(cutCZ(), [0, 1])
    circuit.append(cutCZ(), [0, 1])
    cut_circuit = ck.get_locations_and_subcircuits(circuit)
    assert len(cut_circuit.cut_locations) == 1
    qpd = qpd_for_location(cut_circuit.cut_locations[0])
    assert gamma(qpd) == pytest.approx(1.0)
    assert len(qpd) == 1


def _run(circuit, observables, options):
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
    values = ck.estimate_expectation_values(results)
    return values, len(cut_circuit.cut_locations), experiment.num_groups


@pytest.mark.slow
@pytest.mark.sim
def test_merging_lowers_the_cost_and_keeps_the_answer():
    """Two marked rzz cost gamma 3.16 over 36 groups apart, 2.43 over 6 merged."""
    observables = ["IZ", "ZI", "ZZ"]

    def build():
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ry(0.7, 1)
        circuit.append(**cutGate(RZZGate(0.4), 0, 1))
        circuit.append(**cutGate(RZZGate(0.4), 0, 1))
        return circuit

    reference = QuantumCircuit(2)
    reference.h(0)
    reference.ry(0.7, 1)
    reference.rzz(0.8, 0, 1)
    state = Statevector(reference)
    exact = [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]

    off, cuts_off, groups_off = _run(
        build(), observables, CutOptions(consolidate=False)
    )
    on, cuts_on, groups_on = _run(build(), observables, CutOptions(consolidate=True))

    assert (cuts_off, groups_off) == (2, 36)
    assert (cuts_on, groups_on) == (1, 6)
    for expected, without, with_ in zip(exact, off, on):
        assert abs(expected - without) < TOLERANCE
        assert abs(expected - with_) < TOLERANCE


@pytest.mark.slow
@pytest.mark.sim
def test_find_cuts_consolidates_before_partitioning():
    """A merged run is one graph edge to cut rather than several."""
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.4 + 0.2 * qubit, qubit)
    circuit.rzz(0.3, 0, 1)
    circuit.rzz(0.4, 1, 2)
    circuit.rzz(0.4, 1, 2)
    circuit.rzz(0.5, 2, 3)

    with_merge = ck.find_cuts(
        circuit.copy(), options=CutOptions(finder_num_partitions=2, consolidate=True)
    )
    without = ck.find_cuts(
        circuit.copy(), options=CutOptions(consolidate=False, finder_num_partitions=2)
    )
    observables = ["IIIZ", "IIZI", "IZII", "ZIII"]
    state = Statevector(circuit)
    exact = [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]

    for cut_circuit in (with_merge, without):
        experiment = ck.get_experiment_circuits(cut_circuit, observables)
        results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
        values = ck.estimate_expectation_values(results)
        for expected, actual in zip(exact, values):
            assert abs(expected - actual) < TOLERANCE


def test_options_reach_the_cut_circuit():
    options = CutOptions(consolidate=False)
    circuit = QuantumCircuit(2)
    circuit.append(cutCZ(), [0, 1])
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    assert cut_circuit.options is options
    experiment = ck.get_experiment_circuits(cut_circuit, ["IZ", "ZI"])
    assert experiment.options is options


def test_marker_gate_recovers_the_per_gate_markers():
    assert marker_gate(cutCZ()).name == CZGate().name
    assert marker_gate(cut()) is None


def _blocked_bundle_circuit():
    """A run of gates about two different axes on one pair, with a parallel partner.

    Merging pair (0, 2) turns it into a generic two-qubit unitary, which is no longer a
    single-axis rotation and so can no longer join a joint decomposition with the gate
    on pair (1, 3). Here that costs more than the merge saves.
    """
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.3 + 0.1 * qubit, qubit)
    circuit.rzz(0.7, 0, 1)
    circuit.rzz(0.7, 2, 3)
    circuit.append(**cutGate(CPhaseGate(1.15), 0, 2))
    circuit.append(**cutGate(RZZGate(1.768), 1, 3))
    circuit.append(**cutGate(RYYGate(0.758), 0, 2))
    return circuit


def _planned(circuit, consolidate):
    from QCut.qpd.qpd_operations import plan_cost

    options = CutOptions(consolidate=consolidate)
    cut_circuit = ck.get_locations_and_subcircuits(circuit.copy(), options=options)
    return plan_cost(cut_circuit, options), len(cut_circuit.cut_locations)


def test_auto_declines_a_merge_that_would_block_a_bundle():
    """Merging is not always cheaper once joint cutting is on, so auto compares."""
    circuit = _blocked_bundle_circuit()
    always, always_cuts = _planned(circuit, "always")
    never, never_cuts = _planned(circuit, "never")
    auto, auto_cuts = _planned(circuit, "auto")

    assert always > never  # merging genuinely costs more on this circuit
    assert auto == pytest.approx(never)
    assert (always_cuts, never_cuts, auto_cuts) == (2, 3, 3)


def test_auto_takes_the_merge_when_it_helps():
    """Two gates on a pair with no bundle to lose, so merging is the cheaper plan."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.ry(0.7, 1)
    circuit.append(**cutGate(RZZGate(0.4), 0, 1))
    circuit.append(**cutGate(RZZGate(0.4), 0, 1))

    always, always_cuts = _planned(circuit, "always")
    never, never_cuts = _planned(circuit, "never")
    auto, auto_cuts = _planned(circuit, "auto")

    assert always < never
    assert auto == pytest.approx(always)
    assert (always_cuts, never_cuts, auto_cuts) == (1, 2, 1)


@pytest.mark.slow
@pytest.mark.sim
def test_auto_keeps_the_answer_on_the_plan_it_picks():
    """The cheaper plan still has to reconstruct the right expectation values."""
    circuit = _blocked_bundle_circuit()
    reference = QuantumCircuit(4)
    for qubit in range(4):
        reference.ry(0.3 + 0.1 * qubit, qubit)
    reference.rzz(0.7, 0, 1)
    reference.rzz(0.7, 2, 3)
    reference.cp(1.15, 0, 2)
    reference.rzz(1.768, 1, 3)
    reference.ryy(0.758, 0, 2)

    observables = ["IIIZ", "IIZI", "IZII", "ZIII"]
    state = Statevector(reference)
    exact = [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]

    values, _, _ = _run(circuit, observables, CutOptions(consolidate="auto"))
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE


def test_consolidate_strategy_accepts_booleans_and_rejects_nonsense():
    """True and False stay valid, and 'never' must not read as truthy."""
    from QCut.errors.qcuterror import QCutError

    assert CutOptions(consolidate=True).consolidate_mode == "always"
    assert CutOptions(consolidate=False).consolidate_mode == "never"
    assert CutOptions().consolidate_mode == "auto"
    for value in ("auto", "always", "never"):
        assert CutOptions(consolidate=value).consolidate_mode == value
    with pytest.raises(QCutError, match="unknown consolidate strategy"):
        CutOptions(consolidate="sometimes")


def test_never_really_means_never():
    """A guard against reading the strategy string as a plain boolean."""
    circuit = QuantumCircuit(2)
    circuit.append(**cutGate(RZZGate(0.4), 0, 1))
    circuit.append(**cutGate(RZZGate(0.4), 0, 1))
    options = CutOptions(consolidate="never")
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    assert len(cut_circuit.cut_locations) == 2


@pytest.mark.slow
@pytest.mark.sim
def test_find_cuts_auto_is_never_worse_than_either_mode():
    """find_cuts runs the whole search both ways under auto.

    Consolidating changes which edges the partitioner sees, so the two plans can cut
    different gates and the comparison cannot be made any other way.
    """
    from QCut.qpd.qpd_operations import plan_cost

    circuit = QuantumCircuit(6)
    for qubit in range(6):
        circuit.ry(0.3 + 0.1 * qubit, qubit)
    for left, right in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        circuit.rzz(0.8, left, right)
    # runs of gates about different axes on the pairs that cross the split
    circuit.cp(1.15, 0, 3)
    circuit.rzz(1.768, 1, 4)
    circuit.ryy(0.758, 0, 3)

    costs = {}
    for mode in ("auto", "always", "never"):
        options = CutOptions(consolidate=mode, finder_num_partitions=2)
        found = ck.find_cuts(circuit.copy(), options=options)
        costs[mode] = plan_cost(found, options)

    assert costs["auto"] <= min(costs["always"], costs["never"]) + 1e-9

    observables = ["IIIIIZ", "IIIIZI", "IIIZII", "ZIIIII"]
    state = Statevector(circuit)
    exact = [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]
    options = CutOptions(consolidate="auto", finder_num_partitions=2)
    found = ck.find_cuts(circuit.copy(), options=options)
    experiment = ck.get_experiment_circuits(found, observables)
    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
    values = ck.estimate_expectation_values(results)
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE
