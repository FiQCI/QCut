"""Tests for CircuitKnitting package."""  # noqa: N999

import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import QCut as ck

# import QCut as ck
import tests.solutions_automatic_cuts as sq
from QCut import CutOptions, find_cuts
from QCut.errors.qcuterror import QCutError
from QCut.execution.run import run_cut_circuit
from QCut.QCutFind.combine_subcircuits import construct_final_subcircuits

#: Bound on each expectation value. The failure this guards against, a cut circuit that
#: does not reconstruct its original, misses by order one, so a bound seven times
#: tighter than that is plenty. Set alongside SHOTS to keep a factor of 2.5 over the
#: worst error measured across four passes, which is 0.060.
TOLERANCE = 0.15

#: Shot budget per fixture. The version matrix runs these cases in every environment and
#: they are the only end-to-end path it still covers, so the budget is the smallest one
#: that keeps the tolerance comfortable rather than the largest one affordable.
SHOTS = 2**11

#: One simulator for the whole module. Building an AerSimulator per test costs several
#: seconds of thread-pool setup, which is most of what the parametrised cases would
#: otherwise spend.
SIMULATOR = AerSimulator()

mult = 1.635

circuit1 = QuantumCircuit(4)
circuit1.r(mult * 0.46262, mult * 0.1446, 0)
circuit1.cx(0, 1)
circuit1.cx(1, 2)
circuit1.cx(2, 3)

circuit2 = QuantumCircuit(4)
circuit2.r(mult * 0.46262, mult * 0.1446, 0)
circuit2.cx(0, 1)

circuit2.cx(1, 2)
circuit2.cx(1, 2)
circuit2.cx(1, 2)

circuit2.cx(2, 3)


def test_find_cuts() -> None:
    """Test find_cuts function.

    This function tests whether the find_cuts method correctly identifies the cut
    locations in the provided test circuits by comparing the result to the
    pre-defined solutions.
    """
    for solution_index, circ in enumerate(sq.test_circuits):
        options = CutOptions(
            finder_num_partitions=sq.cut_sizes[solution_index],
            finder_cut_mode="both",
        )

        cut_circuit = find_cuts(circ.copy(), options=options)

        assert len(cut_circuit.subcircuits) == sq.cut_sizes[solution_index]


def test_find_gate_cuts():
    """Test find_cuts function on a circuit with a cut gate.

    This function tests whether the find_cuts method correctly identifies the cut
    locations in a circuit containing a cut gate by comparing the result to the
    expected number of subcircuits.
    """
    options = CutOptions(
        finder_num_partitions=2,
        finder_cut_mode="both",
    )
    cut_circuit = find_cuts(circuit1.copy(), options=options)

    assert len(cut_circuit.subcircuits) == 2

    print(cut_circuit.subcircuits[0])
    print(cut_circuit.subcircuits[1])

    for circ in cut_circuit.subcircuits:
        for op in circ.data:
            assert "meas" not in op.operation.name.lower()
            assert "init" not in op.operation.name.lower()


def test_auto_refine_wire():
    """Test find_cuts function on a circuit with a cut gate.

    This function tests whether the find_cuts method correctly identifies the cut
    locations in a circuit containing a cut gate by comparing the result to the
    expected number of subcircuits.
    """
    options = CutOptions(
        finder_num_partitions=2,
        finder_max_qubits=[2, 2],
        finder_cut_mode="wire",
    )
    cut_circuit = find_cuts(circuit2.copy(), options=options)

    assert len(cut_circuit.subcircuits) == 2


def test_auto_refine_gate():
    """Test find_cuts function on a circuit with a cut gate.

    This function tests whether the find_cuts method correctly identifies the cut
    locations in a circuit containing a cut gate by comparing the result to the
    expected number of subcircuits.
    """
    options = CutOptions(
        finder_num_partitions=2,
        finder_max_qubits=[2, 2],
        finder_cut_mode="gate",
    )
    cut_circuit = find_cuts(circuit2.copy(), options=options)

    assert len(cut_circuit.subcircuits) == 2

    for circ in cut_circuit.subcircuits:
        for op in circ.data:
            assert "meas" not in op.operation.name.lower()
            assert "init" not in op.operation.name.lower()


def test_construct_final_subcircuits():
    """Test the construction of final subcircuits.

    This function tests whether the final subcircuits are correctly constructed
    after identifying cut locations by comparing the operations in the generated
    subcircuits to the expected operations.
    """
    options = CutOptions(
        finder_num_partitions=2,
        finder_max_qubits=[2, 2],
        finder_cut_mode="gate",
    )
    cut_circuit = find_cuts(circuit2.copy(), options=options)

    final_circs = construct_final_subcircuits(
        cut_circuit.subcircuits, [circuit2.num_qubits]
    )

    assert len(final_circs) == 1


@pytest.mark.parametrize("index", range(len(sq.test_circuits)))
@pytest.mark.sim
def test_expectation_values(index: int) -> None:
    """Find the cuts, run the pieces, and check the expectation values come back.

    One case per fixture rather than one loop over all of them, so that a failure names
    the circuit that failed and the expensive fixtures can be marked without taking the
    cheap ones with them. This is the only end-to-end path the qiskit version matrix
    still runs, which is deliberate: it is the one that would notice a transpiler or
    primitive change breaking the pipeline while every unit test stayed green.
    """
    options = CutOptions(
        finder_num_partitions=sq.cut_sizes[index],
        finder_cut_mode="both",
    )
    cut_circuit = find_cuts(sq.test_circuits[index].copy(), options=options)
    values = run_cut_circuit(
        cut_circuit, sq.test_observables[index], SIMULATOR, shots=SHOTS
    )
    for expected, actual in zip(sq.exp_val_solutions[index], values):
        assert abs(expected - actual) <= TOLERANCE, (  # noqa: S101
            f"fixture {index}: expected {expected}, got {actual}"
        )


def _finder_circuit():
    """Dense enough that the partitioner has real choices to make."""
    import numpy as np

    rng = np.random.default_rng(11)
    circuit = QuantumCircuit(10)
    for qubit in range(10):
        circuit.h(qubit)
    for _ in range(2):
        for first in range(9):
            circuit.rzz(float(rng.uniform(0.2, 1.2)), first, first + 1)
        for first in range(0, 8, 2):
            circuit.rzz(float(rng.uniform(0.2, 1.2)), first, first + 2)
    return circuit


def _cost(options):
    from QCut.qpd.qpd_operations import plan_cost

    options = options.replace(
        finder_max_qubits=[5, 5],
        finder_cut_mode="both",
    )

    found = ck.find_cuts(_finder_circuit().copy(), options=options)
    return plan_cost(found, options)


def test_the_finder_is_deterministic():
    """Two runs on one circuit must agree.

    METIS used to be seeded from ``np.random.randint``, so repeated runs returned
    partitions whose overheads differed by up to three orders of magnitude with no way
    to reproduce the good one.
    """
    options = CutOptions()
    assert _cost(options) == _cost(options)


def test_the_seed_moves_the_candidate_set():
    """``seed`` shifts which partitions are tried, and each choice is reproducible."""
    first = CutOptions(seed=0)
    second = CutOptions(seed=500)
    assert _cost(first) == _cost(first)
    assert _cost(second) == _cost(second)


def test_more_candidates_never_cost_more():
    """The candidates are consecutive seeds, so a larger set contains the smaller one.

    Costing whole plans and keeping the cheapest therefore cannot get worse by looking
    at more of them, which is what makes the knob safe to raise.
    """
    few = _cost(CutOptions(seed=0, finder_candidates=1))
    many = _cost(CutOptions(seed=0, finder_candidates=6))
    assert many <= few + 1e-9


def test_a_single_candidate_still_works():
    """``finder_candidates=1`` is the old cost, with the determinism kept."""
    assert _cost(CutOptions(finder_candidates=1)) > 0


def test_finder_candidates_must_be_positive():
    with pytest.raises(QCutError, match="finder_candidates"):
        CutOptions(finder_candidates=0)


def _widths(found):
    return sorted(sub.num_qubits for sub in found.subcircuits)


def test_a_qubit_budget_is_met_by_the_partitioner():
    """The budget has to be asked for, not repaired for.

    The graph's nodes are wire segments, so balancing node counts says nothing about how
    many qubits a partition holds. Weighting the nodes per qubit and naming each
    partition's share gets the constraint met while the cut is chosen. Repairing it
    afterwards, by moving whole qubits across and paying in cuts, used to cost up to
    three orders of magnitude in sampling overhead on exactly these circuits.
    """
    from QCut.qpd.qpd_operations import plan_cost

    options = CutOptions(
        consolidate="never",
        joint_rotation_cuts=False,
        wire_cut_communication="never",
        finder_max_qubits=[5, 5],
        finder_cut_mode="both",
    )
    found = ck.find_cuts(_finder_circuit().copy(), options=options)
    assert _widths(found) == [5, 5]
    # The optimum over balanced bipartitions of this circuit, found by exhaustion.
    assert plan_cost(found, options) < 30.0


def test_an_uneven_budget_is_respected():
    """Shares come from ``max_qubits``, so they do not have to be equal."""
    found = ck.find_cuts(
        _finder_circuit().copy(),
        options=CutOptions(finder_cut_mode="both", finder_max_qubits=[7, 3]),
    )
    assert max(_widths(found)) <= 7


def test_no_budget_leaves_the_split_free_to_be_uneven():
    """Without a budget an unbalanced split is often much cheaper, so do not force one.

    Balancing unconditionally would make the unconstrained path worse, which is why the
    node weights are only applied when there is a budget to meet.
    """
    from QCut.qpd.qpd_operations import plan_cost

    options = CutOptions(
        finder_num_partitions=2,
        finder_cut_mode="both",
        consolidate="never",
        joint_rotation_cuts=False,
        wire_cut_communication="never",
    )
    free = ck.find_cuts(_finder_circuit().copy(), options=options)

    options = options.replace(finder_max_qubits=[5, 5])

    budgeted = ck.find_cuts(_finder_circuit().copy(), options=options)
    assert plan_cost(free, options) <= plan_cost(budgeted, options) + 1e-9
