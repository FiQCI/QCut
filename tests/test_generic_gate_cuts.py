"""End-to-end tests for cutting two-qubit gates that have no hand-written QPD."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    CRZGate,
    CXGate,
    RZZGate,
    UnitaryGate,
    XXPlusYYGate,
)
from qiskit.quantum_info import Pauli, Statevector, random_unitary
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cutGate
from QCut.errors.qcuterror import QCutError
from QCut.qpd.qpd_gates import CutTwoQubitGate
from QCut.qpd.qpd_operations import qpd_for_location

OBSERVABLES = ["IZ", "ZI", "ZZ"]
# Shot noise on the reconstructed values, in line with the rest of the suite.
TOLERANCE = 0.1


def _reference(gate) -> list[float]:
    """Exact expectation values of the uncut circuit."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.ry(0.7, 1)
    circuit.append(gate, [0, 1])
    state = Statevector(circuit)
    return [float(np.real(state.expectation_value(Pauli(p)))) for p in OBSERVABLES]


def _cut_and_run(gate):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.ry(0.7, 1)
    circuit.append(**cutGate(gate, 0, 1))
    cut_circuit = ck.get_locations_and_subcircuits(circuit)
    experiment = ck.get_experiment_circuits(cut_circuit, OBSERVABLES)
    results = ck.run_experiments(experiment, backend=AerSimulator())
    values = ck.estimate_expectation_values(results)
    return values, experiment.num_groups


# Gates with no entry in QPD_REGISTRY. rzz and crz used to cost 36 groups at gamma 9,
# as two cz cuts each.
GENERATED = [
    ("rzz", RZZGate(0.3), 6),
    ("rzz_large_angle", RZZGate(1.9), 6),
    ("crz", CRZGate(0.8), 6),
    ("cx", CXGate(), 6),
    ("xx_plus_yy", XXPlusYYGate(0.9, 0.4), 30),
]


@pytest.mark.parametrize(
    ("name", "gate", "expected_groups"), GENERATED, ids=[g[0] for g in GENERATED]
)
@pytest.mark.sim
def test_generated_cut_reproduces_the_uncut_circuit(name, gate, expected_groups):
    values, num_groups = _cut_and_run(gate)
    assert num_groups == expected_groups
    for expected, actual in zip(_reference(gate), values):
        assert abs(expected - actual) < TOLERANCE, f"{name}: {values}"


@pytest.mark.sim
def test_generic_unitary_can_be_cut():
    """An arbitrary SU(4) needs the full 58-row table."""
    gate = UnitaryGate(random_unitary(4, seed=7).data)
    values, num_groups = _cut_and_run(gate)
    assert num_groups == 58
    for expected, actual in zip(_reference(gate), values):
        assert abs(expected - actual) < TOLERANCE


def test_two_qubit_gate_becomes_a_single_marker():
    """cutGate no longer transpiles a two-qubit gate into cz cuts."""
    marker = cutGate(RZZGate(0.3), 0, 1)["instruction"]
    assert isinstance(marker, CutTwoQubitGate)
    assert marker.name == "CutRZZ"
    assert marker.gate.name == "rzz"


def test_larger_gate_still_transpiles():
    """A three-qubit gate must still be decomposed first."""
    result = cutGate(CXGate().control(1), 0, [1, 2])
    assert not isinstance(result["instruction"], CutTwoQubitGate)
    assert result["qargs"] == [0, 1, 2]


def _location(gate_name, gate):
    """Build a CutLocation directly, without going through a circuit."""
    from qiskit import QuantumRegister

    from QCut.model.cutlocation import CutLocation

    register = QuantumRegister(2, "q")
    return CutLocation(([(register, 0), (register, 1)], 0), gate_name, gate)


def test_registry_gates_still_use_their_hand_written_table():
    """cz must keep using cz_qpd, which needs no local u gates, not generation."""
    from QCut.qpd.qpd import cz_qpd, swap_qpd

    assert qpd_for_location(_location("cz", CXGate())) is cz_qpd
    assert qpd_for_location(_location("swap", None)) is swap_qpd
    generated = qpd_for_location(_location("cx", CXGate()))
    assert generated is not cz_qpd
    assert len(generated) == len(cz_qpd)


def test_qpd_is_resolved_once_per_location():
    """Generation costs a KAK decomposition, so the result must be cached."""
    circuit = QuantumCircuit(2)
    circuit.append(**cutGate(RZZGate(0.3), 0, 1))
    location = ck.get_locations_and_subcircuits(circuit).cut_locations[0]
    first = qpd_for_location(location)
    assert qpd_for_location(location) is first


def test_gate_without_a_qpd_or_a_gate_raises():
    """A cut location with an unknown gate name and no gate cannot be expanded."""
    circuit = QuantumCircuit(2)
    circuit.append(**cutGate(RZZGate(0.3), 0, 1))
    location = ck.get_locations_and_subcircuits(circuit).cut_locations[0]
    location.gate = None
    location._qpd = None
    with pytest.raises(NotImplementedError, match="No QPD available"):
        qpd_for_location(location)


class TestParametrisedCutGate:
    """A cut gate may keep parameters unbound until the experiments are built."""

    THETA = Parameter("theta")

    def _circuit(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ry(0.7, 1)
        circuit.append(**cutGate(RZZGate(self.THETA), 0, 1))
        return circuit

    def test_parameter_is_visible_to_qiskit(self):
        """The marker must re-expose the wrapped gate's parameters."""
        assert self.THETA in self._circuit().parameters

    def test_binding_before_the_cut_works(self):
        bound = self._circuit().assign_parameters({self.THETA: 0.3})
        location = ck.get_locations_and_subcircuits(bound).cut_locations[0]
        assert location.gate.params == [0.3]

    def test_splitting_before_binding_works(self):
        """The split is parameter-agnostic, so it must not require binding."""
        cut_circuit = ck.get_locations_and_subcircuits(self._circuit())
        assert cut_circuit.num_subcircuits == 2
        assert cut_circuit.cut_locations[0].gate.params == [self.THETA]

    @pytest.mark.parametrize("key", ["object", "name"])
    @pytest.mark.parametrize("theta", [0.3, 1.9])
    @pytest.mark.sim
    def test_binding_after_the_split_gives_correct_values(self, key, theta):
        cut_circuit = ck.get_locations_and_subcircuits(self._circuit())
        parameters = {self.THETA: theta} if key == "object" else {"theta": theta}
        bound = cut_circuit.assign_parameters(parameters, inplace=False)
        assert bound.cut_locations[0].gate.params == [theta]

        experiment = ck.get_experiment_circuits(bound, OBSERVABLES)
        assert experiment.num_groups == 6
        results = ck.run_experiments(experiment, backend=AerSimulator())
        values = ck.estimate_expectation_values(results)
        for expected, actual in zip(_reference(RZZGate(theta)), values):
            assert abs(expected - actual) < TOLERANCE

    def test_binding_after_the_split_works_inplace(self):
        cut_circuit = ck.get_locations_and_subcircuits(self._circuit())
        cut_circuit.assign_parameters({self.THETA: 0.3}, inplace=True)
        assert cut_circuit.cut_locations[0].gate.params == [0.3]

    def test_expanding_while_unbound_explains_what_to_do(self):
        cut_circuit = ck.get_locations_and_subcircuits(self._circuit())
        with pytest.raises(QCutError, match="assign_parameters"):
            ck.get_experiment_circuits(cut_circuit, OBSERVABLES)


@pytest.mark.sim
def test_automatic_cut_finding_uses_a_generated_qpd():
    """find_cuts must keep a gate whole rather than break it into cz cuts.

    rzz used to be transpiled away before the interaction graph was built. Now it
    survives and is cut at its own gamma = 1 + 2|sin(theta)|.
    """
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.4 + 0.3 * qubit, qubit)
    circuit.rzz(0.3, 0, 1)
    circuit.rzz(0.4, 1, 2)
    circuit.rzz(0.5, 2, 3)

    options = CutOptions(
        finder_num_partitions=2,
    )

    cut_circuit = ck.find_cuts(circuit.copy(), options=options)
    assert len(cut_circuit.cut_locations) == 1
    location = cut_circuit.cut_locations[0]
    assert location.gate_name == "rzz"

    observables = ["IIIZ", "IIZI", "IZII", "ZIII"]
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    assert experiment.num_groups == 6
    theta = float(location.gate.params[0])
    assert sum(abs(c) for c in experiment.coefficients) == pytest.approx(
        1 + 2 * abs(np.sin(theta))
    )

    results = ck.run_experiments(experiment, backend=AerSimulator())
    values = ck.estimate_expectation_values(results)
    state = Statevector(circuit)
    for pauli, actual in zip(observables, values):
        exact = float(np.real(state.expectation_value(Pauli(pauli))))
        assert abs(exact - actual) < TOLERANCE


def test_gate_weight_is_log_gamma_and_handles_unknown_gates():
    """Weights must distinguish gammas that used to truncate to the same int."""
    import math

    from QCut.QCutFind.graph_circuit_utils import (
        WEIGHT_SCALE,
        cut_weight,
        get_gate_weight,
        get_wire_weight,
    )

    assert get_gate_weight("cz") == round(math.log(3) * WEIGHT_SCALE)
    assert get_gate_weight("swap") == round(math.log(7) * WEIGHT_SCALE)
    assert get_wire_weight("both") == round(math.log(4) * WEIGHT_SCALE)
    # Both of these gammas used to truncate to 1.
    small = get_gate_weight("rzz", "gate", RZZGate(0.05))
    larger = get_gate_weight("rzz", "gate", RZZGate(0.3))
    assert small < larger < get_gate_weight("cz")
    # An rzz(pi/2) is CZ-equivalent, so it must weigh the same.
    assert get_gate_weight("rzz", "gate", RZZGate(np.pi / 2)) == get_gate_weight("cz")
    # A free cut costs no shots but still adds circuits, so it does not weigh nothing.
    assert cut_weight(1.0) == 1
    with pytest.raises(ValueError, match="no gate was supplied"):
        get_gate_weight("rzz")


def test_log_weights_rank_cut_sets_by_their_real_cost():
    """The overhead of a cut set is a product, so the weights have to add up like one.

    Two CZ cuts cost gamma 9 and one SWAP cut costs 7, so the SWAP is cheaper. Adding
    gamma directly says the opposite, since 3 + 3 is less than 7. That is the bug the
    logarithm fixes.
    """
    from QCut.QCutFind.graph_circuit_utils import get_gate_weight

    two_cz = 2 * get_gate_weight("cz")
    one_swap = get_gate_weight("swap")
    assert 3 + 3 < 7  # what the old additive gamma weights compared
    assert two_cz > one_swap  # what the real cost, 9 against 7, demands
