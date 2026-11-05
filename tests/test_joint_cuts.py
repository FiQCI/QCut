"""Tests for cutting several parallel two-qubit rotation gates together."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library import (
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CXGate,
    CZGate,
    DCXGate,
    RXXGate,
    RYYGate,
    RZXGate,
    RZZGate,
    SwapGate,
    XXPlusYYGate,
    iSwapGate,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import (
    Operator,
    Pauli,
    Statevector,
    random_unitary,
)
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cutCZ, cutGate
from QCut.qpd.bundle import plan_bundles
from QCut.qpd.qpd_generate import qpd_from_gate
from QCut.qpd.qpd_joint import (
    gamma_joint,
    gamma_separate,
    joint_rotation_qpd,
    joint_rotation_qpd_from_gates,
    single_axis_frame,
)

#: Shot budget for the end-to-end tests. These assert that a bundle keeps the answer,
#: whose failure mode is a grossly wrong value rather than a drift, so they do not need
#: shot-noise precision. Measured over six runs the worst error here is 0.042, leaving
#: the tolerance below a comfortable factor of two.
SHOTS = 2**12

TOLERANCE = 0.1

_SUP = lambda m: np.kron(m, np.conj(m))  # noqa: E731


def _embed(single: np.ndarray, qubit: int, width: int) -> np.ndarray:
    """Place a one-qubit operator, little-endian to match qiskit's Operator."""
    out = np.eye(1)
    for position in range(width - 1, -1, -1):
        out = np.kron(out, single if position == qubit else np.eye(2))
    return out


def _side_superop(circuit: QuantumCircuit) -> np.ndarray:
    """Superoperator of one side's operation, measurements weighted by their outcome."""
    width = circuit.num_qubits
    out = np.eye((2**width) ** 2, dtype=complex)
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure):
            qubit = circuit.find_bit(instruction.qubits[0]).index
            projectors = [
                _embed(np.diag([1 - bit, bit]).astype(complex), qubit, width)
                for bit in (0, 1)
            ]
            out = (_SUP(projectors[0]) - _SUP(projectors[1])) @ out
        else:
            padded = QuantumCircuit(width)
            padded.append(
                instruction.operation,
                [circuit.find_bit(q).index for q in instruction.qubits],
            )
            out = _SUP(Operator(padded).data) @ out
    return out


def _channel_from_qpd(qpd: list[dict], width: int) -> np.ndarray:
    """Reassemble the channel a joint QPD represents.

    Side 1 holds the higher qubit indices of the joined system, so under qiskit's
    little-endian ordering it is the left factor of the tensor product.
    """
    dim = 2**width
    total = np.zeros(((dim * dim) ** 2,) * 2, dtype=complex)
    for term in qpd:
        left = _side_superop(term["op_1"]).reshape((dim,) * 4)
        right = _side_superop(term["op_0"]).reshape((dim,) * 4)
        joint = np.einsum("ijkl,IJKL->iIjJkKlL", left, right)
        total += term["c"] * joint.reshape((dim * dim) ** 2, (dim * dim) ** 2)
    return total


def _apply_qpd(qpd: list[dict], width: int, rho: np.ndarray) -> np.ndarray:
    """Act with the channel a joint QPD represents, without ever forming it.

    :func:`_channel_from_qpd` builds the joint superoperator, which at three gates is a
    4096x4096 array per term. Contracting each side against the input instead costs a
    few hundred thousand operations rather than millions, and needs no large array. The
    index order is the one that function documents: side 1 is the left tensor factor, so
    reshaping a row-major ``rho`` splits both of its indices into (side 1, side 0).
    """
    dim = 2**width
    inp = rho.reshape((dim,) * 4)
    out = np.zeros_like(inp)
    for term in qpd:
        left = _side_superop(term["op_1"]).reshape((dim,) * 4)
        right = _side_superop(term["op_0"]).reshape((dim,) * 4)
        out += term["c"] * np.einsum(
            "ijkl,IJKL,kKlL->iIjJ", left, right, inp, optimize=True
        )
    return out.reshape(dim * dim, dim * dim)


def _target_action(thetas: list[float], rho: np.ndarray) -> np.ndarray:
    """Act with the product of the rotation channels on ``rho``."""
    width = len(thetas)
    circuit = QuantumCircuit(2 * width)
    for gate, theta in enumerate(thetas):
        circuit.rzz(theta, gate, width + gate)
    unitary = Operator(circuit).data
    return unitary @ rho @ unitary.conj().T


def _target_channel(thetas: list[float]) -> np.ndarray:
    width = len(thetas)
    circuit = QuantumCircuit(2 * width)
    for gate, theta in enumerate(thetas):
        circuit.rzz(theta, gate, width + gate)
    return _SUP(Operator(circuit).data)


ANGLE_SETS = [
    [0.3],
    [1.7],
    [np.pi / 2],
    [0.4, 0.9],
    [np.pi / 2, np.pi / 2],
    [2.4, 0.15],
    [0.3, 1.1, 2.0],
]


@pytest.mark.parametrize("thetas", ANGLE_SETS, ids=lambda t: f"n{len(t)}")
def test_joint_qpd_reconstructs_the_channel(thetas):
    """The joint table must sum to exactly the product of the rotation channels.

    Pins Eq. (C16)'s signs, the i > j ordering and the parity-measurement convention.
    """
    qpd = joint_rotation_qpd(thetas)
    width = len(thetas)
    if width <= 2:
        # Small enough to compare the channels outright, which is the strongest form.
        error = np.abs(_channel_from_qpd(qpd, width) - _target_channel(thetas)).max()
    else:
        # The full superoperator is 4096x4096 per term here. Two linear maps agree iff
        # their difference annihilates a spanning set, so a difference that is not
        # identically zero shows up on a generic input with probability one. Several
        # random inputs make that decisive numerically at a fraction of the cost.
        dim = 2**width
        rng = np.random.default_rng(20240117)
        error = 0.0
        for _ in range(3):
            rho = rng.normal(size=(dim * dim,) * 2) + 1j * rng.normal(
                size=(dim * dim,) * 2
            )
            difference = _apply_qpd(qpd, width, rho) - _target_action(thetas, rho)
            error = max(error, float(np.abs(difference).max() / np.abs(rho).max()))
    assert error < 1e-12


@pytest.mark.parametrize("thetas", ANGLE_SETS, ids=lambda t: f"n{len(t)}")
def test_joint_gamma_matches_the_closed_form(thetas):
    qpd = joint_rotation_qpd(thetas)
    assert sum(abs(term["c"]) for term in qpd) == pytest.approx(gamma_joint(thetas))


def test_joint_gamma_never_loses_to_separate_cuts():
    for thetas in ANGLE_SETS:
        assert gamma_joint(thetas) <= gamma_separate(thetas) + 1e-12


@pytest.mark.parametrize(
    ("size", "expected"),
    [(1, 6), (2, 30), (3, 132), (4, 552)],
    ids=["n1", "n2", "n3", "n4"],
)
def test_joint_term_count(size, expected):
    """Bundling lowers the group count as well as gamma, so the counts are pinned."""
    thetas = [0.3 + 0.2 * index for index in range(size)]
    assert len(joint_rotation_qpd(thetas)) == expected
    assert expected <= 6**size


@pytest.mark.parametrize("theta", [0.3, 1.7, np.pi / 2, 2.4])
def test_one_gate_agrees_with_the_single_gate_generator(theta):
    """A bundle of one has to be the table qpd_from_gate already produces."""
    joint = _channel_from_qpd(joint_rotation_qpd([theta]), 1)
    separate = _channel_from_qpd(
        [
            {"op_0": term["op_0"], "op_1": term["op_1"], "c": term["c"]}
            for term in qpd_from_gate(RZZGate(theta))
        ],
        1,
    )
    assert np.abs(joint - separate).max() < 1e-12


SINGLE_AXIS = [
    ("cx", CXGate()),
    ("cz", CZGate()),
    ("rzz", RZZGate(0.3)),
    ("rzz-folded", RZZGate(2.4)),
    ("rxx", RXXGate(1.1)),
    ("ryy", RYYGate(0.7)),
    ("rzx", RZXGate(1.9)),
    ("crz", CRZGate(0.8)),
    ("crx", CRXGate(1.9)),
    ("cry", CRYGate(0.55)),
    ("cp", CPhaseGate(0.7)),
]


@pytest.mark.parametrize(
    ("name", "gate"), SINGLE_AXIS, ids=[name for name, _ in SINGLE_AXIS]
)
def test_single_axis_frame_reproduces_the_gate(name, gate):
    """The returned angle and locals must rebuild the gate, fold included."""
    frame = single_axis_frame(gate)
    assert frame is not None
    theta, locals_ = frame
    rebuilt = QuantumCircuit(2)
    for side, (pre, _) in enumerate(locals_):
        if pre is not None:
            rebuilt.append(pre, [side])
    rebuilt.rzz(theta, 0, 1)
    for side, (_, post) in enumerate(locals_):
        if post is not None:
            rebuilt.append(post, [side])
    error = np.abs(_SUP(Operator(rebuilt).data) - _SUP(gate.to_matrix())).max()
    assert error < 1e-12


@pytest.mark.parametrize(
    "gate",
    [SwapGate(), iSwapGate(), DCXGate(), XXPlusYYGate(0.9, 0.4)],
    ids=["swap", "iswap", "dcx", "xx_plus_yy"],
)
def test_two_axis_gates_are_rejected(gate):
    assert single_axis_frame(gate) is None


def test_random_gate_is_rejected():
    gate = Operator(random_unitary(4, seed=5)).to_instruction()
    assert single_axis_frame(gate) is None


def test_flipping_a_gate_swaps_only_its_locals():
    """A single-axis gate is symmetric about its axis, so a reversed cut bundles."""
    gates = [CRZGate(0.8), CRZGate(1.4)]
    flipped = joint_rotation_qpd_from_gates(gates, [False, True])
    assert flipped is not None

    reference = QuantumCircuit(4)
    reference.append(gates[0], [0, 2])
    reference.append(gates[1], [3, 1])  # gate 1 the other way round
    error = np.abs(_channel_from_qpd(flipped, 2) - _SUP(Operator(reference).data)).max()
    assert error < 1e-12


def test_from_gates_returns_none_for_a_two_axis_gate():
    assert joint_rotation_qpd_from_gates([RZZGate(0.3), SwapGate()]) is None


def _two_partition_circuit(thetas, cut=True, marker=None):
    """Four qubits split as {0,1} against {2,3}, joined by two parallel gates."""
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.ry(0.7, 1)
    circuit.rx(0.4, 2)
    circuit.h(3)
    circuit.rzz(0.5, 0, 1)
    circuit.rzz(0.6, 2, 3)
    if not cut:
        circuit.rzz(thetas[0], 0, 2)
        circuit.rzz(thetas[1], 1, 3)
    elif marker is None:
        circuit.append(**cutGate(RZZGate(thetas[0]), 0, 2))
        circuit.append(**cutGate(RZZGate(thetas[1]), 1, 3))
    else:
        circuit.append(marker(), [0, 2])
        circuit.append(marker(), [1, 3])
    circuit.rz(0.3, 0)
    circuit.ry(0.2, 3)
    return circuit


OBSERVABLES = ["IIIZ", "IIZI", "IZII", "ZIII", "IIZZ", "ZZII"]


def _exact(circuit):
    state = Statevector(circuit)
    return [float(np.real(state.expectation_value(Pauli(p)))) for p in OBSERVABLES]


def _run(circuit, options):
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    experiment = ck.get_experiment_circuits(cut_circuit, OBSERVABLES)
    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
    values = ck.estimate_expectation_values(results)
    gamma = sum(abs(c) for c in experiment.coefficients)
    return values, experiment.num_groups, gamma


@pytest.mark.slow
@pytest.mark.sim
def test_bundling_lowers_the_cost_and_keeps_the_answer():
    """Two parallel rzz cost gamma 7.51 over 36 groups apart, 6.00 over 30 together."""
    thetas = (0.9, 1.3)
    exact = _exact(_two_partition_circuit(thetas, cut=False))

    together, groups_on, gamma_on = _run(_two_partition_circuit(thetas), CutOptions())
    apart, groups_off, gamma_off = _run(
        _two_partition_circuit(thetas), CutOptions(joint_rotation_cuts=False)
    )

    assert (groups_on, groups_off) == (30, 36)
    assert gamma_on == pytest.approx(gamma_joint(list(thetas)))
    assert gamma_off == pytest.approx(gamma_separate(list(thetas)))
    assert gamma_on < gamma_off
    for expected, with_, without in zip(exact, together, apart):
        assert abs(expected - with_) < TOLERANCE
        assert abs(expected - without) < TOLERANCE


@pytest.mark.sim
def test_two_parallel_cz_cuts_bundle_through_the_named_marker():
    """cutCZ carries no gate of its own, so the bundle has to rebuild it."""
    circuit = _two_partition_circuit((0, 0), marker=cutCZ)
    reference = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name == "CutCZ":
            reference.cz(*[circuit.find_bit(q).index for q in instruction.qubits])
        else:
            reference.append(
                instruction.operation, instruction.qubits, instruction.clbits
            )

    values, groups, gamma = _run(circuit, CutOptions())
    assert (groups, gamma) == (30, pytest.approx(7.0))
    for expected, actual in zip(_exact(reference), values):
        assert abs(expected - actual) < TOLERANCE


@pytest.mark.sim
def test_sampling_works_over_bundles():
    """The sampler draws per bundle, so its coefficients must still sum to gamma."""
    thetas = (0.9, 1.3)
    options = CutOptions(expansion="sample", num_samples=400, seed=11)
    values, groups, gamma = _run(_two_partition_circuit(thetas), options)
    assert groups <= 30
    assert gamma == pytest.approx(gamma_joint(list(thetas)))
    # Four hundred draws off a fixed seed pick a deterministic subset of the terms, and
    # that subset is off by about 0.23 however many shots it is given, so this bound is
    # about the sampling and not the statistics. It is loose because the failure it
    # guards against, a sampler that draws the wrong terms, misses by order one.
    for expected, actual in zip(
        _exact(_two_partition_circuit(thetas, cut=False)), values
    ):
        assert abs(expected - actual) < 0.4


def _bundles(circuit, options=None):
    cut_circuit = ck.get_locations_and_subcircuits(
        circuit, options=options or CutOptions()
    )
    from QCut.utils.circuit import _remove_obsm_2

    _remove_obsm_2(cut_circuit.subcircuits)
    return plan_bundles(
        cut_circuit.cut_locations, cut_circuit.subcircuits, cut_circuit.options
    )


def test_cuts_joining_different_subcircuit_pairs_do_not_bundle():
    """A bundle side has to sit in one subcircuit, or its operation cannot be placed."""
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.h(qubit)
    circuit.append(**cutGate(RZZGate(0.9), 0, 2))
    circuit.append(**cutGate(RZZGate(1.3), 1, 3))
    assert all(bundle.size == 1 for bundle in _bundles(circuit))


def _staggered_circuit(blockers):
    """Two parallel cuts with local gates wedged between their two slots.

    ``blockers`` lists the qubits to put an ``rz`` on, between the first cut's slot and
    the second's.
    """
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.h(qubit)
    circuit.rzz(0.5, 0, 1)
    circuit.rzz(0.6, 2, 3)
    circuit.append(**cutGate(RZZGate(0.9), 0, 2))
    for qubit in blockers:
        circuit.rz(0.3, qubit)
    circuit.append(**cutGate(RZZGate(1.3), 1, 3))
    return circuit


def test_a_gate_on_the_later_wire_still_allows_bundling():
    """The block can go in at the later slot instead, since rz(1) misses qubit 0."""
    (bundle,) = [b for b in _bundles(_staggered_circuit([1])) if b.size > 1]
    assert bundle.cuts == (0, 1)
    # Each side picks its own slot. Only subcircuit 0 carries the blocker.
    assert bundle.place[0] == "last"


def test_a_gate_on_each_wire_between_the_cuts_prevents_bundling():
    """Now neither slot works, so the cuts are not parallel and stay separate."""
    assert all(bundle.size == 1 for bundle in _bundles(_staggered_circuit([1, 0])))


@pytest.mark.sim
def test_delayed_placement_keeps_the_answer():
    """Placing the block at the later slot moves a gate forward, so check the value."""
    circuit = _staggered_circuit([1])
    reference = circuit.copy_empty_like()
    thetas = iter((0.9, 1.3))
    pairs = iter(((0, 2), (1, 3)))
    for instruction in circuit.data:
        if instruction.operation.name.startswith("Cut"):
            reference.rzz(next(thetas), *next(pairs))
        else:
            reference.append(
                instruction.operation, instruction.qubits, instruction.clbits
            )
    exact = _exact(reference)

    values, groups, _ = _run(circuit, CutOptions())
    assert groups == 30
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE


def test_two_axis_cuts_do_not_bundle():
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.h(qubit)
    circuit.rzz(0.5, 0, 1)
    circuit.rzz(0.6, 2, 3)
    circuit.append(**cutGate(SwapGate(), 0, 2))
    circuit.append(**cutGate(SwapGate(), 1, 3))
    assert all(bundle.size == 1 for bundle in _bundles(circuit))


def test_bundling_can_be_turned_off():
    circuit = _two_partition_circuit((0.9, 1.3))
    assert all(
        bundle.size == 1
        for bundle in _bundles(circuit, CutOptions(joint_rotation_cuts=False))
    )


def test_bundle_layout_covers_every_placeholder_once():
    """Each side holds one placeholder per cut, else the block gets wrong qubits."""
    (bundle,) = [b for b in _bundles(_two_partition_circuit((0.9, 1.3))) if b.size > 1]
    assert bundle.size == 2
    assert sorted(bundle.cuts) == list(bundle.cuts)
    flat = [entry for side in bundle.layout for entry in side]
    assert len(flat) == 2 * bundle.size
    assert len(set(flat)) == len(flat)
    for side in bundle.layout:
        assert [cut for cut, _ in side] == list(bundle.cuts)


def test_three_parallel_gates_bundle_into_one_decomposition():
    circuit = QuantumCircuit(6)
    for qubit in range(6):
        circuit.h(qubit)
    circuit.rzz(0.5, 0, 1)
    circuit.rzz(0.5, 1, 2)
    circuit.rzz(0.6, 3, 4)
    circuit.rzz(0.6, 4, 5)
    thetas = [0.9, 1.3, 0.5]
    for index, theta in enumerate(thetas):
        circuit.append(**cutGate(RZZGate(theta), index, 3 + index))

    bundles = _bundles(circuit)
    joint = [bundle for bundle in bundles if bundle.size > 1]
    assert len(joint) == 1
    assert joint[0].size == 3
    assert gamma_joint(thetas) < gamma_separate(thetas)


def test_every_cut_lands_in_exactly_one_bundle():
    bundles = _bundles(_two_partition_circuit((0.9, 1.3)))
    covered = sorted(cut for bundle in bundles for cut in bundle.cuts)
    assert covered == list(range(len(covered)))


def test_all_bit_patterns_appear_in_the_diagonal():
    """One diagonal term per bit pattern, so a dropped pattern would be visible."""
    qpd = joint_rotation_qpd([0.7, 1.1])
    diagonal = [term for term in qpd if term["op_0"].num_clbits == 0]
    names = {term["op_0"].name for term in diagonal}
    assert {
        "Z_" + "".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=2)
    } <= names


def _three_gate_circuit(thetas, cut=True):
    """Six qubits split as {0,1,2} against {3,4,5}, joined by three parallel gates."""
    circuit = QuantumCircuit(6)
    for qubit in range(6):
        circuit.h(qubit)
    circuit.rzz(0.5, 0, 1)
    circuit.rzz(0.5, 1, 2)
    circuit.rzz(0.6, 3, 4)
    circuit.rzz(0.6, 4, 5)
    for index, theta in enumerate(thetas):
        if cut:
            circuit.append(**cutGate(RZZGate(theta), index, 3 + index))
        else:
            circuit.rzz(theta, index, 3 + index)
    return circuit


SIX_QUBIT_OBSERVABLES = [
    "IIIIIZ",
    "IIIIZI",
    "IIIZII",
    "IIZIII",
    "IZIIII",
    "ZIIIII",
    "IIIIZZ",
    "ZZIIII",
]


@pytest.mark.slow
@pytest.mark.sim
def test_three_parallel_gates_run_end_to_end():
    """Three cuts in one decomposition: 132 groups against 216, gamma 9.36 against 27.

    Exercises the joint operations at width three, where the parity measurement and the
    rotation both need a real CNOT ladder rather than a single-qubit gate.
    """
    thetas = [0.9, 1.3, 0.5]
    state = Statevector(_three_gate_circuit(thetas, cut=False))
    exact = [
        float(np.real(state.expectation_value(Pauli(p)))) for p in SIX_QUBIT_OBSERVABLES
    ]

    cut_circuit = ck.get_locations_and_subcircuits(
        _three_gate_circuit(thetas), options=CutOptions()
    )
    experiment = ck.get_experiment_circuits(cut_circuit, SIX_QUBIT_OBSERVABLES)
    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
    values = ck.estimate_expectation_values(results)

    assert experiment.num_groups == 132
    gamma = sum(abs(c) for c in experiment.coefficients)
    assert gamma == pytest.approx(gamma_joint(thetas))
    assert gamma < gamma_separate(thetas)
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE


def test_a_trailing_gate_on_one_wire_shrinks_the_bundle():
    """Cut 0 can slide neither back past rzz(1,2) nor forward past its own rz."""
    circuit = _three_gate_circuit([0.9, 1.3, 0.5])
    circuit.rz(0.25, 0)
    bundles = [bundle for bundle in _bundles(circuit) if bundle.size > 1]
    assert len(bundles) == 1
    assert bundles[0].cuts == (1, 2)


def test_joint_blocks_transpile_to_the_iqm_basis():
    """The blocks add CNOT ladders and parity measurements, so check they compile."""
    backend = GenericBackendV2(num_qubits=5, basis_gates=["cz", "r"])
    cut_circuit = ck.get_locations_and_subcircuits(
        _two_partition_circuit((0.9, 1.3)), options=CutOptions()
    )
    experiment = ck.get_experiment_circuits(cut_circuit, OBSERVABLES)
    assert experiment.num_groups == 30

    transpiled = ck.transpile_experiments(experiment, backend, optimization_level=3)
    assert transpiled.num_circuits == experiment.num_circuits
    allowed = {"cz", "r", "measure", "barrier"}
    for group in transpiled.experiments:
        for obs_set in group:
            for circuit in obs_set.values():
                for instruction in circuit.data:
                    assert instruction.operation.name.lower() in allowed


def _triangles_circuit():
    """Two dense triangles joined by two parallel rzz, the cheapest cuts available."""
    circuit = QuantumCircuit(6)
    for qubit in range(6):
        circuit.ry(0.3 + 0.1 * qubit, qubit)
    for left, right in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        circuit.rzz(1.4, left, right)
    circuit.rzz(0.6, 0, 3)
    circuit.rzz(0.6, 1, 4)
    return circuit


@pytest.mark.slow
@pytest.mark.sim
def test_find_cuts_bundles_the_cuts_it_chooses():
    """Automatic cut finding gets joint cutting without asking for it."""
    circuit = _triangles_circuit()
    state = Statevector(circuit)
    exact = [
        float(np.real(state.expectation_value(Pauli(p)))) for p in SIX_QUBIT_OBSERVABLES
    ]

    seen = {}
    for label, options in [
        ("joint", CutOptions(finder_num_partitions=2)),
        ("separate", CutOptions(finder_num_partitions=2, joint_rotation_cuts=False)),
    ]:
        cut_circuit = ck.find_cuts(circuit.copy(), options=options)
        experiment = ck.get_experiment_circuits(cut_circuit, SIX_QUBIT_OBSERVABLES)
        results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
        values = ck.estimate_expectation_values(results)
        seen[label] = (
            experiment.num_groups,
            sum(abs(c) for c in experiment.coefficients),
        )
        for expected, actual in zip(exact, values):
            assert abs(expected - actual) < TOLERANCE

    assert seen["joint"][0] < seen["separate"][0]
    assert seen["joint"][1] < seen["separate"][1]
