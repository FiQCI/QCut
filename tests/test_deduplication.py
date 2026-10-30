"""Submitting each distinct experiment circuit once, and sharing its result.

A subcircuit's experiment circuit depends on the QPD terms of the cuts touching that
subcircuit and on the measurement setting restricted to its own qubits, so most of an
experiment's slots repeat a circuit. The slot count grows exponentially in the number
of pieces while the distinct count grows linearly.
"""

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate, QuantumRegister
from qiskit.circuit.library import CXGate, CZGate, UnitaryGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Pauli, Statevector, random_unitary
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

import QCut as ck
from QCut import CutOptions, cutGate
from QCut.execution import circuit_knitting as knit
from QCut.utils.circuit_utils import circuit_fingerprint

#: Shot noise on the reconstructed values, in line with the rest of the suite.
TOLERANCE = 0.1

#: Enumerated rather than sampled, and with every decomposition that could merge cuts
#: turned off, so the circuit counts below are fixed numbers rather than a strategy's
#: current preference.
OPTIONS = CutOptions(
    consolidate="never",
    joint_rotation_cuts=False,
    wire_cut_communication="never",
    expansion="exact",
)

CHAIN_OBSERVABLES = ["ZZZZZZ", "IIZZII", "XIIIIZ", "ZIIIIX", "IIIIZZ"]
CHAIN_SLOTS = 324
CHAIN_DISTINCT = 45


def _chain(cut: bool = True) -> QuantumCircuit:
    """Three two-qubit pieces in a row, joined by two cut CX gates."""
    circuit = QuantumCircuit(6)
    for qubit in range(6):
        circuit.ry(0.3 + 0.2 * qubit, qubit)
    circuit.cx(0, 1)
    if cut:
        circuit.append(**cutGate(CXGate(), 1, 2))
    else:
        circuit.cx(1, 2)
    circuit.cx(2, 3)
    if cut:
        circuit.append(**cutGate(CXGate(), 3, 4))
    else:
        circuit.cx(3, 4)
    circuit.cx(4, 5)
    return circuit


def _wings(cut: bool = True) -> QuantumCircuit:
    """Two mirrored wings cut off a middle piece, so two subcircuits come out equal.

    Both cuts put their controlled side on the middle piece, and the wings are built
    alike on wires that map to the same local indices, so the wing subcircuits are
    structurally identical. They must still never share a sample.
    """
    circuit = QuantumCircuit(6)
    for qubit in range(6):
        circuit.ry(0.3, qubit)
    circuit.cx(0, 1)
    circuit.cx(4, 5)
    circuit.cx(2, 3)
    if cut:
        circuit.append(**cutGate(CZGate(), 2, 1))
        circuit.append(**cutGate(CZGate(), 3, 5))
    else:
        circuit.cz(2, 1)
        circuit.cz(3, 5)
    return circuit


WING_OBSERVABLES = ["ZZZZZZ", "IZZZZI"]
WING_DISTINCT = 35
#: What the same experiment would collapse to if the key dropped the subcircuit index.
WING_DISTINCT_WITHOUT_SUBCIRCUIT = 30


def _experiment(circuit, observables, deduplicate=True, backend=None):
    cut_circuit = ck.get_locations_and_subcircuits(
        circuit, options=OPTIONS.replace(deduplicate=deduplicate)
    )
    if backend is not None:
        cut_circuit = ck.transpile_subcircuits(
            cut_circuit, backend, optimization_level=3
        )
    return ck.get_experiment_circuits(cut_circuit, observables)


def _exact(circuit, observables):
    state = Statevector(circuit)
    return np.array(
        [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]
    )


def _runnable_circuits(experiment):
    return [circuit for _slot, circuit in knit._runnable(experiment)[0]]


# --------------------------------------------------------------------------- #
# The fingerprint itself
# --------------------------------------------------------------------------- #


def _base() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.rz(0.3, 1)
    circuit.measure(0, 0)
    return circuit


def test_a_copy_fingerprints_the_same():
    assert circuit_fingerprint(_base()) == circuit_fingerprint(_base().copy())


def test_the_fingerprint_is_hashable():
    assert len({circuit_fingerprint(_base()), circuit_fingerprint(_base())}) == 1


def test_an_added_gate_changes_the_fingerprint():
    other = _base()
    other.x(1)
    assert circuit_fingerprint(_base()) != circuit_fingerprint(other)


def test_a_changed_parameter_changes_the_fingerprint():
    other = QuantumCircuit(2, 2)
    other.h(0)
    other.rz(0.4, 1)
    other.measure(0, 0)
    assert circuit_fingerprint(_base()) != circuit_fingerprint(other)


def test_a_different_clbit_target_changes_the_fingerprint():
    other = QuantumCircuit(2, 2)
    other.h(0)
    other.rz(0.3, 1)
    other.measure(0, 1)
    assert circuit_fingerprint(_base()) != circuit_fingerprint(other)


def test_a_different_register_layout_changes_the_fingerprint():
    """The layout decides how a counts key is split, so it is part of the circuit.

    These two agree on every instruction and on the number of classical bits. They
    differ only in how those bits are grouped, which is exactly what
    :func:`QCut.utils.circuit_utils.compact_qpd_register` changes and what decides
    whether the counts come back as one field or two.
    """
    one = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(2, "meas"))
    two = QuantumCircuit(
        QuantumRegister(1, "q"),
        ClassicalRegister(1, "qpd_meas"),
        ClassicalRegister(1, "meas"),
    )
    one.measure(0, 0)
    two.measure(0, 0)
    assert one.num_clbits == two.num_clbits
    assert circuit_fingerprint(one) != circuit_fingerprint(two)


def test_same_name_different_definition_differs():
    """QPD operations are custom instructions, and their names do collide."""

    def custom(angle):
        body = QuantumCircuit(1)
        body.rx(angle, 0)
        gate = Gate(name="qpd_op", num_qubits=1, params=[])
        gate.definition = body
        return gate

    one = QuantumCircuit(1)
    one.append(custom(0.3), [0])
    two = QuantumCircuit(1)
    two.append(custom(0.7), [0])
    assert circuit_fingerprint(one) != circuit_fingerprint(two)


def test_ndarray_parameters_are_handled():
    one = QuantumCircuit(2)
    one.append(UnitaryGate(random_unitary(4, seed=1)), [0, 1])
    two = QuantumCircuit(2)
    two.append(UnitaryGate(random_unitary(4, seed=2)), [0, 1])
    assert circuit_fingerprint(one) == circuit_fingerprint(one.copy())
    assert circuit_fingerprint(one) != circuit_fingerprint(two)


def test_the_fingerprint_agrees_with_qiskits_own_equality():
    """The key is hand-rolled, so check it against the thing it stands in for.

    Qiskit's circuit equality ignores name and metadata, which this does too, and
    compares the global phase, which this does not -- so it may merge more, never less.
    Anything inside one bucket must compare equal, and no two buckets may.
    """
    circuits = _runnable_circuits(_experiment(_chain(), CHAIN_OBSERVABLES))
    cache: dict = {}
    buckets: dict = {}
    for circuit in circuits:
        buckets.setdefault(circuit_fingerprint(circuit, cache), []).append(circuit)

    for members in buckets.values():
        assert all(other == members[0] for other in members[1:])

    representatives = [members[0] for members in buckets.values()]
    for index, one in enumerate(representatives):
        assert not any(one == other for other in representatives[index + 1 :])


# --------------------------------------------------------------------------- #
# How many circuits a run submits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("deduplicate", "circuits"), [(True, CHAIN_DISTINCT), (False, CHAIN_SLOTS)]
)
def test_the_estimate_counts_the_circuits_that_are_submitted(deduplicate, circuits):
    experiment = _experiment(_chain(), CHAIN_OBSERVABLES, deduplicate=deduplicate)
    assert experiment.num_circuits == CHAIN_SLOTS
    assert ck.estimate_run(experiment, shots=1024).circuits == circuits


@pytest.mark.sim
def test_the_estimate_matches_what_the_backend_is_asked_to_run():
    experiment = _experiment(_chain(), CHAIN_OBSERVABLES)
    estimate = ck.estimate_run(experiment, shots=256, max_batch_size=20)

    calls = []
    backend = AerSimulator()
    original = backend.run

    def counting(circuits, **kwargs):
        calls.append((len(circuits), kwargs.get("shots")))
        return original(circuits, **kwargs)

    backend.run = counting
    ck.run_experiments(experiment, shots=256, backend=backend, max_batch_size=20)

    assert estimate.jobs == len(calls)
    assert estimate.circuits == sum(circuits for circuits, _ in calls)
    assert estimate.shots == sum(circuits * shots for circuits, shots in calls)


# --------------------------------------------------------------------------- #
# That the results end up in the right places
# --------------------------------------------------------------------------- #


@pytest.mark.sim
def test_every_slot_receives_the_result_of_an_identical_circuit():
    """The guard against a dedup bug that is invisible in an expectation value.

    A slot handed the wrong circuit's counts still produces a number, just a wrong one,
    and shot noise hides a lot. Comparing fingerprints catches it outright.
    """
    experiment = _experiment(_chain(), CHAIN_OBSERVABLES)
    batches = []
    original = knit._submit

    def spy(backend, circuits, shots, max_batch_size, run_options):
        job = original(backend, circuits, shots, max_batch_size, run_options)
        batches.append((list(circuits), job))
        return job

    knit._submit = spy
    try:
        results = ck.run_experiments(
            experiment, shots=128, backend=AerSimulator(seed_simulator=1)
        )
    finally:
        knit._submit = original

    assert sum(len(circuits) for circuits, _job in batches) == CHAIN_DISTINCT

    # ``index`` is a position within the circuit's own job, so the job has to be found
    # first. The result object is what ties a slot back to the batch it came from.
    cache: dict = {}
    by_result = {
        id(knit._result(job)): [circuit_fingerprint(c, cache) for c in circuits]
        for circuits, job in batches
    }
    checked = 0
    for group, circuit_group in enumerate(experiment.experiments):
        for obs, obs_group in enumerate(circuit_group):
            for sub, circuit in obs_group.items():
                held = results.results[group][obs][sub]
                if isinstance(held.result, dict):
                    continue  # nothing to measure, so nothing was submitted
                keys = by_result[id(held.result)]
                assert keys[held.index] == circuit_fingerprint(circuit, cache)
                checked += 1
    assert checked == CHAIN_SLOTS


@pytest.mark.sim
def test_slots_sharing_a_circuit_hold_one_result_object():
    experiment = _experiment(_chain(), CHAIN_OBSERVABLES)
    results = ck.run_experiments(
        experiment, shots=128, backend=AerSimulator(seed_simulator=1)
    )
    held = {
        (group, obs, sub): results.results[group][obs][sub]
        for group, circuit_group in enumerate(results.results)
        for obs, obs_group in enumerate(circuit_group)
        for sub in obs_group
    }
    assert len(held) == CHAIN_SLOTS
    assert len({id(one) for one in held.values()}) == CHAIN_DISTINCT

    # The slots of one distinct circuit hold the same object, not equal copies.
    by_object: dict = {}
    for slot, result in held.items():
        by_object.setdefault(id(result), []).append(slot)
    shared = max(by_object.values(), key=len)
    assert len(shared) > 1
    assert all(held[slot] is held[shared[0]] for slot in shared)


# --------------------------------------------------------------------------- #
# That the numbers stay right
# --------------------------------------------------------------------------- #


@pytest.mark.sim
@pytest.mark.parametrize("deduplicate", [True, False])
def test_the_chain_reproduces_the_uncut_circuit(deduplicate):
    exact = _exact(_chain(cut=False), CHAIN_OBSERVABLES)
    experiment = _experiment(_chain(), CHAIN_OBSERVABLES, deduplicate=deduplicate)
    results = ck.run_experiments(
        experiment, shots=8192, backend=AerSimulator(seed_simulator=3)
    )
    values = np.array(ck.estimate_expectation_values(results))
    assert np.abs(values - exact).max() < TOLERANCE, f"{values} against {exact}"


@pytest.mark.sim
def test_a_sampler_gives_the_same_answer_as_a_backend():
    exact = _exact(_chain(cut=False), CHAIN_OBSERVABLES)
    experiment = _experiment(_chain(), CHAIN_OBSERVABLES)
    results = ck.run_experiments(experiment, shots=8192, backend=SamplerV2())
    values = np.array(ck.estimate_expectation_values(results))
    assert np.abs(values - exact).max() < TOLERANCE, f"{values} against {exact}"


@pytest.mark.sim
def test_transpiled_subcircuits_still_deduplicate():
    """Layout is recorded in metadata, which the key leaves out, so nothing splits."""
    backend = GenericBackendV2(num_qubits=5, basis_gates=["cz", "r"], seed=11)
    experiment = _experiment(_chain(), CHAIN_OBSERVABLES, backend=backend)
    assert ck.estimate_run(experiment, shots=1024).circuits == CHAIN_DISTINCT

    exact = _exact(_chain(cut=False), CHAIN_OBSERVABLES)
    results = ck.run_experiments(
        experiment, shots=8192, backend=AerSimulator(seed_simulator=5)
    )
    values = np.array(ck.estimate_expectation_values(results))
    assert np.abs(values - exact).max() < TOLERANCE, f"{values} against {exact}"


# --------------------------------------------------------------------------- #
# The subcircuit index in the key
# --------------------------------------------------------------------------- #


def test_identical_subcircuits_are_not_merged():
    """Two factors of one product must never share a sample.

    Each group's estimate is a product with one factor per subcircuit. Two subcircuits
    sharing a sample would make that ``E[X**2]`` where ``E[X]**2`` is meant, which is
    biased rather than merely noisier. The subcircuit index in the key is what stops
    it, and this fixture is one where dropping it really would merge.
    """
    experiment = _experiment(_wings(), WING_OBSERVABLES)
    assert ck.estimate_run(experiment, shots=1024).circuits == WING_DISTINCT

    cache: dict = {}
    without = {
        circuit_fingerprint(circuit, cache)
        for circuit in _runnable_circuits(experiment)
    }
    assert len(without) == WING_DISTINCT_WITHOUT_SUBCIRCUIT, (
        "fixture no longer has identical subcircuits, so it guards nothing"
    )


@pytest.mark.sim
def test_the_symmetric_circuit_still_reproduces_the_uncut_one():
    exact = _exact(_wings(cut=False), WING_OBSERVABLES)
    experiment = _experiment(_wings(), WING_OBSERVABLES)
    results = ck.run_experiments(
        experiment, shots=8192, backend=AerSimulator(seed_simulator=4)
    )
    values = np.array(ck.estimate_expectation_values(results))
    assert np.abs(values - exact).max() < TOLERANCE, f"{values} against {exact}"
