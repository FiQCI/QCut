"""Tests for sampling the quasiprobability decomposition instead of enumerating it."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, UnitaryGate
from qiskit.quantum_info import Pauli, Statevector, random_unitary
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cutGate
from QCut.errors.qcuterror import QCutError
from QCut.qpd.qpd_operations import qpd_for_location, sample_qpd_combinations

OBSERVABLES = ["IZ", "ZI", "ZZ"]


def _circuit(gate):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.ry(0.7, 1)
    circuit.append(**cutGate(gate, 0, 1))
    return circuit


def _reference(gate):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.ry(0.7, 1)
    circuit.append(gate, [0, 1])
    state = Statevector(circuit)
    return np.array(
        [float(np.real(state.expectation_value(Pauli(p)))) for p in OBSERVABLES]
    )


def _run(gate, options, shots=2**12):
    cut_circuit = ck.get_locations_and_subcircuits(_circuit(gate), options=options)
    experiment = ck.get_experiment_circuits(cut_circuit, OBSERVABLES)
    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=shots)
    values = ck.estimate_expectation_values(results)
    return np.array(values), experiment


@pytest.mark.sim
def test_auto_enumerates_a_small_decomposition():
    """The default must not sample where enumeration is affordable and exact."""
    _, experiment = _run(RZZGate(0.9), CutOptions())
    assert not experiment.sampled
    assert experiment.num_groups == 6


def test_auto_samples_once_the_product_is_large():
    circuit = QuantumCircuit(3)
    for qubit in range(3):
        circuit.ry(0.4, qubit)
    for i, (a, b) in enumerate(((0, 1), (1, 2))):
        circuit.append(
            **cutGate(UnitaryGate(random_unitary(4, seed=30 + i).data), a, b)
        )
    options = CutOptions(max_exact_groups=100, num_samples=120, seed=2)
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    exact_groups = 1
    for location in cut_circuit.cut_locations:
        exact_groups *= len(qpd_for_location(location))
    assert exact_groups == 58 * 58

    experiment = ck.get_experiment_circuits(cut_circuit, ["IIZ", "IZI", "ZII"])
    assert experiment.sampled
    assert experiment.num_draws == 120
    assert experiment.num_groups <= 120


@pytest.mark.sim
def test_sampled_gamma_matches_the_exact_gamma():
    """The coefficients are scaled so their one-norm is the true gamma.

    That is what lets the estimator treat a sampled experiment like an enumerated one.
    """
    exact_values, exact_experiment = _run(RZZGate(0.9), CutOptions(expansion="exact"))
    _, sampled = _run(
        RZZGate(0.9), CutOptions(expansion="sample", num_samples=500, seed=1)
    )
    exact_gamma = sum(abs(c) for c in exact_experiment.coefficients)
    sampled_gamma = sum(abs(c) for c in sampled.coefficients)
    assert sampled_gamma == pytest.approx(exact_gamma)
    assert exact_gamma == pytest.approx(1 + 2 * abs(np.sin(0.9)))


@pytest.mark.parametrize("num_samples", [500, 2000])
@pytest.mark.sim
def test_sampling_converges_on_the_exact_answer(num_samples):
    gate = RZZGate(0.9)
    values, experiment = _run(
        gate, CutOptions(expansion="sample", num_samples=num_samples, seed=7)
    )
    assert experiment.sampled
    # Sampling error falls off with the draw count, so the tolerance is loose at 500 and
    # tight at 2000. Enumerating this decomposition exactly needs only 6 groups, so this
    # test is about correctness of the estimator rather than about saving work.
    tolerance = 0.35 if num_samples == 500 else 0.1
    assert np.abs(_reference(gate) - values).max() < tolerance


@pytest.mark.sim
def test_a_seed_makes_the_experiment_reproducible():
    first = _run(RZZGate(0.9), CutOptions(expansion="sample", num_samples=50, seed=11))[
        1
    ]
    second = _run(
        RZZGate(0.9), CutOptions(expansion="sample", num_samples=50, seed=11)
    )[1]
    assert np.allclose(list(first.coefficients), list(second.coefficients))

    other = _run(RZZGate(0.9), CutOptions(expansion="sample", num_samples=50, seed=12))[
        1
    ]
    assert len(other.coefficients) == len(first.coefficients)


def _experiment(options):
    cut_circuit = ck.get_locations_and_subcircuits(
        _circuit(RZZGate(0.9)), options=options
    )
    return ck.get_experiment_circuits(cut_circuit, OBSERVABLES)


def test_an_unseeded_sample_records_its_seed_and_can_be_rebuilt():
    unseeded = _experiment(CutOptions(expansion="sample", num_samples=50))
    assert isinstance(unseeded.seed, int)

    rebuilt = _experiment(
        CutOptions(expansion="sample", num_samples=50, seed=unseeded.seed)
    )
    assert rebuilt.seed == unseeded.seed
    assert np.allclose(list(rebuilt.coefficients), list(unseeded.coefficients))
    assert [
        [{sub: str(c) for sub, c in obs.items()} for obs in group]
        for group in rebuilt.experiments
    ] == [
        [{sub: str(c) for sub, c in obs.items()} for obs in group]
        for group in unseeded.experiments
    ]


def test_a_given_seed_is_kept_and_an_enumeration_has_none():
    seeded = _experiment(CutOptions(expansion="sample", num_samples=50, seed=11))
    assert seeded.seed == 11
    assert _experiment(CutOptions(expansion="exact")).seed is None


@pytest.mark.sim
def test_repeated_draws_are_collapsed():
    """Drawing 500 times from a 6 term decomposition must not build 500 circuits."""
    _, experiment = _run(
        RZZGate(0.9), CutOptions(expansion="sample", num_samples=500, seed=1)
    )
    assert experiment.num_groups <= 6
    assert experiment.num_draws == 500


def test_sampler_probabilities_follow_the_coefficients():
    """Terms are drawn proportional to |c|, which is the low variance choice."""
    circuit = _circuit(RZZGate(0.9))
    locations = ck.get_locations_and_subcircuits(circuit).cut_locations
    qpd = qpd_for_location(locations[0])
    combinations, coefficients, draws = sample_qpd_combinations(
        locations, num_samples=20000, seed=4
    )
    assert draws == 20000
    gamma = sum(abs(term["c"]) for term in qpd)
    assert sum(abs(c) for c in coefficients) == pytest.approx(gamma)

    # Recover the empirical frequency of each term and compare with |c| / gamma.
    frequency = {}
    for terms, coefficient in zip(combinations, coefficients):
        index = next(i for i, t in enumerate(qpd) if t is terms[0])
        frequency[index] = abs(coefficient) / gamma
    for index, term in enumerate(qpd):
        assert frequency[index] == pytest.approx(abs(term["c"]) / gamma, abs=0.02)


def test_bad_options_are_rejected():
    with pytest.raises(QCutError, match="unknown expansion strategy"):
        CutOptions(expansion="nonsense")
    with pytest.raises(QCutError, match="max_exact_groups"):
        CutOptions(max_exact_groups=0)
    with pytest.raises(QCutError, match="num_samples"):
        CutOptions(num_samples=0)


def test_options_must_be_a_cutoptions():
    with pytest.raises(QCutError, match="must be a CutOptions"):
        ck.get_locations_and_subcircuits(_circuit(RZZGate(0.4)), options="exact")


def test_should_sample_thresholds():
    options = CutOptions(max_exact_groups=100)
    assert not options.should_sample(100)
    assert options.should_sample(101)
    assert not CutOptions(expansion="exact").should_sample(10**9)
    assert CutOptions(expansion="sample").should_sample(1)


def test_sample_count_defaults_to_the_threshold():
    assert CutOptions(max_exact_groups=250).sample_count == 250
    assert CutOptions(max_exact_groups=250, num_samples=7).sample_count == 7
