"""Tests for the shot-noise bound on estimated expectation values."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut, cutGate
from QCut.job.cutresult import RawResult

SHOTS = 1000


def _experiment(observables=None, options=None, qubits=None):
    """A gate cut and a wire cut, whose 48 groups all weigh 0.25."""
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.4 + 0.3 * qubit, qubit)
    circuit.cx(0, 1)
    circuit.append(**cutGate(CXGate(), 1, 2))
    circuit.append(cut(), [2])
    circuit.cx(2, 3)
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    return ck.get_experiment_circuits(cut_circuit, observables, qubits)


def _errors(experiment, shots=SHOTS):
    # Only the experiment and the shots are read, so nothing needs running.
    return ck.estimate_standard_errors(RawResult([], shots, experiment))


def test_an_enumerated_experiment_is_bounded_by_its_coefficients():
    experiment = _experiment("ZZZZ")
    coefficients = np.asarray(experiment.coefficients)

    assert np.allclose(np.abs(coefficients), 0.25)
    assert _errors(experiment) == pytest.approx(np.sqrt(48 * 0.25**2 / SHOTS))


def test_more_shots_lower_the_bound():
    experiment = _experiment("ZZZZ")
    assert _errors(experiment, 4 * SHOTS) == pytest.approx(_errors(experiment) / 2)


def test_a_sampled_experiment_adds_the_draws():
    experiment = _experiment(
        "ZZZZ", CutOptions(expansion="sample", num_samples=10, seed=3)
    )
    coefficients = np.asarray(experiment.coefficients)
    expected = np.sqrt(
        np.sum(coefficients**2) / SHOTS + np.sum(np.abs(coefficients)) ** 2 / 10
    )

    assert experiment.sampled
    assert _errors(experiment) == pytest.approx(expected)


def test_a_weighted_observable_scales_by_its_coefficients():
    per_term = _errors(_experiment("ZZZZ"))
    weighted = SparsePauliOp(["ZZZZ", "IIIZ"], [2.0, -0.5])

    assert _errors(_experiment(weighted)) == pytest.approx(2.5 * per_term)


def test_the_errors_are_shaped_like_the_values():
    errors = _errors(_experiment([["IIIZ", "ZIII"], ["ZZZZ", "IIZI"]]))
    assert errors.shape == (2, 2)
    assert _errors(_experiment("ZZZZ")).shape == ()


@pytest.mark.sim
def test_a_distribution_is_bounded_like_its_z_strings():
    """Each probability is a weighted sum of the Z strings, read from the same shots."""
    experiment = _experiment(qubits=[0, 1, 2, 3])
    results = ck.submit_experiments(
        experiment, backend=AerSimulator(seed_simulator=1), shots=SHOTS
    ).result()
    distribution = ck.estimate_probabilities(results)
    z_strings = ck.estimate_expectation_values(results)

    # p(x) = 2**-k (1 + sum_S (-1)**(x.S) <Z_S>), so the bound carries over.
    labels = [next(iter(term)) for term in experiment.observables.tolist()]
    for bitstring in distribution:
        signs = [
            (-1) ** sum(int(b) for b, z in zip(bitstring, label) if z == "Z")
            for label in labels
        ]
        rebuilt = (1 + np.dot(signs, z_strings)) / 2**4
        assert distribution[bitstring] == pytest.approx(rebuilt)

    per_z_string = _errors(experiment)[0]
    assert distribution.standard_error == pytest.approx((1 - 2**-4) * per_z_string)
    marginal = distribution.marginal([0])
    assert marginal.standard_error == pytest.approx(per_z_string / 2)


def test_a_distribution_built_by_hand_has_no_standard_error():
    assert ck.QuasiProbabilities({"0": 0.5, "1": 0.5}).standard_error is None


def test_results_without_an_experiment_are_refused():
    with pytest.raises(ValueError, match="carry no experiment"):
        ck.estimate_standard_errors(RawResult([], SHOTS))
