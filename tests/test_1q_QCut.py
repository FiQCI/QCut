"""Tests for CircuitKnitting package."""  # noqa: N999

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import QCut as ck
import tests.solutions_1q as sq
from QCut.execution.circuit_knitting import run
from QCut.job.cutresult import SubResult, TotalResult
from QCut.post_process.postprocess import _process_results

# import QCut as ck
from QCut.pre_process.circuit_preparation import (
    _get_cut_locations,
    get_locations_and_subcircuits,
)


def _remove_obsm(subcircuits: list[QuantumCircuit]):

    for circ in subcircuits:
        j = 0
        while j < len(circ.data):
            if "obs" in circ[j].operation.name:
                circ.data.remove(circ[j])
            else:
                j += 1


def test_get_cut_locations() -> None:
    """Test get_cut_locations function.

    This function tests whether the get_cut_location method correctly identifies
    the cut locations in the provided test circuits by comparing the result to
    the pre-defined solutions.
    """
    for solution_index, circ in enumerate(sq.test_circuits):
        print(solution_index)
        assert np.array_equal(
            _get_cut_locations(circ.copy()),
            sq.cut_location_solutions[solution_index],
        )


def test_separate_subcircuits() -> None:
    """Test separate_subcircuits function.

    This function tests whether the get_locations_and_subcircuits method correctly
    identifies the locations and separates the subcircuits for each test circuit
    by comparing the operations in the generated subcircuits to the pre-defined
    solutions.
    """
    count = 0
    for solution_index, circ in enumerate(sq.test_circuits):
        cut_circuit = get_locations_and_subcircuits(circ.copy())
        circs = cut_circuit.subcircuits
        _remove_obsm(circs)
        count += 1

        assert len(circs) == sq.number_of_subcircuits[solution_index]

        for circ_index, subcirc in enumerate(circs):
            assert len(subcirc.data) == sq.subcircuit_len[solution_index][circ_index]


@pytest.mark.sim
def test_results_exist() -> None:
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.append(ck.cutCZ(), [0, 1])

    cut_circuit = get_locations_and_subcircuits(circ)

    obs = ["ZZ"]

    cut_experiment = ck.get_experiment_circuits(cut_circuit, obs)

    res = ck.run_experiments(cut_experiment, backend=AerSimulator())

    result = _process_results(res.result(), res._shots)

    assert isinstance(result[0][0], TotalResult)

    assert str(result[0][0])

    assert isinstance(result[0][0].subcircuits[0][0][0], SubResult)

    assert str(result[0][0].subcircuits[0][0][0])


@pytest.mark.sim
def test_expectation_values() -> None:
    """Test the expectation values of the test circuits.

    This function tests whether the run method correctly calculates the expectation
    values for each test circuit and its corresponding observable by comparing the
    results to the pre-defined solutions within a specified error tolerance.

    The test runs each circuit on the AerSimulator backend without error mitigation.
    """
    # Initialize the simulator
    sim = AerSimulator()

    # Iterate over each test circuit and its corresponding expected solutions
    for solution_index, circ in enumerate(sq.test_circuits):
        print(solution_index)
        # Calculate expectation values using the run method
        expvals = run(circ, sq.test_observables[solution_index], backend=sim)
        # Check each calculated expectation value against the corresponding
        # expected value
        tolerance = 0.1
        print("CIrcuit index: ", solution_index)
        print(expvals)
        print(sq.exp_val_solutions[solution_index])
        for check in [
            abs(a - b) <= tolerance
            for a, b in zip(expvals, sq.exp_val_solutions[solution_index])
        ]:
            assert check  # noqa: S101
