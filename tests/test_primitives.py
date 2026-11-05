"""Tests for the cut primitives: CutEstimator and CutQuasiSampler."""

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.result import QuasiDistribution
from qiskit_aer import AerSimulator

from QCut import CutEstimator, CutOptions, CutQuasiSampler, cut
from QCut.errors.qcuterror import QCutError
from QCut.post_process.probabilities import QuasiProbabilities
from QCut.primitives import CutRunEstimate

#: Shot noise on the reconstructed values, in line with the rest of the suite.
TOLERANCE = 0.1
SHOTS = 2**13


def _gates(circuit):
    for qubit in range(circuit.num_qubits):
        circuit.ry(0.3 + 0.2 * qubit, qubit)
    circuit.cx(0, 1)


def _cut(width=4):
    """A chain with one wire cut in the middle."""
    circuit = QuantumCircuit(width)
    _gates(circuit)
    circuit.append(cut(), [1])
    for qubit in range(1, width - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def _uncut(width=4):
    circuit = QuantumCircuit(width)
    _gates(circuit)
    for qubit in range(1, width - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def _exact(circuit, observable):
    return float(
        np.real(Statevector(circuit).expectation_value(SparsePauliOp(observable)))
    )


class _NeverDone:
    """A job that stays queued until it is cancelled."""

    def __init__(self, number):
        self.number = number
        self.cancelled = False

    def job_id(self):
        return f"never-{self.number}"

    def status(self):
        return JobStatus.CANCELLED if self.cancelled else JobStatus.QUEUED

    def cancel(self):
        self.cancelled = True


class _NeverDoneBackend:
    """Accepts every batch and never finishes any, so nothing is simulated."""

    def __init__(self):
        self.jobs = []

    def run(self, circuits, shots=None, **options):
        job = _NeverDone(len(self.jobs))
        self.jobs.append(job)
        return job


# -- estimator --------------------------------------------------------------------- #


@pytest.mark.sim
def test_estimator_reconstructs_each_circuit():
    estimator = CutEstimator(AerSimulator(seed_simulator=1))
    job = estimator.run([_cut(4), _cut(3)], [["ZZZZ", "IIXZ"], "ZZI"], shots=SHOTS)
    values = job.result().expectation_values()

    assert len(values) == 2
    assert values[0].shape == (2,)
    assert values[1].shape == ()
    expected = [_exact(_uncut(4), "ZZZZ"), _exact(_uncut(4), "IIXZ")]
    assert np.allclose(values[0], expected, atol=TOLERANCE)
    assert abs(values[1] - _exact(_uncut(3), "ZZI")) < TOLERANCE


@pytest.mark.sim
def test_estimator_entries_carry_their_raw_results():
    job = CutEstimator(AerSimulator(seed_simulator=2)).run(_cut(), "ZZZZ", shots=1024)
    result = job.result()

    assert len(result) == 1
    assert result[0].raw.experiment is job.experiments[0]
    assert np.array_equal(job.expectation_values(0), result[0].expectation_values)
    assert job.result() is result


@pytest.mark.sim
def test_estimator_runs_communicating_wire_cuts():
    """The waves run on each circuit's own job thread, behind the primitive's job."""
    options = CutOptions(wire_cut_communication="always")
    job = CutEstimator(AerSimulator(seed_simulator=3), options=options).run(
        [_cut(), _cut()], "ZZZZ", shots=SHOTS
    )

    assert all(experiment.communicates for experiment in job.experiments)
    for value in job.result().expectation_values():
        assert abs(value - _exact(_uncut(), "ZZZZ")) < TOLERANCE


def test_one_circuit_takes_its_observables_as_given():
    job = CutEstimator(_NeverDoneBackend()).run(_cut(), ["ZZZZ", "IIIZ"])
    assert job.experiments[0].observables.shape == (2,)


def test_a_single_observable_is_used_for_every_circuit():
    job = CutEstimator(_NeverDoneBackend()).run([_cut(), _cut()], "ZZZZ")
    assert [e.observables.shape for e in job.experiments] == [(), ()]


def test_a_list_of_observables_needs_one_entry_per_circuit():
    with pytest.raises(ValueError, match="one entry per circuit"):
        CutEstimator(_NeverDoneBackend()).run([_cut()], ["ZZZZ", "IIIZ"])


# -- sampler ----------------------------------------------------------------------- #


@pytest.mark.sim
def test_sampler_reconstructs_the_measured_distribution():
    circuit = _cut(3)
    circuit.measure_all()
    result = (
        CutQuasiSampler(AerSimulator(seed_simulator=4))
        .run(circuit, shots=SHOTS)
        .result()
    )

    expected = Statevector(_uncut(3)).probabilities_dict()
    distribution = result[0].quasi_probabilities
    assert isinstance(distribution, QuasiProbabilities)
    for bitstring, probability in expected.items():
        assert abs(distribution[bitstring] - probability) < TOLERANCE


@pytest.mark.sim
def test_sampler_reads_bits_in_classical_bit_order():
    """Classical bit zero is rightmost, as in qiskit, whatever qubit fills it."""
    circuit = _cut(3)
    circuit.add_register(*QuantumCircuit(0, 2).cregs)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    result = (
        CutQuasiSampler(AerSimulator(seed_simulator=5))
        .run(circuit, shots=SHOTS)
        .result()
    )

    expected = Statevector(_uncut(3)).probabilities_dict(qargs=[2, 0])
    distribution = result[0].quasi_probabilities
    assert distribution.width == 2
    for bitstring, probability in expected.items():
        assert abs(distribution[bitstring] - probability) < TOLERANCE


@pytest.mark.sim
def test_sampler_gives_a_qiskit_quasi_distribution():
    circuit = _cut(3)
    circuit.measure_all()
    job = CutQuasiSampler(AerSimulator(seed_simulator=6)).run(
        [circuit, circuit], shots=1024
    )
    result = job.result()

    dists = result.quasi_dists()
    assert len(dists) == 2
    assert all(isinstance(dist, QuasiDistribution) for dist in dists)
    assert len(dists[0]) == 2**3
    assert len(result[0].quasi_dist(top=2)) == 2
    assert job.quasi_probabilities(1) is result[1].quasi_probabilities


def test_sampler_needs_final_measurements():
    with pytest.raises(QCutError, match="measures nothing"):
        CutQuasiSampler(_NeverDoneBackend()).run(_cut())


def test_sampler_refuses_unmeasured_classical_bits():
    circuit = _cut(3)
    circuit.add_register(*QuantumCircuit(0, 2).cregs)
    circuit.measure(0, 0)
    with pytest.raises(QCutError, match="never measures into classical bit"):
        CutQuasiSampler(_NeverDoneBackend()).run(circuit)


# -- inputs both primitives refuse ------------------------------------------------- #


@pytest.mark.parametrize("primitive", ["estimator", "sampler"])
def test_circuits_need_cut_markers(primitive):
    circuit = _uncut()
    circuit.measure_all()
    backend = _NeverDoneBackend()
    with pytest.raises(QCutError, match="no cut markers"):
        if primitive == "estimator":
            CutEstimator(backend).run(circuit, "ZZZZ")
        else:
            CutQuasiSampler(backend).run(circuit)
    assert backend.jobs == []


def test_mid_circuit_measurements_are_refused():
    circuit = QuantumCircuit(4, 1)
    _gates(circuit)
    circuit.measure(0, 0)
    circuit.h(0)  # the measured qubit is used again, so the measurement is not final
    circuit.append(cut(), [1])
    circuit.cx(1, 2)
    with pytest.raises(QCutError, match="before its end"):
        CutEstimator(_NeverDoneBackend()).run(circuit, "ZZZZ")


def test_a_bad_circuit_submits_nothing():
    """Every circuit is checked and built before the first job is queued."""
    backend = _NeverDoneBackend()
    with pytest.raises(QCutError, match="circuit 1 has no cut markers"):
        CutEstimator(backend).run([_cut(), _uncut()], "ZZZZ")
    assert backend.jobs == []


def test_a_backend_is_needed():
    with pytest.raises(QCutError, match="backend"):
        CutEstimator(None)


# -- the job ----------------------------------------------------------------------- #


def test_job_answers_for_every_circuit():
    backend = _NeverDoneBackend()
    job = CutEstimator(backend).run([_cut(), _cut(3)], ["ZZZZ", "ZZI"])

    assert len(job.jobs) == 2
    assert job.job_ids() == [i for one in job.jobs for i in one.job_ids()]
    assert job.job_ids() == [f"never-{n}" for n in range(len(backend.jobs))]
    assert job.status() == JobStatus.QUEUED
    assert [len(batches) for batches in job.partial_results()] == [
        len(one.statuses()) for one in job.jobs
    ]

    job.cancel()
    assert all(one.cancelled for one in backend.jobs)
    assert job.cancelled()
    assert job.in_final_state()


def test_primitives_do_not_warn_about_qcut_3():
    """They call the warned-about functions from inside QCut, which stays quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        estimator = CutEstimator(_NeverDoneBackend())
        estimator.estimate(_cut(), "ZZZZ")
        estimator.run(_cut(), "ZZZZ")
        circuit = _cut()
        circuit.measure_all()
        CutQuasiSampler(_NeverDoneBackend()).run(circuit)


# -- cost ------------------------------------------------------------------------- #


def test_estimate_matches_what_the_run_submits():
    backend = _NeverDoneBackend()
    estimator = CutEstimator(backend)
    circuits, observables = [_cut(), _cut(3)], ["ZZZZ", "ZZI"]

    estimate = estimator.estimate(circuits, observables, shots=512, max_batch_size=4)
    job = estimator.run(circuits, observables, shots=512, max_batch_size=4)

    assert isinstance(estimate, CutRunEstimate)
    assert estimate.exact
    assert len(estimate) == 2
    assert estimate.jobs == len(backend.jobs)
    assert estimate.circuits == job.num_circuits
    assert estimate.shots == job.total_shots
    for index, one in enumerate(job.jobs):
        assert estimate[index].circuits == one.num_circuits
        assert estimate[index].shots == one.total_shots


def test_estimate_is_not_exact_for_communicating_cuts():
    estimator = CutEstimator(
        _NeverDoneBackend(), options=CutOptions(wire_cut_communication="always")
    )
    assert not estimator.estimate(_cut(), "ZZZZ").exact


def test_options_given_to_run_win_over_the_primitives():
    estimator = CutEstimator(_NeverDoneBackend(), options=CutOptions(deduplicate=True))
    deduplicated = estimator.estimate(_cut(), "ZZZZ")
    every_slot = estimator.estimate(
        _cut(), "ZZZZ", options=CutOptions(deduplicate=False)
    )
    job = estimator.run(_cut(), "ZZZZ", options=CutOptions(deduplicate=False))

    assert every_slot.circuits > deduplicated.circuits
    assert every_slot.circuits == job.experiments[0].num_circuits
