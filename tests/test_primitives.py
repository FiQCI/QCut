"""Tests for the cut primitives: CutEstimator and CutQuasiSampler."""

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import JobStatus
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.result import QuasiDistribution
from qiskit_aer import AerSimulator

from QCut import (
    CutEstimator,
    CutOptions,
    CutQuasiSampler,
    ParallelBackend,
    cut,
    estimate_standard_errors,
    find_cuts,
    get_locations_and_subcircuits,
    transpile_subcircuits,
)
from QCut.errors.qcuterror import QCutError
from QCut.post_process.probabilities import QuasiProbabilities
from QCut.pre_process.circuit_preparation import _final_measurements
from QCut.primitives import CutRunEstimate

#: Shot noise on the reconstructed values, in line with the rest of the suite.
TOLERANCE = 0.1
SHOTS = 2**13


def _gates(circuit):
    for qubit in range(circuit.num_qubits):
        circuit.ry(0.3 + 0.2 * qubit, qubit)
    circuit.cx(0, 1)


def _marked(width=4):
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


def _cut(width=4, options=None):
    return get_locations_and_subcircuits(_marked(width), options=options)


def _measured_cut(width=3, options=None):
    circuit = _marked(width)
    circuit.measure_all()
    return get_locations_and_subcircuits(circuit, options=options)


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
    assert job.standard_errors(0) == estimate_standard_errors(result[0].raw)
    assert result.standard_errors() == [result[0].standard_errors]
    assert job.result() is result


@pytest.mark.sim
def test_estimator_runs_communicating_wire_cuts():
    """The waves run on each circuit's own job thread, behind the primitive's job."""
    options = CutOptions(wire_cut_communication="always")
    job = CutEstimator(AerSimulator(seed_simulator=3)).run(
        [_cut(options=options), _cut(options=options)], "ZZZZ", shots=SHOTS
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


def test_a_cut_circuit_runs_with_its_own_options():
    estimator = CutEstimator(_NeverDoneBackend())
    deduplicated = estimator.estimate(
        _cut(options=CutOptions(deduplicate=True)), "ZZZZ"
    )
    every_slot = estimator.estimate(_cut(options=CutOptions(deduplicate=False)), "ZZZZ")

    assert every_slot.circuits > deduplicated.circuits


# -- sampler ----------------------------------------------------------------------- #


@pytest.mark.sim
def test_sampler_reconstructs_the_measured_distribution():
    sampler = CutQuasiSampler(AerSimulator(seed_simulator=4))
    result = sampler.run(_measured_cut(3), shots=SHOTS).result()

    expected = Statevector(_uncut(3)).probabilities_dict()
    distribution = result[0].quasi_probabilities
    assert isinstance(distribution, QuasiProbabilities)
    for bitstring, probability in expected.items():
        assert abs(distribution[bitstring] - probability) < TOLERANCE


@pytest.mark.sim
def test_sampler_reads_bits_in_classical_bit_order():
    """Classical bit zero is rightmost, as in qiskit, whatever qubit fills it."""
    circuit = _marked(3)
    circuit.add_register(*QuantumCircuit(0, 2).cregs)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    sampler = CutQuasiSampler(AerSimulator(seed_simulator=5))
    result = sampler.run(get_locations_and_subcircuits(circuit), shots=SHOTS).result()

    expected = Statevector(_uncut(3)).probabilities_dict(qargs=[2, 0])
    distribution = result[0].quasi_probabilities
    assert distribution.width == 2
    for bitstring, probability in expected.items():
        assert abs(distribution[bitstring] - probability) < TOLERANCE


@pytest.mark.sim
def test_sampler_runs_what_find_cuts_found():
    """find_cuts strips the measurements before cutting, and still records them."""
    circuit = _uncut(4)
    circuit.measure_all()
    found = find_cuts(circuit, options=CutOptions(finder_num_partitions=2))
    assert found.final_measurements == ((0, 0), (1, 1), (2, 2), (3, 3))

    sampler = CutQuasiSampler(AerSimulator(seed_simulator=7))
    distribution = sampler.run(found, shots=SHOTS).result()[0].quasi_probabilities
    for bitstring, probability in Statevector(_uncut(4)).probabilities_dict().items():
        assert abs(distribution[bitstring] - probability) < TOLERANCE


@pytest.mark.sim
def test_sampler_gives_a_qiskit_quasi_distribution():
    job = CutQuasiSampler(AerSimulator(seed_simulator=6)).run(
        [_measured_cut(3), _measured_cut(3)], shots=1024
    )
    result = job.result()

    dists = result.quasi_dists()
    assert len(dists) == 2
    assert all(isinstance(dist, QuasiDistribution) for dist in dists)
    assert len(dists[0]) == 2**3
    assert len(result[0].quasi_dist(top=2)) == 2
    assert job.quasi_probabilities(1) is result[1].quasi_probabilities


def test_sampler_needs_final_measurements():
    with pytest.raises(QCutError, match="no final measurements"):
        CutQuasiSampler(_NeverDoneBackend()).run(_cut())


def test_sampler_refuses_unmeasured_classical_bits():
    circuit = _marked(3)
    circuit.add_register(*QuantumCircuit(0, 2).cregs)
    circuit.measure(0, 0)
    with pytest.raises(QCutError, match="never measures into classical bit"):
        CutQuasiSampler(_NeverDoneBackend()).run(get_locations_and_subcircuits(circuit))


# -- what a cut circuit remembers of its measurements ------------------------------ #


def test_splitting_records_the_final_measurements():
    circuit = _marked(3)
    circuit.add_register(*QuantumCircuit(0, 2).cregs)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    cut_circuit = get_locations_and_subcircuits(circuit)

    assert cut_circuit.final_measurements == ((2, 0), (0, 1))
    assert cut_circuit.uncut_num_clbits == 2


def test_a_measurement_followed_by_a_gate_is_not_final():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(0)
    circuit.measure(0, 1)
    circuit.measure(1, 0)
    assert _final_measurements(circuit) == ((0, 1), (1, 0))


def test_transpiling_and_binding_keep_the_final_measurements():
    theta = Parameter("theta")
    circuit = _marked(3)
    circuit.rz(theta, 0)
    circuit.measure_all()
    cut_circuit = get_locations_and_subcircuits(circuit)

    bound = cut_circuit.assign_parameters({theta: 0.4})
    transpiled = transpile_subcircuits(cut_circuit, GenericBackendV2(5, seed=1))
    for kept in (bound, transpiled):
        assert kept.final_measurements == cut_circuit.final_measurements
        assert kept.uncut_num_clbits == 3


# -- inputs both primitives refuse ------------------------------------------------- #


@pytest.mark.parametrize("primitive", ["estimator", "sampler"])
def test_primitives_take_cut_circuits_only(primitive):
    circuit = _marked()
    circuit.measure_all()
    backend = _NeverDoneBackend()
    with pytest.raises(QCutError, match="take CutCircuits"):
        if primitive == "estimator":
            CutEstimator(backend).run(circuit, "ZZZZ")
        else:
            CutQuasiSampler(backend).run(circuit)
    assert backend.jobs == []


def test_a_backend_with_a_coupling_map_needs_transpiled_circuits():
    device = GenericBackendV2(5, seed=1)
    with pytest.raises(QCutError, match="not transpiled"):
        CutEstimator(device).estimate(_cut(), "ZZZZ")


def test_circuits_transpiled_for_another_backend_are_refused():
    device, other = GenericBackendV2(5, seed=1), GenericBackendV2(5, seed=2)
    with pytest.raises(QCutError, match="different backend"):
        CutEstimator(device).estimate(transpile_subcircuits(_cut(), other), "ZZZZ")


def test_circuits_transpiled_for_the_same_backend_are_taken():
    device = GenericBackendV2(5, seed=1)
    transpiled = transpile_subcircuits(_measured_cut(3), device)

    assert CutEstimator(device).estimate(transpiled, "ZZZ").circuits > 0
    assert CutQuasiSampler(device).estimate(transpiled).circuits > 0


def test_a_backend_without_a_coupling_map_takes_anything():
    """An ideal simulator or a sampler runs the subcircuits as they are."""
    transpiled = transpile_subcircuits(_cut(), GenericBackendV2(5, seed=1))
    estimator = CutEstimator(_NeverDoneBackend())

    assert estimator.estimate([_cut(), transpiled], "ZZZZ").circuits > 0


def test_a_bad_circuit_submits_nothing():
    """Every circuit is checked and built before the first job is queued."""
    backend = _NeverDoneBackend()
    with pytest.raises(QCutError, match="circuit 1 is a QuantumCircuit"):
        CutEstimator(backend).run([_cut(), _marked()], "ZZZZ")
    assert backend.jobs == []


def test_a_backend_is_needed():
    with pytest.raises(QCutError, match="backend"):
        CutEstimator(None)


@pytest.mark.sim
def test_a_parallel_backend_shares_the_circuits_evenly():
    """Each input is submitted in batches of its own, so they all come in the same
    sizes, here 8 then 2. Taking turns used to put both 8s on one backend, [16, 4]."""
    backend = ParallelBackend([AerSimulator(), AerSimulator()])
    job = CutEstimator(backend).run([_cut(), _cut()], "ZZZZ", max_batch_size=8)

    assert len(job.result()) == 2
    assert backend.submitted == [10, 10]


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
        CutQuasiSampler(_NeverDoneBackend()).run(_measured_cut())


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
    options = CutOptions(wire_cut_communication="always")
    estimate = CutEstimator(_NeverDoneBackend()).estimate(_cut(options=options), "ZZZZ")
    assert not estimate.exact
