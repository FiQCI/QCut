"""Tests for submitting an experiment and collecting its results later."""

import threading

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus
from qiskit.providers.exceptions import JobTimeoutError
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut
from QCut.errors.qcuterror import QCutError

OBSERVABLES = ["IIIZ", "ZIII"]


def _circuit():
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.3 + 0.1 * qubit, qubit)
    circuit.cx(0, 1)
    circuit.append(cut(), [1])
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    return circuit


def _experiment(communication="never"):
    cut_circuit = ck.get_locations_and_subcircuits(
        _circuit(), options=CutOptions(wire_cut_communication=communication)
    )
    return ck.get_experiment_circuits(cut_circuit, OBSERVABLES)


class _GatedJob:
    def __init__(self, inner, gate, number):
        self.inner = inner
        self.gate = gate
        self.number = number
        self.cancelled = False

    def job_id(self):
        return f"gated-{self.number}"

    def status(self):
        return self.inner.status() if self.gate.opened.is_set() else JobStatus.QUEUED

    def result(self):
        assert self.gate.opened.wait(60), "the gate was never opened"
        return self.inner.result()

    def cancel(self):
        self.cancelled = True
        return True


class _Gate:
    """Runs on Aer, but holds every result back until the gate is opened.

    ``refuse`` names the ``run`` call, counting from one, that raises instead.
    """

    def __init__(self, refuse=None):
        self.inner = AerSimulator(seed_simulator=1)
        self.opened = threading.Event()
        self.refuse = refuse
        self.jobs = []

    def run(self, circuits, shots=1024, **options):
        if len(self.jobs) + 1 == self.refuse:
            self.refuse = None
            raise RuntimeError("refused by the target")
        job = _GatedJob(
            self.inner.run(circuits, shots=shots, **options), self, len(self.jobs) + 1
        )
        self.jobs.append(job)
        return job


def _waves(job):
    return {batch.wave for batch in job.partial_results()}


@pytest.mark.sim
def test_submitting_returns_before_the_results_are_in():
    gate = _Gate()
    job = ck.submit_experiments(
        _experiment(), shots=128, backend=gate, max_batch_size=4
    )

    assert isinstance(job, ck.QCutJob)
    assert job.job_ids() == [f"gated-{n}" for n in range(1, len(gate.jobs) + 1)]
    assert job.job_id() == "gated-1"
    assert job.status() == JobStatus.QUEUED
    assert not job.in_final_state()

    gate.opened.set()
    raw = job.result()
    assert isinstance(raw, ck.RawResult)
    assert raw.experiment is job.experiment
    assert job.result() is raw
    assert job.done()


@pytest.mark.sim
@pytest.mark.parametrize("communication", ["never", "always"])
def test_run_experiments_is_submit_then_result(communication):
    """The blocking call is a wrapper, so both give the same counts on a seeded run."""
    experiment = _experiment(communication)
    blocking = ck.run_experiments(
        experiment, shots=256, backend=AerSimulator(seed_simulator=7)
    )
    job = ck.submit_experiments(
        experiment, shots=256, backend=AerSimulator(seed_simulator=7)
    )

    assert np.allclose(
        ck.estimate_expectation_values(blocking),
        ck.estimate_expectation_values(job.result()),
    )


@pytest.mark.sim
def test_what_was_submitted_matches_the_estimate():
    experiment = _experiment()
    estimate = ck.estimate_run(experiment, shots=256, max_batch_size=5)
    job = ck.submit_experiments(
        experiment, shots=256, backend=AerSimulator(), max_batch_size=5
    )

    assert job.num_circuits == estimate.circuits
    assert job.total_shots == estimate.shots
    assert len(job.job_ids()) == estimate.jobs

    batches = job.partial_results()
    assert [batch.index for batch in batches] == list(range(estimate.jobs))
    assert sum(batch.circuits for batch in batches) == estimate.circuits
    assert all(batch.wave == 0 and batch.shots == 256 for batch in batches)
    job.result()
    assert all(batch.result is not None for batch in job.partial_results())


@pytest.mark.sim
def test_later_waves_go_out_without_asking_for_the_result():
    """A thread of the job's own submits each wave once the one before it is in."""
    gate = _Gate()
    job = ck.submit_experiments(_experiment("always"), shots=256, backend=gate)

    assert job.num_waves == 2
    assert job.wave == 0
    assert _waves(job) == {0}
    assert not job.in_final_state()
    assert "wave 1 of 2" in repr(job)

    gate.opened.set()
    # Only reads the status, which drives nothing: the thread does the submitting.
    job.wait_for_final_state(timeout=60, wait=0.05)

    assert job.done()
    assert job.wave == 1
    assert _waves(job) == {0, 1}
    assert len(job.job_ids()) == len(gate.jobs)
    values = ck.estimate_expectation_values(job.result())
    assert values.shape == (len(OBSERVABLES),)


@pytest.mark.sim
def test_a_refused_batch_keeps_the_ones_before_it():
    gate = _Gate(refuse=2)
    job = ck.submit_experiments(
        _experiment(), shots=128, backend=gate, max_batch_size=2
    )

    statuses = job.statuses()
    assert statuses[:2] == [JobStatus.QUEUED, JobStatus.ERROR]
    assert set(statuses[2:]) == {JobStatus.CANCELLED}
    assert job.status() == JobStatus.ERROR
    assert job.job_ids()[0] == "gated-1"
    assert set(job.job_ids()[1:]) == {None}
    assert job.num_circuits == 2

    # Raised at once, without waiting for the batch that did go out.
    with pytest.raises(RuntimeError, match="refused by the target"):
        job.result()


@pytest.mark.sim
def test_a_refusal_in_a_later_wave_is_raised_from_the_result():
    gate = _Gate()
    gate.opened.set()
    experiment = _experiment("always")
    probe = ck.submit_experiments(experiment, shots=256, backend=gate)
    first = sum(1 for batch in probe.partial_results() if batch.wave == 0)
    probe.result()

    gate = _Gate(refuse=first + 1)
    job = ck.submit_experiments(experiment, shots=256, backend=gate)
    gate.opened.set()

    with pytest.raises(RuntimeError, match="refused by the target"):
        job.result(timeout=60)
    assert job.status() == JobStatus.ERROR


@pytest.mark.sim
@pytest.mark.parametrize("communication", ["never", "always"])
def test_waiting_too_long_times_out(communication):
    gate = _Gate()
    job = ck.submit_experiments(_experiment(communication), shots=128, backend=gate)

    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.1)

    gate.opened.set()
    assert isinstance(job.result(timeout=60), ck.RawResult)


@pytest.mark.sim
def test_cancelling_stops_the_waves():
    gate = _Gate()
    job = ck.submit_experiments(_experiment("always"), shots=256, backend=gate)

    job.cancel()
    assert all(submitted.cancelled for submitted in gate.jobs)
    gate.opened.set()

    with pytest.raises(QCutError, match="cancelled"):
        job.result(timeout=60)
    assert job.status() == JobStatus.CANCELLED
    assert _waves(job) == {0}


class _Splitting:
    """A target that splits each batch into two jobs of its own, as fiqci-ems can."""

    def __init__(self):
        self.inner = AerSimulator()
        self.calls = 0

    def run(self, circuits, shots=1024, **options):
        self.calls += 1
        job = self.inner.run(circuits, shots=shots, **options)
        job.job_ids = lambda n=self.calls: [f"{n}a", f"{n}b"]
        return job


@pytest.mark.sim
def test_a_target_batching_on_its_own_lists_every_job():
    job = ck.submit_experiments(
        _experiment(), shots=128, backend=_Splitting(), max_batch_size=4
    )
    batches = len(job.partial_results())
    assert job.job_ids() == [
        f"{n}{part}" for n in range(1, batches + 1) for part in "ab"
    ]


def test_a_backend_is_needed():
    with pytest.raises(QCutError, match="backend or sampler"):
        ck.submit_experiments(_experiment(), shots=128)
