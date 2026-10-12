"""Reading counts from IQM's counts endpoint when a job's raw measurements are withheld.

VTT Q50 hands back only a histogram for large jobs, and ``IQMJob.result()`` fails on
them. The run falls back to ``client.get_job_measurement_counts`` and has to key what
comes back exactly as ``job.result()`` would have.
"""

from collections import Counter
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import ParallelBackend
from QCut.errors.qcuterror import QCutError
from QCut.execution import circuit_knitting
from QCut.execution.iqm_counts import qiskit_counts, withholds_measurements
from tests.test_parallel import _communicating, _experiment, _gate_cut

SEED = 11


def _withheld() -> ValidationError:
    """The error ``IQMJob.result()`` raises when the server offers only counts."""
    try:
        TypeAdapter(list[dict[str, list[list[int]]]]).validate_json(
            '{"detail": ["Raw measurements are not available for this job, '
            'use measurement_counts instead."]}'
        )
    except ValidationError as error:
        return error
    raise AssertionError("that json should not have validated")


def test_the_refusal_is_recognised_though_pydantic_shortens_it():
    error = _withheld()

    assert "measurement_counts" not in str(error), "pydantic no longer truncates"
    assert withholds_measurements(error)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("No results were available for job abc."),
        ConnectionError("connection reset by peer"),
    ],
)
def test_other_failures_are_not_mistaken_for_it(error):
    assert not withholds_measurements(error)


def test_registers_and_bits_are_laid_out_as_qiskit_does():
    """Last-added register first, highest bit first, whatever order IQM lists keys."""
    keys = ["c_2_0_1", "qpd_meas_1_1_0", "c_2_0_0"]

    assert qiskit_counts(keys, {"101": 5, "010": 3}) == {"0 11": 5, "1 00": 3}


def test_an_unmeasured_bit_reads_zero():
    assert qiskit_counts(["c_3_0_2"], {"1": 4, "0": 2}) == {"100": 4, "000": 2}


def test_a_bitstring_of_the_wrong_length_is_refused():
    with pytest.raises(QCutError, match="does not match"):
        qiskit_counts(["c_2_0_0", "c_2_0_1"], {"1": 4})


def test_the_keys_match_qiskit_iqms_own_formatting():
    """The counts path must agree with what ``IQMJob`` builds from raw measurements."""
    iqm = pytest.importorskip("iqm.qiskit_iqm.iqm_job")
    rng = np.random.default_rng(SEED)
    keys = ["meas_3_1_0", "qpd_meas_2_0_1", "meas_3_1_2", "qpd_meas_2_0_0"]
    shots = rng.integers(0, 2, size=(200, len(keys)))

    raw = {key: [[int(bit)] for bit in shots[:, i]] for i, key in enumerate(keys)}
    expected = Counter(iqm.IQMJob._iqm_format_measurement_results(raw, len(shots)))
    histogram = Counter("".join(map(str, shot)) for shot in shots)

    assert qiskit_counts(keys, dict(histogram)) == dict(expected)


def _as_iqm(circuit, counts: dict, rng) -> SimpleNamespace:
    """Qiskit counts as IQM's endpoint hands them back, keys in a shuffled order."""
    keys = {}
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        clbit = instruction.clbits[0]
        register, index = circuit.find_bit(clbit).registers[0]
        creg_idx = circuit.cregs.index(register)
        keys[f"{register.name}_{register.size}_{creg_idx}_{index}"] = (
            creg_idx,
            register.size,
            index,
        )
    order = [list(keys)[i] for i in rng.permutation(len(keys))]

    histogram = Counter()
    for key, count in counts.items():
        fields = key.split(" ")
        bits = []
        for name in order:
            creg_idx, size, index = keys[name]
            bits.append(fields[len(circuit.cregs) - 1 - creg_idx][size - 1 - index])
        histogram["".join(bits)] += count
    return SimpleNamespace(measurement_keys=order, counts=dict(histogram))


class _Client:
    """``IQMClient.get_job_measurement_counts``, and a record of who was asked."""

    def __init__(self):
        self.jobs = {}
        self.asked = []

    def get_job_measurement_counts(self, job_id: UUID):
        self.asked.append(job_id)
        return self.jobs[job_id]


class _CountsOnlyJob:
    def __init__(self, backend, job_id: str):
        self._backend, self._job_id = backend, job_id

    def backend(self):
        return self._backend

    def job_id(self):
        return self._job_id

    def result(self):
        raise _withheld()


class _CountsOnlyBackend(AerSimulator):
    """Aer, behaving as Q50 does for a large job: counts only, via the client."""

    def __init__(self, **options):
        super().__init__(seed_simulator=SEED, **options)
        self.client = _Client()
        self._rng = np.random.default_rng(SEED)

    def run(self, circuits, shots=1024, **options):
        circuits = list(circuits)
        result = super().run(circuits, shots=shots, **options).result()
        job_id = uuid4()
        self.client.jobs[job_id] = [
            _as_iqm(circuit, result.get_counts(i), self._rng)
            for i, circuit in enumerate(circuits)
        ]
        return _CountsOnlyJob(self, str(job_id))


def _estimate(experiment, backend):
    return np.array(
        ck.estimate_expectation_values(
            ck.run_experiments(
                experiment, shots=2000, backend=backend, max_batch_size=4
            )
        )
    )


@pytest.mark.sim
@pytest.mark.parametrize("case", [_gate_cut, _communicating])
def test_a_counts_only_run_gives_the_same_estimate(case):
    """Seeded alike, the counts path must reproduce the ordinary one exactly."""
    marked, _plain, observables = case()
    experiment = _experiment(marked, observables)
    backend = _CountsOnlyBackend()

    via_counts = _estimate(experiment, backend)
    via_result = _estimate(experiment, AerSimulator(seed_simulator=SEED))

    assert backend.client.asked, "the counts endpoint was never asked"
    assert np.array_equal(via_counts, via_result)


@pytest.mark.sim
def test_each_job_is_read_from_the_machine_that_ran_it():
    marked, _plain, observables = _gate_cut()
    experiment = _experiment(marked, observables)
    first, second = _CountsOnlyBackend(), _CountsOnlyBackend()

    _estimate(experiment, ParallelBackend([first, second]))

    for backend in (first, second):
        assert backend.client.asked
        assert set(backend.client.asked) <= set(backend.client.jobs)


def test_a_job_with_no_client_says_so():
    job = SimpleNamespace(result=lambda: (_ for _ in ()).throw(_withheld()))

    with pytest.raises(QCutError, match="no IQM client"):
        circuit_knitting._result(job)
