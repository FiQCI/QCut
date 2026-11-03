import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate, Parameter
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

import QCut as ck
from QCut import cut, cutGate
from QCut.execution import job as job_module
from QCut.execution.qcutresult import CircuitResult
from QCut.options import CutOptions

cut_qc = QuantumCircuit(4)

mult = Parameter("mult")
cut_qc.r(mult * 0.46262, mult * 0.1446, 0)
cut_qc.append(**cutGate(CXGate(), 0, 1))
cut_qc.append(cut(), [1])
cut_qc.cx(1, 2)
cut_qc.cx(2, 3)


cut_circuit = ck.get_locations_and_subcircuits(cut_qc)
num_subcircuits = 3

num_qubits = [1, 1, 3]


def test_cut_circuit_properties():
    assert cut_circuit.num_qubits == num_qubits
    assert cut_circuit.num_subcircuits == num_subcircuits


def test_cut_circuit_assign_parameters():
    cut_circuit_param = cut_circuit.assign_parameters({"mult": 2}, inplace=False)

    for exp_ind, subcircuits in enumerate(cut_circuit_param.subcircuits):
        for circ_ind, value in enumerate(subcircuits):
            for par in value.params:
                if hasattr(par, "parameters"):
                    for elem in par.parameters:
                        assert "mult" not in elem.name
                else:
                    assert True


observables = ["IIIZ", "IIZI", "IZII", "IIZZ"]
cut_experiment = ck.get_experiment_circuits(cut_circuit, observables)

exp_num_qubits = [1, 1, 3]
exp_num_circuits = 144
exp_group_size = 3
exp_num_groups = 48
exp_num_obs_groups = 1


def test_cut_experiment_properties():
    assert cut_experiment.num_qubits == exp_num_qubits
    assert cut_experiment.num_circuits == exp_num_circuits
    assert cut_experiment.group_size == exp_group_size
    assert cut_experiment.num_groups == exp_num_groups
    assert cut_experiment.num_obs_groups == exp_num_obs_groups


def test_cut_experiment_assign_parameters():
    cut_experiment_param = cut_experiment.assign_parameters({"mult": 2}, inplace=False)

    for exp_ind, subcircuits in enumerate(cut_experiment_param.experiments):
        for obs_circ in subcircuits:
            for circ in obs_circ.values():
                for circ_ind, value in enumerate(circ):
                    for par in value.params:
                        if hasattr(par, "parameters"):
                            for elem in par.parameters:
                                assert "mult" not in elem.name
                        else:
                            assert True


def test_results_carry_their_own_experiment():
    """The estimator is called on the result alone, with nothing threaded through."""
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.append(**cutGate(CXGate(), 0, 1))
    observables = ["IZ", "ZI"]

    cut_circuit = ck.get_locations_and_subcircuits(circuit)
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    results = ck.run_experiments(
        experiment, backend=AerSimulator(seed_simulator=17), shots=2048
    )

    assert results.experiment is experiment
    assert ck.estimate_expectation_values(results).shape == (len(observables),)


def test_results_without_an_experiment_say_so():
    """A hand-built result carries nothing, so the estimator cannot interpret it."""
    from QCut.execution.qcutresult import RawResult

    bare = RawResult([], 1024)
    assert bare.experiment is None
    with pytest.raises(ValueError, match="no experiment"):
        ck.estimate_expectation_values(bare)


def _wire_and_gate_cut():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.ry(0.4 + 0.2 * qubit, qubit)
    marked.append(**cutGate(CXGate(), 0, 1))
    plain.cx(0, 1)
    marked.append(cut(), [1])
    for circuit in (marked, plain):
        circuit.cx(1, 2)
        circuit.cx(2, 3)
    return marked, plain, ["IIIZ", "IIZI", "IZII", "ZIII"]


def _communicating_pair():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.ry(0.4 + 0.2 * qubit, qubit)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
    marked.append(cut(), [1])
    marked.append(cut(), [2])
    for circuit in (marked, plain):
        circuit.cx(1, 2)
        circuit.cx(2, 3)
    return marked, plain, ["IIIZ", "IIZI", "IZII", "ZIII"]


@pytest.mark.sim
@pytest.mark.parametrize(
    ("name", "case", "options"),
    [
        ("plain", _wire_and_gate_cut, CutOptions()),
        (
            "communicating",
            _communicating_pair,
            CutOptions(wire_cut_communication="always"),
        ),
    ],
)
def test_a_sampler_can_run_the_experiment(name, case, options):
    """``backend`` also takes a V2 sampler, on both execution paths.

    The communicating path is the one worth pinning: it runs in waves and allocates each
    wave's shots from what the last one measured, so the sampler has to be read back
    mid-run and not only at the end.
    """
    marked, plain, observables = case()
    state = Statevector(plain)
    exact = np.array(
        [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]
    )

    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked, options=options), observables
    )
    results = ck.run_experiments(experiment, shots=8192, backend=SamplerV2())
    values = np.array(ck.estimate_expectation_values(results))

    assert np.abs(values - exact).max() < 0.1, f"{name}: {values} against {exact}"


def test_raw_results_hold_results_rather_than_counts():
    """``RawResult.result()`` keeps its shape, with a result per subcircuit.

    Holding what the backend returned is what lets the estimate be recomputed without
    re-running, and it is what a sampler's own result object slots into.
    """
    marked, _plain, observables = _wire_and_gate_cut()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), observables
    )
    raw = ck.run_experiments(
        experiment, backend=AerSimulator(seed_simulator=11), shots=1024
    )

    leaves = [leaf for group in raw.result() for obs in group for leaf in obs.values()]
    assert leaves and all(isinstance(leaf, CircuitResult) for leaf in leaves)
    assert all(isinstance(leaf.counts(), dict) for leaf in leaves)

    # Re-running the estimate on the same results has to give the same answer.
    first = ck.estimate_expectation_values(raw)
    assert np.allclose(first, ck.estimate_expectation_values(raw))


class _CountingBackend:
    """Records every submission, so an estimate can be checked against the real run."""

    def __init__(self, backend):
        self._backend = backend
        self.calls = []

    def run(self, circuits, shots=1024, **options):
        self.calls.append((len(circuits), shots))
        return self._backend.run(circuits, shots=shots, **options)


class _RecordingSimulator(AerSimulator):
    """A simulator that notes what each submission carried besides its circuits.

    A plain stub would do, but the shorthands transpile for anything that is not an
    ``AerSimulator``, and a stub has no target to transpile against.
    """

    def __init__(self):
        super().__init__()
        self.submitted = []

    def run(self, circuits, shots=1024, **options):
        self.submitted.append(options)
        return super().run(circuits, shots=shots, **options)


@pytest.mark.sim
@pytest.mark.parametrize("shorthand", ["run", "run_cut_circuit"])
def test_the_shorthands_pass_run_options_to_the_backend(shorthand):
    """Whatever the backend takes beyond shots is only reachable through these."""
    marked, _plain, observables = _wire_and_gate_cut()
    target = marked if shorthand == "run" else ck.get_locations_and_subcircuits(marked)
    backend = _RecordingSimulator()

    getattr(ck, shorthand)(
        target,
        observables,
        backend=backend,
        shots=1024,
        run_options={"seed_simulator": 7},
    )

    assert backend.submitted, "the backend was never called"
    assert all(options["seed_simulator"] == 7 for options in backend.submitted)


@pytest.mark.sim
@pytest.mark.parametrize("max_batch_size", [100, 40])
def test_a_plain_run_is_estimated_exactly(max_batch_size):
    """Nothing about a run without communication depends on what it measures."""
    marked, _plain, observables = _wire_and_gate_cut()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), observables
    )
    estimate = ck.estimate_run(experiment, shots=1024, max_batch_size=max_batch_size)

    backend = _CountingBackend(AerSimulator())
    ck.run_experiments(
        experiment, shots=1024, backend=backend, max_batch_size=max_batch_size
    )

    assert estimate.exact
    assert estimate.jobs == len(backend.calls)
    assert estimate.circuits == sum(circuits for circuits, _ in backend.calls)
    assert estimate.shots == sum(circuits * shots for circuits, shots in backend.calls)


@pytest.mark.sim
def test_a_communicating_run_is_bounded_not_exact():
    """Later waves spend their shots on what the wave before them measured.

    So the circuit count is the most that can be submitted and the job count the fewest,
    since circuits wanting very different shot counts cannot share a job.
    """
    marked, _plain, observables = _communicating_pair()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(
            marked, options=CutOptions(wire_cut_communication="always")
        ),
        observables,
    )
    estimate = ck.estimate_run(experiment, shots=1024)

    backend = _CountingBackend(AerSimulator())
    ck.run_experiments(experiment, shots=1024, backend=backend)

    assert not estimate.exact
    assert estimate.circuits >= sum(circuits for circuits, _ in backend.calls)
    assert estimate.jobs <= len(backend.calls)
    spent = sum(circuits * shots for circuits, shots in backend.calls)
    assert estimate.shots == pytest.approx(spent, rel=0.01)


class _NotReadyJob:
    """A job that refuses its results, the way one not quite ready does."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def result(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "counts"


def _not_ready():
    """The pydantic error IQM's client raises when the measurements are not there."""
    try:
        TypeAdapter(list[dict[str, list[list[int]]]]).validate_json(
            '{"detail": ["No results available for this job."]}'
        )
    except ValidationError as error:
        return error
    raise AssertionError("that json should not have validated")


@pytest.mark.parametrize(
    "error",
    [
        _not_ready(),
        RuntimeError("No results were available for job abc even though it is done."),
    ],
    ids=["validation_error", "execution_error"],
)
def test_results_are_asked_for_again_when_they_are_not_ready(error, monkeypatch):
    """A device can report a job done before its measurements can be fetched."""
    monkeypatch.setattr(job_module, "RESULT_RETRY_DELAY", 0)
    job = _NotReadyJob(error)

    assert job_module._result(job) == "counts"
    assert job.calls == 2


def test_a_failure_that_is_not_about_readiness_is_raised():
    """Only the not-ready-yet failure is worth asking again about."""
    job = _NotReadyJob(ConnectionError("connection reset by peer"))

    with pytest.raises(ConnectionError):
        job_module._result(job)
    assert job.calls == 1


def test_results_still_missing_after_the_retry_raise(monkeypatch):
    monkeypatch.setattr(job_module, "RESULT_RETRY_DELAY", 0)
    job = _NotReadyJob(_not_ready(), _not_ready())

    with pytest.raises(ValidationError):
        job_module._result(job)
    assert job.calls == 2


def test_the_private_qiskit_primitives_building_relies_on():
    """Building experiment circuits goes past QuantumCircuit's public API twice.

    ``data.insert`` runs the conversion meant for the deprecated three-tuple form and
    ``copy`` copies every operation, and neither is wanted hundreds of thousands of
    times. The replacements are private, so a qiskit that changes them should fail here
    rather than somewhere inside a cut.
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.append(Gate("marker", 1, []), [1])

    instruction = CircuitInstruction(Gate("inserted", 1, []), (circuit.qubits[0],), ())
    circuit._data.insert(1, instruction)
    assert [i.operation.name for i in circuit.data] == ["h", "inserted", "marker"]
    assert circuit.data[1].operation is instruction.operation

    shared = circuit.copy_empty_like()
    shared._data = circuit._data.copy(copy_instructions=False)
    assert [i.operation.name for i in shared.data] == ["h", "inserted", "marker"]
    assert all(a.operation is b.operation for a, b in zip(shared.data, circuit.data))

    # the public copy does not share, which is the whole reason for the flag
    assert circuit.copy().data[2].operation is not circuit.data[2].operation


@pytest.mark.sim
def test_experiment_circuits_share_their_operations():
    """A consequence of the above, and what keeps the memory down at scale."""
    marked, _plain, observables = _wire_and_gate_cut()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), observables
    )

    circuits = [
        circuit
        for group in experiment.experiments
        for obs_group in group
        for circuit in obs_group.values()
    ]
    instructions = sum(len(circuit.data) for circuit in circuits)
    distinct = len({id(i.operation) for c in circuits for i in c.data})

    assert instructions > 10 * distinct, (
        f"{instructions} instructions over only {distinct} operations was expected"
    )


def test_every_exported_name_exists():
    """``from QCut import *`` must not raise.

    ``__all__`` listed a name that had been removed, so the star import failed with an
    AttributeError. Nothing else in the suite exercised it.
    """
    missing = [name for name in ck.__all__ if not hasattr(ck, name)]
    assert missing == []
