"""Tests for wire cuts that exchange the measured outcome between the partitions."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut, cutGate
from QCut.errors.qcuterror import QCutError
from QCut.execution import waves
from QCut.execution.waves import MEASURE_SHARE
from QCut.qpd.bundle import plan_bundles
from QCut.qpd.qpd_locc import (
    MAX_BLOCK,
    _label_sign,
    _lagrangians,
    _parity,
    _span,
    _symplectic,
    gamma_local,
    gamma_locc,
    locc_wire_qpd,
    mub_unitaries,
)
from QCut.utils.circuit import _remove_obsm_2

TOLERANCE = 0.1

#: Bound for the tests whose point is that the plumbing pairs the right operations
#: together. Getting that wrong moves an expectation value by order one, not by a
#: fraction, so these can run on a quarter of the shots and still separate right from
#: wrong by a factor of five. Keeps the same margin over the measured worst error that
#: the tighter bound had at four times the budget.
STRUCTURAL_TOLERANCE = 0.2
WIDTHS = (1, 2, 3, 4)


def _column(label: tuple[int, ...]) -> int:
    """Basis index of a label, its first bit least significant to match qiskit."""
    return sum(bit << position for position, bit in enumerate(label))


def _prepare_and_project(width: int, unitary: np.ndarray, projector, state):
    """Superoperator of ``rho -> Tr[projector rho] * state``, row-major vec order."""
    return np.outer(state.reshape(-1), projector.T.reshape(-1))


@pytest.mark.parametrize("width", WIDTHS, ids=[f"n{w}" for w in WIDTHS])
def test_bases_are_mutually_unbiased(width):
    """Every pair of bases must overlap uniformly, else it is not a spread."""
    unitaries = mub_unitaries(width)
    dimension = 1 << width
    assert len(unitaries) == dimension + 1
    for unitary in unitaries:
        assert np.abs(unitary.conj().T @ unitary - np.eye(dimension)).max() < 1e-10
    for first, second in itertools.combinations(unitaries, 2):
        overlap = np.abs(first.conj().T @ second) ** 2
        assert np.abs(overlap - 1 / dimension).max() < 1e-10


@pytest.mark.parametrize("width", WIDTHS, ids=[f"n{w}" for w in WIDTHS])
def test_the_pauli_partition_is_a_symplectic_spread(width):
    """The commuting sets must partition the non-identity Paulis exactly once each.

    Pinned because the subspaces are only isotropic for the coordinate form once the Z
    component is written in the trace-dual basis. Using the plain polynomial basis is
    right up to two qubits and silently wrong from three.
    """
    subspaces = _lagrangians(width)
    spans = [_span(generators) for generators in subspaces]
    assert len(spans) == (1 << width) + 1
    for span in spans:
        assert len(span) == 1 << width
        assert not any(_symplectic(a, b) for a in span for b in span)
    seen = set()
    for span in spans:
        fresh = span - {(0, 0)}
        assert not (fresh & seen)
        seen |= fresh
    assert len(seen) == 4**width - 1


@pytest.mark.parametrize("width", WIDTHS, ids=[f"n{w}" for w in WIDTHS])
def test_decomposition_reconstructs_the_identity_channel(width):
    """The terms must sum to the identity on all the cut wires at once.

    The measured label is an outcome rather than a term, so its ``2**n`` values share
    one channel's coefficient. Undoing that share, the label's sign and the wire cut
    parity leaves the bare theorem, which is what has to sum to the identity.
    """
    unitaries = mub_unitaries(width)
    dimension = 1 << width
    total = np.zeros((dimension * dimension,) * 2, dtype=complex)
    for term in locc_wire_qpd(width):
        unitary = unitaries[term["channel"]]
        measured = unitary[:, _column(term["label"])]
        prepared_label = tuple(int(bit) for bit in term["op_1"].name.split("-")[-1])
        prepared = unitary[:, _column(prepared_label)]
        weight = term["c"] / _label_sign(term["label"]) / _parity(width) * dimension
        total += weight * _prepare_and_project(
            width,
            unitary,
            np.outer(measured, measured.conj()),
            np.outer(prepared, prepared.conj()),
        )
    assert np.abs(total - np.eye(dimension * dimension)).max() < 1e-12


@pytest.mark.parametrize("width", WIDTHS, ids=[f"n{w}" for w in WIDTHS])
def test_gamma_and_channel_counts(width):
    terms = locc_wire_qpd(width)
    assert sum(abs(term["c"]) for term in terms) == pytest.approx(gamma_locc(width))
    assert gamma_locc(width) == 2 ** (width + 1) - 1
    assert gamma_local(width) == 4**width
    assert len({term["channel"] for term in terms}) == (1 << width) + 1
    assert len(terms) == (1 << width) * (2 ** (width + 1) - 1)


def test_gamma_beats_local_operations_from_two_wires_up():
    """One wire gains little, and the gain grows quickly after that."""
    assert gamma_locc(1) == 3 and gamma_local(1) == 4
    for width in (2, 3, 4):
        assert gamma_locc(width) < gamma_local(width) / 2


def test_out_of_range_width_is_refused():
    with pytest.raises(QCutError, match="out of range"):
        mub_unitaries(MAX_BLOCK + 1)
    with pytest.raises(QCutError, match="out of range"):
        mub_unitaries(0)


def _blocks(n_wires, with_cut=True):
    """``n_wires`` cut in parallel, with both sides internally connected."""
    width = 2 * n_wires
    circuit = QuantumCircuit(width)
    for qubit in range(width):
        circuit.ry(0.3 + 0.13 * qubit, qubit)
    for index in range(n_wires - 1):
        circuit.cx(index, index + 1)
    for index in range(n_wires):
        circuit.rz(0.25 + 0.1 * index, index)
    if with_cut:
        for index in range(n_wires):
            circuit.append(cut(), [index])
    for index in range(n_wires):
        circuit.cx(index, n_wires + index)
    for index in range(n_wires - 1):
        circuit.cx(n_wires + index, n_wires + index + 1)
    for qubit in range(n_wires, width):
        circuit.ry(0.2, qubit)
    return circuit


def _zs(width):
    return ["I" * (width - 1 - k) + "Z" + "I" * k for k in range(width)]


def _run(circuit, observables, options, shots=2**12):
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=shots)
    values = ck.estimate_expectation_values(results)
    return values, experiment


def _exact(circuit, observables):
    state = Statevector(circuit)
    return [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]


@pytest.mark.parametrize(
    ("n_wires", "groups", "gamma"), [(1, 6, 3), (2, 28, 7), (3, 120, 15)]
)
@pytest.mark.sim
def test_a_block_of_wires_runs_end_to_end(n_wires, groups, gamma):
    """The whole two-phase path, against the exact expectation values."""
    observables = _zs(2 * n_wires)
    exact = _exact(_blocks(n_wires, with_cut=False), observables)

    values, experiment = _run(
        _blocks(n_wires),
        observables,
        CutOptions(wire_cut_communication="always"),
    )
    assert experiment.communicates
    assert experiment.num_groups == groups
    assert sum(abs(c) for c in experiment.coefficients) == pytest.approx(gamma)
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE


@pytest.mark.parametrize("n_wires", [1, 2])
@pytest.mark.sim
def test_communicating_and_local_agree(n_wires):
    observables = _zs(2 * n_wires)
    exact = _exact(_blocks(n_wires, with_cut=False), observables)
    for strategy in ("always", "never"):
        values, _ = _run(
            _blocks(n_wires),
            observables,
            CutOptions(wire_cut_communication=strategy),
        )
        for expected, actual in zip(exact, values):
            assert abs(expected - actual) < TOLERANCE


def test_a_block_uses_far_fewer_circuits():
    """Circuit count is the reliable win, dropping from 8**n to (2**n)(2**(n+1) - 1)."""
    observables = _zs(6)
    counts = {}
    for strategy in ("always", "never"):
        cut_circuit = ck.get_locations_and_subcircuits(
            _blocks(3), options=CutOptions(wire_cut_communication=strategy)
        )
        experiment = ck.get_experiment_circuits(cut_circuit, observables)
        counts[strategy] = experiment.num_circuits
    assert counts["always"] * 4 < counts["never"]


def _bundles(circuit, options=None):
    cut_circuit = ck.get_locations_and_subcircuits(
        circuit, options=options or CutOptions()
    )
    subcircuits = [sub.copy() for sub in cut_circuit.subcircuits]
    _remove_obsm_2(subcircuits)
    return plan_bundles(cut_circuit.cut_locations, subcircuits, cut_circuit.options)


def test_auto_leaves_a_single_wire_alone():
    """One wire gains little from communicating and pays for the emulation, so auto
    only uses it from two wires up."""
    single = _bundles(_blocks(1), CutOptions())
    assert all(bundle.kind == "single" for bundle in single)
    forced = _bundles(_blocks(1), CutOptions(wire_cut_communication="always"))
    assert any(bundle.kind == "cc_wire" for bundle in forced)


def test_auto_takes_a_block_of_two():
    bundles = _bundles(_blocks(2), CutOptions())
    communicating = [bundle for bundle in bundles if bundle.kind == "cc_wire"]
    assert len(communicating) == 1
    assert communicating[0].size == 2


def test_never_leaves_every_wire_alone():
    bundles = _bundles(_blocks(3), CutOptions(wire_cut_communication="never"))
    assert all(bundle.kind == "single" for bundle in bundles)


def test_strategy_accepts_booleans_and_rejects_nonsense():
    assert CutOptions(wire_cut_communication=True).min_communicating_block == 1
    assert CutOptions(wire_cut_communication=False).min_communicating_block == 0
    assert CutOptions().min_communicating_block == 2
    with pytest.raises(QCutError, match="unknown wire_cut_communication"):
        CutOptions(wire_cut_communication="sometimes")


def _cyclic(with_cut=True):
    """Wire cuts running both ways between the same two subcircuits."""
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.3 + 0.1 * qubit, qubit)
    circuit.cx(0, 1)
    if with_cut:
        circuit.append(cut(), [1])
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    if with_cut:
        circuit.append(cut(), [2])
    circuit.cx(0, 2)
    return circuit


@pytest.mark.sim
def test_a_cycle_falls_back_for_one_direction():
    """Running both directions in order is impossible, so one keeps the local table."""
    bundles = _bundles(_cyclic(), CutOptions(wire_cut_communication="always"))
    kinds = sorted(bundle.kind for bundle in bundles)
    assert kinds == ["cc_wire", "single"]

    observables = _zs(4)
    exact = _exact(_cyclic(with_cut=False), observables)
    values, experiment = _run(
        _cyclic(), observables, CutOptions(wire_cut_communication="always")
    )
    assert sum(abs(c) for c in experiment.coefficients) == pytest.approx(3 * 4)
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE


def _wires_and_gate(with_cut=True):
    circuit = QuantumCircuit(5)
    for qubit in range(5):
        circuit.ry(0.4 + 0.1 * qubit, qubit)
    circuit.cx(0, 1)
    if with_cut:
        circuit.append(cut(), [0])
        circuit.append(cut(), [1])
        circuit.append(**cutGate(RZZGate(0.8), 2, 4))
    else:
        circuit.rzz(0.8, 2, 4)
    circuit.cx(0, 2)
    circuit.cx(1, 3)
    circuit.cx(2, 3)
    circuit.ry(0.2, 4)
    return circuit


@pytest.mark.slow
@pytest.mark.sim
def test_a_gate_cut_alongside_a_communicating_block():
    """The shared measuring runs must not hand a group another group's gate cut."""
    observables = _zs(5)
    exact = _exact(_wires_and_gate(with_cut=False), observables)
    values, experiment = _run(_wires_and_gate(), observables, CutOptions())
    assert experiment.communicates
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < STRUCTURAL_TOLERANCE


def test_cuts_joining_different_subcircuit_pairs_do_not_form_a_block():
    """A block side has to sit in one subcircuit, or its operation cannot be placed."""
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.3 + 0.1 * qubit, qubit)
    circuit.append(cut(), [0])
    circuit.append(cut(), [1])
    # The two preparing sides end up in separate subcircuits, so the cuts join different
    # pairs and no single operation can span either side.
    circuit.cx(0, 2)
    circuit.cx(1, 3)
    bundles = _bundles(circuit, CutOptions(wire_cut_communication="always"))
    assert all(bundle.size == 1 for bundle in bundles)


def test_two_cuts_on_one_wire_do_not_form_a_block():
    """Sequential cuts on a wire are not parallel, whatever else is true of them."""
    circuit = QuantumCircuit(3)
    circuit.ry(0.6, 0)
    circuit.cx(0, 1)
    circuit.append(cut(), [1])
    circuit.cx(1, 2)
    circuit.append(cut(), [1])
    circuit.cx(1, 0)
    bundles = _bundles(circuit, CutOptions(wire_cut_communication="always"))
    assert all(bundle.size == 1 for bundle in bundles)


def _chain(with_cut=True):
    """A feeds B feeds C, so the dependencies need three waves rather than two."""
    circuit = QuantumCircuit(6)
    for qubit in range(6):
        circuit.ry(0.3 + 0.1 * qubit, qubit)
    circuit.cx(0, 1)
    if with_cut:
        circuit.append(cut(), [0])
        circuit.append(cut(), [1])
    circuit.cx(0, 2)
    circuit.cx(1, 3)
    circuit.cx(2, 3)
    if with_cut:
        circuit.append(cut(), [2])
        circuit.append(cut(), [3])
    circuit.cx(2, 4)
    circuit.cx(3, 5)
    circuit.cx(4, 5)
    return circuit


@pytest.mark.sim
def test_chained_blocks_need_a_wave_each():
    """B is a preparing side and a measuring side at once, so it cannot share a wave.

    Two blocks in series would silently lose every group if the shot allocation were
    computed from the first wave alone, since B's own outcome is not known until B runs.
    """
    observables = _zs(6)
    exact = _exact(_chain(with_cut=False), observables)

    values, experiment = _run(
        _chain(),
        observables,
        CutOptions(wire_cut_communication="always", expansion="exact"),
        shots=2**11,
    )
    assert experiment.communicates
    assert experiment.plan.last_wave == 2
    assert sorted(experiment.plan.waves.values()) == [0, 1, 2]
    assert sum(abs(c) for c in experiment.coefficients) == pytest.approx(49)
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < 0.2


def _count_jobs(experiment, max_batch_size, shots=2048):
    """Run an experiment and report how many backend jobs it took."""
    jobs = []
    original = AerSimulator.run

    def counting(self, circuits, **kwargs):
        jobs.append(len(circuits) if isinstance(circuits, list) else 1)
        return original(self, circuits, **kwargs)

    AerSimulator.run = counting
    try:
        results = ck.run_experiments(
            experiment,
            backend=AerSimulator(),
            shots=shots,
            max_batch_size=max_batch_size,
        )
    finally:
        AerSimulator.run = original
    return jobs, results


@pytest.mark.sim
def test_a_wave_costs_few_jobs():
    """Batching keeps a wave to a handful of jobs rather than one per group."""
    cut_circuit = ck.get_locations_and_subcircuits(
        _blocks(2), options=CutOptions(wire_cut_communication="always")
    )
    experiment = ck.get_experiment_circuits(cut_circuit, _zs(4))
    jobs, _ = _count_jobs(experiment, 100)
    assert len(jobs) <= 12  # against 28 groups over two waves
    assert all(count <= 100 for count in jobs)


@pytest.mark.sim
def test_a_generous_batch_size_does_not_collapse_the_allocation():
    """Shots split by label, so a batch must not span every request at once.

    With only a size limit, a large ``max_batch_size`` would put a whole wave in one job
    at one shot count, which is uniform allocation and throws the split away. The spread
    limit keeps the job count and the accuracy roughly independent of the batch size.
    """
    observables = _zs(6)
    exact = _exact(_blocks(3, with_cut=False), observables)
    cut_circuit = ck.get_locations_and_subcircuits(
        _blocks(3), options=CutOptions(wire_cut_communication="always")
    )
    experiment = ck.get_experiment_circuits(cut_circuit, observables)

    counts = {}
    for max_batch_size in (100, 10**9):
        jobs, results = _count_jobs(experiment, max_batch_size, shots=2**12)
        counts[max_batch_size] = len(jobs)
        values = ck.estimate_expectation_values(results)
        for expected, actual in zip(exact, values):
            assert abs(expected - actual) < 0.2

    # A whole wave never collapses into a single job, however large the batches may be.
    assert counts[10**9] > 2
    assert counts[10**9] <= 2 * counts[100]


def _shots_per_wave(experiment, shots=2048, max_batch_size=100):
    """Return how many shots each wave consumed, in wave order.

    One :func:`QCut.execution.dispatch._dispatch` call is one wave, so counting inside
    the backend and starting a new tally per call splits the run up the way the waves
    do. It is swapped where :mod:`QCut.execution.waves` looks it up.
    """
    per_wave: list[int] = []
    original_dispatch = waves._dispatch
    original_run = AerSimulator.run

    def counting(self, circuits, **kwargs):
        count = len(circuits) if isinstance(circuits, list) else 1
        per_wave[-1] += count * kwargs.get("shots", 1024)
        return original_run(self, circuits, **kwargs)

    def dispatch(*args, **kwargs):
        per_wave.append(0)
        return original_dispatch(*args, **kwargs)

    waves._dispatch = dispatch
    AerSimulator.run = counting
    try:
        ck.run_experiments(
            experiment,
            backend=AerSimulator(),
            shots=shots,
            max_batch_size=max_batch_size,
        )
    finally:
        waves._dispatch = original_dispatch
        AerSimulator.run = original_run
    return per_wave


@pytest.mark.parametrize(
    ("builder", "width", "pieces"),
    [(lambda: _blocks(2), 4, 2), (_chain, 6, 3)],
    ids=["two_pieces", "three_pieces"],
)
@pytest.mark.sim
def test_the_measuring_wave_takes_a_smaller_share(builder, width, pieces):
    """Sharing makes a measuring shot worth more, so wave zero gets less than its head.

    Dividing the budget evenly over the subcircuits would hand the measuring wave one
    share in ``pieces``, which measures worse than :data:`MEASURE_SHARE` at both two and
    three pieces. The reweighting has to leave the run's total alone.
    """
    cut_circuit = ck.get_locations_and_subcircuits(
        builder(), options=CutOptions(wire_cut_communication="always")
    )
    experiment = ck.get_experiment_circuits(cut_circuit, _zs(width))
    shots = 2048
    per_wave = _shots_per_wave(experiment, shots=shots)

    assert len(per_wave) == pieces
    share = per_wave[0] / sum(per_wave)
    even = 1 / pieces
    assert abs(share - MEASURE_SHARE) < abs(share - even)

    # Budget neutral: every subcircuit still costs shots * groups between them.
    nominal = shots * experiment.num_groups * pieces
    assert 0.8 * nominal < sum(per_wave) < 1.2 * nominal
