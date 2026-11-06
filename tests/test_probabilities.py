"""Reconstructing a distribution over a chosen set of qubits from a cut circuit."""

from collections.abc import Mapping

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut, cutGate
from QCut.post_process.postprocess import estimate_expectation_values, walsh_hadamard
from QCut.post_process.probabilities import (
    DEFAULT_TOP,
    QuasiProbabilities,
    _dense_values,
    _separable_form,
)
from QCut.utils.observables import _all_z_paulis_for_subset

SHOTS = 40000

#: The reconstruction and the observable formulation it replaced are the same sum in a
#: different order, so they agree to rounding rather than to shot noise. Measured worst
#: case over the cases below is 8e-16, which leaves three orders of margin.
EXACT = 1e-12


def _bell_with_a_gate_cut():
    """A cut Bell pair with a spectator, so the marginal is not the whole state."""
    marked, plain = QuantumCircuit(3), QuantumCircuit(3)
    marked.h(0)
    plain.h(0)
    marked.append(**cutGate(CXGate(), 0, 1))
    plain.cx(0, 1)
    for circuit in (marked, plain):
        circuit.x(2)
    return marked, plain


def _spread_state():
    """Distinct marginals on every qubit, so a mis-ordered key cannot pass."""
    marked, plain = QuantumCircuit(3), QuantumCircuit(3)
    for circuit in (marked, plain):
        circuit.ry(0.7, 0)
    marked.append(**cutGate(CXGate(), 0, 1))
    plain.cx(0, 1)
    for circuit in (marked, plain):
        circuit.ry(1.3, 2)
    return marked, plain


def _reconstruct(marked, qubits, shots=SHOTS):
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), qubits=qubits
    )
    results = ck.run_experiments(experiment, shots=shots, backend=AerSimulator())
    return ck.estimate_probabilities(results)


def _exact_marginal(plain, qubits):
    """The true distribution over ``qubits``, keyed the way QCut keys it."""
    marginal: dict[str, float] = {}
    for bits, probability in Statevector(plain).probabilities_dict().items():
        per_qubit = bits[::-1]
        key = "".join(per_qubit[qubit] for qubit in reversed(qubits))
        marginal[key] = marginal.get(key, 0.0) + probability
    return marginal


@pytest.mark.sim
def test_the_reconstruction_matches_the_exact_distribution():
    marked, plain = _bell_with_a_gate_cut()
    probs = _reconstruct(marked, [0, 1])
    exact = _exact_marginal(plain, [0, 1])

    assert set(probs) == {"00", "01", "10", "11"}
    for key, value in probs.nearest_probabilities().items():
        assert value == pytest.approx(exact.get(key, 0.0), abs=0.05)


@pytest.mark.sim
def test_a_subset_reconstructs_the_marginal_over_those_qubits():
    """The chosen qubits need not be contiguous, and the rest are summed over."""
    marked, plain = _spread_state()
    probs = _reconstruct(marked, [0, 2])
    exact = _exact_marginal(plain, [0, 2])

    for key, value in probs.nearest_probabilities().items():
        assert value == pytest.approx(exact.get(key, 0.0), abs=0.05)


@pytest.mark.sim
def test_the_first_qubit_given_is_the_last_character_of_the_key():
    """Keys follow the order ``qubits`` was given, not the circuit's own order."""
    marked, plain = _spread_state()
    forwards = _reconstruct(marked, [0, 2]).nearest_probabilities()
    backwards = _reconstruct(marked, [2, 0]).nearest_probabilities()

    for key, value in forwards.items():
        assert backwards[key[::-1]] == pytest.approx(value, abs=0.05)

    exact = _exact_marginal(plain, [0, 2])
    assert forwards["10"] == pytest.approx(exact["10"], abs=0.05)


@pytest.mark.sim
def test_the_values_are_quasi_probabilities_and_the_projection_is_not():
    marked, _plain = _bell_with_a_gate_cut()
    probs = _reconstruct(marked, [0, 1], shots=2000)

    assert isinstance(probs, Mapping)
    assert not isinstance(probs, dict), (
        "deliberately not a dict: a real one would have to hold every bitstring, "
        "which is what top() and probability_of() exist to avoid"
    )
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)

    nearest = probs.nearest_probabilities()
    assert min(nearest.values()) >= 0.0
    assert sum(nearest.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(nearest) == set(probs), "clipped bitstrings are kept, as zeros"
    assert probs.quasi_probabilities() == dict(probs)


def test_the_total_is_one_however_wrong_the_estimates_are():
    """The identity term fixes the sum, so the total cannot be used as a check."""
    wrong = np.array([1 + 0.5 - 0.5, 1 - 0.5 + 0.5, 1 + 0.5 + 0.5, 1 - 0.5 - 0.5]) / 4
    nonsense = QuasiProbabilities(dict(zip(["00", "01", "10", "11"], wrong)))

    assert sum(nonsense.values()) == pytest.approx(1.0)


@pytest.mark.sim
def test_counts_scale_by_the_shots_the_experiment_ran_at():
    marked, _plain = _bell_with_a_gate_cut()
    probs = _reconstruct(marked, [0, 1], shots=2000)

    assert probs.shots == 2000
    assert sum(probs.counts().values()) == pytest.approx(2000, abs=1e-6)
    assert sum(probs.counts(shots=137).values()) == pytest.approx(137, abs=1e-6)
    assert min(probs.counts().values()) >= 0.0, "counts are projected first"


def _spread_over_five_bits():
    """A width-5 distribution, so the default view has to leave something out."""
    values = np.zeros(32)
    values[[3, 7, 11, 19, 27]] = [0.4, 0.3, 0.2, 0.15, -0.05]
    values[0] = 1.0 - values.sum()
    return QuasiProbabilities.from_array(values, shots=500)


def test_the_dict_views_show_the_leading_bitstrings_by_default():
    """A distribution over more than a few qubits has more entries than anyone reads."""
    probs = _spread_over_five_bits()

    assert len(probs) == 32
    for view in (probs.quasi_probabilities(), probs.nearest_probabilities()):
        assert len(view) == DEFAULT_TOP
        assert list(view.values()) == sorted(view.values(), reverse=True)
    assert len(probs.counts()) == DEFAULT_TOP


def test_asking_for_none_gives_the_whole_distribution():
    """``top=None`` is the unabridged view, in numerical order as a dict always was."""
    probs = _spread_over_five_bits()

    for view in (
        probs.quasi_probabilities(top=None),
        probs.nearest_probabilities(top=None),
        probs.counts(top=None),
    ):
        assert len(view) == 32
        assert list(view) == [format(x, "05b") for x in range(32)]

    assert probs.quasi_probabilities(top=None) == dict(probs)
    assert sum(probs.counts(top=None).values()) == pytest.approx(500, abs=1e-6)


def test_a_narrow_distribution_is_unaffected_by_the_default():
    """Everything fits inside the default, so the views are the whole thing."""
    probs = QuasiProbabilities({"00": 0.5, "01": 0.3, "10": 0.25, "11": -0.05}, 1000)

    assert probs.quasi_probabilities() == dict(probs)
    assert set(probs.nearest_probabilities()) == set(probs)
    assert sum(probs.counts().values()) == pytest.approx(1000, abs=1e-6)


@pytest.mark.parametrize("top", [1, 3, 10, 32, 1000])
def test_the_views_agree_with_the_full_distribution_they_abridge(top):
    probs = _spread_over_five_bits()
    whole = probs.quasi_probabilities(top=None)
    projected = probs.nearest_probabilities(top=None)

    quasi = probs.quasi_probabilities(top)
    nearest = probs.nearest_probabilities(top)
    assert len(quasi) == len(nearest) == min(top, 32)
    for key, value in quasi.items():
        assert value == pytest.approx(whole[key], abs=1e-12)
    for key, value in nearest.items():
        assert value == pytest.approx(projected[key], abs=1e-12)


def test_counts_only_sum_to_the_shots_when_nothing_is_left_out():
    """Truncating drops the mass with the bitstrings, as the docstring warns."""
    probs = _spread_over_five_bits()

    assert sum(probs.counts(top=None).values()) == pytest.approx(500, abs=1e-6)
    assert sum(probs.counts(top=2).values()) < 500
    assert sum(probs.counts(shots=100, top=None).values()) == pytest.approx(
        100, abs=1e-6
    )


@pytest.mark.parametrize(
    "view", ["quasi_probabilities", "nearest_probabilities", "counts"]
)
@pytest.mark.parametrize("top", [0, -1])
def test_the_views_want_a_positive_count_or_none(view, top):
    probs = _spread_over_five_bits()
    with pytest.raises(ValueError, match="at least one"):
        getattr(probs, view)(top=top)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8])
def test_the_repr_never_shows_more_than_it_would_for_a_wider_distribution(width):
    """A bigger distribution used to print fewer entries, over a width threshold."""
    values = np.full(1 << width, 2.0**-width)
    shown = repr(QuasiProbabilities.from_array(values)).count(":")

    assert shown == min(1 << width, DEFAULT_TOP)
    if (1 << width) > DEFAULT_TOP:
        assert f"{1 << width} bitstrings" in repr(QuasiProbabilities.from_array(values))


def test_counts_need_a_shot_count_from_somewhere():
    bare = QuasiProbabilities({"0": 0.6, "1": 0.4})

    with pytest.raises(ValueError, match="no shot count"):
        bare.counts()
    assert bare.counts(shots=10) == {"0": 6.0, "1": 4.0}


@pytest.mark.sim
def test_every_z_string_shares_one_measurement_setting():
    """The observables multiply with the qubit count, the circuits do not."""
    marked, _plain = _spread_state()
    counts = []
    for width in (1, 2, 3):
        experiment = ck.get_experiment_circuits(
            ck.get_locations_and_subcircuits(marked), qubits=list(range(width))
        )
        assert experiment.observables.size == 2**width - 1
        assert experiment.num_obs_groups == 1
        counts.append(experiment.num_circuits)

    assert len(set(counts)) == 1, f"circuit count should not grow: {counts}"


def test_probabilities_need_an_experiment_built_from_qubits():
    marked, _plain = _bell_with_a_gate_cut()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), observables=["IIZ"]
    )
    results = ck.run_experiments(experiment, shots=128, backend=AerSimulator())

    assert not experiment.can_reconstruct_probabilities
    with pytest.raises(ValueError, match="Cannot reconstruct probabilities"):
        ck.estimate_probabilities(results)


def test_observables_and_qubits_are_mutually_exclusive():
    marked, _plain = _bell_with_a_gate_cut()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    with pytest.raises(ValueError, match="Either observables or qubits"):
        ck.get_experiment_circuits(cut_circuit)
    with pytest.raises(ValueError, match="Only one of observables or qubits"):
        ck.get_experiment_circuits(cut_circuit, observables=["IIZ"], qubits=[0])


@pytest.mark.sim
@pytest.mark.parametrize("shorthand", ["run", "run_cut_circuit"])
def test_the_shorthands_reconstruct_when_given_qubits(shorthand):
    """``run`` and ``run_cut_circuit`` take ``qubits`` the way the long form does."""
    marked, plain = _bell_with_a_gate_cut()
    target = marked if shorthand == "run" else ck.get_locations_and_subcircuits(marked)

    probs = getattr(ck, shorthand)(
        target, backend=AerSimulator(), shots=SHOTS, qubits=[0, 1]
    )

    assert isinstance(probs, QuasiProbabilities)
    assert probs.shots == SHOTS
    exact = _exact_marginal(plain, [0, 1])
    for key, value in probs.nearest_probabilities().items():
        assert value == pytest.approx(exact.get(key, 0.0), abs=0.05)


@pytest.mark.parametrize("shorthand", ["run", "run_cut_circuit"])
def test_the_shorthands_also_need_exactly_one_of_the_two(shorthand):
    """Checked before any circuit is built, not once the experiment is assembled."""
    marked, _plain = _bell_with_a_gate_cut()
    target = marked if shorthand == "run" else ck.get_locations_and_subcircuits(marked)

    with pytest.raises(ValueError, match="Either observables or qubits"):
        getattr(ck, shorthand)(target)
    with pytest.raises(ValueError, match="Only one of observables or qubits"):
        getattr(ck, shorthand)(target, ["IIZ"], qubits=[0])


@pytest.mark.parametrize(
    ("qubits", "message"),
    [
        ([], "at least one qubit"),
        ([0, 0], "must not repeat"),
        ([3], "outside the 3-qubit circuit"),
        ([-1], "outside the 3-qubit circuit"),
    ],
)
def test_the_qubits_asked_for_have_to_exist(qubits, message):
    """A qubit off the end used to wrap round onto a different one."""
    with pytest.raises(ValueError, match=message):
        _all_z_paulis_for_subset(3, qubits)


def _wire_cut():
    """One wire cut, so the parity term is exercised as well as the gate-cut path."""
    circuit = QuantumCircuit(4)
    circuit.ry(0.7, 0)
    circuit.cx(0, 1)
    circuit.append(cut(), [1])
    circuit.cx(1, 2)
    circuit.ry(1.1, 3)
    circuit.cx(2, 3)
    return circuit


def _parallel_wires(n_wires=2):
    """Wires that can be bundled, which is what classical communication needs."""
    width = 2 * n_wires
    circuit = QuantumCircuit(width)
    for qubit in range(width):
        circuit.ry(0.3 + 0.13 * qubit, qubit)
    for index in range(n_wires - 1):
        circuit.cx(index, index + 1)
    for index in range(n_wires):
        circuit.rz(0.25 + 0.1 * index, index)
    for index in range(n_wires):
        circuit.append(cut(), [index])
    for index in range(n_wires):
        circuit.cx(index, n_wires + index)
    for index in range(n_wires - 1):
        circuit.cx(n_wires + index, n_wires + index + 1)
    return circuit


def _via_observables(results):
    """The distribution the way it used to be reconstructed.

    Every Pauli Z over the chosen qubits, estimated one observable at a time, through an
    inverse Walsh-Hadamard transform. Kept here rather than in the package because it is
    only wanted as an independent second opinion -- and because going through
    ``experiment.observables`` is what checks they are still there to be asked for.
    """
    width = len(results.experiment.qubits)
    coefficients = np.empty(1 << width)
    coefficients[0] = 1.0
    coefficients[1:] = estimate_expectation_values(results)
    return walsh_hadamard(coefficients) / (1 << width)


#: The reconstruction is an algebraic identity, so it has to hold for every shape of
#: experiment rather than on average: gate cuts, wire cuts, bundled wires with and
#: without classical communication, a sampled decomposition, subsets, a subset given
#: back to front, one qubit, and every qubit.
EQUIVALENCE_CASES = [
    ("gate cut, pair", _bell_with_a_gate_cut()[0], [0, 1], None),
    ("gate cut, spread subset", _spread_state()[0], [0, 2], None),
    ("gate cut, reversed subset", _spread_state()[0], [2, 0], None),
    ("gate cut, one qubit", _spread_state()[0], [1], None),
    ("gate cut, every qubit", _spread_state()[0], [0, 1, 2], None),
    ("one wire cut", _wire_cut(), [0, 1, 2, 3], None),
    ("one wire cut, subset", _wire_cut(), [3, 0], None),
    (
        "bundled wires, communicating",
        _parallel_wires(),
        [0, 2, 3],
        CutOptions(wire_cut_communication="always"),
    ),
    (
        "bundled wires, not communicating",
        _parallel_wires(),
        [0, 2, 3],
        CutOptions(wire_cut_communication="never"),
    ),
    (
        "sampled decomposition",
        _parallel_wires(),
        [0, 1, 2, 3],
        CutOptions(expansion="sample", num_samples=120, seed=7),
    ),
]


@pytest.mark.sim
@pytest.mark.parametrize(
    ("circuit", "qubits", "options"),
    [case[1:] for case in EQUIVALENCE_CASES],
    ids=[case[0] for case in EQUIVALENCE_CASES],
)
def test_the_reconstruction_agrees_with_the_observable_formulation(
    circuit, qubits, options
):
    """The product form and the 2**k observables must be the same number, not close.

    Shot noise cancels because both read the same counts, so this compares the two
    reconstructions rather than either against the truth. A few shots are enough.
    """
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(circuit, options=options), qubits=qubits
    )
    results = ck.run_experiments(experiment, shots=512, backend=AerSimulator())

    probs = ck.estimate_probabilities(results)
    assert np.abs(probs.probabilities() - _via_observables(results)).max() < EXACT


@pytest.mark.sim
@pytest.mark.parametrize(
    ("circuit", "qubits", "options"),
    [case[1:] for case in EQUIVALENCE_CASES],
    ids=[case[0] for case in EQUIVALENCE_CASES],
)
def test_expanding_group_by_group_gives_the_same_table(circuit, qubits, options):
    """The fallback for results with no one separable form has to agree with the form.

    Nothing reaches it on these experiments -- a subcircuit measuring none of the
    chosen qubits is folded into its group's coefficient, which is what used to make
    the layouts differ -- so it is checked against the path that does run.
    """
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(circuit, options=options), qubits=qubits
    )
    results = ck.run_experiments(experiment, shots=512, backend=AerSimulator())

    separable = _separable_form(results)
    assert separable is not None, "these experiments all have one form"
    assert np.abs(_dense_values(results) - separable.array()).max() < EXACT


@pytest.mark.sim
def test_the_circuits_do_not_depend_on_how_many_qubits_are_asked_for():
    """The whole saving rests on this: the observables shape nothing.

    If a wider distribution ever changed a circuit, deriving the measurement setting
    instead of folding the observables into it would be wrong.
    """
    marked, _plain = _spread_state()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    one = ck.get_experiment_circuits(cut_circuit, qubits=[0])
    every = ck.get_experiment_circuits(cut_circuit, qubits=[0, 1, 2])

    assert one.num_obs_groups == every.num_obs_groups == 1
    assert one.qpd_bits == every.qpd_bits
    for narrow, wide in zip(one.experiments, every.experiments):
        for obs_narrow, obs_wide in zip(narrow, wide):
            assert obs_narrow.keys() == obs_wide.keys()
            for index, circuit in obs_narrow.items():
                assert circuit == obs_wide[index]


def test_the_observables_are_only_built_when_something_asks_for_them():
    """2**k Pauli labels are what used to make a wide distribution unaffordable."""
    marked, _plain = _spread_state()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), qubits=[0, 1, 2]
    )

    assert experiment._observables is None, "not built while nothing has asked"
    assert experiment.observables.size == 2**3 - 1
    assert experiment._observables is not None, "and kept once it has"


def test_an_experiment_without_the_circuit_width_cannot_build_its_observables():
    """A hand-built experiment can be missing what the observables are derived from."""
    marked, _plain = _spread_state()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), qubits=[0, 1]
    )
    experiment._uncut_num_qubits = None

    with pytest.raises(ValueError, match="not the width of the circuit"):
        _ = experiment.observables


@pytest.mark.sim
def test_the_top_bitstrings_are_the_ones_the_whole_distribution_would_name():
    """``top`` skips most of the distribution, so it has to land on the same answer."""
    marked, _plain = _spread_state()
    probs = _reconstruct(marked, [0, 1, 2], shots=4000)
    table = probs.probabilities()

    for count in (1, 3, 8):
        top = probs.top(count)
        assert len(top) == count
        assert list(top.values()) == sorted(top.values(), reverse=True)
        best = np.sort(table)[::-1][:count]
        assert np.abs(np.sort(list(top.values()))[::-1] - best).max() < EXACT
        for key, value in top.items():
            assert value == pytest.approx(probs[key], abs=EXACT)


@pytest.mark.sim
def test_one_bitstring_can_be_had_without_the_others():
    marked, _plain = _spread_state()
    probs = _reconstruct(marked, [0, 1, 2], shots=4000)

    for key in probs:
        assert probs.probability_of(key) == pytest.approx(probs[key], abs=EXACT)


@pytest.mark.sim
def test_a_marginal_is_the_distribution_summed_over_the_qubits_dropped():
    """Exact, and it is the same answer as reconstructing that subset from scratch."""
    marked, plain = _spread_state()
    probs = _reconstruct(marked, [0, 1, 2], shots=SHOTS)

    marginal = probs.marginal([0, 2])
    assert marginal.qubits == [0, 2]

    summed: dict[str, float] = {}
    for key, value in probs.items():
        # key is qubit 0 last, so qubits [0, 2] keep characters -1 and -3.
        summed[key[-1] + key[-3]] = summed.get(key[-1] + key[-3], 0.0) + value
    for key, value in marginal.items():
        assert value == pytest.approx(summed[key[::-1]], abs=EXACT)

    exact = _exact_marginal(plain, [0, 2])
    for key, value in marginal.nearest_probabilities().items():
        assert value == pytest.approx(exact.get(key, 0.0), abs=0.05)


@pytest.mark.sim
def test_a_marginal_can_only_name_qubits_the_distribution_spans():
    marked, _plain = _spread_state()
    probs = _reconstruct(marked, [0, 2], shots=2000)

    with pytest.raises(ValueError, match=r"qubits \[1\] are not among"):
        probs.marginal([0, 1])


def test_the_queries_work_on_a_distribution_built_from_a_plain_mapping():
    """It is public, so the queries have to answer without a separable form too."""
    built = QuasiProbabilities({"00": 0.5, "01": 0.25, "10": 0.25, "11": 0.0})

    assert built.width == 2
    assert built.top(2) == {"00": 0.5, "01": 0.25}
    assert built.probability_of("10") == pytest.approx(0.25)
    assert built.marginal([1])["1"] == pytest.approx(0.25)
    assert built.probabilities().tolist() == [0.5, 0.25, 0.25, 0.0]


def test_a_missing_bitstring_is_a_key_error():
    built = QuasiProbabilities({"00": 1.0, "01": 0.0, "10": 0.0, "11": 0.0})

    assert len(built) == 4
    for key in ("0", "000", "0z", "", 0):
        with pytest.raises(KeyError):
            _ = built[key]


def test_a_circuit_with_measurements_is_accepted_and_left_alone():
    """Final measurements are stripped before splitting, on a copy."""
    marked, _plain = _bell_with_a_gate_cut()
    marked.measure_all()
    before = dict(marked.count_ops())

    ck.get_locations_and_subcircuits(marked)

    assert dict(marked.count_ops()) == before
