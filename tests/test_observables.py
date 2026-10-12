"""Observables are taken the way qiskit's estimator takes them.

Every accepted form is checked against ``Statevector.expectation_value`` on the uncut
circuit, so what is really being pinned is that the coercion in front of the pipeline
does not change the number that comes out the back.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut.execution.basis_transform import _combine_pauli_ops
from QCut.execution.observables import (
    MAX_PROJECTOR_TERMS,
    SUPPORTS_SPARSE_OBSERVABLE,
    _expand,
    coerce_observables,
    per_term_observables,
)

SHOTS = 200000
TOLERANCE = 0.05

# The class turned up in qiskit 1.4 but its estimators would not take one until 2.1,
# so whether QCut accepts it is a question about this qiskit rather than about QCut,
# and the two have to be told apart: below 1.4 there is nothing even to pass.
try:
    from qiskit.quantum_info import SparseObservable

    HAS_SPARSE_OBSERVABLE = True
except ImportError:  # qiskit below 1.4
    HAS_SPARSE_OBSERVABLE = False


def _pair():
    """A wire-cut circuit and the uncut circuit it is equivalent to."""
    marked, plain = QuantumCircuit(3), QuantumCircuit(3)
    marked.h(0)
    plain.h(0)
    for circuit in (marked, plain):
        circuit.cx(0, 1)
    marked.append(ck.cut(), [1])
    for circuit in (marked, plain):
        circuit.cx(1, 2)
        circuit.ry(0.7, 2)
    return marked, plain


def _exact(plain, observable):
    return float(np.real(Statevector(plain).expectation_value(observable)))


def _run(marked, observables, shots=SHOTS):
    return ck.run(marked, observables, AerSimulator(), shots=shots)


# --- what the coercion accepts ------------------------------------------------------


@pytest.mark.sim
@pytest.mark.parametrize(
    ("name", "observables", "expected"),
    [
        ("label", "ZZI", SparsePauliOp("ZZI")),
        ("pauli", Pauli("ZZI"), SparsePauliOp("ZZI")),
        (
            "mapping",
            {"ZZI": 0.5, "IZZ": 0.5},
            SparsePauliOp(["ZZI", "IZZ"], [0.5, 0.5]),
        ),
        (
            "sparse_pauli_op",
            SparsePauliOp(["ZZI", "IZZ"], [2.0, 3.0]),
            SparsePauliOp(["ZZI", "IZZ"], [2.0, 3.0]),
        ),
    ],
)
def test_a_single_observable_gives_a_single_value(name, observables, expected):
    """One observable in, one number out, with its coefficients applied."""
    marked, plain = _pair()
    value = _run(marked, observables)

    assert value.shape == (), f"{name} should give a scalar, got {value.shape}"
    assert float(value) == pytest.approx(_exact(plain, expected), abs=TOLERANCE)


@pytest.mark.sim
def test_a_sparse_pauli_op_is_one_observable_not_a_list_of_them():
    """The qiskit reading: a SparsePauliOp is the weighted sum of its terms.

    This is the one input where QCut used to disagree with the estimator, reading it as
    one observable per Pauli, so it is worth pinning on its own.
    """
    marked, plain = _pair()
    op = SparsePauliOp(["ZZI", "IZZ"], [2.0, 3.0])

    together = _run(marked, op)
    apart = _run(marked, ["ZZI", "IZZ"])

    assert together.shape == ()
    assert apart.shape == (2,)
    assert float(together) == pytest.approx(2 * apart[0] + 3 * apart[1], abs=TOLERANCE)


# --- shape --------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.parametrize(
    ("observables", "shape"),
    [
        ("ZZI", ()),
        (["ZZI", "IZZ", "ZII"], (3,)),
        ([["ZZI", "IZZ"], ["ZII", "III"]], (2, 2)),
    ],
)
def test_the_values_are_shaped_like_the_observables(observables, shape):
    """Output shape follows input shape, as it does for an estimator."""
    marked, plain = _pair()
    values = _run(marked, observables, shots=20000)

    assert values.shape == shape
    assert (
        ck.get_experiment_circuits(
            ck.get_locations_and_subcircuits(marked), observables
        ).observables.shape
        == shape
    )


@pytest.mark.sim
def test_a_nested_array_keeps_its_layout():
    """Every entry of a 2x2 array is its own observable, in place."""
    marked, plain = _pair()
    labels = [["ZZI", "IZZ"], ["ZII", "IIZ"]]

    values = _run(marked, labels)

    for row, row_labels in enumerate(labels):
        for column, label in enumerate(row_labels):
            assert values[row, column] == pytest.approx(
                _exact(plain, SparsePauliOp(label)), abs=TOLERANCE
            )


# --- terms --------------------------------------------------------------------------


def test_observables_sharing_a_term_measure_it_once():
    """The circuits are built for the distinct terms, not for the observables."""
    marked, _plain = _pair()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    one = ck.get_experiment_circuits(cut_circuit, [{"ZZI": 1.0}])
    two = ck.get_experiment_circuits(
        cut_circuit, [{"ZZI": 1.0}, {"ZZI": 2.0, "IZZ": 1.0}]
    )

    assert one.observable_terms.paulis.to_labels() == ["ZZI"]
    assert two.observable_terms.paulis.to_labels() == ["ZZI", "IZZ"]
    assert two.num_circuits == one.num_circuits, "a shared term is not measured twice"


def test_the_identity_term_costs_no_circuits_and_reads_as_one():
    """``<I> = 1``, and it shares whatever setting the rest of the observable needs."""
    marked, _plain = _pair()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    experiment = ck.get_experiment_circuits(cut_circuit, {"III": 0.25, "ZZI": 0.75})
    results = ck.run_experiments(experiment, shots=SHOTS, backend=AerSimulator())
    value = ck.estimate_expectation_values(results)

    plain = _pair()[1]
    expected = 0.25 + 0.75 * _exact(plain, SparsePauliOp("ZZI"))
    assert float(value) == pytest.approx(expected, abs=TOLERANCE)


def test_the_qubits_path_still_reads_one_value_per_z_string():
    """``qubits`` asks for every Z string separately, not for their sum.

    The weights between the observables and the terms are the identity, which is held
    as None rather than written out: this path has ``2**k`` terms, and an identity
    between them is the one thing on it that would grow as its square.
    """
    op = SparsePauliOp(["IIZ", "IZI", "IZZ"])
    spec = per_term_observables(op)

    assert spec.shape == (3,)
    assert spec.weights is None
    assert spec.terms.paulis.to_labels() == ["IIZ", "IZI", "IZZ"]
    assert np.array_equal(spec.combine([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])


# --- projectors ---------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        # |0><0| = (I+Z)/2 and |1><1| = (I-Z)/2 on the named qubit, so these read as the
        # probability of that outcome. Written as labels rather than as a
        # SparseObservable, which only qiskit 2.1 and later will coerce.
        ("I00", SparsePauliOp(["III", "IZI", "IIZ", "IZZ"], [0.25] * 4)),
        ("II1", SparsePauliOp(["III", "IIZ"], [0.5, -0.5])),
        ("I+I", SparsePauliOp(["III", "IXI"], [0.5, 0.5])),
    ],
)
def test_a_projector_label_reads_as_its_pauli_expansion(label, expected):
    """``0 1 + - r l`` work wherever qiskit's own coercion accepts them."""
    marked, plain = _pair()

    value = _run(marked, {label: 1.0})

    assert value.shape == ()
    assert float(value) == pytest.approx(_exact(plain, expected), abs=TOLERANCE)


@pytest.mark.sim
@pytest.mark.skipif(
    not SUPPORTS_SPARSE_OBSERVABLE, reason="qiskit coerces SparseObservable from 2.1"
)
@pytest.mark.parametrize("label", ["I00", "I+0", "r1I", "ZI1"])
def test_projector_terms_are_expanded_into_paulis(label):
    """``0 1 + - r l`` are supported, by writing each as ``(I +- P)/2``."""
    marked, plain = _pair()
    observable = SparseObservable.from_label(label)
    as_paulis = SparsePauliOp.from_sparse_list(
        observable.as_paulis().to_sparse_list(), num_qubits=3
    ).simplify()

    value = _run(marked, observable)

    assert value.shape == ()
    assert float(value) == pytest.approx(_exact(plain, as_paulis), abs=TOLERANCE)


@pytest.mark.skipif(
    not SUPPORTS_SPARSE_OBSERVABLE, reason="qiskit coerces SparseObservable from 2.1"
)
def test_a_projector_adds_no_measurement_settings():
    """Each character of a projector fixes one qubit's basis, so the sum is one setting.

    This is why the expansion is affordable: the terms multiply, the circuits do not.
    """
    marked, _plain = _pair()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    plain_z = ck.get_experiment_circuits(cut_circuit, "ZZI")
    projector = ck.get_experiment_circuits(
        cut_circuit, SparseObservable.from_label("00I")
    )

    assert projector.observable_terms is not None
    assert len(projector.observable_terms) == 4, "|00><00| is a sum of four Paulis"
    assert len(_combine_pauli_ops(projector.observable_terms)) == 1
    assert projector.num_circuits == plain_z.num_circuits


def test_a_projector_over_too_many_qubits_is_refused():
    """Past the cap the question is a distribution, which ``qubits`` answers."""
    width = MAX_PROJECTOR_TERMS.bit_length()  # one projector more than the cap allows

    with pytest.raises(ValueError, match="qubits"):
        coerce_observables("0" * width)


# --- errors -------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_SPARSE_OBSERVABLE or SUPPORTS_SPARSE_OBSERVABLE,
    reason="only qiskit 1.4 and 2.0 have the class without coercing it",
)
def test_a_sparse_observable_is_refused_where_qiskit_will_not_take_one():
    """The class exists from qiskit 1.4, but nothing would coerce it until 2.1.

    Left to itself qiskit raises a bare TypeError naming the class, which does not
    hint that a newer qiskit would have accepted it.
    """
    with pytest.raises(ValueError, match="2.1"):
        coerce_observables(SparseObservable.from_label("ZZI"))


def test_observables_must_span_the_whole_circuit():
    marked, _plain = _pair()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    with pytest.raises(ValueError, match="uncut circuit"):
        ck.get_experiment_circuits(cut_circuit, ["ZZ"])


def test_observables_of_differing_widths_are_refused():
    with pytest.raises(ValueError, match="number of qubits"):
        coerce_observables(["ZZZ", "ZZ"])


def test_non_hermitian_coefficients_are_refused():
    with pytest.raises(ValueError, match="Hermitian"):
        coerce_observables(SparsePauliOp(["ZZ"], [1j]))


def test_a_letter_outside_the_alphabet_is_refused():
    """qiskit's own coercion catches this first, and says so clearly enough.

    What it says is not matched on: the wording changed in qiskit 2.1, and the newer
    message does not even name the offending letter.
    """
    with pytest.raises(ValueError):
        coerce_observables({"ZQ": 1.0})


def test_the_expansion_names_a_letter_it_cannot_place():
    """The backstop for a qiskit whose own coercion let something unexpected through."""
    with pytest.raises(ValueError, match=r"'ZQ' uses \['Q'\], which is neither"):
        _expand("ZQ", 1.0)
