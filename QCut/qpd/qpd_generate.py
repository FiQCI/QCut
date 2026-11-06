"""Generate quasiprobability decompositions for arbitrary two-qubit gates.

The hand-derived tables in :mod:`QCut.qpd` cover ``cz``, ``swap`` and ``iswap``. This
module generalises the derivation in ``docs/theory/SWAP_derivation.md`` so a QPD can be
produced for any two-qubit gate.

The construction is Eq. (19) of K. Mitarai and K. Fujii, "Overhead for simulating a
non-local channel with local channels by quasiprobability sampling", Quantum 5, 388
(2021), `arXiv:2006.11174 <https://arxiv.org/abs/2006.11174>`_. Its third line is the
term ``SWAP_derivation.md`` omits, which vanishes for SWAP but is needed in general.

Two conventions differ from that page and both are pinned by
``tests/test_qpd_generate.py``. ``TwoQubitWeylDecomposition`` puts a plus sign in the
KAK exponent where the page writes a minus, which conjugates :math:`u` and flips the
sign of the third line. And in :mod:`QCut.qpd_gates` the ``axyp``/``axym`` and
``ayzp``/``ayzm`` pairs are labelled the opposite way round from the paper, so the
mapping below crosses them over. See ``docs/theory/General_2q_derivation.md``.
"""

from __future__ import annotations

import logging

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitWeylDecomposition

from QCut.errors.qcuterror import QCutError
from QCut.qpd.qpd_gates import (
    axym,
    axyp,
    ayzm,
    ayzp,
    azxm,
    azxp,
    bxy,
    byz,
    bzx,
    i_gate,
    ry_m,
    ry_p,
    s_gate,
    sdg_gate,
    sx_gate,
    sxdg_gate,
    x_gate,
    xmeas,
    y_gate,
    ymeas,
    z_gate,
    zmeas,
)

logger: logging.Logger = logging.getLogger(__name__)

#: Rows whose coefficient is smaller than this in absolute value are dropped.
DEFAULT_TOL: float = 1e-9

# Paper-convention operator to QCut primitive. _A0 is indexed X, Y, Z and the rest
# XY, YZ, ZX.
_A0 = (xmeas, ymeas, zmeas)
_AP = (axym, ayzm, azxp)
_AM = (axyp, ayzp, azxm)
_B0P = (sx_gate, ry_p, s_gate)
_B0M = (sxdg_gate, ry_m, sdg_gate)
_B = (bxy, byz, bzx)
_PAULI = (i_gate, x_gate, y_gate, z_gate)

_1Q_DECOMPOSER = OneQubitEulerDecomposer(basis="U")


def u_from_kak(a: float, b: float, c: float) -> np.ndarray:
    r"""Exponentiate the non-local part of a KAK decomposition.

    Returns the four complex coefficients :math:`u_\alpha` of

    .. math::

        \exp[i(a\, XX + b\, YY + c\, ZZ)]
        = \sum_{\alpha=0}^{3} u_\alpha \sigma_\alpha \otimes \sigma_\alpha

    The exponent's summation is a real symmetric matrix in the basis
    ``(II, XX, YY, ZZ)`` whose eigenvectors do not depend on ``a``, ``b`` or ``c``, so
    only its eigenvalues need exponentiating.
    """
    theta = np.asarray([a, b, c], dtype=float)
    eigvals = np.array(
        [
            -theta.sum(),
            -theta[0] + theta[1] + theta[2],
            -theta[1] + theta[2] + theta[0],
            -theta[2] + theta[0] + theta[1],
        ]
    )
    eigvecs = np.ones([1, 1]) / 2 - np.eye(4)
    return np.transpose(eigvecs) @ (np.exp(1j * eigvals) * eigvecs[:, 0])


def gamma(qpd: list[dict]) -> float:
    """Return the one-norm of a QPD's coefficients.

    The shot count needed scales as ``gamma ** 2``.
    """
    return float(sum(abs(term["c"]) for term in qpd))


def gamma_optimal(u: np.ndarray) -> float:
    r"""Return the provably optimal :math:`\gamma` for the gate described by ``u``.

    :math:`\gamma_{\mathrm{opt}} = 2(\sum_\alpha |u_\alpha|)^2 - 1`, from L. Schmitt,
    C. Piveteau and D. Sutter, "Cutting circuits with multiple two-qubit unitaries",
    Quantum 9, 1634 (2025), `arXiv:2312.11638 <https://arxiv.org/abs/2312.11638>`_.

    :func:`qpd_from_u` attains this for every named gate but runs above it for a generic
    two-qubit unitary. Use :func:`optimality_gap` to tell the two cases apart.
    """
    return float(2 * np.sum(np.abs(np.asarray(u))) ** 2 - 1)


def gamma_from_u(u: np.ndarray) -> float:
    """Return the one-norm of the Eq. (19) coefficients without building the table.

    Equal to ``gamma(qpd_from_u(u))`` but closed form, which is what makes it cheap
    enough to cost a whole cutting plan before committing to it.
    """
    u = np.asarray(u)
    total = float(np.sum(np.abs(u) ** 2))
    for alpha in range(4):
        for beta in range(alpha + 1, 4):
            z = u[alpha] * np.conj(u[beta])
            total += 4 * (abs(z.real) + abs(z.imag))
    return total


def gamma_for_gate(gate: Gate) -> float:
    """Return the gamma :func:`qpd_from_gate` would give, without building the table."""
    decomposition = TwoQubitWeylDecomposition(gate.to_matrix())
    return gamma_from_u(u_from_kak(decomposition.a, decomposition.b, decomposition.c))


def optimality_gap(u: np.ndarray) -> float:
    r"""Return ``gamma(qpd_from_u(u)) - gamma_optimal(u)`` without building the table.

    A non-zero value means some pair product has both a real and an imaginary part, and
    a Schmitt-Piveteau-Sutter decomposition would be cheaper for this gate.
    """
    return gamma_from_u(u) - gamma_optimal(u)


Row = tuple[float, QuantumCircuit, QuantumCircuit]


def _pair_products(u: np.ndarray) -> tuple[list[complex], list[complex]]:
    """Return the pair products of ``u``, ordered to index into _A0/_AP/_AM/_B.

    The first list holds the pairs involving the identity, the second the purely
    non-identity pairs in cyclic order.
    """
    with_identity = [u[0] * np.conj(u[k + 1]) for k in range(3)]
    cyclic = [
        u[1] * np.conj(u[2]),  # XY
        u[2] * np.conj(u[3]),  # YZ
        u[3] * np.conj(u[1]),  # ZX
    ]
    return with_identity, cyclic


def _signed(plus: QuantumCircuit, minus: QuantumCircuit):
    """Return ``(sign, primitive)`` for the two members of a Kraus pair."""
    return ((1, plus), (-1, minus))


def _pauli_rows(u: np.ndarray) -> list[Row]:
    """First line of Eq. (19), ``sum_a |u_a|^2 sigma_a^(x)2``."""
    return [
        (float(abs(u[alpha]) ** 2), _PAULI[alpha], _PAULI[alpha]) for alpha in range(4)
    ]


def _real_rows(with_identity: list[complex], cyclic: list[complex]) -> list[Row]:
    """Second line of Eq. (19), ``2 Re(u_a u*_b) (A^(x)2 - B^(x)2)``."""
    rows: list[Row] = []
    for k in range(3):
        re = float(with_identity[k].real)
        # One measurement covers both signs of A_{0 alpha}.
        rows.append((2 * re, _A0[k], _A0[k]))
        for s_0, p_0 in _signed(_B0P[k], _B0M[k]):
            for s_1, p_1 in _signed(_B0P[k], _B0M[k]):
                rows.append((-0.5 * re * s_0 * s_1, p_0, p_1))
    for k in range(3):
        re = float(cyclic[k].real)
        for s_0, p_0 in _signed(_AP[k], _AM[k]):
            for s_1, p_1 in _signed(_AP[k], _AM[k]):
                rows.append((0.5 * re * s_0 * s_1, p_0, p_1))
        rows.append((-2 * re, _B[k], _B[k]))
    return rows


def _imaginary_rows(with_identity: list[complex], cyclic: list[complex]) -> list[Row]:
    """Third line of Eq. (19), ``Im(u_a u*_b) (A (x) B + B (x) A)``.

    Asymmetric in the plus/minus label, which is why the ``_AP``/``_AM`` crossing
    matters here but not in :func:`_real_rows`.
    """
    rows: list[Row] = []
    for k in range(3):
        im = float(with_identity[k].imag)
        for sign, primitive in _signed(_B0P[k], _B0M[k]):
            rows.append((im * sign, _A0[k], primitive))
            rows.append((im * sign, primitive, _A0[k]))
    for k in range(3):
        im = float(cyclic[k].imag)
        for sign, primitive in _signed(_AP[k], _AM[k]):
            rows.append((im * sign, primitive, _B[k]))
            rows.append((im * sign, _B[k], primitive))
    return rows


def _rows(u: np.ndarray, tol: float) -> list[Row]:
    """Return the ``(coefficient, op_0, op_1)`` rows of Eq. (19) with ``|c| > tol``."""
    with_identity, cyclic = _pair_products(u)
    rows = (
        _pauli_rows(u)
        + _real_rows(with_identity, cyclic)
        + _imaginary_rows(with_identity, cyclic)
    )
    return [row for row in rows if abs(row[0]) > tol]


def _local_gate(matrix: np.ndarray) -> Gate | None:
    """Return a ``UGate`` implementing ``matrix``, or ``None`` if it is the identity."""
    matrix = np.asarray(matrix, dtype=complex)
    # Strip the global phase before comparing against the identity.
    idx = np.unravel_index(np.argmax(np.abs(matrix)), matrix.shape)
    phased = matrix / (matrix[idx] / abs(matrix[idx]))
    if np.abs(phased - np.eye(2)).max() < 1e-12:
        return None
    circuit = _1Q_DECOMPOSER(matrix)
    if len(circuit.data) != 1:  # pragma: no cover - defensive
        raise QCutError(
            f"expected a single-gate Euler decomposition, got {len(circuit.data)} gates"
        )
    return circuit.data[0].operation


_MERGED: dict[bytes, Gate | None] = {}


def _merged_gate(matrix: np.ndarray) -> Gate | None:
    """:func:`_local_gate`, memoised.

    A QPD's terms are built from the same handful of local products, and the Euler
    decomposition costs more than every other part of the merge together.
    """
    key = matrix.round(12).tobytes()
    if key not in _MERGED:
        _MERGED[key] = _local_gate(matrix)
    return _MERGED[key]


def _run_gate(run: list[Gate]) -> Gate | None:
    """The one gate a run of single-qubit gates comes to, or None for the identity."""
    if len(run) == 1:
        return run[0]
    matrix = np.eye(2)
    for gate in run:
        matrix = gate.to_matrix() @ matrix
    return _merged_gate(matrix)


def merge_locals(
    op: QuantumCircuit, locals_: list[tuple[Gate | None, Gate | None]]
) -> QuantumCircuit:
    """Return ``op`` wrapped in the per-qubit locals, one pair per qubit.

    Every run of single-qubit gates on a qubit becomes one gate, and a run that comes
    to the identity none at all. That costs a few matrix products per QPD, once, and
    every experiment circuit built from the result then carries fewer instructions.
    With no locals at all, ``op`` comes back as it is.
    """
    if all(pre is None and post is None for pre, post in locals_):
        return op
    out = QuantumCircuit(op.num_qubits, op.num_clbits, name=f"{op.name}'")
    runs: list[list[Gate]] = [[] if pre is None else [pre] for pre, _ in locals_]
    index = {bit: i for i, bit in enumerate(op.qubits)}
    index.update({bit: i for i, bit in enumerate(op.clbits)})

    def flush(qubit: int) -> None:
        run = runs[qubit]
        if not run:
            return
        merged = _run_gate(run)
        if merged is not None:
            out.append(merged, [qubit])
        run.clear()

    for instruction in op.data:
        qubits = [index[qubit] for qubit in instruction.qubits]
        if len(qubits) == 1 and isinstance(instruction.operation, Gate):
            runs[qubits[0]].append(instruction.operation)
            continue
        # anything else ends the runs on the qubits it touches, and gates on the other
        # qubits keep accumulating: they cannot be reordered past it either way
        for qubit in qubits:
            flush(qubit)
        out.append(
            instruction.operation,
            qubits,
            [index[clbit] for clbit in instruction.clbits],
        )

    for qubit, (_, post) in enumerate(locals_):
        if post is not None:
            runs[qubit].append(post)
        flush(qubit)
    return out


def qpd_from_u(u: np.ndarray, tol: float = DEFAULT_TOL) -> list[dict]:
    """Build a QPD from the non-local KAK coefficients ``u``.

    Args:
        u: the four complex coefficients from :func:`u_from_kak`.
        tol: rows whose coefficient is smaller than this are dropped.

    Returns:
        A list of ``{"op_0", "op_1", "c"}`` entries shaped like the hand-written tables
        in :mod:`QCut.qpd`, where ``op_0`` acts on the first qubit and ``op_1`` on the
        second. At most 58 rows, of which 34 survive for swap and 6 for cz.
    """
    u = np.asarray(u)
    if u.shape != (4,):
        raise QCutError(f"u must have shape (4,), got {u.shape}")
    return [
        {"op_0": op_0.copy(), "op_1": op_1.copy(), "c": coeff}
        for coeff, op_0, op_1 in _rows(u, tol)
    ]


def qpd_from_gate(gate: Gate, tol: float = DEFAULT_TOL) -> list[dict]:
    """Generate a QPD for any two-qubit gate.

    Args:
        gate: a two-qubit gate with all parameters bound.
        tol: rows whose coefficient is smaller than this are dropped.

    Returns:
        A list of ``{"op_0", "op_1", "c"}`` entries, as :func:`qpd_from_u` returns.

    Raises:
        QCutError: the gate is not two-qubit, or has no bound matrix.
    """
    if gate.num_qubits != 2:
        raise QCutError(
            f"QPD generation needs a two-qubit gate, got {gate.name} on "
            f"{gate.num_qubits} qubits"
        )
    try:
        matrix = gate.to_matrix()
    except Exception as exc:
        raise QCutError(
            f"could not get a matrix for gate '{gate.name}', which usually means it "
            "still has unbound parameters. A QPD's coefficients depend on the gate's "
            "numeric parameters, so they must be bound before the experiment circuits "
            "are generated. Call assign_parameters on the circuit or on the CutCircuit "
            "returned by get_locations_and_subcircuits, then retry."
        ) from exc

    decomposition = TwoQubitWeylDecomposition(matrix)
    u = u_from_kak(decomposition.a, decomposition.b, decomposition.c)
    qpd = qpd_from_u(u, tol)

    # Qiskit's tensor ordering puts qubit 0 on the right of the Weyl decomposition, so
    # op_0 takes the "r" locals and op_1 the "l" ones. The global phase cancels.
    pre_0, post_0 = _local_gate(decomposition.K2r), _local_gate(decomposition.K1r)
    pre_1, post_1 = _local_gate(decomposition.K2l), _local_gate(decomposition.K1l)
    if any(g is not None for g in (pre_0, post_0, pre_1, post_1)):
        for term in qpd:
            term["op_0"] = merge_locals(term["op_0"], [(pre_0, post_0)])
            term["op_1"] = merge_locals(term["op_1"], [(pre_1, post_1)])

    logger.debug(
        "generated %d-term QPD for '%s', gamma=%.6f (optimal %.6f, gap %.2e)",
        len(qpd),
        gate.name,
        gamma(qpd),
        gamma_optimal(u),
        optimality_gap(u),
    )
    return qpd
