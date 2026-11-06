"""Joint quasiprobability decomposition for several parallel two-qubit rotation gates.

Cutting gates one at a time costs the product of their sampling overheads. For gates
that are locally equivalent to a rotation about a single Cartan axis this is not
optimal. Cutting ``n`` of them together costs

.. math::

    \\gamma = 2 \\prod_s (1 + |\\sin\\theta_s|) - 1

against :math:`\\prod_s (1 + 2|\\sin\\theta_s|)` for separate cuts, so two CNOT gates
cost 7 rather than 9 and three cost 15 rather than 27. The number of terms falls too,
from :math:`6^n` to :math:`O(4^n)`.

The construction is Eq. (C16) of C. Ufrecht et al., "Optimal joint cutting of two-qubit
rotation gates", Phys. Rev. A 109, 052440 (2024),
`arXiv:2312.09679 <https://arxiv.org/abs/2312.09679>`_. It needs no ancilla qubits and
no classical communication between the partitions. See
``docs/theory/Joint_rotation_derivation.md``.

The gates have to be parallel, meaning schedulable in one time slice, because the
operations act jointly on subsets of one partition's qubits rather than on single
qubits.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.synthesis import TwoQubitWeylDecomposition

from QCut.errors.qcuterror import QCutError
from QCut.qpd.qpd_generate import DEFAULT_TOL, _local_gate, merge_locals

logger: logging.Logger = logging.getLogger(__name__)

#: A Cartan coordinate below this counts as zero when testing for a single axis.
AXIS_TOL: float = 1e-8


def gamma_joint(thetas: list[float]) -> float:
    """Return the joint sampling overhead for rotation angles ``thetas``."""
    return float(2 * np.prod([1 + abs(np.sin(theta)) for theta in thetas]) - 1)


def gamma_separate(thetas: list[float]) -> float:
    """Return the sampling overhead of cutting each angle in ``thetas`` on its own."""
    return float(np.prod([1 + 2 * abs(np.sin(theta)) for theta in thetas]))


def _bit_value(bits: tuple[int, ...]) -> int:
    """Read a bit tuple as an integer, first element most significant."""
    return int("".join(str(bit) for bit in bits), 2)


def _z_gates(circuit: QuantumCircuit, bits: tuple[int, ...]) -> None:
    """Apply ``Z`` to every qubit whose bit is set."""
    for qubit, bit in enumerate(bits):
        if bit:
            circuit.z(qubit)


def _ladder(circuit: QuantumCircuit, subset: list[int]) -> None:
    """Map ``Z`` on ``subset[0]`` to ``Z`` over all of ``subset``, and back.

    Every gate shares the same target so they commute, which also makes the ladder its
    own inverse. Applying it before and after an operation on ``subset[0]`` therefore
    conjugates that operation into the parity frame.
    """
    for qubit in subset[1:]:
        circuit.cx(qubit, subset[0])


def _rotation_op(width: int, subset: list[int], sign: int) -> QuantumCircuit:
    r"""Return the channel of :math:`(I \mp i Z^{\otimes T})/\sqrt 2`.

    A multi-qubit ``Z`` rotation by :math:`\pm\pi/2`, which is a single ``rz`` inside a
    ladder over ``subset``.
    """
    name = "R" + ("+" if sign > 0 else "-") + "_" + "".join(str(q) for q in subset)
    circuit = QuantumCircuit(width, name=name)
    _ladder(circuit, subset)
    circuit.rz(sign * np.pi / 2, subset[0])
    _ladder(circuit, subset)
    return circuit


def _parity_op(width: int, subset: list[int]) -> QuantumCircuit:
    r"""Return the parity measurement channel :math:`P^0 - P^1` over ``subset``.

    The measurement outcome carries the sign, so this costs no sampling overhead of its
    own. The ladder has to be undone after the measurement because it entangled the rest
    of the subset onto ``subset[0]``.
    """
    name = "P_" + "".join(str(q) for q in subset)
    circuit = QuantumCircuit(width, 1, name=name)
    _ladder(circuit, subset)
    circuit.measure(subset[0], 0)
    _ladder(circuit, subset)
    return circuit


def _diagonal_terms(
    width: int, coefficients: dict[tuple[int, ...], float]
) -> list[dict]:
    """Return the ``i == j`` terms, one per bit pattern, with no measurements."""
    terms = []
    for bits, coefficient in coefficients.items():
        ops = []
        for _ in range(2):
            circuit = QuantumCircuit(width, name="Z_" + "".join(map(str, bits)))
            _z_gates(circuit, bits)
            ops.append(circuit)
        terms.append({"op_0": ops[0], "op_1": ops[1], "c": coefficient**2})
    return terms


def _prefixed(prefix: tuple[int, ...], op: QuantumCircuit) -> QuantumCircuit:
    """Return ``op`` with a layer of ``Z`` gates in front of it.

    ``Z`` commutes with everything the joint operations are built from, so the layer
    could equally go after. It goes in front so a term reads the same way round as
    Eq. (C16).
    """
    out = QuantumCircuit(op.num_qubits, op.num_clbits, name=op.name)
    _z_gates(out, prefix)
    out.compose(
        op, qubits=range(op.num_qubits), clbits=range(op.num_clbits), inplace=True
    )
    return out


def _off_diagonal_terms(
    width: int, i: tuple[int, ...], j: tuple[int, ...], prefactor: float
) -> list[dict]:
    """Return the terms of one ``i > j`` pair of Eq. (C16)."""
    subset = [index for index, (a, b) in enumerate(zip(i, j)) if a != b]
    nu = sum(b - a for a, b in zip(i, j))
    rotations = {sign: _rotation_op(width, subset, sign) for sign in (1, -1)}
    parity = _parity_op(width, subset)

    pairs: list[tuple[float, QuantumCircuit, QuantumCircuit]] = []
    if nu % 2 == 0:
        sign = (-1) ** (nu // 2)
        pairs.append((sign, parity, parity))
        for left in (1, -1):
            for right in (1, -1):
                pairs.append(
                    (-sign * left * right / 4, rotations[left], rotations[right])
                )
    else:
        sign = (-1) ** ((nu - 1) // 2)
        for which in (1, -1):
            pairs.append((sign * which / 2, rotations[which], parity))
            pairs.append((sign * which / 2, parity, rotations[which]))

    return [
        {
            "op_0": _prefixed(i, op_0),
            "op_1": _prefixed(i, op_1),
            "c": prefactor * coefficient,
        }
        for coefficient, op_0, op_1 in pairs
    ]


def joint_rotation_qpd(thetas: list[float], tol: float = DEFAULT_TOL) -> list[dict]:
    r"""Build the joint QPD for ``n`` parallel two-qubit rotation gates.

    Args:
        thetas: the rotation angle of each gate, in the convention
            :math:`R_{zz}(\theta) = \cos(\theta/2) I - i \sin(\theta/2) Z \otimes Z`.
        tol: terms whose coefficient is smaller than this are dropped.

    Returns:
        A list of ``{"op_0", "op_1", "c"}`` entries shaped like the single-gate tables,
        except that ``op_0`` and ``op_1`` act on ``len(thetas)`` qubits. Qubit ``s`` of
        each side belongs to gate ``s``.
    """
    width = len(thetas)
    if width < 1:
        raise QCutError("joint_rotation_qpd needs at least one angle")

    patterns = list(itertools.product((0, 1), repeat=width))
    coefficients = {
        bits: float(
            np.prod(
                [
                    np.cos(theta / 2) if bit == 0 else np.sin(theta / 2)
                    for theta, bit in zip(thetas, bits)
                ]
            )
        )
        for bits in patterns
    }

    terms = _diagonal_terms(width, coefficients)
    for i, j in itertools.combinations(patterns, 2):
        if _bit_value(i) < _bit_value(j):
            i, j = j, i
        prefactor = 2 * coefficients[i] * coefficients[j]
        if abs(prefactor) <= tol:
            continue
        terms.extend(_off_diagonal_terms(width, i, j, prefactor))

    terms = [term for term in terms if abs(term["c"]) > tol]
    logger.debug(
        "joint QPD for %d gate(s): %d terms, gamma=%.6f against %.6f separately",
        width,
        len(terms),
        gamma_joint(thetas),
        gamma_separate(thetas),
    )
    return terms


def single_axis_frame(
    gate: Gate,
) -> tuple[float, list[tuple[Gate | None, Gate | None]]] | None:
    r"""Put a two-qubit gate in the frame the joint decomposition needs.

    A gate qualifies when its Cartan coordinates have a single non-zero entry, which
    holds for ``rzz``, ``rxx``, ``ryy``, ``rzx``, the controlled rotations, ``cp``,
    ``cx`` and ``cz``. The Weyl chamber folds the coordinate into
    :math:`[0, \pi/4]`, which changes the angle but not the gate, so the returned
    ``theta`` is the folded one and the locals absorb the difference.

    Returns:
        ``(theta, locals)`` where ``locals[side]`` is a ``(pre, post)`` pair of
        single-qubit gates or ``None``, or ``None`` if the gate is not single-axis.
        Side 0 takes the qubit-0 locals, matching ``op_0`` in the single-gate tables.
    """
    if gate.num_qubits != 2:
        return None
    try:
        matrix = gate.to_matrix()
    except Exception:
        return None

    decomposition = TwoQubitWeylDecomposition(matrix)
    if abs(decomposition.b) > AXIS_TOL or abs(decomposition.c) > AXIS_TOL:
        return None

    # The Weyl form is exp(i a XX). Conjugating by H on both qubits turns that into
    # exp(i a ZZ) = R_zz(-2a), which is what Eq. (C16) decomposes.
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    locals_ = [
        (
            _local_gate(hadamard @ pre),
            _local_gate(post @ hadamard),
        )
        for pre, post in (
            (decomposition.K2r, decomposition.K1r),
            (decomposition.K2l, decomposition.K1l),
        )
    ]
    return -2 * decomposition.a, locals_


def single_axis_theta(gate: Gate) -> float:
    """The folded rotation angle of a single-axis gate.

    Only for gates a plan has already accepted as single-axis, so one that is not is a
    bug rather than a case to handle.
    """
    frame = single_axis_frame(gate)
    if frame is None:
        raise QCutError(
            f"gate '{gate.name}' is not equivalent to a single-axis rotation, "
            "so it has no joint decomposition"
        )
    return frame[0]


def joint_rotation_qpd_from_gates(
    gates: list[Gate], flipped: list[bool] | None = None, tol: float = DEFAULT_TOL
) -> list[dict] | None:
    """Build the joint QPD for a list of gates, or None if one is not single-axis.

    Args:
        gates: the gates being cut together, in bundle qubit order.
        flipped: per gate, whether its two qubits are the other way round relative to
            the bundle's sides. A single-axis gate is symmetric about its interaction
            axis, so flipping one only swaps its own local unitaries.
        tol: terms whose coefficient is smaller than this are dropped.

    Returns:
        A list of ``{"op_0", "op_1", "c"}`` entries whose operations act on
        ``len(gates)`` qubits.
    """
    flipped = [False] * len(gates) if flipped is None else list(flipped)
    frames = []
    for gate in gates:
        frame = single_axis_frame(gate)
        if frame is None:
            return None
        frames.append(frame)

    thetas = [frame[0] for frame in frames]
    per_side: list[list[tuple]] = [[], []]
    for frame, flip in zip(frames, flipped):
        locals_ = list(reversed(frame[1])) if flip else list(frame[1])
        per_side[0].append(locals_[0])
        per_side[1].append(locals_[1])

    # Gate s owns qubit s of each side. Its locals do not commute with the joint
    # operation, so merge_locals keeps them around it: pre-locals, joint, post-locals.
    terms = joint_rotation_qpd(thetas, tol)
    return [
        {
            "op_0": merge_locals(term["op_0"], per_side[0]),
            "op_1": merge_locals(term["op_1"], per_side[1]),
            "c": term["c"],
        }
        for term in terms
    ]
