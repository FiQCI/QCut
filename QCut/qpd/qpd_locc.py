"""Wire cut decomposition that uses classical communication between the partitions.

Cutting a wire without classical communication costs :math:`\\gamma = 4`, and cutting
``n`` of them costs :math:`4^n`. That is already optimal for local operations alone, so
cutting wires as a block buys nothing by itself. Letting the partitions exchange
classical information changes the picture completely, to

.. math::

    \\gamma = 2^{n+1} - 1

so one wire costs 3 rather than 4, two cost 7 rather than 16, and three cost 15 rather
than 64. The number of channels drops from :math:`8^n` to :math:`2^n + 1`.

The construction is Theorem 3 of H. Harada, K. Wada and N. Yamamoto, "Doubly optimal
parallel wire cutting without ancilla qubits", PRX Quantum 5, 040308 (2024),
`arXiv:2303.07340 <https://arxiv.org/abs/2303.07340>`_,

.. math::

    \\mathrm{Id}^{\\otimes n} = \\sum_{i} \\sum_{j}
        \\mathrm{Tr}[U_i |j\\rangle\\langle j| U_i^\\dagger (\\cdot)]\\,
        U_i |j\\rangle\\langle j| U_i^\\dagger
      - (2^n - 1) \\sum_j \\mathrm{Tr}[|j\\rangle\\langle j| (\\cdot)]\\, \\rho_j

where the :math:`2^n` unitaries :math:`U_i` rotate the computational basis into the
other mutually unbiased bases and :math:`\\rho_j` is the uniform mixture over the
computational basis states other than :math:`|j\\rangle`. In words, measure in one of
the :math:`2^n + 1` mutually unbiased bases and prepare the state you just measured,
except in the computational basis where you prepare any of the others and flip the sign.

It needs no ancilla qubits, unlike the teleportation-based protocol of Brenner et al.,
but it does need the prepared state to depend on the measured outcome. That is the
classical communication, and it is why these cuts are executed in two phases. See
``docs/theory/LOCC_wire_derivation.md``.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli

from QCut.errors.qcuterror import QCutError

logger: logging.Logger = logging.getLogger(__name__)

#: Largest block this module will build. Beyond it the overhead 2^(n+1) - 1 makes the
#: cut pointless long before the construction becomes expensive.
MAX_BLOCK: int = 6

#: Irreducible polynomial over GF(2) per degree, as a bit mask. Used to multiply in
#: GF(2^n), which is what generates the mutually unbiased bases.
_POLYNOMIALS: dict[int, int] = {
    1: 0b11,
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
}


def gamma_locc(width: int) -> float:
    """Return the overhead of cutting ``width`` wires with classical communication."""
    return float(2 ** (width + 1) - 1)


def gamma_local(width: int) -> float:
    """Return the overhead of cutting ``width`` wires with local operations only."""
    return float(4**width)


def _gf_multiply(left: int, right: int, width: int) -> int:
    """Multiply two elements of GF(2^width), each packed as an integer of bits."""
    polynomial = _POLYNOMIALS[width]
    result = 0
    for bit in range(width):
        if right >> bit & 1:
            result ^= left << bit
    for bit in range(2 * width - 2, width - 1, -1):
        if result >> bit & 1:
            result ^= polynomial << (bit - width)
    return result & ((1 << width) - 1)


def _trace(element: int, width: int) -> int:
    """Return the absolute trace of a GF(2^width) element, which is 0 or 1."""
    total, power = 0, element
    for _ in range(width):
        total ^= power
        power = _gf_multiply(power, power, width)
    if total not in (0, 1):  # pragma: no cover - defensive
        raise QCutError(f"the trace of {element} in GF(2^{width}) is not in GF(2)")
    return total


def _lagrangians(width: int) -> list[list[tuple[int, int]]]:
    r"""Return the ``2^n + 1`` maximal sets of commuting Pauli strings, as generators.

    Pauli strings modulo phase are vectors :math:`(a|b)` over :math:`\mathbb{F}_2^{2n}`,
    two of them commute when the symplectic form :math:`a\cdot b' + a'\cdot b` vanishes,
    and a maximal commuting set is a Lagrangian subspace. Identifying
    :math:`\mathbb{F}_2^n` with :math:`\mathrm{GF}(2^n)` gives them all at once as
    :math:`L_\alpha = \{(x \mid \alpha x)\}` for each field element :math:`\alpha`,
    plus :math:`L_\infty = \{(0 \mid y)\}`.

    The catch is that those subspaces are isotropic for the field's *trace* form
    :math:`\mathrm{Tr}(xy')`, not for the coordinate dot product that Pauli commutation
    actually uses. The two agree only up to :math:`n = 2`. Writing the :math:`Z`
    component in the trace-dual basis, so its coordinates are
    :math:`\mathrm{Tr}(e_l y)`, turns the trace form into the dot product and makes the
    construction correct for every width.

    Each entry is a list of ``width`` generators ``(a, b)``, and ``L_infinity`` comes
    last so the all-Z set is the one the decomposition treats separately.
    """
    basis = [1 << bit for bit in range(width)]

    def encode(x: int, y: int) -> tuple[int, int]:
        dual = sum(
            _trace(_gf_multiply(element, y, width), width) << bit
            for bit, element in enumerate(basis)
        )
        return x, dual

    subspaces = [
        [encode(element, _gf_multiply(alpha, element, width)) for element in basis]
        for alpha in range(1 << width)
    ]
    # That these form a symplectic spread is checked for every width up to MAX_BLOCK
    # in tests/test_locc_wire_cuts.py. A subspace that is not isotropic still
    # diagonalises to something, and would only show up as a decomposition that does
    # not sum to the identity channel.
    subspaces.append([encode(0, element) for element in basis])
    return subspaces


def _pauli_matrix(a: int, b: int, width: int) -> np.ndarray:
    """Return the Pauli string for ``(a|b)``, qubit ``k`` taking bits ``k`` of each."""
    bits = range(width)
    return Pauli(([b >> k & 1 for k in bits], [a >> k & 1 for k in bits])).to_matrix()


def mub_unitaries(width: int) -> list[np.ndarray]:
    r"""Return the ``2^n + 1`` unitaries that rotate the computational basis into MUBs.

    Column ``j`` of entry ``i`` is the ``j``-th vector of basis ``i``. The last entry is
    the identity, since the last Lagrangian is the all-Z one whose eigenbasis is the
    computational basis, and the decomposition treats that basis separately.

    The basis of a Lagrangian is pinned down by weighting its generators by distinct
    powers of two before diagonalising. Every sign pattern of the generators then gives
    a distinct eigenvalue, so the eigenvectors come out unique rather than merely
    spanning the right space.
    """
    if not 1 <= width <= MAX_BLOCK:
        raise QCutError(
            f"a block of {width} wire cuts is out of range, expected 1 to {MAX_BLOCK}"
        )
    dimension = 1 << width
    unitaries = []
    for generators in _lagrangians(width):
        combination = sum(
            (1 << index) * _pauli_matrix(a, b, width)
            for index, (a, b) in enumerate(generators)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(combination)
        if np.min(np.diff(eigenvalues)) < 1e-6:  # pragma: no cover - defensive
            raise QCutError(
                "the mutually unbiased basis construction did not separate its "
                "eigenvalues, so the basis is not unique"
            )
        unitaries.append(eigenvectors)
    unitaries[-1] = np.eye(dimension, dtype=complex)
    return unitaries


def _measure_op(width: int, unitary: np.ndarray, name: str) -> QuantumCircuit:
    """Return the circuit that measures in the basis whose vectors are the columns.

    Rotating by the adjoint first turns the basis into the computational one, so the
    recorded bits are the index of the basis vector that was found.
    """
    circuit = QuantumCircuit(width, width, name=name)
    if not np.allclose(unitary, np.eye(1 << width)):
        circuit.append(UnitaryGate(unitary.conj().T, label=f"{name}-rot"), range(width))
    for qubit in range(width):
        circuit.measure(qubit, qubit)
    return circuit


def _prepare_op(
    width: int, unitary: np.ndarray, index: tuple[int, ...], name: str
) -> QuantumCircuit:
    """Return the circuit preparing basis vector ``index`` of the given basis."""
    circuit = QuantumCircuit(width, name=name)
    for qubit, bit in enumerate(index):
        if bit:
            circuit.x(qubit)
    if not np.allclose(unitary, np.eye(1 << width)):
        circuit.append(UnitaryGate(unitary, label=f"{name}-rot"), range(width))
    return circuit


def _sign(label: tuple[int, ...]) -> int:
    """Return the sign a term carries to cancel two of QCut's sign conventions.

    QCut folds every qpd measurement bit into the estimator's sign as ``0 -> -1`` and
    ``1 -> +1``, which is what makes a Pauli measurement carry its own eigenvalue. Here
    the bits are a label saying which state to prepare, not a sign, so the term carries
    the inverse: ``-1`` per zero in the label.

    ``QCut.postprocess`` also multiplies every group by ``(-1)**(wire_cuts + 1)``, a
    convention the non-communicating table was written against. A block of
    ``len(label)`` wires adds that many to the exponent, which the term cancels too.

    Together that is ``(-1)**(zeros + len(label))``, or ``(-1)**sum(label)``.
    """
    return (-1) ** sum(label)


def locc_wire_qpd(width: int) -> list[dict]:
    r"""Build the decomposition for a block of ``width`` wire cuts.

    Args:
        width: how many wires are cut together.

    Returns:
        One entry per ``(channel, measured label, prepared state)``, in the usual
        ``{"op_0", "op_1", "c"}`` shape, with ``op_0`` measuring and ``op_1`` preparing
        on ``width`` qubits. Two extra keys say how to execute it. ``"channel"`` groups
        the entries sharing a measurement, so it only has to be run once, and
        ``"label"`` says which measured outcome the entry's preparation answers.

        The measured label is an outcome rather than a term of the decomposition, so its
        :math:`2^n` values share one channel's coefficient between them. The one-norm of
        the coefficients is therefore :math:`2^{n+1} - 1` as it should be, and the
        estimator recovers the factor by weighting each label by how often it came up.
    """
    unitaries = mub_unitaries(width)
    labels = list(itertools.product((0, 1), repeat=width))
    share = 1 / len(labels)
    terms: list[dict] = []

    for channel, unitary in enumerate(unitaries):
        computational = channel == len(unitaries) - 1
        measure = _measure_op(width, unitary, f"locc-meas-{channel}")
        for label in labels:
            # In the computational basis the prepared state is the uniform mixture
            # over every *other* basis state, with the sign flipped. Each of those is
            # its own entry, and they sum to the -(2^n - 1) of the theorem.
            prepared = (
                [other for other in labels if other != label]
                if computational
                else [label]
            )
            base = -1.0 if computational else 1.0
            for index in prepared:
                terms.append(
                    {
                        "op_0": measure,
                        "op_1": _prepare_op(
                            width,
                            unitary,
                            index,
                            f"locc-prep-{channel}-{''.join(map(str, index))}",
                        ),
                        "c": base * share * _sign(label),
                        "channel": channel,
                        "label": label,
                    }
                )

    logger.debug(
        "built the communicating decomposition for %d wire(s): %d term(s) over %d "
        "channel(s), gamma=%.4f against %.4f without communication",
        width,
        len(terms),
        len(unitaries),
        gamma_locc(width),
        gamma_local(width),
    )
    return terms


@dataclass(frozen=True)
class CommunicationPlan:
    """What execution needs to know to run communicating wire cuts.

    ``label_clbits`` says, per ``(group, observable setting, subcircuit)``, which
    classical bits of that circuit's qpd register hold each bundle's measured outcome,
    in the bundle's own cut order.

    ``labels`` says, per group, which measured outcome each bundle answers.

    ``keys`` identifies each group by everything except which label it answers. Groups
    sharing a key therefore run identical circuits everywhere apart from the preparing
    side, so their measuring runs only have to be executed once. It has to cover every
    bundle, not just the communicating ones, or a group would inherit another's
    operations for the cuts they do not share.

    ``prepare_subs`` names the subcircuits whose contents depend on a measured label.
    Those are the only ones that have to be run per group.

    ``waves`` says which wave each subcircuit belongs to and ``bundle_waves`` which wave
    each bundle's outcome becomes known in. A cut forces its measuring side to run
    before its preparing side, and those constraints chain. Cutting a circuit into A, B
    and C so that A feeds B and B feeds C needs three waves rather than two, because B's
    own measured outcome is what decides what C prepares.
    """

    label_clbits: dict[tuple[int, int, int], list]
    labels: list[dict]
    keys: list[tuple]
    prepare_subs: frozenset[int] = frozenset()
    waves: dict[int, int] = field(default_factory=dict)
    bundle_waves: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Total number of wires cut with communication."""
        return sum(bundle.size for bundle in self.labels[0]) if self.labels else 0

    @property
    def last_wave(self) -> int:
        """Index of the final wave, so zero means everything runs at once."""
        return max(self.waves.values(), default=0)
