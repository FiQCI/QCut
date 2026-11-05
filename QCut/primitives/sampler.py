"""Reconstructing the measured distribution of cut circuits, shaped like a sampler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.result import QuasiDistribution

from QCut.errors.qcuterror import QCutError
from QCut.execution.dispatch import DEFAULT_SHOTS
from QCut.job.cutresult import RawResult
from QCut.options import CutOptions
from QCut.post_process.probabilities import QuasiProbabilities, estimate_probabilities
from QCut.primitives.base import (
    CutRunEstimate,
    _as_list,
    _check_circuit,
    _CutPrimitive,
)
from QCut.primitives.job import QCutPrimitiveJob


@dataclass(frozen=True)
class CutQuasiSamplerEntry:
    """The reconstructed distribution of one input circuit."""

    #: The distribution over the measured bits, built lazily.
    quasi_probabilities: QuasiProbabilities
    #: The raw counts it was reconstructed from.
    raw: RawResult

    def quasi_dist(self, top: int | None = None) -> QuasiDistribution:
        """The distribution as a qiskit :class:`~qiskit.result.QuasiDistribution`.

        Args:
            top (int): keep only this many of the most likely outcomes. ``None``, the
                default, keeps all ``2**n`` of them, which for many measured bits is
                expensive; :attr:`quasi_probabilities` answers without expanding.
        """
        return QuasiDistribution(
            self.quasi_probabilities.quasi_probabilities(top=top),
            shots=self.quasi_probabilities.shots,
        )


class CutQuasiSamplerResult(Sequence):
    """What a :class:`CutQuasiSampler` run gives back: one entry per input circuit."""

    def __init__(self, entries: list[CutQuasiSamplerEntry]) -> None:
        self._entries = list(entries)

    def __getitem__(self, index):
        """The entry of one input circuit."""
        return self._entries[index]

    def __len__(self) -> int:
        """How many circuits the run took."""
        return len(self._entries)

    def quasi_probabilities(self) -> list[QuasiProbabilities]:
        """Every circuit's distribution, in the order the circuits were given."""
        return [entry.quasi_probabilities for entry in self._entries]

    def quasi_dists(self, top: int | None = None) -> list[QuasiDistribution]:
        """Every circuit's distribution as a qiskit ``QuasiDistribution``.

        Args:
            top (int): as in :meth:`CutQuasiSamplerEntry.quasi_dist`.
        """
        return [entry.quasi_dist(top) for entry in self._entries]

    def __repr__(self) -> str:
        """Represent without expanding any distribution."""
        return f"CutQuasiSamplerResult({self.quasi_probabilities()})"


class CutQuasiSamplerJob(QCutPrimitiveJob):
    """A :class:`CutQuasiSampler` run. :meth:`result` gives a
    :class:`CutQuasiSamplerResult`."""

    def result(self, timeout: float | None = None) -> CutQuasiSamplerResult:
        """Wait for every circuit and return their distributions.

        Args:
            timeout (float): seconds to wait for the whole run before giving up, or
                ``None`` to wait as long as it takes.
        """
        return super().result(timeout)

    def quasi_probabilities(
        self, index: int | None = None
    ) -> list[QuasiProbabilities] | QuasiProbabilities:
        """Wait for the run and return the distributions.

        Args:
            index (int): the input circuit to return it for, or ``None`` for every
                circuit's, in the order the circuits were given.
        """
        result = self.result()
        if index is None:
            return result.quasi_probabilities()
        return result[index].quasi_probabilities

    def _build(self, raws: list[RawResult]) -> CutQuasiSamplerResult:
        return CutQuasiSamplerResult(
            [CutQuasiSamplerEntry(estimate_probabilities(raw), raw) for raw in raws]
        )


def _measured_qubits(circuit: QuantumCircuit, index: int) -> list[int]:
    """The qubit each classical bit is measured from, in classical bit order.

    That order is what puts the reconstructed bitstrings in qiskit's convention, with
    classical bit zero rightmost.

    Raises:
        QCutError: the circuit measures nothing, measures into a classical bit twice,
            or leaves a classical bit unmeasured.
    """
    measured: dict[int, int] = {}
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        clbit = circuit.find_bit(instruction.clbits[0]).index
        if clbit in measured:
            raise QCutError(
                f"circuit {index} measures into classical bit {clbit} more than once"
            )
        measured[clbit] = circuit.find_bit(instruction.qubits[0]).index
    if not measured:
        raise QCutError(
            f"circuit {index} measures nothing. CutQuasiSampler reconstructs the "
            "distribution of the measured qubits, so end the circuit in measurements."
        )
    unmeasured = [clbit for clbit in range(circuit.num_clbits) if clbit not in measured]
    if unmeasured:
        raise QCutError(
            f"circuit {index} never measures into classical bit(s) {unmeasured}, so "
            "their place in the outcome cannot be filled"
        )
    return [measured[clbit] for clbit in range(circuit.num_clbits)]


class CutQuasiSampler(_CutPrimitive):
    """Quasi-probability distributions of cut circuits, run on any backend or sampler.

    Shaped like FiQCI-EMS's sampler, but what comes back is reconstructed rather than
    counted: a quasi-probability distribution over the measured bits, which can dip
    below zero by the estimate's noise. The qubits are read off the circuit's final
    measurements, and its bitstrings follow qiskit's convention, classical bit zero
    rightmost.

    Each circuit has to carry its cut markers, placed with :func:`QCut.cut` and
    :func:`QCut.cutGate` or found with :func:`QCut.find_cuts`.

    Args:
        backend: the backend or V2 sampler to run on. A sampler is not transpiled for,
            so it must take the circuits as built.
        options (CutOptions): how to cut, for every run that is not given its own.
            Defaults to :data:`QCut.DEFAULT_OPTIONS`.
    """

    def __init__(self, backend, options: CutOptions | None = None) -> None:
        super().__init__(backend, options)

    def run(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit],
        shots: int = DEFAULT_SHOTS,
        max_batch_size: int = 100,
        options: CutOptions | None = None,
        **run_options,
    ) -> CutQuasiSamplerJob:
        """Submit the circuits and return without waiting for their results.

        Args:
            circuits: a circuit or a list of them, each carrying its cut markers and
                ending in the measurements whose distribution is wanted. Every
                classical bit must be measured into exactly once.
            shots (int): shots per experiment circuit, as in
                :func:`QCut.submit_experiments`.
            max_batch_size (int): the most circuits submitted in one ``run`` call. Each
                input circuit is submitted in batches of its own.
            options (CutOptions): how to cut, for this run only.
            **run_options: passed on to every ``run`` call on the backend.

        Returns:
            CutQuasiSamplerJob: a handle on every circuit's experiment, whose
            :meth:`~CutQuasiSamplerJob.result` gives the distributions.

        Raises:
            QCutError: a circuit has no cut markers, measures before its end, or its
                final measurements do not fill its classical bits once each.
        """
        experiments = self._experiments(circuits, options)
        return CutQuasiSamplerJob(
            self._submit(experiments, shots, max_batch_size, run_options)
        )

    def estimate(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit],
        shots: int = DEFAULT_SHOTS,
        max_batch_size: int = 100,
        options: CutOptions | None = None,
    ) -> CutRunEstimate:
        """Work out what :meth:`run` will cost with the same arguments, submitting
        nothing.

        Returns:
            CutRunEstimate: the jobs, circuits and shots of the whole run, and indexed
            by circuit, each circuit's own share.
        """
        experiments = self._experiments(circuits, options)
        return self._estimate(experiments, shots, max_batch_size)

    def _experiments(self, circuits, options):
        listed = _as_list(circuits)
        qubits = []
        for index, circuit in enumerate(listed):
            _check_circuit(circuit, index)
            qubits.append(_measured_qubits(circuit, index))
        return [
            self._experiment(circuit, options, qubits=measured)
            for circuit, measured in zip(listed, qubits)
        ]
