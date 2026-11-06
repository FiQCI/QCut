"""Reconstructing the measured distribution of cut circuits, shaped like a sampler."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.result import QuasiDistribution

from QCut.errors.qcuterror import QCutError
from QCut.execution.dispatch import DEFAULT_SHOTS
from QCut.execution.experiments import get_experiment_circuits
from QCut.job.cutresult import RawResult
from QCut.model.cutcircuit import CutCircuit
from QCut.post_process.probabilities import QuasiProbabilities, estimate_probabilities
from QCut.primitives.base import CutRunEstimate, _CutPrimitive, _CutResult
from QCut.primitives.job import QCutPrimitiveJob


@dataclass(frozen=True)
class CutQuasiSamplerEntry:
    """The reconstructed distribution of one input circuit."""

    #: The distribution over the measured bits, built lazily.
    quasi_probabilities: QuasiProbabilities
    #: The raw counts it was reconstructed from.
    raw: RawResult

    @property
    def standard_error(self) -> float | None:
        """An upper bound on each quasi-probability's shot noise, the same for all.

        See :attr:`QCut.QuasiProbabilities.standard_error`.
        """
        return self.quasi_probabilities.standard_error

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


class CutQuasiSamplerResult(_CutResult):
    """What a :class:`CutQuasiSampler` run gives back: one entry per input circuit."""

    def quasi_probabilities(self) -> list[QuasiProbabilities]:
        """Every circuit's distribution, in the order the circuits were given."""
        return [entry.quasi_probabilities for entry in self._entries]

    def quasi_dists(self, top: int | None = None) -> list[QuasiDistribution]:
        """Every circuit's distribution as a qiskit ``QuasiDistribution``.

        Args:
            top (int): as in :meth:`CutQuasiSamplerEntry.quasi_dist`.
        """
        return [entry.quasi_dist(top) for entry in self._entries]

    def standard_errors(self) -> list[float | None]:
        """Every circuit's standard error, in the order the circuits were given."""
        return [entry.standard_error for entry in self._entries]

    def __repr__(self) -> str:
        """Represent without expanding any distribution."""
        return f"CutQuasiSamplerResult({self.quasi_probabilities()})"


class CutQuasiSamplerJob(QCutPrimitiveJob[CutQuasiSamplerResult]):
    """A :class:`CutQuasiSampler` run. :meth:`result` gives a
    :class:`CutQuasiSamplerResult`."""

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

    def standard_errors(
        self, index: int | None = None
    ) -> list[float | None] | float | None:
        """Wait for the run and return the standard errors.

        Args:
            index (int): the input circuit to return it for, or ``None`` for every
                circuit's, in the order the circuits were given.
        """
        result = self.result()
        if index is None:
            return result.standard_errors()
        return result[index].standard_error

    def _build(self, raws: list[RawResult]) -> CutQuasiSamplerResult:
        return CutQuasiSamplerResult(
            [CutQuasiSamplerEntry(estimate_probabilities(raw), raw) for raw in raws]
        )


def _measured_qubits(cut_circuit: CutCircuit, index: int) -> list[int]:
    """The qubit each classical bit was measured from, in classical bit order.

    Read from the final measurements the uncut circuit had, which splitting strips and
    the cut circuit remembers. Classical bit order is what puts the reconstructed
    bitstrings in qiskit's convention, with classical bit zero rightmost.

    Raises:
        QCutError: the uncut circuit measured nothing at its end, measured into a
            classical bit twice, or left a classical bit unmeasured.
    """
    measured: dict[int, int] = {}
    for qubit, clbit in cut_circuit.final_measurements:
        if clbit in measured:
            raise QCutError(
                f"circuit {index} measures into classical bit {clbit} more than once"
            )
        measured[clbit] = qubit
    if not measured:
        raise QCutError(
            f"circuit {index} was split from a circuit with no final measurements. "
            "CutQuasiSampler reconstructs the distribution of the measured qubits, so "
            "end the circuit in measurements before cutting it."
        )
    num_clbits = cut_circuit.uncut_num_clbits
    unmeasured = [clbit for clbit in range(num_clbits) if clbit not in measured]
    if unmeasured:
        raise QCutError(
            f"circuit {index} never measures into classical bit(s) {unmeasured}, so "
            "their place in the outcome cannot be filled"
        )
    return [measured[clbit] for clbit in range(num_clbits)]


class CutQuasiSampler(_CutPrimitive):
    """Quasi-probability distributions of cut circuits, run on any backend or sampler.

    Shaped like FiQCI-EMS's sampler, but what comes back is reconstructed rather than
    counted: a quasi-probability distribution over the measured bits, which can dip
    below zero by the estimate's noise. The qubits are the ones the circuit measured at
    its end before it was cut, and the bitstrings follow qiskit's convention, classical
    bit zero rightmost.

    A cut circuit comes from :func:`QCut.get_locations_and_subcircuits` or
    :func:`QCut.find_cuts`, and is run with the options it was made with. The sampler
    does not transpile: for a backend with a coupling map, transpile each cut circuit
    for that same backend with :func:`QCut.transpile_subcircuits` first.

    Args:
        backend: the backend or V2 sampler to run on. A sampler, like a backend
            without a coupling map, takes the circuits as they are.
    """

    def run(
        self,
        circuits: CutCircuit | list[CutCircuit],
        shots: int = DEFAULT_SHOTS,
        max_batch_size: int = 100,
        parameter_values: dict | list[dict] | None = None,
        **run_options,
    ) -> CutQuasiSamplerJob:
        """Submit the circuits and return without waiting for their results.

        Args:
            circuits: a cut circuit or a list of them, each split from a circuit that
                ended in the measurements whose distribution is wanted, measuring into
                every classical bit exactly once.
            shots (int): shots per experiment circuit, as in
                :func:`QCut.submit_experiments`.
            max_batch_size (int): the most circuits submitted in one ``run`` call. Each
                input circuit is submitted in batches of its own.
            parameter_values: ``{parameter or name: value}`` to bind into the circuit.
                With a list of circuits, a list with one entry per circuit, or one
                for every circuit. For a sweep, list the same circuit once per set of
                values.
            **run_options: passed on to every ``run`` call on the backend.

        Returns:
            CutQuasiSamplerJob: a handle on every circuit's experiment, whose
            :meth:`~CutQuasiSamplerJob.result` gives the distributions.

        Raises:
            QCutError: something other than a cut circuit was given, one is not
                transpiled for this backend although it has a coupling map, has
                parameters left unbound, or its final measurements do not fill its
                classical bits once each.
            ValueError: the parameter values do not have one entry per circuit.
        """
        experiments = self._experiments(circuits, parameter_values)
        return CutQuasiSamplerJob(
            self._submit(experiments, shots, max_batch_size, run_options)
        )

    def estimate(
        self,
        circuits: CutCircuit | list[CutCircuit],
        shots: int = DEFAULT_SHOTS,
        max_batch_size: int = 100,
        parameter_values: dict | list[dict] | None = None,
    ) -> CutRunEstimate:
        """Work out what :meth:`run` will cost with the same arguments, submitting
        nothing.

        Returns:
            CutRunEstimate: the jobs, circuits and shots of the whole run, and indexed
            by circuit, each circuit's own share.
        """
        experiments = self._experiments(circuits, parameter_values)
        return self._estimate(experiments, shots, max_batch_size)

    def _experiments(self, circuits, parameter_values):
        bound = self._bound(circuits, parameter_values)
        qubits = [_measured_qubits(cut, index) for index, cut in enumerate(bound)]
        return [
            get_experiment_circuits(cut_circuit, qubits=measured)
            for cut_circuit, measured in zip(bound, qubits)
        ]
