"""Estimating expectation values of cut circuits, shaped like a qiskit estimator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from QCut.execution.dispatch import DEFAULT_SHOTS
from QCut.job.cutresult import RawResult
from QCut.options import CutOptions
from QCut.post_process.postprocess import estimate_expectation_values
from QCut.primitives.base import (
    CutRunEstimate,
    _as_list,
    _check_circuit,
    _CutPrimitive,
)
from QCut.primitives.job import QCutPrimitiveJob
from QCut.utils.observables import ObservablesLike


@dataclass(frozen=True)
class CutEstimatorEntry:
    """The expectation values of one input circuit."""

    #: One value per observable, shaped like the observables given for the circuit.
    #: A single observable gives a zero-dimensional array.
    expectation_values: np.ndarray
    #: The raw counts the values were estimated from.
    raw: RawResult


class CutEstimatorResult(Sequence):
    """What a :class:`CutEstimator` run gives back: one entry per input circuit."""

    def __init__(self, entries: list[CutEstimatorEntry]) -> None:
        self._entries = list(entries)

    def __getitem__(self, index):
        """The entry of one input circuit."""
        return self._entries[index]

    def __len__(self) -> int:
        """How many circuits the run took."""
        return len(self._entries)

    def expectation_values(self) -> list[np.ndarray]:
        """Every circuit's expectation values, in the order the circuits were given."""
        return [entry.expectation_values for entry in self._entries]

    def __repr__(self) -> str:
        """Represent as the values."""
        return f"CutEstimatorResult({self.expectation_values()})"


class CutEstimatorJob(QCutPrimitiveJob):
    """A :class:`CutEstimator` run, whose :meth:`result` is a
    :class:`CutEstimatorResult`."""

    def result(self, timeout: float | None = None) -> CutEstimatorResult:
        """Wait for every circuit and return their expectation values.

        Args:
            timeout (float): seconds to wait for the whole run before giving up, or
                ``None`` to wait as long as it takes.
        """
        return super().result(timeout)

    def expectation_values(
        self, index: int | None = None
    ) -> list[np.ndarray] | np.ndarray:
        """Wait for the run and return the expectation values.

        Args:
            index (int): the input circuit to return them for, or ``None`` for every
                circuit's, in the order the circuits were given.
        """
        result = self.result()
        if index is None:
            return result.expectation_values()
        return result[index].expectation_values

    def _build(self, raws: list[RawResult]) -> CutEstimatorResult:
        return CutEstimatorResult(
            [CutEstimatorEntry(estimate_expectation_values(raw), raw) for raw in raws]
        )


def _pair_observables(circuits, observables, count: int) -> list:
    """One observables entry per circuit, broadcast the way FiQCI's estimator does it.

    One circuit takes the observables exactly as given. With a list of circuits, a list
    or tuple of observables gives one entry per circuit, and anything else is one
    observable used for every circuit.

    Raises:
        ValueError: the list of observables does not have one entry per circuit.
    """
    if isinstance(circuits, QuantumCircuit):
        return [observables]
    if isinstance(observables, (list, tuple)):
        if len(observables) != count:
            raise ValueError(
                f"{count} circuit(s) were given but {len(observables)} observable "
                "entries. With a list of circuits, give one entry per circuit, each an "
                "observable or a list of them, or a single observable for all of them."
            )
        return list(observables)
    return [observables] * count


class CutEstimator(_CutPrimitive):
    """Expectation values of cut circuits, run on any backend or sampler.

    Shaped like FiQCI-EMS's estimator: made once for a backend, then :meth:`run` with
    the circuits and the observables to estimate on them. Each circuit has to carry its
    cut markers, placed with :func:`QCut.cut` and :func:`QCut.cutGate` or found with
    :func:`QCut.find_cuts`. Every circuit is cut, transpiled for the backend when it has
    a coupling map, and submitted before :meth:`run` returns.

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
        observables: ObservablesLike | list[ObservablesLike],
        shots: int = DEFAULT_SHOTS,
        max_batch_size: int = 100,
        options: CutOptions | None = None,
        **run_options,
    ) -> CutEstimatorJob:
        """Submit the circuits and return without waiting for their results.

        Args:
            circuits: a circuit or a list of them, each carrying its cut markers.
                Final measurements are ignored.
            observables: what to estimate, taken as qiskit's estimator takes them --
                a label, a ``Pauli``, a ``SparsePauliOp``, a ``SparseObservable``, a
                ``{label: coefficient}`` mapping, or a nested sequence of those. One
                circuit takes them as given. With a list of circuits, give a list with
                one entry per circuit, or one observable for every circuit.
            shots (int): shots per experiment circuit, as in
                :func:`QCut.submit_experiments`.
            max_batch_size (int): the most circuits submitted in one ``run`` call. Each
                input circuit is submitted in batches of its own.
            options (CutOptions): how to cut, for this run only.
            **run_options: passed on to every ``run`` call on the backend.

        Returns:
            CutEstimatorJob: a handle on every circuit's experiment, whose
            :meth:`~CutEstimatorJob.result` gives the expectation values.

        Raises:
            QCutError: a circuit has no cut markers, or measures before its end.
            ValueError: the observables do not have one entry per circuit, or do not
                fit their circuit.
        """
        experiments = self._experiments(circuits, observables, options)
        return CutEstimatorJob(
            self._submit(experiments, shots, max_batch_size, run_options)
        )

    def estimate(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit],
        observables: ObservablesLike | list[ObservablesLike],
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
        experiments = self._experiments(circuits, observables, options)
        return self._estimate(experiments, shots, max_batch_size)

    def _experiments(self, circuits, observables, options):
        listed = _as_list(circuits)
        paired = _pair_observables(circuits, observables, len(listed))
        for index, circuit in enumerate(listed):
            _check_circuit(circuit, index)
        return [
            self._experiment(circuit, options, observables=entry)
            for circuit, entry in zip(listed, paired)
        ]
