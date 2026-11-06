"""Estimating expectation values of cut circuits, shaped like a qiskit estimator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from QCut.execution.dispatch import DEFAULT_SHOTS
from QCut.job.cutresult import RawResult
from QCut.model.cutcircuit import CutCircuit
from QCut.post_process.postprocess import (
    estimate_expectation_values,
    estimate_standard_errors,
)
from QCut.primitives.base import CutRunEstimate, _as_list, _CutPrimitive, _pair
from QCut.primitives.job import QCutPrimitiveJob
from QCut.utils.observables import ObservablesLike


@dataclass(frozen=True)
class CutEstimatorEntry:
    """The expectation values of one input circuit."""

    #: One value per observable, shaped like the observables given for the circuit.
    #: A single observable gives a zero-dimensional array.
    expectation_values: np.ndarray
    #: An upper bound on each value's shot noise, shaped the same. See
    #: :func:`QCut.estimate_standard_errors`.
    standard_errors: np.ndarray
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

    def standard_errors(self) -> list[np.ndarray]:
        """Every circuit's standard errors, in the order the circuits were given."""
        return [entry.standard_errors for entry in self._entries]

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

    def standard_errors(
        self, index: int | None = None
    ) -> list[np.ndarray] | np.ndarray:
        """Wait for the run and return the standard errors.

        Args:
            index (int): the input circuit to return them for, or ``None`` for every
                circuit's, in the order the circuits were given.
        """
        result = self.result()
        if index is None:
            return result.standard_errors()
        return result[index].standard_errors

    def _build(self, raws: list[RawResult]) -> CutEstimatorResult:
        return CutEstimatorResult(
            [
                CutEstimatorEntry(
                    estimate_expectation_values(raw),
                    estimate_standard_errors(raw),
                    raw,
                )
                for raw in raws
            ]
        )


class CutEstimator(_CutPrimitive):
    """Expectation values of cut circuits, run on any backend or sampler.

    Shaped like FiQCI-EMS's estimator: made once for a backend, then :meth:`run` with
    the cut circuits and the observables to estimate on them. A cut circuit comes from
    :func:`QCut.get_locations_and_subcircuits` or :func:`QCut.find_cuts`, and is run
    with the options it was made with.

    The estimator does not transpile, much as qiskit's primitives take circuits already
    written for their target. For a backend with a coupling map, transpile each cut
    circuit for that same backend with :func:`QCut.transpile_subcircuits` first.

    Args:
        backend: the backend or V2 sampler to run on. A sampler, like a backend
            without a coupling map, takes the circuits as they are.
    """

    def __init__(self, backend) -> None:
        super().__init__(backend)

    def run(
        self,
        circuits: CutCircuit | list[CutCircuit],
        observables: ObservablesLike | list[ObservablesLike],
        shots: int = DEFAULT_SHOTS,
        max_batch_size: int = 100,
        parameter_values: dict | list[dict] | None = None,
        **run_options,
    ) -> CutEstimatorJob:
        """Submit the circuits and return without waiting for their results.

        Args:
            circuits: a cut circuit or a list of them.
            observables: what to estimate, taken as qiskit's estimator takes them --
                a label, a ``Pauli``, a ``SparsePauliOp``, a ``SparseObservable``, a
                ``{label: coefficient}`` mapping, or a nested sequence of those. One
                circuit takes them as given. With a list of circuits, give a list with
                one entry per circuit, or one observable for every circuit.
            shots (int): shots per experiment circuit, as in
                :func:`QCut.submit_experiments`.
            max_batch_size (int): the most circuits submitted in one ``run`` call. Each
                input circuit is submitted in batches of its own.
            parameter_values: ``{parameter or name: value}`` to bind into the circuit,
                paired with the circuits the way ``observables`` is. For a sweep, list
                the same circuit once per set of values.
            **run_options: passed on to every ``run`` call on the backend.

        Returns:
            CutEstimatorJob: a handle on every circuit's experiment, whose
            :meth:`~CutEstimatorJob.result` gives the expectation values.

        Raises:
            QCutError: something other than a cut circuit was given, one is not
                transpiled for this backend although it has a coupling map, or one
                has parameters left unbound.
            ValueError: the observables or parameter values do not have one entry per
                circuit, or the observables do not fit their circuit.
        """
        experiments = self._experiments(circuits, observables, parameter_values)
        return CutEstimatorJob(
            self._submit(experiments, shots, max_batch_size, run_options)
        )

    def estimate(
        self,
        circuits: CutCircuit | list[CutCircuit],
        observables: ObservablesLike | list[ObservablesLike],
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
        experiments = self._experiments(circuits, observables, parameter_values)
        return self._estimate(experiments, shots, max_batch_size)

    def _experiments(self, circuits, observables, parameter_values):
        listed = _as_list(circuits)
        paired = _pair(circuits, observables, len(listed), "observable")
        values = _pair(circuits, parameter_values, len(listed), "parameter_values")
        bound = []
        for index, (cut_circuit, entry) in enumerate(zip(listed, values)):
            self._check(cut_circuit, index)
            bound.append(self._bind(cut_circuit, entry, index))
        return [
            self._experiment(cut_circuit, observables=entry)
            for cut_circuit, entry in zip(bound, paired)
        ]
