"""Running one experiment on several backends at once."""

from __future__ import annotations

import logging

from QCut.errors.qcuterror import QCutError
from QCut.transpilation.transpile_cut_circuits import transpile_circuits

logger: logging.Logger = logging.getLogger(__name__)


class ParallelBackend:
    """Spread an experiment's jobs over several backends, one batch each in turn.

    QCut submits every batch of a wave before collecting any of it, so handing
    successive batches to different backends leaves them all queued at once. Each batch
    is transpiled for the backend about to run it, so the backends do not have to be
    alike, and only one batch is ever held in its transpiled form. The experiment itself
    stays as it was built.

    Pass it where a backend goes::

        backend = ParallelBackend([first, second])
        job = ck.submit_experiments(experiment, shots=4096, backend=backend)

    A batch goes only to a backend wide enough for it, so a fleet of different sizes
    works. The pieces that fit anywhere are shared, and the widest go to the backends
    that can hold them. The batches are dealt out in turn among those that fit.

    The experiment does not have to be transpiled beforehand. Each batch is transpiled
    for the backend.

    Attributes:
        backends (list): the backends, in the order the batches are dealt out.
        submitted (list[int]): how many circuits each has been given, which is what to
            read to see how the work was shared.
    """

    def __init__(
        self,
        backends,
        optimization_level: int = 3,
        transpile_options: dict | None = None,
        use_iqm_transpiler: bool = True,
    ):
        """Init.

        Args:
            backends: the backends to run on, two or more.
            optimization_level (int): optimization level for transpilation (0-3).
            transpile_options (dict): arguments passed to the transpiler.
            use_iqm_transpiler (bool): whether an IQM backend may use IQM's transpiler.

        Raises:
            QCutError: fewer than two backends were given.
        """
        self.backends = list(backends)
        if len(self.backends) < 2:
            raise QCutError(
                f"ParallelBackend needs at least two backends, got "
                f"{len(self.backends)}. Pass the backend itself to run on one."
            )
        self._optimization_level = optimization_level
        self._transpile_options = transpile_options
        self._use_iqm_transpiler = use_iqm_transpiler
        self._turns: dict[tuple[int, ...], int] = {}
        self.submitted = [0] * len(self.backends)

    @property
    def max_shots(self) -> int | None:
        """The most shots every one of them takes, so a batch fits wherever it lands."""
        declared = [
            shots
            for backend in self.backends
            if isinstance(shots := getattr(backend, "max_shots", None), int)
            and shots > 0
        ]
        return min(declared) if declared else None

    def _fitting(self, width: int) -> tuple[int, ...]:
        """Which backends are wide enough, by index. One that does not say is tried."""
        return tuple(
            index
            for index, backend in enumerate(self.backends)
            if not isinstance(getattr(backend, "num_qubits", None), int)
            or backend.num_qubits >= width
        )

    def run(self, circuits, shots: int = 1024, **options):
        """Transpile this batch for the backend whose turn it is, and submit it there.

        Args:
            circuits: the batch, as QCut sized it.
            shots (int): what to run them at. Passed on unchanged: it is what the
                estimator divides by, and the waves allocate it themselves.
            **options: passed on to the backend.

        Returns:
            The backend's own job, so the results are read exactly as they would be
            had the backend been given the batch directly.

        Raises:
            QCutError: no backend is wide enough for the batch.
        """
        circuits = list(circuits)
        width = max(circuit.num_qubits for circuit in circuits)
        fitting = self._fitting(width)
        if not fitting:
            raise QCutError(
                f"none of the {len(self.backends)} backends has room for a "
                f"{width}-qubit circuit. Cut into smaller pieces, with max_qubits, or "
                "give a backend that fits."
            )

        # Counted per set of backends that fit, so the wide circuits having fewer
        # places to go does not stop the narrow ones from being shared evenly.
        turn = self._turns.get(fitting, 0)
        self._turns[fitting] = turn + 1
        index = fitting[turn % len(fitting)]
        backend = self.backends[index]

        transpiled = transpile_circuits(
            circuits,
            backend,
            self._optimization_level,
            self._transpile_options,
            self._use_iqm_transpiler,
        )
        self.submitted[index] += len(transpiled)
        logger.info(
            f"Batch of {len(transpiled)} circuits, {width} qubits wide, to backend "
            f"{index + 1} of {len(self.backends)}, {backend}"
        )
        return backend.run(transpiled, shots=shots, **options)
