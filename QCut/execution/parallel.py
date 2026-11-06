"""Running one experiment on several backends at once."""

from __future__ import annotations

import logging
import threading

from QCut.errors.qcuterror import QCutError
from QCut.transpilation.transpile_cut_circuits import transpile_circuits

logger: logging.Logger = logging.getLogger(__name__)


class ParallelBackend:
    """Spread an experiment's jobs over several backends, giving each batch to the least
    loaded.

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
    that can hold them. Among those that fit, a batch goes to the one given the fewest
    shots so far, the first listed on a tie.

    Pass untranspiled cut circuits and experiments: each batch is transpiled for the
    device it lands on.

    Attributes:
        backends (list): the backends, in the order given.
        submitted (list[int]): how many circuits each has been given, which is what to
            read to see how the work was shared.
        submitted_shots (list[int]): how many shots each has been given, which is what
            the work is shared by.
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
        self._lock = threading.Lock()
        self.submitted = [0] * len(self.backends)
        self.submitted_shots = [0] * len(self.backends)

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
        """Transpile this batch for the least loaded backend that fits, and submit it.

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

        # Reserved before the batch is transpiled, so a call running alongside sees
        # this one's work already. Transpiling and submitting stay outside the lock,
        # so one slow submission does not hold the others up.
        work = len(circuits) * shots
        with self._lock:
            # min() keeps the first of equals, and fitting is in the order given.
            index = min(fitting, key=lambda fit: self.submitted_shots[fit])
            self.submitted[index] += len(circuits)
            self.submitted_shots[index] += work
        backend = self.backends[index]

        try:
            transpiled = transpile_circuits(
                circuits,
                backend,
                self._optimization_level,
                self._transpile_options,
                self._use_iqm_transpiler,
            )
            logger.info(
                f"Batch of {len(transpiled)} circuits, {width} qubits wide, to backend "
                f"{index + 1} of {len(self.backends)}, {backend}"
            )
            return backend.run(transpiled, shots=shots, **options)
        except Exception:
            # Never handed over, so it is not counted as given.
            with self._lock:
                self.submitted[index] -= len(circuits)
                self.submitted_shots[index] -= work
            raise
