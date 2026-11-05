"""Init circuit knitting."""  # noqa: N999

from importlib.metadata import PackageNotFoundError, version

from QCut.execution.estimate import JobEstimate, RunEstimate, estimate_run
from QCut.execution.experiments import get_experiment_circuits
from QCut.execution.parallel import ParallelBackend
from QCut.execution.run import run, run_cut_circuit, run_experiments
from QCut.execution.submit import submit_experiments
from QCut.job.cutresult import CircuitResult, RawResult
from QCut.job.handle import PartialBatch, QCutJob
from QCut.options import DEFAULT_OPTIONS, CutOptions
from QCut.post_process.postprocess import (
    estimate_expectation_values,
)
from QCut.post_process.probabilities import (
    QuasiProbabilities,
    estimate_probabilities,
)
from QCut.pre_process.circuit_preparation import (
    get_locations_and_subcircuits,
)
from QCut.pre_process.consolidate import consolidate_two_qubit_blocks
from QCut.primitives import CutEstimator, CutQuasiSampler
from QCut.QCutFind import find_cuts
from QCut.qpd.qpd_gates import cut, cutCZ, cutGate, cutISWAP, cutSWAP
from QCut.transpilation.transpile_cut_circuits import (
    transpile_circuits,
    transpile_experiments,
    transpile_subcircuits,
)
from QCut.utils.deprecation import install_moved_module_finder
from QCut.utils.observables import ObservablesLike

install_moved_module_finder()
del install_moved_module_finder

try:
    __version__ = version("QCut")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = [
    "CutEstimator",
    "CutQuasiSampler",
    "ParallelBackend",
    "transpile_circuits",
    "transpile_experiments",
    "transpile_subcircuits",
    "estimate_expectation_values",
    "estimate_run",
    "get_experiment_circuits",
    "get_locations_and_subcircuits",
    "run",
    "run_cut_circuit",
    "run_experiments",
    "submit_experiments",
    "QCutJob",
    "PartialBatch",
    "cut",
    "cutCZ",
    "cutSWAP",
    "cutISWAP",
    "cutGate",
    "find_cuts",
    "CutOptions",
    "DEFAULT_OPTIONS",
    "consolidate_two_qubit_blocks",
    "estimate_probabilities",
    "QuasiProbabilities",
    "RunEstimate",
    "JobEstimate",
    "RawResult",
    "CircuitResult",
    "ObservablesLike",
]
