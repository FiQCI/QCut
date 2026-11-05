"""Warning callers about planned changes to the public interface."""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def deprecated(
    message: str, category: type[Warning] = FutureWarning
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Warn whoever calls the decorated function from outside QCut.

    ``FutureWarning`` by default, since it is shown to end users where a
    ``DeprecationWarning`` is hidden outside ``__main__``. Calls from within QCut stay
    quiet, so one public function can build on another, and a wrapper whose own
    behaviour is not changing does not warn on its behalf.

    Args:
        message (str): what changes, and what to use instead.
        category (type[Warning]): the warning to raise.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            caller = sys._getframe(1).f_globals.get("__name__", "")
            if caller != "QCut" and not caller.startswith("QCut."):
                warnings.warn(message, category, stacklevel=2)
            return function(*args, **kwargs)

        return wrapper

    return decorate


#: Modules that moved when the source was reorganised, old path to new. Importing an
#: old path warns and hands back the new module itself, so a class imported either way
#: is the same class, and a result pickled under an old path still loads.
#:
#: A few private helpers moved further than their module did, and are not reachable
#: under the old path: ``_combine_pauli_ops`` and ``_get_observable_circuit_index``
#: (now in ``QCut.utils.observables``), ``is_resonator_backend`` (now in
#: ``QCut.utils.backend``) and ``NonCommutingGate`` (now in ``QCut.utils.circuit``).
#: ``QCut.execution.circuit_knitting`` was split, and the old path keeps only its public
#: names. Its private helpers are in ``experiments``, ``dispatch``, ``waves``,
#: ``submit`` and ``estimate`` under ``QCut.execution``.
MOVED_MODULES: dict[str, str] = {
    "QCut.cutcircuit": "QCut.model.cutcircuit",
    "QCut.cutlocation": "QCut.model.cutlocation",
    "QCut.cutting": "QCut.pre_process",
    "QCut.cutting.circuit_preparation": "QCut.pre_process.circuit_preparation",
    "QCut.cutting.consolidate": "QCut.pre_process.consolidate",
    "QCut.execution.circuit_knitting": "QCut.execution.run",
    "QCut.execution.backend_utility": "QCut.transpilation.transpile_cut_circuits",
    "QCut.execution.basis_transform": "QCut.transpilation.basis_transform",
    "QCut.execution.move_routing": "QCut.transpilation.move_routing",
    "QCut.execution.distribution": "QCut.post_process.distribution",
    "QCut.execution.postprocess": "QCut.post_process.postprocess",
    "QCut.execution.probabilities": "QCut.post_process.probabilities",
    "QCut.execution.job": "QCut.job.handle",
    "QCut.execution.qcutresult": "QCut.job.cutresult",
    "QCut.execution.observables": "QCut.utils.observables",
    "QCut.utils.circuit_utils": "QCut.utils.circuit",
}


class _MovedModuleLoader(importlib.abc.Loader):
    """Load an old module path by importing the module it moved to."""

    def __init__(self, new: str) -> None:
        self.new = new
        self._spec: importlib.machinery.ModuleSpec | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        # Called straight from the import machinery, whose frames the warnings module
        # skips, so a stacklevel of two lands on the caller's import statement.
        warnings.warn(
            f"{spec.name} has moved to {self.new}, and the old path is planned to be "
            "removed in QCut 3.0. Import from the new path, or from QCut itself.",
            FutureWarning,
            stacklevel=2,
        )
        module = importlib.import_module(self.new)
        self._spec = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        """Nothing to run: the module was executed under its new name.

        The import machinery has just set ``__spec__`` to the old path's spec, which
        would leave the real module claiming the old name and this loader. Put its own
        back.
        """
        module.__spec__ = self._spec


class _MovedModuleFinder(importlib.abc.MetaPathFinder):
    """Answer imports of the paths in :data:`MOVED_MODULES`."""

    def find_spec(self, fullname, path, target=None):
        new = MOVED_MODULES.get(fullname)
        if new is None:
            return None
        # No origin, so the import machinery leaves the new module's own __file__ alone.
        return importlib.machinery.ModuleSpec(fullname, _MovedModuleLoader(new))


def install_moved_module_finder() -> None:
    """Make the old module paths importable again, with a warning. Idempotent."""
    if not any(isinstance(finder, _MovedModuleFinder) for finder in sys.meta_path):
        # First, so a stale copy of an old module left in an installed tree cannot
        # shadow the redirect.
        sys.meta_path.insert(0, _MovedModuleFinder())
