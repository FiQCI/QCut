"""Warning callers about planned changes to the public interface."""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Callable
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
