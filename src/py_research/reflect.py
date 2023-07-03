"""Utils for Python code reflection."""

import inspect
from collections.abc import Callable
from typing import Any


def _get_calling_frame(offset=0):
    stack = inspect.stack()
    if len(stack) < offset + 3:
        raise RuntimeError("No caller!")
    return stack[offset + 2]


def get_calling_module_name():
    """Return the name of the module calling the current function."""
    mod = inspect.getmodule(_get_calling_frame(offset=1).frame)
    return mod.__name__ if mod is not None else None


def get_full_args_dict(
    func: Callable, args: list, kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return dict of all args + kwargs with names."""
    argspec = inspect.getfullargspec(func)
    argnames = [
        a for i, a in enumerate(argspec.args) if i not in list(argspec.defaults or [])
    ]

    return {**dict(zip(argnames, args)), **(kwargs or {})}
