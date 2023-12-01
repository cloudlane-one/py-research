"""Utils for Python code reflection."""

import inspect
from collections.abc import Callable
from typing import Any, Sequence, TypeVar


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
    func: Callable, args: Sequence, kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return dict of all args + kwargs with names."""
    argspec = inspect.getfullargspec(func)

    arg_defaults = argspec.defaults or []
    kwdefaults = dict(zip(argspec.args[-len(arg_defaults) :], arg_defaults))

    posargs = dict(zip(argspec.args[: len(args)], args))

    return {**kwdefaults, **posargs, **(kwargs or {})}


T = TypeVar("T")


def get_all_subclasses(cls: type[T]) -> set[type[T]]:
    """Return all subclasses of given class."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )
