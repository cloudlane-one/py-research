"""Utils for reflecting the Python runtime."""

import inspect
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, TypeVar

T = TypeVar("T")


def _get_calling_frame(offset: int = 0):
    stack = inspect.stack()
    if len(stack) < offset + 3:
        raise RuntimeError("No caller!")
    return stack[offset + 2]


def get_calling_module(offset: int = 0) -> ModuleType | None:
    """Return the name of the module calling the current function."""
    return inspect.getmodule(_get_calling_frame(offset + 1).frame)


def get_full_args_dict(
    func: Callable, args: Sequence, kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return dict of all args + kwargs.

    Args:
        func: Function to inspect.
        args: Positional arguments given to the function.
        kwargs: Keyword arguments given to the function.

    Returns:
        Dictionary of all args + kwargs.
    """
    argspec = inspect.getfullargspec(func)

    arg_defaults = argspec.defaults or []
    kwdefaults = dict(zip(argspec.args[-len(arg_defaults) :], arg_defaults))

    posarg_names = argspec.args

    # Handle bound methods
    if posarg_names[0] == "self" and hasattr(func, "__self__"):
        posarg_names = posarg_names[1:]

    posargs = dict(zip(posarg_names[: len(args)], args))

    return {**kwdefaults, **posargs, **(kwargs or {})}


def get_return_type(func: Callable) -> type | None:
    """Get the return type annotation of given function, if any."""
    sig = inspect.signature(func)
    return (
        sig.return_annotation
        if sig.return_annotation != inspect.Signature.empty
        else None
    )


def get_subclasses(
    cls: type[T], max_level: int | None = None, _level: int = 1
) -> set[type[T]]:
    """Return all subclasses of given class."""
    if max_level is not None and _level > max_level:
        return set()

    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_subclasses(c, _level=_level + 1)]
    )
