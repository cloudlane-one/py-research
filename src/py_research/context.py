"""Convenience functions for working with context variables."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


@contextmanager
def set_ctx_var[T](var: ContextVar[T], value: T) -> Iterator[T]:
    """Set a context variable within a context manager, restoring the previous value on exit.

    Args:
        var: The context variable to set.
        value: The value to set the context variable to.

    Returns:
        A context manager that yields the set value.
    """
    token = var.set(value)
    try:
        yield value
    finally:
        var.reset(token)
