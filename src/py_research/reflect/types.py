"""Reflection utilities for types."""

from collections.abc import Iterable
from typing import Any, Protocol, TypeGuard, TypeVar

from beartype.door import is_bearable, is_subhint

T = TypeVar("T")
T_cov = TypeVar("T_cov", covariant=True)
U_cov = TypeVar("U_cov", covariant=True)


class SupportsItems(Protocol[T_cov, U_cov]):
    """Protocol for objects that support item access."""

    def items(self) -> Iterable[tuple[T_cov, U_cov]]: ...  # noqa: D102


def is_subtype(type_: type, supertype: T) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_subhint(type_, supertype)


def has_type(obj: Any, type_: type[T]) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_bearable(obj, type_)
