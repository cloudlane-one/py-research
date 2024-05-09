"""Reflection utilities for types."""

from typing import Any, TypeGuard, TypeVar

from beartype.door import is_bearable, is_subhint

T = TypeVar("T")


def is_subtype(type_: type, supertype: T) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_subhint(type_, supertype)


def has_type(obj: Any, type_: type[T]) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_bearable(obj, type_)
