"""Reflection utilities for types."""

from typing import TypeGuard, TypeVar

from beartype.door import is_subhint

T = TypeVar("T")


def is_subtype(type_: type, supertype: T) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_subhint(type_, supertype)
