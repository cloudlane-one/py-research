"""Utils for types reflection."""

from typing import Any, TypeGuard, TypeVar

from beartype.door import is_bearable, is_subhint

T = TypeVar("T")


def has_type(obj: Any, type_: type[T]) -> TypeGuard[T]:
    """Check if the object has the given type."""
    return is_bearable(obj, type_)


def is_subtype(obj: type, type_: type[T]) -> TypeGuard[type[T]]:
    """Check if the object is a subtype of the given type."""
    return is_subhint(obj, type_)
