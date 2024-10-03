"""Reflection utilities for types."""

from collections.abc import Iterable
from functools import reduce
from inspect import getmro
from types import UnionType
from typing import (
    Any,
    NewType,
    Protocol,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

from beartype.door import is_bearable, is_subhint
from sqlalchemy.util.typing import GenericProtocol

T = TypeVar("T")
T_cov = TypeVar("T_cov", covariant=True)
U_cov = TypeVar("U_cov", covariant=True)


type SingleTypeDef[T] = GenericProtocol[T] | TypeAliasType | NewType | type[T]


@runtime_checkable
class SupportsItems(Protocol[T_cov, U_cov]):
    """Protocol for objects that support item access."""

    def items(self) -> Iterable[tuple[T_cov, U_cov]]: ...  # noqa: D102


def is_subtype(type_: SingleTypeDef | UnionType, supertype: T) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_subhint(type_, supertype)


def has_type(obj: Any, type_: SingleTypeDef[T]) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_bearable(obj, type_)


def get_lowest_common_base(types: Iterable[type]) -> type:
    """Return the lowest common base of given types."""
    bases_of_all = reduce(set.intersection, (set(getmro(t)) for t in types))
    return max(bases_of_all, key=lambda b: sum(issubclass(b, t) for t in bases_of_all))
