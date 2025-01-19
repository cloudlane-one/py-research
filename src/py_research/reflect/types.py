"""Reflection utilities for types."""

from collections.abc import Iterable
from functools import reduce
from inspect import getmro
from types import GenericAlias, NoneType, UnionType
from typing import (
    Any,
    ForwardRef,
    NewType,
    Protocol,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

from beartype.door import is_bearable, is_subhint

T = TypeVar("T", covariant=True)
T_cov = TypeVar("T_cov", covariant=True)
U_cov = TypeVar("U_cov", covariant=True)

_AnnotationScanType = Union[
    type[Any], str, ForwardRef, NewType, TypeAliasType, "GenericProtocol[Any]"
]


@runtime_checkable
class GenericProtocol(Protocol[T]):  # type: ignore
    """protocol for generic types.

    this since Python.typing _GenericAlias is private

    """

    __args__: tuple[_AnnotationScanType, ...]
    __origin__: type[T]


type SingleTypeDef[T] = GenericProtocol[T] | TypeAliasType | NewType | type[
    T
] | GenericAlias


@runtime_checkable
class SupportsItems(Protocol[T_cov, U_cov]):
    """Protocol for objects that support item access."""

    def items(self) -> Iterable[tuple[T_cov, U_cov]]: ...  # noqa: D102


def is_subtype(type_: SingleTypeDef | UnionType, supertype: T) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return (
        is_subhint(type_, supertype)
        if not isinstance(type_, TypeAliasType)
        else is_subhint(type_.__value__, supertype)
    )


def has_type(obj: Any, type_: SingleTypeDef[T] | UnionType) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_bearable(obj, type_)


def get_lowest_common_base(types: Iterable[type]) -> type:
    """Return the lowest common base of given types."""
    if len(list(types)) == 0:
        return object

    bases_of_all = reduce(set.intersection, (set(getmro(t)) for t in types))
    return max(bases_of_all, key=lambda b: sum(issubclass(b, t) for t in bases_of_all))


def extract_nullable_type(type_: SingleTypeDef[T | None] | UnionType) -> type[T] | None:
    """Extract the non-none base type of a union."""
    args = get_args(type_)

    if len(args) == 0 and has_type(type_, SingleTypeDef):
        return type_

    notna_args = {arg for arg in args if get_origin(arg) is not NoneType}
    return get_lowest_common_base(notna_args) if notna_args else None


def get_inheritance_distance(cls: type, base: type) -> int | None:
    """Return the inheritance distance between two classes.

    Note: Positive direction is from subclass to base class. If arguments are
    are reversed, the sign will be negative.

    Warning: Untested function.

    Args:
        cls: The subclass.
        base: The base class.

    Returns:
        The signed inheritance distance between the two classes. If the base class is
        not a base of the subclass, None is returned.
    """
    if not isinstance(cls, type) or not isinstance(base, type):
        return None

    if cls is base:
        return 0

    if issubclass(cls, base):
        cls, base, sign = (cls, base, 1)
    elif issubclass(base, cls):
        cls, base, sign = (base, cls, -1)
    else:
        return None

    distance = 1
    bases = set(cls.__bases__)
    while base not in bases and distance < 100:
        bases = reduce(set.union, (set(b.__bases__) for b in bases))
        distance += 1

    return distance * sign
