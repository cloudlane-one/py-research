"""Common types."""

from collections.abc import Iterable
from dataclasses import Field, dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import auto
from typing import (
    Any,
    ClassVar,
    ForwardRef,
    Generic,
    NewType,
    Protocol,
    TypeAliasType,
    TypeVar,
    final,
    runtime_checkable,
)

from py_research.enums import StrEnum


@runtime_checkable
class DataclassInstance(Protocol):
    """Protocol for dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


@runtime_checkable
class StatePicklable(Protocol):
    """Protocol for dataclass instances."""

    def __getstate__(self) -> Any: ...  # noqa: D105


@runtime_checkable
class ArgsPicklable(Protocol):
    """Protocol for dataclass instances."""

    def __getnewargs__(self) -> tuple: ...  # noqa: D105


@runtime_checkable
class ArgsPicklableEx(Protocol):
    """Protocol for dataclass instances."""

    def __getnewargs_ex__(self) -> tuple: ...  # noqa: D105


class UUID4(str):
    """UUID4 string."""


type Ordinal = (
    bool | int | float | Decimal | datetime | date | time | timedelta | UUID4 | str
)


class Not(StrEnum):
    """Demark some kind of unresolved or unhandled status."""

    defined = auto()
    """State in question is entirely undefined."""

    resolved = auto()
    """State in question is defined, but not fully resolved."""

    handled = auto()
    """Requested state change is not handled."""

    changed = auto()
    """State in question is not changed."""


T = TypeVar("T", covariant=True)
T_cov = TypeVar("T_cov", covariant=True)
U_cov = TypeVar("U_cov", covariant=True)

type _AnnotationScanType = type[Any] | TypeAliasType | GenericAlias[
    Any
] | NewType | ForwardRef | str


@runtime_checkable
class GenericAlias(Protocol[T]):  # type: ignore
    """protocol for generic types.

    this since Python.typing _GenericAlias is private

    """

    __args__: tuple[_AnnotationScanType, ...]
    __origin__: type[T]


type SingleTypeDef[T] = GenericAlias[T] | TypeAliasType | type[T] | NewType


@runtime_checkable
class SupportsItems(Protocol[T_cov, U_cov]):
    """Protocol for objects that support item access."""

    def keys(self) -> Iterable[T_cov]: ...  # noqa: D102

    def values(self) -> Iterable[U_cov]: ...  # noqa: D102

    def items(self) -> Iterable[tuple[T_cov, U_cov]]: ...  # noqa: D102


T_contra = TypeVar("T_contra", contravariant=True)


@final
@dataclass
class ContraType(Generic[T_contra]):
    """Represent a contravariant type."""

    type_: type[T_contra] | None = None
