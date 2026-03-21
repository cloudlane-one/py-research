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

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from py_research.enums import StrEnum


class UUID4(str):
    """UUID4 string."""


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


T_contra = TypeVar("T_contra", contravariant=True)


@final
@dataclass
class ContraType(Generic[T_contra]):
    """Represent a contravariant type."""

    type_: type[T_contra] | None = None


@dataclass(frozen=True)
@final
class Undefined:
    """Undefined typevar marker."""

    pass


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


type Ordinal = (bool | int | float | Decimal | datetime | date | time | timedelta | str)


T = TypeVar("T", covariant=True)
T_cov = TypeVar("T_cov", covariant=True)
U_cov = TypeVar("U_cov", covariant=True)


type AnnotationScanType = type[Any] | TypeAliasType | GenericAlias[
    Any
] | NewType | ForwardRef | str


@runtime_checkable
class GenericAlias[T](Protocol):
    """protocol for generic types.

    this since Python.typing _GenericAlias is private

    """

    __args__: tuple[AnnotationScanType, ...]
    __origin__: type[T]


type SingleTypeDef[T] = GenericAlias[T] | TypeAliasType | type[T] | NewType


@runtime_checkable
class SupportsItems(Protocol[T_cov, U_cov]):
    """Protocol for objects that support item access."""

    def keys(self) -> Iterable[T_cov]: ...  # noqa: D102

    def values(self) -> Iterable[U_cov]: ...  # noqa: D102

    def items(self) -> Iterable[tuple[T_cov, U_cov]]: ...  # noqa: D102


@runtime_checkable
class SupportsKeysAndGetItem[K, V](Protocol):
    """A protocol for objects that support keys() and __getitem__()."""

    def keys(self) -> Iterable[K]: ...  # noqa: D102
    def __getitem__(self, key: K, /) -> V: ...  # noqa: D105


@runtime_checkable
class RuntimeValidated(Protocol):
    """Protocol for objects that support pydantic runtime validation."""

    def __get_pydantic_core_schema__(  # noqa: D105
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema: ...


@dataclass(frozen=True)
class Attr:
    """Reference to an attribute."""

    name: str


operator_methods = [
    "__abs__",
    "__add__",
    "__aenter__",
    "__aexit__",
    "__aiter__",
    "__and__",
    "__anext__",
    "__await__",
    "__bool__",
    "__buffer__",
    "__bytes__",
    "__call__",
    "__ceil__",
    "__complex__",
    "__concat__",
    "__contains__",
    "__delitem__",
    "__dir__",
    "__enter__",
    "__eq__",
    "__exit__",
    "__float__",
    "__floor__",
    "__floordiv__",
    "__format__",
    "__ge__",
    "__getitem__",
    "__gt__",
    "__hash__",
    "__iadd__",
    "__iand__",
    "__iconcat__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__index__",
    "__instancecheck__",
    "__int__",
    "__inv__",
    "__invert__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__iter__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__len__",
    "__length_hint__",
    "__lshift__",
    "__lt__",
    "__match_args__",
    "__matmul__",
    "__missing__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__next__",
    "__not__",
    "__or__",
    "__pos__",
    "__pow__",
    "__release_buffer__",
    "__repr__",
    "__reversed__",
    "__round__",
    "__rshift__",
    "__setitem__",
    "__str__",
    "__sub__",
    "__truediv__",
    "__trunc__",
    "__xor__",
]
