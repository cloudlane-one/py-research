"""Common types."""

from dataclasses import Field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import auto
from typing import Any, ClassVar, Protocol, runtime_checkable

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
