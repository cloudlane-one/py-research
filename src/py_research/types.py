"""Common types."""

from dataclasses import Field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, ClassVar, Protocol, final, runtime_checkable


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


@final
class Undef:
    """Demark undefined status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class Keep:
    """Demark unchanged status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]
