"""Common types."""

from dataclasses import Field
from typing import Any, ClassVar, Protocol, runtime_checkable


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
