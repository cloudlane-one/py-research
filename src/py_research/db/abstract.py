"""Abstract base for connections to relational databases."""

from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

from yarl import URL

from .spec import (
    S2,
    TI,
    AttrRef,
    AttrSet,
    Data,
    DataArray,
    DataPoint,
    DataSet,
    S,
    S2_cov,
    S3_cov,
    S_cov,
    UI_cov,
    V,
    V2_cov,
    V_cov,
)

R_cov = TypeVar("R_cov", covariant=True, bound="Ref")
D_cov = TypeVar("D_cov", covariant=True, bound="Data")


@dataclass(frozen=True, kw_only=True)
class Ref(ABC, Generic[R_cov, D_cov]):
    """Readable data connection."""

    spec: D_cov
    """Spec of the referenced data."""

    container: R_cov
    """Container of the referenced data."""

    @property
    def value(self: "Ref[Any, Data[Any, V]]") -> V:
        """Return the referenced data."""
        ...


@dataclass(frozen=True, kw_only=True)
class Var(Ref[R_cov, D_cov], ABC):
    """Writable data connection."""

    @property
    def value(self: "Ref[Any, Data[Any, V]]") -> V:
        """Return the referenced data."""
        ...

    @value.setter
    def value(self: "Var[Any, Data[Any, V]]", value: V) -> None:
        """Set the referenced data."""
        ...


@dataclass(frozen=True, kw_only=True)
class ArrayRef(
    Ref[R_cov, DataArray[S_cov, V_cov, V2_cov, UI_cov, *TI]],
    ABC,
    Generic[R_cov, S_cov, V_cov, V2_cov, UI_cov, *TI],
):
    """Readable connection to a data array."""


@dataclass(frozen=True, kw_only=True)
class ArrayVar(
    ArrayRef[R_cov, S_cov, V_cov, V2_cov, UI_cov, *TI],
    ABC,
):
    """Writable connection to a data array."""


@dataclass(frozen=True, kw_only=True)
class PointRef(
    Ref[R_cov, DataPoint[S_cov, V_cov, S2_cov, S3_cov]],
    ABC,
    Generic[R_cov, S_cov, V_cov, S2_cov, S3_cov],
):
    """Readable connection to a data point."""


@dataclass(frozen=True, kw_only=True)
class PointVar(PointRef[R_cov, S_cov, V_cov, S2_cov, S3_cov], ABC):
    """Writable connection to a data point."""


@dataclass(frozen=True, kw_only=True)
class SetRef(
    Ref[R_cov, DataSet[S_cov, V_cov, V2_cov, S2_cov, S3_cov, UI_cov, *TI]],
    ABC,
    Generic[R_cov, S_cov, V_cov, V2_cov, S2_cov, S3_cov, UI_cov, *TI],
):
    """Readable connection to a data set."""


@dataclass(frozen=True, kw_only=True)
class SetVar(SetRef[R_cov, S_cov, V_cov, V2_cov, S2_cov, S3_cov, UI_cov, *TI], ABC):
    """Writable connection to a data set."""


@dataclass(frozen=True, kw_only=True)
class DataSource(
    ABC,
    Generic[S_cov, V_cov, V2_cov, S2_cov, S3_cov, UI_cov, *TI],
):
    """Readable database connection."""

    url: URL
    namespace: Hashable | None = None

    @overload
    def __getitem__(
        self, key: AttrSet[S_cov, S, Any]
    ) -> "DataSource[S, Any, Any, Any, Any, Any, Any]": ...

    @overload
    def __getitem__(
        self, key: AttrSet[S2_cov, S2, Any]
    ) -> "DataSource[Any, S2, Any, Any, Any, Any, Any]": ...

    @overload
    def __getitem__(self, key: AttrRef[Data[S_cov, V], Any, S_cov]) -> V: ...

    @overload
    def __getitem__(
        self, key: AttrRef[Data[S2_cov, V], Any, S2_cov] | str
    ) -> V | None: ...

    def __getitem__(
        self,
        key: (
            AttrSet[S_cov | S2_cov, Any, Any]
            | AttrRef[Data[S_cov | S2_cov, V], Any, S_cov | S2_cov]
            | str
        ),
    ) -> "DataSource | V | None":
        """Get a column of this table."""
        ...


@dataclass(frozen=True, kw_only=True)
class DataBase(DataSource[S_cov, V_cov, V2_cov, S2_cov, S3_cov, UI_cov, *TI], ABC):
    """Writable database connection."""
