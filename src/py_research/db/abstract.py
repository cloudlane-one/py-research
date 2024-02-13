"""Abstract base for connections to relational databases."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

from .spec import (
    S2,
    TI,
    AttrRef,
    AttrSet,
    Data,
    DataArray,
    DataBackend,
    DataBaseSchema,
    DataNode,
    DataSet,
    EmptySchema,
    N,
    RI_cov,
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

DBS_cov = TypeVar("DBS_cov", covariant=True, bound="DataBaseSchema")
DBS2_cov = TypeVar("DBS2_cov", covariant=True, bound="DataBaseSchema")


@dataclass(frozen=True, kw_only=True)
class Ref(ABC, Generic[N, D_cov]):
    """Readable data connection."""

    spec: D_cov
    """Spec of the referenced data."""

    container: "ArrayRef[N, *tuple[Any, ...]]"
    """Container of the referenced data."""

    @property
    def value(self: "Ref[Any, Data[Any, V]]") -> V:
        """Return the referenced data."""
        ...


@dataclass(frozen=True, kw_only=True)
class Var(Ref[N, D_cov], ABC):
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
    Ref[N, DataArray[S_cov, V_cov, V2_cov, RI_cov, UI_cov, *TI]],
    ABC,
    Generic[N, S_cov, V_cov, V2_cov, RI_cov, UI_cov, *TI],
):
    """Readable connection to a data array."""


@dataclass(frozen=True, kw_only=True)
class ArrayVar(
    ArrayRef[N, S_cov, V_cov, V2_cov, RI_cov, UI_cov, *TI],
    ABC,
):
    """Writable connection to a data array."""


@dataclass(frozen=True, kw_only=True)
class NodeRef(
    Ref[N, DataNode[S_cov, V_cov, S2_cov, S3_cov]],
    ABC,
    Generic[N, S_cov, V_cov, S2_cov, S3_cov],
):
    """Readable connection to a data point."""


@dataclass(frozen=True, kw_only=True)
class NodeVar(NodeRef[N, S_cov, V_cov, S2_cov, S3_cov], ABC):
    """Writable connection to a data point."""


@dataclass(frozen=True, kw_only=True)
class SetRef(
    Ref[N, DataSet[S_cov, V_cov, V2_cov, S2_cov, S3_cov, RI_cov, UI_cov, *TI]],
    ABC,
    Generic[N, S_cov, V_cov, V2_cov, S2_cov, S3_cov, RI_cov, UI_cov, *TI],
):
    """Readable connection to a data set."""


@dataclass(frozen=True, kw_only=True)
class SetVar(SetRef[N, S_cov, V_cov, V2_cov, S2_cov, S3_cov, RI_cov, UI_cov, *TI], ABC):
    """Writable connection to a data set."""


@dataclass(frozen=True, kw_only=True)
class DataSource(
    DataNode[EmptySchema, V_cov, DBS_cov, DBS2_cov],
    ABC,
    Generic[N, V_cov, DBS_cov, DBS2_cov],
):
    """Readable database connection."""

    backend: DataBackend[N]

    @overload
    def __getitem__(
        self, key: AttrSet[DBS_cov, S, Any]
    ) -> "DataSource[N, S, Any, Any]": ...

    @overload
    def __getitem__(
        self, key: AttrSet[DBS2_cov, S2, Any]
    ) -> "DataSource[Any, S2, Any, Any]": ...

    @overload
    def __getitem__(self, key: AttrRef[Data[DBS_cov, V], Any, DBS_cov]) -> V: ...

    @overload
    def __getitem__(
        self, key: AttrRef[Data[DBS2_cov, V], Any, DBS2_cov] | str
    ) -> V | None: ...

    def __getitem__(
        self,
        key: (
            AttrSet[DBS_cov | DBS2_cov, Any, Any]
            | AttrRef[Data[DBS_cov | DBS2_cov, V], Any, DBS_cov | DBS2_cov]
            | str
        ),
    ) -> "DataSource | V | None":
        """Get a dataset in this database."""
        ...


@dataclass(frozen=True, kw_only=True)
class DataBase(DataSource[N, V_cov, DBS_cov, DBS2_cov], ABC):
    """Writable database connection."""
