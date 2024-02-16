"""Abstract base for connections to relational databases."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

from .spec import (
    D2,
    DI,
    IL,
    IL2,
    S2,
    S3,
    S4,
    TI,
    V2,
    V3,
    AttrRef,
    AttrSet,
    D,
    Data,
    DataArray,
    DataBackend,
    DataBaseSchema,
    DataNode,
    DataSet,
    DI_tup,
    EmptySchema,
    IndexSubset,
    IndexValue,
    N,
    S,
    S2_contrav,
    S2_cov,
    S3_cov,
    S_cov,
    TI_tup,
    V,
    V2_cov,
    V_cov,
)

R_cov = TypeVar("R_cov", covariant=True, bound="Ref")
Da_cov = TypeVar("Da_cov", covariant=True, bound="Data")

DBS_cov = TypeVar("DBS_cov", covariant=True, bound="DataBaseSchema")
DBS2_cov = TypeVar("DBS2_cov", covariant=True, bound="DataBaseSchema")

IV = TypeVar("IV", bound="IndexValue")


@dataclass(frozen=True, kw_only=True)
class Ref(ABC, Generic[N, Da_cov]):
    """Readable data connection."""

    spec: Da_cov
    """Spec of the referenced data."""

    container: "ArrayRef[N, Any, Any, Any, Any, Any, Any]"
    """Container of the referenced data."""

    @property
    def value(self: "Ref[Any, Data[Any, V]]") -> V:
        """Return the referenced data."""
        ...


@dataclass(frozen=True, kw_only=True)
class Var(Ref[N, Da_cov], ABC):
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
    Ref[N, DataArray[S_cov, V_cov, V2_cov, TI, DI, IL]],
    ABC,
    Generic[N, S_cov, V_cov, V2_cov, TI, DI, IL],
):
    """Readable connection to a data array."""

    @overload
    def __getitem__(  # type: ignore
        self: "ArrayRef[N, S_cov, V_cov, V2_cov, tuple[*TI_tup], DI, IL]",
        key: tuple[*TI_tup],
    ) -> V2_cov | None:
        # Return fully rsolved array item, if it exists.
        ...

    @overload
    def __getitem__(
        self: "ArrayRef[N, S_cov, V_cov, V2_cov, TI, tuple[*DI_tup], IL]",
        key: tuple[*DI_tup],
    ) -> V2_cov:
        # Return fully rsolved array item, guaranteed to exist as per domain index.
        ...

    @overload
    def __getitem__(
        self: "ArrayRef[N, S_cov, V_cov, V2_cov, tuple[V, *TI_tup], DI, IL]",
        key: V,
    ) -> "ArrayRef[N, S_cov, V_cov, V2_cov, tuple[*TI_tup], None, Any]":
        # Return sub-array, possibly empty.
        ...

    @overload
    def __getitem__(
        self: "ArrayRef[N, S_cov, V_cov, V2_cov, tuple[Any, *TI_tup], tuple[IV, *DI_tup], IL]",  # noqa: E501
        key: IV,
    ) -> "ArrayRef[N, S_cov, V_cov, V2_cov, tuple[*TI_tup], tuple[*DI_tup], Any]":
        # Return sub-array, guaranteed to be non-empty as per domain index.
        ...

    @overload
    def __getitem__(
        self,
        key: slice,
    ) -> "ArrayRef[N, S_cov, V_cov, V2_cov, TI, None, IL]":
        # Return sub-array via slice, possibly empty.
        ...

    @overload
    def __getitem__(
        self: "ArrayRef[N, S_cov, V_cov, V2_cov, TI, tuple[IndexValue[IL2, D, V], *DI_tup], IL]",  # noqa: E501
        key: IndexSubset[IL2, D, V, D2],
    ) -> "ArrayRef[N, S_cov, V_cov, V2_cov, TI, tuple[IndexValue[IL2, D2, V], *DI_tup], IL]":  # noqa: E501
        # Return sub-array, guaranteed to be non-empty as per domain index.
        ...

    @overload
    def __getitem__(
        self: "ArrayRef[N, S_cov, V_cov, V2_cov, TI, DI, IL]",
        key: "ArrayRef[N, Any, Any, bool, tuple[V], tuple[IV], IL]",
    ) -> "ArrayRef[N, S_cov, V_cov, V2_cov, TI, None, IL]":
        # Return sub-array via boolean mask, possibly empty.
        ...

    def __getitem__(
        self,
        key: "V_cov | tuple | slice | IndexValue | IndexSubset | ArrayRef",
    ) -> "ArrayRef | V2_cov | None":
        """Get an item of this array or a sub-array."""
        ...


@dataclass(frozen=True, kw_only=True)
class ArrayVar(
    ArrayRef[N, S, V, V2, TI, DI, IL],
    ABC,
):
    """Writable connection to a data array."""

    @overload
    def __setitem__(
        self: "ArrayRef[N, S, V, V2, tuple[*TI_tup], tuple[*DI_tup], IL]",
        key: tuple[*TI_tup] | tuple[*DI_tup],
        value: V2,
    ) -> None:
        # Set array item.
        ...

    @overload
    def __setitem__(
        self: "ArrayRef[N, S, V, V2, tuple[V3, *TI_tup], DI, IL]",
        key: V3,
        value: "ArrayRef[N, S, V, V2, tuple[*TI_tup], Any, Any]",
    ) -> None:
        # Set sub-array via type index.
        ...

    @overload
    def __setitem__(
        self: "ArrayRef[N, S, V, V2, TI, tuple[IV, *DI_tup], IL]",
        key: IV,
        value: "ArrayRef[N, S, V, V2, Any, tuple[*DI_tup], Any]",
    ) -> None:
        # Set sub-array via domain index.
        ...

    @overload
    def __setitem__(
        self, key: slice, value: "ArrayRef[N, S, V, V2, TI, Any, IL]"
    ) -> None:
        # Set sub-array via slice.
        ...

    @overload
    def __setitem__(
        self: "ArrayRef[N, S, V, V2, TI, tuple[IndexValue[IL2, D, V], *DI_tup], IL]",  # noqa: E501
        key: IndexSubset[IL2, D, V, D2],
        value: "ArrayRef[N, S, V, V2, TI, tuple[IndexValue[IL2, D2, V], *DI_tup], IL]",  # noqa: E501
    ) -> None:
        # Set sub-array via subset of domain index.
        ...

    @overload
    def __setitem__(
        self: "ArrayRef[N, S, V, V2, TI, DI, IL]",
        key: "ArrayRef[N, Any, Any, bool, tuple[V], tuple[IV], IL]",
        value: "ArrayRef[N, S, V, V2, TI, Any, IL]",
    ) -> None:
        # Set sub-array via boolean mask.
        ...

    def __setitem__(
        self,
        key: "Any | tuple | slice | IndexValue | IndexSubset | ArrayRef",
        value: "ArrayRef | V2",
    ) -> None:
        """Get an item of this array or a sub-array."""
        ...


@dataclass(frozen=True, kw_only=True)
class NodeRef(
    Ref[N, DataNode[S_cov, V_cov, S2_cov, S3_cov]],
    ABC,
    Generic[N, S_cov, V_cov, S2_cov, S3_cov],
):
    """Readable connection to a data node."""

    @overload
    def __getitem__(
        self, key: AttrSet[S2_cov, S, Any]
    ) -> "NodeRef[N, S_cov, S, Any, Any]": ...

    @overload
    def __getitem__(
        self, key: AttrSet[S3_cov, S2, Any]
    ) -> "NodeRef[N, S_cov, S2, Any, Any] | None": ...

    @overload
    def __getitem__(self, key: AttrRef[Data[S2_cov, V], Any, S2_cov]) -> V: ...

    @overload
    def __getitem__(
        self, key: AttrRef[Data[S3_cov, V], Any, S3_cov] | str
    ) -> V | None: ...

    def __getitem__(
        self,
        key: (
            AttrSet[S2_cov | S3_cov, Any, Any]
            | AttrRef[Data[S2_cov | S3_cov, V], Any, S2_cov | S3_cov]
            | str
        ),
    ) -> "NodeRef | V | None":
        """Get an item or subset of this data node."""
        ...


@dataclass(frozen=True, kw_only=True)
class NodeVar(NodeRef[N, S, V, S2, S3], ABC):
    """Writable connection to a data node."""

    @overload
    def __setitem__(
        self, key: AttrRef[Data[S2 | S3, V2], Any, S2 | S3], value: V2
    ) -> None: ...

    @overload
    def __setitem__(
        self, key: AttrSet[S2, S4, Any], value: "NodeRef[N, S, S4, Any, Any]"
    ) -> None: ...

    @overload
    def __setitem__(
        self, key: AttrSet[S3, S4, Any], value: "NodeRef[N, S, S4, Any, Any]"
    ) -> None: ...

    def __setitem__(
        self,
        key: AttrSet[S2 | S3, Any, Any] | AttrRef[Data[S2 | S3, V], Any, S2 | S3] | str,
        value: "NodeRef | V | None",
    ) -> None:
        """Set an item or subset of this data node."""
        ...


@dataclass(frozen=True, kw_only=True)
class SetRef(
    Ref[N, DataSet[S_cov, V_cov, V2_cov, TI, DI, S2_cov, S3_cov, S2_contrav]],
    ABC,
    Generic[N, S_cov, V_cov, V2_cov, TI, DI, S2_cov, S3_cov, S2_contrav],
):
    """Readable connection to a dataset."""


@dataclass(frozen=True, kw_only=True)
class SetVar(SetRef[N, S, V, V2, TI, DI, S2, S3, S4], ABC):
    """Writable connection to a dataset."""


@dataclass(kw_only=True)
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
    ) -> "DataSource[N, S2, Any, Any] | None": ...

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


@dataclass(kw_only=True)
class DataBase(DataSource[N, V_cov, DBS_cov, DBS2_cov], ABC):
    """Writable database connection."""
