"""Base for universal relational data schemas."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    LiteralString,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    cast,
    overload,
)

from yarl import URL

from py_research.reflect.types import is_subtype

IL = TypeVar("IL", bound="LiteralString | Hashable | None")
IL2 = TypeVar("IL2", bound="LiteralString | Hashable | None")

D = TypeVar("D", bound="Domain")
D2 = TypeVar("D2", bound="Domain")

TI_tup = TypeVarTuple("TI_tup")
TI = TypeVar("TI")

DI_tup = TypeVarTuple("DI_tup")
DI = TypeVar("DI")

S = TypeVar("S", bound="Schema")
S2 = TypeVar("S2", bound="Schema")
S3 = TypeVar("S3", bound="Schema")
S4 = TypeVar("S4", bound="Schema")
S5 = TypeVar("S5", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")
S2_cov = TypeVar("S2_cov", covariant=True, bound="Schema")
S3_cov = TypeVar("S3_cov", covariant=True, bound="Schema")
S4_cov = TypeVar("S4_cov", covariant=True, bound="Schema")
S_contrav = TypeVar("S_contrav", contravariant=True, bound="Schema")

A = TypeVar("A", bound="Attribute")
A_cov = TypeVar("A_cov", covariant=True, bound="Attribute")
A2_cov = TypeVar("A2_cov", covariant=True, bound="Attribute")

V = TypeVar("V")
V2 = TypeVar("V2")
V3 = TypeVar("V3")
V4 = TypeVar("V4")
V_cov = TypeVar("V_cov", covariant=True)
V2_cov = TypeVar("V2_cov", covariant=True)
V_contrav = TypeVar("V_contrav", contravariant=True)

L_cov = TypeVar("L_cov", bound="ArrayLink", covariant=True)

DA_cov = TypeVar("DA_cov", covariant=True, bound="DataArray")

N = TypeVar("N", bound="LiteralString")


class Domain(Protocol[V_contrav]):
    """Domain of values."""

    def __contains__(self, __key: V_contrav) -> bool:
        """Check containment of given value representing an element or subset."""
        ...


class All(Domain):
    """Domain of all values."""

    def __contains__(self, __key: Any) -> Literal[True]:
        """Check containment of given value representing an element or subset."""
        return True


@dataclass(frozen=True)
class Index(Generic[IL, D, V]):
    """Array index."""

    value_type: type[V]
    domain: D
    key: IL

    @overload
    def __getitem__(self: "Index[IL, D, None]", v: V2) -> "IndexValue[IL, D, V2]": ...

    @overload
    def __getitem__(self, v: V) -> "IndexValue[IL, D, V]": ...

    @overload
    def __getitem__(self, v: D2 | slice) -> "IndexSubset[IL, D, V, D2]": ...

    def __getitem__(
        self, v: V | D2 | slice
    ) -> "IndexValue[IL, D, Any] | IndexSubset[IL, D, V, D2]":
        """Check containment of given value representing an element or subset.

        Returns:
            Special object representing the validated element / subset.
        """
        if isinstance(v, self.value_type):
            return IndexValue(self, v)
        elif isinstance(v, slice):
            return IndexSubset(self, v)
        else:
            return IndexSubset(self, Index(self.value_type, v, self.key))

    def __mul__(
        self, __other: "Index[IL2, D2, V2]"
    ) -> "IndexSpec[tuple[V, V2], tuple[IndexValue[IL, D, V], IndexValue[IL2, D2, V2]], IL | IL2]":  # noqa: E501
        """Combine two indexes into a new index spec."""
        return IndexSpec([self, __other])


class IndexFactory:
    """Factory for index creation."""

    def __getitem__(self, key: type[V]) -> Index[None, All, V]:
        """Create a new index."""
        return Index(key, All(), None)


idx = IndexFactory()


@dataclass(frozen=True)
class IndexValue(Generic[IL, D, V]):
    """Single value of an index."""

    index: Index[IL, D, V]
    value: V


@dataclass(frozen=True)
class IndexSubset(Generic[IL, D, V, D2]):
    """Subset of an index."""

    super: Index[IL, D, V]
    sub: Index[IL, D2, V] | slice


@dataclass(frozen=True)
class IndexSpec(Generic[TI, DI, IL]):
    """Full index spec for an array."""

    indexes: list[Index]

    def __mul__(
        self: "IndexSpec[tuple[*TI_tup], tuple[*DI_tup], IL]",
        __other: "Index[IL2, D, V]",
    ) -> (
        "IndexSpec[tuple[*TI_tup, V], tuple[*DI_tup, IndexValue[IL2, D, V]], IL | IL2]"
    ):
        """At an index at the end of the index spec."""
        return IndexSpec(self.indexes + [__other])


class Schema(Generic[S_cov, A_cov, S2_cov]):
    """Base class for static data schemas."""

    _attr_type: type[A_cov]

    @classmethod
    def attrs(cls) -> set["AttrRef[Any, Any, Self]"]:
        """Return the attributes of this schema."""
        return {
            AttrRef(cls, attr)
            for attr in cls.__dict__.values()
            if is_subtype(attr, cls._attr_type)  # type: ignore
        }

    @classmethod
    def all(cls) -> "AttrSet[Self | S_cov, Self | S_cov, A_cov]":
        """Select all attributes in a schema."""
        return AttrSet(set([cls]), cls.attrs())

    @classmethod
    def refs(cls) -> set[type[S2_cov]]:
        """Return all schemas referenced by this one."""
        return {
            attr.schema for attr in cls.__dict__.values() if isinstance(attr, DataNode)
        }


class EmptySchema(Schema):
    """Empty schema."""


@dataclass(kw_only=True)
class Attribute(Generic[S_cov]):
    """Attribute."""

    name: str | None = None

    def __set_name__(self, cls: type, name: str) -> None:
        """Set the name of this attribute."""
        if isinstance(cls, Schema):
            self.name = name

    @overload
    def __get__(self, instance: S | None, cls: type[S]) -> "AttrSet[S, S, Self]": ...

    @overload
    def __get__(self, instance: Any | None, cls: type) -> Self: ...

    def __get__(
        self, instance: Any | None, cls: type[S]
    ) -> "AttrSet[S, S, Self] | Self":
        """Return a fully typed version of this attribute."""
        if not isinstance(cls, Schema):
            return self

        assert instance is None, "Attribute must be accessed from the class"
        return AttrSet(set([cls]), set([AttrRef(schema=cls, attr=self)]))


@dataclass(frozen=True)
class AttrRef(Generic[A_cov, A2_cov, S_cov]):
    """Attribute of a schema."""

    schema: type[S_cov]
    """Schema, which this attribute belongs to."""

    attr: A_cov | A2_cov
    """Type of the value of this attribute."""


@dataclass(frozen=True)
class AttrSet(Generic[S_cov, S2_cov, A_cov]):
    """Select all attributes in a schema."""

    schemas: Iterable[type[S_cov | S2_cov]]
    attrs: Iterable["AttrRef[A_cov, A_cov, S_cov | S2_cov]"]

    def __add__(
        self, other: "AttrSet[S, S, A]"
    ) -> "AttrSet[S_cov | S, S_cov | S, A_cov | A]":
        """Create a schema select union."""
        return AttrSet[S_cov | S, S_cov | S, A_cov | A](
            set(self.schemas) | set(other.schemas), set(self.attrs) | set(other.attrs)
        )


class NestedData:
    """Declare data as nested."""


IndexKey: TypeAlias = (
    AttrRef["Data[S, V]", "Data[S, V]", S] | set[AttrRef["Data[S, V]", "Data[S, V]", S]]
)


@dataclass(kw_only=True, frozen=True)
class ArrayLink(Generic[S, V, S2, TI]):
    """Link to a data array."""

    attr: AttrRef["Data[S, V]", "Data[S, V]", S]
    target: "DataArray[S2, Any, V, TI, Any, Any] | NestedData"
    source_key: IndexKey[S, TI] | None = None


@dataclass(kw_only=True, frozen=True)
class SetLink(ArrayLink[S, V, S2, TI], Generic[S, V, S2, S3, S4, TI]):
    """Link to a dataset."""

    attr: AttrRef["DataNode[S, V, S3, S4]", Any, S]
    target: "DataSet[S2, Any, V, TI, Any, S3, S4, Any] | NestedData"
    target_key: IndexKey[S3, TI]


@dataclass
class Data(Attribute[S_cov], Generic[S_cov, V_cov]):
    """Data point within a relational database."""

    value_type: type[V_cov]
    """Type of the value of this attribute."""


@dataclass
class DataArray(Data[S_cov, V_cov], Generic[S_cov, V_cov, V2_cov, TI, DI, IL]):
    """Data array in a relational database."""

    item_type: type[V2_cov]
    """Value type of this array's items."""

    index: IndexSpec[TI, DI, IL]
    """Index spec of this array."""

    default: bool = False
    """Whether this is the default array for data of its spec."""


@dataclass
class DataNode(
    Data[S_cov, V_cov],
    Generic[S_cov, V_cov, S2_cov, S3_cov],
):
    """Data point consisting of multiple data values following a schema."""

    schema: type[S2_cov]
    """Schema of this data node."""

    partial_schema: type[S3_cov] = cast(type[S3_cov], Schema)
    """Partial schema of this data node, if any."""

    links: set[ArrayLink[S2_cov | S3_cov, V_cov, S_cov, Any]] = set()
    """External sources of linked attributes in this node."""


default_index = IndexSpec(
    [
        Index(
            Hashable,
            All(),
            AttrRef(Schema, Data(Hashable, name="_id")),
        )
    ]
)


@dataclass
class DataSet(
    DataArray[S_cov, V_cov, V2_cov, TI, DI, Any],
    DataNode[S_cov, V_cov, S2_cov, S3_cov],
    Generic[S_cov, V_cov, V2_cov, TI, DI, S2_cov, S3_cov, S4_cov],
):
    """Dataset in a relational database."""

    index: IndexSpec[TI, DI, AttrRef[Data[S4_cov, Hashable], Any, S2_cov]] = (
        default_index
    )
    """Index spec of this dataset."""


class DataBaseSchema(Schema[S_cov, DA_cov, S2_cov]):
    """Base class for database schemas."""


@dataclass(frozen=True)
class DataBackend(Generic[N]):
    """Data backend."""

    name: N
    """Name of the backend."""

    location: Path | URL
    """Location of the backend."""

    mime_type: str | None = None
    """MIME type of the backend (in case of file)."""

    object_storages: dict[type, Path | URL] = {}
    """Object storages associated with this backend."""
