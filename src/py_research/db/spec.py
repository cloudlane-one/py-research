"""Base for universal relational data schemas."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    LiteralString,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from yarl import URL

from py_research.reflect.types import is_subtype

IL = TypeVar("IL", bound="LiteralString | Schema")
IL2 = TypeVar("IL2", bound="LiteralString | Schema")
IL3 = TypeVar("IL3", bound="LiteralString | Schema")
IL4 = TypeVar("IL4", bound="LiteralString | Schema")

D = TypeVar("D", bound="Domain")
D2 = TypeVar("D2", bound="Domain")
D3 = TypeVar("D3", bound="Domain")
D4 = TypeVar("D4", bound="Domain")

S = TypeVar("S", bound="Schema")
S2 = TypeVar("S2", bound="Schema")
S3 = TypeVar("S3", bound="Schema")
S4 = TypeVar("S4", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")
S2_cov = TypeVar("S2_cov", covariant=True, bound="Schema")
S3_cov = TypeVar("S3_cov", covariant=True, bound="Schema")
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

UI = TypeVar("UI", bound="Index")
UI2 = TypeVar("UI2", bound="Index")
UI_cov = TypeVar("UI_cov", covariant=True)
UI2_cov = TypeVar("UI2_cov", covariant=True)

L_cov = TypeVar("L_cov", bound="ArrayLink", covariant=True)

DA_cov = TypeVar("DA_cov", covariant=True, bound="DataArray")

N = TypeVar("N", bound="LiteralString")


class Domain(Protocol[V_contrav]):
    """Domain of values."""

    def __contains__(self, __key: V_contrav) -> bool:
        """Check containment of given value representing an element or subset."""
        ...


@dataclass(frozen=True)
class Index(Generic[IL, D, V]):
    """Domain of values."""

    label: IL
    domain: D
    value_type: type[V]

    @overload
    def __getitem__(self, v: V) -> "IndexValue[IL, D, V]": ...

    @overload
    def __getitem__(self, v: D2) -> "IndexSubset[IL, D, V, D2]": ...

    def __getitem__(
        self, v: V | D2
    ) -> "IndexValue[IL, D, V] | IndexSubset[IL, D, V, D2]":
        """Check containment of given value representing an element or subset.

        Returns:
            Special object representing the validated element / subset.
        """
        ...


@dataclass(frozen=True)
class IndexValue(Generic[IL, D, V]):
    """Range value protocol."""

    value: V


@dataclass(frozen=True)
class IndexSubset(Generic[IL, D, V, D2]):
    """Range value protocol."""

    super: Index[IL, D, V]
    sub: Index[IL, D2, V]


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
        return AttrSet(set([cls]), set([AttrRef(schema=cls, attr_type=type(self))]))


@dataclass(frozen=True)
class AttrRef(Generic[A_cov, A2_cov, S_cov]):
    """Attribute of a schema."""

    schema: type[S_cov]
    """Schema, which this attribute belongs to."""

    attr_type: A_cov | A2_cov
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
class ArrayLink(Generic[S, V, S2, UI]):
    """Link to a table."""

    attr: AttrRef["Data[S, V]", "Data[S, V]", S]
    target: "DataArray[S2, Any, V, UI, Any] | NestedData"
    source_key: IndexKey[S, UI] | None = None


@dataclass(kw_only=True, frozen=True)
class SetLink(ArrayLink[S, V, S2, UI], Generic[S, V, S2, S3, S4, UI]):
    """Link to a table."""

    attr: AttrRef["DataNode[S, V, S3, S4]", Any, S]
    target: "DataSet[S2, Any, V, UI, Any, S3, S4] | NestedData"
    target_key: IndexKey[S3, UI]


@dataclass(frozen=True)
class Data(Attribute[S_cov], Generic[S_cov, V_cov]):
    """Data point within a relational database."""

    value_type: type[V_cov]
    """Type of the value of this attribute."""


@dataclass(frozen=True)
class DataArray(Data[S_cov, V_cov], Generic[S_cov, V_cov, V2_cov, UI_cov, UI2_cov]):
    """Data array in a relational database."""

    item_type: type[V2_cov]
    """Value type of this array's items."""

    index: Iterable[UI_cov] = []
    """Type of the index of this array.
    Use ``tuple`` for multi-dimensional arrays."
    Add ``| slice`` to index types to make them sliceable.
    """

    partial_index: Iterable[UI2_cov] = []
    """Index type union for partial indexing."""

    default: bool = False
    """Whether this is the default array for data of its spec."""


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class DataSet(
    DataArray[S_cov, V_cov, V2_cov, UI_cov, UI2_cov],
    DataNode[S_cov, V_cov, S2_cov, S3_cov],
    Generic[S_cov, V_cov, V2_cov, UI_cov, UI2_cov, S2_cov, S3_cov],
):
    """Data set in a relational database."""

    index_attrs: Iterable[IndexKey[S_cov, UI_cov]] = set()
    """Schema attributes to use as indexes of this dataset."""


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
