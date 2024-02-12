"""Base for universal relational data schemas."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Self, TypeAlias, TypeVar, TypeVarTuple, cast, overload

from py_research.reflect.types import is_subtype

S = TypeVar("S", bound="Schema")
S2 = TypeVar("S2", bound="Schema")
S3 = TypeVar("S3", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")
S2_cov = TypeVar("S2_cov", covariant=True, bound="Schema")
S3_cov = TypeVar("S3_cov", covariant=True, bound="Schema")
S_contrav = TypeVar("S_contrav", contravariant=True, bound="Schema")

A = TypeVar("A", bound="Attribute")
A_cov = TypeVar("A_cov", covariant=True, bound="Attribute")
A2_cov = TypeVar("A2_cov", covariant=True, bound="Attribute")

V = TypeVar("V")
V2 = TypeVar("V2")
V_cov = TypeVar("V_cov", covariant=True)
V2_cov = TypeVar("V2_cov", covariant=True)

UI = TypeVar("UI", bound=Hashable)
UI_cov = TypeVar("UI_cov", covariant=True)

TI = TypeVarTuple("TI")

L_cov = TypeVar("L_cov", bound="ArrayLink", covariant=True)

I_cov = TypeVar("I_cov", covariant=True, bound="DataArray")


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
            attr.schema for attr in cls.__dict__.values() if isinstance(attr, DataPoint)
        }


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


@dataclass
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


@dataclass
class Data(Attribute[S_cov], Generic[S_cov, V_cov]):
    """Data point within a relational database."""

    value_type: type[V_cov]
    """Type of the value of this attribute."""


@dataclass
class DataArray(Data[S_cov, V_cov], Generic[S_cov, V_cov, V2_cov, UI_cov, *TI]):
    """Data array in a relational database."""

    item_type: type[V2_cov]
    """Value type of this array's items."""

    full_key_type: type[tuple[*TI]]
    """Type of the index of this array.
    Use ``tuple`` for multi-dimensional arrays."
    Add ``| slice`` to index types to make them sliceable.
    """

    partial_key_type: type[UI_cov]
    """Index type union for partial indexing."""

    default: bool = False
    """Whether this is the default array for data of its spec."""


IndexKey: TypeAlias = (
    AttrRef[Data[S, V], Data[S, V], S] | set[AttrRef[Data[S, V], Data[S, V], S]]
)


@dataclass
class DataPoint(
    Data[S_cov, V_cov],
    Generic[S_cov, V_cov, S2_cov, S3_cov],
):
    """Data point consisting of multiple data values following a schema."""

    schema: type[S2_cov]
    """Schema of this dataset."""

    partial_schema: type[S3_cov] = cast(type[S3_cov], Schema)
    """Partial schema of this dataset, if any."""


class NestedData:
    """Declare data as nested."""


@dataclass(kw_only=True, frozen=True)
class ArrayLink(Generic[S, V, UI]):
    """Link to a table."""

    attr: AttrRef[Data[S, V], Any, S]
    target: "DataArray[Any, V, Any, UI, *tuple[Any, ...]] | NestedData"
    source_key: IndexKey[S, UI] | None = None


@dataclass(kw_only=True, frozen=True)
class SetLink(ArrayLink[S, V, UI], Generic[S, V, S2, S3, UI]):
    """Link to a table."""

    attr: AttrRef[DataPoint[S, V, S2, S3], Any, S]
    target: "DataSet[Any, V, Any, S2, S3, UI, *tuple[Any, ...]] | NestedData"
    target_key: IndexKey[S2, UI]


@dataclass
class DataSet(
    DataArray[S_cov, V_cov, V2_cov, UI_cov, *TI],
    DataPoint[S_cov, V_cov, S2_cov, S3_cov],
    Generic[S_cov, V_cov, V2_cov, S2_cov, S3_cov, UI_cov, *TI],
):
    """Data set in a relational database."""

    schema: type[S2_cov]
    """Schema of this dataset."""

    partial_schema: type[S3_cov] = cast(type[S3_cov], Schema)
    """Partial schema of this dataset, if any."""

    index_keys: Iterable[IndexKey[S_cov, UI_cov]] = set()
    """Schema attributes to use as indexes of this dataset."""

    links: set[ArrayLink[S_cov, Any, Any]] = set()
    """External sources of linked attributes in this table."""


class DbSchema(Schema[S_cov, I_cov, S2_cov]):
    """Base class for database schemas."""
