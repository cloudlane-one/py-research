"""Base for universal relational data schemas."""

from abc import ABC
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Self, TypeAlias, TypeVar, cast, overload

from yarl import URL

from py_research.reflect.types import is_subtype

S = TypeVar("S", bound="Schema")
S2 = TypeVar("S2", bound="Schema")
S3 = TypeVar("S3", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")
S2_cov = TypeVar("S2_cov", covariant=True, bound="Schema")
S3_cov = TypeVar("S3_cov", covariant=True, bound="Schema")

A = TypeVar("A", bound="Attribute")
A_cov = TypeVar("A_cov", covariant=True, bound="Attribute")
A2_cov = TypeVar("A2_cov", covariant=True, bound="Attribute")

V = TypeVar("V")
V_cov = TypeVar("V_cov", covariant=True)

Id = TypeVar("Id")
Id_cov = TypeVar("Id_cov", covariant=True, bound="Hashable")

L = TypeVar("L", bound="IndexLink")
L2 = TypeVar("L2", bound="IndexLink")


class Schema(Generic[S_cov, A_cov]):
    """Base class for static data schemas."""

    _attr_type: type[A_cov]

    @classmethod
    def attrs(cls) -> set["AttrRef[Any, Any, Self]"]:
        """Return the attributes of this schema."""
        return {
            AttrRef(cls, attr)
            for attr in cls.__dict__.values()
            if is_subtype(attr, cls._attr_type)
        }

    @classmethod
    def all(cls) -> "AttrSet[Self | S_cov, Self | S_cov, A_cov]":
        """Select all attributes in a schema."""
        return AttrSet(set([cls]), cls.attrs())


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

    def __get__(self, instance: Any | None, cls: type) -> "AttrSet[S, S, Self] | Self":
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

    parent: "DataArray[S_cov, V_cov, Any]"
    """"""


@dataclass
class DataArray(Data[S_cov, V_cov], Generic[S_cov, V_cov, Id_cov]):
    """Data array in a relational database."""

    key_type: type[Id_cov]
    """Type of the index of this array.
    Use ``tuple`` for multi-dimensional arrays."
    Add ``| slice`` to index types to make them sliceable.
    """


IndexKey: TypeAlias = (
    AttrRef[Data[S, V], Data[S, V], S] | set[AttrRef[Data[S, V], Data[S, V], S]]
)


@dataclass
class DataSet(
    DataArray[S_cov, V_cov, Id_cov], Generic[S_cov, V_cov, Id_cov, S2_cov, S3_cov]
):
    """Data set in a relational database."""

    schema: type[S2_cov]
    """Schema of this dataset."""

    primary_key: IndexKey[S2_cov, Hashable]
    """Primary index of this dataset."""

    partial_schema: type[S3_cov] = cast(type[S3_cov], Schema)
    """Partial schema of this dataset, if any."""

    other_keys: Iterable[IndexKey[S_cov, Hashable]] = set()
    """Other unique indexes of this dataset."""


@dataclass(kw_only=True, frozen=True)
class IndexLink(Generic[A, S, V, Id]):
    """Link to a table."""

    attr: AttrRef[A, Data[S, V], S]
    target: DataArray[Any, V, Id]
    source_key: IndexKey[S, Id] | None = None


@dataclass(kw_only=True, frozen=True)
class SetLink(IndexLink[A, S, V, Id], Generic[A, S, V, Id, S2, S3]):
    """Link to a table."""

    attr: AttrRef[A, DataSet[S, V, Id, S2, S3], S]
    target: DataSet[Any, V, Id, S2, S3]
    target_key: IndexKey[S2, Id]


class DataHub(
    DataSet[S_cov, V_cov, Id_cov, S2_cov, S3_cov],
    Generic[S_cov, V_cov, Id_cov, S2_cov, S3_cov, L],
):
    """Data collection in a relational database."""

    links: set[L] = set()


@dataclass(kw_only=True, frozen=True)
class UrlLink(Generic[L]):
    """Link to a table."""

    link: L
    url: URL


class DataBase(
    ABC,
    DataHub[S_cov, V_cov, Id_cov, S2_cov, S3_cov, L],
    Generic[S_cov, V_cov, Id_cov, S2_cov, S3_cov, L, L2],
):
    """data container."""

    url_links: set[UrlLink[L2]] = set()
