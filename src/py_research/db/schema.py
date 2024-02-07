"""Universal relational data schemas."""

from dataclasses import dataclass
from typing import Any, Generic, Self, TypeVar, overload


class TableSchema:
    """Base class for static table schemas."""

    @classmethod
    def attrs(cls) -> set["Attribute[Self, Any]"]:
        """Return the attributes of this schema."""
        return {attr for attr in cls.__dict__.values() if isinstance(attr, Attribute)}


T = TypeVar("T", bound=TableSchema)

T_cov = TypeVar("T_cov", covariant=True, bound=TableSchema)
T2_cov = TypeVar("T2_cov", covariant=True, bound=TableSchema)

V = TypeVar("V")
V_cov = TypeVar("V_cov", covariant=True)


@dataclass
class Attribute(Generic[T_cov, V_cov]):
    """Attribute of a schema."""

    value_type: type[V_cov]
    """Type of the value of this attribute."""

    schema: type[T_cov] | None = None
    """Schema, which this attribute belongs to."""

    name: str | None = None
    """Name of this attribute."""

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of this attribute."""
        if isinstance(owner, TableSchema):
            self.schema = owner
            self.name = name

    @overload
    def __get__(self, instance: T | None, cls: type[T]) -> "Attribute[T, V_cov]": ...

    @overload
    def __get__(self, instance: Any | None, cls: type) -> "Attribute[Any, V_cov]": ...

    def __get__(self, instance: Any | None, cls: type) -> "Attribute[Any, V_cov]":
        """Return a fully typed version of this attribute."""
        assert instance is None, "Attribute must be accessed from the class"
        assert self.schema is not None
        assert issubclass(cls, self.schema)
        return self

    def __set__(self, instance: Any, value: Self) -> None:
        """Set the value of this attribute."""
        raise AttributeError("Attribute is read-only")

    def __hash__(self) -> int:  # noqa: D105
        return super().__hash__()


@dataclass
class Relation(Attribute[T_cov, T2_cov]):
    """Relation attribute of a schema."""


class DatabaseSchema:
    """Base class for schemas of databases."""

    _namespace: str | None = None
    """Default namespace for tables in this schema."""

    _ro: bool = False
    """Whether to default to ro tables."""
