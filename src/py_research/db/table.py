"""Universal relational data tables."""

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from types import NoneType
from typing import Any, Generic, TypeAlias, TypeVar, cast

import pandas as pd
import sqlalchemy as sqla
import sqlalchemy.sql.elements as sqla_elements

from .schema import Attribute, DatabaseSchema, Relation, TableSchema

T = TypeVar("T", bound=TableSchema)
T2 = TypeVar("T2", bound=TableSchema)

T_cov = TypeVar("T_cov", covariant=True, bound=TableSchema)
T2_cov = TypeVar("T2_cov", covariant=True, bound=TableSchema)

V = TypeVar("V")
V_cov = TypeVar("V_cov", covariant=True)


class Link(Generic[T, T2, V]):
    """Link to a table."""

    def __init__(  # noqa: D107
        self,
        relation: Relation[T, T2],
        target_table: "Table[T2, Any]",
        target_attribute: Attribute[T2, V] | None = None,
        source_attribute: Attribute[T, V] | None = None,
    ) -> None:
        self.relation = relation
        self.target_table = target_table
        self.target_attribute = target_attribute or target_table.primary_key
        self.__source_attribute = source_attribute
        super().__init__()

    def __hash__(self) -> int:  # noqa: D105
        return super().__hash__()

    def source_attribute(self, source_schema: "type[T]") -> Attribute[T, V]:
        """Get an automatic source attribute definition."""
        return self.__source_attribute or Attribute(
            value_type=self.target_attribute.value_type,
            schema=source_schema,
            name=f"{self.target_table.name}.{self.target_attribute.name}",
        )


Index: TypeAlias = Attribute[T, Any] | set[Attribute[T, Any]]


@dataclass
class Table(Generic[T_cov, T2_cov], ABC):
    """Definition of a tabular data in a relational database."""

    schema: type[T_cov]
    """Schema of this table."""

    partial_schema: type[T2_cov] = cast(type[T2_cov], TableSchema)
    """Partial schema of this table, if any."""

    links: set[Link[T_cov, Any, Any]] = set()
    """Links to other tables."""

    indexes: Index | list[Index] = []
    """Unique indexes of this table.
    In case of list, the first one is taken to be the primary.
    """

    namespace: str | None = None
    """Namespace of this table. If not set, the db schema's default is used."""

    ro: bool | None = None
    """Whether this table is read-only. If not set, the db schema's default is used."""

    db_schema: "DatabaseSchema | None" = None
    """Schema of the database this table belongs to."""

    name: str | None = None
    """Name of this table."""

    query: sqla.Select | None = None
    """SQL select statement for this table."""

    @property
    def foreign_keys(self) -> dict[Attribute[T_cov | T2_cov, Any], sqla.ForeignKey]:
        """Return the foreign keys of this table."""
        return {
            (link.source_attribute(self.schema)): sqla.ForeignKey(
                link.target_table[link.target_attribute]
            )
            for link in self.links
        }

    @property
    def primary_key(self) -> Attribute[T_cov, Any]:
        """Primary key of this table."""
        if isinstance(self.indexes, Attribute):
            return self.indexes

        elif isinstance(self.indexes, list):
            for index in self.indexes:
                if isinstance(index, Attribute):
                    return index

        raise AttributeError("Table has no primary key")

    @property
    def columns(self) -> dict[Attribute[T_cov | T2_cov, V], sqla.Column[V]]:
        """Return the columns of this table."""
        fks = self.foreign_keys
        return {
            attr: sqla.Column(
                name=attr.name,
                type_=attr.value_type,
                *([fks[attr]] if attr in fks else []),
                nullable=issubclass(attr.value_type, NoneType),
                primary_key=(attr == self.primary_key),
            )
            for attr in self.schema.attrs() | self.partial_schema.attrs()
        }

    @property
    def unique_constraints(self) -> list[sqla.UniqueConstraint]:
        """Return the unique constraints of this table."""
        indexes = (
            [self.indexes]
            if isinstance(self.indexes, Attribute | set)
            else self.indexes
        )

        return [
            sqla.UniqueConstraint(
                *[self[i] for i in (index if isinstance(index, set) else {index})]
            )
            for index in indexes
        ]

    @property
    def sql(self) -> sqla.Select:
        """Return a SQL select statement for this table."""
        if self.query is not None:
            return self.query
        elif self.name is not None:
            return sqla.Table(
                self.name,
                sqla.MetaData(),
                *self.columns.values(),
                *self.unique_constraints,
                schema=self.namespace,
            ).select()
        else:
            raise AttributeError("Table has no name")

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of this attribute."""
        if isinstance(owner, DatabaseSchema):
            self.name = name
            self.db_schema = owner

            if self.namespace is None:
                self.namespace = owner._namespace

            if self.ro is None:
                self.ro = owner._ro

    def __clause_element__(self) -> sqla.Subquery:
        """Return the SQL clause element of this table."""
        return self.sql.subquery()

    def __getitem__(self, attr: Attribute[T, V]) -> sqla_elements.KeyedColumnElement[V]:
        """Get a column of this table."""
        assert attr.name is not None
        return self.sql.c[attr.name]

    def __contains__(self, attr: Attribute[T, Any] | str) -> bool:
        """Check if a column exists in this table."""
        if isinstance(attr, Attribute):
            assert attr.name is not None
            name = attr.name
        else:
            name = attr

        return name in self.sql.c

    def keys(self) -> Iterable[str]:
        """Return the column names of this table."""
        return self.sql.c.keys()

    def get(
        self, attr: Attribute[T | T2, V] | str
    ) -> sqla_elements.KeyedColumnElement[V] | None:
        """Get a column of this table."""
        if isinstance(attr, Attribute):
            assert attr.name is not None
            name = attr.name
        else:
            name = attr

        return self.sql.c.get(name)

    def select(self, *attrs: Attribute[T_cov, Any]) -> "Table[Any, T_cov]":
        """Return a table with only the given attributes."""
        return Table(
            schema=TableSchema,
            partial_schema=self.schema,
            links={
                ln for ln in self.links if ln.source_attribute(self.schema) in attrs
            },
            indexes=[
                idx
                for idx in (
                    self.indexes if isinstance(self.indexes, list) else [self.indexes]
                )
                if (isinstance(idx, set) and idx & set(attrs)) or idx in attrs
            ],
            db_schema=self.db_schema,
            ro=True,
            query=self.sql.with_only_columns(*[self[attr] for attr in attrs]),
        )

    def filter(
        self, *conditions: sqla_elements.ColumnElement[bool]
    ) -> "Table[T_cov, T2_cov]":
        """Return a table with the given conditions."""
        return Table(
            schema=self.schema,
            partial_schema=self.partial_schema,
            links=self.links,
            indexes=self.indexes,
            db_schema=self.db_schema,
            ro=True,
            query=self.sql.where(*conditions),
        )


T3 = TypeVar("T3", bound=TableSchema)


@dataclass
class JoinSchema(TableSchema, Generic[T, T2, T3]):
    """Default schema of a join table."""

    left: "Relation[JoinSchema[T, T2, T3], T]"
    """Left table of the join."""

    right: "Relation[JoinSchema[T, T2, T3], T2]"
    """Right table of the join."""

    left_index: "Attribute[JoinSchema[T, T2, T3], Any]"
    """Index of the left table."""

    right_index: "Attribute[JoinSchema[T, T2, T3], Any]"
    """Index of the right table."""

    link_attrs: "Relation[JoinSchema[T, T2, T3], T3]"
    """Attributes of the join table."""


def join_table(
    ltr: Link[T, T2, Any], rtl: Link[T2, T, Any], extra_schema: type[T3] = TableSchema
) -> Table[JoinSchema[T, T2, T3], Any]:
    """Create a join table from two tables."""
    schema = cast(
        type[JoinSchema[T, T2, extra_schema]],
        type(
            f"join_{ltr.target_table.schema.__name__}_{rtl.target_table.schema.__name__}",
            (JoinSchema[T, T2, extra_schema],),
            {},
        ),
    )

    schema.left = Relation(
        name=rtl.relation.name,
        value_type=rtl.target_table.schema,
        schema=schema,
    )
    schema.left_index = Link(
        schema.left, rtl.target_table, rtl.target_attribute
    ).source_attribute(schema)

    schema.right = Relation(
        name=ltr.relation.name,
        value_type=ltr.target_table.schema,
        schema=schema,
    )
    schema.right_index = Link(
        schema.right, ltr.target_table, ltr.target_attribute
    ).source_attribute(schema)

    return Table(
        schema=schema,
        links={
            Link(
                schema.left, rtl.target_table, rtl.target_attribute, schema.left_index
            ),
            Link(
                schema.right, ltr.target_table, ltr.target_attribute, schema.right_index
            ),
        },
        indexes={schema.left_index, schema.right_index},
    )


@dataclass(kw_only=True)
class ConnectedTable(Table[T_cov, T2_cov]):
    """Connection to tabular data in a relational database."""

    engine: sqla.engine.Engine
    """Engine to connect to."""

    def df(self) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        return pd.read_sql(self.sql, self.engine)
