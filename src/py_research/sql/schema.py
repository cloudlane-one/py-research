"""Static schemas for SQL databases."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Self,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import sqlalchemy as sqla
import sqlalchemy.orm as orm

from py_research.reflect.runtime import get_all_subclasses

V = TypeVar("V")

T = TypeVar("T", bound="Table")
T2 = TypeVar("T2", bound="Table")
T3 = TypeVar("T3", bound="Table")


class ColRef(sqla.ColumnClause[V], Generic[V, T]):
    """Reference a column by table schema, name and value type."""

    def __init__(  # noqa: D107
        self, table: type[T], name: str, value_type: type[V] | None
    ) -> None:
        self.table_schema = table
        self.name = name
        self.value_type = value_type

        super().__init__(
            self.name,
            type_=self.table_schema._sqla_table.c[self.name].type,
        )


@dataclass
class Col(Generic[V]):
    """Define table column within a table schema."""

    primary_key: bool = False
    col_name: str | None = None
    _attr_name: str | None = None

    @property
    def name(self) -> str:
        """Column name."""
        name = self.col_name or self._attr_name
        assert name is not None, "Column name not set."
        return name

    def __setname__(self, _, name: str) -> None:  # noqa: D105
        self._attr_name = name

    @overload
    def __get__(self, instance: None, owner: type[T]) -> ColRef[V, T]: ...

    @overload
    def __get__(self, instance: object, owner: type) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type[T]
    ) -> ColRef[V, T] | Self:
        if instance is not None or not issubclass(owner, Table):
            return self

        assert self._attr_name is not None
        return cast(ColRef[V, T], owner._columns[self._attr_name])


@dataclass
class Rel(Generic[T, T2, V]):
    """Define foreign key column(s) for a table."""

    target: type[T]
    foreign_key: Col | Iterable[Col] | dict[Col[V], ColRef[V, T]] | None = None
    _source: type[T2] | None = None
    _attr_name: str | None = None

    @property
    def cols(self) -> tuple[ColRef[Any, T2], ...]:
        """Foreign key columns."""
        assert self._source is not None, "Source table not set."
        assert self._attr_name is not None, "Attribute name not set."

        match self.foreign_key:
            case dict():
                source_cols = list(self.foreign_key.keys())
                target_cols = list(self.foreign_key.values())
            case Col() as col:
                source_cols = [self._source._columns[col.name]]
                target_cols = self.target._primary_keys
            case Iterable() as cols:
                source_cols = [self._source._columns[col.name] for col in cols]
                target_cols = self.target._primary_keys
            case None:
                return tuple(
                    ColRef(
                        self._source,
                        f"{self._attr_name}.{target_col.name}",
                        target_col.value_type,
                    )
                    for target_col in self.target._primary_keys
                )

        assert len(list(source_cols)) == len(
            target_cols
        ), "Count of foreign keys must match count of primary keys in target table."
        assert all(
            issubclass(self._source._value_types[fk_col.name], pk_col.value_type)
            for fk_col, pk_col in zip(source_cols, target_cols)
            if pk_col.value_type is not None
        ), "Foreign key value types must match primary key value types."

        return tuple(
            ColRef(self._source, fk_col.name, self._source._value_types[fk_col.name])
            for fk_col in source_cols
        )

    def __setname__(self, _, name: str) -> None:  # noqa: D105
        self._attr_name = name

    @overload
    def __get__(self, instance: None, owner: type[T3]) -> "Rel[T, T3, V]": ...

    @overload
    def __get__(self, instance: object, owner: type) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type[T3]
    ) -> "Rel[T, T3, V] | Self":
        if instance is not None or not issubclass(owner, Table):
            return self

        assert self._attr_name is not None
        return Rel(self.target, self.foreign_key, owner, self._attr_name)


class Schema:
    """Base class for declarative SQL schemas."""

    _type_map: dict[type, sqla.types.TypeEngine] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
    }

    @classmethod
    def _tables(cls) -> set[type["Table"]]:
        """Return all tables defined by subclasses."""
        subclasses = get_all_subclasses(cls)
        return {s for s in subclasses if isinstance(s, Table)}


class Table(Schema):
    """Protocol for table schemas."""

    _default_name: str

    _value_types: dict[str, type]
    _primary_keys: tuple[ColRef["Table", Self], ...]
    _relations: tuple[Rel["Table", Self, Any], ...]

    _columns: dict[str, ColRef["Table", Self]]

    def __clause_element__(self) -> sqla.TableClause:  # noqa: D105
        assert self._default_name is not None
        return sqla.table(self._default_name)

    def __init_subclass__(cls) -> None:  # noqa: D105
        if not hasattr(cls, "_default_name"):
            # Generate a default table name from the class name
            cls._default_name = cls.__name__.lower()

        # Retrieve column type definitions from class annotations
        cls._value_types = {
            attr: get_args(hint)[0]
            for attr, hint in get_type_hints(cls)
            if get_origin(hint) is Col
        }

        # Create a dictionary of columns explicitly defined in the class
        defined_cols = {
            attr: ColRef(cls, attr, cls._value_types[attr])
            for attr, col in cls.__dict__.items()
            if isinstance(col, Col)
        }

        # Create a tuple of primary key columns
        cls._primary_keys = tuple(c for c in defined_cols.values() if c.primary_key)

        # Create a tuple of relation objects defined in the class
        cls._relations = tuple(
            Rel(rel.target, rel.foreign_key, cls, rel_name)
            for rel_name, rel in cls.__dict__.items()
            if isinstance(rel, Rel)
        )

        # Create a dictionary of all columns in the table
        cls._columns = {
            **defined_cols,
            **{c.name: c for rel in cls._relations for c in rel.cols},
        }

        return super().__init_subclass__()

    @classmethod
    def _sqla_table(
        cls, metadata: sqla.MetaData, subs: dict[type["Table"], sqla.Table]
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        registry = orm.registry(metadata=metadata, type_annotation_map=cls._type_map)

        # Create a SQLAlchemy table object from the class definition
        return sqla.Table(
            cls._default_name,
            registry.metadata,
            *(
                sqla.Column(
                    col.name,
                    registry._resolve_type(col.value_type) if col.value_type else None,
                    primary_key=col.primary_key,
                )
                for col in cls._columns.values()
            ),
            *(
                sqla.ForeignKeyConstraint(
                    [col.name for col in rel.cols],
                    [col.name for col in rel.target._primary_keys],
                    table=(
                        subs[rel.target]
                        if rel.target in subs
                        else rel.target._sqla_table(metadata, subs)
                    ),
                    name=f"{cls._default_name}_{rel._attr_name}_fk",
                )
                for rel in cls._relations
            ),
        )
