"""Static schemas for SQL databases."""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.orm as orm

from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_all_subclasses
from py_research.reflect.types import is_subtype

RecordValue: TypeAlias = "Record | Iterable[Record] | Mapping[Any, Record]"

Idx = TypeVar("Idx", bound="Hashable")
Val = TypeVar("Val")
Rec = TypeVar("Rec", bound="Record")
Recs = TypeVar("Recs", bound=RecordValue)


DataFrame: TypeAlias = pd.DataFrame | pl.DataFrame


class AttrRef(sqla.ColumnClause[Val], Generic[Val, Rec]):
    """Reference a property by its containing record type, name and value type."""

    def __init__(  # noqa: D107
        self, record_type: type[Rec], name: str, value_type: type[Val] | None
    ) -> None:
        self.record_type = record_type
        self.name = name
        self.value_type = value_type

        super().__init__(
            self.name,
            type_=self.record_type._sqla_table.c[self.name].type,
        )


@dataclass
class Prop(Generic[Val]):
    """Define property of a record."""

    primary_key: bool = False

    _name: str | None = None
    _record: type["Record"] | None = None
    _value_type: type[Val] | None = None
    _value: Val | None = None

    @property
    def name(self) -> str:
        """Property name."""
        assert self._name is not None, "Property name not set."
        return self._name

    @property
    def record(self) -> type["Record"]:
        """Record type."""
        assert self._record is not None, "Record type not set."
        return self._record

    @property
    def value_type(self) -> type[Val]:
        """Property value type."""
        assert self._value_type is not None, "Property value type not set."
        return self._value_type

    @property
    def value(self) -> Val:
        """Property value."""
        assert self._value is not None, "Property value not set."
        return self._value

    def __setname__(self, _, name: str) -> None:  # noqa: D105
        self._name = name

    @overload
    def __get__(self, instance: "Record", owner: type["Record"]) -> Val: ...

    @overload
    def __get__(self: "Rel[Rec]", instance: None, owner: type[Rec]) -> type[Rec]: ...

    @overload
    def __get__(
        self: "Attr[Val]", instance: None, owner: type[Rec]
    ) -> AttrRef[Val, Rec]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: "object | None", owner: type[Rec] | None
    ) -> Val | type[Rec] | AttrRef[Val, Rec] | Self:
        if owner is None or not issubclass(owner, Record):
            return self

        if instance is not None:
            return self.value

        if is_subtype(self.value_type, RecordValue):
            return cast(type[Rec], self.value_type)

        if isinstance(self, Attr):
            return cast(AttrRef[Val, Rec], owner._cols()[self.name])

        raise NotImplementedError()

    def __set__(self, instance: "Record", value: Val) -> None:
        """Set the value of the property."""
        self._value = value


@dataclass
class Attr(Prop[Val]):
    """Define attribute of a record."""

    col_name: str | None = None

    @property
    def name(self) -> str:
        """Column name."""
        return self.col_name or super().name


@dataclass
class Rel(Prop[Recs]):
    """Define relation to another record."""

    foreign_key: Attr | Iterable[Attr] | dict[Attr, AttrRef] | None = None

    _target: type["Record"] | None = None

    @property
    def target(self) -> type["Record"]:
        """Target record type."""
        assert self._target is not None, "Target record type not set."
        return self._target

    @property
    def cols(self) -> list[Attr]:
        """Foreign key columns."""
        match self.foreign_key:
            case dict():
                source_cols = list(self.foreign_key.keys())
                target_cols = list(self.foreign_key.values())
            case Attr() as col:
                source_cols = [self.record._cols()[col.name]]
                target_cols = self.target._primary_keys()
            case Iterable() as cols:
                source_cols = [self.record._cols()[col.name] for col in cols]
                target_cols = self.target._primary_keys()
            case None:
                return [
                    Attr(
                        **asdict(target_col),
                        col_name=f"{self._name}.{target_col.name}",
                    )
                    for target_col in self.target._primary_keys()
                ]

        assert len(list(source_cols)) == len(
            target_cols
        ), "Count of foreign keys must match count of primary keys in target table."
        assert all(
            issubclass(self.record._value_types[fk_col.name], pk_col.value_type)
            for fk_col, pk_col in zip(source_cols, target_cols)
            if pk_col.value_type is not None
        ), "Foreign key value types must match primary key value types."

        return [
            Attr(
                primary_key=self.primary_key,
                col_name=fk_col.name,
                _record=self.record,
                _value_type=self.record._value_types[fk_col.name],
            )
            for fk_col in source_cols
        ]


def _extract_record_type(hint: Any) -> type["Record"]:
    origin: type = get_origin(hint)

    if issubclass(origin, Record):
        return origin
    if isinstance(origin, Mapping):
        return get_args(hint)[1]
    if isinstance(origin, Iterable):
        return get_args(hint)[0]

    raise ValueError("Invalid record value.")


class Record(Generic[Idx, Val]):
    """Schema for a record in a database."""

    _table_name: str
    _type_map: dict[type, sqla.types.TypeEngine] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
    }

    _value_types: dict[str, type]
    _defined_attrs: list[Attr]
    _rels: list[Rel]

    def __init_subclass__(cls) -> None:  # noqa: D105
        # Retrieve column type definitions from class annotations
        cls._value_types = {
            name: get_args(hint)[0]
            for name, hint in get_type_hints(cls)
            if issubclass(get_origin(hint) or type, Attr)
        }

        cls._defined_attrs = [
            Attr(
                **{
                    **(asdict(attr) if isinstance(attr, Attr) else {}),
                    "_name": name,
                    "_record": cls,
                    "_value_type": cls._value_types[name],
                }
            )
            for name, attr in cls.__dict__.items()
            if isinstance(attr, Attr)
        ]

        cls._rels = [
            Rel(
                **{
                    **(asdict(rel) if isinstance(rel, Rel) else {}),
                    "_name": name,
                    "_record": cls,
                    "_value_type": cls._value_types[name],
                    "_target": _extract_record_type(cls._value_types[name]),
                }
            )
            for name, rel in cls.__dict__.items()
            if isinstance(rel, Rel)
        ]

        return super().__init_subclass__()

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        return (
            cls._table_name
            if hasattr(cls, "_table_name")
            else PyObjectRef.reference(cls).fqn.replace(".", "_")
        )

    @classmethod
    def _cols(cls) -> dict[str, Attr]:
        """Return the columns of this schema."""
        return {
            **{a.name: a for a in cls._defined_attrs},
            **{c.name: c for rel in cls._rels for c in rel.cols},
        }

    @classmethod
    def _primary_keys(cls) -> list[Prop]:
        """Return the primary key columns of this schema."""
        return [p for p in cls._cols().values() if p.primary_key]

    @classmethod
    def _sqla_table(
        cls,
        metadata: sqla.MetaData,
        subs: dict[type["Record"], sqla.Table],
        name: str | None = None,
        schema_name: str | None = None,
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        registry = orm.registry(metadata=metadata, type_annotation_map=cls._type_map)

        # Create a SQLAlchemy table object from the class definition
        return sqla.Table(
            name or cls._default_table_name(),
            registry.metadata,
            *(
                sqla.Column(
                    col.name,
                    registry._resolve_type(col.value_type) if col.value_type else None,
                    primary_key=col.primary_key,
                )
                for col in cls._cols().values()
            ),
            *(
                sqla.ForeignKeyConstraint(
                    [col.name for col in rel.cols],
                    [col.name for col in rel.target._primary_keys()],
                    table=(
                        subs[rel.target]
                        if rel.target in subs
                        else rel.target._sqla_table(metadata, subs)
                    ),
                    name=f"{cls._default_table_name()}_{rel._name}_fk",
                )
                for rel in cls._rels
            ),
            schema=schema_name,
        )

    def __clause_element__(self) -> sqla.TableClause:  # noqa: D105
        assert self._default_table_name() is not None
        return sqla.table(self._default_table_name())


class Schema:
    """Group multiple record types into a schema."""

    _record_types: set[type["Record"]]
    _assoc_tables: set[type["Record"]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        subclasses = get_all_subclasses(cls)

        cls._record_types = {s for s in subclasses if isinstance(s, Record)}

        cls._assoc_tables = set()
        for table in cls._record_types:
            pks = set([col.name for col in table._primary_keys()])
            fks = set([col.name for rel in table._rels for col in rel.cols])
            if pks in fks:
                cls._assoc_tables.add(table)

        super().__init_subclass__()
