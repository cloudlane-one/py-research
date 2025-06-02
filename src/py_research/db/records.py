"""Basis for relational data records."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    Set,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cmp_to_key, reduce
from inspect import getmodule
from pathlib import Path
from secrets import token_hex
from sqlite3 import PARSE_DECLTYPES
from types import ModuleType, new_class
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    LiteralString,
    Self,
    cast,
    get_args,
)
from uuid import uuid4

import networkx as nx
import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.orm as orm
from cloudpathlib import CloudPath
from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema
from typing_extensions import TypeVar
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.caching import cached_method, cached_prop
from py_research.data import copy_and_override
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.types import SupportsItems, TypeRef, has_type
from py_research.types import UUID4, Not

from .data import (
    PL,
    SQL,
    AutoIdx,
    AutoIndexable,
    Base,
    C,
    Col,
    Ctx,
    D,
    Data,
    Expand,
    Frame,
    Idx,
    InputData,
    InputFrame,
    Interface,
    KeySelect,
    KeyT,
    KeyTt,
    R,
    Registry,
    Root,
    RwxT,
    RwxT2,
    SxT2,
    Tab,
    Tabs,
    U,
    ValT,
    ValT2,
    frame_coalesce,
)
from .models import IdxT, Model, ModelMeta, Prop, RuT, Var
from .utils import (
    get_pl_schema,
    register_sqlite_adapters,
    remove_cross_fk,
    safe_delete,
    safe_insert,
    safe_update,
    validate_sql_table,
)

RecT = TypeVar("RecT", bound="Record", contravariant=True, default=Any)
RecT2 = TypeVar("RecT2", bound="Record")


@dataclass(kw_only=True, eq=False)
class Attr(
    Var[ValT, Idx[()], SQL, Col, ValT, RecT],
    Generic[ValT, RecT],
):
    """Single-value attribute or column.

    Note:
        This always serializes directly into its owner record instance,
        which may be in form of a SQL column, a Python `__dict__` entry,
        or a JSON object property.
    """

    def _getter(self, instance: RecT) -> ValT | Literal[Not.resolved]:
        """Get the value of this attribute on an instance."""
        if self.name not in instance.__dict__:
            return Not.resolved

        return instance.__dict__[self.name]

    def _setter(self: Attr[ValT2], instance: RecT, value: ValT2) -> None:
        """Set the value of this attribute on an instance."""
        instance.__dict__[self.name] = value

    def _id(self) -> str:
        """Identity of the attribute."""
        # TODO: Implement this method for the Attr class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> Expand[Idx[()]]:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, Col, Any, Any, Root, Ctx[Any, Any, Tab]],
    ) -> Frame[PL, Col]:
        """Get SQL-side reference to this property."""
        parent = self.parent()
        assert parent is not None
        parent_df = parent.load()

        return Frame(parent_df[self._id()])

    def _mutation(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Prop[Any, Any, RuT],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[RuT]] = {U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        raise NotImplementedError()


TupT = TypeVar("TupT", bound=tuple[Any, ...], default=Any)


@dataclass(kw_only=True, eq=False)
class AttrTuple(
    Prop[TupT, Idx[()], R, PL, Col | Tab, TupT, RecT],
):
    """Index property for a record type."""

    attrs: tuple[Attr[Any, RecT], ...]

    def __post_init__(self) -> None:  # noqa: D105
        """Get the value of this attribute."""
        if len(self.attrs) > 0:
            data = (
                reduce(Data.__matmul__, self.attrs)
                if len(self.attrs) > 1
                else self.attrs[0]
            )
            self.data = cast(
                Data[
                    TupT, Expand[Idx[()]], Col | Tab, PL, R, Interface[RecT, Any, Tab]
                ],
                data,
            )

    def _getter(self, instance: RecT) -> TupT | Literal[Not.resolved]:
        """Get the value of this attribute tuple."""
        val = tuple(attr.__get__(instance, type(instance)) for attr in self.attrs)
        return cast(TupT, val[0] if len(val) == 1 else val)

    def gen_value_map(self, val: Any) -> dict[str, Hashable]:
        """Generate a key-value map for this key."""
        idx_names = [attr.name for attr in self.attrs]

        if len(idx_names) == 1:
            return {idx_names[0]: val}

        assert isinstance(val, tuple) and len(val) == len(idx_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(idx_names, val)}

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self.context, self.attrs))


@dataclass(eq=False)
class Key(AttrTuple[tuple[*KeyTt]]):
    """Index property for a record type."""

    attrs: tuple[Attr, ...] = ()

    def __set_name__(self, owner: type[Record], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if len(self.attrs) == 0:
            self.attrs = (
                Attr(
                    alias=self.name,
                    typeref=TypeRef(Attr[Hashable]),
                    context=Interface(owner),
                ),
            )
            setattr(owner, f"{name}_attr", self.attrs[0])
            self.__post_init__()


def records_to_df(
    models: Iterable[dict[str, Any] | Record],
) -> pl.DataFrame:
    """Convert values to a DataFrame."""
    model_types: set[type[Record]] = {
        type(val) for val in models if isinstance(val, Record)
    }
    df_data = [
        (val._to_dict(keys="names") if isinstance(val, Record) else val)
        for val in models
    ]

    return pl.DataFrame(
        df_data,
        schema=get_pl_schema(
            {
                attr.name: attr.common_value_type
                for rec in model_types
                for attr in rec._attrs().values()
            }
        ),
    )


class Record(Model, AutoIndexable[*KeyTt]):
    """Schema for a table in a database."""

    _root_class = True

    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID4: sqla.types.CHAR(36),
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)

        for superclass in cls._model_superclasses:
            if superclass in cls.__bases__ and issubclass(superclass, Record):
                super_pk = superclass._primary_key()
                col_map = {
                    Attr[Hashable, Self](
                        alias=pk_attr.name,
                        typeref=pk_attr.typeref,
                        context=Interface(cls),
                    ): pk_attr
                    for pk_attr in super_pk.attrs
                }
                for col in col_map.keys():
                    setattr(cls, col.name, col)

                pk = Key[cast(type[tuple], super_pk.typeargs[TupT]), cls](
                    attrs=tuple(col_map.keys())
                )
                setattr(cls, pk.name, pk)

                fk = Link[superclass, cls](on=col_map)
                setattr(cls, fk.name, fk)

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        name = cls._table_name or cls._fqn
        fqn_parts = name.split(".")

        name = fqn_parts[-1]
        for part in reversed(fqn_parts[:-1]):
            name = part + "_" + name
            if len(name) > 40:
                break

        return name

    @classmethod
    def _attrs(cls) -> dict[str, Attr]:
        """Columns of this record type's table."""
        return {
            prop.name: prop
            for prop in cls._get_class_props().values()
            if isinstance(prop, Attr)
        }

    @classmethod
    def _keys(cls) -> dict[str, Key]:
        """Columns of this record type's table."""
        return {
            prop.name: prop
            for prop in cls._get_class_props().values()
            if isinstance(prop, Key)
        }

    @classmethod
    def _primary_key(cls) -> Key:
        """Primary key of this record type's table."""
        return cls.__dict__["_pk"]

    @classmethod
    def _links(cls) -> dict[str, Link]:
        """Columns of this record type's table."""
        return {
            prop.name: prop
            for prop in cls._get_class_props().values()
            if isinstance(prop, Link)
        }

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: D105
        cls, source: type[Any], handler: Callable[[Any], CoreSchema]
    ) -> CoreSchema:
        # TODO: Extend superclass method to handle opportunistic record upload and links
        # depends on context supplied via info object.
        # Same Base: Use a simple reference mechanism by storing the record's ID.
        # Different Base: Also reference by id + copy record to other base if not there.
        # Other Root with public source and online target: Public URL instead of raw ID.
        # Other Root with public source and local-only tgt: Full JSON + source URL
        # Other Root with private source: Full JSON
        raise NotImplementedError()

    @classmethod
    def __get_pydantic_json_schema__(  # noqa: D105
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # TODO: Extend superclass method to handle opportunistic record upload and links
        # depends on context supplied via info object.
        raise NotImplementedError()

    _published: bool = False
    _base: Attr[DataBase] = Attr(default_factory=lambda: DataBase())

    _pk: Key[*KeyTt]

    @cached_prop
    def _table(self) -> Table[Self, Any]:
        """Return the singleton class for this record."""
        return self._base.table(type(self))

    @cached_method
    def _data(self) -> Data[Self, Idx[()], Tab, SQL, R | U, Base]:
        """Return the singleton class for this record."""
        registry = cast(Registry[Record[*tuple[Any, ...]]], self._table)
        single_rec = registry[KeySelect(self._pk)]
        return cast(Data[Self, Idx[()], Tab, SQL, R | U, Base], single_rec)

    def _load_dict(self) -> None:
        if not self._published or self._pk in self._table._instance_map:
            return

        query = self._data().select()

        result = self._base.connection.execute(query)
        row = result.fetchone()

        if row is not None:
            self.__dict__.update(row._asdict())

    def _save_dict(self) -> None:
        data = self._data()
        data <<= self

    def __hash__(self) -> int:
        """Identify the record by database and id."""
        return gen_int_hash((self._base if self._published else None, self._pk))

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)


register_sqlite_adapters()


type TableInput = (
    Record
    | Iterable[Record]
    | Mapping[Hashable, Record]
    | pl.DataFrame
    | pd.DataFrame
    | sqla.Select
    | sqla.FromClause
    | Mapping[str, pl.Series | pd.Series | sqla.ColumnElement]
)


type OverlayType = Literal["transaction", "name_prefix", "db_schema"]


TabT = TypeVar("TabT", bound=Record)


@dataclass
class Table(Registry[TabT, RwxT, "DataBase"]):
    """Table matching a table model, may be filtered."""

    input_data: InputData | None = None

    def _id(self) -> str:
        """Identity of the data object."""
        # TODO: implement this for Table class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> AutoIdx[TabT]:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2, Any, Any, Root, *tuple[Any, ...]],
    ) -> Frame[PL, SxT2]:
        """Get SQL expression or Polars data of this property."""
        raise NotImplementedError()

    def _mutation(
        self: Table[Any, RwxT2],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[RwxT2]] = {C, U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        tables = {rec: table for rec, (table, _) in self._base_table_map.items()}
        base = self.root()

        assert has_type(
            input_data,
            TableInput,
        )

        if has_type(input_data, Record | Iterable[Record]):
            input_data = records_to_df(input_data)
        elif has_type(input_data, Mapping[Hashable, Record]):
            input_data = records_to_df(input_data.values())
        elif has_type(input_data, Mapping[str, pl.Series]):
            input_data = pl.concat(
                [s.rename(k) for k, s in input_data.items()], how="horizontal"
            )
        elif has_type(input_data, Mapping[str, pd.Series]):
            input_data = pd.concat(input_data, axis="columns")
        elif has_type(input_data, Mapping[str, sqla.ColumnElement]):
            input_data = sqla.select(*(c.label(k) for k, c in input_data.items()))

        if isinstance(input_data, pl.DataFrame | pd.DataFrame):
            input_sql = self._upload_df(input_data)
        else:
            assert isinstance(input_data, sqla.Select | sqla.FromClause)
            input_sql = input_data

        statements: list[sqla.Executable] = []

        if D in mode:
            # Delete current records first.
            statements += [
                safe_delete(table, input_sql, base.connection.engine)
                for table in tables.values()
            ]

        if {C, R, U} & mode:
            for base_type, table in tables.items():
                # Rename input columns to match base table columns.
                input_sql = input_sql.select().with_only_columns(
                    *(
                        input_sql.c[col.name].label(name)
                        for name, col in self._base_table_map[base_type][
                            0
                        ].columns.items()
                    )
                )

                if U in mode:
                    statements.append(
                        safe_update(table, input_sql, base.connection.engine)
                    )
                else:
                    statements.append(
                        safe_insert(
                            table,
                            input_sql,
                            base.connection.engine,
                            upsert=(U in mode and C in mode),
                        )
                    )

        return statements

    def validate(
        self, inspector: sqla.Inspector | None = None, required: bool = False
    ) -> None:
        """Perform pre-defined schema validations."""
        inspector = inspector or sqla.inspect(self.root().engine)

        for table, _ in self._base_table_map.values():
            validate_sql_table(inspector, table, required)

    @cached_prop
    def _base_table_map(
        self,
    ) -> dict[type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]]:
        """Return the base SQLAlchemy table objects for all record types."""
        base = self.root()
        base_table_map: dict[
            type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]
        ] = {}

        rec_list = list(
            sorted(
                self.value_type_set,
                key=cmp_to_key(
                    lambda x, y: (
                        -1 if issubclass(y, x) else 1 if issubclass(x, y) else 0
                    )
                ),
            )
        )

        first_rec = rec_list[0]
        first_table = base._get_base_table(first_rec)
        base_table_map[first_rec] = (first_table, None)

        for leaf_rec in rec_list[1:]:
            leaf_table = base._get_base_table(leaf_rec)
            base_table_map[leaf_rec] = (
                leaf_table,
                reduce(
                    sqla.and_,
                    (
                        pk_first == pk_leaf
                        for pk_first, pk_leaf in zip(
                            first_table.primary_key.columns,
                            leaf_table.primary_key.columns,
                            strict=False,
                        )
                    ),
                ),
            )

            super_recs = (
                r for r in leaf_rec._model_superclasses if issubclass(r, Record)
            )
            for super_rec in super_recs:
                if super_rec not in base_table_map:
                    super_table = base._get_base_table(leaf_rec)
                    base_table_map[super_rec] = (
                        super_table,
                        reduce(
                            sqla.and_,
                            (
                                pk_leaf == pk_super
                                for pk_leaf, pk_super in zip(
                                    leaf_table.primary_key.columns,
                                    super_table.primary_key.columns,
                                    strict=True,
                                )
                            ),
                        ),
                    )

        return dict(
            sorted(
                base_table_map.items(),
                key=cmp_to_key(
                    lambda x, y: (
                        -1
                        if issubclass(y[0], x[0])
                        else 1 if issubclass(x[0], y[0]) else 0
                    )
                ),
            )
        )

    def _get_base_union(self) -> sqla.FromClause:
        """Recursively join all bases of this record to get the full data."""
        bm_vals = list(self._base_table_map.values())

        joined = reduce(
            lambda x, y: x.join(y[0], y[1]), bm_vals[1:], bm_vals[0][0]
        ).select()

        split = {
            rec._fqn: joined.with_only_columns(*tab.columns)
            for rec, (tab, _) in self._base_table_map.items()
        }

        coalesced = frame_coalesce(Frame(split)).get()

        query = coalesced.subquery()
        setattr(query, "_tab", self)
        return query

    def _get_upload_table(self) -> sqla.Table:
        cols = {
            name: sqla.Column(col.key, col.type)
            for name, col in self.select().c.items()
        }

        return sqla.Table(
            f"upload/{self.fqn.replace('.', '_')}/{token_hex(5)}",
            self.root()._metadata,
            *cols.values(),
        )

    def _upload_df(self, df: pd.DataFrame | pl.DataFrame) -> sqla.Table:
        table = self._get_upload_table()

        if isinstance(df, pl.DataFrame):
            df.write_database(
                str(table),
                self.root().connection,
                if_table_exists="replace",
            )
        else:
            df.reset_index().to_sql(
                table.name,
                self.root().connection,
                if_exists="replace",
                index=False,
            )

        return table


class DynRecordMeta(ModelMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(self, name: str) -> Attr:
        """Get dynamic attribute by dynamic name."""
        return Attr(alias=name, context=Interface(self))

    def __getattr__(self, name: str) -> Attr:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)
        return Attr(alias=name, context=Interface(self))


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


x = DynRecord


def dynamic_record_type[T: Record](
    base: type[T] | tuple[type[T], ...],
    name: str,
    props: Iterable[Prop] = [],
    src_module: ModuleType | None = None,
    extra_attrs: dict[str, Any] = {},
) -> type[T]:
    """Create a dynamically defined record type."""
    base = base if isinstance(base, tuple) else (base,)
    return cast(
        type[T],
        new_class(
            name,
            base,
            None,
            lambda ns: ns.update(
                {
                    **{p.name: p for p in props},
                    "__annotations__": {
                        p.name: p.typeref.single_typedef
                        for p in props
                        if p.typeref is not None
                    },
                    "_src_mod": src_module or base[0]._src_mod or getmodule(base[0]),
                    **extra_attrs,
                }
            ),
        ),
    )


class HashRec(Record[str]):
    """Record type with a default hashed primary key."""

    _template = True

    def __post_init__(self) -> None:  # noqa: D105
        setattr(
            self,
            self._primary_key().attrs[0].name,
            gen_str_hash(
                {a.name: getattr(self, a.name) for a in type(self)._attrs().values()}
            ),
        )


class Entity(Record[UUID4]):
    """Record type with a default UUID4 primary key."""

    _template = True
    _pk: Key[UUID4] = Key(default_factory=lambda: (UUID4(uuid4()),))


type SingleJoinMap = (Prop | SupportsItems[Prop, Prop] | Set[Prop])

LnT = TypeVar(
    "LnT",
    bound=Record | Mapping[Hashable, Record] | Iterable[Record] | None,
    default=Any,
)


@dataclass(kw_only=True, eq=False)
class Link(
    Var[LnT, Idx[()], SQL, Tab, LnT, RecT],
    Generic[LnT, RecT],
):
    """Link to a single record."""

    on: (
        SingleJoinMap
        | SupportsItems[type[Record], SingleJoinMap]
        | (
            Prop[RecT, Idx[()], Any, Any, Any, RecT, Any]
            | Set[Prop[RecT, Idx[()], Any, Any, Any, RecT, Any]]
        )
        | None
    ) = None

    @cached_prop
    def join_maps(self) -> SupportsItems[type[Record], SupportsItems[Prop, Prop]]:
        """Return the foreign key for this link."""
        if self.on is None:
            return {
                rec_type: {
                    Attr(
                        alias=f"{rec_type._fqn}.{pk.name}",
                        typeref=pk.typeref,
                    ): pk
                    for pk in rec_type._primary_key().attrs
                }
                for rec_type in self.value_type_set
                if issubclass(rec_type, Record)
            }

        if has_type(self.on, SingleJoinMap):
            rec_type = self.common_value_type
            assert issubclass(rec_type, Record)
            self.on = {rec_type: self.on}

        assert has_type(self.on, Mapping[type[Record], SingleJoinMap])

        return (
            {
                rec_type: {
                    Attr(
                        alias=f"{rec_type._fqn}.{fk.name}",
                        typeref=fk.typeref,
                    ): fk
                }
                for rec_type, fk in self.on.items()
                if isinstance(fk, Prop)
            }
            | {
                rec_type: {
                    Attr(
                        alias=f"{rec_type._fqn}.{fk.name}",
                        typeref=fk.typeref,
                    ): fk
                    for fk in fks
                }
                for rec_type, fks in self.on.items()
                if has_type(fks, Set[Prop])
            }
            | {
                rec_type: fk_map
                for rec_type, fk_map in self.on.items()
                if has_type(fk_map, Mapping[Prop, Prop])
            }
        )


TgtT = TypeVar("TgtT", bound="Record", covariant=True, default=Any)


class Edge(Record[str, str], Generic[RecT, TgtT]):
    """Automatically defined relation record type."""

    _template = True

    _from: Link[RecT] = Link()
    _to: Link[RecT] = Link()

    _pk: Key[str, str] = Key(
        attrs=(
            *(
                a
                for fk_map in _from.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            *(
                a
                for fk_map in _to.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
        )
    )


class LabelEdge(Record[str, str, KeyT], Generic[RecT, TgtT, KeyT]):
    """Automatically defined relation record type with index substitution."""

    _template = True

    _from: Link[RecT] = Link()
    _to: Link[RecT] = Link()
    _rel_idx: Attr[KeyT] = Attr()

    _pk: Key[str, str, KeyT] = Key(
        attrs=(
            *(
                a
                for fk_map in _from.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            *(
                a
                for fk_map in _to.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            _rel_idx,
        )
    )


EdgeT = TypeVar("EdgeT", bound=Edge | LabelEdge, default=Any)


@dataclass(kw_only=True, eq=False)
class Rel(Var[LnT, IdxT, SQL, Col, TgtT, RecT], Generic[LnT, EdgeT, IdxT, TgtT, RecT]):
    """Backlink record set."""

    backlink: (
        Prop[RecT, Idx[()], Any, Any, Any, RecT, TgtT]
        | set[Prop[RecT, Idx[()], Any, Any, Any, RecT, TgtT]]
    )
    link: (
        Prop[RecT, Idx[()], Any, Any, Any, RecT, TgtT]
        | set[Prop[RecT, Idx[()], Any, Any, Any, RecT, TgtT]]
    )


class Item(Record[KeyT], Generic[ValT, KeyT, RecT]):
    """Dynamically defined scalar record type."""

    _template = True
    _array: ClassVar[Array]

    _from: Link[RecT] = Link()
    _idx: Attr[KeyT] = Attr()
    _val: Attr[ValT]

    _pk: Key[*tuple[Any, ...], KeyT] = Key(
        attrs=(
            *(
                a
                for fk_map in _from.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            _idx,
        )
    )


_item_classes: dict[str, type[Item]] = {}


@dataclass(eq=False)
class Array(Var[ValT, IdxT, SQL, Col, ValT, RecT], Generic[ValT, IdxT, RecT]):
    """Set / array of scalar values."""

    @cached_prop
    def _key_type(self) -> type:
        idx_type = self.typeargs[KeyT]
        key_tuple = get_args(idx_type)
        return key_tuple[0] if len(key_tuple) == 1 else tuple[*key_tuple]

    @cached_prop
    def relation_type(self) -> type[Item[ValT, KeyT, RecT]]:
        """Return the dynamic relation record type."""
        assert self.owner is not None
        base_array_fqn = self.fqn

        rec = _item_classes.get(
            base_array_fqn,
            dynamic_record_type(
                Item[self.common_value_type, self._key_type, self.owner],
                f"{self.owner.__name__}.{self.name}",
                src_module=self.owner._src_mod or getmodule(self.owner),
                extra_attrs={"_array": self},
            ),
        )
        _item_classes[base_array_fqn] = rec

        return rec


class Schema:
    """Group multiple record types into a schema."""

    _schema_types: set[type[Record]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        cls._schema_types = set()
        super().__init_subclass__()


@dataclass
class Require:
    """Mark schema or record type as required."""

    present: bool = True


BackT = TypeVar("BackT", bound=LiteralString | None, default=None)


@dataclass(eq=False)
class DataBase(
    Data[Record, Idx[*tuple[Any, ...]], Tabs, SQL, RwxT, Base[Record, RwxT]],
    Base[Record, RwxT],
    Generic[BackT, RwxT],
):
    """Database connection."""

    context: Base | Data[Any, Any, Any, Any, Any, Base] = field(init=False)

    backend: BackT = None  # pyright: ignore[reportAssignmentType]
    """Unique name to identify this database's backend by."""
    url: sqla.URL | CloudPath | Path | None = None
    """Connection URL or path."""

    schema: (
        type[Schema]
        | Mapping[
            type[Schema] | type[Record],
            Literal[True] | Require | str | sqla.TableClause,
        ]
        | Set[type[Schema] | type[Record]]
        | None
    ) = None
    validate_on_init: bool = False

    overlay: Literal[True] | str | None = None
    overlay_type: OverlayType = "transaction"

    remove_cross_fks: bool = False

    _rec_types: dict[type[Record], Literal[True] | Require | str | sqla.TableClause] = (
        field(default_factory=dict)
    )
    _subs: dict[type[Record], sqla.TableClause] = field(default_factory=dict)

    _metadata: sqla.MetaData = field(default_factory=sqla.MetaData)
    _valid_caches: dict[type[Record], set[Hashable]] = field(default_factory=dict)
    _instance_map: dict[type[Record], dict[Hashable, Record]] = field(
        default_factory=dict
    )

    _db_id: str | None = None
    _overlay_name: str | None = None
    _transaction: sqla.Transaction | None = None

    def __post_init__(self):  # noqa: D105
        self.context = self

        tables: dict[type[Record], Literal[True] | Require | str | sqla.TableClause] = {
            tab: tab_def
            for tab, tab_def in self._schema_map.items()
            if issubclass(tab, Record)
        }

        # Handle Record classes in schema argument.
        self._subs = {
            **self._subs,
            **{
                tab: (sub if isinstance(sub, sqla.TableClause) else sqla.table(sub))
                for tab, sub in tables.items()
                if not isinstance(sub, Require | bool)
            },
        }
        self._rec_types |= tables

        self._overlay_name = (
            f"{self.db_id}_overlay_{token_hex(4)}"
            if self.overlay is True
            else self.overlay
        )

        if self._overlay_name is not None and self.overlay_type == "db_schema":
            self._ensure_sql_schema_exists(self._overlay_name)

        if self._overlay_name is not None and self.overlay_type == "transaction":
            self._transaction = sqla.Transaction(self.engine.connect())

        if self.validate_on_init:
            self.validate()

    def _id(self) -> str:
        """Identity of the data object."""
        raise NotImplementedError()

    def _index(
        self,
    ) -> Idx:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2, Any, Any, Root, *tuple[Any, ...]],
    ) -> Frame[PL, SxT2]:
        """Get SQL expression or Polars data of this property."""
        raise NotImplementedError()

    def _mutation(
        self: Data[Any, Any, Any, Any, RwxT2],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[RwxT2]] = {C, U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        raise NotImplementedError()

    @property
    def db_id(self) -> str:
        """Return the unique database ID."""
        db_id = self._db_id
        if db_id is None:
            if self.backend is None:
                db_id = token_hex(4)
            else:
                db_id = self.backend + (
                    gen_str_hash(self.url) if self.url is not None else ""
                )

        self._db_id = db_id
        return db_id

    @cached_prop
    def backend_type(
        self,
    ) -> Literal["sql-connection", "sqlite-file", "excel-file", "in-memory"]:
        """Type of the backend."""
        match self.url:
            case Path() | CloudPath():
                return "excel-file" if "xls" in self.url.suffix else "sqlite-file"
            case sqla.URL():
                return (
                    "sqlite-file"
                    if self.url.drivername == "sqlite"
                    else "sql-connection"
                )
            case _:
                return "in-memory"

    @cached_prop
    def engine(self) -> sqla.engine.Engine:
        """SQLA Engine for this DB."""
        # Create engine based on backend type
        # For Excel-backends, use sqlite in-memory engine
        return sqla.create_engine(
            (
                (self.url if isinstance(self.url, sqla.URL) else str(self.url))
                if self.backend_type in ("sql-connection", "sqlite-file")
                else f"sqlite:///file:{self.db_id}?uri=true&mode=memory&cache=shared"
            ),
        )

    @cached_prop
    def df_engine(self) -> sqla.engine.Engine:
        """SQLA Engine for this DB."""
        # Create engine based on backend type
        # For Excel-backends, use sqlite in-memory engine
        return sqla.create_engine(
            (
                (self.url if isinstance(self.url, sqla.URL) else str(self.url))
                if self.backend_type in ("sql-connection", "sqlite-file")
                else f"sqlite:///file:{self.db_id}?uri=true&mode=memory&cache=shared"
            ),
            connect_args=(
                {"detect_types": PARSE_DECLTYPES}
                if self.backend_type in ("sqlite-file", "in-memory", "excel-file")
                else {}
            ),
        )

    @property
    def connection(self) -> sqla.engine.Connection:
        """SQLA Connection for this DB."""
        return (
            self._transaction.connection
            if self._transaction is not None
            else self.engine.connect()
        )

    def registry[T: AutoIndexable](
        self: Base[T], value_type: type[T]
    ) -> Registry[T, RwxT, Base[T, RwxT]]:
        """Get the registry for a type in this base."""
        ...

    def table[T: Record](self, value_type: type[T]) -> Table[T, RwxT]:
        """Get the registry for a type in this base."""
        ...

    def validate(self) -> None:
        """Perform pre-defined schema validations."""
        recs = {rec: isinstance(req, Require) for rec, req in self._rec_types.items()}
        inspector = sqla.inspect(self.engine)

        # Iterate over all tables and perform validations for each
        for rec, req in recs.items():
            self.table(rec).validate(inspector, required=req)

    def commit(self) -> None:
        """Commit the transaction."""
        if self._transaction is not None:
            self._transaction.commit()
            self._transaction = None
        # Note: Commiting overlays of other types than "transaction"
        # is not supported yet.

    @contextmanager
    def edit(
        self: DataBase[Any, C | U | D], overlay_type: OverlayType = "transaction"
    ) -> Generator[DataBase[BackT, RwxT]]:
        """Context manager to create temp overlay of base and auto-commit on exit."""
        assert self.overlay is None, "Cannot edit base with already active overlay."
        edit_base = copy_and_override(
            DataBase[BackT, RwxT], self, overlay=True, backend=self.backend
        )

        try:
            yield edit_base
        finally:
            edit_base.commit()

    @cached_prop
    def _schema_map(
        self,
    ) -> Mapping[
        type[Schema | Record], Literal[True] | Require | str | sqla.TableClause
    ]:
        return (
            {self.schema: True}
            if isinstance(self.schema, type)
            else (
                cast(
                    dict[type[Schema | Record], Literal[True]],
                    {rec: True for rec in self.schema},
                )
                if isinstance(self.schema, Set)
                else self.schema if self.schema is not None else {}
            )
        )

    @cached_method
    def _get_base_table_name(
        self,
        rec_type: type[Record],
    ) -> str:
        """Return the name of the table for this class."""
        sub = self._subs.get(rec_type)
        return sub.name if sub is not None else rec_type._default_table_name()

    def _get_base_table(
        self,
        rec_type: type[Record],
        mode: Set[type[C | R | U | D]] = {R},
    ) -> sqla.Table:
        """Return the base SQLAlchemy table object for this data's record type."""
        orig_table: sqla.Table | None = None

        if (
            {C, U, D} & mode
            and self._overlay_name is not None
            and rec_type not in self._subs
        ):
            orig_table = self._get_base_table(rec_type, {R})

            # Create an empty overlay table for the record type
            self._subs[rec_type] = sqla.table(
                (
                    (self._overlay_name + "/" + rec_type._default_table_name())
                    if self.overlay_type == "name_prefix"
                    else rec_type._default_table_name()
                ),
                schema=(
                    self._overlay_name if self.overlay_type == "db_schema" else None
                ),
            )

        table_name = self._get_base_table_name(rec_type)

        if table_name in self._metadata.tables:
            # Return the table object from metadata if it already exists.
            # This is necessary to avoid circular dependencies.
            return self._metadata.tables[table_name]

        sub = self._subs.get(rec_type)

        # Create a partial SQLAlchemy table object from the class definition
        # without foreign keys to avoid circular dependencies.
        # This adds the table to the metadata.
        sqla.Table(
            table_name,
            self._metadata,
            *self._map_cols(rec_type).values(),
            self._map_pk(rec_type),
            *self._map_keys(rec_type).values(),
            schema=(sub.schema if sub is not None else None),
        )

        # Re-create the table object with foreign keys and return it.
        table = sqla.Table(
            table_name,
            self._metadata,
            *self._map_fks(rec_type).values(),
            schema=(sub.schema if sub is not None else None),
            extend_existing=True,
        )

        self._create_sql_table(table)

        if orig_table is not None and mode == {C, U}:
            with self.engine.begin() as conn:
                conn.execute(
                    sqla.insert(table).from_select(
                        orig_table.columns.keys(), orig_table.select()
                    )
                )

        if self.backend_type == "excel-file":
            self._load_base_table_from_excel(table)

        return table

    @cached_method
    def _map_cols(self, rec_type: type[Record]) -> dict[Attr, sqla.Column]:
        """Columns of this record type's table."""
        registry = orm.registry(
            metadata=self._metadata,
            type_annotation_map=rec_type._type_map,
        )

        return {
            attr: sqla.Column(
                attr.name,
                registry._resolve_type(
                    attr.value_typeform  # pyright: ignore[reportArgumentType]
                ),
                autoincrement=False,
                nullable=has_type(None, attr.value_typeform),
            )
            for attr in rec_type._attrs().values()
        }

    @cached_method
    def _map_keys(self, rec_type: type[Record]) -> dict[Key, sqla.Index]:
        """Columns of this record type's table."""
        return {
            k: sqla.Index(
                f"{self._get_base_table_name(rec_type)}_key_{k.name}",
                *(col.name for col in k.attrs),
                unique=True,
            )
            for k in rec_type._keys().values()
        }

    @cached_method
    def _map_pk(self, rec_type: type[Record]) -> sqla.PrimaryKeyConstraint:
        """Columns of this record type's table."""
        pk = rec_type._primary_key()
        return sqla.PrimaryKeyConstraint(
            *[col.name for col in pk.attrs],
        )

    @cached_method
    def _map_fks(self, rec_type: type[Record]) -> dict[Link, sqla.ForeignKeyConstraint]:
        fks: dict[Link, sqla.ForeignKeyConstraint] = {}

        for link in rec_type._links().values():
            target_type = cast(type[Record], link.common_value_type)
            target_table = self._get_base_table(target_type)

            fks[link] = sqla.ForeignKeyConstraint(
                [
                    col.name
                    for fk_map in link.join_maps.values()
                    for col in fk_map.keys()
                    if isinstance(col, Attr)
                ],
                [
                    target_table.c[col.name]
                    for fk_map in link.join_maps.values()
                    for col in fk_map.values()
                    if isinstance(col, Attr)
                ],
                name=f"{self._get_base_table_name(rec_type)}_fk_{link.name}_{target_type._fqn}",
            )

        return fks

    def _get_valid_cache_set(self, rec_type: type[Record]) -> set[Hashable]:
        """Get the valid cache set for a record type."""
        if rec_type not in self._valid_caches:
            self._valid_caches[rec_type] = set()

        return self._valid_caches[rec_type]

    def _get_instance_map(self, rec_type: type[Record]) -> dict[Hashable, Record]:
        """Get the instance map for a record type."""
        if rec_type not in self._instance_map:
            self._instance_map[rec_type] = {}

        return self._instance_map[rec_type]

    def _load_base_table_from_excel(self, table: sqla.Table) -> None:
        """Load all tables from Excel."""
        assert isinstance(self.url, Path | CloudPath)
        path = self.url

        if isinstance(path, Path) and not path.exists():
            return

        with self.engine.begin() as conn:
            conn.execute(table.delete().where(sqla.true()))

        with open(path, "rb") as file:
            df = pl.read_excel(file, sheet_name=table.name)

        df.write_database(
            str(table),
            self.df_engine,
            if_table_exists="append",
        )

    def _create_sql_table(self, table: sqla.Table) -> None:
        """Create SQL-side table from Table class."""
        if self.remove_cross_fks:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            table = table.to_metadata(sqla.MetaData())  # temporary metadata
            remove_cross_fk(table)

        table.create(self.engine, checkfirst=True)

    def _ensure_sql_schema_exists(self, schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_schema(schema_name):
            with self.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name

    def _save_to_excel(self) -> None:
        """Save all (or selected) tables to Excel."""
        assert isinstance(self.url, Path | CloudPath)
        file = self.url

        with ExcelWorkbook(file) as wb:
            for table in self._metadata.tables.values():
                pl.read_database(
                    table.select(),
                    self.engine.connect(),
                ).write_excel(wb, worksheet=table.name)

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(
            (
                self.db_id,
                self.url,
                self._subs,
            )
        )

    def graph(
        self,
    ) -> nx.Graph:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        raise NotImplementedError("Graph export is not implemented yet.")
