"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import MISSING, Field, asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import cache, partial, reduce
from inspect import get_annotations, getmodule
from io import BytesIO
from itertools import groupby
from pathlib import Path
from secrets import token_hex
from sqlite3 import PARSE_DECLTYPES
from types import ModuleType, NoneType, UnionType, new_class
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ForwardRef,
    Generic,
    Literal,
    LiteralString,
    ParamSpec,
    Self,
    TypeAliasType,
    TypeVarTuple,
    Union,
    cast,
    dataclass_transform,
    final,
    get_args,
    get_origin,
    overload,
)
from uuid import uuid4

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.dialects.mysql as mysql
import sqlalchemy.dialects.postgresql as postgresql
import sqlalchemy.dialects.sqlite as sqlite
import sqlalchemy.orm as orm
import sqlalchemy.sql.selectable as sqla_sel
import sqlalchemy.sql.visitors as sqla_visitors
import sqlparse
import yarl
from bidict import bidict
from cloudpathlib import CloudPath
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from typing_extensions import TypeVar
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.caching import cached_prop
from py_research.data import copy_and_override
from py_research.files import HttpFile
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericProtocol,
    SingleTypeDef,
    get_lowest_common_base,
    has_type,
    is_subtype,
)

from .sqlite_utils import register_sqlite_adapters

register_sqlite_adapters()


class UUID4(str):
    """UUID4 string."""


RecT = TypeVar("RecT", bound="Record", covariant=True, default=Any)
RecT2 = TypeVar("RecT2", bound="Record")
RecT3 = TypeVar("RecT3", bound="Record")
RecT4 = TypeVar("RecT4", bound="Record")

RefT = TypeVar("RefT", bound="Record | None", covariant=True)
RefT2 = TypeVar("RefT2", bound="Record | None")

RelT = TypeVar("RelT", bound="Record | None", covariant=True, default=None)
RelT2 = TypeVar("RelT2", bound="Record | None")
RelT3 = TypeVar("RelT3", bound="Record | None")

OwnT = TypeVar("OwnT", bound="Record", default=Any)
ParT = TypeVar("ParT", contravariant=True, bound="Record", default=Any)

SchemaT = TypeVar("SchemaT", bound="Record | Schema", covariant=True)
SchemaT2 = TypeVar("SchemaT2", bound="Record | Schema")


type Ordinal = (
    bool | int | float | Decimal | datetime | date | time | timedelta | UUID4 | str
)

OrdT = TypeVar("OrdT", bound=Ordinal)


@final
class Undef:
    """Demark undefined status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class Keep:
    """Demark unchanged status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


type Input[Val, Key: Hashable] = pd.DataFrame | pl.DataFrame | Iterable[Val] | Mapping[
    Key, Val
] | sqla.Select | Val

ValT = TypeVar("ValT", covariant=True)
ValT2 = TypeVar("ValT2")
ValT3 = TypeVar("ValT3")
ValTt = TypeVarTuple("ValTt")
ValTt2 = TypeVarTuple("ValTt2")
ValTt3 = TypeVarTuple("ValTt3")

KeyT = TypeVar("KeyT", bound=Hashable, default=Any)
KeyT2 = TypeVar("KeyT2", bound=Hashable)
KeyT3 = TypeVar("KeyT3", bound=Hashable)
KeyTt = TypeVarTuple("KeyTt")
KeyTt2 = TypeVarTuple("KeyTt2")
KeyTt3 = TypeVarTuple("KeyTt3")
KeyTt4 = TypeVarTuple("KeyTt4")

type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]


@final
class Idx[*K]:
    """Define the custom index type of a dataset."""


@final
class RootIdx[T]:
    """Database root index."""


@final
class BaseIdx[R: Record]:
    """Singleton to mark dataset as having the record type's base index."""


@final
class ArrayIdx[R: Record, *K]:
    """Record type's base index + custom array index."""


IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound=Idx | RootIdx | BaseIdx | ArrayIdx,
    default=BaseIdx,
)
IdxT2 = TypeVar(
    "IdxT2",
    bound=Idx | RootIdx | BaseIdx | ArrayIdx,
)


class C:
    """Singleton to allow creation of new records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


class R:
    """Singleton to allow reading of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


class RU(R):
    """Singleton to allow reading and updating of records."""


class CR(C, R):
    """Singleton to allow creation and reading of records."""


class CRU(C, RU):
    """Singleton to allow creation, reading, and updating of records."""


@final
class CRUD(CRU):
    """Singleton to allow creation, reading, updating, and deleting of records."""


CrudT = TypeVar("CrudT", bound=C | R, default=CRUD, covariant=True)
CrudT2 = TypeVar("CrudT2", bound=C | R)
CrudT3 = TypeVar("CrudT3", bound=C | R)

RwT = TypeVar("RwT", bound=C | R, default=RU, covariant=True)


@final
@dataclass
class Ctx(Generic[ParT]):
    """Context record of a dataset."""

    record_type: type[ParT]


CtxT = TypeVar("CtxT", bound="Ctx| None", covariant=True, default=None)
CtxT2 = TypeVar("CtxT2", bound="Ctx | None")
CtxT3 = TypeVar("CtxT3", bound="Ctx | None")


@final
class Symbolic:
    """Local backend."""


type DynBackendID = LiteralString | None


BaseT = TypeVar(
    "BaseT",
    bound=DynBackendID | Symbolic,
    covariant=True,
    default=None,
)
BaseT2 = TypeVar(
    "BaseT2",
    bound=DynBackendID | Symbolic,
)

BackT = TypeVar(
    "BackT",
    bound=DynBackendID,
    covariant=True,
    default=None,
)
BackT2 = TypeVar(
    "BackT2",
    bound=DynBackendID,
)
BackT3 = TypeVar(
    "BackT3",
    bound=DynBackendID,
)

# OverT = TypeVar("OverT", bound=LiteralString | None, default=None)


# class Overlay(Generic[BackT, OverT]):
#     """Overlay type."""


# class DynOverlay(Overlay[BackT, None]):
#     """Dynamic overlay type."""


DfT = TypeVar("DfT", bound=pd.DataFrame | pl.DataFrame)

Params = ParamSpec("Params")


type LinkItem = Table[Record | None, None, Any, Any, Any, Any]

type PropPath[LeafT] = tuple[Data[LeafT, Any, Any, None, None, Any]] | tuple[
    Table[Record, None, Any, Any, None, Any],
    *tuple[Table[Record | None, Any, Any, Any, Ctx, Any], ...],
    Data[LeafT, Any, Any, Any, Ctx, Any],
]
type JoinDict = dict[LinkItem, JoinDict]

type SqlJoin = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


type OverlayType = Literal["name_prefix", "db_schema"]


type AggMap[Rec: Record] = dict[
    Value[Any, Any, Any, Rec, Any],
    Value[Any, Any, Any, Any, Any] | sqla.Function,
]


@dataclass(kw_only=True, frozen=True)
class Agg(Generic[RecT]):
    """Define an aggregation map."""

    target: type[RecT]
    map: AggMap[RecT]


_pl_type_map: dict[
    Any, pl.DataType | type | Callable[[SingleTypeDef | UnionType], pl.DataType]
] = {
    **{
        t: t
        for t in (
            int,
            float,
            complex,
            str,
            bool,
            datetime,
            date,
            time,
            timedelta,
            bytes,
        )
    },
    Literal: lambda t: pl.Enum([str(a) for a in get_args(t)]),
    UUID4: pl.String,
}


def _get_pl_schema(
    col_map: Mapping[str, Value[Any, Any, Any, Any, Any]]
) -> dict[str, pl.DataType | type | None]:
    """Return the schema of the dataset."""
    exact_matches = {
        name: (_pl_type_map.get(col.value_type), col) for name, col in col_map.items()
    }
    matches = {
        name: (
            (match, col.value_type)
            if match is not None
            else (_pl_type_map.get(col.target_type), col.value_type)
        )
        for name, (match, col) in exact_matches.items()
    }

    return {
        name: (
            match if isinstance(match, pl.DataType | type | None) else match(match_type)
        )
        for name, (match, match_type) in matches.items()
    }


def _pd_to_py_dtype(c: pd.Series | pl.Series) -> type | None:
    """Map pandas dtype to Python type."""
    if isinstance(c, pd.Series):
        if is_datetime64_dtype(c):
            return datetime
        elif is_bool_dtype(c):
            return bool
        elif is_integer_dtype(c):
            return int
        elif is_numeric_dtype(c):
            return float
        elif is_string_dtype(c):
            return str
    else:
        if c.dtype.is_temporal():
            return datetime
        elif c.dtype.is_integer():
            return int
        elif c.dtype.is_float():
            return float
        elif c.dtype.is_(pl.String):
            return str

    return None


def _sql_to_py_dtype(c: sqla.ColumnElement) -> type | None:
    """Map sqla column type to Python type."""
    match c.type:
        case sqla.DateTime():
            return datetime
        case sqla.Date():
            return date
        case sqla.Time():
            return time
        case sqla.Boolean():
            return bool
        case sqla.Integer():
            return int
        case sqla.Float():
            return float
        case sqla.String() | sqla.Text() | sqla.Enum():
            return str
        case sqla.LargeBinary():
            return bytes
        case _:
            return None


def _gen_prop(
    name: str,
    data: pd.Series | pl.Series | sqla.ColumnElement,
    pk: bool = False,
    fks: Mapping[str, Value[Any, Any, Any, Any, Symbolic]] = {},
) -> Data[Any, Any, Any, Any, Any, Any]:
    is_link = name in fks
    value_type = (
        _pd_to_py_dtype(data)
        if isinstance(data, pd.Series | pl.Series)
        else _sql_to_py_dtype(data)
    ) or Any
    attr = Value(
        primary_key=pk,
        _name=name if not is_link else f"fk_{name}",
        _type=Value[value_type],
    )
    return (
        attr
        if not is_link
        else Link(fks=fks[name], _type=fks[name]._type, _name=f"link_{name}")
    )


def props_from_data(
    data: pd.DataFrame | pl.DataFrame | sqla.Select,
    foreign_keys: Mapping[str, Value[Any, Any, Any, Any, Symbolic]] | None = None,
    primary_keys: list[str] | None = None,
) -> list[Data]:
    """Extract prop definitions from dataframe or query."""
    foreign_keys = foreign_keys or {}

    if isinstance(data, pd.DataFrame):
        data.index.rename(
            [n or f"index_{i}" for i, n in enumerate(data.index.names)], inplace=True
        )
        primary_keys = primary_keys or list(data.index.names)
        data = data.reset_index()

    primary_keys = primary_keys or []

    columns = (
        [data[col] for col in data.columns]
        if isinstance(data, pd.DataFrame | pl.DataFrame)
        else list(data.columns)
    )

    return [
        _gen_prop(str(col.name), col, col.name in primary_keys, foreign_keys)
        for col in columns
    ]


def _remove_cross_fk(table: sqla.Table):
    """Dirty vodoo to remove external FKs from existing table."""
    for c in table.columns.values():
        c.foreign_keys = set(
            fk
            for fk in c.foreign_keys
            if fk.constraint and fk.constraint.referred_table.schema == table.schema
        )

    table.foreign_keys = set(  # pyright: ignore[reportAttributeAccessIssue]
        fk
        for fk in table.foreign_keys
        if fk.constraint and fk.constraint.referred_table.schema == table.schema
    )

    table.constraints = set(
        c
        for c in table.constraints
        if not isinstance(c, sqla.ForeignKeyConstraint)
        or c.referred_table.schema == table.schema
    )


@cache
def _prop_type_name_map() -> dict[str, type[Data]]:
    return {cls.__name__: cls for cls in get_subclasses(Data) if cls is not Data}


def _map_prop_type_name(name: str) -> type[Data | None]:
    """Map property type name to class."""
    name_map = _prop_type_name_map()
    matches = [name_map[n] for n in name_map if name.startswith(n + "[")]
    return matches[0] if len(matches) == 1 else NoneType


_type_alias_classes: dict[TypeAliasType, type] = {}


@dataclass(kw_only=True, eq=False)
class Data(Generic[ValT, IdxT, CrudT, RelT, CtxT, BaseT]):
    """Relational dataset."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: 1,
        CrudT: 2,
        RelT: 3,
        CtxT: 4,
        BaseT: 5,
    }

    _name: str | None = None
    _type: (
        str
        | SingleTypeDef[Data[ValT, Any, Any, Any, Any, Any]]
        | SingleTypeDef[ValT]
        | None
    ) = None
    _typevar_map: dict[TypeVar, SingleTypeDef | UnionType] = field(default_factory=dict)

    _db: DataBase[Any, Any, CrudT | CRUD, BaseT] | None = None
    _ctx: CtxT | Data[Record | None, Any, R, Any, Any, BaseT] | None = None

    _filters: list[
        sqla.ColumnElement[bool]
        | list[tuple[Hashable, ...]]
        | tuple[slice | Hashable, ...]
    ] = field(default_factory=list)
    _tuple_selection: tuple[Data[Any, Any, Any, Any, Any, BaseT], ...] | None = None

    _sql_col: sqla.ColumnElement | None = None
    _sql_from: sqla_sel.NamedFromClause | None = None
    """Override the SQL from clause of this dataset."""

    @cached_prop
    def name(self) -> str:
        """Defined name or generated identifier of this dataset."""
        return self._name if self._name is not None else token_hex(5)

    @cached_prop
    def value_type(self) -> SingleTypeDef[ValT] | UnionType:
        """Value typehint of this dataset."""
        if is_subtype(self._generic_type, Data[Any, Any, Any, Any, Any, Any]):
            typearg = self._typearg_map[ValT]
            if isinstance(typearg, int):
                typearg = self._hint_to_typedef(self._generic_args[typearg])

            return typearg

        return self._generic_type

    @cached_prop
    def target_type(self) -> type[ValT]:
        """Value type of this dataset."""
        return self._typedef_to_type(self.value_type)

    @cached_prop
    def record_type(self: Data[RecT2 | None, Any, Any, Any, Any, Any]) -> type[RecT2]:
        """Record target type of this dataset."""
        t = self.target_type
        assert issubclass(t, Record)
        return t

    @cached_prop
    def relation_type(self) -> type[RelT]:
        """Relation record type, if any."""
        typearg = self._typearg_map[RelT]
        if isinstance(typearg, int):
            typearg = self._hint_to_typedef(self._generic_args[typearg])

        rec_type = self._typedef_to_type(typearg)
        if not issubclass(rec_type, Record):
            return cast(type[RelT], NoneType)

        return cast(type[RelT], rec_type)

    @cached_prop
    def db(self) -> DataBase[Any, Any, CrudT | CRUD, BaseT]:
        """Database, which this dataset belongs to."""
        if self._db is not None:
            return self._db

        db = cast(DataBase[Any, Any, CRUD, BaseT], DataBase())
        self._db = db
        return db

    @property
    def ctx(
        self: Data[Any, Any, Any, Any, Ctx[RecT2], Any]
    ) -> Table[RecT2, Any, Any, Any, Any, BaseT]:
        """Context table of this dataset."""
        assert self._ctx is not None
        return (
            cast(Table[RecT2, Any, Any, Any, Any, BaseT], self._ctx)
            if isinstance(self._ctx, Data)
            else Table(_db=self.db, _type=Table[self._ctx.record_type])
        )

    @cached_prop
    def path(self) -> PropPath[ValT]:
        """Relational path of this dataset."""
        if self._ctx_table is None:
            return (cast(Data[ValT, Any, Any, None, None, Any], self),)

        return cast(PropPath[ValT], (*self._ctx_table.path, self))

    @property
    def rel(
        self: Data[Any, Any, Any, RecT2, Ctx[RecT3], Any]
    ) -> BackLink[RecT2, Any, Any, RecT3, BaseT]:
        """Table pointing to the relations of this dataset, if any."""
        assert self._rel is not None
        return self._rel

    @cached_prop
    def x(self: Data[RecT2 | None, Any, Any, Any, Any, Any]) -> type[RecT2]:
        """Path-aware accessor to the record type of this dataset."""
        return cast(
            type[RecT2],
            new_class(
                self.record_type.__name__,
                (self.record_type,),
                None,
                lambda ns: ns.update(
                    {
                        "_ctx": self,
                        "_derivate": True,
                        "_src_mod": getmodule(self.record_type),
                    }
                ),
            ),
        )

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if self._ctx_table is None:
            if issubclass(self.target_type, Record):
                return self.target_type._fqn
            else:
                return self.name

        fqn = f"{self._ctx_table.fqn}.{self.name}"

        if len(self._filters) > 0:
            fqn += f"[{gen_str_hash(self._filters, length=6)}]"

        return fqn

    @property
    def tuple_selection(
        self: Data[tuple, Any, Any, Any, Any, Any]
    ) -> tuple[Data[Any, Any, Any, Any, Any, Any], ...]:
        """Tuple-selection of data in the sub-grpah, if any."""
        assert self._tuple_selection is not None
        return self._tuple_selection

    @cached_prop
    def select(
        self,
        cols: Mapping[str, Value[Any, Any, Any, Any, Any]] | None = None,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        if cols is not None:
            abs_cols = {
                name: (
                    col._prepend_ctx(self._table)
                    if isinstance(col.db.backend, Symbolic)
                    else col
                )
                for name, col in cols.items()
            }
        else:
            abs_cols = self._abs_idx_cols | self._abs_cols

        sql_cols: list[sqla.ColumnElement] = []
        for col_name, col in abs_cols.items():
            sql_col = col.sql_col.label(col_name)
            if sql_col not in sql_cols:
                sql_cols.append(sql_col)

        select = sqla.select(*sql_cols)

        if self._root._table is not None:
            select = select.select_from(self._root._table.sql_from)

        for join in self.sql_joins:
            select = select.join(*join)

        for filt in self.sql_filters:
            select = select.where(filt)

        return select.distinct()

    @cached_prop
    def query(
        self,
    ) -> sqla.Subquery:
        """Return select statement for this dataset."""
        return self.select.subquery()

    @property
    def select_str(self) -> str:
        """Return select statement for this dataset."""
        return sqlparse.format(
            str(self.select.compile(self.db.engine)),
            reindent=True,
            keyword_case="upper",
        )

    @property
    def sql_col(self: Data[Any, Any, Any, None, Ctx, Any]) -> sqla.ColumnElement:
        """Return the SQL column of this dataset."""
        if self._sql_col is not None:
            return self._sql_col

        col = sqla.column(
            self.name,
            _selectable=self._table.sql_from if self._table is not None else None,
        )
        setattr(col, "_data", self)

        return col

    @property
    def sql_from(
        self: Data[Record | None, Any, Any, Any, Any, Any]
    ) -> sqla_sel.NamedFromClause:
        """Recursively join all bases of this record to get the full data."""
        if self.db.backend_type == "excel-file":
            self.db._load_from_excel(
                {self.record_type, *self.record_type._record_superclasses}
            )

        if self._sql_from is not None:
            return self._sql_from

        base_table = self._gen_sql_base_table("read")

        if len(self.record_type._record_superclasses) == 0:
            table = base_table.alias(self.fqn)
        else:
            table = base_table
            cols = {col.name: col for col in base_table.columns}
            for superclass in self.record_type._record_superclasses:
                superclass_table = Table(_db=self.db, _type=Table[superclass]).sql_from
                cols |= {col.key: col for col in superclass_table.columns}

                table = table.join(
                    superclass_table,
                    reduce(
                        sqla.and_,
                        (
                            base_table.c[pk_name] == superclass_table.c[pk_name]
                            for pk_name in self.record_type._pk_values
                        ),
                    ),
                )

            table = (
                sqla.select(*(col.label(col_name) for col_name, col in cols.items()))
                .select_from(table)
                .alias(self.fqn)
            )

        self._sql_from = table
        return table

    @cached_prop
    def sql_joins(
        self,
        _subtree: JoinDict | None = None,
        _parent: Data[Record | None, Any, Any, Any, Any, Any] | None = None,
    ) -> list[SqlJoin]:
        """Extract join operations from the relational tree."""
        joins: list[SqlJoin] = []
        _subtree = _subtree if _subtree is not None else self._total_join_dict
        _parent = _parent if _parent is not None else self._root

        for target, next_subtree in _subtree.items():
            joins.append(
                (
                    target.sql_from,
                    reduce(
                        sqla.and_,
                        (
                            (
                                target[fk] == _parent[pk]
                                for fk, pk in target.link._fk_map.items()
                            )
                            if isinstance(target, BackLink)
                            else (
                                _parent[fk] == target[pk]
                                for fk, pk in target._fk_map.items()
                            )
                        ),
                    ),
                )
            )

            joins.extend(type(self).sql_joins(self, next_subtree, target))

        return joins

    @cached_prop
    def sql_filters(self) -> list[sqla.ColumnElement[bool]]:
        """Get the SQL filters for this table."""
        if not isinstance(self, Table):
            return []

        return self._abs_filters[0] + (
            self._ctx_table.sql_filters if self._ctx_table is not None else []
        )

    @overload
    def keys(  # pyright: ignore[reportOverlappingOverload]
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
    ) -> Sequence[KeyT2]: ...

    @overload
    def keys(
        self: Data[
            Any,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Sequence[tuple[*KeyTt2]]: ...

    @overload
    def keys(
        self: Data[
            Any,
            Idx[KeyT2] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Sequence[KeyT2 | tuple[*KeyTt2]]: ...

    @overload
    def keys(
        self: Data[
            Any,
            ArrayIdx[Record[*KeyTt2], KeyT3],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Sequence[tuple[*KeyTt2, KeyT3]]: ...

    def keys(
        self: Data[Any, Any, Any, Any, Any, Any],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        df = self.df(index_only=True)
        if len(self._abs_idx_cols) == 1:
            return [tup[0] for tup in df.iter_rows()]

        return list(df.iter_rows())

    def values(
        self: Data[ValT2, Any, Any, Any, Any, DynBackendID],
    ) -> Sequence[ValT2]:
        """Iterable over this dataset's values."""
        dfs = self.df()
        if isinstance(dfs, pl.DataFrame):
            dfs = (dfs,)

        selection = (
            list(self._tuple_selection) if self._tuple_selection is not None else [self]
        )

        valid_caches = {
            sel.record_type: self.db._get_valid_cache_set(sel.record_type)
            for sel in selection
            if isinstance(sel, Table)
        }
        instance_maps = {
            sel.record_type: self.db._get_instance_map(sel.record_type)
            for sel in selection
            if isinstance(sel, Table)
        }

        vals = []
        for rows in zip(*(df.iter_rows(named=True) for df in dfs)):
            rows = cast(tuple[dict[str, Any], ...], rows)

            val_list = []
            for sel, row in zip(selection, rows):
                if isinstance(sel, Table):
                    rec_type = sel.record_type
                    assert issubclass(rec_type, Record)

                    new_rec = rec_type(**row)

                    if new_rec._index in valid_caches[rec_type]:
                        rec = instance_maps[rec_type][new_rec._index]
                    else:
                        rec = new_rec
                        rec._database = self.db
                        valid_caches[rec_type].add(rec._index)
                        instance_maps[rec_type][rec._index] = rec

                    val_list.append(rec)
                elif isinstance(sel, Array):
                    val_list.append(row["value"])
                else:
                    val_list.append(row[sel.name])

            vals.append(tuple(val_list) if len(val_list) > 1 else val_list[0])

        return vals

    @overload
    def items(  # pyright: ignore[reportOverlappingOverload]
        self: Data[
            ValT2, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
    ) -> Iterable[tuple[KeyT2, ValT2]]: ...

    @overload
    def items(
        self: Data[
            ValT2,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Iterable[tuple[tuple[*KeyTt2], ValT2]]: ...

    @overload
    def items(
        self: Data[
            ValT2,
            Idx[KeyT2] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Iterable[tuple[KeyT2 | tuple[*KeyTt2], ValT2]]: ...

    @overload
    def items(
        self: Data[
            ValT2,
            ArrayIdx[Record[*KeyTt2], KeyT3],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Iterable[tuple[tuple[*KeyTt2, KeyT3], ValT2]]: ...

    def items(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
    ) -> Iterable[tuple[Any, Any]]:
        """Iterable over this dataset's items."""
        return list(zip(self.keys(), self.values()))

    @overload
    def get(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
        default: ValT2,
    ) -> ValT | ValT2: ...

    @overload
    def get(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, DynBackendID
        ],
        key: tuple[*KeyTt2],
        default: ValT2,
    ) -> ValT | ValT2: ...

    @overload
    def get(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
        default: None = ...,
    ) -> ValT | None: ...

    @overload
    def get(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, DynBackendID
        ],
        key: tuple[*KeyTt2],
        default: None = ...,
    ) -> ValT | None: ...

    def get(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        key: Hashable | None = None,
        default: Hashable | None = None,
    ) -> Record | Hashable | None:
        """Get a record by key."""
        try:
            return (self[key] if key is not None else self).values()[0]
        except KeyError | IndexError:
            return default

    @overload
    def df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT],
        sort_by: (
            Literal["index"] | Iterable[Data[Any, Idx[()], Any, None, Ctx, Symbolic]]
        ) = ...,
        index_only: bool = ...,
        without_index: bool = ...,
        force_fqns: bool = ...,
    ) -> DfT: ...

    @overload
    def df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: None = ...,
        sort_by: (
            Literal["index"] | Iterable[Data[Any, Idx[()], Any, None, Ctx, Symbolic]]
        ) = ...,
        index_only: bool = ...,
        without_index: bool = ...,
        force_fqns: bool = ...,
    ) -> pl.DataFrame: ...

    def df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT] | None = None,
        sort_by: (
            Literal["index"] | Iterable[Data[Any, Idx[()], Any, None, Ctx, Symbolic]]
        ) = "index",
        index_only: bool = False,
        without_index: bool = False,
        force_fqns: bool = False,
    ) -> DfT:
        """Load dataset as dataframe."""
        if self._table is None:
            return cast(DfT, kind() if kind is not None else pl.DataFrame())

        all_cols = {
            **(self._abs_idx_cols if not without_index else {}),
            **(self._abs_cols if not index_only else {}),
        }
        if force_fqns:
            all_cols = {col.fqn: col for col in all_cols.values()}

        select = type(self).select(self, all_cols)

        col_names = cast(
            Mapping[Data[Any, Any, Any, None, Ctx, DynBackendID], str],
            {col: col_name for col_name, col in all_cols.items()},
        )

        merged_df = None
        if kind is pd.DataFrame:
            with self.db.engine.connect() as con:
                merged_df = pd.read_sql(select, con)

                merged_df = merged_df.set_index(
                    list(self._abs_idx_cols.keys()), drop=True
                )

                merged_df = (
                    merged_df.sort_index()
                    if sort_by == "index"
                    else merged_df.sort_values(
                        by=[col_names[self._table[c]] for c in sort_by]
                    )
                )
        else:
            merged_df = pl.read_database(
                select,
                self.db.df_engine.connect(),
            )

            sort_cols = (
                list(self._abs_idx_cols.keys())
                if sort_by == "index"
                else [col_names[self._table[c]] for c in sort_by]
            )
            merged_df = merged_df.sort(sort_cols)

            if not index_only and not without_index and isinstance(self, Table):
                rec_type: type[Record] = self.record_type
                drop_fqns = {
                    copy_and_override(Value, pk, _ctx=self).fqn
                    for pk in rec_type._pk_values.values()
                }
                merged_df = merged_df.drop(
                    [
                        col_name
                        for col_name in self._abs_idx_cols.keys()
                        if col_name in drop_fqns
                    ]
                )
            elif without_index:
                merged_df = merged_df.drop(
                    [col_name for col_name in self._abs_idx_cols.keys()]
                )

        return cast(DfT, merged_df)

    @overload
    def extract(
        self: Data[RecT2 | None, Any, Any, Any, Any, BackT2],
        *,
        aggs: (
            Mapping[
                Table[Record | None, Any, Any, Any, Ctx, Symbolic],
                Agg,
            ]
            | None
        ) = ...,
        to_db: None = ...,
        overlay_type: OverlayType = ...,
    ) -> DataBase[RecT2, Any, CRUD, BackT2]: ...

    @overload
    def extract(
        self: Data[Any, Any, Any, Any, Any, BackT2],
        *,
        aggs: (
            Mapping[
                Table[Record | None, Any, Any, Any, Ctx, Symbolic],
                Agg,
            ]
            | None
        ) = ...,
        to_db: None = ...,
        overlay_type: OverlayType = ...,
    ) -> DataBase[Any, Any, CRUD, BackT2]: ...

    @overload
    def extract(
        self: Data[Record | None, Any, R, Any, Any, DynBackendID],
        *,
        aggs: (
            Mapping[
                Table[Record | None, Any, Any, Any, Ctx, Symbolic],
                Agg,
            ]
            | None
        ) = ...,
        to_db: DataBase[SchemaT2, Any, CRUD, BackT3],
        overlay_type: OverlayType = ...,
    ) -> DataBase[SchemaT2, Any, CRUD, BackT3]: ...

    def extract(
        self: Data[Any, Any, Any, Any, Any, BackT2],
        aggs: (
            Mapping[
                Table[Record | None, Any, Any, Any, Ctx, Symbolic],
                Agg,
            ]
            | None
        ) = None,
        to_db: DataBase[Any, Any, CRUD, BackT3] | None = None,
        overlay_type: OverlayType = "name_prefix",
    ) -> DataBase[Any, Any, CRUD, BackT2 | BackT3]:
        """Extract a new database instance from the current dataset."""
        if aggs is not None:
            # TODO: Implement aggregations.
            raise NotImplementedError("Aggregations are not yet supported.")

        if isinstance(self, DataBase):
            overlay_db = self
        else:
            # Create a new database overlay for the results.
            assert self._table is not None
            rec_types = self._table.record_type._rel_types()

            overlay_db = copy_and_override(
                DataBase[Any, Any, CRUD, BackT2 | BackT3],
                self.db,
                backend=self.db.backend,
                schema=rec_types,
                write_to_overlay=f"extract/{token_hex(10)}",
                overlay_type=overlay_type,
                _def_types={},
                _metadata=sqla.MetaData(),
                _instance_map={},
            )

            # Empty all tables in the overlay.
            for rec in rec_types:
                overlay_db[rec] = {}

            # Traverse all relation paths and extract data to overlay.
            to_traverse: dict[
                Data[Record | None, Any, Any, Any, Ctx, DynBackendID],
                Data[Record | None, Any, Any, Any, Ctx, DynBackendID],
            ] = {
                self[rel]: self[tab]
                for tab in self.record_type._tables.values()
                for rel in tab._abs_link_path
            }
            while to_traverse:
                rel = next(iter(to_traverse.keys()))
                tab = to_traverse.pop(rel)

                if len(rel) == 0:
                    continue

                select = Data[Any, Any, Any, Any, Any, Any].select(
                    rel, cols=rel._abs_cols
                )
                overlay_db[rel.record_type] |= select

                if rel._rel is not None:
                    overlay_db[rel._rel.record_type] |= rel._rel.select

                for sub_tab in rel.record_type._tables.values():
                    sub_tab = tab[sub_tab]
                    for sub_rel in sub_tab._abs_link_path:
                        sub_rel = rel[sub_rel]

                        if not issubclass(sub_tab.relation_type, tab.relation_type):
                            to_traverse[sub_tab] = sub_rel

        # # Extract rel paths, which contain an aggregated rel.
        # aggs_per_type: dict[
        #     type[Record],
        #     list[
        #         tuple[
        #             Table[Record | None, Any, Any, Any, Any, Symbolic],
        #             Agg,
        #         ]
        #     ],
        # ] = {}
        # if aggs is not None:
        #     for rel, agg in aggs.items():
        #         for path_rel in all_paths_rels:
        #             if path_rel._has_ancestor(rel):
        #                 aggs_per_type[rel._ctx_type.record_type] = [
        #                     *aggs_per_type.get(rel._ctx_type.record_type, []),
        #                     (rel, agg),
        #                 ]
        #                 all_paths_rels.remove(path_rel)

        # aggregations: dict[type[Record], sqla.Select] = {}
        # for rec, rec_aggs in aggs_per_type.items():
        #     selects = []
        #     for rel, agg in rec_aggs:
        #         selects.append(
        #             sqla.select(
        #                 *[
        #                     (
        #                         sa._add_ctx(self)._sql_col
        #                         if isinstance(sa, Value)
        #                         else sqla_visitors.replacement_traverse(
        #                             sa,
        #                             {},
        #                             replace=lambda element, **kw: (
        #                                 element._add_ctx(cast(Table, self))._sql_col
        #                                 if isinstance(element, Value)
        #                                 else None
        #                             ),
        #                         )
        #                     ).label(ta.name)
        #                     for ta, sa in agg.map.items()
        #                 ]
        #             )
        #         )

        #     aggregations[rec] = sqla.union(*selects).select()

        # for rec, agg_select in aggregations.items():
        #     overlay_db[rec] &= agg_select

        # Transfer tables to the new database, if supplied.
        if to_db is not None:
            other_db = copy_and_override(
                DataBase[Any, Any, CRUD, BackT3],
                overlay_db,
                backend=to_db.backend,
                url=to_db.url,
                _def_types={},
                _metadata=sqla.MetaData(),
                _instance_map={},
            )

            for rec in overlay_db._record_types:
                other_db[rec] = overlay_db[rec].df()

            overlay_db = other_db

        return overlay_db

    def to_graph(
        self: Data[Schema | Record, Any, Any, Any, Any, DynBackendID],
        nodes: (
            Sequence[type[Record] | Data[Record, BaseIdx, Any, None, None, Symbolic]]
            | None
        ) = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        db = self.extract()

        nodes = (
            nodes
            if nodes is not None
            else list(
                DataBase[Any, Any, Any, Any]._record_types(db, with_relations=False)
            )
        )
        node_tables = [db[n] for n in nodes]
        node_types = [n if isinstance(n, type) else n.record_type for n in nodes]

        # Concat all node tables into one.
        node_dfs = [
            n.df(kind=pd.DataFrame, force_fqns=True)
            .reset_index()
            .assign(table=n.target_type._default_table_name())
            for n in node_tables
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )
        node_df = node_df[
            [
                "node_id",
                "table",
                *(c for c in node_df.columns if c not in ("node_id", "table")),
            ]
        ]

        directed_edges = reduce(
            set.union, (set((n, r) for r in n._links.values()) for n in node_types)
        )

        undirected_edges: dict[
            type[Record],
            set[
                tuple[
                    Link[Record, Any, Any, Symbolic], Link[Record, Any, Any, Symbolic]
                ]
            ],
        ] = {t: set() for t in db._relation_types}

        for rel in db._relation_types:
            left, right = cast(
                tuple[
                    Link[Record, Any, Any, Symbolic],
                    Link[Record, Any, Any, Symbolic],
                ],
                (rel._from, rel._to),  # type: ignore
            )
            if left.target_type in node_types and right.target_type in node_types:
                undirected_edges[rel].add((left, right))

        direct_edge_dfs = [
            node_df.loc[node_df["table"] == str(parent._default_table_name())]
            .rename(columns={"node_id": "source"})
            .merge(
                node_df.loc[
                    node_df["table"] == str(link.record_type._default_table_name())
                ],
                left_on=[c.fqn for c in link._fk_map.keys()],
                right_on=[c.fqn for c in link._fk_map.values()],
            )
            .rename(columns={"node_id": "target"})[["source", "target"]]
            .assign(
                ltr=",".join(c.name for c in link._fk_map.keys()),
                rtl=None,
            )
            for parent, link in directed_edges
        ]

        rel_edge_dfs = []
        for assoc_table, rels in undirected_edges.items():
            for left, right in rels:
                rel_df = (
                    self[assoc_table]
                    .df(kind=pd.DataFrame, force_fqns=True)
                    .reset_index()
                )

                left_merged = rel_df.merge(
                    node_df.loc[
                        node_df["table"] == str(left.record_type._default_table_name())
                    ][[*(c.fqn for c in left._fk_map.values()), "node_id"]],
                    left_on=[c.fqn for c in left._fk_map.keys()],
                    right_on=[c.fqn for c in left._fk_map.values()],
                    how="inner",
                ).rename(columns={"node_id": "source"})

                both_merged = left_merged.merge(
                    node_df.loc[
                        node_df["table"] == str(right.record_type._default_table_name())
                    ][[*(c.fqn for c in right._fk_map.values()), "node_id"]],
                    left_on=[c.fqn for c in right._fk_map.keys()],
                    right_on=[c.fqn for c in right._fk_map.values()],
                    how="inner",
                ).rename(columns={"node_id": "target"})[
                    list(
                        {
                            "source",
                            "target",
                            *(c.fqn for c in assoc_table._col_values.values()),
                        }
                    )
                ]

                rel_edge_dfs.append(
                    both_merged.assign(
                        ltr=",".join(c.name for c in right._fk_map.keys()),
                        rtl=",".join(c.name for c in left._fk_map.keys()),
                    )
                )

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                *direct_edge_dfs,
                *rel_edge_dfs,
            ],
            ignore_index=True,
        )

        return node_df, edge_df

    @overload
    def to_dfs(
        self: Data[tuple, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT],
    ) -> tuple[DfT, ...]: ...

    @overload
    def to_dfs(
        self: Data[tuple, Any, Any, Any, Any, DynBackendID],
        kind: None = ...,
    ) -> tuple[pl.DataFrame, ...]: ...

    def to_dfs(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT] | None = None,
    ) -> DfT | tuple[DfT, ...]:
        """Load tuple-valued dataset as tuple of dataframes."""
        merged_df = self.df(kind=kind)

        name_map = [
            {col.fqn: col.name for col in col_set}
            for col_set in self._abs_col_sets.values()
        ]

        return cast(
            tuple[DfT, ...],
            (merged_df[list(cols.keys())].rename(cols) for cols in name_map),
        )

    def isin(
        self: Data[Any, Any, Any, None, Ctx, BaseT2], other: Iterable[ValT2] | slice
    ) -> sqla.ColumnElement[bool]:
        """Test values of this dataset for membership in the given iterable."""
        return (
            self.sql_col.between(other.start, other.stop)
            if isinstance(other, slice)
            else self.sql_col.in_(other)
        )

    # 1. DB-level type selection
    @overload
    def __getitem__(
        self: Data[Schema | Record | None, RootIdx, Any, None, None, Any],
        key: type[RecT3],
    ) -> Data[RecT3, BaseIdx[RecT3], CrudT, None, None, BaseT]: ...

    # 2. DB-level nested prop selection
    @overload
    def __getitem__(
        self: Data[Schema | Record | None, RootIdx, Any, None, None, Any],
        key: Data[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Data[
        ValT3,
        Idx[*tuple[Any, ...], Any, *KeyTt3] | Idx[*KeyTt3],
        CrudT3,
        RelT3,
        CtxT3,
        BaseT,
    ]: ...

    # 3. DB-level nested array selection
    @overload
    def __getitem__(
        self: Data[Schema | Record | None, RootIdx, Any, None, None, Any],
        key: Data[
            ValT3,
            ArrayIdx[Record[*KeyTt3], *KeyTt4],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Data[
        ValT3,
        Idx[*tuple[Any, ...], *KeyTt3, *KeyTt4],
        CrudT3,
        RelT3,
        CtxT3,
        BaseT,
    ]: ...

    # 4. Top-level prop selection
    @overload
    def __getitem__(
        self: Data[
            RecT2 | None, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any
        ],
        key: Data[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            RelT3,
            Ctx[RecT2],
            Symbolic,
        ],
    ) -> Data[ValT3, Idx[*KeyTt2, *KeyTt3], CrudT3, RelT3, Ctx[RecT2], BaseT]: ...

    # 5. Top-level array selection
    @overload
    def __getitem__(
        self: Data[
            RecT2 | None, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any
        ],
        key: Data[
            ValT3,
            ArrayIdx[Record[*KeyTt3], *KeyTt4],
            CrudT3,
            RelT3,
            Ctx[RecT2],
            Symbolic,
        ],
    ) -> Data[
        ValT3, Idx[*KeyTt2, *KeyTt3, *KeyTt4], CrudT3, RelT3, Ctx[RecT2], BaseT
    ]: ...

    # 6. Nested prop selection
    @overload
    def __getitem__(
        self: Data[
            Record | None, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any
        ],
        key: Data[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt2, *tuple[Any, ...], *KeyTt3],
        CrudT3,
        RelT3,
        CtxT3,
        BaseT,
    ]: ...

    # 7. Nested array selection
    @overload
    def __getitem__(
        self: Data[
            Record | None, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any
        ],
        key: Data[
            ValT3,
            ArrayIdx[Record[*KeyTt3], *KeyTt4],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt2, *tuple[Any, ...], *KeyTt3, *KeyTt4],
        CrudT3,
        RelT3,
        CtxT3,
        BaseT,
    ]: ...

    # 8. Key filtering, scalar index type
    @overload
    def __getitem__(
        self: Data[Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Any],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 9. Key / slice filtering
    @overload
    def __getitem__(
        self: Data[Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 10. Key / slice filtering, array index
    @overload
    def __getitem__(
        self: Data[Any, ArrayIdx[Record[*KeyTt2], KeyT3], Any, Any, Any, Any],
        key: list[tuple[*KeyTt2, KeyT3]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 11. Expression filtering
    @overload
    def __getitem__(
        self: Data[Any, Idx | BaseIdx, Any, Any, Any, Any],
        key: sqla.ColumnElement[bool],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 12. Key selection, scalar index type, symbolic context
    @overload
    def __getitem__(
        self: Data[Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Symbolic],
        key: KeyT2,
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 13. Key selection, tuple index type, symbolic context
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Symbolic
        ],
        key: tuple[*KeyTt2],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 14. Key selection, tuple index type, symbolic context, array index
    @overload
    def __getitem__(
        self: Data[Any, ArrayIdx[Record[*KeyTt2], KeyT3], Any, Any, Any, Symbolic],
        key: tuple[*KeyTt2, KeyT3],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 15. Key selection, scalar index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
    ) -> ValT: ...

    # 16. Key selection, tuple index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, DynBackendID
        ],
        key: tuple[*KeyTt2],
    ) -> ValT: ...

    # 17. Key selection, tuple index type, array index
    @overload
    def __getitem__(
        self: Data[Any, ArrayIdx[Record[*KeyTt2], KeyT3], Any, Any, Any, DynBackendID],
        key: tuple[*KeyTt2, KeyT3],
    ) -> ValT: ...

    # Implementation:

    def __getitem__(
        self: Data[Any, Any, Any, Any, Any, Any],
        key: (
            type[Record]
            | Data[Any, Any, Any, Any, Any, Symbolic]
            | sqla.ColumnElement[bool]
            | list
            | slice
            | tuple[slice, ...]
            | Hashable
        ),
    ) -> Data[Any, Any, Any, Any, Any, Any] | ValT:
        """Select into the relational subgraph or filte ron the current level."""
        match key:
            case type():
                return Table(_db=self.db, _type=Table[key])
            case Data():
                return (
                    copy_and_override(type(key), key, _db=self)
                    if isinstance(self, DataBase)
                    else key._prepend_ctx(self._table)
                )
            case list() | slice() | Hashable() | sqla.ColumnElement():
                if not isinstance(key, tuple | list | sqla.ColumnElement):
                    key = (key,)
                elif isinstance(key, list) and not has_type(key, list[tuple]):
                    key = [(k,) for k in key]

                key_set = copy_and_override(
                    type(self),
                    self,
                    _filters=[
                        *self._filters,
                        cast(
                            sqla.ColumnElement[bool]
                            | list[tuple[Hashable, ...]]
                            | tuple[slice | Hashable, ...],
                            key,
                        ),
                    ],
                )

                if (
                    not isinstance(key, list | sqla.ColumnElement)
                    and not has_type(key, tuple[slice, ...])
                    and not isinstance(self.db.backend, Symbolic)
                ):
                    try:
                        return list(iter(key_set))[0]
                    except IndexError as e:
                        raise KeyError(key) from e

                return key_set

    @overload
    def __setitem__(
        self: Data[Schema | Record | None, RootIdx, CRUD, None, None, DynBackendID],
        key: type[RecT3],
        value: Data[RecT3, Any, Any, Any, Any, DynBackendID] | Input[RecT3, Hashable],
    ) -> None: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        key: Data[
            RecT3 | None,
            BaseIdx[Record[KeyT3]],
            Any,
            Any,
            Any,
            Symbolic,
        ],
        value: (
            Data[
                RecT3,
                Any,
                Any,
                Any,
                Any,
                DynBackendID,
            ]
            | Input[RecT3 | KeyT3, Hashable]
        ),
    ) -> None: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        key: Data[
            RecT3 | None,
            BaseIdx[Record[*KeyTt3]],
            Any,
            Any,
            Any,
            Symbolic,
        ],
        value: (
            Data[
                RecT3,
                Any,
                Any,
                Any,
                Any,
                DynBackendID,
            ]
            | Input[RecT3 | tuple[*KeyTt3], Hashable]
        ),
    ) -> None: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        key: Data[
            ValT3,
            Any,
            Any,
            Any,
            Any,
            Symbolic,
        ],
        value: (
            Data[
                ValT3,
                Any,
                Any,
                Any,
                Any,
                DynBackendID,
            ]
            | Input[ValT, Hashable]
        ),
    ) -> None: ...

    @overload
    def __setitem__(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], CRUD, Any, Any, DynBackendID
        ],
        key: KeyT2,
        value: Data[ValT, Idx[()], Any, Any, Any, DynBackendID] | Input[ValT, None],
    ) -> None: ...

    @overload
    def __setitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], CRUD, Any, Any, DynBackendID
        ],
        key: tuple[*KeyTt2],
        value: Data[ValT, Idx[()], Any, Any, Any, DynBackendID] | Input[ValT, None],
    ) -> None: ...

    @overload
    def __setitem__(
        self: Data[Any, ArrayIdx[Record[*KeyTt2], KeyT3], CRUD, Any, Any, DynBackendID],
        key: tuple[*KeyTt2, KeyT3],
        value: Data[ValT, Idx[()], Any, Any, Any, DynBackendID] | Input[ValT, None],
    ) -> None: ...

    # Implementation:

    def __setitem__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        key: type[Record] | Data[Any, Any, Any, Any, Any, Symbolic] | Hashable,
        value: Data[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> None:
        """Replacing assignment."""
        item = self[key]

        if hash(item) != hash(value):
            cast(Data, item)._mutate(value, mode="replace")

        return

    # 1. Type deletion
    @overload
    def __delitem__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        key: type[RecT2],
    ) -> None: ...

    # 2. Filter deletion
    @overload
    def __delitem__(
        self: Data[
            Any,
            BaseIdx[Record[KeyT2]]
            | Idx[KeyT2]
            | BaseIdx[Record[*KeyTt2]]
            | Idx[*KeyTt2],
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        key: (
            Iterable[KeyT2 | tuple[*KeyTt2]]
            | KeyT2
            | tuple[*KeyTt2]
            | slice
            | tuple[slice, ...]
            | Data[bool, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Symbolic]
        ),
    ) -> None: ...

    # Implementation:

    def __delitem__(  # noqa: D105
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        key: (
            list[Hashable]
            | Hashable
            | slice
            | tuple[slice, ...]
            | sqla.ColumnElement[bool]
        ),
    ) -> None:
        if not isinstance(key, slice) or key != slice(None):
            del self[
                key if isinstance(key, list | slice | sqla.ColumnElement) else [key]
            ][:]
            return

        self._mutate([], mode="delete")

    @overload
    def __irshift__(
        self: Data[
            Record[KeyT2] | None,
            Any,
            RU,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: Data[ValT, Any, Any, Any, Any, DynBackendID] | Input[ValT | KeyT2, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __irshift__(
        self: Data[
            Record[*KeyTt2] | None,
            Any,
            RU,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, DynBackendID]
            | Input[ValT | tuple[*KeyTt2], Any]
        ),
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __irshift__(
        self: Data[
            ValT2,
            Any,
            RU,
            Any,
            Any,
            DynBackendID,
        ],
        other: Data[ValT2, Any, Any, Any, Any, DynBackendID] | Input[ValT2, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __irshift__(
        self: Data[Any, Any, RU, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Updating assignment."""
        cast(Data, self)._mutate(other, mode="update")
        return cast(
            Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT],
            self,
        )

    @overload
    def __ilshift__(
        self: Data[
            Record[KeyT2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: Data[ValT, Any, Any, Any, Any, DynBackendID] | Input[ValT | KeyT2, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __ilshift__(
        self: Data[
            Record[*KeyTt2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, DynBackendID]
            | Input[ValT | tuple[*KeyTt2], Any]
        ),
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __ilshift__(
        self: Data[
            ValT2,
            Any,
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: Data[ValT2, Any, Any, Any, Any, DynBackendID] | Input[ValT2, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __ilshift__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Inserting assignment."""
        self._mutate(other, mode="insert")
        return cast(
            Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT],
            self,
        )

    @overload
    def __ior__(
        self: Data[
            Record[KeyT2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: Data[ValT, Any, Any, Any, Any, DynBackendID] | Input[ValT | KeyT2, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __ior__(
        self: Data[
            Record[*KeyTt2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, DynBackendID]
            | Input[ValT | tuple[*KeyTt2], Any]
        ),
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __ior__(
        self: Data[
            ValT2,
            Any,
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: Data[ValT2, Any, Any, Any, Any, DynBackendID] | Input[ValT2, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __ior__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Upserting assignment."""
        self._mutate(other, mode="upsert")
        return cast(
            Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT],
            self,
        )

    @overload
    def __eq__(
        self: Data[Any, Any, Any, None, Ctx, BaseT2],
        other: Any | Data[Any, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    @overload
    def __eq__(
        self,
        other: Any,
    ) -> bool: ...

    def __eq__(  # noqa: D105
        self: Data[Any, Any, Any, Any, Any, Any],
        other: Any,
    ) -> sqla.ColumnElement[bool] | bool:
        identical = hash(self) == hash(other)

        if identical or self._ctx is None:
            return identical

        if isinstance(other, Data):
            return self.sql_col == other.sql_col

        return self.sql_col == other

    def __neq__(  # noqa: D105
        self: Data[Any, Any, Any, None, Ctx, BaseT2],
        other: Any | Data[Any, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self.sql_col != other.sql_col
        return self.sql_col != other

    def __lt__(  # noqa: D105
        self: Data[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Data[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self.sql_col < other.sql_col
        return self.sql_col < other

    def __le__(  # noqa: D105
        self: Data[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Data[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self.sql_col <= other.sql_col
        return self.sql_col <= other

    def __gt__(  # noqa: D105
        self: Data[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Data[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self.sql_col > other.sql_col
        return self.sql_col > other

    def __ge__(  # noqa: D105
        self: Data[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Data[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self.sql_col >= other.sql_col
        return self.sql_col >= other

    @overload
    def __matmul__(
        self: Data[tuple, Any, Any, Any, Any, Any],
        other: Data[tuple, Any, Any, Any, Any, Any],
    ) -> Data[tuple, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __matmul__(
        self: Data[ValT2, Any, Any, Any, Any, Any],
        other: Data[tuple[*ValTt3], Any, Any, Any, Any, Any],
    ) -> Data[tuple[ValT2, *ValTt3], IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, Any, Any, Any, Any],
        other: Data[ValT3, Any, Any, Any, Any, Any],
    ) -> Data[tuple[*ValTt2, ValT3], IdxT, CrudT, RelT, CtxT, BaseT]: ...

    @overload
    def __matmul__(
        self: Data[ValT2, Any, Any, Any, Any, Any],
        other: Data[ValT3, Any, Any, Any, Any, Any],
    ) -> Data[tuple[ValT2, ValT3], IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __matmul__(
        self,
        other: Data[ValT2, Any, Any, Any, Any, BaseT],
    ) -> Data[tuple, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Align and merge this dataset with another into a tuple-valued dataset."""
        return copy_and_override(
            Data[tuple, IdxT, CrudT, RelT, CtxT, BaseT],
            self,
            _tuple_selection=(
                *(
                    self._tuple_selection
                    if self._tuple_selection is not None
                    else [self]
                ),
                *(
                    other._tuple_selection
                    if other._tuple_selection is not None
                    else [other]
                ),
            ),
        )

    # def __set_name__(self, _, name: str) -> None:  # noqa: D105
    #     if self._name is None:
    #         self._name = name
    #     else:
    #         assert name == self._name

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, tuple[*KeyTt2]], Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, KeyT2], Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[ValT, Idx[KeyT2], CrudT, RelT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Record[*KeyTt2], BaseIdx, Any, Any, Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Any, ArrayIdx[Any, KeyT2], Any, Any, Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[ValT, ArrayIdx[RecT4, KeyT2], CrudT, RelT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, Any, Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[ValT, IdxT, CrudT, RelT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Any, Idx[()], Any, Any, Any, Any],
        instance: Record,
        owner: type[Record],
    ) -> ValT: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, tuple[*KeyTt2]], Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, KeyT2], Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[ValT, Idx[KeyT2], CrudT, RelT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Data[Record[*KeyTt2], BaseIdx, Any, Any, Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Data[Any, ArrayIdx[Any, KeyT2], Any, Any, Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[ValT, ArrayIdx[RecT4, KeyT2], CrudT, RelT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, Any, Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[ValT, IdxT, CrudT, RelT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(
        self: Data[Any, Any, Any, Any, Ctx | None, Any],
        instance: object | None,
        owner: type | None,
    ) -> Data[Any, Any, Any, Any, Any, Any] | ValT:
        """Get the value of this dataset when used as property."""
        owner = (
            self._symbolic_ctx.record_type if self._symbolic_ctx is not None else owner
        )

        if (
            owner is not None
            and issubclass(owner, Record)
            and isinstance(instance, owner)
        ):
            if isinstance(self, Value):
                if (
                    self.pub_status is Public
                    and instance._published
                    and instance._index not in instance._db._get_valid_cache_set(owner)
                ):
                    instance._update_dict()

                value = (
                    self.getter(instance)
                    if self.getter is not None
                    else instance.__dict__.get(self.name, Undef)
                )

                if value is Undef:
                    if self.default_factory is not None:
                        value = self.default_factory()
                    else:
                        value = self.default

                    assert (
                        value is not Undef
                    ), f"Property value for `{self.name}` could not be fetched."
                    setattr(instance, self.name, value)

                return value
            else:
                self_ref = cast(
                    Data[ValT, Idx[()], CrudT, RelT, Ctx, Symbolic],
                    getattr(owner, self.name),
                )
                return instance._db[type(instance)][self_ref][instance._index]

        return self

    @overload
    def __set__(
        self,
        instance: Record,
        value: Data[ValT, Idx[()], Any, Any, Any, Any] | ValT | type[Keep],
    ) -> None: ...

    @overload
    def __set__(
        self,
        instance: Any,
        value: Any,
    ) -> None: ...

    def __set__(
        self,
        instance: Any,
        value: Any,
    ) -> None:
        """Set the value of this dataset when used as property."""
        if value is Keep:
            return

        if isinstance(instance, Record):
            if isinstance(self, Value):
                if self.setter is not None:
                    self.setter(instance, value)
                else:
                    instance.__dict__[self.name] = value

            if not isinstance(self, Value) or (
                self.pub_status is Public and instance._published
            ):
                owner = type(instance)
                sym_rel: Data[Any, Any, Any, Any, Any, Symbolic] = getattr(
                    owner, self.name
                )
                instance._table[[instance._index]][sym_rel]._mutate(value)

            if not isinstance(self, Value):
                instance._update_dict()
        else:
            instance.__dict__[self.name] = value

        return

    def __len__(self: Data[Any, Any, Any, Any, Any, DynBackendID]) -> int:
        """Get the number of items in the dataset."""
        with self.db.engine.connect() as conn:
            count = conn.execute(
                sqla.select(sqla.func.count()).select_from(self.query)
            ).scalar()
            assert count is not None
            return count

    def __iter__(  # noqa: D105
        self: Data[ValT2, Any, Any, Any, Any, DynBackendID],
    ) -> Iterator[ValT2]:
        return iter(self.values())

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(
            (
                self.db,
                self.name,
                self.value_type,
                self.relation_type,
                self._ctx_table,
                self._filters,
                self._tuple_selection,
            )
        )

    @property
    def _data_type(self) -> type[Data] | type[None]:
        """Resolve the data container type reference."""
        hint = self._type

        if hint is None:
            return NoneType

        if has_type(hint, SingleTypeDef):
            base = get_origin(hint)
            if base is None or not issubclass(base, Data):
                return NoneType

            return base
        elif isinstance(hint, str):
            return _map_prop_type_name(hint)
        else:
            return NoneType

    @cached_prop
    def _generic_type(self) -> SingleTypeDef | UnionType:
        """Resolve the generic property type reference."""
        hint = self._type or Data
        generic = self._hint_to_typedef(hint)

        return generic

    @cached_prop
    def _generic_args(self) -> tuple[SingleTypeDef, ...]:
        args = get_args(self._generic_type)
        if len(args) == 0:
            args = tuple(
                cast(TypeVar, t).__default__
                for t in getattr(type(self), "__parameters__")
            )

        return args

    @cached_prop
    def _symbolic_ctx(self) -> CtxT:
        """Context record type."""
        return (
            cast(CtxT, Ctx(self._ctx.record_type))
            if isinstance(self._ctx, Data)
            else self._ctx if self._ctx else cast(CtxT, None)
        )

    @cached_prop
    def _ctx_table(self) -> Table[Record | None, Any, Any, Any, Any, BaseT] | None:
        """Context table."""
        return (
            self._ctx
            if isinstance(self._ctx, Table)
            else (
                Table(_db=self.db, _type=Table[self._ctx.record_type])
                if self._ctx is not None
                else None
            )
        )

    @cached_prop
    def _ctx_module(self) -> ModuleType | None:
        """Module of the context record type."""
        if self._symbolic_ctx is None:
            return None

        assert issubclass(self._symbolic_ctx.record_type, Record)
        return self._symbolic_ctx.record_type._src_mod or getmodule(self._symbolic_ctx)

    @cached_prop
    def _owner_type(self) -> type[Record] | None:
        """Get the original owner type of the property."""
        if self._symbolic_ctx is None:
            return None

        direct_owner = self._symbolic_ctx.record_type
        assert issubclass(direct_owner, Record)
        all_bases = [
            direct_owner,
            *direct_owner._record_superclasses,
        ]

        original_owners = [
            base for base in all_bases if self.name in base._get_class_props()
        ]
        assert len(original_owners) == 1
        return original_owners[0]

    @cached_prop
    def _backlink(
        self: Data[RecT2 | Any, Any, Any, RecT3 | None, Ctx[RecT4] | None, Any]
    ) -> BackLink[RecT2 | RecT3, Any, Any, RecT4, BaseT] | None:
        """Table pointing to the relations of this dataset, if any."""
        if isinstance(self, BackLink):
            return self

        if self._ctx_table is None:
            return None

        rec_type = (
            self.relation_type
            if is_subtype(self.relation_type, Record)
            else self.record_type if isinstance(self, Table) else None
        )
        if rec_type is None:
            return None

        links = [
            r
            for r in rec_type._links.values()
            if isinstance(r, Link)
            and issubclass(self._ctx_table.record_type, r.record_type)
        ]

        if len(links) != 1:
            return None

        index_by = self.index_by if isinstance(self, Table) else None

        link = links[0]
        return BackLink[RecT2 | RecT3, Any, Any, RecT4, BaseT](
            _db=self.db,
            _ctx=self._ctx,
            _name=(
                self.name
                if issubclass(rec_type, Item)
                else (
                    f"{self.name}._rel"
                    if issubclass(rec_type, Relation)
                    else gen_str_hash((link._ctx, link._name), 6)
                )
            ),
            _type=BackLink[cast(type[RecT2 | RecT3], rec_type)],
            link=cast(Link[Any, Any, Any, Any], link),
            index_by=index_by,
        )

    @cached_prop
    def _link(
        self: Data[RecT2 | Any, Any, Any, RecT3 | None, Ctx[RecT4] | None, Any]
    ) -> Link[RecT2, Any, RecT3 | RecT4, BaseT] | None:
        if isinstance(self, Link):
            return self

        if self._ctx_table is None:
            return None

        rec_type = (
            self.relation_type
            if is_subtype(self.relation_type, Record)
            else self.record_type if isinstance(self, Table) else None
        )
        if rec_type is None:
            return None

        links = [
            r
            for r in rec_type._links.values()
            if isinstance(r, Link) and issubclass(self.target_type, r.record_type)
        ]

        if len(links) != 1:
            return None

        return copy_and_override(
            Link[RecT2, Any, RecT3 | RecT4, BaseT],
            links[0],
            _db=self.db,
            _ctx=self._backlink if self._backlink is not None else self._ctx,  # type: ignore
            _sql_from=self.sql_from,
        )

    @property
    def _table(self) -> Table[Record | None, Any, Any, Any, Any, BaseT] | None:
        if isinstance(self, Table):
            return self
        if issubclass(self.target_type, Record):
            return Table(_db=self.db, _type=self.target_type)
        if self._backlink is not None:
            return self._backlink
        if self._ctx_table is not None:
            return self._ctx_table

        return None

    @property
    def _rel(
        self: Data[Any, Any, Any, RecT2 | None, Ctx[RecT3] | None, Any]
    ) -> BackLink[RecT2, Any, Any, RecT3, BaseT] | None:
        """Table pointing to the relations of this dataset, if any."""
        if not issubclass(self.relation_type, Record) or self._backlink is None:
            return None

        return self._backlink

    @property
    def _root(self) -> Data[Any, Any, Any, None, None, BaseT]:
        return self.path[0]

    @cached_prop
    def _abs_link_path(self) -> dict[LinkItem, Data[Any, Any, Any, Any, Any, Any]]:
        return {
            t: p for p in self.path for t in (p._backlink, p._link) if t is not None
        }

    @cached_prop
    def _abs_cols(self) -> dict[str, Value[Any, Any, Any, Any, BaseT]]:
        cols = (
            [
                v._prepend_ctx(self._table)
                for sel in self._tuple_selection
                for v in sel._abs_cols.values()
            ]
            if self._tuple_selection is not None
            else (
                [
                    copy_and_override(
                        Value[Any, Any, Any, Any, BaseT], v, _db=self.db, _ctx=self
                    )
                    for v in self.target_type._col_values.values()
                ]
                if has_type(self, Table[Record, Any, Any, Any, Any, Any])
                else [
                    cast(Value[Any, Any, Any, Any, BaseT], v)
                    for v in (
                        (self.relation_type.value._prepend_ctx(self._rel),)  # type: ignore
                        if issubclass(self.relation_type, Item)
                        else (self._prepend_ctx(self._ctx_table),)
                    )
                ]
            )
        )

        return (
            {col.fqn: col for col in cols}
            if issubclass(self.target_type, tuple)
            else {col.name: col for col in cols}
        )

    @cached_prop
    def _abs_idx_cols(self) -> dict[str, Value[Any, Any, Any, Any, BaseT]]:
        abs_idx_cols = {}
        path_dict: dict[
            Data[Record | None, Any, Any, Any, Any, Any],
            Data[Any, Any, Any, Any, Any, Any],
        ] = {self._root: self._root, **self._abs_link_path}

        for link_node, node in path_dict.items():
            if isinstance(link_node, Link) and issubclass(node.relation_type, Relation):
                abs_idx_cols |= {
                    copy_and_override(
                        Value,
                        pk,
                        _ctx=node,
                    ).fqn: link_node[pk]
                    for pk in link_node._fk_map.values()
                }
            elif isinstance(link_node, Table) and not issubclass(
                node.relation_type, Relation
            ):
                cols = []
                rec_type: type[Record] = link_node.record_type

                if link_node.index_by is not None:
                    cols = (
                        [link_node.index_by]
                        if isinstance(link_node.index_by, Value)
                        else list(link_node.index_by)
                    )
                elif issubclass(rec_type, Item):
                    cols = [rec_type.idx]  # type: ignore
                elif issubclass(rec_type, IndexedRelation):
                    cols = [rec_type._rel_id]  # type: ignore
                else:
                    cols = rec_type._pk_values.values()

                abs_cols = [
                    copy_and_override(
                        Value[Any, Any, Any, Any, BaseT],
                        col,
                        _db=self.db,
                        _ctx=link_node,
                    )
                    for col in cols
                ]
                abs_idx_cols |= {col.fqn: col for col in abs_cols}

        return abs_idx_cols

    @cached_prop
    def _abs_col_sets(
        self: Data[tuple, Any, Any, Any, Any, Any]
    ) -> dict[
        Data[Any, Any, Any, Any, Any, Any], set[Value[Any, Any, Any, Any, BaseT]]
    ]:
        return {
            sel: {
                copy_and_override(
                    Value[Any, Any, Any, Any, BaseT], v, _db=self.db, _ctx=self._ctx
                )
                for v in sel._abs_cols.values()
            }
            for sel in self.tuple_selection
        }

    @cached_prop
    def _base_table_map(
        self: Data[Record | None, Any, Any, Any, Any, Any],
    ) -> dict[
        Data[Record | None, Any, Any, Any, Any, Any],
        dict[str, Value[Any, Any, Any, Any, BaseT]],
    ]:
        return {
            self.db[base_rec]: {
                v_name: v
                for v_name, v in (*self._abs_idx_cols.items(), *self._abs_cols.items())
                if v.name in base_rec._values
            }
            for base_rec in [
                self.record_type,
                *self.record_type._record_superclasses,
            ]
        }

    @cached_prop
    def _abs_filters(
        self,
    ) -> tuple[
        list[sqla.ColumnElement[bool]], list[Table[Any, Any, Any, Any, Any, BaseT]]
    ]:
        sql_filt = [f for f in self._filters if isinstance(f, sqla.ColumnElement)]

        join_set: set[Table[Any, Any, Any, Any, Any, BaseT]] = set()
        replace_func = partial(self._visit_filter_col, join_set=join_set)
        sql_filt = [
            sqla_visitors.replacement_traverse(f, {}, replace=replace_func)
            for f in sql_filt
        ]
        merge = list(join_set)

        key_filt = [
            (
                reduce(
                    sqla.and_,
                    (
                        (idx.isin(filt) if isinstance(filt, slice) else idx == filt)
                        for idx, filt in zip(self._abs_idx_cols.values(), key_filt)
                    ),
                )
                if isinstance(key_filt, tuple)
                else reduce(
                    sqla.or_,
                    (
                        reduce(
                            sqla.and_,
                            (
                                (idx == filt)
                                for idx, filt in zip(
                                    self._abs_idx_cols.values(), single_filt
                                )
                            ),
                        )
                        for single_filt in key_filt
                    ),
                )
            )
            for key_filt in self._filters
            if not isinstance(key_filt, sqla.ColumnElement)
        ]

        return [
            *sql_filt,
            *key_filt,
        ], merge

    @cached_prop
    def _abs_froms(self) -> list[Table[Record | None, Any, Any, Any, Any, BaseT]]:
        return (
            [
                v._prepend_ctx(self._table)
                for sel in self._tuple_selection
                for v in sel._abs_froms
            ]
            if self._tuple_selection is not None
            else [self._table] if self._table is not None else []
        )

    @cached_prop
    def _total_froms(self) -> list[Table[Record | None, Any, Any, Any, Any, BaseT]]:
        return self._abs_froms + self._abs_filters[1]

    @cached_prop
    def _total_join_dict(self) -> JoinDict:
        tree: JoinDict = {}

        for rel in self._total_froms:
            subtree = tree
            for node in rel._abs_link_path.keys():
                if node not in subtree:
                    subtree[node] = {}
                subtree = subtree[node]

        return tree

    @cached_prop
    def _fk_map(self: Data[Record | None, Idx[()], Any, None, Ctx, Any]) -> bidict[
        Value[Any, Any, Any, Any, Symbolic],
        Value[Any, Any, Any, Any, Symbolic],
    ]:
        """Map source link foreign keys to target cols."""
        fks = self.fks if isinstance(self, Link) else None

        match fks:
            case None:
                return bidict(
                    {
                        Value[Any, Any, Any, Any, Symbolic](
                            _name=f"{self.name}_{pk.name}",
                            _type=pk._type,
                            init=False,
                            index=self.index if isinstance(self, Link) else False,
                            primary_key=(
                                self.primary_key if isinstance(self, Link) else False
                            ),
                            _ctx=Ctx(self.ctx.record_type),
                        ): pk
                        for pk in self.record_type._pk_values.values()
                    }
                )
            case dict():
                return bidict({fk: pk for fk, pk in fks.items()})
            case Value() | list():
                fks = fks if isinstance(fks, list) else [fks]

                pks = [
                    getattr(self.record_type, name)
                    for name in self._symbolic_ctx.record_type._pk_values
                ]

                return bidict(dict(zip(fks, pks)))

    def _hint_to_typedef(
        self,
        hint: SingleTypeDef | UnionType | TypeVar | str | ForwardRef,
    ) -> SingleTypeDef | UnionType:
        typedef = hint

        if isinstance(typedef, str):
            typedef = eval(
                typedef,
                {**globals(), **(vars(self._ctx_module) if self._ctx_module else {})},
            )

        if isinstance(typedef, TypeVar):
            type_res = self._typevar_map.get(typedef, Undef)

            if type_res is Undef and typedef.has_default():
                type_res = typedef.__default__
            if type_res is Undef:
                raise TypeError(
                    f"Type variable `{typedef}` not bound for typehint `{hint}`."
                )

            typedef = type_res

        if isinstance(typedef, ForwardRef):
            typedef = typedef._evaluate(
                {**globals(), **(vars(self._ctx_module) if self._ctx_module else {})},
                {},
                recursive_guard=frozenset(),
            )

        return cast(SingleTypeDef, typedef)

    def _typedef_to_type(
        self, hint: SingleTypeDef | UnionType | None, remove_null: bool = False
    ) -> type:
        if hint is None:
            return NoneType

        if isinstance(hint, type):
            return hint

        typedef = (
            self._hint_to_typedef(hint.__value__)
            if isinstance(hint, TypeAliasType)
            else self._hint_to_typedef(hint) if isinstance(hint, TypeVar) else hint
        )

        if isinstance(typedef, UnionType):
            union_types: set[type] = {
                get_origin(union_arg) or union_arg for union_arg in get_args(typedef)
            }

            if remove_null:
                union_types = {
                    t
                    for t in union_types
                    if t is not None and not issubclass(t, NoneType)
                }
                assert len(union_types) == 1
                return next(iter(union_types))

            typedef = get_lowest_common_base(union_types)

        orig = get_origin(typedef)

        if orig is None or orig is Literal:
            return object

        assert isinstance(orig, type)
        if isinstance(hint, TypeAliasType):
            cls = _type_alias_classes.get(hint) or new_class(
                hint.__name__,
                (typedef,),
                None,
                lambda ns: ns.update(
                    {"_src_mod": (self._ctx_module or getmodule(hint))}
                ),
            )
            _type_alias_classes[hint] = cls
            return cls
        else:
            return new_class(
                orig.__name__ + "_" + gen_str_hash(get_args(typedef), 5),
                (typedef,),
                None,
                lambda ns: ns.update(
                    {"_src_mod": (self._ctx_module or getmodule(hint))}
                ),
            )

    def _has_ancestor(
        self, other: Table[Record | None, Any, Any, Any, Any, BaseT]
    ) -> bool:
        """Check if ``other`` is ancestor of this data."""
        return other is self._ctx or (
            self._ctx_table is not None and self._ctx_table._has_ancestor(other)
        )

    def _prepend_ctx(
        self,
        left: Data[Record | None, Any, Any, Any, Any, Any] | None,
    ) -> Self:
        """Prefix this dataset with another table as context."""
        if left is None:
            return self

        root, *path = self.path
        assert (
            left.record_type == root.target_type
        ), "Context table must be of same type as root table."

        prefixed = cast(
            Self,
            reduce(
                lambda x, y: copy_and_override(
                    cast(type[Data[Record, Any, Any, Any, Any, BaseT]], type(y)),
                    y,
                    _db=x.db,
                    _ctx=x,
                ),
                path,
                left,
            ),
        )
        return prefixed

    def _visit_filter_col(
        self,
        element: sqla_visitors.ExternallyTraversible,
        join_set: set[Table[Any, Any, Any, Any, Any, BaseT]] = set(),
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if hasattr(element, "_data"):
            data = cast(
                Data[Any, Idx[()], Any, None, Ctx, Any], getattr(element, "_data")
            )
            prefixed = data._prepend_ctx(self._table)

            if prefixed.ctx != self._root:
                join_set.add(prefixed.ctx)

            return prefixed.sql_col

        return None

    def _gen_sql_base_cols(
        self: Data[Record | None, Any, Any, Any, Any, Any]
    ) -> dict[str, sqla.Column]:
        """Columns of this record type's table."""
        registry = orm.registry(
            metadata=self.db._metadata,
            type_annotation_map=self.record_type._type_map,
        )

        return {
            name: sqla.Column(
                attr.name,
                registry._resolve_type(
                    attr.value_type  # pyright: ignore[reportArgumentType]
                ),
                primary_key=attr.primary_key,
                autoincrement=False,
                index=attr.index,
                nullable=has_type(None, attr.value_type),
            )
            for name, attr in self.record_type._class_values.items()
        }

    def _gen_sql_base_fks(
        self: Data[Record | None, Any, Any, Any, Any, Any]
    ) -> list[sqla.ForeignKeyConstraint]:
        fks: list[sqla.ForeignKeyConstraint] = []

        for rt in self.record_type._links.values():
            rel_table = self[rt]._gen_sql_base_table()
            fks.append(
                sqla.ForeignKeyConstraint(
                    [fk.name for fk in rt._fk_map.keys()],
                    [rel_table.c[pk.name] for pk in rt._fk_map.values()],
                    name=f"{self.record_type._get_table_name(self.db._subs)}_{rt.name}_fk",
                )
            )

        for superclass in self.record_type._record_superclasses:
            base_table = Table(
                _db=self.db, _type=Table[superclass]
            )._gen_sql_base_table()

            fks.append(
                sqla.ForeignKeyConstraint(
                    [pk_name for pk_name in self.record_type._pk_values],
                    [base_table.c[pk_name] for pk_name in self.record_type._pk_values],
                    name=(
                        self.record_type._get_table_name(self.db._subs)
                        + "_base_fk_"
                        + gen_str_hash(superclass._get_table_name(self.db._subs), 5)
                    ),
                )
            )

        return fks

    def _gen_sql_base_table(
        self: Data[Record | None, Any, Any, Any, Any, Any],
        mode: Literal["read", "replace", "upsert"] = "read",
        without_auto_fks: bool = False,
    ) -> sqla.Table:
        """Return the base SQLAlchemy table object for this data's record type."""
        orig_table: sqla.Table | None = None

        if (
            mode != "read"
            and self.db.write_to_overlay is not None
            and self.record_type not in self.db._subs
        ):
            orig_table = self._gen_sql_base_table("read")

            # Create an empty overlay table for the record type
            self.db._subs[self.record_type] = sqla.table(
                (
                    (
                        self.db.write_to_overlay
                        + "/"
                        + self.record_type._default_table_name()
                    )
                    if self.db.overlay_type == "name_prefix"
                    else self.record_type._default_table_name()
                ),
                schema=(
                    self.db.write_to_overlay
                    if self.db.overlay_type == "db_schema"
                    else None
                ),
            )

        table_name = self.record_type._get_table_name(self.db._subs)

        if not without_auto_fks and table_name in self.db._metadata.tables:
            # Return the table object from metadata if it already exists.
            # This is necessary to avoid circular dependencies.
            return self.db._metadata.tables[table_name]

        sub = self.db._subs.get(self.record_type)

        cols = self._gen_sql_base_cols()
        if without_auto_fks:
            cols = {
                name: col
                for name, col in cols.items()
                if name in self.record_type._values
            }

        # Create a partial SQLAlchemy table object from the class definition
        # without foreign keys to avoid circular dependencies.
        # This adds the table to the metadata.
        sqla.Table(
            table_name,
            self.db._metadata,
            *cols.values(),
            schema=(sub.schema if sub is not None else None),
        )

        fks = self._gen_sql_base_fks()
        if without_auto_fks:
            fks = [
                fk
                for fk in fks
                if not any(
                    c.name in self.record_type._fk_values
                    and c.name not in self.record_type._values
                    for c in fk.columns
                )
            ]

        # Re-create the table object with foreign keys and return it.
        table = sqla.Table(
            table_name,
            self.db._metadata,
            *fks,
            schema=(sub.schema if sub is not None else None),
            extend_existing=True,
        )

        self.db._create_sqla_table(table)

        if orig_table is not None and mode == "upsert":
            with self.db.engine.begin() as conn:
                conn.execute(
                    sqla.insert(table).from_select(
                        orig_table.columns.keys(), orig_table.select()
                    )
                )

        return table

    def _gen_idx_value_map(self, idx: Any) -> dict[str, Hashable]:
        idx_names = list(self._abs_idx_cols.keys())

        if len(idx_names) == 1:
            return {idx_names[0]: idx}

        assert isinstance(idx, tuple) and len(idx) == len(idx_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(idx_names, idx)}

    def _gen_fk_value_map(
        self: Data[Record | None, Any, Any, None, Any, Any], val: Any
    ) -> dict[str, Hashable]:
        fk_names = [col.name for col in self._fk_map.keys()]

        if len(fk_names) == 1:
            return {fk_names[0]: val}

        assert isinstance(val, tuple) and len(val) == len(fk_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(fk_names, val)}

    def _values_to_df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        values: Mapping[Any, Any],
        include: list[str] | None = None,
    ) -> pl.DataFrame:
        df_data = []
        for idx, vals in values.items():
            row_data = {
                idx_name: v
                for idx_name, v in zip(
                    self._abs_idx_cols,
                    idx if isinstance(idx, tuple) else (idx,),
                )
            }
            for val_def, val in (
                zip(self._tuple_selection, vals)
                if self._tuple_selection is not None and len(self._tuple_selection) > 1
                else [(self, vals)]
            ):
                if issubclass(val_def.target_type, Record):
                    if isinstance(val, Record):
                        rec_dict = val._to_dict()
                        row_data |= {
                            c_name: rec_dict[c.name]
                            for c_name, c in val_def._abs_cols.items()
                            if include is None or c_name in include
                        }
                    else:
                        row_data |= val_def._gen_idx_value_map(val)
                else:
                    assert len(val_def._abs_cols) == 1
                    row_data[list(val_def._abs_cols.keys())[0]] = val

            df_data.append(row_data)

        return pl.DataFrame(
            df_data,
            schema=_get_pl_schema(
                {
                    col_name: col
                    for col_name, col in (self._abs_cols | self._abs_idx_cols).items()
                    if include is None or col_name in include
                }
            ),
        )

    def _gen_upload_table(
        self,
        include: list[str] | None = None,
    ) -> sqla.Table:
        type_map = (
            self.target_type._type_map
            if issubclass(self.target_type, Record)
            else (
                self._ctx_table.record_type._type_map
                if self._ctx_table is not None
                else {}
            )
        )

        metadata = sqla.MetaData()
        registry = orm.registry(
            metadata=self.db._metadata,
            type_annotation_map=type_map,
        )

        cols = [
            sqla.Column(
                col_name,
                registry._resolve_type(
                    col.value_type  # pyright: ignore[reportArgumentType]
                ),
                primary_key=col.primary_key,
                autoincrement=False,
                index=col.index,
                nullable=has_type(None, col.value_type),
            )
            for col_name, col in (self._abs_cols | self._abs_idx_cols).items()
            if include is None or col_name in include
        ]

        table_name = f"upload/{token_hex(5)}/{self.fqn.replace('.', '_')}"
        table = sqla.Table(
            table_name,
            metadata,
            *cols,
        )

        return table

    def _df_to_table(
        self,
        df: pd.DataFrame | pl.DataFrame,
    ) -> sqla.Table:
        if isinstance(df, pd.DataFrame) and any(
            name is None for name in df.index.names
        ):
            idx_names = list(self._abs_idx_cols.keys())
            df.index.set_names(idx_names, inplace=True)

        value_table = self._gen_upload_table(include=list(df.columns))

        if isinstance(df, pd.DataFrame):
            df.reset_index().to_sql(
                value_table.name,
                self.db.engine,
                if_exists="replace",
                index=False,
            )
        else:
            df.write_database(
                str(value_table), self.db.engine, if_table_exists="append"
            )

        return value_table

    def _mutate(
        self: Data[ValT2, Any, CRUD, Any, Any, DynBackendID],
        value: Data[ValT2, Any, Any, Any, Any, Any] | Input[ValT2, Hashable],
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        record_ids: dict[Hashable, Hashable] | None = None
        valid_caches = (
            self.db._get_valid_cache_set(self._table.record_type)
            if self._table is not None
            else set()
        )

        match value:
            case sqla.Select():
                assert self._table is not None
                self._table._mutate_from_sql(value.subquery(), mode)
                valid_caches.clear()
            case Data():
                if hash(value.db) == hash(self.db) and self._table is not None:
                    # Other database is exactly the same,
                    # so updating target table is enough.
                    self._table._mutate_from_sql(value.query, mode)
                elif issubclass(value.target_type, Record | Schema):
                    # Other database is not exactly the same,
                    # hence may be on a different overlay or backend.
                    # Related records have to be mutated alongside.
                    other_db = value.extract()

                    if self.db.db_id == other_db.db_id:
                        for rec in other_db._record_types:
                            self.db[rec]._mutate_from_sql(other_db[rec].query, mode)
                    else:
                        for rec in other_db._record_types:
                            value_table = self.db[rec]._df_to_table(
                                other_db[rec].df(),
                            )
                            self.db[rec]._mutate_from_sql(
                                value_table,
                                mode,
                            )
                            value_table.drop(self.db.engine)
                elif self._table is not None and value._table is not None:
                    # Not a record type, so no need to mutate related records.
                    value_table = self._df_to_table(value.df())
                    self._table._mutate_from_sql(value_table, mode)
                    value_table.drop(self.db.engine)

                valid_caches -= set(value.keys())
            case pd.DataFrame() | pl.DataFrame():
                assert self._table is not None

                value_table = self._df_to_table(value)
                self._table._mutate_from_sql(
                    value_table,
                    mode,
                )
                value_table.drop(self.db.engine)

                base_idx_cols = list(self._table.record_type._pk_values.keys())
                base_idx_keys = set(
                    value[base_idx_cols].iter_rows()
                    if isinstance(value, pl.DataFrame)
                    else value[base_idx_cols].itertuples(index=False)
                )

                valid_caches -= base_idx_keys
            case Record():
                cast(
                    Data[Record, Any, CRUD, Any, Any, DynBackendID],
                    self,
                )._mutate_from_records({value._index: value}, mode)
                valid_caches -= {value._index}
            case Iterable():
                if not issubclass(self.target_type, Record):
                    assert isinstance(
                        value, Mapping
                    ), "Inserting via values requires a mapping."
                    self._mutate_from_values(value, mode)
                    valid_caches -= set(value.keys())
                else:
                    if isinstance(value, Mapping):
                        records = {
                            idx: rec
                            for idx, rec in value.items()
                            if isinstance(rec, Record)
                        }
                        record_ids = {
                            idx: rec
                            for idx, rec in value.items()
                            if not isinstance(rec, Record)
                        }
                    else:
                        records = {
                            idx: rec
                            for idx, rec in enumerate(value)
                            if isinstance(rec, Record)
                        }
                        record_ids = {
                            idx: rec
                            for idx, rec in enumerate(value)
                            if not isinstance(rec, Record)
                        }

                    cast(
                        Data[Record | None, Any, CRUD, Any, Any, DynBackendID],
                        self,
                    )._mutate_from_records(
                        records,
                        mode,
                    )
                    valid_caches -= {rec._index for rec in records.values()}

                    if len(record_ids) > 0:
                        cast(
                            Data[Record | None, Any, CRUD, Any, Any, DynBackendID],
                            self,
                        )._mutate_rels_from_rec_ids(record_ids, mode)
                        valid_caches -= set(record_ids.values())
            case Hashable():
                if not issubclass(self.target_type, Record):
                    self._mutate_from_values({None: cast(ValT2, value)}, mode)
                    valid_caches -= {self.keys()}
                else:
                    assert issubclass(
                        self.relation_type, Record
                    ), "Inserting via ids requires a relation."
                    cast(
                        Data[Record, Any, CRUD, Record, Ctx, DynBackendID],
                        self,
                    )._mutate_rels_from_rec_ids({value: value}, mode)
                    valid_caches -= {value}

        return

    def _mutate_from_values(
        self: Data[ValT2, Any, CRUD, Any, Any, DynBackendID],
        values: Mapping[Any, ValT2],
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        assert self._table is not None

        df = self._values_to_df(values)
        value_table = self._df_to_table(df)
        self._table._mutate_from_sql(
            value_table,
            mode,
        )
        value_table.drop(self.db.engine)
        return

    def _mutate_from_records(
        self: Data[Record | None, Any, CRUD, Relation | None, Ctx | None, DynBackendID],
        values: Mapping[Any, Record],
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        db_grouped = {
            db: dict(recs)
            for db, recs in groupby(
                sorted(
                    values.items(), key=lambda x: x[1]._published and x[1]._db.db_id
                ),
                lambda x: None if not x[1]._published else x[1]._db,
            )
        }

        unconnected_records = db_grouped.get(None, {})
        local_records = db_grouped.get(self.db, {})

        remote_records = {
            db: recs
            for db, recs in db_grouped.items()
            if db is not None and hash(db) != hash(self.db)
        }

        # Update with local records first.
        df_data = self._values_to_df(unconnected_records)
        value_table = self._df_to_table(df_data)
        self._mutate_from_sql(
            value_table,
            mode,
        )
        value_table.drop(self.db.engine)

        if local_records and has_type(
            self, Data[Record | None, Any, Any, Record, Ctx, Any]
        ):
            # Only update relations for records already existing in this db.
            self._mutate_rels_from_rec_ids(
                {idx: rec._index for idx, rec in local_records.items()},
                mode,
            )

        for db, recs in remote_records.items():
            rec_ids = [rec._index for rec in recs.values()]
            remote_set = db[self.record_type][rec_ids]

            remote_db = (
                db if all(rec._root for rec in recs.values()) else remote_set.extract()
            )
            for s in remote_db._def_types:
                if remote_db.db_id == self.db.db_id:
                    self.db[s]._mutate_from_sql(remote_db[s].query, "upsert")
                else:
                    value_table = self._df_to_table(
                        remote_db[s].df(),
                    )
                    self.db[s]._mutate_from_sql(
                        value_table,
                        "upsert",
                    )
                    value_table.drop(self.db.engine)

            self._mutate_from_sql(remote_set.query, mode)

        return

    def _mutate_rels_from_rec_ids(
        self: Data[Record | None, Any, CRUD, Record | None, Ctx, DynBackendID],
        values: Mapping[Hashable, Hashable],
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        if self._link is None or self._table is None:
            return

        if not issubclass(self.relation_type, NoneType):
            return self._table._mutate_rels_from_rec_ids(values, mode)

        value_table = self._df_to_table(
            self._values_to_df(
                values, include=list(self.record_type._pk_values.keys())
            ),
        )

        to_fk_cols: list[sqla.ColumnElement] = [
            value_table.c[self[pk].fqn].label(fk.name)
            for fk, pk in self._link._fk_map.items()
        ]

        self.ctx._mutate_from_sql(
            sqla.select(value_table).with_only_columns(*to_fk_cols).subquery(),
            mode,
        )

    def _mutate_from_sql(  # noqa: C901
        self: Data[Record | None, Any, Any, Any, Any, Any],
        value_table: sqla.FromClause,
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        if not issubclass(self.relation_type, NoneType):
            assert self._link is not None
            return self._link._mutate_from_sql(value_table, mode)

        if isinstance(self, BackLink):
            # If this table links back to its parent, query and join
            # the parent's primary key columns to the value table.
            ctx_table = Data.select(self.ctx, cols=self.ctx._abs_idx_cols).subquery()

            pk_fk_cols = {
                self.ctx[pk].fqn: fk.name for fk, pk in self.link._fk_map.items()
            }

            if not all(col in value_table.c for col in pk_fk_cols.keys()):
                idx_col_map = {
                    value_table.c[col_name]: ctx_table.c[col_name]
                    for col_name in self._abs_idx_cols.keys()
                    if col_name in ctx_table.columns
                }

                value_table = (
                    sqla.select(value_table)
                    .join(
                        ctx_table,
                        reduce(
                            sqla.and_,
                            (
                                left_idx == right_idx
                                for left_idx, right_idx in idx_col_map.items()
                            ),
                        ),
                    )
                    .add_columns(
                        *(
                            ctx_table.c[col].label(label)
                            for col, label in pk_fk_cols.items()
                        )
                    )
                    .subquery()
                )
            else:
                value_table = (
                    sqla.select(value_table)
                    .add_columns(
                        *(
                            value_table.c[col].label(label)
                            for col, label in pk_fk_cols.items()
                        )
                    )
                    .subquery()
                )

        table_values: dict[sqla.Table, dict[str, Value[Any, Any, Any, Any, Any]]] = {}
        for tab, vals in self._base_table_map.items():
            unique_vals: dict[str, Value[Any, Any, Any, Any, Any]] = {}
            for col_name, col in vals.items():
                if col_name in value_table.columns and col not in unique_vals.values():
                    unique_vals[col_name] = col

            sqla_tab = tab._gen_sql_base_table(
                "upsert" if mode in ("update", "insert", "upsert") else "replace"
            )
            table_values[sqla_tab] = unique_vals

        statements: list[sqla.Executable] = []

        if mode in ("replace", "delete"):
            # Delete all records in the current selection.
            for table, vals in table_values.items():
                # Prepare delete statement.
                if self.db.engine.dialect.name in (
                    "postgres",
                    "postgresql",
                    "duckdb",
                    "mysql",
                    "mariadb",
                ):
                    # Delete-from.
                    statements.append(
                        table.delete().where(
                            reduce(
                                sqla.and_,
                                (
                                    table.c[col.name] == self.query.c[col_name]
                                    for col_name, col in vals.items()
                                    if col.name in table.primary_key.columns
                                ),
                            )
                        )
                    )
                elif self.db.engine.dialect.name in ("sqlite",):
                    statements.append(
                        table.delete().where(
                            sqla.column("rowid").in_(
                                sqla.select(sqla.column("rowid"))
                                .select_from(table)
                                .join(
                                    self.query,
                                    reduce(
                                        sqla.and_,
                                        (
                                            table.c[col.name] == self.query.c[col_name]
                                            for col_name, col in vals.items()
                                            if col.name in table.primary_key.columns
                                        ),
                                    ),
                                )
                            )
                        )
                    )
                else:
                    raise NotImplementedError(
                        "Replacement not supported for this dialect."
                    )

        if mode in ("replace", "upsert", "insert"):
            # Construct the insert statements.

            assert len(self._filters) == 0, "Can only upsert into unfiltered datasets."

            for table, vals in table_values.items():
                if self.db.engine.dialect.name in (
                    "postgres",
                    "postgresql",
                    "duckdb",
                    "sqlite",
                ):
                    if self.db.engine.dialect.name in (
                        "postgres",
                        "postgresql",
                        "duckdb",
                    ):
                        # For Postgres / DuckDB, use: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#updating-using-the-excluded-insert-values
                        statement = postgresql.Insert(table)
                    else:
                        # For SQLite, use: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#updating-using-the-excluded-insert-values
                        statement = sqlite.Insert(table)

                    statement = statement.from_select(
                        [col.name for col in vals.values()],
                        sqla.select(
                            *(
                                value_table.c[col_name].label(col.name)
                                for col_name, col in vals.items()
                            )
                        )
                        .select_from(value_table)
                        .where(sqla.text("true")),
                    )
                    if mode == "upsert":
                        statement = statement.on_conflict_do_update(
                            index_elements=[
                                col.name for col in table.primary_key.columns
                            ],
                            set_={
                                c.name: statement.excluded[c.name]
                                for c in vals.values()
                                if c.name not in table.primary_key.columns
                            },
                        )
                    else:
                        statement = statement.on_conflict_do_nothing()
                elif self.db.engine.dialect.name in (
                    "mysql",
                    "mariadb",
                ):
                    # For MySQL / MariaDB, use: https://docs.sqlalchemy.org/en/20/dialects/mysql.html#insert-on-duplicate-key-update-upsert
                    statement = (
                        mysql.Insert(table)
                        .from_select(
                            [c.name for c in vals.values()],
                            sqla.select(
                                *(
                                    value_table.c[col_name].label(col.name)
                                    for col_name, col in vals.items()
                                )
                            ).select_from(value_table),
                        )
                        .prefix_with("INSERT INTO")
                    )
                    if mode == "upsert":
                        statement = statement.on_duplicate_key_update(
                            **statement.inserted
                        )
                    else:
                        statement = statement.prefix_with("INSERT IGNORE INTO")
                else:
                    # For others, use CTE: https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.cte
                    raise NotImplementedError(
                        "Upsert not supported for this database dialect."
                    )

                statements.append(statement)

        elif mode == "update":
            # Construct the update statements.

            # Derive current select statement and join with value table.
            select = self.select.join(
                value_table,
                reduce(
                    sqla.and_,
                    (
                        col.sql_col == value_table.c[col_name]
                        for col_name, col in (
                            *self._abs_idx_cols.items(),
                            *(
                                (pk.name, pk)
                                for pk in self._abs_cols.values()
                                if pk.primary_key
                            ),
                        )
                        if col_name in value_table.columns
                    ),
                ),
            ).subquery()

            for table, vals in table_values.items():
                col_name_map = {c_name: c.name for c_name, c in vals.items()}
                values = {
                    col_name_map[col_name]: col
                    for col_name, col in select.columns.items()
                    if col_name in col_name_map
                }

                # Prepare update statement.
                if self.db.engine.dialect.name in (
                    "postgres",
                    "postgresql",
                    "duckdb",
                    "mysql",
                    "mariadb",
                    "sqlite",
                ):
                    # Update-from.
                    statements.append(
                        table.update()
                        .values(values)
                        .where(
                            reduce(
                                sqla.and_,
                                (
                                    table.c[col.name] == select.c[col_name]
                                    for col_name, col in vals.items()
                                    if col.name in table.primary_key.columns
                                ),
                            ),
                        )
                    )
                else:
                    # Correlated update.
                    raise NotImplementedError("Correlated update not supported yet.")

        # Execute delete / insert / update statements.
        with self.db.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        if isinstance(self, Link):
            # Update link from parents to this table.
            to_fk_cols: list[sqla.ColumnElement] = [
                value_table.c[self[pk].fqn].label(fk.name)
                for fk, pk in self._fk_map.items()
            ]

            self.ctx._mutate_from_sql(
                sqla.select(value_table).add_columns(*to_fk_cols).subquery(),
                "update",
            )

        if self.db.backend_type == "excel-file":
            self.db._save_to_excel(
                # {self.record_type, *self.record_type._record_superclasses}
            )

        return


@final
class Public:
    """Demark public status of attribute."""


@final
class Private:
    """Demark private status of attribute."""


PubT = TypeVar("PubT", bound="Public | Private", default="Public")


@dataclass(kw_only=True, eq=False)
class Value(
    Data[ValT, Idx[()], RwT, None, Ctx[ParT], BaseT],
    Generic[ValT, RwT, PubT, ParT, BaseT],
):
    """Single-value attribute or column."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: Idx[()],
        CrudT: 1,
        RelT: NoneType,
        CtxT: Ctx,
        BaseT: 4,
    }

    alias: str | None = None
    init: bool = True
    default: ValT | type[Undef] = (
        Undef  # pyright: ignore[reportIncompatibleVariableOverride]
    )
    default_factory: Callable[[], ValT] | None = None

    getter: Callable[[Record], ValT] | None = None
    setter: Callable[[Record, ValT], None] | None = None
    pub_status: type[PubT] = Public  # pyright: ignore[reportAssignmentType]

    primary_key: bool = False
    index: bool = False
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = None

    def __post_init__(self) -> None:
        """Post init."""
        if self.pub_status is Public:
            self._selection = [cast(Value[Any, Any, Public, Any, Any], self)]

    @cached_prop
    def name(self) -> str:
        """Property name."""
        if self.alias is not None:
            return self.alias

        return super().name


# Redefine IdxT in an attempt to fix randomly occuring type error,
# in which IdxT isn't recognized as a type variable below anymore.
TdxT = TypeVar(
    "TdxT",
    covariant=True,
    bound=Idx | BaseIdx,
    default=BaseIdx,
)


@dataclass(kw_only=True, eq=False)
class Table(
    Data[RefT, TdxT, CrudT, RelT, CtxT, BaseT],
    Generic[RefT, RelT, CrudT, TdxT, CtxT, BaseT],
):
    """Record set."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: 3,
        CrudT: 2,
        RelT: 1,
        CtxT: 4,
        BaseT: 5,
    }

    index_by: (
        Value[Any, Any, Public, Any, Symbolic]
        | Iterable[Value[Any, Any, Public, Any, Symbolic]]
        | None
    ) = None
    default: bool = False


@dataclass(kw_only=True, eq=False)
class Link(
    Table[RefT, None, CrudT, Idx[()], Ctx[ParT], BaseT],
    Generic[RefT, CrudT, ParT, BaseT],
):
    """Link to a single record."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: Idx[()],
        CrudT: 1,
        RelT: NoneType,
        CtxT: Ctx[Any],
        BaseT: 3,
    }

    fks: (
        Value[Any, Any, Any, Any, Symbolic]
        | dict[Value[Any, Any, Any, Any, Symbolic], Value[Any, Any, Any, Any, Symbolic]]
        | list[Value[Any, Any, Any, Any, Symbolic]]
    ) | None = None

    index: bool = False
    primary_key: bool = False


@dataclass(kw_only=True, eq=False)
class BackLink(
    Table[RecT, None, RwT, TdxT, Ctx[ParT], BaseT],
    Generic[RecT, RwT, TdxT, ParT, BaseT],
):
    """Backlink record set."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: 2,
        CrudT: 1,
        RelT: NoneType,
        CtxT: Ctx[Any],
        BaseT: 4,
    }

    link: Data[ParT, Idx[()], Any, None, Ctx[RecT], Symbolic]


type ItemType = Item

_item_classes: dict[str, type[Item]] = {}


@dataclass(eq=False)
class Array(
    Data[
        ValT,
        ArrayIdx[Any, KeyT],
        CrudT,
        "Item[ValT, KeyT, OwnT]",
        Ctx[OwnT],
        BaseT,
    ],
    Generic[ValT, KeyT, CrudT, OwnT, BaseT],
):
    """Set / array of scalar values."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: ArrayIdx[Any, Any],
        CrudT: 2,
        RelT: ItemType,
        CtxT: Ctx[Any],
        BaseT: 4,
        KeyT: 1,
    }

    @cached_prop
    def _key_type(self) -> type[KeyT]:
        args = self._generic_args
        if len(args) == 0:
            return cast(type[KeyT], object)

        return cast(type[KeyT], args[1])

    @cached_prop
    def relation_type(self) -> type[Item[ValT, KeyT, OwnT]]:
        """Return the dynamic relation record type."""
        base_array_fqn = copy_and_override(Array, self, _ctx=self._symbolic_ctx).fqn

        rec = _item_classes.get(
            base_array_fqn,
            dynamic_record_type(
                Item[self.target_type, self._key_type, self._symbolic_ctx.record_type],
                f"{self._symbolic_ctx.record_type.__name__}.{self.name}",
                src_module=self._symbolic_ctx.record_type._src_mod
                or getmodule(self._symbolic_ctx.record_type),
                extra_attrs={
                    "_array": copy_and_override(
                        Array, self, _db=symbol_db, _ctx=self._symbolic_ctx
                    )
                },
            ),
        )
        _item_classes[base_array_fqn] = rec

        return rec


RootKeyT = TypeVar("RootKeyT", bound=Hashable, default=None)


@dataclass
class DataBase(Data[SchemaT, RootIdx[RootKeyT], CrudT, None, None, BaseT]):
    """Database connection."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: RootIdx,
        CrudT: 2,
        RelT: NoneType,
        CtxT: NoneType,
        BaseT: 3,
    }

    backend: BaseT = None  # pyright: ignore[reportAssignmentType]
    """Unique name to identify this database's backend by."""

    url: sqla.URL | CloudPath | HttpFile | Path | None = None
    """Connection URL or path."""

    schema: (
        type[SchemaT]
        | Mapping[
            type[SchemaT],
            Literal[True] | Require | str | sqla.TableClause,
        ]
        | set[type[SchemaT]]
        | None
    ) = None

    write_to_overlay: str | None = None
    overlay_type: OverlayType = "name_prefix"

    validate_on_init: bool = False
    remove_cross_fks: bool = False

    _def_types: Mapping[
        type[Record], Literal[True] | Require | str | sqla.TableClause
    ] = field(default_factory=dict)
    _subs: dict[type[Record], sqla.TableClause] = field(default_factory=dict)

    _metadata: sqla.MetaData = field(default_factory=sqla.MetaData)
    _valid_caches: dict[type[Record], set[Hashable]] = field(default_factory=dict)
    _instance_map: dict[type[Record], dict[Hashable, Record]] = field(
        default_factory=dict
    )

    _db_id: str | None = None

    def __post_init__(self):  # noqa: D105
        records = {
            rec: sub for rec, sub in self._schema_map.items() if issubclass(rec, Record)
        }

        # Handle Record classes in schema argument.
        self._subs = {
            **self._subs,
            **{
                rec: (sub if isinstance(sub, sqla.TableClause) else sqla.table(sub))
                for rec, sub in records.items()
                if not isinstance(sub, Require | bool)
            },
        }
        self._def_types = {**self._def_types, **records}  # type: ignore

        schemas = {
            schema: schema_name
            for schema, schema_name in self._schema_map.items()
            if issubclass(schema, Schema)
        }

        # Handle Schema classes in schema argument.
        self._subs = {
            **self._subs,
            **{
                rec: sqla.table(rec._default_table_name(), schema=schema_name)
                for schema, schema_name in schemas.items()
                for rec in schema._schema_types
                if isinstance(schema_name, str)
            },
        }

        self._def_types = cast(
            dict,
            {
                **self._def_types,
                **{
                    rec: (
                        req
                        if not isinstance(req, str)
                        else sqla.table(rec._default_table_name(), schema=req)
                    )
                    for schema, req in schemas.items()
                    for rec in schema._schema_types
                },
            },
        )

        if self.validate_on_init:
            self.validate()

        if self.write_to_overlay is not None and self.overlay_type == "db_schema":
            self._ensure_sqla_schema_exists(self.write_to_overlay)

        self._db = self
        self._name = self.db_id
        self._type = cast(
            type[SchemaT],
            (
                self.schema
                if isinstance(self.schema, type)
                else Union[*self._def_types] if len(self._def_types) > 0 else None
            ),
        )

    @property
    def db_id(self) -> str:
        """Return the unique database ID."""
        db_id = self._db_id
        if db_id is None:
            if isinstance(self.backend, Symbolic):
                db_id = "symbolic"
            elif self.backend is None:
                db_id = token_hex(5)
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
            case HttpFile():
                url = yarl.URL(self.url.url)
                typ = (
                    ("excel-file" if "xls" in Path(url.path).suffix else "sqlite-file")
                    if url.scheme in ("http", "https")
                    else None
                )
                if typ is None:
                    raise ValueError(f"Unsupported URL scheme: {url.scheme}")
                return typ
            case None:
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

    def describe(self) -> dict[str, str | dict[str, str] | None]:
        """Return a description of this database."""
        schema_desc = {}
        if not isinstance(self.db.backend, Symbolic):
            dyn_self = cast(DataBase[Any, Any, Any, DynBackendID], self)

            schema_ref = (
                PyObjectRef.reference(self.schema)
                if isinstance(self.schema, type) and issubclass(self.schema, Schema)
                else None
            )

            schema_desc = {
                **(
                    {
                        "schema": {
                            attr: val
                            for attr, val in asdict(schema_ref).items()
                            if attr not in ["object_type"] and val is not None
                        }
                    }
                    if schema_ref is not None
                    else {}
                ),
                "contents": {
                    "records": {
                        rec._fqn: len(dyn_self[rec])
                        for rec in DataBase[Any, Any, Any, Any]._record_types(
                            self, with_relations=False
                        )
                    },
                    "arrays": {
                        item._fqn: len(dyn_self[item]) for item in self._item_types
                    },
                    "relations": {
                        rel._fqn: len(dyn_self[rel]) for rel in self._relation_types
                    },
                },
            }

        return {
            **schema_desc,
            "backend": (
                str(self.url)
                if self.url is not None and self.backend_type != "in-memory"
                else None
            ),
            **(
                dict(overlay=self.write_to_overlay)
                if self.write_to_overlay is not None
                else {}
            ),
        }

    def validate(self) -> None:
        """Perform pre-defined schema validations."""
        types = {rec: isinstance(req, Require) for rec, req in self._def_types.items()}

        tables = {
            self[b_rec]._gen_sql_base_table(): required
            for rec, required in types.items()
            for b_rec in [rec, *rec._record_superclasses]
        }

        inspector = sqla.inspect(self.engine)

        # Iterate over all tables and perform validations for each
        for table, required in tables.items():
            has_table = inspector.has_table(table.name, table.schema)

            if not has_table and not required:
                continue

            # Check if table exists
            assert has_table

            db_columns = {
                c["name"]: c for c in inspector.get_columns(table.name, table.schema)
            }
            for column in table.columns:
                # Check if column exists
                assert column.name in db_columns

                db_col = db_columns[column.name]

                # Check if column type and nullability match
                assert isinstance(db_col["type"], type(column.type))
                assert db_col["nullable"] == column.nullable or column.nullable is None

            # Check if primary key is compatible
            db_pk = inspector.get_pk_constraint(table.name, table.schema)
            if len(db_pk["constrained_columns"]) > 0:  # Allow source tbales without pk
                assert set(db_pk["constrained_columns"]) == set(
                    table.primary_key.columns.keys()
                )

            # Check if foreign keys are compatible
            db_fks = inspector.get_foreign_keys(table.name, table.schema)
            for fk in table.foreign_key_constraints:
                matches = [
                    (
                        set(db_fk["constrained_columns"]) == set(fk.column_keys),
                        (
                            db_fk["referred_table"].lower()
                            == fk.referred_table.name.lower()
                        ),
                        set(db_fk["referred_columns"])
                        == set(f.column.name for f in fk.elements),
                    )
                    for db_fk in db_fks
                ]

                assert any(all(m) for m in matches)

    def load(
        self: DataBase[Any, Any, CRUD, BackT2],
        data: pd.DataFrame | pl.DataFrame | sqla.Select,
        fks: (
            Mapping[
                str,
                Value[Any, Any, Public, Any, Symbolic],
            ]
            | None
        ) = None,
    ) -> Data[DynRecord, BaseIdx[DynRecord], R, None, None, BackT2]:
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, pd.DataFrame | pl.DataFrame)
            else f"temp_{token_hex(5)}"
        )

        rec = dynamic_record_type(
            DynRecord,
            name,
            props=props_from_data(
                data,
                ({name: col for name, col in fks.items()} if fks is not None else None),
            ),
        )

        self[rec] = data

        return self[rec]

    def __or__(
        self: DataBase[Any, Any, Any, BackT2],
        other: DataBase[SchemaT2, Any, Any, DynBackendID],
    ) -> DataBase[SchemaT | SchemaT2, Any, CRUD, BackT2]:
        """Union two databases, right overriding left."""
        db = copy_and_override(
            DataBase[SchemaT | SchemaT2, Any, CRUD, BackT2],
            self,
            backend=self.backend,
            schema={**self._schema_map, **self._schema_map},  # type: ignore
            write_to_overlay=f"upsert/({self.db_id}|{other.db_id})/{token_hex(4)}",
            _def_types={},
            _metadata=sqla.MetaData(),
            _instance_map={},
        )

        db._mutate(other, "upsert")

        return db

    def __lshift__(
        self: DataBase[Any, Any, Any, BackT2],
        other: DataBase[SchemaT2, Any, Any, DynBackendID],
    ) -> DataBase[SchemaT | SchemaT2, Any, CRUD, BackT2]:
        """Union two databases, left overriding right."""
        db = copy_and_override(
            DataBase[SchemaT | SchemaT2, Any, CRUD, BackT2],
            self,
            backend=self.backend,
            schema={**self._schema_map, **self._schema_map},  # type: ignore
            write_to_overlay=f"insert/({self.db_id}<<{other.db_id})/{token_hex(4)}",
            _def_types={},
            _metadata=sqla.MetaData(),
            _instance_map={},
        )

        db._mutate(other, "insert")

        return db

    def __rshift__(
        self: DataBase[Any, Any, Any, BackT2],
        other: DataBase[SchemaT2, Any, Any, DynBackendID],
    ) -> DataBase[SchemaT | SchemaT2, Any, CRUD, BackT2]:
        """Intersect two databases, right overriding left."""
        db = copy_and_override(
            DataBase[SchemaT | SchemaT2, Any, CRUD, BackT2],
            self,
            backend=self.backend,
            schema={**self._schema_map, **self._schema_map},  # type: ignore
            write_to_overlay=f"update/({self.db_id}>>{other.db_id})/{token_hex(4)}",
            _def_types={},
            _metadata=sqla.MetaData(),
            _instance_map={},
        )

        db._mutate(other, "update")

        return db

    def __hash__(self) -> int:
        """Hash the DB."""
        return gen_int_hash(
            (
                self.db_id,
                self.url,
                self._subs,
            )
        )

    @cached_prop
    def _schema_map(
        self,
    ) -> Mapping[
        type[Record | Schema], Literal[True] | Require | str | sqla.TableClause
    ]:
        return (
            {self.schema: True}
            if isinstance(self.schema, type)
            else (
                cast(
                    dict[type[Record | Schema], Literal[True]],
                    {rec: True for rec in self.schema},
                )
                if isinstance(self.schema, set)
                else self.schema if self.schema is not None else {}
            )
        )

    @cached_prop
    def _record_types(
        self, with_relations: bool = True
    ) -> Mapping[type[Record], Literal[True] | Require | str | sqla.TableClause]:
        """Set of all schema types in this DB."""
        types = dict(self._def_types).copy()

        for rec in types:
            types = {
                **{r: True for r in rec._rel_types(with_relations=with_relations)},
                **types,
            }

        return types

    @property
    def _item_types(self) -> set[type[Item[Any, Any, Record]]]:
        return {t for t in self._record_types if issubclass(t, Item)}

    @property
    def _relation_types(self) -> set[type[Relation[Record, Record]]]:
        return {t for t in self._record_types if issubclass(t, Relation)}

    def _get_valid_cache_set(self, rec: type[Record]) -> set[Hashable]:
        """Get the valid cache set for a record type."""
        if rec not in self._valid_caches:
            self._valid_caches[rec] = set()

        return self._valid_caches[rec]

    def _get_instance_map(self, rec: type[Record]) -> dict[Hashable, Record]:
        """Get the instance map for a record type."""
        if rec not in self._instance_map:
            self._instance_map[rec] = {}

        return self._instance_map[rec]

    def _create_sqla_table(self, sqla_table: sqla.Table) -> None:
        """Create SQL-side table from Table class."""
        if self.remove_cross_fks:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            _remove_cross_fk(sqla_table)

        sqla_table.create(self.engine, checkfirst=True)

    def _ensure_sqla_schema_exists(self, schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_schema(schema_name):
            with self.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name

    def _load_from_excel(self, targets: set[type[Record]] | None = None) -> None:
        """Load all tables from Excel."""
        assert isinstance(self.url, Path | CloudPath | HttpFile)
        path = self.url.get() if isinstance(self.url, HttpFile) else self.url

        if isinstance(path, Path) and not path.exists():
            return

        recs = targets if targets is not None else self._record_types.keys()

        with open(path, "rb") as file:
            for rec in recs:
                table = self[rec]._gen_sql_base_table("replace")

                with self.engine.begin() as conn:
                    conn.execute(table.delete().where(sqla.true()))

                df = pl.read_excel(file, sheet_name=rec._get_table_name(self._subs))
                df.write_database(
                    str(table),
                    self.df_engine,
                    if_table_exists="append",
                )

    def _save_to_excel(self) -> None:
        """Save all (or selected) tables to Excel."""
        assert isinstance(self.url, Path | CloudPath | HttpFile)
        file = self.url.get() if isinstance(self.url, HttpFile) else self.url

        recs = self._record_types.keys()

        with ExcelWorkbook(file) as wb:
            for rec in recs:
                pl.read_database(
                    self[rec]._gen_sql_base_table().select(),
                    self.engine.connect(),
                ).write_excel(wb, worksheet=rec._get_table_name(self._subs))

        if isinstance(self.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.url.set(file)


symbol_db = DataBase[Any, Any, R, Symbolic](backend=Symbolic())
"""Database for all symbolic data instances."""


class RecordMeta(type):
    """Metaclass for record types."""

    _record_superclasses: list[type[Record]]

    _is_root_class: bool = False
    _template: bool = False
    _src_mod: ModuleType | None = None
    _derivate: bool = False

    __class_props: dict[str, Data[Any, Any, Any, Any, Ctx[Record], Symbolic]] | None

    @staticmethod
    def _get_typevar_map(
        c: SingleTypeDef[Record],
    ) -> dict[TypeVar, SingleTypeDef | UnionType]:
        orig = c if isinstance(c, RecordMeta) else get_origin(c)

        if not isinstance(orig, RecordMeta):
            return {}

        own_typevar_map = (
            dict(zip(getattr(orig, "__parameters__"), get_args(c)))
            if orig is not c and hasattr(orig, "__parameters__")
            else {}
        )

        super_typevar_map = reduce(
            lambda x, y: x | y,
            (RecordMeta._get_typevar_map(sup) for sup in orig._super_types),
        )
        return {**super_typevar_map, **own_typevar_map}

    def __init__(cls, name, bases, dct):
        """Initialize a new record type."""
        super().__init__(name, bases, dct)

        if "_src_mod" not in cls.__dict__:
            cls._src_mod = getmodule(cls if not cls._derivate else bases[0])

        cls.__class_props = None
        cls._get_class_props()

    @property
    def __dataclass_fields__(cls) -> dict[str, Field]:  # noqa: D105
        return {
            **{
                name: Field(
                    a.default,
                    a.default_factory,  # pyright: ignore[reportArgumentType]
                    a.init,
                    hash=a.primary_key,
                    repr=True,
                    metadata={},
                    compare=True,
                    kw_only=True,
                )
                for name, a in cls._values.items()
            },
            **{
                name: Field(
                    MISSING,
                    lambda: MISSING,
                    init=True,
                    repr=True,
                    hash=False,
                    metadata={},
                    compare=True,
                    kw_only=True,
                )
                for name, _ in cls._links.items()
            },
        }

    @property
    def _super_types(cls) -> Iterable[type | GenericProtocol]:
        return (
            cls.__dict__["__orig_bases__"]
            if "__orig_bases__" in cls.__dict__
            else cls.__bases__
        )

    def _get_class_props(cls) -> dict[str, Data[Any, Any, Any, Any, Ctx, Symbolic]]:
        if cls.__class_props is not None:
            return cls.__class_props

        ctx: Ctx | Data[Record | None, Any, Any, Any, Any, Any] = (
            Ctx(cast(type["Record"], cls))
            if not hasattr(cls, "_ctx")
            else getattr(cls, "_ctx")
        )

        # Construct prelimiary symbolic data instances for all class annotations.
        props = {
            name: Data[Any, Any, Any, Any, Ctx, Symbolic](
                _type=hint,
                _db=symbol_db,
                _ctx=ctx,
                _name=name,
            )
            for name, hint in get_annotations(cls).items()
        }

        # Filter out all non-Data annotations.
        props = {
            name: prop
            for name, prop in props.items()
            if is_subtype(prop._data_type, Data)
        }

        # Merge data definitions from annotations with those from the class body.
        for prop_name, prop in props.items():
            prop_type = cast(
                type[Data[Any, Any, Any, Any, Ctx["Record"], Symbolic]],
                prop._data_type,
            )

            if prop_name in cls.__dict__:
                prop = copy_and_override(
                    prop_type,
                    cls.__dict__[prop_name],
                    _name=prop._name,
                    _type=prop._type,
                    _ctx=prop._ctx,
                )
            else:
                prop = copy_and_override(prop_type, prop)

            setattr(cls, prop_name, prop)
            props[prop_name] = prop

        # Construct list of Record superclasses and apply template superclasses.
        cls._record_superclasses = []

        typevar_map = {}
        for c in cls._super_types:
            # Get proper origin class of generic supertype.
            orig = get_origin(c) if not isinstance(c, type) else c

            if orig is None:
                continue

            # Handle typevar substitutions.
            typevar_map = cls._get_typevar_map(c)

            # Add self to schema classes, if superclass is a Schema.
            if issubclass(orig, Schema):
                orig._schema_types.add(cls)

            # Skip root Record class and non-Record classes.
            if not isinstance(orig, RecordMeta) or orig._is_root_class:
                continue

            # Apply template classes.
            if orig._template or cls._derivate:
                # Pre-collect all template props
                # to prepend them in order.
                orig_props = {}
                for prop_name, super_prop in orig._get_class_props().items():
                    if prop_name in props:
                        prop = copy_and_override(
                            type(props[prop_name]), (super_prop, props[prop_name])
                        )
                    else:
                        prop = copy_and_override(
                            type(super_prop),
                            super_prop,
                            _typevar_map=typevar_map,
                            _ctx=ctx,
                        )

                    setattr(cls, prop_name, prop)
                    orig_props[prop_name] = prop

                props = {**orig_props, **props}  # Prepend template props.
            else:
                assert orig is c  # Must be concrete class, not a generic
                cls._record_superclasses.append(orig)

        cls.__class_props = props
        return props

    @property
    def _class_links(
        cls,
    ) -> dict[str, Link[Any, Any, Any, Symbolic]]:
        """The relations of this record type without superclasses."""
        return {
            name: ref
            for name, ref in cls._get_class_props().items()
            if isinstance(ref, Link)
        }

    @property
    def _class_fk_values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        """The foreign key columns of this record type without superclasses."""
        return {
            a.name: a
            for ref in cls._class_links.values()
            if isinstance(ref, Link)
            for a in ref._fk_map.keys()
        }

    @property
    def _props(cls) -> dict[str, Data[Any, Any, Any, Any, Ctx, Symbolic]]:
        """The statically defined properties of this record type."""
        own_props = {**cls._get_class_props(), **cls._class_fk_values}
        return reduce(
            lambda x, y: {**x, **y},
            (c._props for c in cls._record_superclasses),
            own_props,
        )

    @property
    def _pk_values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        """The primary key columns of this record type."""
        return {
            name: a
            for name, a in cls._props.items()
            if isinstance(a, Value) and a.primary_key
        }

    @property
    def _class_values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        """The columns of this record type without superclasses."""
        return (
            cls._class_fk_values
            | cls._pk_values
            | {
                k: a
                for k, a in cls._get_class_props().items()
                if isinstance(a, Value) and a.pub_status is Public
            }
        )

    @property
    def _fk_values(cls) -> Mapping[str, Value[Any, Any, Any, Any, Symbolic]]:
        """The foreign key columns of this record type."""
        return reduce(
            lambda x, y: {**x, **y},
            (c._class_fk_values for c in cls._record_superclasses),
            cls._class_fk_values,
        )

    @property
    def _values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        return {k: c for k, c in cls._props.items() if isinstance(c, Value)}

    @property
    def _col_values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        return {k: a for k, a in cls._values.items() if a.pub_status is Public}

    @property
    def _non_fk_values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        return {k: c for k, c in cls._col_values.items() if k not in cls._fk_values}

    @property
    def _tables(cls) -> dict[str, Table[Record | None, Any, Any, Any, Ctx, Symbolic]]:
        return {k: r for k, r in cls._props.items() if isinstance(r, Table)}

    @property
    def _links(cls) -> dict[str, Link[Record | None, Any, Any, Symbolic]]:
        return {k: r for k, r in cls._tables.items() if isinstance(r, Link)}

    @property
    def _arrays(cls) -> dict[str, Array[Any, Any, Any, Any, Symbolic]]:
        return {k: c for k, c in cls._props.items() if isinstance(c, Array)}

    @property
    def _fqn(cls) -> str:
        mod_name: str = (
            getattr(cls._src_mod, "__name__")
            if cls._src_mod is not None
            else cls.__module__
        )

        return mod_name + "." + cls.__name__

    def _rel_types(
        cls, _traversed: set[type[Record]] | None = None, with_relations: bool = True
    ) -> set[type[Record]]:
        direct_rel_types = {t.record_type for t in cls._tables.values()}

        if with_relations:
            direct_rel_types |= {
                t.relation_type
                for t in (*cls._tables.values(), *cls._arrays.values())
                if issubclass(t.relation_type, Record)
            }

        _traversed = _traversed or set()
        to_traverse = direct_rel_types - _traversed

        return (
            direct_rel_types
            if len(to_traverse) == 0
            else direct_rel_types
            | reduce(
                set.union,
                (
                    t._rel_types(_traversed | to_traverse, with_relations)
                    for t in to_traverse
                ),
            )
        )


class Schema:
    """Group multiple record types into a schema."""

    _schema_types: set[type[Record]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        cls._schema_types = set()
        super().__init_subclass__()


class RecordTable:
    """Table, which a record belongs to."""

    @overload
    def __get__(
        self, instance: None, owner: type[RecT2]
    ) -> Table[RecT2, None, Any, BaseIdx[RecT2], None, Symbolic]: ...

    @overload
    def __get__(
        self, instance: RecT2, owner: type[RecT2]
    ) -> Table[RecT2, None, Any, BaseIdx[RecT2], None, DynBackendID]: ...

    def __get__(  # noqa: D105
        self, instance: Record | None, owner: type[Record]
    ) -> Table[Record, None, Any, BaseIdx[Record], None, Any]:
        if instance is not None:
            table = instance._db[owner]
            if not instance._published:
                table |= instance
                instance._published = True
        else:
            table = symbol_db[owner]

        assert isinstance(table, Table)
        return table


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Value, Link, BackLink, Table, Array),
    eq_default=False,
)
class Record(Generic[*KeyTt], metaclass=RecordMeta):
    """Schema for a record in a database."""

    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID4: sqla.types.CHAR(36),
    }

    _template: ClassVar[bool]
    _derivate: ClassVar[bool] = False

    _is_root_class: ClassVar[bool] = True
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    _table: RecordTable = RecordTable()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)
        cls._is_root_class = False
        if "_template" not in cls.__dict__:
            cls._template = False

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
    def _get_table_name(
        cls,
        subs: Mapping[type[Record], sqla.TableClause],
    ) -> str:
        """Return a SQLAlchemy table object for this schema."""
        sub = subs.get(cls)
        return sub.name if sub is not None else cls._default_table_name()

    @classmethod
    def _find_backlinks(
        cls, target: type[RecT2]
    ) -> set[BackLink[RecT2, Any, Any, Self, Symbolic]]:
        """Get all direct relations from a target record type to this type."""
        return {
            BackLink(
                link=cast(Link[Self, Any, RecT2, Symbolic], ln),
                _ctx=Ctx(cls),
                _type=BackLink[target],
                _db=symbol_db,
            )
            for ln in target._links.values()
            if issubclass(cls, ln.target_type)
        }

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:  # noqa: D105
        assert cls._default_table_name() is not None
        return sqla.table(cls._default_table_name())

    @classmethod  # pyright: ignore[reportArgumentType]
    def _from_existing(
        cls: Callable[Params, RecT2],
        rec: Record[KeyT],
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> RecT2:
        return copy_and_override(
            cls,
            rec,
            *(arg if arg is not Keep else MISSING for arg in args),
            **{k: v if v is not Keep else MISSING for k, v in kwargs.items()},
        )  # pyright: ignore[reportCallIssue]

    _database: Value[DataBase[Any, Any, CRUD, DynBackendID] | None, CRUD, Private] = (
        Value(
            pub_status=Private,
            default=None,
        )
    )
    _published: Value[bool, CRUD, Private] = Value(pub_status=Private, default=False)
    _root: Value[bool, CRUD, Private] = Value(pub_status=Private, default=True)
    _index: Value[tuple[*KeyTt], CRUD, Private] = Value(pub_status=Private, init=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record instance."""
        super().__init__()

        cls = type(self)

        values = {name: val for name, val in kwargs.items() if name in cls._values}
        links = {name: val for name, val in kwargs.items() if name in cls._links}
        arrays = {name: val for name, val in kwargs.items() if name in cls._arrays}
        tables = {
            name: val
            for name, val in kwargs.items()
            if name in cls._tables and name not in cls._links
        }
        others = {name: val for name, val in kwargs.items() if name not in cls._props}

        # First set all attributes.
        for name, value in values.items():
            setattr(self, name, value)

        # Then set all direct relations.
        for name, value in links.items():
            setattr(self, name, value)

        # Then all attribute sets.
        for name, value in arrays.items():
            setattr(self, name, value)

        # Set all indirect relations.
        for name, value in tables.items():
            setattr(self, name, value)

        # Set all other attributes.
        for name, value in others.items():
            setattr(self, name, value)

        self.__post_init__()

        pks = type(self)._pk_values
        if len(pks) == 1:
            self._index = getattr(self, next(iter(pks)))
        else:
            self._index = cast(tuple[*KeyTt], tuple(getattr(self, pk) for pk in pks))

        if self._database is not None and not self._published:
            self._db[type(self)] |= self
            self._published = True

        return

    @property
    def _db(self) -> DataBase[Any, Any, CRUD, DynBackendID]:
        if self._database is None:
            self._database = DataBase[Any, Any, CRUD, DynBackendID]()

        return self._database

    def __post_init__(self) -> None:  # noqa: D105
        pass

    def __hash__(self) -> int:
        """Identify the record by database and id."""
        return gen_int_hash((self._db if self._published else None, self._index))

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)

    @overload
    def _to_dict(
        self,
        name_keys: Literal[True] = ...,
        include: set[type[Data]] | None = ...,
        with_fks: bool = ...,
        with_private: bool = ...,
    ) -> dict[str, Any]: ...

    @overload
    def _to_dict(
        self,
        name_keys: Literal[False],
        include: set[type[Data]] | None = ...,
        with_fks: bool = ...,
        with_private: bool = ...,
    ) -> dict[Data[Any, Any, Any, None, Ctx[Self], Symbolic], Any]: ...

    def _to_dict(
        self,
        name_keys: bool = True,
        include: set[type[Data[Any, Any, Any, Any, Any, Any]]] | None = None,
        with_fks: bool = True,
        with_private: bool = False,
    ) -> dict[Data[Any, Any, Any, None, Ctx[Self], Symbolic], Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Data[Any, Any, Any, Any, Any, Any]], ...] = (
            tuple(include) if include is not None else (Value,)
        )

        vals = {
            p if not name_keys else p.name: getattr(self, p.name)
            for p in type(self)._props.values()
            if isinstance(p, include_types)
            and (with_fks or p.name not in type(self)._fk_values)
            and (not isinstance(p, Value) or p.pub_status is Public or with_private)
        }

        return cast(dict, vals)

    @classmethod
    def _is_complete_dict(
        cls,
        data: Mapping[Data[Any, Any], Any] | Mapping[str, Any],
    ) -> bool:
        """Check if dict data contains all required info for record type."""
        data = {(p if isinstance(p, str) else p.name): v for p, v in data.items()}
        return all(
            a.name in data
            or a.getter is not None
            or a.default is not Undef
            or a.default_factory is not None
            for a in cls._non_fk_values.values()
            if a.init is not False
        ) and all(
            r.name in data or all(fk.name in data for fk in r._fk_map.keys())
            for r in cls._links.values()
        )

    def _update_dict(self) -> None:
        table = self._table[[self._index]]
        df = table.df()
        rec_dict = list(df.iter_rows(named=True))[0]
        self.__dict__.update(rec_dict)

    def __repr__(self) -> str:
        """Return a string representation of the record."""
        return f"{type(self).__name__}({repr(self._to_dict(with_fks=False))})"


@dataclass
class Require:
    """Mark schema or record type as required."""

    present: bool = True


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(
        cls: type[Record], name: str
    ) -> Value[Any, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by dynamic name."""
        return Value(_db=symbol_db, _type=Value[cls], _ctx=Ctx(cls), _name=name)

    def __getattr__(
        cls: type[Record], name: str
    ) -> Value[Any, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)
        return Value(_db=symbol_db, _type=Value[cls], _ctx=Ctx(cls), _name=name)


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


x = DynRecord


def dynamic_record_type(
    base: type[RecT2],
    name: str,
    props: Iterable[Data[Any, Any, Any, Any, Any, Any]] = [],
    src_module: ModuleType | None = None,
    extra_attrs: dict[str, Any] = {},
) -> type[RecT2]:
    """Create a dynamically defined record type."""
    return cast(
        type[RecT2],
        new_class(
            name,
            (base,),
            None,
            lambda ns: ns.update(
                {
                    **{p.name: p for p in props},
                    "__annotations__": {p.name: p._type for p in props},
                    "_src_mod": src_module or base._src_mod or getmodule(base),
                    **extra_attrs,
                }
            ),
        ),
    )


class RecUUID(Record[UUID4]):
    """Record type with a default UUID4 primary key."""

    _template = True
    _id: Value[UUID4] = Value(primary_key=True, default_factory=lambda: UUID4(uuid4()))


class RecHashed(Record[str]):
    """Record type with a default hashed primary key."""

    _template = True

    _id: Value[str] = Value(primary_key=True, init=False)

    def __post_init__(self) -> None:  # noqa: D105
        self._id = gen_str_hash(
            {
                a.name: getattr(self, a.name)
                for a in type(self)._values.values()
                if a.name != "_id"
            }
        )


class BacklinkRecord(Record[KeyT2], Generic[KeyT2, RecT2]):
    """Dynamically defined record type."""

    _template = True

    _from: Link[RecT2]


class Item(BacklinkRecord[KeyT2, RecT2], Generic[ValT, KeyT2, RecT2]):
    """Dynamically defined scalar record type."""

    _template = True
    _array: ClassVar[Array[Any, Any, Any, Any, Symbolic]]

    _from: Link[RecT2] = Link(primary_key=True)
    idx: Value[KeyT2] = Value(primary_key=True)
    value: Value[ValT]


class Relation(RecHashed, BacklinkRecord[str, RecT2], Generic[RecT2, RecT3]):
    """Automatically defined relation record type."""

    _template = True

    _to: Link[RecT3] = Link(index=True)


class IndexedRelation(Relation[RecT2, RecT3], Generic[RecT2, RecT3, KeyT]):
    """Automatically defined relation record type with index substitution."""

    _template = True

    _rel_id: Value[KeyT] = Value(index=True)
