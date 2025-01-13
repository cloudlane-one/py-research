"""Static schemas for universal relational databases."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import MISSING, Field, dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import cache, partial, reduce
from inspect import get_annotations, getmodule
from io import BytesIO
from itertools import chain, groupby
from pathlib import Path
from secrets import token_hex
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
    TypeVarTuple,
    cast,
    dataclass_transform,
    final,
    get_args,
    get_origin,
    overload,
)
from uuid import UUID, uuid4

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.dialects.mysql as mysql
import sqlalchemy.dialects.postgresql as postgresql
import sqlalchemy.dialects.sqlite as sqlite
import sqlalchemy.orm as orm
import sqlalchemy.sql.visitors as sqla_visitors
import sqlalchemy.util as sqla_util
import yarl
from bidict import bidict
from cloudpathlib import CloudPath
from duckdb_engine import DuckDBEngineWarning
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from sqlalchemy_utils import UUIDType
from typing_extensions import TypeVar
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.caching import cached_method, cached_prop
from py_research.data import copy_and_override
from py_research.files import HttpFile
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericProtocol,
    SingleTypeDef,
    extract_nullable_type,
    get_lowest_common_base,
    has_type,
    is_subtype,
)

ValT = TypeVar("ValT", covariant=True)
ValT2 = TypeVar("ValT2")
ValT3 = TypeVar("ValT3")
ValT4 = TypeVar("ValT4")
ValTt = TypeVarTuple("ValTt")
ValTt2 = TypeVarTuple("ValTt2")
ValTt3 = TypeVarTuple("ValTt3")

CrudT = TypeVar("CrudT", bound="C | R", default="CRUD", covariant=True)
CrudT2 = TypeVar("CrudT2", bound="C | R")
CrudT3 = TypeVar("CrudT3", bound="C | R")

RwT = TypeVar("RwT", bound="C | R", default="RU", covariant=True)
RwT2 = TypeVar("RwT2", bound="C | R")
RwT3 = TypeVar("RwT3", bound="C | R")


RecT = TypeVar("RecT", bound="Record", covariant=True, default=Any)
RecT2 = TypeVar("RecT2", bound="Record")
RecT3 = TypeVar("RecT3", bound="Record")
RecT4 = TypeVar("RecT4", bound="Record")

OwnT = TypeVar("OwnT", default=Any, bound=Record)

CtxT = TypeVar("CtxT", bound="Ctx| None", covariant=True, default=None)
CtxT2 = TypeVar("CtxT2", bound="Ctx | None")
CtxT3 = TypeVar("CtxT3", bound="Ctx | None")

ParT = TypeVar("ParT", contravariant=True, bound="Record", default=Any)


KeyT = TypeVar("KeyT", bound="Hashable | NoIdx", default=Any)
KeyT2 = TypeVar("KeyT2", bound="Hashable | NoIdx")
KeyT3 = TypeVar("KeyT3", bound="Hashable | NoIdx")
KeyTt = TypeVarTuple("KeyTt")
KeyTt2 = TypeVarTuple("KeyTt2")
KeyTt3 = TypeVarTuple("KeyTt3")

IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound="Idx | BaseIdx",
    default="BaseIdx",
)
IdxT2 = TypeVar(
    "IdxT2",
    bound="Idx | BaseIdx",
)
IdxT3 = TypeVar(
    "IdxT3",
    bound="Idx | BaseIdx",
)

LnT = TypeVar("LnT", bound="Record | None", covariant=True, default=None)
LnT2 = TypeVar("LnT2", bound="Record | None")
LnT3 = TypeVar("LnT3", bound="Record | None")

RefT = TypeVar("RefT", bound="Record | None", covariant=True)
RefT2 = TypeVar("RefT2", bound="Record | None")

BaseT = TypeVar(
    "BaseT",
    bound="LiteralString | Symbolic | None",
    covariant=True,
    default=None,
)
BaseT2 = TypeVar(
    "BaseT2",
    bound="LiteralString | Symbolic | None",
)
BaseT3 = TypeVar(
    "BaseT3",
    bound="LiteralString | Symbolic | None",
)

BackT = TypeVar(
    "BackT",
    bound="LiteralString | None",
    covariant=True,
    default=None,
)
BackT2 = TypeVar(
    "BackT2",
    bound="LiteralString | None",
)
BackT3 = TypeVar(
    "BackT3",
    bound="LiteralString | None",
)

PubT = TypeVar("PubT", bound="Public | Private", default="Public")

OrdT = TypeVar("OrdT", bound="Ordinal")


DfT = TypeVar("DfT", bound=pd.DataFrame | pl.DataFrame)

Params = ParamSpec("Params")

type Ordinal = bool | int | float | Decimal | datetime | date | time | timedelta | UUID | str


@final
class Undef:
    """Demark undefined status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class Keep:
    """Demark unchanged status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


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


@final
class Symbolic:
    """Local backend."""


type DynBackendID = LiteralString | None


type Input[
    Val: Hashable, Key: Hashable, RKey: Hashable
] = pd.DataFrame | pl.DataFrame | Iterable[Val | RKey] | Mapping[
    Key, Val | RKey
] | sqla.Select | Val | RKey


class Idx(Generic[*KeyTt]):
    """Define the custom index type of a dataset."""


@final
class BaseIdx(Generic[RecT]):
    """Singleton to mark dataset as having the record type's base index."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


type NoIdx = Idx[()]


type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]


@final
@dataclass
class Ctx(Generic[ParT]):
    """Context record of a dataset."""

    record_type: type[ParT]


@final
class Public:
    """Demark public status of attribute."""


@final
class Private:
    """Demark private status of attribute."""


type LinkItem = Table[Record | None, None, Any, Any, Any, Any]

type PropPath[RootT: Record, LeafT] = tuple[
    Table[RootT, Any, Any, Any, None, Any]
] | tuple[
    Table[RootT, Any, Any, Any, None, Any],
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


_pl_type_map: dict[type, pl.DataType | type] = {
    UUID: pl.String,
}


def _get_pl_schema(attr_map: Mapping[str, Value[Any, Any, Any, Any, Any]]) -> pl.Schema:
    """Return the schema of the dataset."""
    return pl.Schema(
        {
            name: _pl_type_map.get(a.target_type, a.target_type)
            for name, a in attr_map.items()
        }
    )


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
        _typehint=Value[value_type],
    )
    return (
        attr
        if not is_link
        else Link(fks=fks[name], _typehint=fks[name]._typehint, _name=f"link_{name}")
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


@dataclass(kw_only=True, eq=False)
class Data(Generic[ValT, IdxT, CrudT, LnT, CtxT, BaseT]):
    """Relational data."""

    _typearg_map: ClassVar[dict[TypeVar, int]] = {
        ValT: 0,
        IdxT: 1,
        CrudT: 2,
        LnT: 3,
        CtxT: 4,
        BaseT: 5,
    }

    _db: DataBase[CrudT | CRUD, BaseT] | None = None
    _ctx: CtxT | Table[Record, Any, R, Any, Any, BaseT] | None = None

    _name: str | None = None
    _typehint: str | SingleTypeDef[Data[ValT, Any, Any, Any, Any, Any]] | None = None
    _typevar_map: dict[TypeVar, SingleTypeDef] = field(default_factory=dict)

    _filters: list[
        sqla.ColumnElement[bool] | tuple[slice | list[Hashable] | Hashable, ...]
    ] = field(default_factory=list)
    _tuple_selection: tuple[Data[Any, Any, Any, Any, Any, BaseT], ...] | None = None

    @cached_prop
    def db(self) -> DataBase[CrudT | CRUD, BaseT]:
        db = self._db

        if db is None and isinstance(self.base_type, NoneType):
            db = cast(DataBase[CRUD, BaseT], DataBase())

        if db is None:
            raise ValueError("Missing backend.")

        return db

    @property
    def ctx(
        self: Data[Any, Any, Any, Any, Ctx[RecT2], Any]
    ) -> Table[RecT2, Any, Any, Any, Any, BaseT]:
        assert self._ctx is not None
        return (
            cast(Table[RecT2, Any, Any, Any, Any, BaseT], self._ctx)
            if isinstance(self._ctx, Data)
            else Table(_db=self.db, _typehint=Table[self._ctx.record_type])
        )

    @cached_prop
    def name(self) -> str:
        return self._name if self._name is not None else token_hex(5)

    @cached_prop
    def value_type(self) -> SingleTypeDef[ValT]:
        """Resolve the value type reference."""
        args = self._generic_args
        if len(args) == 0:
            return cast(type[ValT], object)

        arg = args[self._typearg_map[ValT]]
        return cast(SingleTypeDef[ValT], self._hint_to_typedef(arg))

    @cached_prop
    def target_type(self) -> type[ValT]:
        """Value type of the property."""
        return cast(
            type[ValT],
            (
                self.value_type
                if isinstance(self.value_type, type)
                else get_origin(self.value_type) or object
            ),
        )

    @cached_prop
    def record_type(self: Data[RecT2 | None, Any, Any, Any, Any, Any]) -> type[RecT2]:
        """Value type of the property."""
        t = extract_nullable_type(self.value_type)
        assert t is not None and issubclass(t, Record)
        return t

    @cached_prop
    def relation_type(self) -> type[LnT]:
        """Link record type."""
        args = self._generic_args
        rec = args[self._typearg_map[LnT]]
        rec_type = self._hint_to_type(rec)

        if not issubclass(rec_type, Record):
            return cast(type[LnT], NoneType)

        return cast(type[LnT], rec_type)

    @cached_prop
    def base_type(self) -> type[BaseT]:
        """Link record type."""
        args = self._generic_args
        base = args[self._typearg_map[BaseT]]
        base_type = self._hint_to_type(base)

        return cast(type[BaseT], base_type)

    @cached_prop
    def fqn(self) -> str:
        """String representation of the relation path."""
        if self._ctx_data is None:
            if issubclass(self.target_type, Record):
                return self.target_type._default_table_name()
            else:
                return self.name

        fqn = f"{self._ctx_data.fqn}.{self.name}"

        if len(self._filters) > 0:
            fqn += f"[{gen_str_hash(self._filters)}]"

        return fqn

    @cached_prop
    def rel(
        self: Data[Any, Any, Any, RecT2, Ctx[RecT3], Any]
    ) -> BackLink[RecT2, Any, Any, RecT3, BaseT]:
        """Reference props of the relation record type."""
        link = (
            self.relation_type._from  # type: ignore
            if issubclass(self.relation_type, BacklinkRecord)
            else [
                r
                for r in self.relation_type._links.values()
                if isinstance(r, Link)
                and issubclass(self.ctx.record_type, r.record_type)
            ][0]
        )
        return cast(
            BackLink[RecT2, Any, Any, RecT3, BaseT],
            BackLink(link=link)._add_ctx(self.ctx),
        )

    @cached_prop
    def rec(self: Data[RecT2 | None, Any, Any, Any, Any, Any]) -> type[RecT2]:
        return cast(
            type[RecT2],
            type(
                self.target_type.__name__ + "_" + token_hex(5),
                (self.target_type,),
                {
                    "_rel": self,
                    "_derivate": True,
                    "_src_mod": getmodule(self.target_type),
                },
            ),
        )

    @property
    def tuple_selection(
        self: Data[tuple, Any, Any, Any, Any, Any]
    ) -> tuple[Data[Any, Any, Any, Any, Any, Any], ...]:
        assert self._tuple_selection is not None
        return self._tuple_selection

    @cached_prop
    def select(
        self,
        *,
        index_only: bool = False,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        select = sqla.select(
            *(col._sql_col.label(col.fqn) for col in self._abs_idx_values),
            *(
                (val._sql_col.label(val.fqn) for val in self._abs_cols)
                if not index_only
                else []
            ),
        ).select_from(self._prop_path[0]._sql_table)

        for join in self._sql_joins:
            select = select.join(*join)

        for filt in self._sql_filters:
            select = select.where(filt)

        return select

    @cached_prop
    def query(
        self,
    ) -> sqla.Subquery:
        """Return select statement for this dataset."""
        return self.select.subquery()

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(
            (
                self._db,
                self._ctx,
                self._name,
                self._typehint,
                self._typevar_map,
                self._filters,
            )
        )

    @overload
    def keys(  # type: ignore
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
    ) -> Sequence[KeyT2]: ...

    @overload
    def keys(
        self: Data[
            Any,
            BaseIdx[Record[KeyT2, *KeyTt2]] | Idx[KeyT2, *KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Sequence[tuple[KeyT2, *KeyTt2]]: ...

    def keys(
        self: Data[Any, Any, Any, Any, Any, Any],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        df = self.to_df(index_only=True)
        if len(self._idx_cols) == 1:
            return [tup[0] for tup in df.iter_rows()]

        return list(df.iter_rows())

    def values(  # noqa: D102
        self: Data[ValT2, Any, Any, Any, Any, DynBackendID],
    ) -> Sequence[ValT2]:
        dfs = self.to_df()
        if isinstance(dfs, pl.DataFrame):
            dfs = (dfs,)

        val_selection = [val._owner_type for val in self.selection]

        valid_caches = {
            rt: self.db._get_valid_cache_set(rt)
            for rt in val_selection
            if isinstance(rt, type)
        }
        instance_maps = {
            rt: self.db._get_instance_map(rt)
            for rt in val_selection
            if isinstance(rt, type)
        }

        vals = []
        for rows in zip(*(df.iter_rows(named=True) for df in dfs)):
            rows = cast(tuple[dict[str, Any], ...], rows)

            val_list = []
            for sel, row in zip(val_selection, rows):
                if isinstance(sel, type):
                    assert issubclass(sel, Record)
                    rec_type = sel

                    new_rec = rec_type(**row)

                    if new_rec._index in valid_caches[rec_type]:
                        rec = instance_maps[rec_type][new_rec._index]
                    else:
                        rec = new_rec
                        rec._db = self.db
                        valid_caches[rec_type].add(rec._index)
                        instance_maps[rec_type][rec._index] = rec

                    val_list.append(rec)
                else:
                    val_list.append(row[sel.name])

            vals.append(tuple(val_list) if len(val_list) > 1 else val_list[0])

        return vals

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
    def to_df(  # pyright: ignore[reportOverlappingOverload]
        self: Data[tuple[*ValTt2], Any, Any, Any, Any, DynBackendID],
        kind: type[DfT],
        index_only: Literal[False] = ...,
    ) -> tuple[DfT, ...]: ...

    @overload
    def to_df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT],
        index_only: bool = ...,
    ) -> DfT: ...

    @overload
    def to_df(  # pyright: ignore[reportOverlappingOverload]
        self: Data[tuple[*ValTt2], Any, Any, Any, Any, DynBackendID],
        kind: None = ...,
        index_only: Literal[False] = ...,
    ) -> tuple[pl.DataFrame, ...]: ...

    @overload
    def to_df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: None = ...,
        index_only: bool = ...,
    ) -> pl.DataFrame: ...

    def to_df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT] | None = None,
        index_only: bool = False,
    ) -> DfT | tuple[DfT, ...]:
        """Download selection."""
        select = type(self).select(self, index_only=index_only)

        idx_cols = [idx.fqn for idx in self._abs_idx_values]

        merged_df = None
        if kind is pd.DataFrame:
            with self.db.engine.connect() as con:
                merged_df = pd.read_sql(select, con)
                merged_df = merged_df.set_index(idx_cols, drop=False)
        else:
            merged_df = pl.read_database(
                str(select.compile(self.db.engine)), self.db.engine
            )

        if index_only:
            return cast(DfT, merged_df)

        name_map = [
            {col.fqn: col.name for col in col_set}
            for col_set in self._abs_value_sets.values()
        ]

        return cast(
            tuple[DfT, ...],
            (merged_df[list(cols.keys())].rename(cols) for cols in name_map),
        )

    @overload
    def extract(
        self: Data[Record | None, Any, Any, Any, Any, BackT2],
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
    ) -> DataBase[CRUD, BackT2]: ...

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
        to_db: DataBase[CRUD, BackT3],
        overlay_type: OverlayType = ...,
    ) -> DataBase[CRUD, BackT3]: ...

    def extract(  # pyright: ignore[reportInconsistentOverload]
        self: Data[Record | None, Any, Any, Any, Any, BackT2],
        aggs: (
            Mapping[
                Table[Record | None, Any, Any, Any, Ctx, Symbolic],
                Agg,
            ]
            | None
        ) = None,
        to_db: DataBase[CRUD, BackT3] | None = None,
        overlay_type: OverlayType = "name_prefix",
    ) -> DataBase[CRUD, BackT2 | BackT3]:
        """Extract a new database instance from the current selection."""
        assert isinstance(self, Table)

        # Get all rec types in the schema.
        rec_types = {self.target_type, *self.record_type._rel_types}

        # Get the entire subdag of this target type.
        all_paths_rels = {
            r
            for rel in self.record_type._links.values()
            for r in rel._add_ctx(self)._get_subdag(rec_types)
        }

        # Extract rel paths, which contain an aggregated rel.
        aggs_per_type: dict[
            type[Record],
            list[
                tuple[
                    Table[Record | None, Any, Any, Any, Any, Symbolic],
                    Agg,
                ]
            ],
        ] = {}
        if aggs is not None:
            for rel, agg in aggs.items():
                for path_rel in all_paths_rels:
                    if path_rel._has_ancestor(rel):
                        aggs_per_type[rel._ctx_type.record_type] = [
                            *aggs_per_type.get(rel._ctx_type.record_type, []),
                            (rel, agg),
                        ]
                        all_paths_rels.remove(path_rel)

        replacements: dict[type[Record], sqla.Select] = {}
        for rec in rec_types:
            # For each table, create a union of all results from the direct routes.
            selects = [
                rel._add_ctx(self).select
                for rel in all_paths_rels
                if issubclass(rec, rel.target_type)
            ]
            replacements[rec] = sqla.union(*selects).select()

        aggregations: dict[type[Record], sqla.Select] = {}
        for rec, rec_aggs in aggs_per_type.items():
            selects = []
            for rel, agg in rec_aggs:
                selects.append(
                    sqla.select(
                        *[
                            (
                                sa._add_ctx(self)._sql_col
                                if isinstance(sa, Value)
                                else sqla_visitors.replacement_traverse(
                                    sa,
                                    {},
                                    replace=lambda element, **kw: (
                                        element._add_ctx(cast(Table, self))._sql_col
                                        if isinstance(element, Value)
                                        else None
                                    ),
                                )
                            ).label(ta.name)
                            for ta, sa in agg.map.items()
                        ]
                    )
                )

            aggregations[rec] = sqla.union(*selects).select()

        # Create a new database overlay for the results.
        overlay_db = copy_and_override(
            DataBase[CRUD, BackT2 | BackT3],
            self.db,
            backend=self.db.backend,
            write_to_overlay=f"temp_{token_hex(10)}",
            overlay_type=overlay_type,
            schema=None,
            types=rec_types,
            _def_types={},
            _metadata=sqla.MetaData(),
            _instance_map={},
        )

        # Overlay the new tables onto the new database.
        for rec in replacements:
            overlay_db[rec] &= replacements[rec]

        for rec, agg_select in aggregations.items():
            overlay_db[rec] &= agg_select

        # Transfer table to the new database.
        if to_db is not None:
            other_db = copy_and_override(
                DataBase[CRUD, BackT3],
                overlay_db,
                backend=to_db.backend,
                url=to_db.url,
                _def_types={},
                _metadata=sqla.MetaData(),
                _instance_map={},
            )

            for rec in set(replacements) | set(aggregations):
                other_db[rec] &= overlay_db[rec].to_df()

            overlay_db = other_db

        return overlay_db

    def isin(  # noqa: D102
        self: Data[ValT2, Any, Any, Any, Any, BaseT2], other: Iterable[ValT2] | slice
    ) -> sqla.ColumnElement[bool]: ...

    # 1. DB-level type selection
    @overload
    def __getitem__(
        self: DataBase[Any, Any],
        key: type[RecT3],
    ) -> Data[RecT3, BaseIdx[RecT3], CrudT, None, None, BaseT]: ...

    # 2. Top-level prop selection
    @overload
    def __getitem__(
        self: Data[RecT2, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any],
        key: Data[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            LnT3,
            Ctx[RecT2],
            Symbolic,
        ],
    ) -> Data[ValT3, Idx[*KeyTt2, *KeyTt3], CrudT3, LnT3, Ctx[RecT2], BaseT]: ...

    # 3. Nested prop selection
    @overload
    def __getitem__(
        self: Data[RecT2, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any],
        key: Data[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            LnT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt2, *tuple[Any, ...], *KeyTt3],
        CrudT3,
        LnT3,
        CtxT3,
        BaseT,
    ]: ...

    # 4. Key selection, scalar index type, symbolic context
    @overload
    def __getitem__(
        self: Data[Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Symbolic],
        key: KeyT2,
    ) -> Data[ValT, IdxT, RU, LnT, CtxT, Symbolic]: ...

    # 5. Key selection, tuple index type, symbolic context
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Symbolic
        ],
        key: tuple[*KeyTt2],
    ) -> Data[ValT, IdxT, RU, LnT, CtxT, Symbolic]: ...

    # 6. Key selection, scalar index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
    ) -> ValT: ...

    # 7. Key selection, tuple index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, DynBackendID
        ],
        key: tuple[*KeyTt2],
    ) -> ValT: ...

    # 8. Key filtering, scalar index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: list[KeyT2],
    ) -> Data[ValT, IdxT, RU, LnT, CtxT, BaseT]: ...

    # 9. Expression / key / slice filtering
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, DynBackendID
        ],
        key: (
            list[tuple[*KeyTt2]] | sqla.ColumnElement[bool] | slice | tuple[slice, ...]
        ),
    ) -> Data[ValT, IdxT, RU, LnT, CtxT, BaseT]: ...

    # Implementation:

    def __getitem__(  # noqa: D105
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
        match key:
            case type():
                assert issubclass(
                    key,
                    self.target_type,
                )
                return copy_and_override(Data[key], self, selection=key)
            case Data():
                return self._suffix(key)
            case list() | slice() | Hashable() | sqla.ColumnElement():
                if not isinstance(key, list | sqla.ColumnElement) and not has_type(
                    key, tuple[slice, ...]
                ):
                    if not isinstance(key, slice):
                        assert (
                            self._single_key is None or key == self._single_key
                        ), "Cannot select multiple single record keys"

                    key = cast(Sequence[slice | list[Hashable] | Hashable], [key])

                key_set = copy_and_override(
                    type(self),
                    self,
                    filters=[*self.filters, key],
                )

                if key_set._single_key is not None and isinstance(
                    self.db.backend, Symbolic
                ):
                    try:
                        return list(iter(key_set))[0]
                    except IndexError as e:
                        raise KeyError(key) from e

                return key_set

    def __iter__(  # noqa: D105
        self: Data[ValT2, Any, Any, Any, Any, DynBackendID],
    ) -> Iterator[ValT2]:
        return iter(self.values())

    @overload
    def __imatmul__(
        self: Data[
            Record[KeyT2] | Record[*KeyTt2],
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            RU,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, BaseT]
            | Input[ValT, KeyT3 | tuple[*KeyTt3], KeyT2 | tuple[*KeyTt2]]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    @overload
    def __imatmul__(
        self: Data[
            ValT2,
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            RU,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT2, Any, Any, Any, Any, BaseT]
            | Input[ValT2, KeyT3 | tuple[*KeyTt3], None]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    def __imatmul__(
        self: Data[Any, Any, RU, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, Any] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]:
        """Aligned assignment."""
        self._mutate(other, mode="update")
        return cast(
            Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT],
            self,
        )

    @overload
    def __iadd__(
        self: Data[
            Record[KeyT2] | Record[*KeyTt2],
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CR,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, BaseT]
            | Input[ValT, KeyT3 | tuple[*KeyTt3], KeyT2 | tuple[*KeyTt2]]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    @overload
    def __iadd__(
        self: Data[
            ValT2,
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CR,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT2, Any, Any, Any, Any, BaseT]
            | Input[ValT2, KeyT3 | tuple[*KeyTt3], None]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    def __iadd__(
        self: Data[Any, Any, CR, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, Any] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]:
        """Aligned assignment."""
        self._mutate(other, mode="insert")
        return cast(
            Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT],
            self,
        )

    @overload
    def __ior__(
        self: Data[
            Record[KeyT2] | Record[*KeyTt2],
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CRU,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, BaseT]
            | Input[ValT, KeyT3 | tuple[*KeyTt3], KeyT2 | tuple[*KeyTt2]]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    @overload
    def __ior__(
        self: Data[
            ValT2,
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CRU,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT2, Any, Any, Any, Any, BaseT]
            | Input[ValT2, KeyT3 | tuple[*KeyTt3], None]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    def __ior__(
        self: Data[Any, Any, CRU, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]:
        """Aligned assignment."""
        self._mutate(other, mode="upsert")
        return cast(
            Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT],
            self,
        )

    @overload
    def __iand__(
        self: Data[
            Record[KeyT2] | Record[*KeyTt2],
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, BaseT]
            | Input[ValT, KeyT3 | tuple[*KeyTt3], KeyT2 | tuple[*KeyTt2]]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    @overload
    def __iand__(
        self: Data[
            ValT2,
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT2, Any, Any, Any, Any, BaseT]
            | Input[ValT2, KeyT3 | tuple[*KeyTt3], None]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    def __iand__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, BaseT] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]:
        """Aligned assignment."""
        self._mutate(other, mode="replace")
        return cast(
            Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT],
            self,
        )

    @overload
    def __isub__(
        self: Data[
            Record[KeyT2] | Record[*KeyTt2],
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT, Any, Any, Any, Any, BaseT]
            | Input[ValT, KeyT3 | tuple[*KeyTt3], KeyT2 | tuple[*KeyTt2]]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    @overload
    def __isub__(
        self: Data[
            ValT2,
            BaseIdx[Record[KeyT3]]
            | Idx[KeyT3]
            | BaseIdx[Record[*KeyTt3]]
            | Idx[*KeyTt3],
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: (
            Data[ValT2, Any, Any, Any, Any, BaseT]
            | Input[ValT2, KeyT3 | tuple[*KeyTt3], None]
        ),
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]: ...

    def __isub__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, LnT, Ctx[CtxT], BaseT]:
        """Aligned assignment."""
        raise NotImplementedError("Subtraction not supported yet.")

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
            type[Record]
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice, ...]
            | Data[bool, Any, Any, Any, Any, Symbolic]
        ),
    ) -> None:
        if not isinstance(key, slice) or key != slice(None):
            del self[key][:]

        tables = {
            self.db[rec]._get_sql_base_table(mode="upsert")
            for rec in self.target_type._record_superclasses
        }

        statements = []

        for table in tables:
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
                                col == self[self._pk_attrs[col.name]]
                                for col in table.primary_key.columns
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
                                type(self).select(self, index_only=True).subquery(),
                                reduce(
                                    sqla.and_,
                                    (
                                        col == self[self._pk_attrs[col.name]]
                                        for col in table.primary_key.columns
                                    ),
                                ),
                            )
                        )
                    )
                )
            else:
                raise NotImplementedError("Deletion not supported for this dialect.")

        # Execute delete statements.
        with self.db.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        if self.db.backend_type == "excel-file":
            self._save_to_excel()

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]  # noqa: D105
        self: Data[Any, IdxT2, Any, Any, Any, BaseT2],
        other: Any | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    def __lt__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    def __lte__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    def __gt__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    def __gte__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    @overload
    def __matmul__(
        self: Data[tuple, Any, Any, Any, Any, Any],
        other: Data[tuple, Any, Any, Any, Any, Any],
    ) -> Data[tuple, IdxT, CrudT, LnT, CtxT, BaseT]: ...

    @overload
    def __matmul__(
        self: Data[ValT2, Any, Any, Any, Any, Any],
        other: Data[tuple[*ValTt3], Any, Any, Any, Any, Any],
    ) -> Data[tuple[ValT2, *ValTt3], IdxT, CrudT, LnT, CtxT, BaseT]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, Any, Any, Any, Any],
        other: Data[ValT3, Any, Any, Any, Any, Any],
    ) -> Data[tuple[*ValTt2, ValT3], IdxT, CrudT, LnT, CtxT, BaseT]: ...

    @overload
    def __matmul__(
        self: Data[ValT2, Any, Any, Any, Any, Any],
        other: Data[ValT3, Any, Any, Any, Any, Any],
    ) -> Data[tuple[ValT2, ValT3], IdxT, CrudT, LnT, CtxT, BaseT]: ...

    def __matmul__(
        self,
        other: Data[ValT2, Any, Any, Any, Any, BaseT],
    ) -> Data[tuple, IdxT, CrudT, LnT, CtxT, BaseT]:
        return copy_and_override(
            Data[tuple, IdxT, CrudT, LnT, CtxT, BaseT],
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

    # TODO: Implement construction of merges via matmul

    def __setitem__(
        self,
        key: Any,
        other: Data[Any, Any, Any, Any, Any, Any],
    ) -> None:
        """Catchall setitem."""
        return

    def __set_name__(self, _, name: str) -> None:  # noqa: D105
        if self._name is None:
            self._name = name
        else:
            assert name == self._name

    @overload
    def __get__(
        self: Data[Any, IdxT, Any, Any, Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[RecT, IdxT, CrudT, LnT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Record[*KeyTt2], Any, Any, None, Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[RecT, Idx[*KeyTt2], CrudT, LnT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, tuple[*KeyTt2]], Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[RecT, Idx[*KeyTt2], CrudT, LnT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, KeyT2], Any, Any],
        instance: None,
        owner: type[RecT4],
    ) -> Data[RecT, Idx[KeyT2], CrudT, LnT, Ctx[RecT4], Symbolic]: ...

    @overload
    def __get__(
        self: Data[Any, NoIdx, Any, Any, Any, Any],
        instance: Record,
        owner: type[Record],
    ) -> ValT: ...

    @overload
    def __get__(
        self: Data[Any, IdxT, Any, Any, Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[RecT, IdxT, CrudT, LnT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Data[Record[*KeyTt2], Any, Any, None, Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[RecT, Idx[*KeyTt2], CrudT, LnT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, tuple[*KeyTt2]], Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[RecT, Idx[*KeyTt2], CrudT, LnT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Data[Any, Any, Any, IndexedRelation[Any, Any, KeyT2], Any, Any],
        instance: RecT4,
        owner: type[RecT4],
    ) -> Data[RecT, Idx[KeyT2], CrudT, LnT, Ctx[RecT4], DynBackendID]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(
        self: Data[Any, Any, Any, Any, Ctx | None, Any],
        instance: object | None,
        owner: type | None,
    ) -> Data[Any, Any, Any, Any, Any, Any] | ValT:
        owner = self._ctx_type.record_type if self._ctx_type is not None else owner

        if owner is not None and issubclass(owner, Record):
            if isinstance(instance, Record):
                if isinstance(self, Value):
                    if (
                        self.pub_status is Public
                        and instance._connected
                        and instance._index
                        not in instance._db._get_valid_cache_set(owner)
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
                        Data[ValT, NoIdx, CrudT, LnT, Ctx, Symbolic],
                        getattr(owner, self.name),
                    )
                    return instance._db[type(instance)][self_ref][instance._index]

            if instance is None:
                return copy_and_override(
                    Data[ValT, NoIdx, CrudT, LnT, Ctx, Symbolic],
                    self,
                    _db=DataBase(backend=Symbolic()),
                    _ctx=Ctx(owner),
                )

        return self

    @overload
    def __set__(
        self,
        instance: Record,
        value: Data[ValT, NoIdx, Any, Any, Any, Any] | ValT | type[Keep],
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
        if value is Keep:
            return

        if isinstance(instance, Record):
            if isinstance(self, Value):
                if self.setter is not None:
                    self.setter(instance, value)
                else:
                    instance.__dict__[self.name] = value

            if not isinstance(self, Value) or self.pub_status is Public:
                owner = type(instance)
                sym_rel: Data[Any, Any, Any, Any, Any, Symbolic] = getattr(
                    owner, self.name
                )
                instance._table[sym_rel]._mutate(value)

            if not isinstance(self, Value):
                instance._update_dict()
        else:
            instance.__dict__[self.name] = value

        return

    @cached_prop
    def _data_type(self) -> type[Data] | type[None]:
        """Resolve the property type reference."""
        hint = self._typehint

        if hint is None:
            return NoneType

        if has_type(hint, SingleTypeDef):
            base = get_origin(hint)
            if base is None or not issubclass(base, Data):
                return NoneType

            return base
        elif isinstance(hint, str):
            return self._map_prop_type_name(hint)
        else:
            return NoneType

    @cached_prop
    def _generic_type(self) -> SingleTypeDef | UnionType:
        """Resolve the generic property type reference."""
        hint = self._typehint or Data
        generic = self._hint_to_typedef(hint)
        assert is_subtype(generic, Data)

        return generic

    @cached_prop
    def _generic_args(self) -> tuple[SingleTypeDef | UnionType | TypeVar, ...]:
        """Resolve the generic property type reference."""
        args = get_args(self._generic_type)
        return tuple(self._hint_to_typedef(hint) for hint in args)

    @cached_prop
    def _ctx_type(self) -> CtxT:
        """Link record type."""
        return (
            cast(CtxT, Ctx(self._ctx.target_type))
            if isinstance(self._ctx, Data)
            else self._ctx if self._ctx else cast(CtxT, None)
        )

    @cached_prop
    def _ctx_data(self) -> Table[Record | None, Any, Any, Any, Any, BaseT] | None:
        """Parent reference of the property."""
        return (
            self._ctx
            if isinstance(self._ctx, Table)
            else Table(_db=self.db, _ctx=self._ctx) if self._ctx is not None else None
        )

    @cached_prop
    def _ctx_module(self) -> ModuleType | None:
        """Get the module of the context."""
        if self._ctx_type is None:
            return None

        assert issubclass(self._ctx_type.record_type, Record)
        return self._ctx_type.record_type._src_mod or getmodule(self._ctx_type)

    @cached_prop
    def _owner_type(self) -> type[Record] | None:
        """Get the original owner type of the property."""
        if self._ctx_type is None:
            return None

        direct_owner = self._ctx_type.record_type
        assert issubclass(direct_owner, Record)
        all_bases = [
            direct_owner,
            *direct_owner._record_superclasses,
        ]

        original_owners = [base for base in all_bases if self.name in base._class_props]
        assert len(original_owners) == 1
        return original_owners[0]

    @cached_prop
    def _prop_path(self) -> PropPath[Record, ValT]:
        if self._ctx_data is None:
            assert issubclass(self.target_type, Record)
            return (cast(Table[Record, Any, Any, Any, None, Any], self),)

        return cast(PropPath[Record, ValT], (*self._ctx_data._prop_path, self))

    @cached_prop
    def _abs_link_path(self) -> list[LinkItem]:
        path: list[LinkItem] = []

        for p in self._prop_path:
            if issubclass(p.relation_type, NoneType):
                path.append(cast(Table[Record, None, Any, Any, Any, Any], p))
            elif issubclass(p.relation_type, Relation):
                p = cast(Table[Record, Relation, Any, Any, Any, Any], p)
                path.extend([p.rel, p.rel.rec._to])  # type: ignore
            else:
                p = cast(Table[Record, BacklinkRecord, Any, Any, Any, Any], p)
                path.append(p.rel)

        return path

    @property
    def _abs_idx_values(self) -> list[Value[Any, Any, Any, Any, BaseT]]:
        indexes: list[Value[Any, Any, Any, Any, BaseT]] = []

        for node in self._abs_link_path:
            if isinstance(node, Table):
                for pk in (
                    [node.record_type._rel_id]  # type: ignore
                    if issubclass(node.record_type, IndexedRelation)
                    else node.record_type._pk_values.values()
                ):
                    if pk not in indexes:
                        indexes.append(
                            copy_and_override(
                                Value[Any, Any, Any, Any, BaseT],
                                pk._add_ctx(node),
                                _db=self.db,
                            )
                        )
        return indexes

    @cached_prop
    def _abs_cols(self) -> set[Value[Any, Any, Any, Any, BaseT]]:
        return {
            copy_and_override(
                Value[Any, Any, Any, Any, BaseT], v, _db=self.db, _ctx=self._ctx
            )
            for v in (
                (v for sel in self.tuple_selection for v in sel._abs_cols)
                if has_type(self, Data[tuple, Any, Any, Any, Ctx, Any])
                else (
                    self.record_type._values.values()
                    if has_type(self, Table[Record | None, Any, Any, Any, Any, Any])
                    else (self,)
                )
            )
        }

    @cached_prop
    def _abs_values_per_base(
        self: Data[Record | None, Any, Any, Any, Any, Any],
    ) -> dict[
        Table[Record | None, Any, Any, Any, Any, Any],
        set[Value[Any, Any, Any, Any, BaseT]],
    ]:
        return {
            Table(_db=self.db, _typehint=Table[base_rec]): {
                v for v in self._abs_cols if v.name in base_rec._values
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
        key_filt = [f for f in self._filters if not isinstance(f, sqla.ColumnElement)]
        key_filt = (
            [
                (idx.isin(val) if isinstance(val, list | slice) else idx == val)
                for f in key_filt
                for idx, val in zip(self._abs_idx_values, f)
            ]
            if len(key_filt) > 0
            else []
        )

        filters = [
            *sql_filt,
            *key_filt,
        ]

        join_set: set[Table[Any, Any, Any, Any, Any, BaseT]] = set()
        replace_func = partial(self._visit_filter_col, join_set=join_set, render=False)
        parsed_filt = [
            sqla_visitors.replacement_traverse(f, {}, replace=replace_func)
            for f in filters
        ]
        merge = list(join_set)

        return parsed_filt, merge

    @cached_prop
    def _abs_joins(self) -> list[Table[Record | None, Any, Any, Any, Any, BaseT]]:
        return [
            copy_and_override(
                Table[Record | None, Any, Any, Any, Any, BaseT],
                v,
                _db=self.db,
                _ctx=self._ctx,
            )
            for v in (
                (v for sel in self.tuple_selection for v in sel._abs_joins)
                if has_type(self, Data[tuple, Any, Any, Any, Ctx, Any])
                else (
                    (self,)
                    if has_type(self, Table[Record | None, Any, Any, Any, Any, Any])
                    else (
                        (self.ctx,)
                        if has_type(self, Data[Record, Any, Any, Any, Ctx, Any])
                        else ()
                    )
                )
            )
        ]

    @cached_prop
    def _total_joins(self) -> list[Table[Record | None, Any, Any, Any, Any, BaseT]]:
        sel = self._abs_joins + self._abs_filters[1]
        return sel if self._ctx_data is None else sel + self._ctx_data._total_joins

    @cached_prop
    def _total_join_dict(self) -> JoinDict:
        tree: JoinDict = {}

        for rel in self._total_joins:
            subtree = tree
            for node in rel._abs_link_path:
                if node not in subtree:
                    subtree[node] = {}
                subtree = subtree[node]

        return tree

    @cached_prop
    def fk_map(self: Data[Record | None, NoIdx, Any, None, Ctx, Any]) -> bidict[
        Value[Any, Any, Any, Any, Symbolic],
        Value[Any, Any, Any, Any, Symbolic],
    ]:
        """Map source foreign keys to target cols."""
        if not isinstance(self, Link) or self.fks is None:
            from_rec: type[Record]
            to_rec: type[Record]
            from_rec, to_rec = (
                (self.ctx.record_type, self.record_type)
                if isinstance(self, Link)
                else (self.record_type, self.ctx.record_type)
            )

            return bidict(
                {
                    Value[Any, Any, Any, Any, Symbolic](
                        _name=f"{self.name}_{pk.name}",
                        _typehint=pk._typehint,
                        init=False,
                        index=self.index if isinstance(self, Link) else False,
                        primary_key=(
                            self.primary_key if isinstance(self, Link) else False
                        ),
                        _ctx=Ctx(from_rec),
                    ): pk
                    for pk in to_rec._pk_values.values()
                }
            )

        match self.fks:
            case dict():
                return bidict({fk: pk for fk, pk in self.fks.items()})
            case Value() | list():
                fks = self.fks if isinstance(self.fks, list) else [self.fks]

                pks = [
                    getattr(self.record_type, name)
                    for name in self._ctx_type.record_type._pk_values
                ]

                return bidict(dict(zip(fks, pks)))

    @cached_prop
    def _sql_joins(
        self,
        _subtree: JoinDict | None = None,
        _parent: Table[Record | None, Any, Any, Any, Any, Any] | None = None,
    ) -> list[SqlJoin]:
        """Extract join operations from the relation tree."""
        joins: list[SqlJoin] = []
        _subtree = _subtree if _subtree is not None else self._total_join_dict
        _parent = _parent if _parent is not None else self._prop_path[0]

        for target, next_subtree in _subtree.items():
            joins.append(
                (
                    target._sql_table,
                    reduce(
                        sqla.and_,
                        (
                            (
                                _parent[fk] == target[pk]
                                for fk, pk in target.fk_map.items()
                            )
                            if isinstance(target, Link)
                            else (
                                target[fk] == _parent[pk]
                                for fk, pk in target.fk_map.items()
                            )
                        ),
                    ),
                )
            )

            joins.extend(type(self)._sql_joins(self, next_subtree, target))

        return joins

    @cached_prop
    def _sql_filters(self) -> list[sqla.ColumnElement[bool]]:
        """Get the SQL filters for this table."""
        if not isinstance(self, Table):
            return []

        return self._abs_filters[0] + (
            self._ctx_data._sql_filters if self._ctx_data is not None else []
        )

    @cached_prop
    def _sql_base_cols(
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

    @cached_prop
    def _sql_base_fks(
        self: Data[Record | None, Any, Any, Any, Any, Any]
    ) -> list[sqla.ForeignKeyConstraint]:
        fks: list[sqla.ForeignKeyConstraint] = []

        for rt in self.record_type._links.values():
            rel_table = rt._get_sql_base_table()
            fks.append(
                sqla.ForeignKeyConstraint(
                    [fk.name for fk in rt.fk_map.keys()],
                    [rel_table.c[pk.name] for pk in rt.fk_map.values()],
                    name=f"{self.record_type._get_table_name(self.db._subs)}_{rt.name}_fk",
                )
            )

        for superclass in self.record_type._record_superclasses:
            base_table = Table(
                _db=self.db, _typehint=Table[superclass]
            )._get_sql_base_table()

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

    @cached_prop
    def _sql_col(self: Data[Any, Any, Any, Any, Ctx, Any]) -> sqla.ColumnElement:
        assert self._ctx_data is not None
        return self._ctx_data._sql_table.c[self.name]

    @cached_prop
    def _sql_table(
        self: Data[Record | None, Any, Any, Any, Any, Any]
    ) -> sqla.FromClause:
        """Recursively join all bases of this record to get the full data."""
        base_table = self._get_sql_base_table("read")

        table = base_table
        cols = {col.name: col for col in base_table.columns}
        for superclass in self.record_type._record_superclasses:
            superclass_table = Table(
                _db=self.db, _typehint=Table[superclass]
            )._sql_table
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

        return (
            sqla.select(*(col.label(col_name) for col_name, col in cols.items()))
            .select_from(table)
            .subquery()
        )

    def _map_prop_type_name(self, name: str) -> type[Data | None]:
        """Map property type name to class."""
        name_map = _prop_type_name_map()
        matches = [name_map[n] for n in name_map if name.startswith(n + "[")]
        return matches[0] if len(matches) == 1 else NoneType

    def _hint_to_typedef(
        self, hint: SingleTypeDef | UnionType | TypeVar | str | ForwardRef
    ) -> SingleTypeDef:
        typedef = hint

        if isinstance(typedef, str):
            typedef = eval(
                typedef,
                {**globals(), **(vars(self._ctx_module) if self._ctx_module else {})},
            )

        if isinstance(typedef, TypeVar):
            typedef = self._typevar_map.get(typedef) or typedef.__bound__ or object

        if isinstance(typedef, ForwardRef):
            typedef = typedef._evaluate(
                {**globals(), **(vars(self._ctx_module) if self._ctx_module else {})},
                {},
                recursive_guard=frozenset(),
            )

        if isinstance(typedef, UnionType):
            union_types: set[type] = {
                get_origin(union_arg) or union_arg for union_arg in get_args(typedef)
            }
            typedef = get_lowest_common_base(union_types)

        return cast(SingleTypeDef, typedef)

    def _hint_to_type(self, hint: SingleTypeDef | UnionType | TypeVar) -> type:
        if isinstance(hint, type):
            return hint

        typedef = (
            self._hint_to_typedef(hint)
            if isinstance(hint, UnionType | TypeVar)
            else hint
        )
        orig = get_origin(typedef)

        if orig is None or orig is Literal:
            return object

        assert isinstance(orig, type)
        return new_class(
            orig.__name__ + "_" + token_hex(5),
            (hint,),
            None,
            lambda ns: ns.update({"_src_mod": self._ctx}),
        )

    def _has_ancestor(
        self, other: Table[Record | None, Any, Any, Any, Any, BaseT]
    ) -> bool:
        """Check if other is ancestor of this data."""
        return other is self._ctx or (
            self._ctx_data is not None and self._ctx_data._has_ancestor(other)
        )

    def _add_ctx(
        self,
        left: Data[Record | None, Any, Any, Any, Any, Any],
    ) -> Self:
        """Prefix this reltable with a reltable or record type."""
        return cast(
            Self,
            reduce(
                lambda x, y: copy_and_override(
                    type(y), y, _ctx=copy_and_override(Table[Record | None, Any, Any, Any, Any, Any], x, _db=y.db)  # type: ignore
                ),
                self._prop_path,
                left,
            ),
        )

    def _get_subdag(
        self: Data[Record | None, Any, Any, Any, Any, Any],
        target_records: set[type[Record]] | None = None,
        _traversed: set[Table[Any, Any, Any, Any, Any, Symbolic]] | None = None,
    ) -> set[Table[Any, Any, Any, Any, Ctx, Symbolic]]:
        """Find all paths to the target record types."""
        target_records = target_records or set()
        _traversed = _traversed or set()

        # Get relations of the target type as next relations
        next_rels = set(
            tab
            for tab in self.record_type._tables.values()
            if tab.record_type in target_records
        )

        for backlink_record in target_records:
            next_rels |= backlink_record._find_backlinks(self.record_type)

        # Filter out already traversed relations
        next_rels = {rel for rel in next_rels if rel not in _traversed}

        # Add next relations to traversed set
        _traversed |= next_rels

        next_rels = {rel._add_ctx(self) for rel in next_rels}

        # Return next relations + recurse
        return next_rels | {
            rel
            for next_rel in next_rels
            for rel in next_rel._get_subdag(target_records, _traversed)
        }

    def _visit_filter_col(
        self,
        element: sqla_visitors.ExternallyTraversible,
        join_set: set[Table[Any, Any, Any, Any, Any, BaseT]] = set(),
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if isinstance(element, Value) and self._ctx_data is not None:
            prefixed = element._add_ctx(self._ctx_data)
            join_set.add(prefixed.ctx)
            return prefixed._sql_col

        return None

    def _get_sql_base_table(
        self: Data[Record | None, Any, Any, Any, Any, Any],
        mode: Literal["read", "replace", "upsert"] = "read",
        without_auto_fks: bool = False,
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        orig_table: sqla.Table | None = None

        if (
            mode != "read"
            and self.db.write_to_overlay is not None
            and self.record_type not in self.db._subs
        ):
            orig_table = self._get_sql_base_table("read")

            # Create an empty overlay table for the record type
            self.db._subs[self.record_type] = sqla.table(
                (
                    (
                        self.db.write_to_overlay
                        + "_"
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

        cols = self._sql_base_cols
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

        fks = self._sql_base_fks
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
            *cols.values(),
            *fks,
            schema=(sub.schema if sub is not None else None),
            extend_existing=True,
        )

        self.db._create_sqla_table(table)

        if orig_table is not None and mode == "upsert":
            with self.db.engine.connect() as conn:
                conn.execute(
                    sqla.insert(table).from_select(
                        orig_table.columns.keys(), orig_table.select()
                    )
                )

        return table

    def _df_to_table(
        self,
        df: pd.DataFrame | pl.DataFrame,
    ) -> sqla.Table:
        if isinstance(df, pd.DataFrame) and any(
            name is None for name in df.index.names
        ):
            idx_names = [col.fqn for col in self._abs_idx_values]
            df.index.set_names(idx_names, inplace=True)

        value_table = self._gen_upload_table()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DuckDBEngineWarning)
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

    @overload
    def _values_to_df(
        self: Data[
            ValT2, Idx[KeyT2] | BaseIdx[Record[KeyT2]], Any, Any, Any, DynBackendID
        ],
        values: Mapping[KeyT2, ValT2],
    ) -> pl.DataFrame: ...

    @overload
    def _values_to_df(
        self: Data[
            ValT2, Idx[*KeyTt2] | BaseIdx[Record[*KeyTt2]], Any, Any, Any, DynBackendID
        ],
        values: Mapping[tuple[*KeyTt2], ValT2],
    ) -> pl.DataFrame: ...

    def _values_to_df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        values: Mapping[Any, Any],
    ) -> pl.DataFrame:
        col_data = [
            tuple(
                *(idx if len(self._abs_idx_values) > 1 else tuple(idx)),
                *(
                    v
                    for val in (vals if len(self._selection) > 1 else [vals])
                    for v in (
                        val._to_dict().values() if isinstance(val, Record) else [val]
                    )
                ),
            )
            for idx, vals in values.items()
        ]

        col_map = {
            **{col.fqn: col for col in self._abs_idx_values},
            **{
                col.fqn: col for v_set in self._abs_value_sets.values() for col in v_set
            },
        }
        return pl.DataFrame(col_data, schema=_get_pl_schema(col_map))

    def _gen_upload_table(
        self,
    ) -> sqla.Table:
        type_map = {
            t: st
            for tab in self._selection
            if isinstance(tab, Table)
            for t, st in tab.record_type._type_map.items()
        }

        metadata = sqla.MetaData()
        registry = orm.registry(
            metadata=self.db._metadata,
            type_annotation_map=type_map,
        )

        cols = [
            sqla.Column(
                col.fqn,
                registry._resolve_type(
                    col.value_type  # pyright: ignore[reportArgumentType]
                ),
                primary_key=col.primary_key,
                autoincrement=False,
                index=col.index,
                nullable=has_type(None, col.value_type),
            )
            for v_set in self._abs_value_sets.values()
            for col in v_set
        ]

        table_name = f"{self.fqn}[{token_hex(5)}]"
        table = sqla.Table(
            table_name,
            metadata,
            *cols,
        )

        return table

    def _mutate(
        self: Data[ValT2, Any, CR | RU, Any, Any, DynBackendID],
        value: Data[ValT2, Any, Any, Any, Any, Any] | Input[ValT2, Hashable, Hashable],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        record_ids: dict[Hashable, Hashable] | None = None
        valid_caches = self.db._get_valid_cache_set(self.target_type)

        match value:
            case sqla.Select():
                self._mutate_from_sql(value.subquery(), mode)
                valid_caches.clear()
            case Data():
                if hash(value.db) != hash(self.db):
                    remote_db = (
                        value if isinstance(value, DataBase) else value.extract()
                    )
                    for s in remote_db._def_types:
                        if remote_db.db_id == self.db.db_id:
                            self.db[s]._mutate_from_sql(
                                remote_db[s]._sql_query, "upsert"
                            )
                        else:
                            value_table = self._df_to_table(
                                remote_db[s].to_df(),
                            )
                            self.db[s]._mutate_from_sql(
                                value_table,
                                "upsert",
                            )
                            value_table.drop(self.db.engine)

                self._mutate_from_sql(value.select.subquery(), mode)
                valid_caches -= set(value.keys())
            case pd.DataFrame() | pl.DataFrame():
                value_table = self._df_to_table(value)
                self._mutate_from_sql(
                    value_table,
                    mode,
                )
                value_table.drop(self.db.engine)

                base_idx_cols = list(self.target_type._pk_attrs.keys())
                base_idx_keys = set(
                    value[base_idx_cols].iter_rows()
                    if isinstance(value, pl.DataFrame)
                    else value[base_idx_cols].itertuples(index=False)
                )

                valid_caches -= base_idx_keys
            case Record():
                cast(
                    Data[Record, Any, CRUD, Any, DynBackendID],
                    self,
                )._mutate_from_records({value._index: value}, mode)
                valid_caches -= {value._index}
            case Iterable():
                if self._attr is not None:
                    assert isinstance(
                        value, Mapping
                    ), "Inserting via values requires a mapping."
                    cast(
                        Data[Any, Any, CRUD, Any, DynBackendID],
                        self,
                    )._mutate_from_values(value, mode)
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
                        Data[Record, Any, CRUD, Any, Any, DynBackendID],
                        self,
                    )._mutate_from_records(
                        records,
                        mode,
                    )
                    valid_caches -= {rec._index for rec in records.values()}

                    if len(record_ids) > 0:
                        cast(
                            Data[Record, Any, CRUD, Any, Any, DynBackendID],
                            self,
                        )._mutate_from_rec_ids(record_ids, mode)
                        valid_caches -= set(record_ids.values())
            case Hashable():
                if self._attr is not None:
                    cast(
                        Data[Any, Any, CRUD, Any, Any, DynBackendID],
                        self,
                    )._mutate_from_values({None: value}, mode)
                    valid_caches -= {self.keys()}
                else:
                    assert (
                        self._ref is not None
                    ), "Inserting via ids requires a relation."
                    cast(
                        Data[Record, Any, CRUD, Any, Any, DynBackendID],
                        self,
                    )._mutate_from_rec_ids({value: value}, mode)
                    valid_caches -= {value}

        return

    def _mutate_from_values(
        self: Data[ValT2, Any, CRUD, Any, Any, DynBackendID],
        values: Mapping[Any, ValT2],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        df = self._values_to_df(values)
        value_table = self._df_to_table(df)
        self._mutate_from_sql(
            value_table,
            mode,
        )
        value_table.drop(self.db.engine)
        return

    def _mutate_from_sql(
        self,
        value_table: sqla.FromClause,
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        table_values = {
            tab._get_sql_base_table(
                "upsert" if mode in ("update", "insert", "upsert") else "replace"
            ): vals
            for tab, vals in self._abs_values_per_base.items()
        }

        statements: list[sqla.Executable] = []

        if mode == "replace":
            # Delete all records in the current selection.
            for table in table_values:
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
                                    col == self.query.corresponding_column(col)
                                    for col in table.primary_key.columns
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
                                            col == self.query.corresponding_column(col)
                                            for col in table.primary_key.columns
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
                # Do an insert-from-select operation, which updates on conflict:
                if mode == "upsert":
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
                            [col.name for col in vals],
                            sqla.select(
                                *(
                                    value_table.c[col.fqn].label(col.name)
                                    for col in vals
                                )
                            ).select_from(value_table),
                        )
                        statement = statement.on_conflict_do_update(
                            index_elements=[
                                col.name for col in table.primary_key.columns
                            ],
                            set_={
                                c.name: statement.excluded[c.name]
                                for c in vals
                                if c.name not in table.primary_key.columns
                            },
                        )
                    elif self.db.engine.dialect.name in (
                        "mysql",
                        "mariadb",
                    ):
                        # For MySQL / MariaDB, use: https://docs.sqlalchemy.org/en/20/dialects/mysql.html#insert-on-duplicate-key-update-upsert
                        statement = (
                            mysql.Insert(table)
                            .from_select(
                                [c.name for c in vals],
                                sqla.select(
                                    *(
                                        value_table.c[col.fqn].label(col.name)
                                        for col in vals
                                    )
                                ).select_from(value_table),
                            )
                            .prefix_with("INSERT INTO")
                        )
                        statement = statement.on_duplicate_key_update(
                            **statement.inserted
                        )
                    else:
                        # For others, use CTE: https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.cte
                        raise NotImplementedError(
                            "Upsert not supported for this database dialect."
                        )
                else:
                    statement = table.insert().from_select(
                        [c.name for c in vals],
                        sqla.select(
                            *(value_table.c[col.fqn].label(col.name) for col in vals)
                        ).select_from(value_table),
                    )

                statements.append(statement)
        else:
            # Construct the update statements.

            # Derive current select statement and join with value table, if exists.
            value_join_on = reduce(
                sqla.and_,
                (
                    self.query.corresponding_column(idx_col) == idx_col
                    for idx_col in value_table.primary_key
                ),
            )
            select = self.query.join(
                value_table,
                value_join_on,
            )

            for table, vals in table_values.items():
                col_names = {c.fqn: c.name for c in vals}
                values = {
                    col_names[col_fqn]: col
                    for col_fqn, col in value_table.columns.items()
                    if col_fqn in col_names
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
                                    col == select.corresponding_column(col)
                                    for col in table.primary_key.columns
                                ),
                            )
                        )
                    )
                else:
                    # Correlated update.
                    raise NotImplementedError("Correlated update not supported yet.")

        # Execute delete / insert / update statements.
        with self.db.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        # Update relations with parent records.
        if isinstance(self._ref, Ref):
            # Case: parent links directly to child (n -> 1)
            idx_cols = [value_table.c[col.name] for col in self._idx_cols]
            fk_cols = [
                value_table.c[pk.name].label(fk.name) for fk, pk in self._fk_map.items()
            ]
            self.ctx._mutate_from_sql(
                sqla.select(*idx_cols, *fk_cols).select_from(value_table).subquery(),
                "update",
            )
        elif issubclass(self.relation_type, Record):
            # Case: parent and child are linked via assoc table (n <--> m)
            # Update link table with new child indexes.
            idx_cols = [value_table.c[col.fqn] for col in self._idx_cols]
            fk_cols = [
                value_table.c[pk.name].label(fk.name)
                for fk, pk in self._target_path[1]._fk_map.items()
            ]
            link_cols = [
                value_table.c[col.fqn].label(col.name)
                for col in self._idx_cols
                if col.name in self.relation_type._col_values
            ]
            cast(
                Data[Record, Any, CRUD, Any, DynBackendID],
                self,
            ).rels._mutate_from_sql(
                sqla.select(*idx_cols, *fk_cols, *link_cols)
                .select_from(value_table)
                .subquery(),
                mode,
            )
        else:
            # The (1 <- n) case is already covered by updating
            # the child record directly, which includes all its foreign keys.
            pass

        if self.db.backend_type == "excel-file":
            self._save_to_excel()

        return

    def _mutate_from_values(
        self: Table[Any, Relation | None, CRUD, Any, Ctx | None, DynBackendID],
        values: Mapping[Hashable, Record],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        db_grouped = {
            db: dict(recs)
            for db, recs in groupby(
                sorted(
                    values.items(), key=lambda x: x[1]._connected and x[1]._db.db_id
                ),
                lambda x: None if not x[1]._connected else x[1]._db,
            )
        }

        unconnected_records = db_grouped.get(None, {})
        local_records = db_grouped.get(self.db, {})

        remote_records = {
            db: recs
            for db, recs in db_grouped.items()
            if db is not None and hash(db) != hash(self.db)
        }

        if unconnected_records:
            df_data = self._values_to_df(unconnected_records)
            value_table = self._df_to_table(df_data)
            self._mutate_from_sql(
                value_table,
                mode,
            )
            value_table.drop(self.db.engine)

        if local_records and self._rel_data is not None:
            # Only update relations for records already existing in this db.
            rel_rec_type = self._rel_data.record_type
            assert self._ctx_type is not None
            from_type = self._ctx_type.record_type
            rel_recs: dict[Hashable, Record] = {}

            for idx, rec in local_records.items():
                rel_rec = rel_rec_type(_from=from_type, _to=rec)
            self._rel_data._mutate_from_values({}, mode)

        for db, recs in remote_records.items():
            rec_ids = [rec._index for rec in recs.values()]
            remote_set = db[self.target_type][rec_ids]

            remote_db = (
                db if all(rec._root for rec in recs.values()) else remote_set.extract()
            )
            for s in remote_db._def_types:
                if remote_db.db_id == self.db.db_id:
                    self.db[s]._mutate_from_sql(remote_db[s].query, "upsert")
                else:
                    value_table = self._df_to_table(
                        remote_db[s].to_df(),
                    )
                    self.db[s]._mutate_from_sql(
                        value_table,
                        "upsert",
                    )
                    value_table.drop(self.db.engine)

            self._mutate_from_sql(remote_set.select().subquery(), mode)

        return


@dataclass(kw_only=True, eq=False)
class Value(
    Data[ValT, NoIdx, RwT, None, Ctx[ParT], BaseT],
    Generic[ValT, RwT, PubT, ParT, BaseT],
):
    """Record attribute."""

    _typearg_map: ClassVar[dict[TypeVar, int]] = {
        ValT: 0,
        CrudT: 1,
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


@dataclass(kw_only=True, eq=False)
class Table(
    Data[RefT, IdxT, CrudT, LnT, CtxT, BaseT],
    Generic[RefT, LnT, CrudT, IdxT, CtxT, BaseT],
):
    _typearg_map: ClassVar[dict[TypeVar, int]] = {
        ValT: 0,
        LnT: 1,
        CrudT: 2,
        BaseT: 5,
    }

    default: bool = False


@dataclass(kw_only=True, eq=False)
class Link(
    Table[RefT, None, CrudT, NoIdx, Ctx[ParT], BaseT],
    Generic[RefT, CrudT, ParT, BaseT],
):
    _typearg_map: ClassVar[dict[TypeVar, int]] = {
        ValT: 0,
        CrudT: 1,
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
    Table[RecT, None, CrudT, IdxT, Ctx[ParT], BaseT],
    Generic[RecT, IdxT, CrudT, ParT, BaseT],
):
    """Backlink record set."""

    link: Data[ParT, NoIdx, Any, None, Ctx[RecT], Symbolic]


@dataclass(eq=False)
class Array(
    Data[ValT, Idx[KeyT], CrudT, "ArrayRecord[ValT, KeyT, OwnT]", Ctx[OwnT], BaseT],
    Generic[ValT, KeyT, CrudT, OwnT, BaseT],
):
    """Record attribute set."""

    _typearg_map: ClassVar[dict[TypeVar, int]] = {
        ValT: 0,
        KeyT: 1,
        CrudT: 2,
        BaseT: 4,
    }

    @cached_prop
    def _key_type(self) -> type[KeyT]:
        args = self._generic_args
        if len(args) == 0:
            return cast(type[KeyT], object)

        return cast(type[KeyT], args[self._typearg_map[KeyT]])

    @cached_prop
    def relation_type(self) -> type[ArrayRecord[ValT, KeyT, OwnT]]:
        """Return the dynamic relation record type."""
        return dynamic_record_type(
            ArrayRecord[self.target_type, self._key_type, self._ctx_type.record_type],
            f"{self._ctx_type.record_type._default_table_name()}_{self.name}",
        )


@dataclass
class DataBase(Data[Any, NoIdx, CrudT, None, None, BaseT]):
    """Database."""

    backend: BaseT = None  # pyright: ignore[reportAssignmentType]
    """Unique name to identify this database's backend by."""

    url: sqla.URL | CloudPath | HttpFile | Path | None = None
    """Connection URL or path."""

    types: (
        Mapping[
            type[Record],
            Literal[True] | Require | str | sqla.TableClause,
        ]
        | set[type[Record]]
        | None
    ) = None
    schema: (
        type[Schema]
        | Mapping[
            type[Schema],
            Literal[True] | Require | str,
        ]
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

    def __post_init__(self):  # noqa: D105
        self.db = self

        if self.types is not None:
            types = self.types
            if isinstance(types, set):
                types = cast(
                    dict[type[Record], Literal[True]], {rec: True for rec in self.types}
                )

            self._subs = {
                **self._subs,
                **{
                    rec: (sub if isinstance(sub, sqla.TableClause) else sqla.table(sub))
                    for rec, sub in types.items()
                    if not isinstance(sub, Require) and sub is not True
                },
            }
            self._def_types = {**self._def_types, **types}

        if self.schema is not None:
            if isinstance(self.schema, Mapping):
                self._subs = {
                    **self._subs,
                    **{
                        rec: sqla.table(rec._default_table_name(), schema=schema_name)
                        for schema, schema_name in self.schema.items()
                        for rec in schema._record_types
                    },
                }
                schemas = self.schema
            else:
                schemas = {self.schema: True}

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
                        for rec in schema._record_types
                    },
                },
            )

        if self.validate_on_init:
            self.validate()

        if self.write_to_overlay is not None and self.overlay_type == "db_schema":
            self._ensure_sqla_schema_exists(self.write_to_overlay)

    @cached_prop
    def db_id(self) -> str:
        """Return the unique database ID."""
        if isinstance(self.backend, Symbolic):
            return "symbolic"
        return self.backend or gen_str_hash(self.url) or token_hex(5)

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
        # For Excel-backends, use duckdb in-memory engine
        return (
            sqla.create_engine(
                self.url if isinstance(self.url, sqla.URL) else str(self.url)
            )
            if (
                self.backend_type == "sql-connection"
                or self.backend_type == "sqlite-file"
            )
            else (sqla.create_engine(f"duckdb:///:memory:{self.db_id}"))
        )

    def describe(self) -> dict[str, str | dict[str, str] | None]:
        """Return a description of this database."""
        schema_desc = {}
        if isinstance(self.schema, type):
            schema_ref = PyObjectRef.reference(self.schema)

            schema_desc = {
                "schema": {
                    "repo": schema_ref.repo,
                    "package": schema_ref.package,
                    "class": f"{schema_ref.module}.{schema_ref.object}",
                }
            }

            if schema_ref.object_version is not None:
                schema_desc["schema"]["version"] = schema_ref.object_version
            elif schema_ref.package_version is not None:
                schema_desc["schema"]["version"] = schema_ref.package_version

            if schema_ref.repo_revision is not None:
                schema_desc["schema"]["revision"] = schema_ref.repo_revision

            if schema_ref.docs_url is not None:
                schema_desc["schema"]["docs"] = schema_ref.docs_url

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
        types: dict[type[Record], Any] = {}

        if self.types is not None:
            types |= {
                rec: (isinstance(req, Require) and req.present) or req is True
                for rec, req in self._def_types.items()
            }

        if isinstance(self.schema, Mapping):
            types |= {
                rec: isinstance(req, Require) and req.present
                for schema, req in self.schema.items()
                for rec in schema._record_types
            }

        tables = {
            self[b_rec]._get_sql_base_table(): required
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
        self,
        data: pd.DataFrame | pl.DataFrame | sqla.Select,
        fks: (
            Mapping[
                str,
                Data[Value, BaseIdx, Any, Any, Any, Symbolic],
            ]
            | None
        ) = None,
    ) -> Data[DynRecord, BaseIdx, R, Any, BaseT]:
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, pd.DataFrame | pl.DataFrame)
            else f"temp_{token_hex(5)}"
        )

        rec = dynamic_record_type(
            name,
            props=props_from_data(
                data,
                ({name: col for name, col in fks.items()} if fks is not None else None),
            ),
        )
        ds = Data(db=self, selection=rec)

        ds &= data

        return ds

    def execute[
        *T
    ](
        self,
        stmt: sqla.Select[tuple[*T]] | sqla.Insert | sqla.Update | sqla.Delete,
    ) -> sqla.Result[tuple[*T]]:
        """Execute a SQL statement in this database's context."""
        stmt = self._parse_expr(stmt)
        with self.engine.begin() as conn:
            return conn.execute(self._parse_expr(stmt))

    def to_graph(
        self: DataBase[R, Any], nodes: Sequence[type[Record]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        node_tables = [self[n] for n in nodes]

        # Concat all node tables into one.
        node_dfs = [
            n.to_df(kind=pd.DataFrame)
            .reset_index()
            .assign(table=n.target_type._default_table_name())
            for n in node_tables
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )

        directed_edges = reduce(
            set.union, (set((n, r) for r in n._refs.values()) for n in nodes)
        )

        undirected_edges: dict[type[Record], set[tuple[Ref, ...]]] = {
            t: set() for t in nodes
        }
        for n in nodes:
            for at in self._assoc_types:
                if len(at._refs) == 2:
                    left, right = (r for r in at._refs.values())
                    assert left is not None and right is not None
                    if left.target_type == n:
                        undirected_edges[n].add((left, right))
                    elif right.target_type == n:
                        undirected_edges[n].add((right, left))

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                *[
                    node_df.loc[node_df["table"] == str(parent._default_table_name())]
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(link.target_type._default_table_name())
                        ],
                        left_on=[c.name for c in link.fk_map.keys()],
                        right_on=[c.prop.name for c in link.fk_map.values()],
                    )
                    .rename(columns={"node_id": "target"})[["source", "target"]]
                    .assign(
                        ltr=",".join(c.name for c in link.fk_map.keys()),
                        rtl=None,
                    )
                    for parent, link in directed_edges
                ],
                *[
                    self[assoc_table]
                    .to_df(kind=pd.DataFrame)
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(left_rel.value_origin_type._default_table_name())
                        ].dropna(axis="columns", how="all"),
                        left_on=[c.name for c in left_rel.fk_map.keys()],
                        right_on=[c.name for c in left_rel.fk_map.values()],
                        how="inner",
                    )
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(left_rel.value_origin_type._default_table_name())
                        ].dropna(axis="columns", how="all"),
                        left_on=[c.name for c in right_rel.fk_map.keys()],
                        right_on=[c.name for c in right_rel.fk_map.values()],
                        how="inner",
                    )
                    .rename(columns={"node_id": "target"})[
                        list(
                            {
                                "source",
                                "target",
                                *(a for a in self[assoc_table]._col_attrs),
                            }
                        )
                    ]
                    .assign(
                        ltr=",".join(c.name for c in right_rel.fk_map.keys()),
                        rtl=",".join(c.name for c in left_rel.fk_map.keys()),
                    )
                    for assoc_table, rels in undirected_edges.items()
                    for left_rel, right_rel in rels
                ],
            ],
            ignore_index=True,
        )

        return node_df, edge_df

    def __hash__(self) -> int:
        """Hash the DB."""
        return gen_int_hash(
            (
                self.name,
                self.url,
                self._subs,
            )
        )

    @cached_prop
    def _schema_types(
        self,
    ) -> Mapping[type[Record], Literal[True] | Require | str | sqla.TableClause]:
        """Set of all schema types in this DB."""
        types = dict(self._def_types).copy()

        for rec in types:
            types = {
                **{r: True for r in rec._rel_types},
                **types,
            }

        return types

    @cached_prop
    def _assoc_types(self) -> set[type[Record]]:
        """Set of all association tables in this DB."""
        assoc_types = set()
        for rec in self._def_types:
            assoc_table = self[rec]  # move to DataSet
            pks = set([col.name for col in assoc_table._pk_attrs.values()])
            fks = set(
                [col.name for rel in rec._links.values() for col in rel.fk_map.keys()]
            )
            if pks == fks:
                assoc_types.add(rec)

        return assoc_types

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
        if self.db.remove_cross_fks:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            _remove_cross_fk(sqla_table)

        sqla_table.create(self.db.engine, checkfirst=True)

    def _ensure_sqla_schema_exists(self, schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_schema(schema_name):
            with self.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name

    def _load_from_excel(self) -> None:
        """Load all tables from Excel."""
        assert isinstance(self.db.url, Path | CloudPath | HttpFile)
        path = self.db.url.get() if isinstance(self.db.url, HttpFile) else self.db.url

        recs: list[type[Record]] = [
            self.target_type,
            *self.target_type._record_superclasses,
        ]
        with open(path, "rb") as file:
            for rec in recs:
                table = self.db[rec]._get_sql_base_table("replace")

                with self.db.engine.connect() as conn:
                    conn.execute(table.delete())

                pl.read_excel(
                    file, sheet_name=rec._get_table_name(self.db._subs)
                ).write_database(
                    str(table),
                    str(self.db.engine.url),
                    if_table_exists="append",
                )

    def _save_to_excel(self) -> None:
        """Save all (or selected) tables to Excel."""
        assert isinstance(self.db.url, Path | CloudPath | HttpFile)
        file = self.db.url.get() if isinstance(self.db.url, HttpFile) else self.db.url

        recs: list[type[Record]] = [
            self.target_type,
            *self.target_type._record_superclasses,
        ]
        with ExcelWorkbook(file) as wb:
            for rec in recs:
                pl.read_database(
                    str(self.db[rec]._get_sql_base_table().select()),
                    self.db.engine,
                ).write_excel(wb, worksheet=rec._get_table_name(self.db._subs))

        if isinstance(self.db.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.db.url.set(file)


class RecordMeta(type):
    """Metaclass for record types."""

    _record_superclasses: list[type[Record]]
    _class_props: dict[str, Data[Any, Any, Any, Any, Ctx, Symbolic]]

    _is_root_class: bool = False
    _template: bool = False
    _src_mod: ModuleType | None = None
    _derivate: bool = False

    def __init__(cls, name, bases, dct):
        """Initialize a new record type."""
        super().__init__(name, bases, dct)

        if "_src_mod" not in cls.__dict__:
            cls._src_mod = getmodule(cls if not cls._derivate else bases[0])

        props = {
            name: Data[Any, Any, Any, Any, Ctx, Symbolic](
                _typehint=hint,
                _db=DataBase(backend=Symbolic()),
                _ctx=Ctx(cls),
            )
            for name, hint in get_annotations(cls).items()
        }
        props = {
            name: prop
            for name, prop in props.items()
            if is_subtype(prop._data_type, Data)
        }

        for prop_name, prop in props.items():
            prop_type = cast(
                type[Data[Any, Any, Any, Any, Ctx[Record], Symbolic]], prop._data_type
            )

            if prop_name in cls.__dict__:
                prop = copy_and_override(
                    prop_type,
                    cls.__dict__[prop_name],
                    _name=prop._name,
                    _typehint=prop._typehint,
                    _ctx=Ctx(cls),
                )
            else:
                prop = prop_type(
                    _name=prop._name,
                    _typehint=prop._typehint,
                    _ctx=Ctx(cls),
                )

            setattr(cls, prop_name, prop)
            props[prop_name] = prop

        cls._record_superclasses = []
        super_types: Iterable[type | GenericProtocol] = (
            cls.__dict__["__orig_bases__"]
            if "__orig_bases__" in cls.__dict__
            else cls.__bases__
        )
        for c in super_types:
            orig = get_origin(c) if not isinstance(c, type) else c
            if orig is not c and hasattr(orig, "__parameters__"):
                typevar_map = dict(zip(getattr(orig, "__parameters__"), get_args(c)))
            else:
                typevar_map = {}

            if not isinstance(orig, RecordMeta) or orig._is_root_class:
                continue

            if orig._template or cls._derivate:
                for prop_name, prop in orig._class_props.items():
                    if prop_name in cls.__dict__:
                        prop = copy_and_override(
                            type(props[prop_name]), (prop, props[prop_name])
                        )
                    else:
                        prop = copy_and_override(
                            type(prop),
                            props.get(prop_name, prop),
                            _typevar_map=typevar_map,
                            _ctx=Ctx(cls),
                        )

                    setattr(cls, prop_name, prop)
                    props[prop_name] = prop
            else:
                assert orig is c  # Must be concrete class, not a generic
                cls._record_superclasses.append(orig)

        cls._class_props = props

    @property
    def _props(cls) -> dict[str, Data[Any, Any, Any, Any, Ctx, Symbolic]]:
        """The statically defined properties of this record type."""
        return reduce(
            lambda x, y: {**x, **y},
            (c._props for c in cls._record_superclasses),
            cls._class_props,
        )

    @property
    def _class_links(
        cls,
    ) -> dict[str, Link[Any, Any, Any, Symbolic]]:
        """The relations of this record type without superclasses."""
        return {
            name: ref for name, ref in cls._class_props.items() if isinstance(ref, Link)
        }

    @property
    def _class_fk_values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        """The foreign key columns of this record type without superclasses."""
        return {
            a.name: a
            for ref in cls._class_links.values()
            if isinstance(ref, Link)
            for a in ref.fk_map.keys()
        }

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
                for k, a in cls._class_props.items()
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
    def _data_values(cls) -> dict[str, Value[Any, Any, Any, Any, Symbolic]]:
        return {k: c for k, c in cls._col_values.items() if k not in cls._fk_values}

    @property
    def _tables(cls) -> dict[str, Table[Any, Any, Any, Any, Ctx, Symbolic]]:
        return {k: r for k, r in cls._props.items() if isinstance(r, Table)}

    @property
    def _links(cls) -> dict[str, Link[Any, Any, Any, Symbolic]]:
        return {k: r for k, r in cls._tables.items() if isinstance(r, Link)}

    @property
    def _arrays(cls) -> dict[str, Array[Any, Any, Any, Ctx, Symbolic]]:
        return {k: c for k, c in cls._props.items() if isinstance(c, Array)}

    @property
    def _rel_types(cls) -> set[type[Record]]:
        return {rel_set.target_type for rel_set in cls._tables.values()}


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
        UUID: UUIDType(binary=False),
    }

    _template: ClassVar[bool]
    _derivate: ClassVar[bool] = False

    _is_root_class: ClassVar[bool] = True
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)
        cls._is_root_class = False

        cls.__dataclass_fields__ = {
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

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        if cls._table_name is not None:
            return cls._table_name

        fqn_parts = (cls.__module__ + "." + cls.__name__).split(".")

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
            BackLink(link=cast(Link[Self, Any, RecT2, Symbolic], ln), _ctx=Ctx(cls))
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

    _db: Value[DataBase[CRUD, DynBackendID], CRUD, Private] = Value(
        pub_status=Private,
        default_factory=lambda: DataBase[CRUD, DynBackendID](),
    )
    _connected: Value[bool, CRUD, Private] = Value(pub_status=Private, default=False)
    _root: Value[bool, CRUD, Private] = Value(pub_status=Private, default=True)
    _index: Value[tuple[*KeyTt], CRUD, Private] = Value(pub_status=Private, init=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record instance."""
        super().__init__()

        cls = type(self)

        attrs = {name: val for name, val in kwargs.items() if name in cls._values}
        direct_rels = {name: val for name, val in kwargs.items() if name in cls._links}
        attr_sets = {name: val for name, val in kwargs.items() if name in cls._arrays}
        indirect_rels = {
            name: val for name, val in kwargs.items() if name in cls._tables
        }

        # First set all attributes.
        for name, value in attrs.items():
            setattr(self, name, value)

        # Then set all direct relations.
        for name, value in direct_rels.items():
            setattr(self, name, value)

        # Then all attribute sets.
        for name, value in attr_sets.items():
            setattr(self, name, value)

        # Finally set all indirect relations.
        for name, value in indirect_rels.items():
            setattr(self, name, value)

        self.__post_init__()

        pks = type(self)._pk_values
        if len(pks) == 1:
            self._index = getattr(self, next(iter(pks)))
        else:
            self._index = cast(tuple[*KeyTt], tuple(getattr(self, pk) for pk in pks))

        return

    @cached_prop
    def _table(self) -> Table[Self, Any, CRUD, BaseIdx[Self], None, DynBackendID]:
        if not self._connected:
            self._db[type(self)] |= self
            self._connected = True

        data = self._db[type(self)]
        assert isinstance(data, Table)
        return data

    def __post_init__(self) -> None:  # noqa: D105
        pass

    def __hash__(self) -> int:
        """Identify the record by database and id."""
        return gen_int_hash((self._db if self._connected else None, self._index))

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)

    @overload
    def _to_dict(
        self,
        name_keys: Literal[False] = ...,
        include: set[type[Data]] | None = ...,
        with_fks: bool = ...,
    ) -> dict[Data, Any]: ...

    @overload
    def _to_dict(
        self,
        name_keys: Literal[True],
        include: set[type[Data]] | None = ...,
        with_fks: bool = ...,
    ) -> dict[str, Any]: ...

    def _to_dict(
        self,
        name_keys: bool = True,
        include: set[type[Data[Any, Any, Any, Any, Any, Any]]] | None = None,
        with_fks: bool = True,
    ) -> dict[Data, Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Data[Any, Any, Any, Any, Any, Any]], ...] = (
            tuple(include) if include is not None else (Value,)
        )

        vals = {
            p if not name_keys else p.name: getattr(self, p.name)
            for p in (type(self)._props if with_fks else type(self)._props).values()
            if isinstance(p, include_types)
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
            or a.getter is None
            or a.default is Undef
            or a.default_factory is None
            for a in cls._data_values.values()
            if a.init is not False
        ) and all(
            r.name in data or all(fk.name in data for fk in r.fk_map.keys())
            for r in cls._links.values()
        )

    def _update_dict(self) -> None:
        table = self._table[[self._index]]
        df = table.to_df()
        rec_dict = list(df.iter_rows(named=True))[0]
        self.__dict__.update(rec_dict)


class Schema:
    """Group multiple record types into a schema."""

    _record_types: set[type[Record]]
    _rel_record_types: set[type[Record]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        subclasses = get_subclasses(cls, max_level=1)
        cls._record_types = {s for s in subclasses if issubclass(s, Record)}
        cls._rel_record_types = {rr for r in cls._record_types for rr in r._rel_types}
        super().__init_subclass__()


@dataclass
class Require:
    """Mark schema or record type as required."""

    present: bool = True


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(
        cls: type[Record], name: str
    ) -> Data[Value, BaseIdx, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by dynamic name."""
        return Data(db=cls._sym_db, record_type=cls, _attr=Value(_name=name))

    def __getattr__(
        cls: type[Record], name: str
    ) -> Data[Value, BaseIdx, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)
        return Data(db=cls._sym_db, record_type=cls, _attr=Value(_name=name))


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


a = DynRecord


def dynamic_record_type[
    RecT: Record
](
    base: type[RecT],
    name: str,
    props: Iterable[Data[Any, Any, Any, Any, Any, Any]] = [],
) -> type[RecT]:
    """Create a dynamically defined record type."""
    return cast(
        type[RecT],
        new_class(
            name,
            (base,),
            {},
            lambda ns: ns.update(
                {
                    **{p.name: p for p in props},
                    "__annotations__": {p.name: p._typehint for p in props},
                }
            ),
        ),
    )


class RecUUID(Record[UUID]):
    """Record type with a default UUID primary key."""

    _template = True
    _id: Value[UUID] = Value(primary_key=True, default_factory=uuid4)


class RecHashed(Record[int]):
    """Record type with a default hashed primary key."""

    _template = True

    _id: Value[int] = Value(primary_key=True, init=False)

    def __post_init__(self) -> None:  # noqa: D105
        self._id = gen_int_hash(
            {a.name: getattr(self, a.name) for a in type(self)._values.values()}
        )


class BacklinkRecord(Record[KeyT], Generic[KeyT, RecT]):
    """Dynamically defined record type."""

    _template = True

    _from: Link[RecT]


class ArrayRecord(BacklinkRecord[KeyT, RecT], Generic[ValT, KeyT, RecT]):
    """Dynamically defined record type."""

    _template = True

    _id: Value[KeyT] = Value(primary_key=True)
    _value: Value[ValT]


class Relation(RecHashed, BacklinkRecord[int, RecT2], Generic[RecT2, RecT3]):
    """Automatically defined relation record type."""

    _template = True

    _to: Link[RecT3] = Link(index=True)


class IndexedRelation(Relation[RecT2, RecT3], Generic[RecT2, RecT3, KeyT]):
    """Automatically defined relation record type."""

    _template = True

    _rel_id: Value[KeyT] = Value(index=True)
