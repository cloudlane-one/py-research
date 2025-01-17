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
from itertools import groupby
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
    TypeGuard,
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
import sqlparse
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

from py_research.caching import cached_prop
from py_research.data import copy_and_override
from py_research.files import HttpFile
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericProtocol,
    SingleTypeDef,
    has_type,
    is_subtype,
)

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


type Ordinal = (
    bool | int | float | Decimal | datetime | date | time | timedelta | UUID | str
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


type Input[Val, Key: Hashable, RKey: Hashable] = pd.DataFrame | pl.DataFrame | Iterable[
    Val | RKey
] | Mapping[Key, Val | RKey] | sqla.Select | Val | RKey

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

type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]


class Idx[*K]:
    """Define the custom index type of a dataset."""


@final
class BaseIdx[R: Record]:
    """Singleton to mark dataset as having the record type's base index."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound=Idx | BaseIdx,
    default=BaseIdx,
)
IdxT2 = TypeVar(
    "IdxT2",
    bound=Idx | BaseIdx,
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


@final
class Public:
    """Demark public status of attribute."""


@final
class Private:
    """Demark private status of attribute."""


PubT = TypeVar("PubT", bound="Public | Private", default="Public")


DfT = TypeVar("DfT", bound=pd.DataFrame | pl.DataFrame)

Params = ParamSpec("Params")


type LinkItem = Table[Record | None, None, Any, Any, Any, Any]

type PropPath[RootT: Record, LeafT] = tuple[
    Table[RootT, None, Any, Any, None, Any]
] | tuple[
    Table[RootT, None, Any, Any, None, Any],
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


def _map_prop_type_name(name: str) -> type[Data | None]:
    """Map property type name to class."""
    name_map = _prop_type_name_map()
    matches = [name_map[n] for n in name_map if name.startswith(n + "[")]
    return matches[0] if len(matches) == 1 else NoneType


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
    _typehint: str | SingleTypeDef[Data[ValT, Any, Any, Any, Any, Any]] | None = None
    _typevar_map: dict[TypeVar, SingleTypeDef] = field(default_factory=dict)

    _db: DataBase[CrudT | CRUD, BaseT] | None = None
    _ctx: CtxT | Data[Record | None, Any, R, Any, Any, BaseT] | None = None

    _filters: list[
        sqla.ColumnElement[bool]
        | list[tuple[Hashable, ...]]
        | tuple[slice | Hashable, ...]
    ] = field(default_factory=list)
    _tuple_selection: tuple[Data[Any, Any, Any, Any, Any, BaseT], ...] | None = None

    @cached_prop
    def db(self) -> DataBase[CrudT | CRUD, BaseT]:
        """Database, which this dataset belongs to."""
        db = self._db

        if db is None and issubclass(cast(type, self.base_type), NoneType):
            db = cast(DataBase[CRUD, BaseT], DataBase())

        if db is None:
            raise ValueError("Missing backend.")

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
            else Table(_db=self.db, _typehint=Table[self._ctx.record_type])
        )

    @cached_prop
    def path(self) -> PropPath[Record, ValT]:
        """Relational path of this dataset."""
        if self._ctx_table is None:
            assert issubclass(self.target_type, Record)
            return (cast(Table[Record, Any, Any, Any, None, Any], self),)

        return cast(PropPath[Record, ValT], (*self._ctx_table.path, self))

    @cached_prop
    def name(self) -> str:
        """Defined name or generated identifier of this dataset."""
        return self._name if self._name is not None else token_hex(5)

    @property
    def value_type(self) -> SingleTypeDef[ValT] | UnionType:
        """Value typehint of this dataset."""
        return self._get_typearg(ValT)

    @cached_prop
    def target_type(self) -> type[ValT]:
        """Value type of this dataset."""
        t = self._get_typearg(ValT, remove_null=True)

        return cast(
            type[ValT],
            (t if isinstance(t, type) else get_origin(t) or object),
        )

    @cached_prop
    def record_type(self: Data[RecT2 | None, Any, Any, Any, Any, Any]) -> type[RecT2]:
        """Record target type of this dataset."""
        t = self.target_type
        assert issubclass(t, Record)
        return t

    @cached_prop
    def relation_type(self) -> type[RelT]:
        """Relation record type, if any."""
        rec = self._get_typearg(RelT)

        rec_type = self._hint_to_type(rec)
        if not issubclass(rec_type, Record):
            return cast(type[RelT], NoneType)

        return cast(type[RelT], rec_type)

    @cached_prop
    def base_type(self) -> type[BaseT]:
        """Base type of this dataset."""
        base = self._get_typearg(BaseT)
        base_type = self._hint_to_type(base)
        return cast(type[BaseT], base_type)

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if self._ctx_table is None:
            if issubclass(self.target_type, Record):
                return self.target_type._default_table_name()
            else:
                return self.name

        fqn = f"{self._ctx_table.fqn}.{self.name}"

        if len(self._filters) > 0:
            fqn += f"[{gen_str_hash(self._filters)}]"

        return fqn

    @cached_prop
    def rel(
        self: Data[Any, Any, Any, RecT2, Ctx[RecT3], Any]
    ) -> BackLink[RecT2, Any, Any, RecT3, BaseT]:
        """Table pointing to the relations of this dataset, if any."""
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
        index_by = self.index_by if isinstance(self, Table) else None

        return BackLink[RecT2, Any, Any, RecT3, BaseT](
            _db=self.db,
            _ctx=self.ctx,
            _typehint=BackLink[self.relation_type],
            link=cast(Link[Any, Any, Any, Any], link),
            index_by=index_by,
        )

    @cached_prop
    def rec(self: Data[RecT2 | None, Any, Any, Any, Any, Any]) -> type[RecT2]:
        """Path-aware accessor to the record type of this dataset."""
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
        """Tuple-selection of data in the sub-grpah, if any."""
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
            *(
                col._sql_col.label(col_name)
                for col_name, col in (
                    self._total_cols if not index_only else self._abs_idx_cols
                ).items()
            ),
        ).select_from(self._root_table._sql_table)

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

    @property
    def select_str(self) -> str:
        """Return select statement for this dataset."""
        return sqlparse.format(
            str(self.select.compile(self.db.engine)),
            reindent=True,
            keyword_case="upper",
        )

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

    def keys(
        self: Data[Any, Any, Any, Any, Any, Any],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        df = self.to_df(index_only=True)
        if len(self._abs_idx_cols) == 1:
            return [tup[0] for tup in df.iter_rows()]

        return list(df.iter_rows())

    def values(
        self: Data[ValT2, Any, Any, Any, Any, DynBackendID],
    ) -> Sequence[ValT2]:
        """Iterable over this dataset's values."""
        dfs = self.to_df()
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

    def items(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
    ) -> Iterable[tuple[Any, Any]]:
        """Iterable over this dataset's items."""
        return zip(self.keys(), self.values())

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
    def to_df(
        self: Data[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT],
        index_only: bool = ...,
    ) -> DfT: ...

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
    ) -> DfT:
        """Load dataset as dataframe."""
        select = type(self).select(self, index_only=index_only)

        merged_df = None
        if kind is pd.DataFrame:
            with self.db.engine.connect() as con:
                merged_df = pd.read_sql(select, con)
                merged_df = merged_df.set_index(
                    list(self._abs_idx_cols.keys()), drop=False
                )
        else:
            merged_df = pl.read_database(
                select,
                self.db.engine.connect(),
            )

        return cast(DfT, merged_df)

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
        merged_df = self.to_df(kind=kind)

        name_map = [
            {col.fqn: col.name for col in col_set}
            for col_set in self._abs_col_sets.values()
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
        """Extract a new database instance from the current dataset."""
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

    def isin(
        self: Data[ValT2, Any, Any, Any, Any, BaseT2], other: Iterable[ValT2] | slice
    ) -> sqla.ColumnElement[bool]:
        """Test values of this dataset for membership in the given iterable."""
        return (
            self._sql_col.between(other.start, other.stop)
            if isinstance(other, slice)
            else self._sql_col.in_(other)
        )

    # 1. DB-level type selection
    @overload
    def __getitem__(
        self: Data[Any, Idx[()], Any, None, None, Any],
        key: type[RecT3],
    ) -> Data[RecT3, BaseIdx[RecT3], CrudT, None, None, BaseT]: ...

    # 2. Top-level prop selection
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

    # 3. Nested prop selection
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

    # 4. Key filtering, scalar index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 5. Key / slice filtering
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, DynBackendID
        ],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 6. Expression filtering
    @overload
    def __getitem__(
        self: Data[Any, Idx | BaseIdx, Any, Any, Any, DynBackendID],
        key: sqla.ColumnElement[bool],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 7. Key selection, scalar index type, symbolic context
    @overload
    def __getitem__(
        self: Data[Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Symbolic],
        key: KeyT2,
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 8. Key selection, tuple index type, symbolic context
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Symbolic
        ],
        key: tuple[*KeyTt2],
    ) -> Data[ValT, IdxT, RU, RelT, CtxT, BaseT]: ...

    # 9. Key selection, scalar index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
    ) -> ValT: ...

    # 10. Key selection, tuple index type
    @overload
    def __getitem__(
        self: Data[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, DynBackendID
        ],
        key: tuple[*KeyTt2],
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
                return Table(_db=self.db, _typehint=Table[key])
            case Data():
                return (
                    key._add_ctx(self._ctx_table)
                    if self._ctx_table is not None
                    else copy_and_override(
                        type(key),
                        key,
                        _db=self.db,
                    )
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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __imatmul__(
        self: Data[Any, Any, RU, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, Any] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Aligned assignment."""
        self._mutate(other, mode="update")
        return cast(
            Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT],
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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __iadd__(
        self: Data[Any, Any, CR, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, Any] | Input[Any, Any, Any],
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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __ior__(
        self: Data[Any, Any, CRU, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Upserting assignment."""
        self._mutate(other, mode="upsert")
        return cast(
            Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT],
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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __iand__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any, BaseT] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Replacing assignment."""
        self._mutate(other, mode="replace")
        return cast(
            Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT],
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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

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
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]: ...

    def __isub__(
        self: Data[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Data[Any, Any, Any, Any, Any] | Input[Any, Any, Any],
    ) -> Data[ValT, IdxT, CrudT, RelT, CtxT, BaseT]:
        """Idempotent deletion."""
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
            | sqla.ColumnElement[bool]
        ),
    ) -> None:
        if not isinstance(key, slice) or key != slice(None):
            del self[key][:]

        self._mutate([], mode="replace")

    @overload
    def __eq__(  # noqa: D105
        self: Data[Any, IdxT2, Any, Any, Ctx, BaseT2],
        other: Any | Data[Any, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    @overload
    def __eq__(  # noqa: D105
        self,
        other: Any,
    ) -> bool: ...

    def __eq__(  # noqa: D105
        self: Data[Any, Any, Any, Any, Any, Any],
        other: Any | Data[Ordinal, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool] | bool:
        if self._ctx is None:
            return hash(self) == hash(other)

        if isinstance(other, Data):
            return self._sql_col == other._sql_col

        return self._sql_col == other

    def __neq__(  # noqa: D105
        self: Data[Any, IdxT2, Any, Any, Ctx, BaseT2],
        other: Any | Data[Ordinal, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self._sql_col != other._sql_col
        return self._sql_col != other

    def __lt__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Ctx, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self._sql_col < other._sql_col
        return self._sql_col < other

    def __lte__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Ctx, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self._sql_col <= other._sql_col
        return self._sql_col <= other

    def __gt__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Ctx, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self._sql_col > other._sql_col
        return self._sql_col > other

    def __gte__(  # noqa: D105
        self: Data[OrdT, IdxT2, Any, Any, Ctx, BaseT2],
        other: OrdT | Data[OrdT, IdxT2, Any, Any, Any, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Data):
            return self._sql_col >= other._sql_col
        return self._sql_col >= other

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

    def __setitem__(
        self,
        key: Any,
        other: Data[Any, Any, Any, Any, Any, Any],
    ) -> None:
        """Catchall setitem."""
        return

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
        owner = self._ctx_type.record_type if self._ctx_type is not None else owner

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

        # # Commented out to have context association done only upon record class
        # # initialization. Thereby subclasses won't override superclasses as context.
        # if owner is not None and issubclass(owner, Record) and instance is None:
        #     return copy_and_override(
        #         Data[ValT, Idx[()], R, RelT, Ctx, Symbolic],
        #         self,
        #         _db=symbol_db,
        #         _ctx=Ctx(owner),
        #     )

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
        return len(list(iter(self)))

    @property
    def _data_type(self) -> type[Data] | type[None]:
        """Resolve the data container type reference."""
        hint = self._typehint

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
        hint = self._typehint or Data
        generic = self._hint_to_typedef(hint)
        assert is_subtype(generic, Data)

        return generic

    @cached_prop
    def _generic_args(self) -> tuple[SingleTypeDef, ...]:
        return get_args(self._generic_type)

    @cached_prop
    def _ctx_type(self) -> CtxT:
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
                Table(_db=self.db, _typehint=Table[self._ctx.record_type])
                if self._ctx is not None
                else None
            )
        )

    @cached_prop
    def _ctx_module(self) -> ModuleType | None:
        """Module of the context record type."""
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

    @property
    def _home_table(self) -> Data[Record | None, Any, Any, Any, Any, BaseT]:
        if Data._has_type(self, Data[Record | None, Any, Any, Any, Any, Any]):
            return self

        if Data._has_type(self, Data[Any, Any, Any, Record, Ctx, Any]):
            return self.rel

        assert self._ctx_table is not None
        return self._ctx_table

    @property
    def _root_table(self) -> Data[Record | None, Any, Any, None, None, BaseT]:
        return self.path[0]

    @property
    def _idx_cols(
        self: Data[Record | None, Any, Any, Any, Any, Any]
    ) -> list[Value[Any, Any, Public, Any, Symbolic]]:
        if isinstance(self, Table) and self.index_by is not None:
            return (
                [self.index_by]
                if isinstance(self.index_by, Value)
                else list(self.index_by)
            )

        if self._ctx is not None and issubclass(self.record_type, IndexedRelation):
            return [self.record_type._rel_id]  # type: ignore

        return list(self.record_type._pk_values.values())

    @cached_prop
    def _rel_to(
        self: Data[RefT2, Any, Any, RecT2, Ctx, Any]
    ) -> Link[RefT2, Any, RecT2, BaseT] | None:
        links = (
            [self.relation_type._to]  # type: ignore
            if issubclass(self.relation_type, Relation)
            else [
                r
                for r in self.relation_type._links.values()
                if isinstance(r, Link) and issubclass(self.record_type, r.record_type)
            ]
        )

        if len(links) != 1:
            return None

        return copy_and_override(
            Link[RefT2, Any, RecT2, BaseT], links[0], _db=self.db, _ctx=self.rel
        )

    @cached_prop
    def _abs_link_path(self) -> list[LinkItem]:
        path: list[LinkItem] = []

        for p in self.path:
            if issubclass(p.relation_type, Record):
                p = cast(Table[Record, Record, Any, Any, Any, Any], p)
                if p._rel_to is not None:
                    path.extend([p.rel, p._rel_to])
                else:
                    path.append(p.rel)
            elif issubclass(p.target_type, Record):
                path.append(cast(Table[Record, None, Any, Any, Any, Any], p))

        return path

    @property
    def _abs_idx_cols(self) -> dict[str, Value[Any, Any, Any, Any, BaseT]]:
        indexes = [
            copy_and_override(
                Value[Any, Any, Any, Any, BaseT],
                pk,
                _db=self.db,
                _ctx=node,
            )
            for node in self._abs_link_path
            for pk in node._idx_cols
        ]

        return (
            {col.fqn: col for col in indexes}
            if issubclass(self.target_type, tuple)
            else {col.name: col for col in indexes}
        )

    @cached_prop
    def _abs_cols(self) -> dict[str, Value[Any, Any, Any, Any, BaseT]]:
        cols = (
            {
                v._add_ctx(self._ctx_table)
                for sel in self.tuple_selection
                for v in sel._abs_cols.values()
            }
            if Data._has_type(self, Data[tuple, Any, Any, Any, Ctx, Any])
            else (
                {
                    copy_and_override(
                        Value[Any, Any, Any, Any, BaseT], v, _db=self.db, _ctx=self
                    )
                    for v in self.record_type._col_values.values()
                }
                if Data._has_type(self, Data[Record | None, Any, Any, Any, Any, Any])
                else {
                    copy_and_override(
                        Value[Any, Any, Any, Any, BaseT], v, _db=self.db, _ctx=self._ctx
                    )
                    for v in (
                        (self.rel.record_type._value,)  # type: ignore
                        if Data._has_type(
                            self, Data[Any, Any, Any, ArrayRecord, Ctx, Any]
                        )
                        else (self,)
                    )
                }
            )
        )

        return (
            {col.fqn: col for col in cols}
            if issubclass(self.target_type, tuple)
            else {col.name: col for col in cols}
        )

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
        self,
    ) -> dict[
        Data[Record | None, Any, Any, Any, Any, Any],
        dict[str, Value[Any, Any, Any, Any, BaseT]],
    ]:
        return (
            {
                tab: {
                    v_name: copy_and_override(
                        Value[Any, Any, Any, Any, BaseT], v, _db=self.db, _ctx=self._ctx
                    )
                    for v_name, v in vals.items()
                }
                for sel in self.tuple_selection
                for tab, vals in sel._base_table_map.items()
            }
            if Data._has_type(self, Data[tuple, Any, Any, Any, Any, Any])
            else (
                {
                    self.db[base_rec]: {
                        v_name: v
                        for v_name, v in self._abs_cols.items()
                        if v.name in base_rec._values
                    }
                    for base_rec in [
                        self.record_type,
                        *self.record_type._record_superclasses,
                    ]
                }
                if Data._has_type(self, Data[Record | None, Any, Any, Any, Any, Any])
                else self._home_table._base_table_map
            )
        )

    @cached_prop
    def _abs_filters(
        self,
    ) -> tuple[
        list[sqla.ColumnElement[bool]], list[Table[Any, Any, Any, Any, Any, BaseT]]
    ]:
        sql_filt = [f for f in self._filters if isinstance(f, sqla.ColumnElement)]

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
                if Data._has_type(self, Data[tuple, Any, Any, Any, Ctx, Any])
                else (self,)
            )
        ]

    @cached_prop
    def _total_cols(self) -> dict[str, Value[Any, Any, Any, Any, BaseT]]:
        return {**self._abs_idx_cols, **self._abs_cols}

    @cached_prop
    def _total_joins(self) -> list[Table[Record | None, Any, Any, Any, Any, BaseT]]:
        sel = self._abs_joins + self._abs_filters[1]
        return sel if self._ctx_table is None else sel + self._ctx_table._total_joins

    @cached_prop
    def _total_join_dict(self) -> JoinDict:
        tree: JoinDict = {}

        for rel in self._total_joins:
            subtree = tree
            for node in rel._abs_link_path[1:]:
                if node not in subtree:
                    subtree[node] = {}
                subtree = subtree[node]

        return tree

    @cached_prop
    def _fk_map(self: Data[Record | None, Idx[()], Any, None, Ctx, Any]) -> bidict[
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
                        _name=f"{to_rec._default_table_name()}_{pk.name}",
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
        _parent: Data[Record | None, Any, Any, Any, Any, Any] | None = None,
    ) -> list[SqlJoin]:
        """Extract join operations from the relational tree."""
        joins: list[SqlJoin] = []
        _subtree = _subtree if _subtree is not None else self._total_join_dict
        _parent = _parent if _parent is not None else self._root_table

        for target, next_subtree in _subtree.items():
            joins.append(
                (
                    target._sql_table,
                    reduce(
                        sqla.and_,
                        (
                            (
                                _parent[fk] == target[pk]
                                for fk, pk in target._fk_map.items()
                            )
                            if isinstance(target, Link)
                            else (
                                target[fk] == _parent[pk]
                                for fk, pk in target._fk_map.items()
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
            self._ctx_table._sql_filters if self._ctx_table is not None else []
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
            rel_table = self[rt]._get_sql_base_table()
            fks.append(
                sqla.ForeignKeyConstraint(
                    [fk.name for fk in rt._fk_map.keys()],
                    [rel_table.c[pk.name] for pk in rt._fk_map.values()],
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
        return self.ctx._sql_table.c[self.name]

    @cached_prop
    def _sql_table(
        self: Data[Record | None, Any, Any, Any, Any, Any]
    ) -> sqla.FromClause:
        """Recursively join all bases of this record to get the full data."""
        base_table = self._get_sql_base_table("read")
        if len(self.record_type._record_superclasses) == 0:
            return base_table

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

    @staticmethod
    def _has_type[
        D: Data[Any, Any, Any, Any, Any, Any]
    ](obj: Data[Any, Any, Any, Any, Any, Any], type_: SingleTypeDef[D]) -> TypeGuard[D]:
        """Check if object is of given type hint."""
        tmpl = Data[Any, Any, Any, Any, Any, Any](_typehint=type_)

        return has_type(obj, type_) and all(
            not isinstance(obj_type := obj._get_typearg(t), Undef)
            and is_subtype(
                obj_type,
                tmpl._get_typearg(t),
            )
            for t in tmpl._typearg_map.keys()
        )

    def _get_typearg(
        self, typevar: TypeVar, remove_null: bool = False
    ) -> SingleTypeDef | UnionType:
        """Get the type argument for a type variable."""
        assert len(self._generic_args) > 0

        typearg = self._typearg_map[typevar]
        if isinstance(typearg, int):
            return self._hint_to_typedef(
                self._generic_args[typearg], remove_null=remove_null
            )

        return typearg

    def _hint_to_typedef(
        self,
        hint: SingleTypeDef | UnionType | TypeVar | str | ForwardRef,
        remove_null: bool = False,
    ) -> SingleTypeDef | UnionType:
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
            if remove_null:
                union_types: set[type] = {
                    get_origin(union_arg) or union_arg
                    for union_arg in get_args(typedef)
                }
                union_types = {
                    t
                    for t in union_types
                    if t is not None and not issubclass(t, NoneType)
                }
                assert len(union_types) == 1
                return next(iter(union_types))

            return typedef

        return cast(SingleTypeDef, typedef)

    def _hint_to_type(self, hint: SingleTypeDef | UnionType | TypeVar | None) -> type:
        if hint is None:
            return NoneType

        if isinstance(hint, type):
            return hint

        typedef = self._hint_to_typedef(hint) if isinstance(hint, TypeVar) else hint
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
        """Check if ``other`` is ancestor of this data."""
        return other is self._ctx or (
            self._ctx_table is not None and self._ctx_table._has_ancestor(other)
        )

    def _add_ctx(
        self,
        left: Data[Record | None, Any, Any, Any, Any, Any] | None,
    ) -> Self:
        """Prefix this dataset with another table as context."""
        if left is None:
            return self

        return cast(
            Self,
            reduce(
                lambda x, y: copy_and_override(
                    cast(type[Data[Record, Any, Any, Any, Any, BaseT]], type(y)),
                    y,
                    _db=x.db,
                    _ctx=x,
                ),
                self.path,
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
        if isinstance(element, Value) and self._ctx_table is not None:
            prefixed = element._add_ctx(self._ctx_table)
            join_set.add(prefixed.ctx)
            return prefixed._sql_col

        return None

    def _get_sql_base_table(
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
        col_data = [
            {
                **{
                    idx_name: v
                    for idx_name, v in zip(
                        self._abs_idx_cols,
                        idx if isinstance(idx, tuple) else (idx,),
                    )
                },
                **{
                    v_name: v
                    for val in (
                        vals
                        if self._tuple_selection is not None
                        and len(self._tuple_selection) > 1
                        else [vals]
                    )
                    for v_name, v in (
                        val._to_dict().items()
                        if isinstance(val, Record)
                        else [(self.name, val)]
                    )
                    if include is None or v_name in include
                },
            }
            for idx, vals in values.items()
        ]

        return pl.DataFrame(
            col_data,
            schema=_get_pl_schema(
                {
                    col_name: col
                    for col_name, col in self._total_cols.items()
                    if include is None or col_name in include
                }
            ),
        )

    def _gen_upload_table(
        self,
        include: list[str] | None = None,
    ) -> sqla.Table:
        type_map = (
            self.record_type._type_map
            if Data._has_type(self, Data[Record | None, Any, Any, Any, Any, Any])
            else (
                self.ctx.record_type._type_map
                if Data._has_type(self, Data[Any, Any, Any, Any, Ctx, Any])
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
            for col_name, col in self._total_cols.items()
            if include is None or col_name in include
        ]

        table_name = f"{self.fqn}[{token_hex(5)}]"
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

    def _mutate(
        self: Data[ValT2, Any, Any, Any, Any, DynBackendID],
        value: Data[ValT2, Any, Any, Any, Any, Any] | Input[ValT2, Hashable, Hashable],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
        autosave: bool = True,
    ) -> None:
        record_ids: dict[Hashable, Hashable] | None = None
        valid_caches = self.db._get_valid_cache_set(self._home_table.record_type)

        match value:
            case sqla.Select():
                self._mutate_from_sql(value.subquery(), mode)
                valid_caches.clear()
            case Data():
                if hash(value.db) != hash(self.db):
                    if has_type(value, Data[Record | None, Any, Any, Any, Any, Any]):
                        remote_db = (
                            value if isinstance(value, DataBase) else value.extract()
                        )
                        for s in remote_db._def_types:
                            if remote_db.db_id == self.db.db_id:
                                self.db[s]._mutate_from_sql(
                                    remote_db[s].query, "upsert"
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
                    else:
                        value_table = self._df_to_table(value.to_df())
                        self._mutate_from_sql(value_table, mode)
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

                base_idx_cols = list(self._home_table.record_type._pk_values.keys())
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
                        )._mutate_from_rec_ids(record_ids, mode)
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
                    )._mutate_from_rec_ids({value: value}, mode)
                    valid_caches -= {value}

        if autosave and self.db.backend_type == "excel-file":
            self.db._save_to_excel()

        return

    def _mutate_from_records(
        self: Data[Record | None, Any, CRUD, Relation | None, Ctx | None, DynBackendID],
        values: Mapping[Any, Record],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
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

        if unconnected_records:
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
            value_table = self._df_to_table(
                self._values_to_df(
                    local_records, include=list(self.record_type._pk_values.keys())
                ),
            )
            self._mutate_rels_from_sql(
                value_table,
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
                        remote_db[s].to_df(),
                    )
                    self.db[s]._mutate_from_sql(
                        value_table,
                        "upsert",
                    )
                    value_table.drop(self.db.engine)

            self._mutate_from_sql(remote_set.query, mode)

        return

    def _mutate_from_rec_ids(
        self: Data[Record | None, Any, CRUD, Record, Ctx, DynBackendID],
        values: Mapping[Hashable, Hashable],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        value_table = self._df_to_table(
            self._values_to_df(
                values, include=list(self.record_type._pk_values.keys())
            ),
        )
        self._mutate_rels_from_sql(
            value_table,
            mode,
        )

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
            for tab, vals in self._base_table_map.items()
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
                            [col.name for col in vals.values()],
                            sqla.select(
                                *(
                                    value_table.c[col_name].label(col.name)
                                    for col_name, col in vals.items()
                                )
                            ).select_from(value_table),
                        )
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
                        [c.name for c in vals.values()],
                        sqla.select(
                            *(
                                value_table.c[col_name].label(col.name)
                                for col_name, col in vals.items()
                            )
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
                col_names = {c_name: c.name for c_name, c in vals.items()}
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
        self._mutate_rels_from_sql(value_table, mode)

        return

    def _mutate_rels_from_sql(
        self,
        value_table: sqla.FromClause,
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        # Update relations with parent records.
        if Data._has_type(self, Link[Record | None, Any, Any]):
            # Case: parent links directly to child (n -> 1)
            idx_cols = [
                value_table.c[col_name] for col_name in self._abs_idx_cols.keys()
            ]
            fk_cols = [
                value_table.c[pk.name].label(fk.name) for fk, pk in self._fk_map.items()
            ]
            self.ctx._mutate_from_sql(
                sqla.select(*idx_cols, *fk_cols).select_from(value_table).subquery(),
                "update",
            )
        elif Data._has_type(self, Data[Any, Any, Any, Record, Ctx, Any]):
            # Case: parent and child are linked via assoc table (n <--> m)
            # Update link table with new child indexes.
            idx_cols = [
                value_table.c[col_name] for col_name in self._abs_idx_cols.keys()
            ]
            to_fk_cols: list[sqla.ColumnElement] = (
                [
                    value_table.c[pk.name].label(fk.name)
                    for fk, pk in self._rel_to._fk_map.items()
                ]
                if self._rel_to is not None
                else []
            )

            ctx_table = Data.select(self.ctx, index_only=True).subquery()

            from_fk_cols = [
                ctx_table.c[pk.name].label(fk.name)
                for fk, pk in self.rel._fk_map.items()
            ]
            idx_col_map = {
                value_table.c[col_name]: ctx_table.c[col_name]
                for col_name in self._abs_idx_cols.keys()
                if col_name in ctx_table.columns
            }

            self.rel._mutate_from_sql(
                sqla.select(*idx_cols, *from_fk_cols, *to_fk_cols)
                .select_from(value_table)
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
                .subquery(),
                mode,
            )

        return


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
    Table[RecT, None, CrudT, TdxT, Ctx[ParT], BaseT],
    Generic[RecT, CrudT, TdxT, ParT, BaseT],
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


@dataclass(eq=False)
class Array(
    Data[ValT, Idx[KeyT], CrudT, "ArrayRecord[ValT, KeyT, OwnT]", Ctx[OwnT], BaseT],
    Generic[ValT, KeyT, CrudT, OwnT, BaseT],
):
    """Set / array of scalar values."""

    _typearg_map: ClassVar[dict[TypeVar, int | SingleTypeDef]] = {
        ValT: 0,
        IdxT: Idx[Any],
        CrudT: 2,
        RelT: NoneType,
        CtxT: Ctx[Any],
        BaseT: 4,
    }

    @cached_prop
    def _key_type(self) -> type[KeyT]:
        args = self._generic_args
        if len(args) == 0:
            return cast(type[KeyT], object)

        return cast(type[KeyT], args[1])

    @cached_prop
    def relation_type(self) -> type[ArrayRecord[ValT, KeyT, OwnT]]:
        """Return the dynamic relation record type."""
        return dynamic_record_type(
            ArrayRecord[self.target_type, self._key_type, self._ctx_type.record_type],
            f"{self._ctx_type.record_type._default_table_name()}_{self.name}",
        )


@dataclass
class DataBase(Data[Any, Idx[()], CrudT, None, None, BaseT]):
    """Database connection."""

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
        self._db = self

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
        self: DataBase[CRUD, Any],
        data: pd.DataFrame | pl.DataFrame | sqla.Select,
        fks: (
            Mapping[
                str,
                Value[Any, Any, Public, Any, Symbolic],
            ]
            | None
        ) = None,
    ) -> Data[DynRecord, BaseIdx[DynRecord], R, None, None, BaseT]:
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

        ds = self[rec]
        ds &= data

        return ds

    # def to_graph(
    #     self: DataBase[R, Any], nodes: Sequence[type[Record]]
    # ) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """Export links between select database objects in a graph format.

    #     E.g. for usage with `Gephi`_

    #     .. _Gephi: https://gephi.org/
    #     """
    #     node_tables = [self[n] for n in nodes]

    #     # Concat all node tables into one.
    #     node_dfs = [
    #         n.to_df(kind=pd.DataFrame)
    #         .reset_index()
    #         .assign(table=n.target_type._default_table_name())
    #         for n in node_tables
    #     ]
    #     node_df = (
    #         pd.concat(node_dfs, ignore_index=True)
    #         .reset_index()
    #         .rename(columns={"index": "node_id"})
    #     )

    #     directed_edges = reduce(
    #         set.union, (set((n, r) for r in n._refs.values()) for n in nodes)
    #     )

    #     undirected_edges: dict[type[Record], set[tuple[Ref, ...]]] = {
    #         t: set() for t in nodes
    #     }
    #     for n in nodes:
    #         for at in self._assoc_types:
    #             if len(at._refs) == 2:
    #                 left, right = (r for r in at._refs.values())
    #                 assert left is not None and right is not None
    #                 if left.target_type == n:
    #                     undirected_edges[n].add((left, right))
    #                 elif right.target_type == n:
    #                     undirected_edges[n].add((right, left))

    #     # Concat all edges into one table.
    #     edge_df = pd.concat(
    #         [
    #             *[
    #                 node_df.loc[node_df["table"] == str(parent._default_table_name())]
    #                 .rename(columns={"node_id": "source"})
    #                 .merge(
    #                     node_df.loc[
    #                         node_df["table"]
    #                         == str(link.target_type._default_table_name())
    #                     ],
    #                     left_on=[c.name for c in link.fk_map.keys()],
    #                     right_on=[c.prop.name for c in link.fk_map.values()],
    #                 )
    #                 .rename(columns={"node_id": "target"})[["source", "target"]]
    #                 .assign(
    #                     ltr=",".join(c.name for c in link.fk_map.keys()),
    #                     rtl=None,
    #                 )
    #                 for parent, link in directed_edges
    #             ],
    #             *[
    #                 self[assoc_table]
    #                 .to_df(kind=pd.DataFrame)
    #                 .merge(
    #                     node_df.loc[
    #                         node_df["table"]
    #                         == str(left_rel.value_origin_type._default_table_name())
    #                     ].dropna(axis="columns", how="all"),
    #                     left_on=[c.name for c in left_rel.fk_map.keys()],
    #                     right_on=[c.name for c in left_rel.fk_map.values()],
    #                     how="inner",
    #                 )
    #                 .rename(columns={"node_id": "source"})
    #                 .merge(
    #                     node_df.loc[
    #                         node_df["table"]
    #                         == str(left_rel.value_origin_type._default_table_name())
    #                     ].dropna(axis="columns", how="all"),
    #                     left_on=[c.name for c in right_rel.fk_map.keys()],
    #                     right_on=[c.name for c in right_rel.fk_map.values()],
    #                     how="inner",
    #                 )
    #                 .rename(columns={"node_id": "target"})[
    #                     list(
    #                         {
    #                             "source",
    #                             "target",
    #                             *(a for a in self[assoc_table]._col_attrs),
    #                         }
    #                     )
    #                 ]
    #                 .assign(
    #                     ltr=",".join(c.name for c in right_rel.fk_map.keys()),
    #                     rtl=",".join(c.name for c in left_rel.fk_map.keys()),
    #                 )
    #                 for assoc_table, rels in undirected_edges.items()
    #                 for left_rel, right_rel in rels
    #             ],
    #         ],
    #         ignore_index=True,
    #     )

    #     return node_df, edge_df

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
            pks = set([col.name for col in assoc_table.record_type._pk_values.values()])
            fks = set(
                [col.name for rel in rec._links.values() for col in rel._fk_map.keys()]
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


symbol_db = DataBase[R, Symbolic](backend=Symbolic())
"""Database for all symbolic data instances."""


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

        # Construct prelimiary symbolic data instances for all class annotations.
        props = {
            name: Data[Any, Any, Any, Any, Ctx, Symbolic](
                _typehint=hint,
                _db=symbol_db,
                _ctx=Ctx(cast(type["Record"], cls)),
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
                    _typehint=prop._typehint,
                    _ctx=prop._ctx,
                )
            else:
                prop = copy_and_override(prop_type, prop)

            setattr(cls, prop_name, prop)
            props[prop_name] = prop

        # Construct list of Record superclasses and apply template superclasses.
        cls._record_superclasses = []
        super_types: Iterable[type | GenericProtocol] = (
            cls.__dict__["__orig_bases__"]
            if "__orig_bases__" in cls.__dict__
            else cls.__bases__
        )
        for c in super_types:
            # Get proper origin class of generic supertype.
            orig = get_origin(c) if not isinstance(c, type) else c

            # Handle typevar substitutions.
            if orig is not c and hasattr(orig, "__parameters__"):
                typevar_map = dict(zip(getattr(orig, "__parameters__"), get_args(c)))
            else:
                typevar_map = {}

            # Skip root Record class and non-Record classes.
            if not isinstance(orig, RecordMeta) or orig._is_root_class:
                continue

            # Apply template classes.
            if orig._template or cls._derivate:
                for prop_name, super_prop in orig._class_props.items():
                    if prop_name in props:
                        prop = copy_and_override(
                            type(props[prop_name]), (super_prop, props[prop_name])
                        )
                    else:
                        prop = copy_and_override(
                            type(super_prop),
                            super_prop,
                            _typevar_map=typevar_map,
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
            for a in ref._fk_map.keys()
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
    def _arrays(cls) -> dict[str, Array[Any, Any, Any, Any, Symbolic]]:
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
            BackLink(
                link=cast(Link[Self, Any, RecT2, Symbolic], ln),
                _ctx=Ctx(cls),
                _typehint=BackLink[target],
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

    _database: Value[DataBase[CRUD, DynBackendID] | None, CRUD, Private] = Value(
        pub_status=Private,
        default=None,
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
    def _db(self) -> DataBase[CRUD, DynBackendID]:
        if self._database is None:
            self._database = DataBase[CRUD, DynBackendID]()

        return self._database

    @cached_prop
    def _table(self) -> Table[Self, Any, CRUD, BaseIdx[Self], None, DynBackendID]:
        table = self._db[type(self)]
        if not self._published:
            table |= self
            self._published = True

        assert isinstance(table, Table)
        return table

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
        name_keys: Literal[False] = ...,
        include: set[type[Data]] | None = ...,
        with_fks: bool = ...,
        with_private: bool = ...,
    ) -> dict[Data, Any]: ...

    @overload
    def _to_dict(
        self,
        name_keys: Literal[True],
        include: set[type[Data]] | None = ...,
        with_fks: bool = ...,
        with_private: bool = ...,
    ) -> dict[str, Any]: ...

    def _to_dict(
        self,
        name_keys: bool = True,
        include: set[type[Data[Any, Any, Any, Any, Any, Any]]] | None = None,
        with_fks: bool = True,
        with_private: bool = False,
    ) -> dict[Data, Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Data[Any, Any, Any, Any, Any, Any]], ...] = (
            tuple(include) if include is not None else (Value,)
        )

        vals = {
            p if not name_keys else p.name: getattr(self, p.name)
            for p in (type(self)._props if with_fks else type(self)._props).values()
            if isinstance(p, include_types)
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
            or a.getter is None
            or a.default is Undef
            or a.default_factory is None
            for a in cls._data_values.values()
            if a.init is not False
        ) and all(
            r.name in data or all(fk.name in data for fk in r._fk_map.keys())
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
    ) -> Value[Any, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by dynamic name."""
        return Value(_db=symbol_db, _typehint=Value[cls], _ctx=Ctx(cls), _name=name)

    def __getattr__(
        cls: type[Record], name: str
    ) -> Value[Any, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)
        return Value(_db=symbol_db, _typehint=Value[cls], _ctx=Ctx(cls), _name=name)


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


a = DynRecord


def dynamic_record_type(
    base: type[RecT2],
    name: str,
    props: Iterable[Data[Any, Any, Any, Any, Any, Any]] = [],
) -> type[RecT2]:
    """Create a dynamically defined record type."""
    return cast(
        type[RecT2],
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


class BacklinkRecord(Record[KeyT2], Generic[KeyT2, RecT2]):
    """Dynamically defined record type."""

    _template = True

    _from: Link[RecT2]


class ArrayRecord(BacklinkRecord[KeyT2, RecT2], Generic[ValT, KeyT2, RecT2]):
    """Dynamically defined scalar record type."""

    _template = True

    _id: Value[KeyT2] = Value(primary_key=True)
    _value: Value[ValT]


class Relation(RecHashed, BacklinkRecord[int, RecT2], Generic[RecT2, RecT3]):
    """Automatically defined relation record type."""

    _template = True

    _to: Link[RecT3] = Link(index=True)


class IndexedRelation(Relation[RecT2, RecT3], Generic[RecT2, RecT3, KeyT]):
    """Automatically defined relation record type with index substitution."""

    _template = True

    _rel_id: Value[KeyT] = Value(index=True)
