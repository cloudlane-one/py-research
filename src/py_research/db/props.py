"""Static schemas for universal relational databases."""

from __future__ import annotations

from abc import abstractmethod
from calendar import day_abbr
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from copy import copy
from dataclasses import dataclass, field
from functools import cache, reduce
from inspect import get_annotations, getmodule
from types import ModuleType, NoneType, UnionType
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeGuard,
    Unpack,
    cast,
    final,
    get_origin,
    overload,
    runtime_checkable,
)

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.sql.visitors as sqla_visitors
import sqlparse
from typing_extensions import TypeVar, TypeVarTuple

from py_research.caching import cached_prop
from py_research.data import copy_and_override
from py_research.hashing import gen_str_hash
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    SingleTypeDef,
    TypeRef,
    get_lowest_common_base,
    get_typevar_map,
    has_type,
    hint_to_typedef,
    is_subtype,
    typedef_to_typeset,
)
from py_research.types import Not, Ordinal

ValT = TypeVar("ValT", covariant=True, default=Any)
ValT2 = TypeVar("ValT2")
ValT3 = TypeVar("ValT3")
ValTi = TypeVar("ValTi", default=Any)

KeyT = TypeVar("KeyT", bound=Hashable)
KeyTt = TypeVarTuple("KeyTt")
KeyTt2 = TypeVarTuple("KeyTt2")
KeyTt3 = TypeVarTuple("KeyTt3")

OrdT = TypeVar("OrdT", bound=Ordinal)

SchemaT = TypeVar("SchemaT", contravariant=True, default=Any)

type SqlExpr = (sqla.SelectBase | sqla.FromClause | sqla.ColumnElement)

SqlT = TypeVar(
    "SqlT",
    bound=SqlExpr | None,
    covariant=True,
    default=Any,
)
SqlT2 = TypeVar(
    "SqlT2",
    bound=SqlExpr | None,
)
SqlT3 = TypeVar(
    "SqlT3",
    bound=SqlExpr | None,
)
SqlTi = TypeVar(
    "SqlTi",
    bound=SqlExpr | None,
    default=Any,
)


DfT = TypeVar("DfT", bound=pd.DataFrame | pl.DataFrame)


@final
class Idx(Generic[*KeyTt]):
    """Define the custom index type of a dataset."""


@final
class SelfIdx(Generic[*KeyTt]):
    """Index by self."""


@final
class HashIdx(Generic[*KeyTt]):
    """Index by hash of self."""


class AutoIndexable(Protocol[*KeyTt]):
    """Base class for indexable objects."""

    @classmethod
    def sql_cols(cls) -> list[sqla.ColumnElement]:
        """Get SQL columns for this auto-indexed type."""
        ...


AutoIdxT = TypeVar("AutoIdxT", bound=AutoIndexable, covariant=True)


@final
class AutoIdx(Generic[AutoIdxT]):
    """Index by custom value derived from self."""


type AnyIdx[*K] = Idx[*K] | SelfIdx[*K] | HashIdx[*K] | AutoIdx[AutoIndexable[*K]]


IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound=AnyIdx,
    default=AnyIdx[*tuple[Any, ...]],
)
IdxT2 = TypeVar(
    "IdxT2",
    bound=AnyIdx,
)
IdxT3 = TypeVar(
    "IdxT3",
    bound=AnyIdx,
)
IdxTi = TypeVar(
    "IdxTi",
    bound=AnyIdx,
    default=AnyIdx[*tuple[Any, ...]],
)


@final
class C:
    """Singleton to allow creation of new records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class R:
    """Singleton to allow reading of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class U:
    """Singleton to allow updating of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class D:
    """Singleton to allow deletion of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


type RU = R | U

type CRUD = C | R | U | D

CrudT = TypeVar("CrudT", bound=CRUD, default=Any, contravariant=True)
CrudT2 = TypeVar("CrudT2", bound=CRUD)

ArgIdxT = TypeVar("ArgIdxT", contravariant=True, bound=AnyIdx, default=Any)
ArgSqlT = TypeVar("ArgSqlT", contravariant=True, bound=SqlExpr | None, default=Any)


class Root(Generic[SchemaT, ArgIdxT, CrudT, ArgSqlT]):
    """Root type of a data instance."""


RootT = TypeVar("RootT", bound=Root, default=Any, covariant=True)
RootT2 = TypeVar("RootT2", bound=Root)
RootT3 = TypeVar("RootT3", bound=Root)


class Base(Root[SchemaT, Any, CrudT, sqla.SelectBase]):
    """Base for retrieving/storing data."""

    @property
    def instance_map(self) -> MutableMapping[Any, Any]:
        """Mapping of FQNs to base-aware instances like FromClauses or Records."""
        ...

    @property
    def connection(self) -> sqla.engine.Connection:
        """SQLAlchemy connection to the database."""
        ...

    def registry[
        T: AutoIndexable
    ](self, value_type: type[T]) -> Registry[T, CrudT, Self]:
        """Get the base data instance for a type in this base."""
        ...


BaseT = TypeVar("BaseT", bound=Base, covariant=True, default=Any)


class Ctx(Generic[ValT, IdxT, CrudT, SqlT]):
    """Node in context path."""


CtxTt = TypeVarTuple("CtxTt", default=Unpack[tuple[Any, ...]])
CtxTt2 = TypeVarTuple("CtxTt2")
CtxTt3 = TypeVarTuple("CtxTt3")


type Input[Val, Sql: SqlExpr | None] = Val | Iterable[Val] | Mapping[
    Hashable, Val
] | pd.DataFrame | pl.DataFrame | Sql | Prop[Val]

Params = ParamSpec("Params")


def _get_prop_type(hint: SingleTypeDef | str) -> type[Prop] | type[None]:
    """Resolve the prop typehint."""
    if has_type(hint, SingleTypeDef):
        base = get_origin(hint)
        if base is None or not issubclass(base, Prop):
            return NoneType

        return base
    elif isinstance(hint, str):
        return _map_data_type_name(hint)
    else:
        return NoneType


@cache
def _data_type_name_map() -> dict[str, type[Prop]]:
    return {cls.__name__: cls for cls in get_subclasses(Prop) if cls is not Prop}


def _map_data_type_name(name: str) -> type[Prop | None]:
    """Map property type name to class."""
    name_map = _data_type_name_map()
    matches = [name_map[n] for n in name_map if name.startswith(n + "[")]
    return matches[0] if len(matches) == 1 else NoneType


@dataclass(kw_only=True)
class Data(Generic[ValT, IdxT, CrudT, SqlT, RootT, *CtxTt]):
    """Property definition for a model."""

    # Core attributes:

    _type: TypeRef[Prop[ValT]] | None = None
    _context: RootT | Data[Any, Any, CrudT, Any, RootT, *tuple[Any, ...]]

    # Extension methods:

    def _name(self) -> str:
        """Name of the property."""
        return gen_str_hash(self)

    def _sql(self) -> SqlT:
        """Get SQL-side reference to this property."""
        assert get_typevar_map(self.resolved_type)[SqlT] is NoneType
        return cast(SqlT, None)

    def _df(
        self,
    ) -> pl.DataFrame | None:
        """Get a dataframe representation of this property's content."""
        return None

    def _value(self, row: Mapping[str, Any]) -> ValT | Literal[Not.handled]:
        """Transform dataframe row to scalar property value."""
        return Not.handled

    def _index(
        self: Data[Any, AnyIdx[*KeyTt2]]
    ) -> (
        Alignment[tuple[*KeyTt2], SelfIdx[*KeyTt2], R, SqlT, Base]
        | Literal[Not.defined]
    ):
        """Get the index of this data."""
        return Not.defined

    def _sql_mutation(
        self: Data[Any, Any, CrudT2, *tuple[Any, ...]],
        input_data: Input[ValT, SqlT],
        mode: set[type[CrudT2]] = {C, U},
    ) -> Sequence[sqla.Executable] | Literal[Not.handled]:
        """Get mutation statements to set this property SQL-side."""
        return Not.handled

    # Type:

    @cached_prop
    def resolved_type(self) -> SingleTypeDef[Prop[ValT]]:
        """Resolved type of this prop."""
        if self._type is None:
            return cast(SingleTypeDef[Prop[ValT]], type(self))

        return hint_to_typedef(
            self._type.hint,
            typevar_map=self._type.var_map,
            ctx_module=self._type.ctx_module,
        )

    @cached_prop
    def value_typeform(self) -> SingleTypeDef[ValT] | UnionType:
        """Target typeform of this prop."""
        return get_typevar_map(self.resolved_type)[ValT]

    @cached_prop
    def value_type_set(self) -> set[type[ValT]]:
        """Target types of this prop (>1 in case of union typeform)."""
        return typedef_to_typeset(self.value_typeform)

    @cached_prop
    def common_value_type(self) -> type:
        """Common base type of the target types."""
        return get_lowest_common_base(
            typedef_to_typeset(
                self.value_typeform,
                remove_null=True,
            )
        )

    @cached_prop
    def typeargs(self) -> dict[TypeVar, SingleTypeDef | UnionType]:
        """Type arguments of this prop."""
        return get_typevar_map(self.resolved_type)

    @staticmethod
    def has_type[T: Data](instance: Data, typedef: type[T]) -> TypeGuard[T]:
        """Check if the dataset has the specified type."""
        orig = get_origin(typedef)
        if orig is None or not issubclass(_get_prop_type(instance.resolved_type), orig):
            return False

        own_typevars = get_typevar_map(instance.resolved_type)
        target_typevars = get_typevar_map(typedef)

        for tv, tv_type in target_typevars.items():
            if tv not in own_typevars:
                return False
            if not is_subtype(own_typevars[tv], tv_type):
                return False

        return True

    # Context:

    def _to_ctx(  # noqa: D107
        self: Data[
            ValT2,
            IdxT2,
            CrudT2,
            SqlT2,
            Root[ValT3, IdxT3, CrudT2, SqlT3],
            *CtxTt2,
        ],
        ctx: Data[ValT3, IdxT3, CrudT2, SqlT3, RootT3, *CtxTt3],
    ) -> Data[ValT2, IdxT2, CrudT2, SqlT2, RootT3, *CtxTt3, *CtxTt2]:
        """Add a context to this property."""
        return copy_and_override(
            Data[ValT2, IdxT2, CrudT2, SqlT2, RootT3, *CtxTt3, *CtxTt2],
            self,
            _context=(
                ctx if isinstance(self._context, Root) else self._context._to_ctx(ctx)
            ),
        )

    @cached_prop
    def root(self) -> RootT:
        """Get the root of this property."""
        if isinstance(self._context, Data):
            return self._context.root

        return self._context

    @cached_prop
    def context(
        self: Data[
            ValT2,
            IdxT2,
            CrudT2,
            SqlT2,
            RootT2,
            *CtxTt2,
            Ctx[ValT3, IdxT3, CrudT2, SqlT3],
        ]
    ) -> Data[ValT3, IdxT3, CrudT2, SqlT3, RootT2, *CtxTt2]:
        """Get the context of this property."""
        assert isinstance(self._context, Data)
        return self._context  # type: ignore

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if not isinstance(self._context, Data):
            return self._name()

        return self._context.fqn + "." + self._name()

    # SQL:

    @property
    def select_str(self) -> str:
        """Return select statement for this dataset."""
        return sqlparse.format(
            str(self.select.compile(self.base.engine)),
            reindent=True,
            keyword_case="upper",
        )

    @cached_prop
    def query(
        self,
    ) -> sqla.Subquery:
        """Return select statement for this dataset."""
        return self.select.subquery()

    # Dataframes:

    @overload
    def df(
        self: Rel[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT],
        sort_by: (
            Literal["index"] | Iterable[Rel[Any, Idx[()], Any, None, Ctx, Symbolic]]
        ) = ...,
        index_only: bool = ...,
        without_index: bool = ...,
        force_fqns: bool = ...,
    ) -> DfT: ...

    @overload
    def df(
        self: Rel[Any, Any, Any, Any, Any, DynBackendID],
        kind: None = ...,
        sort_by: (
            Literal["index"] | Iterable[Rel[Any, Idx[()], Any, None, Ctx, Symbolic]]
        ) = ...,
        index_only: bool = ...,
        without_index: bool = ...,
        force_fqns: bool = ...,
    ) -> pl.DataFrame: ...

    def df(
        self: Rel[Any, Any, Any, Any, Any, DynBackendID],
        kind: type[DfT] | None = None,
        sort_by: (
            Literal["index"] | Iterable[Rel[Any, Idx[()], Any, None, Ctx, Symbolic]]
        ) = "index",
        index_only: bool = False,
        without_index: bool = False,
        force_fqns: bool = False,
    ) -> DfT:
        """Load dataset as dataframe."""
        if self._abs_table is None:
            return cast(DfT, kind() if kind is not None else pl.DataFrame())

        all_cols = {
            **(self._abs_idx_cols if not without_index else {}),
            **(self._abs_cols if not index_only else {}),
        }
        if force_fqns:
            all_cols = {col.fqn: col for col in all_cols.values()}

        select = type(self).select(self, all_cols)

        col_names = cast(
            Mapping[Rel[Any, Any, Any, None, Ctx, DynBackendID], str],
            {col: col_name for col_name, col in all_cols.items()},
        )

        merged_df = None
        if kind is pd.DataFrame:
            with self.base.engine.connect() as con:
                merged_df = pd.read_sql(select, con)

                merged_df = merged_df.set_index(
                    list(self._abs_idx_cols.keys()), drop=True
                )

                merged_df = (
                    merged_df.sort_index()
                    if sort_by == "index"
                    else merged_df.sort_values(
                        by=[col_names[self._abs_table[c]] for c in sort_by]
                    )
                )
        else:
            merged_df = pl.read_database(
                select,
                self.base.df_engine.connect(),
            )

            sort_cols = (
                list(self._abs_idx_cols.keys())
                if sort_by == "index"
                else [col_names[self._abs_table[c]] for c in sort_by]
            )
            merged_df = merged_df.sort(sort_cols)

            if not index_only and not without_index and len(self.record_type_set) == 1:
                rec_type = next(iter(self.record_type_set))
                drop_fqns = {
                    copy_and_override(Var, pk, _ctx=self).fqn
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

    # Items:

    @overload
    def get(
        self: Rel[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
        default: ValT2,
    ) -> ValT | ValT2: ...

    @overload
    def get(
        self: Rel[
            Any,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
        key: tuple[*KeyTt2],
        default: ValT2,
    ) -> ValT | ValT2: ...

    @overload
    def get(
        self: Rel[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
        default: None = ...,
    ) -> ValT | None: ...

    @overload
    def get(
        self: Rel[
            Any,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
        key: tuple[*KeyTt2],
        default: None = ...,
    ) -> ValT | None: ...

    def get(
        self: Rel[Any, Any, Any, Any, Any, DynBackendID],
        key: Hashable | None = None,
        default: Hashable | None = None,
    ) -> Record | Hashable | None:
        """Get a record by key."""
        try:
            return (self[key] if key is not None else self).values()[0]
        except KeyError | IndexError:
            return default

    @overload
    def keys(  # pyright: ignore[reportOverlappingOverload]
        self: Rel[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
    ) -> Sequence[KeyT2]: ...

    @overload
    def keys(
        self: Rel[
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
        self: Rel[
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
        self: Rel[
            Any,
            ExtIdx[Record[*KeyTt2], KeyT3],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Sequence[tuple[*KeyTt2, KeyT3]]: ...

    def keys(
        self: Rel[Any, Any, Any, Any, Any, Any],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        df = self.df(index_only=True)
        if len(self._abs_idx_cols) == 1:
            return [tup[0] for tup in df.iter_rows()]

        return list(df.iter_rows())

    def values(
        self: Rel[ValT2, Any, Any, Any, Any, DynBackendID],
    ) -> Sequence[ValT2]:
        """Iterable over this dataset's values."""
        df = self.df()

        aligned_cols = self._abs_alignment_cols

        rec_types = {
            rec
            for col_set_map in aligned_cols.values()
            for rec in col_set_map.keys()
            if issubclass(rec, Record)
        }
        combi_types = {
            frozenset(type_subset): dynamic_record_type(type_subset, token_hex(5))
            for type_subset in chain.from_iterable(
                combinations(rec_types, r) for r in range(2, len(rec_types) + 1)
            )
        }

        valid_caches = {rec: self.base._get_valid_cache_set(rec) for rec in rec_types}
        instance_maps = {rec: self.base._get_instance_map(rec) for rec in rec_types}

        dfs = [
            df[[col_name for col_set in sel.values() for col_name in col_set.keys()]]
            for sel in aligned_cols.values()
        ]

        vals = []
        for rows in zip(*(df.iter_rows(named=True) for df in dfs)):
            val_list = []
            for (sel, col_set_map), row in zip(aligned_cols.items(), rows):
                if isinstance(sel, Table):
                    recs = {
                        rec_type: rec_type(**rec_dict)
                        for rec_type, col_set in col_set_map.items()
                        if issubclass(rec_type, Record)
                        and rec_type._is_complete_dict(
                            rec_dict := {
                                col.name: row[col_name]
                                for col_name, col in col_set.items()
                            }
                        )
                    }

                    if len(recs) > 1:
                        rec_type = combi_types[frozenset(recs.keys())]
                        new_rec = copy_and_override(
                            rec_type, tuple(r for r in recs.values())
                        )
                    else:
                        rec_type, new_rec = next(iter(recs.items()))

                    if new_rec._index in valid_caches[rec_type]:
                        rec = instance_maps[rec_type][new_rec._index]
                    else:
                        rec = new_rec
                        rec._database = self.base
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
        self: Rel[
            ValT2, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
    ) -> Iterable[tuple[KeyT2, ValT2]]: ...

    @overload
    def items(
        self: Rel[
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
        self: Rel[
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
        self: Rel[
            ValT2,
            ExtIdx[Record[*KeyTt2], KeyT3],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
    ) -> Iterable[tuple[tuple[*KeyTt2, KeyT3], ValT2]]: ...

    def items(
        self: Rel[Any, Any, Any, Any, Any, DynBackendID],
    ) -> Iterable[tuple[Any, Any]]:
        """Iterable over this dataset's items."""
        return list(zip(self.keys(), self.values()))

    def __iter__(  # noqa: D105
        self: Rel[ValT2, Any, Any, Any, Any, DynBackendID],
    ) -> Iterator[ValT2]:
        return iter(self.values())

    def __len__(self: Rel[Any, Any, Any, Any, Any, DynBackendID]) -> int:
        """Get the number of items in the dataset."""
        with self.base.engine.connect() as conn:
            count = conn.execute(
                sqla.select(sqla.func.count()).select_from(self.query)
            ).scalar()
            assert count is not None
            return count

    # Selection and filtering:

    # Comparison:

    @overload
    def __eq__(
        self: Rel[Any, Any, Any, None, Ctx, BaseT2],
        other: Any | Rel[Any, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]: ...

    @overload
    def __eq__(
        self,
        other: Any,
    ) -> bool: ...

    def __eq__(  # noqa: D105
        self: Rel[Any, Any, Any, Any, Any, Any],
        other: Any,
    ) -> sqla.ColumnElement[bool] | bool:
        identical = hash(self) == hash(other)

        if identical or self._context is None:
            return identical

        if isinstance(other, Rel):
            return self._sql_col == other._sql_col

        return self._sql_col == other

    def __neq__(  # noqa: D105
        self: Rel[Any, Any, Any, None, Ctx, BaseT2],
        other: Any | Rel[Any, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Rel):
            return self._sql_col != other._sql_col
        return self._sql_col != other

    def __lt__(  # noqa: D105
        self: Rel[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Rel[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Rel):
            return self._sql_col < other._sql_col
        return self._sql_col < other

    def __le__(  # noqa: D105
        self: Rel[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Rel[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Rel):
            return self._sql_col <= other._sql_col
        return self._sql_col <= other

    def __gt__(  # noqa: D105
        self: Rel[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Rel[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Rel):
            return self._sql_col > other._sql_col
        return self._sql_col > other

    def __ge__(  # noqa: D105
        self: Rel[OrdT, Any, Any, None, Ctx, BaseT2],
        other: OrdT | Rel[OrdT, Any, Any, None, Ctx, BaseT2],
    ) -> sqla.ColumnElement[bool]:
        if isinstance(other, Rel):
            return self._sql_col >= other._sql_col
        return self._sql_col >= other

    def isin(
        self: Rel[Any, Any, Any, None, Ctx, BaseT2], other: Iterable[ValT2] | slice
    ) -> sqla.ColumnElement[bool]:
        """Test values of this dataset for membership in the given iterable."""
        return (
            self._sql_col.between(other.start, other.stop)
            if isinstance(other, slice)
            else self._sql_col.in_(other)
        )

    # Alignment:

    def __matmul__(
        self: Prop[Any, RootT2, CrudT2, Any],
        other: Prop[ValT2, RootT2, CrudT2, IdxT2],
    ) -> Alignment[tuple[ValT, ValT2], RootT2, CrudT2, IdxT | IdxT2]:
        """Align two properties."""
        ...

    # Paths:

    @overload
    def __truediv__(
        self: Data[ValT2, AnyIdx[*KeyTt2], CrudT2, None, Any, RootT2],
        other: (
            Prop[ValT3, AnyIdx[*KeyTt3], CrudT2, Any, BaseT, Ctx[ValT2]]
            | Prop[
                ValT3,
                AnyIdx[*KeyTt3],
                CrudT2,
                Any,
                BaseT,
                Any,
                *tuple[Any, ...],
                Ctx[ValT2],
            ]
        ),
    ) -> Path[ValT3, Idx[*KeyTt2, *KeyTt3], CrudT2, None, BaseT, RootT2]: ...

    @overload
    def __truediv__(
        self: Data[ValT2, AnyIdx[*KeyTt2], CrudT2, sqla.FromClause, Any, RootT2],
        other: (
            Prop[ValT3, AnyIdx[*KeyTt3], CrudT2, SqlT3, BaseT, Ctx[ValT2]]
            | Prop[
                ValT3,
                AnyIdx[*KeyTt3],
                CrudT2,
                SqlT3,
                BaseT,
                Any,
                *tuple[Any, ...],
                Ctx[ValT2],
            ]
        ),
    ) -> Path[ValT3, Idx[*KeyTt2, *KeyTt3], CrudT2, SqlT3, BaseT, RootT2]: ...

    def __truediv__(
        self: Data[ValT2, AnyIdx[*KeyTt2], CrudT2, sqla.FromClause | None, Any, RootT2],
        other: (
            Prop[ValT3, AnyIdx[*KeyTt3], CrudT2, Any, BaseT, Ctx[ValT2]]
            | Prop[
                ValT3,
                AnyIdx[*KeyTt3],
                CrudT2,
                Any,
                BaseT,
                Any,
                *tuple[Any, ...],
                Ctx[ValT2],
            ]
        ),
    ) -> Path[ValT3, Idx[*KeyTt2, *KeyTt3], CrudT2, Any, BaseT, RootT2]:
        """Chain two matching properties together."""
        return Path[ValT3, Idx[*KeyTt2, *KeyTt3], CrudT2, Any, BaseT, RootT2](
            props=(*self.props, other) if isinstance(self, Path) else (self, other)  # type: ignore
        )

    # Index set operations:

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

    def __xor__(
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

    def __lshift__(
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

    # Mutation:

    def stage(
        self: Rel[TabT2 | None, Any, CRUD, Any, Any, BackT2],
        data: pd.DataFrame | pl.DataFrame | sqla.Select,
        fks: (
            Mapping[
                str,
                Var[Any, Any, Public, Any, Symbolic],
            ]
            | None
        ) = None,
    ) -> Table[TabT2, None, R, BaseIdx[TabT2], None, BackT2]:
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        table = (
            self._df_to_table(data) if not isinstance(data, sqla.Select) else data
        ).alias(f"stage/{self.fqn.replace('.', '_')}/{gen_str_hash(data)}")

        return Table(
            _base=self.base,
            _ctx=self._context,
            _name=table.name,
            _type=cast(set[type[TabT2]], self.record_type_set),
            _sql_join=table,
        )

    # Summary:


RegT = TypeVar("RegT", covariant=True, bound=AutoIndexable)


@dataclass(kw_only=True)
class Registry(Data[RegT, AutoIdx[RegT], CrudT, sqla.FromClause, BaseT, None]):
    """Represent a base data type collection."""

    def _name(self) -> str:
        ctx_module = getmodule(self.common_value_type)
        return (
            (
                ctx_module.__name__ + "." + self.common_value_type.__name__
                if ctx_module is not None
                else self.common_value_type.__name__
            )
            + "."
            + self._name()
        )


RuTi = TypeVar("RuTi", bound=R | U, default=R)


@dataclass(kw_only=True)
class Filter(Data[ValTi, IdxTi, RuTi, SqlTi, Root[ValTi, IdxTi, RuTi, SqlTi], *CtxTt]):
    """Property definition for a model."""

    # Attributes:

    _filters: list[
        sqla.ColumnElement[bool]
        | pl.Expr
        | list[tuple[Hashable, ...]]
        | tuple[slice | Hashable, ...]
    ] = field(default_factory=list)

    @cached_prop
    def _sql_filters(
        self,
    ) -> tuple[
        list[sqla.ColumnElement[bool]], list[Table[Any, Any, Any, Any, Any, DbT]]
    ]:
        sql_filt = [f for f in self.__filters if isinstance(f, sqla.ColumnElement)]

        join_set: set[Table[Any, Any, Any, Any, Any, DbT]] = set()
        replace_func = partial(self._parse_sql_filter, join_set=join_set)
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


@dataclass(kw_only=True)
class Prop(Data[ValT, IdxT, CrudT, SqlT, BaseT, RootT, *CtxTt]):
    """Property definition for a model."""

    # Attributes:

    init_after: ClassVar[set[type[Prop]]] = set()

    _owner: type[RootT] | None = None

    alias: str | None = None
    default: ValT | Input[ValT, SqlT] | Literal[Not.defined] = Not.defined
    default_factory: Callable[[], ValT | Input[ValT, SqlT]] | None = None
    init: bool = True
    repr: bool = True
    hash: bool = True
    compare: bool = True

    # Extension methods:

    def _value_get(
        self, instance: RootT
    ) -> ValT | Literal[Not.handled, Not.resolved, Not.defined]:
        """Get the scalar value of this property given an object instance."""
        return Not.handled

    def _value_set(
        self: Prop[Any, Any, C | U], instance: RootT, value: Any
    ) -> None | Literal[Not.handled]:
        """Set the scalar value of this property given an object instance."""
        return Not.handled

    # Ownership:

    def __set_name__(self, owner: type[RootT], name: str) -> None:  # noqa: D105
        if self.alias is None:
            self.alias = name

        if self._owner is None:
            self._owner = owner

        if self._context is None:
            self._context = owner

        if self._type is None:
            self._type = hint_to_typedef(
                get_annotations(owner)[name],
                typevar_map=self.owner_typeargs,
                ctx_module=self.owner_module,
            )

    @cached_prop
    def owner(self: Prop[*tuple[Any, ...], RootT2]) -> type[RootT2]:
        """Module of the owner model type."""
        assert self._owner is not None
        return self._owner

    @cached_prop
    def owner_module(self: Prop[*tuple[Any, ...]]) -> ModuleType | None:
        """Module of the owner model type."""
        return getmodule(self.owner)

    @cached_prop
    def owner_typeargs(
        self: Prop[*tuple[Any, ...]]
    ) -> dict[TypeVar, SingleTypeDef | UnionType]:
        """Type arguments of the owner model type."""
        return get_typevar_map(self.owner)

    # Name:

    @property
    def _name(self) -> str:
        """Name of the property."""
        assert self.alias is not None
        name = self.alias

        if len(self._filters) > 0:
            name += f"[{gen_str_hash(self._filters, length=6)}]"

        return name

    @abstractmethod
    def __set__(  # noqa: D105
        self: Prop[Any, Any, U], instance: Any, value: Any
    ) -> Any:
        return Not.handled

    # Descriptor read/write:

    # @overload
    # def __get__(
    #     self, instance: None, owner: type[CtxT2]
    # ) -> Prop[ValT, CtxT2, CrudT]: ...

    # @overload
    # def __get__(
    #     self: Prop[Any, Any, Any, Idx[()]], instance: CtxT2, owner: type[CtxT2]
    # ) -> ValT: ...

    # @overload
    # def __get__(self, instance: Any, owner: type | None) -> Any: ...

    # @abstractmethod
    # def __get__(self, instance: Any, owner: type | None) -> Any:  # noqa: D105
    #     if owner is None or not issubclass(owner, Model) or instance is None:
    #         return self

    #     return Ungot()


@dataclass
class Path(Data[ValT, IdxT, CrudT, SqlT, BaseT, RootT, *CtxTt]):
    """Alignment of multiple props."""

    props: (
        tuple[Data[ValT, IdxT, CrudT, SqlT, BaseT, RootT],]
        | tuple[
            Data[Any, Any, CrudT, sqla.FromClause | None, BaseT, RootT],
            *tuple[Prop[Any, Any, CrudT, sqla.FromClause | None, BaseT, Any], ...],
            Prop[ValT, Any, CrudT, SqlT, BaseT, Any],
        ]
    )

    @cached_prop
    def _sql_joins(
        self,
        _subtree: JoinDict | None = None,
        _parent: Rel[Record | None, Any, Any, Any, Any] | None = None,
    ) -> list[SqlJoin]:
        """Extract join operations from the relational tree."""
        joins: list[SqlJoin] = []
        _subtree = _subtree if _subtree is not None else self._total_join_dict
        _parent = _parent if _parent is not None else self._root

        if _parent is not None:
            for target, next_subtree in _subtree.items():
                joins.append(
                    (
                        target._sql_alias,
                        reduce(
                            sqla.and_,
                            (
                                (
                                    fk._sql_col == pk._sql_col
                                    for link in target.links
                                    for fk_map in link._abs_fk_maps.values()
                                    for fk, pk in fk_map.items()
                                )
                                if isinstance(target, BackLink)
                                else (
                                    _parent[fk] == target[pk]
                                    for fk_map in target._abs_fk_maps.values()
                                    for fk, pk in fk_map.items()
                                )
                            ),
                        ),
                    )
                )

                joins.extend(type(self)._sql_joins(self, next_subtree, target))

        return joins


TupT = TypeVar("TupT", bound=tuple, covariant=True)


@dataclass
class Alignment(Data[TupT, IdxT, CrudT, SqlT, BaseT, RootT, *CtxTt]):
    """Alignment of multiple props."""

    props: tuple[Prop[Any, Any, CrudT, Any, BaseT, RootT], ...]


RecSqlT = TypeVar("RecSqlT", bound=sqla.CTE | None, covariant=True, default=None)


@dataclass
class Recursion(Data[ValT, Idx[*tuple[Any, ...]], R, RecSqlT, BaseT, RootT, *CtxTt]):
    """Combination of multiple props."""

    paths: tuple[Path[ValT, Any, Any, Any, BaseT, RootT], ...]
