"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from copy import copy
from dataclasses import MISSING, Field, asdict, dataclass, field
from functools import cache, partial, reduce
from inspect import get_annotations, getmodule
from io import BytesIO
from itertools import chain, combinations, groupby, product
from pathlib import Path
from secrets import token_hex
from sqlite3 import PARSE_DECLTYPES
from types import ModuleType, NoneType, UnionType, new_class
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    LiteralString,
    ParamSpec,
    Self,
    TypeGuard,
    TypeVarTuple,
    Union,
    cast,
    dataclass_transform,
    final,
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
from typing_extensions import TypeForm, TypeVar

from py_research.caching import cached_method, cached_prop
from py_research.data import MaskedInit, copy_and_override
from py_research.files import HttpFile
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericProtocol,
    SingleTypeDef,
    get_lowest_common_base,
    get_typevar_map,
    has_type,
    hint_to_typedef,
    is_subtype,
    typedef_to_typeset,
)
from py_research.types import UUID4, Not.changed, Ordinal, Not.defined

from .props import CRUD, RUD, CrudT, Prop, R, ValT
from .sql_tables import Base, Column, DbT, KeyT, Record, RecT, Symbolic
from .utils import pd_to_py_dtype, pl_type_map, sql_to_py_dtype


def _rel_tables(
    rec_type: type[Record], _traversed: set[type[Record]] | None = None
) -> set[type[Record]]:
    direct_rel_types = {
        r
        for t in rec_type._props.values()
        for r in t.value_type_set
        if issubclass(r, Record)
    }

    _traversed = _traversed or set()
    to_traverse = direct_rel_types - _traversed

    return (
        direct_rel_types
        if len(to_traverse) == 0
        else direct_rel_types
        | reduce(
            set.union,
            (_rel_tables(t, _traversed | to_traverse) for t in to_traverse),
        )
    )





type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]


SrcT = TypeVar("SrcT", contravariant=True, bound="Record", default=Any)
TgtT = TypeVar("TgtT", contravariant=True, default=Any)
JdxT = TypeVar("JdxT", contravariant=True, bound=Idx, default=Idx)


@dataclass
class Join(Generic[SrcT, TgtT, JdxT, CrudT]):
    source: type[SrcT]
    target: type[TgtT]

    @property
    def idx_len(self) -> int: ...


RtgT = TypeVar("RtgT", bound="Record | None", contravariant=True, default=Any)


@dataclass
class FkJoin(Join[SrcT, RtgT, Idx[()], RUD]): ...


@dataclass
class RevFkJoin(Join[SrcT, RtgT, Idx[*tuple[Hashable, ...]], CRUD]): ...


@dataclass
class CustomJoin(Join[SrcT, RtgT, JdxT, R]): ...


ParT = TypeVar("ParT", bound="Record", contravariant=True, default=Any)

CtxT = TypeVar(
    "CtxT", covariant=True, bound="Ctx[Any, Any, Any, Any, Any] | None", default=None
)
CtxT2 = TypeVar("CtxT2", bound="Ctx[Any, Any, Any, Any, Any] | None")
CtxT3 = TypeVar("CtxT3", bound="Ctx[Any, Any, Any, Any, Any] | None")


@final
class Ctx(Generic[ParT, IdxT, CrudT, DbT, CtxT]):
    """Context record of a dataset."""


type AnyCtx[R: Record] = Ctx[R, Any, Any, Any, Any]


DfT = TypeVar("DfT", bound=pd.DataFrame | pl.DataFrame)

Params = ParamSpec("Params")


type PropPath[LeafT] = tuple[Rel[LeafT, Any, Any, Any, None]] | tuple[
    Rel[Record | None, Any, Any, Any, None],
    *tuple[Rel[Record | None, Any, Any, Any, Ctx], ...],
    Rel[LeafT, Any, Any, Any, Ctx],
]
type JoinDict = dict[Rel[Record | None, Any, Any, Any, Ctx], JoinDict]

type SqlJoin = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


type AggMap[Rec: Record] = dict[
    Column,
    Column | sqla.Function,
]


@dataclass(kw_only=True, frozen=True)
class Agg(Generic[RecT]):
    """Define an aggregation map."""

    target: type[RecT]
    map: AggMap[RecT]


class Rel(Prop[ValT, Any, CrudT], Generic[ValT, IdxT, CrudT, DbT, CtxT]):
    """Relational dataset."""

    # Internal attributes:

    __target: (set[type[ValT]] | sqla.ColumnElement[ValT] | sqla.Selectable) | None
    __default: ValT | Rel[ValT, Any, Any, DbT, Any] | type[Not.defined] = Not.defined
    __index: Rel[Any, SelfIdx, Any, DbT, Ctx] | None
    __joins: set[Join[Any, ValT, Any, CrudT]] | None
    __crud: CrudT
    __base: Base[Any, DbT]

    __name: str | None
    __type: SingleTypeDef[Rel[ValT]]

    __context: Rel[Record, Any, Any, Any, Any] | None
    __alignments: tuple[Rel[Any, Any, Any, Symbolic, Any]]
    __recursions: Rel[ValT, Any, Any, Symbolic, Ctx] | None
    __filters: list[
        sqla.ColumnElement[bool]
        | list[tuple[Hashable, ...]]
        | tuple[slice | Hashable, ...]
    ]

    __sql_from: sqla_sel.NamedFromClause | None

    # Initialization and copying:

    def __init__(  # noqa: D107
        self: Rel[
            ValT2,
            BaseIdx | SelfIdx | Idx[*KeyTt2, *KeyTt3],
            CrudT2,
            BaseT2,
            Ctx[TabT2, Idx[*KeyTt2], CrudT2, Any, CtxT2] | None,
        ],
        target: (
            set[type[ValT2]]
            | sqla.ColumnElement[ValT2]
            | sqla.Selectable
            | Callable[[Record], ValT2]
        ) | None = None,
        default: ValT2 | Rel[ValT2, Any, Any, DbT, Any] | type[Not.defined] = Not.defined,
        index: Rel[tuple[*KeyTt3], SelfIdx, Any, DbT, Ctx[TabT2]] | None = None,
        joins: set[Join[TabT2, ValT2, Idx[*KeyTt3], CrudT2]] | None = None,
        crud: CrudT2 | None = None,
        base: Base[Any, BaseT2] | BaseT2 | None = None,
    ) -> None:
        self.__target = target
        self.__default = default

        self.__index = index
        if index is None and issubclass(self.common_type, Record):
            self.__index = self.common_type._index

        self.__joins = joins
        self.__crud = crud if crud is not None else cast(CrudT2, CRUD())

        self.__base = (
            base
            if isinstance(base, Base)
            else Base[Any, Any](base) if base is not None else Base[Any, Any]()
        )

        if isinstance(target, sqla.KeyedColumnElement):
            self.__name = target.key

        typedef = (
            getattr(self, "__orig_class__") if hasattr(self, "__orig_class__") else None
        )
        if typedef is None:
            if has_type(target, set[type]):
                typedef = type(self)[  # pyright: ignore[reportInvalidTypeArguments]
                    Union[*target] if len(target) > 1 else next(iter(target))
                ]
            elif has_type(target, set[Join]):
                typedef = type(self)[  # pyright: ignore[reportInvalidTypeArguments]
                    (
                        Union[*(v.target for v in target)]
                        if len(target) > 1
                        else next(iter(target)).target
                    )
                ]
            else:
                assert has_type(target, SingleTypeDef)
                typedef = type(self)[  # pyright: ignore[reportInvalidTypeArguments]
                    cast(TypeForm, target)
                ]
        self.__type = cast(SingleTypeDef, typedef)

        self.__context = None
        self.__alignments = (self.symbol,)
        self.__recursions = None
        self.__filters = []

        self.__sql_from = None

    def __post_init__(self) -> None:
        """Post init."""
        # Call init here to make sure it is called in sub-dataclasses below.
        super().__init__()

    # Representation:

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(
            (
                self.__target,
                self.__index,
                self.__joins,
                self.__crud,
                self.__base,
                self.__name,
                self.__type,
                self.__context,
                self.__alignments,
                self.__recursions,
                self.__filters,
            )
        )

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({repr(self.describe())})"

    # Relational context:

    def __prepend_ctx(
        self,
        left: Rel[Record | None, Any, Any, Any, Any] | None,
    ) -> Self:
        if left is None:
            return self

        if len(self.ctx_path) == 1:
            return self.copy(__context=left)

        root, *path = self.ctx_path
        assert is_subtype(
            Union[*left.record_type_set], Union[*root.record_type_set]
        ), "Context table must be of same type as root table."

        return cast(
            Self,
            reduce(
                lambda x, y: y.copy(__context=x),
                path[:-1],
                left,
            ),
        )

    @property
    def ctx(
        self: Rel[Any, Any, Any, Any, Ctx[TabT2, IdxT2, CrudT2, BaseT2, CtxT2]]
    ) -> Rel[TabT2, IdxT2, CrudT2, BaseT2, CtxT2]:
        """Context table of this dataset."""
        assert self.__context is not None
        return cast(Rel[TabT2, IdxT2, CrudT2, BaseT2, CtxT2], self.__context)

    @cached_prop
    def ctx_path(self) -> PropPath[ValT]:
        """Relational path of this dataset."""
        if self.__context is None:
            return (cast(Rel[ValT, Any, Any, None, Any], self),)

        return cast(PropPath[ValT], (*self.__context.ctx_path, self))

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if self.__context is None:
            if issubclass(self.common_type, Record):
                return "|".join(
                    target_type._fqn for target_type in self.record_type_set
                )
            else:
                return self._name

        fqn = f"{self.__context.fqn}.{self._name}"

        if len(self.__filters) > 0:
            fqn += f"[{gen_str_hash(self.__filters, length=6)}]"

        return fqn

    @property
    def _root(self) -> Rel[Record | None, Any, Any, Any, None] | None:
        first = self.ctx_path[0]
        return (
            first
            if Rel._has_type(first, Rel[Record | None, Any, Any, Any, Any])
            else None
        )

    def _has_ancestor(self, other: Rel[Record | None, Any, Any, Any, Any]) -> bool:
        """Check if ``other`` is ancestor of this data."""
        return other is self.__context or (
            self.__context is not None and self.__context._has_ancestor(other)
        )

    # SQL building blocks:

    @overload
    def _sql_from(  # pyright: ignore[reportOverlappingOverload]
        self: Rel[Record | None, Any, Any, Any, Any]
    ) -> sqla_sel.NamedFromClause: ...

    @overload
    def _sql_from(self: Rel[Any, Any, Any, Any, Any]) -> None: ...

    def _sql_from(
        self: Rel[Any, Any, Any, Any, Any]
    ) -> sqla_sel.NamedFromClause | None:
        """Recursively join all bases of this record to get the full data."""
        if not issubclass(self.common_value_type, Record):
            return None

        if self.__sql_from is not None:
            return self.__sql_from

        if isinstance(self.__target, sqla_sel.NamedFromClause):
            return self.__target

        base_tables: dict[type[Record], sqla.Table] = {}
        super_tables: dict[type[Record], sqla.Table] = {}

        for rec in self.record_type_set:
            base_table, *super_tabs = list(self.base._get_sql_base_tables(rec).items())
            base_tables[base_table[0]] = base_table[1]
            super_tables.update(dict(super_tabs))

        def _union(
            union_stage: tuple[type[Record] | None, sqla.Select],
            next_union: tuple[type[Record], sqla.Join | sqla.Table],
        ) -> tuple[type[Record] | None, sqla.Select]:
            left_rec, left_table = union_stage
            right_rec, right_table = next_union

            right_table = right_table.select().with_only_columns(
                *(
                    right_table.c[col.name].label(col.fqn)
                    for col in right_rec._col_values.values()
                )
            )

            if (
                left_rec is not None
                and len(left_rec._pk_values) == len(right_rec._pk_values)
                and all(
                    hash(pk1) == hash(pk2)
                    for pk1, pk2 in zip(
                        left_rec._pk_values.values(), right_rec._pk_values.values()
                    )
                )
            ):
                return left_rec, left_table.outerjoin(
                    right_table.alias(),
                    reduce(
                        sqla.and_,
                        (
                            left_table.c[pk1] == right_table.c[pk2]
                            for pk1, pk2 in left_rec._pk_values
                        ),
                    ),
                )
            else:
                return None, left_table.union().select()

        if len(base_tables) > 1:
            rec_query_items = list(base_tables.items())
            first_rec, first_select = rec_query_items[0]
            first: tuple[type[Record] | None, sqla.Select] = (
                first_rec,
                first_select.select().with_only_columns(
                    *(
                        first_select.c[col.name].label(col.fqn)
                        for col in first_rec._col_values.values()
                    )
                ),
            )

            _, base_union = reduce(_union, rec_query_items[1:], first)
        else:
            base_union = list(base_tables.values())[0]

        base_join = base_union
        for rec, super_table in super_tables.items():
            base_join = base_join.join(
                super_table,
                reduce(
                    sqla.and_,
                    (
                        base_union.c[pk.fqn if len(base_tables) > 1 else pk.name]
                        == super_table.c[pk.name]
                        for pk in rec._pk_values.values()
                    ),
                ),
            )

        return base_join.alias(self.fqn)

    @overload
    def _sql_col(  # pyright: ignore[reportOverlappingOverload]
        self: Rel[Record | None | tuple, Any, Any, Any, Any]
    ) -> None: ...

    @overload
    def _sql_col(self: Rel[Any, Any, Any, Any, Any]) -> sqla.ColumnElement: ...

    @cached_method
    def _sql_col(self: Rel[Any, Any, Any, Any, Any]) -> sqla.ColumnElement | None:
        """Return the SQL column of this dataset."""
        if isinstance(self.__target, sqla.ColumnElement):
            if self.__target.table is None:
                self.__target.table = self._sql_alias()

            return self.__target

        if issubclass(self.common_type, Record | None | tuple):
            return None

        col = sqla.column(
            (
                self.fqn
                if self.__context is not None
                and len(self.__context.record_type_set) > 1
                else self._name
            ),
            _selectable=(
                self.__context._sql_join() if self.__context is not None else None
            ),
        )
        setattr(col, "_data", self)

        return col

    # Normalized query construction:

    @property
    def _abs_table(self) -> Rel[Record | None, Any, Any, Any, Any] | None:
        if Rel._has_type(self, Rel[Record | None, Any, Any, Any, Any]):
            return self

        return self.__context

    @property
    def _abs_value_col_sets(
        self,
    ) -> dict[type, dict[str, Rel[Any, Any, Any, DbT, Any]]]:
        return (
            {
                rec: {
                    (
                        abs_col := cast(
                            Rel[Any, Any, Any, DbT, Any],
                            col.copy(
                                __base=self.base,
                                __context=self._abs_table,
                            ),
                        )
                    ).fqn: abs_col
                    for col in rec._col_values.values()
                }
                for rec in self.record_type_set
            }
            if len(self.record_type_set) > 0
            else {self.common_type: {self.fqn: self}}
        )

    @cached_prop
    def _abs_alignment_cols(
        self,
    ) -> dict[
        Rel[Any, Any, Any, DbT, Any],
        dict[type[ValT], dict[str, Var[Any, Any, DbT, Any]]],
    ]:
        return {
            abs_sel: {
                t: {col.fqn: col for col in col_set.values()}
                for t, col_set in abs_sel._abs_alignment_cols[abs_sel].items()
            }
            for abs_sel in [
                (
                    sel.__prepend_ctx(self._abs_table)
                    if self._abs_table is not None
                    else sel.copy(__base=self.base)
                )
                for sel in self.__alignments
            ]
        }

    @cached_prop
    def _abs_cols(self) -> dict[str, Var[Any, Any, Any, Any, DbT]]:
        return {
            col_name: col
            for sel in self._abs_alignment_cols.values()
            for col_set in sel.values()
            for col_name, col in col_set.items()
        }

    @cached_prop
    def _abs_idx_cols(self) -> dict[str, Var[Any, Any, Any, Any, DbT]]:
        abs_idx_cols = {}
        path_dict: dict[
            Rel[Record | None, Any, Any, Any, Any, Any],
            Rel[Any, Any, Any, Any, Any, Any],
        ] = {
            **({self._root: self._root} if self._root is not None else {}),
            **self._abs_link_path,
        }

        for link_node, node in path_dict.items():
            if isinstance(link_node, Link) and issubclass(node.relation_type, Relation):
                abs_idx_cols |= {
                    copy_and_override(
                        Var,
                        pk,
                        _ctx=node,
                    ).fqn: link_node[pk]
                    for target_type, fk_map in link_node._abs_fk_maps.items()
                    if is_subtype(Union[*node.record_type_set], target_type)
                    for pk in fk_map.values()
                }
            elif isinstance(link_node, Table) and not issubclass(
                node.relation_type, Relation
            ):
                cols = []
                rec_type = link_node.common_type
                assert issubclass(rec_type, Record)

                if link_node.index_by is not None:
                    cols = (
                        [link_node.index_by]
                        if isinstance(link_node.index_by, Var)
                        else list(link_node.index_by)
                    )
                elif issubclass(rec_type, Item):
                    cols = [rec_type.idx]  # type: ignore
                elif issubclass(rec_type, RelIndex):
                    cols = [rec_type._rel_idx]  # type: ignore
                else:
                    cols = rec_type._pk_values.values()

                abs_cols = [
                    copy_and_override(
                        Var[Any, Any, Any, Any, DbT],
                        col,
                        _base=self.base,
                        _ctx=link_node,
                    )
                    for col in cols
                ]
                abs_idx_cols |= {col.fqn: col for col in abs_cols}

        return abs_idx_cols

    @cached_prop
    def _abs_filters(
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
        ], mergea

    @cached_prop
    def _abs_joins(self) -> list[Table[Record | None, Any, Any, Any, Any, DbT]]:
        return [
            sel._abs_table
            for sel in self._abs_alignment_cols
            if sel._abs_table is not None
        ]

    @property
    def _total_joins(self) -> list[Table[Record | None, Any, Any, Any, Any, DbT]]:
        return self._abs_joins + self._abs_filters[1]

    @property
    def _total_root_tables(
        self,
    ) -> set[Table[Record | None, None, Any, Any, None, DbT]]:
        return {t for t in (self._root, *self._abs_joins) if t is not None}

    @cached_prop
    def _total_join_dict(self) -> JoinDict:
        tree: JoinDict = {}

        for rel in self._total_joins:
            subtree = tree
            for node in rel._abs_link_path.keys():
                if node not in subtree:
                    subtree[node] = {}
                subtree = subtree[node]

        return tree

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

    @cached_prop
    def _sql_filters(self) -> list[sqla.ColumnElement[bool]]:
        """Get the SQL filters for this table."""
        if not isinstance(self, Table):
            return []

        return self._abs_filters[0] + (
            self._ctx_table._sql_filters if self._ctx_table is not None else []
        )

    @cached_prop
    def select(
        self,
        cols: Mapping[str, Var[Any, Any, Any, Any]] | None = None,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        if cols is not None:
            abs_cols = {
                name: (
                    col.__prepend_ctx(self._abs_table)
                    if isinstance(col.base.backend, Symbolic)
                    else col
                )
                for name, col in cols.items()
            }
        else:
            abs_cols = self._abs_idx_cols | self._abs_cols

        sql_cols: list[sqla.ColumnElement] = []
        for col_name, col in abs_cols.items():
            sql_col = col._sql_col.label(col_name)
            if sql_col not in sql_cols:
                sql_cols.append(sql_col)

        select = sqla.select(*sql_cols)

        if len(self._total_root_tables) > 0:
            select = select.select_from(*(t._sql_join for t in self._total_root_tables))

        for join in self._sql_joins:
            select = select.join(*join)

        for filt in self._sql_filters:
            select = select.where(filt)

        return select.distinct()

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

    # Content inspection and loading:

    def describe(self) -> dict[str, str | dict[str, str] | None]:
        """Return a description of this database."""
        desc: dict[str, Any] = {}

        base_desc: dict[str, Any] = {"id": self.base.db_id}
        if self.base.url is not None:
            base_desc["url"] = str(self.base.url)
        if isinstance(self.base.schema, type) and issubclass(self.base.schema, Schema):
            base_desc["schema"] = {
                k: v
                for k, v in asdict(PyObjectRef.reference(self.common_type))
                if k != "object_type"
            }
        desc["base"] = base_desc

        desc["type"] = (
            {
                k: v
                for k, v in asdict(PyObjectRef.reference(self.common_type))
                if k != "object_type"
            }
            if len(self.target_type_set) == 1 and issubclass(self.common_type, Record)
            else str(self._resolved_typehint)
        )

        if not isinstance(self.base.backend, Symbolic):
            dyn_self = cast(Rel[Any, Any, Any, Any, Any, DynBackendID], self)

            desc["contents"] = {
                "records": {
                    rec._fqn: len(dyn_self[rec])
                    for rec in type(self)._rel_record_types(self, with_relations=False)
                },
                "arrays": {
                    rec._fqn: len(dyn_self[rec])
                    for rec in self._rel_record_types
                    if issubclass(rec, Item)
                },
                "relations": {
                    rec._fqn: len(dyn_self[rec])
                    for rec in self._rel_record_types
                    if issubclass(rec, Relation)
                },
            }

        return desc

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
                    val_list.append(row[sel._name])

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

    def graph(
        self: Rel[Schema | Record, Any, Any, Any, Any, DynBackendID],
        nodes: (
            Sequence[type[Record] | Rel[Record, BaseIdx, Any, None, None, Symbolic]]
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
            else list(type(self)._rel_record_types(self, with_relations=False))
        )
        node_tables = [db[n] for n in nodes]
        node_types = {
            t
            for n in nodes
            for t in ([n] if isinstance(n, type) else n.record_type_set)
        }
        edge_types = {t for t in db._rel_record_types if issubclass(t, Relation)}

        # Concat all node tables into one.
        node_dfs = [
            n.df(kind=pd.DataFrame, force_fqns=True).reset_index().assign(table=n.fqn)
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
                    Link[Record, Any, Any, Symbolic],
                    Link[Record, Any, Any, Symbolic],
                ]
            ],
        ] = {t: set() for t in edge_types}

        for rel in edge_types:
            left, right = cast(
                tuple[
                    Link[Record, Any, Any, Symbolic],
                    Link[Record, Any, Any, Symbolic],
                ],
                (rel._from, rel._to),  # type: ignore
            )
            if is_subtype(
                Union[*node_types], Union[*left.record_type_set]
            ) and is_subtype(Union[*node_types], Union[*right.record_type_set]):
                undirected_edges[rel].add((left, right))

        direct_edge_dfs = [
            node_df.loc[node_df["table"] == str(parent._default_table_name())]
            .rename(columns={"node_id": "source"})
            .merge(
                node_df.loc[node_df["table"] == str(rec_type._default_table_name())],
                left_on=[c.fqn for c in fk_map.keys()],
                right_on=[c.fqn for c in fk_map.values()],
            )
            .rename(columns={"node_id": "target"})[["source", "target"]]
            .assign(
                ltr=",".join(c.name for c in fk_map.keys()),
                rtl=None,
            )
            for parent, link in directed_edges
            for rec_type, fk_map in link._abs_fk_maps.items()
        ]

        rel_edge_dfs = []
        for assoc_table, rels in undirected_edges.items():
            for left, right in rels:
                for left_target, right_target in product(
                    left.record_type_set, right.record_type_set
                ):
                    left_fk_map = left._abs_fk_maps[left_target]
                    right_fk_map = right._abs_fk_maps[right_target]

                    rel_df = (
                        self[assoc_table]
                        .df(kind=pd.DataFrame, force_fqns=True)
                        .reset_index()
                    )

                    left_merged = rel_df.merge(
                        node_df.loc[
                            node_df["table"] == str(left_target._default_table_name())
                        ][[*(c.fqn for c in left_fk_map.values()), "node_id"]],
                        left_on=[c.fqn for c in left_fk_map.keys()],
                        right_on=[c.fqn for c in left_fk_map.values()],
                        how="inner",
                    ).rename(columns={"node_id": "source"})

                    both_merged = left_merged.merge(
                        node_df.loc[
                            node_df["table"] == str(right_target._default_table_name())
                        ][[*(c.fqn for c in right_fk_map.values()), "node_id"]],
                        left_on=[c.fqn for c in right_fk_map.keys()],
                        right_on=[c.fqn for c in right_fk_map.values()],
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
                            ltr=",".join(c.name for c in right_fk_map.keys()),
                            rtl=",".join(c.name for c in left_fk_map.keys()),
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

    # Relational selection and filtering:

    @cached_prop
    def symbol(self) -> Rel[ValT, IdxT, CrudT, Symbolic, CtxT]:
        """Symbolic representation of this dataset."""
        return cast(
            Rel[ValT, IdxT, CrudT, Symbolic, CtxT],
            (
                reduce(
                    lambda ctx, data: data.copy(__context=ctx, base=symbol_base),
                    self.ctx_path,
                )
                if len(self.ctx_path) > 1
                else self.copy(base=symbol_base)
            ),
        )

    @cached_prop
    def x(self: Rel[TabT2 | None, Any, Any, Any, Any]) -> type[TabT2]:
        """Path-aware accessor to the record type of this dataset."""
        return cast(
            type[TabT2],
            dynamic_record_type(
                Record,
                f"{self._name}.x",
                props=reduce(
                    set.intersection,
                    (set(t._props.values()) for t in self.record_type_set),
                ),
                extra_attrs={
                    "_ctx": self,
                    "_derivate": True,
                },
            ),
        )

    # 1. DB-level type selection
    @overload
    def __getitem__(
        self: Rel[Schema | Record | None, RootIdx, Any, None, None, Any],
        key: type[TabT3],
    ) -> Rel[TabT3, BaseIdx[TabT3], CrudT, None, None, DbT]: ...

    # 2. DB-level nested prop selection
    @overload
    def __getitem__(
        self: Rel[Schema | Record | None, RootIdx, Any, None, None, Any],
        key: Rel[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Rel[
        ValT3,
        Idx[*tuple[Any, ...], Any, *KeyTt3] | Idx[*KeyTt3],
        CrudT3,
        RelT3,
        CtxT3,
        DbT,
    ]: ...

    # 3. DB-level nested array selection
    @overload
    def __getitem__(
        self: Rel[Schema | Record | None, RootIdx, Any, None, None, Any],
        key: Rel[
            ValT3,
            ExtIdx[Record[*KeyTt3], *KeyTt4],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Rel[
        ValT3,
        Idx[*tuple[Any, ...], *KeyTt3, *KeyTt4],
        CrudT3,
        RelT3,
        CtxT3,
        DbT,
    ]: ...

    # 4. Top-level prop selection
    @overload
    def __getitem__(
        self: Rel[
            TabT2 | None,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            Any,
        ],
        key: Rel[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            RelT3,
            Ctx[ValT],
            Symbolic,
        ],
    ) -> Rel[ValT3, Idx[*KeyTt2, *KeyTt3], CrudT3, RelT3, Ctx[TabT2], DbT]: ...

    # 5. Top-level array selection
    @overload
    def __getitem__(
        self: Rel[
            TabT2 | None,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            Any,
        ],
        key: Rel[
            ValT3,
            ExtIdx[Record[*KeyTt3], *KeyTt4],
            CrudT3,
            RelT3,
            Ctx[TabT2],
            Symbolic,
        ],
    ) -> Rel[ValT3, Idx[*KeyTt2, *KeyTt3, *KeyTt4], CrudT3, RelT3, Ctx[TabT2], DbT]: ...

    # 6. Nested prop selection
    @overload
    def __getitem__(
        self: Rel[
            Record | None,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            Any,
        ],
        key: Rel[
            ValT3,
            BaseIdx[Record[*KeyTt3]] | Idx[*KeyTt3],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Rel[
        ValT3,
        Idx[*KeyTt2, *tuple[Any, ...], *KeyTt3],
        CrudT3,
        RelT3,
        CtxT3,
        DbT,
    ]: ...

    # 7. Nested array selection
    @overload
    def __getitem__(
        self: Rel[
            Record | None,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            Any,
        ],
        key: Rel[
            ValT3,
            ExtIdx[Record[*KeyTt3], *KeyTt4],
            CrudT3,
            RelT3,
            CtxT3,
            Symbolic,
        ],
    ) -> Rel[
        ValT3,
        Idx[*KeyTt2, *tuple[Any, ...], *KeyTt3, *KeyTt4],
        CrudT3,
        RelT3,
        CtxT3,
        DbT,
    ]: ...

    # 8. Key filtering, scalar index type
    @overload
    def __getitem__(
        self: Rel[Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Any],
        key: list[KeyT2] | slice,
    ) -> Rel[ValT, IdxT, RU, RelT, CtxT, DbT]: ...

    # 9. Key / slice filtering
    @overload
    def __getitem__(
        self: Rel[Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Any],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Rel[ValT, IdxT, RU, RelT, CtxT, DbT]: ...

    # 10. Key / slice filtering, array index
    @overload
    def __getitem__(
        self: Rel[Any, ExtIdx[Record[*KeyTt2], KeyT3], Any, Any, Any, Any],
        key: list[tuple[*KeyTt2, KeyT3]] | tuple[slice, ...],
    ) -> Rel[ValT, IdxT, RU, RelT, CtxT, DbT]: ...

    # 11. Expression filtering
    @overload
    def __getitem__(
        self: Rel[Any, Idx | BaseIdx, Any, Any, Any, Any],
        key: sqla.ColumnElement[bool],
    ) -> Rel[ValT, IdxT, RU, RelT, CtxT, DbT]: ...

    # 12. Key selection, scalar index type, symbolic context
    @overload
    def __getitem__(
        self: Rel[Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Symbolic],
        key: KeyT2,
    ) -> Rel[ValT, IdxT, RU, RelT, CtxT, DbT]: ...

    # 13. Key selection, tuple index type, symbolic context
    @overload
    def __getitem__(
        self: Rel[
            Any, BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2], Any, Any, Any, Symbolic
        ],
        key: tuple[*KeyTt2],
    ) -> Rel[ValT, IdxT, RU, RelT, CtxT, DbT]: ...

    # 14. Key selection, tuple index type, symbolic context, array index
    @overload
    def __getitem__(
        self: Rel[Any, ExtIdx[Record[*KeyTt2], KeyT3], Any, Any, Any, Symbolic],
        key: tuple[*KeyTt2, KeyT3],
    ) -> Rel[ValT, IdxT, RU, RelT, CtxT, DbT]: ...

    # 15. Key selection, scalar index type
    @overload
    def __getitem__(
        self: Rel[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, DynBackendID
        ],
        key: KeyT2,
    ) -> ValT: ...

    # 16. Key selection, tuple index type
    @overload
    def __getitem__(
        self: Rel[
            Any,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            Any,
            Any,
            Any,
            DynBackendID,
        ],
        key: tuple[*KeyTt2],
    ) -> ValT: ...

    # 17. Key selection, tuple index type, array index
    @overload
    def __getitem__(
        self: Rel[Any, ExtIdx[Record[*KeyTt2], KeyT3], Any, Any, Any, DynBackendID],
        key: tuple[*KeyTt2, KeyT3],
    ) -> ValT: ...

    def __getitem__(
        self: Rel[Any, Any, Any, Any, Any, Any],
        key: (
            type[Record]
            | Rel[Any, Any, Any, Any, Any, Symbolic]
            | sqla.ColumnElement[bool]
            | list
            | slice
            | tuple[slice, ...]
            | Hashable
        ),
    ) -> Rel[Any, Any, Any, Any, Any, Any] | ValT:
        """Select into the relational subgraph or filte ron the current level."""
        match key:
            case type() | UnionType():
                return Table(_base=self.base, _type=key)
            case Rel():
                return (
                    copy_and_override(type(key), key, _base=self)
                    if isinstance(self, DataBase)
                    else key.__prepend_ctx(self._abs_table)
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
                    and not isinstance(self.base.backend, Symbolic)
                ):
                    try:
                        return list(iter(key_set))[0]
                    except IndexError as e:
                        raise KeyError(key) from e

                return key_set

    @overload
    def __get__(
        self: Rel[Any, Any, Any, RelIndex[Any, Any, tuple[*KeyTt2]], Any, Any],
        instance: None,
        owner: type[TabT4],
    ) -> Rel[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[TabT4], Symbolic]: ...

    @overload
    def __get__(
        self: Rel[Any, Any, Any, RelIndex[Any, Any, KeyT2], Any, Any],
        instance: None,
        owner: type[TabT4],
    ) -> Rel[ValT, Idx[KeyT2], CrudT, RelT, Ctx[TabT4], Symbolic]: ...

    @overload
    def __get__(
        self: Rel[Record[*KeyTt2], BaseIdx, Any, Any, Any, Any],
        instance: None,
        owner: type[TabT4],
    ) -> Rel[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[TabT4], Symbolic]: ...

    @overload
    def __get__(
        self: Rel[Any, ExtIdx[Any, KeyT2], Any, Any, Any, Any],
        instance: None,
        owner: type[TabT4],
    ) -> Rel[ValT, ExtIdx[TabT4, KeyT2], CrudT, RelT, Ctx[TabT4], Symbolic]: ...

    @overload
    def __get__(
        self: Rel[Any, Any, Any, Any, Any, Any],
        instance: None,
        owner: type[TabT4],
    ) -> Rel[ValT, IdxT, CrudT, RelT, Ctx[TabT4], Symbolic]: ...

    @overload
    def __get__(
        self: Rel[Any, Idx[()], Any, Any, Any, Any],
        instance: Record,
        owner: type[Record],
    ) -> ValT: ...

    @overload
    def __get__(
        self: Rel[Any, Any, Any, RelIndex[Any, Any, tuple[*KeyTt2]], Any, Any],
        instance: TabT4,
        owner: type[TabT4],
    ) -> Rel[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[TabT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Rel[Any, Any, Any, RelIndex[Any, Any, KeyT2], Any, Any],
        instance: TabT4,
        owner: type[TabT4],
    ) -> Rel[ValT, Idx[KeyT2], CrudT, RelT, Ctx[TabT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Rel[Record[*KeyTt2], BaseIdx, Any, Any, Any, Any],
        instance: TabT4,
        owner: type[TabT4],
    ) -> Rel[ValT, Idx[*KeyTt2], CrudT, RelT, Ctx[TabT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Rel[Any, ExtIdx[Any, KeyT2], Any, Any, Any, Any],
        instance: TabT4,
        owner: type[TabT4],
    ) -> Rel[ValT, ExtIdx[TabT4, KeyT2], CrudT, RelT, Ctx[TabT4], DynBackendID]: ...

    @overload
    def __get__(
        self: Rel[Any, Any, Any, Any, Any, Any],
        instance: TabT4,
        owner: type[TabT4],
    ) -> Rel[ValT, IdxT, CrudT, RelT, Ctx[TabT4], DynBackendID]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(
        self: Rel[Any, Any, Any, Any, Ctx | None, Any],
        instance: object | None,
        owner: type | None,
    ) -> Rel[Any, Any, Any, Any, Any, Any] | ValT:
        """Get the value of this dataset when used as property."""
        owner = (
            self._ctx_table.common_type if isinstance(self._ctx_table, Rel) else owner
        )

        if owner is not None and issubclass(owner, Record):
            if isinstance(instance, owner):
                if isinstance(self, Var):
                    if (
                        self.pub_status is Public
                        and instance._published
                        and instance._index
                        not in instance._base._get_valid_cache_set(owner)
                    ):
                        instance._update_dict()

                    value = (
                        self.getter(instance)
                        if self.getter is not None
                        else instance.__dict__.get(self._name, Not.defined)
                    )

                    if value is Not.defined:
                        if self.default_factory is not None:
                            value = self.default_factory()
                        else:
                            value = self.default

                        assert (
                            value is not Not.defined
                        ), f"Property value for `{self._name}` could not be fetched."
                        setattr(instance, self._name, value)

                    return value
                else:
                    self_ref = cast(
                        Rel[ValT, Idx[()], CrudT, RelT, Ctx, Symbolic],
                        getattr(owner, self._name),
                    )
                    table = Table(_base=instance._base, _type=owner)
                    return table[self_ref][instance._index]
            else:
                self.copy()

        return self

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

    # Transformation:

    @overload
    def __matmul__(
        self: Rel[tuple, Any, Any, Any, Any, Any],
        other: Rel[tuple, Any, Any, Any, Any, Any],
    ) -> Rel[tuple, IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __matmul__(
        self: Rel[ValT2, Any, Any, Any, Any, Any],
        other: Rel[tuple[*ValTt3], Any, Any, Any, Any, Any],
    ) -> Rel[tuple[ValT2, *ValTt3], IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __matmul__(
        self: Rel[tuple[*ValTt2], Any, Any, Any, Any, Any],
        other: Rel[ValT3, Any, Any, Any, Any, Any],
    ) -> Rel[tuple[*ValTt2, ValT3], IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __matmul__(
        self: Rel[ValT2, Any, Any, Any, Any, Any],
        other: Rel[ValT3, Any, Any, Any, Any, Any],
    ) -> Rel[tuple[ValT2, ValT3], IdxT, CrudT, RelT, CtxT, DbT]: ...

    def __matmul__(
        self,
        other: Rel[ValT2, Any, Any, Any, Any, DbT],
    ) -> Rel[tuple, IdxT, CrudT, RelT, CtxT, DbT]:
        """Align and merge this dataset with another into a tuple-valued dataset."""
        return copy_and_override(
            Rel[tuple, IdxT, CrudT, RelT, CtxT, DbT],
            self,
            _alignments=(
                *(self._alignments if self._alignments is not None else [self.symbol]),
                *(
                    other._alignments
                    if other._alignments is not None
                    else [other.symbol]
                ),
            ),
        )

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

    @overload
    def extract(
        self: Rel[TabT2 | None, Any, Any, Any, Any, BackT2],
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
    ) -> DataBase[TabT2, Any, CRUD, BackT2]: ...

    @overload
    def extract(
        self: Rel[Any, Any, Any, Any, Any, BackT2],
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
        self: Rel[Record | None, Any, R, Any, Any, DynBackendID],
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
        self: Rel[Any, Any, Any, Any, Any, BackT2],
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
            assert self._abs_table is not None
            rec_types = self._abs_table.record_type._rel_types()

            overlay_db = copy_and_override(
                DataBase[Any, Any, CRUD, BackT2 | BackT3],
                self.base,
                backend=self.base.backend,
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
                Rel[Record | None, Any, Any, Any, Ctx, DynBackendID],
                Rel[Record | None, Any, Any, Any, Ctx, DynBackendID],
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

                select = Rel[Any, Any, Any, Any, Any, Any].select(
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

    # Mutation interface:

    def _gen_upload_table(
        self,
        include: list[str] | None = None,
    ) -> sqla.Table:
        assert self._abs_table is not None

        type_map = reduce(
            lambda x, y: x | y,
            (rec._type_map for rec in self._abs_table.record_type_set),
        )

        metadata = sqla.MetaData()
        registry = orm.registry(
            metadata=self.base._metadata,
            type_annotation_map=type_map,
        )

        cols = [
            sqla.Column(
                col_name,
                registry._resolve_type(
                    col.target_typeform  # pyright: ignore[reportArgumentType]
                ),
                primary_key=col.primary_key,
                autoincrement=False,
                index=col.index,
                nullable=has_type(None, col.target_typeform),
            )
            for col_name, col in (self._abs_cols | self._abs_idx_cols).items()
            if include is None or col_name in include
        ]

        table_name = f"upload/{self.fqn.replace('.', '_')}/{token_hex(5)}"
        table = sqla.Table(
            table_name,
            metadata,
            *cols,
        )

        return table

    def _mutate(  # noqa: C901
        self: Rel[ValT2, Any, CRUD, Any, Any, DynBackendID],
        value: Rel[ValT2, Any, Any, Any, Any, Any] | Input[ValT2, Hashable],
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        assert self._abs_table is not None

        record_ids: dict[Hashable, Hashable] | None = None
        valid_caches = {
            rec: self.base._get_valid_cache_set(rec)
            for rec in self._abs_table._all_record_base_types
        }

        match value:
            case sqla.Select():
                assert self._abs_table is not None
                self._abs_table._mutate_from_query(value.subquery(), mode)
                for cache_set in valid_caches.values():
                    cache_set.clear()

            case Rel():
                if hash(value.base) == hash(self.base) and self._abs_table is not None:
                    # Other database is exactly the same,
                    # so updating target table is enough.
                    self._abs_table._mutate_from_query(value.query, mode)
                elif issubclass(value.common_type, Record):
                    # Other database is not exactly the same,
                    # hence may be on a different overlay or backend.
                    # Related records have to be mutated alongside.
                    other_db = value.extract()

                    if self.base.db_id == other_db.db_id:
                        for rec in other_db._rel_record_types:
                            Table(_base=self.base, _type=rec)._mutate_from_query(
                                other_db[rec].query, "upsert"
                            )
                    else:
                        for rec in other_db._rel_record_types:
                            value_table = other_db[rec]._df_to_table(
                                Table(_base=other_db, _type=rec).df(),
                            )
                            Table(_base=self.base, _type=rec)._mutate_from_query(
                                value_table,
                                "upsert",
                            )
                            value_table.drop(self.base.engine)
                elif self._abs_table is not None and value._abs_table is not None:
                    # Not a record type, so no need to mutate related records.
                    value_table = self._df_to_table(value.df())
                    self._abs_table._mutate_from_query(value_table, mode)
                    value_table.drop(self.base.engine)

                for rec_type in value._all_record_base_types:
                    if rec_type in valid_caches:
                        valid_caches[rec_type].clear()

            case pd.DataFrame() | pl.DataFrame():
                assert self._abs_table is not None

                value_table = self._df_to_table(value)
                self._abs_table._mutate_from_query(
                    value_table,
                    mode,
                )
                value_table.drop(self.base.engine)

                for cache_set in valid_caches.values():
                    cache_set.clear()

            case Record():
                assert isinstance(self, Table)
                self._mutate_from_records({value._index: value}, mode)
                valid_caches[type(value)].remove(value._index)

            case Iterable():
                if not issubclass(self.common_type, Record):
                    assert isinstance(
                        value, Mapping
                    ), "Inserting via values requires a mapping."
                    self._mutate_from_values(value, mode)
                    for cache_set in valid_caches.values():
                        cache_set.clear()
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

                    assert isinstance(self, Table)
                    self._mutate_from_records(
                        records,
                        mode,
                    )

                    for rec in records.values():
                        valid_caches[type(rec)].remove(rec._index)

                    if len(record_ids) > 0:
                        self._mutate_rels_from_pks(record_ids, mode)
                        valid_caches[self.common_type] -= set(record_ids.values())

            case _:
                if not issubclass(self.common_type, Record):
                    self._mutate_from_values({None: cast(ValT2, value)}, mode)

                    for cache_set in valid_caches.values():
                        cache_set.clear()
                else:
                    assert isinstance(self, Table)
                    self._mutate_rels_from_pks({value: value}, mode)

                    valid_caches[self.common_type] -= {value}

        return

    def _mutate_from_values(
        self: Rel[ValT2, Any, CRUD, Any, Any, DynBackendID],
        values: Mapping[Any, ValT2],
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        assert self._abs_table is not None

        df = self._values_to_df(values)
        value_table = self._df_to_table(df)
        self._abs_table._mutate_from_query(
            value_table,
            mode,
        )
        value_table.drop(self.base.engine)
        return

    def _mutate_from_models(
        self,
        values: Mapping[Any, Record],
        mode: MutationMode = "update",
    ) -> list[sqla.Executable]:
        db_grouped = {
            db: dict(recs)
            for db, recs in groupby(
                sorted(
                    values.items(), key=lambda x: x[1]._published and x[1]._base.db_id
                ),
                lambda x: None if not x[1]._published else x[1]._base,
            )
        }

        unconnected_records = db_grouped.get(None, {})
        local_records = db_grouped.get(cast(Base[Any, Any], self.base), {})

        remote_records = {
            db: recs
            for db, recs in db_grouped.items()
            if db is not None and hash(db) != hash(self.base)
        }

        # Update with local records first.
        df_data = self._values_to_df(unconnected_records)
        value_table = self._df_to_table(df_data)
        self._mutate_from_query(
            value_table,
            mode,
        )
        value_table.drop(self.base.engine)

        if local_records and has_type(
            self, Table[Record | None, Record, Any, Any, Ctx, Any]
        ):
            # Only update relations for records already existing in this db.
            self._mutate_rels_from_rec_ids(
                {idx: rec._index for idx, rec in local_records.items()},
                mode,
            )

        for base, recs in remote_records.items():
            rec_ids = [rec._index for rec in recs.values()]
            remote_table = Table(_base=base, _type=self.record_type_set)[rec_ids]
            self._mutate(remote_table, mode)

        return

    def _mutate_rels_from_pks(
        self: Rel[Record | None, Any, Any, Record, Ctx, DynBackendID],
        values: Mapping[Hashable, Hashable],
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        assert (
            self._link is not None
            and self._abs_table is not None
            and len(self.record_type_set) == 1
        )

        if not issubclass(self.relation_type, NoneType):
            return self._abs_table._mutate_rels_from_pks(values, mode)

        value_table = self._df_to_table(
            self._values_to_df(values),
        )

        to_fk_cols: list[sqla.ColumnElement] = [
            value_table.c[pk.fqn].label(fk.name)
            for fk_map in self._link._abs_fk_maps.values()
            for fk, pk in fk_map.items()
        ]

        self.ctx._mutate_from_query(
            sqla.select(value_table).with_only_columns(*to_fk_cols).subquery(),
            mode,
        )

    def _mutate_from_query(
        self: Rel[Record | None, Any, Any, Any, Any, DynBackendID],
        query: sqla.FromClause,
        mode: Literal["update", "upsert", "replace", "insert", "delete"] = "update",
    ) -> None:
        if len(self._abs_alignment_cols) > 1:
            for sel in self._abs_alignment_cols:
                sel._mutate_from_query(query, mode)

        col_sets = next(iter(self._abs_alignment_cols.values()))

        if not issubclass(self.relation_type, NoneType):
            assert self._link is not None
            return self._link._mutate_from_query(query, mode)

        if isinstance(self, BackLink):
            # If this table links back to its parent, query and join
            # the parent's primary key columns to the value table.
            ctx_table = Rel.select(self.ctx, cols=self.ctx._abs_idx_cols).subquery()

            pk_fk_cols = {
                pk.fqn: fk.name
                for link in self.links
                for target_type, fk_map in link._abs_fk_maps.items()
                if is_subtype(Union[*self.ctx.record_type_set], target_type)
                for fk, pk in fk_map.items()
            }

            if not all(col in query.c for col in pk_fk_cols.keys()):
                idx_col_map = {
                    query.c[col_name]: ctx_table.c[col_name]
                    for col_name in self._abs_idx_cols.keys()
                    if col_name in ctx_table.columns
                }

                query = (
                    sqla.select(query)
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
                query = (
                    sqla.select(query)
                    .add_columns(
                        *(
                            query.c[col].label(label)
                            for col, label in pk_fk_cols.items()
                        )
                    )
                    .subquery()
                )

        if mode == "update":
            # In case of update, the input query may only contain a subset of columns
            # to be updated. In that case make sure that primary key columns are
            # included by joining it with this table's main query.
            query = self.select.join(
                query,
                reduce(
                    sqla.and_,
                    (
                        col._sql_col == query.c[col_name]
                        for col_name, col in (
                            *self._abs_idx_cols.items(),
                            *(
                                (pk.name, pk)
                                for pk in self._abs_cols.values()
                                if pk.primary_key
                            ),
                        )
                        if col_name in query.columns
                    ),
                ),
            ).alias()

        for target, col_set in col_sets.items():
            assert issubclass(target, Record)

            rec_cols = {
                col.name: query.c[col_name] for col_name, col in col_set.items()
            }
            input_table = (
                query.select()
                .with_only_columns(*(col.label(name) for name, col in rec_cols.items()))
                .alias()
            )

            self.base._mutate_sql_base_tables(target, input_table, mode)

        if isinstance(self, Link):
            # Update link from parents to this table.
            to_fk_cols: list[sqla.ColumnElement] = [
                query.c[pk.fqn].label(fk.name)
                for fk_map in self._abs_fk_maps.values()
                for fk, pk in fk_map.items()
            ]

            self.ctx._mutate_from_query(
                sqla.select(query).add_columns(*to_fk_cols).subquery(),
                "update",
            )

        return

    @overload
    def __setitem__(
        self: Rel[Schema | Record | None, RootIdx, CRUD, None, None, DynBackendID],
        key: type[TabT3],
        value: Rel[TabT3, Any, Any, Any, Any, DynBackendID] | Input[TabT3, Hashable],
    ) -> None: ...

    @overload
    def __setitem__(
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
        key: Rel[
            TabT3 | None,
            BaseIdx[Record[KeyT3]],
            Any,
            Any,
            Any,
            Symbolic,
        ],
        value: (
            Rel[
                TabT3,
                Any,
                Any,
                Any,
                Any,
                DynBackendID,
            ]
            | Input[TabT3 | KeyT3, Hashable]
        ),
    ) -> None: ...

    @overload
    def __setitem__(
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
        key: Rel[
            TabT3 | None,
            BaseIdx[Record[*KeyTt3]],
            Any,
            Any,
            Any,
            Symbolic,
        ],
        value: (
            Rel[
                TabT3,
                Any,
                Any,
                Any,
                Any,
                DynBackendID,
            ]
            | Input[TabT3 | tuple[*KeyTt3], Hashable]
        ),
    ) -> None: ...

    @overload
    def __setitem__(
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
        key: Rel[
            ValT3,
            Any,
            Any,
            Any,
            Any,
            Symbolic,
        ],
        value: (
            Rel[
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
        self: Rel[
            Any, BaseIdx[Record[KeyT2]] | Idx[KeyT2], CRUD, Any, Any, DynBackendID
        ],
        key: KeyT2,
        value: Rel[ValT, Idx[()], Any, Any, Any, DynBackendID] | Input[ValT, None],
    ) -> None: ...

    @overload
    def __setitem__(
        self: Rel[
            Any,
            BaseIdx[Record[*KeyTt2]] | Idx[*KeyTt2],
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        key: tuple[*KeyTt2],
        value: Rel[ValT, Idx[()], Any, Any, Any, DynBackendID] | Input[ValT, None],
    ) -> None: ...

    @overload
    def __setitem__(
        self: Rel[Any, ExtIdx[Record[*KeyTt2], KeyT3], CRUD, Any, Any, DynBackendID],
        key: tuple[*KeyTt2, KeyT3],
        value: Rel[ValT, Idx[()], Any, Any, Any, DynBackendID] | Input[ValT, None],
    ) -> None: ...

    # Implementation:

    def __setitem__(
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
        key: type[Record] | Rel[Any, Any, Any, Any, Any, Symbolic] | Hashable,
        value: Rel[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> None:
        """Replacing assignment."""
        item = self[key]

        if hash(item) != hash(value):
            cast(Rel, item)._mutate(value, mode="replace")

        return

    # 1. Type deletion
    @overload
    def __delitem__(
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
        key: type[TabT2],
    ) -> None: ...

    # 2. Filter deletion
    @overload
    def __delitem__(
        self: Rel[
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
            | Rel[bool, BaseIdx[Record[KeyT2]] | Idx[KeyT2], Any, Any, Any, Symbolic]
        ),
    ) -> None: ...

    # Implementation:

    def __delitem__(  # noqa: D105
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
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
    def __ilshift__(
        self: Rel[
            Record[KeyT2] | None,
            Any,
            RU,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: Rel[ValT, Any, Any, Any, Any, DynBackendID] | Input[ValT | KeyT2, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __ilshift__(
        self: Rel[
            Record[*KeyTt2] | None,
            Any,
            RU,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: (
            Rel[ValT, Any, Any, Any, Any, DynBackendID]
            | Input[ValT | tuple[*KeyTt2], Any]
        ),
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __ilshift__(
        self: Rel[
            ValT2,
            Any,
            RU,
            Any,
            Any,
            DynBackendID,
        ],
        other: Rel[ValT2, Any, Any, Any, Any, DynBackendID] | Input[ValT2, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    def __ilshift__(
        self: Rel[Any, Any, RU, Any, Any, DynBackendID],
        other: Rel[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]:
        """Updating assignment."""
        cast(Rel, self)._mutate(other, mode="update")
        return cast(
            Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT],
            self,
        )

    @overload
    def __ixor__(
        self: Rel[
            Record[KeyT2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: Rel[ValT, Any, Any, Any, Any, DynBackendID] | Input[ValT | KeyT2, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __ixor__(
        self: Rel[
            Record[*KeyTt2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: (
            Rel[ValT, Any, Any, Any, Any, DynBackendID]
            | Input[ValT | tuple[*KeyTt2], Any]
        ),
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __ixor__(
        self: Rel[
            ValT2,
            Any,
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: Rel[ValT2, Any, Any, Any, Any, DynBackendID] | Input[ValT2, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    def __ixor__(
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Rel[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]:
        """Inserting assignment."""
        self._mutate(other, mode="insert")
        return cast(
            Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT],
            self,
        )

    @overload
    def __ior__(
        self: Rel[
            Record[KeyT2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: Rel[ValT, Any, Any, Any, Any, DynBackendID] | Input[ValT | KeyT2, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __ior__(
        self: Rel[
            Record[*KeyTt2] | None,
            Any,
            CRUD,
            Any,
            Ctx,
            DynBackendID,
        ],
        other: (
            Rel[ValT, Any, Any, Any, Any, DynBackendID]
            | Input[ValT | tuple[*KeyTt2], Any]
        ),
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    @overload
    def __ior__(
        self: Rel[
            ValT2,
            Any,
            CRUD,
            Any,
            Any,
            DynBackendID,
        ],
        other: Rel[ValT2, Any, Any, Any, Any, DynBackendID] | Input[ValT2, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]: ...

    def __ior__(
        self: Rel[Any, Any, CRUD, Any, Any, DynBackendID],
        other: Rel[Any, Any, Any, Any, Any, DynBackendID] | Input[Any, Any],
    ) -> Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT]:
        """Upserting assignment."""
        self._mutate(other, mode="upsert")
        return cast(
            Rel[ValT, IdxT, CrudT, RelT, CtxT, DbT],
            self,
        )

    @overload
    def __set__(
        self,
        instance: Record,
        value: Rel[ValT, Idx[()], Any, Any, Any, Any] | ValT | type[Not.changed],
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
        if value is Not.changed:
            return

        if isinstance(instance, Record):
            if isinstance(self, Var):
                if self.setter is not None:
                    self.setter(instance, value)
                else:
                    instance.__dict__[self._name] = value

            if not isinstance(self, Var) or (
                self.pub_status is Public and instance._published
            ):
                owner = type(instance)
                sym_rel: Rel[Any, Any, Any, Any, Any, Symbolic] = getattr(
                    owner, self._name
                )
                instance._table[[instance._index]][sym_rel]._mutate(value)

            if not isinstance(self, Var):
                instance._update_dict()
        else:
            instance.__dict__[self._name] = value

        return
