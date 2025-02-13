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
from xlsxwriter import Workbook as ExcelWorkbook

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

from .utils import pd_to_py_dtype, pl_type_map, sql_to_py_dtype


class Schema:
    """Group multiple record types into a schema."""

    _schema_types: set[type[Model]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        cls._schema_types = set()
        super().__init_subclass__()


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls: type[Record], name: str) -> Var[Any, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by dynamic name."""
        return Var(_base=symbol_base, _type=Var[cls], _ctx=Ctx({cls}), _name=name)

    def __getattr__(cls: type[Record], name: str) -> Var[Any, Any, Any, Any, Symbolic]:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)
        return Var(_base=symbol_base, _type=Var[cls], _ctx=Ctx({cls}), _name=name)


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


x = DynRecord


def dynamic_record_type(
    base: type[RecT2] | tuple[type[RecT2], ...],
    name: str,
    props: Iterable[Prop] = [],
    src_module: ModuleType | None = None,
    extra_attrs: dict[str, Any] = {},
) -> type[RecT2]:
    """Create a dynamically defined record type."""
    base = base if isinstance(base, tuple) else (base,)
    return cast(
        type[RecT2],
        new_class(
            name,
            base,
            None,
            lambda ns: ns.update(
                {
                    **{p.name: p for p in props},
                    "__annotations__": {p.name: p._resolved_typehint for p in props},
                    "_src_mod": src_module or base[0]._src_mod or getmodule(base[0]),
                    **extra_attrs,
                }
            ),
        ),
    )


class HashRec(Record[str]):
    """Record type with a default hashed primary key."""

    _template = True

    _id: Var[str] = Var(primary_key=True, init=False)

    def __post_init__(self) -> None:  # noqa: D105
        self._id = gen_str_hash(
            {
                a.name: getattr(self, a.name)
                for a in type(self)._values.values()
                if a.name != "_id"
            }
        )


class Entity(Record[UUID4]):
    """Record type with a default UUID4 primary key."""

    _template = True
    _id: Var[UUID4] = Var(primary_key=True, default_factory=lambda: UUID4(uuid4()))


class BacklinkRecord(Record[*KeyTt], Generic[RecT2, *KeyTt]):
    """Dynamically defined record type."""

    _template = True

    _from: Link[RecT2]


class Item(BacklinkRecord[RecT2, *KeyTt], Generic[ValT, RecT2, *KeyTt]):
    """Dynamically defined scalar record type."""

    _template = True
    _array: ClassVar[Array[Any, Any, Any, Any, Symbolic]]

    _from: Link[RecT2] = Link(primary_key=True)
    idx: Data[KeyT2] = Var(primary_key=True)
    value: Var[ValT]


class Relation(BacklinkRecord[str, RecT2], Generic[RecT2, RecT3]):
    """Automatically defined relation record type."""

    _template = True

    _from: Link[RecT2] = Link(primary_key=True)
    _to: Link[RecT3] = Link(primary_key=True)


class RelIndex(Relation[RecT2, RecT3], Generic[RecT2, RecT3, KeyT]):
    """Automatically defined relation record type with index substitution."""

    _template = True

    _to: Link[RecT3] = Link(primary_key=False)
    _rel_idx: Var[KeyT] = Var(primary_key=True)


@dataclass(kw_only=True, eq=False)
class Var(
    Data[ValT, Idx[()], RwT, BaseT, AnyCtx[ParT]],
    Generic[ValT, RwT, BaseT, ParT],
):
    """Single-value attribute or column."""

    alias: str | None = None
    init: bool = True


# Redefine IdxT in an attempt to fix randomly occuring type error,
# in which IdxT isn't recognized as a type variable below anymore.
TdxT = TypeVar(
    "TdxT",
    bound=Idx | BaseIdx,
    default=BaseIdx,
)


@dataclass(kw_only=True, eq=False)
class Table(
    Data[RefT, TdxT, CrudT, BaseT, Ctx[RelT, TdxT, CrudT, BaseT, CtxT]],
    Generic[RefT, RelT, CrudT, TdxT, BaseT, CtxT],
):
    """Record set."""

    index_by: (
        Var[Any, Any, Public, Any, Symbolic]
        | Iterable[Var[Any, Any, Public, Any, Symbolic]]
        | None
    ) = None
    default: bool = False


type SymbolicValue = Var[Any, Any, Public, Any, Symbolic]
type SingleFkMap = (
    SymbolicValue | dict[SymbolicValue, SymbolicValue] | set[SymbolicValue]
)
type MultiFkMap = dict[type[Record], SingleFkMap]


@dataclass(kw_only=True, eq=False)
class Link(
    Data[RefT, Idx[()], CrudT, BaseT, AnyCtx[ParT]],
    Generic[RefT, CrudT, BaseT, ParT],
):
    """Link to a single record."""

    fks: SingleFkMap | MultiFkMap | None = None

    index: bool = True
    primary_key: bool = False


@dataclass(kw_only=True, eq=False)
class BackLink(
    Data[RecT, TdxT, CrudT, BaseT, AnyCtx[ParT]],
    Generic[RecT, CrudT, TdxT, ParT, BaseT],
):
    """Backlink record set."""

    to: (
        Data[ParT, Idx[()], Any, Symbolic, AnyCtx[RecT]]
        | set[Data[ParT, Idx[()], Any, Symbolic, AnyCtx[RecT]]]
    )


type ItemType = Item

_item_classes: dict[str, type[Item]] = {}


@dataclass(eq=False)
class Array(
    Data[
        ValT,
        ArrayIdx[Any, *KeyTt],
        CrudT,
        BaseT,
        Ctx["Item[ValT, KeyT, OwnT]", Idx[*KeyTt], CrudT, BaseT, AnyCtx[OwnT]],
    ],
    Generic[ValT, KeyT, CrudT, OwnT, BaseT],
):
    """Set / array of scalar values."""

    @cached_prop
    def _key_type(self) -> type[KeyT]:
        typedef = get_typevar_map(self._resolved_typehint)[KeyT]
        return typedef_to_typeset(typedef).pop()

    @cached_prop
    def relation_type(self) -> type[Item[ValT, KeyT, OwnT]]:
        """Return the dynamic relation record type."""
        assert self._ctx_table is not None
        assert issubclass(self._ctx_table.common_type, Record)
        base_array_fqn = copy_and_override(Array, self, _ctx=self._ctx_table).fqn

        rec = _item_classes.get(
            base_array_fqn,
            dynamic_record_type(
                Item[self.target_typeform, self._key_type, self._ctx_table.common_type],
                f"{self._ctx_table.common_type.__name__}.{self.name}",
                src_module=self._ctx_table.common_type._src_mod
                or getmodule(self._ctx_table.common_type),
                extra_attrs={
                    "_array": copy_and_override(
                        Array, self, _base=symbol_base, _ctx=self._ctx_table
                    )
                },
            ),
        )
        _item_classes[base_array_fqn] = rec

        return rec


RootKeyT = TypeVar("RootKeyT", bound=Hashable, default=None)


@dataclass(eq=False)
class DataBase(Data[SchemaT, BaseIdx, CrudT, BaseT, None], Base[SchemaT, BaseT]):
    """Database connection."""

    def __post_init__(self):  # noqa: D105
        super().__post_init__()

        self._base = self
        self._name = self.db_id
        self._type = cast(
            type[SchemaT],
            (
                self.schema
                if isinstance(self.schema, type)
                else Union[*self._def_types] if len(self._def_types) > 0 else None
            ),
        )
