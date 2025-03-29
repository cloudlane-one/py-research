"""Static schemas for universal relational databases."""

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
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
)

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.sql.visitors as sqla_visitors
import sqlparse
from typing_extensions import TypeVar, TypeVarTuple

from py_research.caching import cached_method, cached_prop
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
ValTo = TypeVar("ValTo", default=None)
ValTt2 = TypeVarTuple("ValTt2")
ValTt3 = TypeVarTuple("ValTt3")

KeyT = TypeVar("KeyT", bound=Hashable)
KeyT2 = TypeVar("KeyT2", bound=Hashable)
KeyT3 = TypeVar("KeyT3", bound=Hashable)

KeyTt = TypeVarTuple("KeyTt")
KeyTt2 = TypeVarTuple("KeyTt2")
KeyTt3 = TypeVarTuple("KeyTt3")
KeyTt4 = TypeVarTuple("KeyTt4")

OrdT = TypeVar("OrdT", bound=Ordinal)

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
CrudT3 = TypeVar("CrudT3", bound=CRUD)

ArgT = TypeVar("ArgT", contravariant=True, default=Any)
ArgIdxT = TypeVar("ArgIdxT", contravariant=True, bound=Idx, default=Any)
ArgSqlT = TypeVar("ArgSqlT", contravariant=True, bound=SqlExpr | None, default=Any)

PassIdxT = TypeVar("PassIdxT", bound=Idx, default=Idx[()], contravariant=True)
PassIdxT2 = TypeVar("PassIdxT2", bound=Idx, default=Idx[()])


class Interface(Generic[ArgT, PassIdxT, ArgIdxT, ArgSqlT, CrudT]):
    """Interface."""


RootT = TypeVar("RootT", bound=Interface, default=Any, covariant=True)
RootT2 = TypeVar("RootT2", bound=Interface)
RootT3 = TypeVar("RootT3", bound=Interface)


@dataclass
class Owner(Interface[ArgT, Idx[*tuple[Any, ...], Any, Any, Any]]):
    """Owner type of a dataset (prop)."""


class Base(Interface[ArgT, Idx[()], Any, Any, CrudT], ABC):
    """Base for retrieving/storing data."""

    @property
    @abstractmethod
    def connection(self) -> sqla.engine.Connection:
        """SQLAlchemy connection to the database."""
        ...

    @abstractmethod
    def registry[T: AutoIndexable](
        self: Base[T, Any], value_type: type[T]
    ) -> Registry[T, CrudT, Base[T, CrudT]]:
        """Get the registry for a type in this base."""
        ...


BaseT = TypeVar("BaseT", bound=Base, covariant=True, default=Any)


class Ctx(Generic[ValT, IdxT, CrudT, SqlT]):
    """Node in context path."""


CtxTt = TypeVarTuple("CtxTt", default=Unpack[tuple[Any, ...]])
CtxTt2 = TypeVarTuple("CtxTt2")
CtxTt3 = TypeVarTuple("CtxTt3")


type InputData[V, S: SqlExpr | None] = V | Iterable[V] | Mapping[
    Hashable, V
] | pd.DataFrame | pl.DataFrame | S | Prop[V]

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
    _context: RootT | Data[Any, Any, Any, Any, RootT]

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

    def _value(self, data: Mapping[str, Any]) -> ValT:
        """Transform dict-like data (e.g. dataframe row) to declared value type."""
        raise NotImplementedError()

    def _index(
        self: Data[Any, AnyIdx[*KeyTt2]],
    ) -> Alignment[tuple[*KeyTt2], SelfIdx[*KeyTt2], R, SqlT, Base]:
        """Get the index of this data."""
        raise NotImplementedError()

    def _sql_mutation(
        self: Data[Any, Any, CrudT2, *tuple[Any, ...]],
        input_data: InputData[ValT, SqlT],
        mode: set[type[CrudT2]] = {C, U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        raise NotImplementedError()

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

    # def _to_ctx(  # noqa: D107
    #     self: Data[
    #         ValT2,
    #         IdxT2,
    #         CrudT2,
    #         SqlT2,
    #         Interface[ValT3, IdxT3, SqlT3, CrudT2],
    #         *CtxTt2,
    #     ],
    #     ctx: Data[ValT3, IdxT3, CrudT2, SqlT3, RootT3, *CtxTt3],
    # ) -> Data[ValT2, IdxT2, CrudT2, SqlT2, RootT3, *CtxTt3, *CtxTt2]:
    #     """Add a context to this property."""
    #     return copy_and_override(
    #         Data[ValT2, IdxT2, CrudT2, SqlT2, RootT3, *CtxTt3, *CtxTt2],
    #         self,
    #         _context=(
    #             ctx
    #             if isinstance(self._context, Interface | Base | Owner)
    #             else self._context._to_ctx(ctx)
    #         ),
    #     )

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
        ],
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

    @overload
    def select(self: Data[Any, Any, Any, None, Any, *tuple[Any, ...]]) -> None: ...

    @overload
    def select(self) -> sqla.Select: ...

    def select(self) -> sqla.SelectBase | None:
        """Return select statement for this dataset."""
        sql = self._sql()
        match sql:
            case None:
                return None
            case sqla.SelectBase():
                return sql
            case _:
                return sqla.select(sql)

    @overload
    def select_str(self: Data[Any, Any, Any, None, Any, *tuple[Any, ...]]) -> None: ...

    @overload
    def select_str(self) -> str: ...

    def select_str(self) -> str | None:
        """Return select statement for this dataset."""
        select = self.select()
        if select is None:
            return None

        assert isinstance(self.root, Base)

        return sqlparse.format(
            str(select.compile(self.root.connection)),
            reindent=True,
            keyword_case="upper",
        )

    @overload
    def query(self: Data[Any, Any, Any, None, Any, *tuple[Any, ...]]) -> None: ...

    @overload
    def query(self) -> sqla.Subquery: ...

    @cached_method
    def query(
        self,
    ) -> sqla.Subquery | None:
        """Return select statement for this dataset."""
        select = self.select()
        if select is None:
            return None

        return select.subquery()

    # Dataframes:

    def df(
        self: Data[Any, Any, Any, Any, Base],
    ) -> pl.DataFrame:
        """Load dataset as dataframe."""
        df = self._df()

        if df is not None:
            return df

        select = self.select()
        assert select is not None

        return pl.read_database(
            select,
            self.root.connection,
        )

    # Collection interface:

    def values(
        self: Data[Any, Any, Any, Any, Base],
    ) -> Sequence[ValT]:
        """Iterable over this dataset's values."""
        return [self._value(row) for row in self.df().iter_rows(named=True)]

    @overload
    def keys(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, AnyIdx[KeyT2], Any, Any, Base],
    ) -> Sequence[KeyT2]: ...

    @overload
    def keys(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Any, Base],
    ) -> Sequence[tuple[*KeyTt2]]: ...

    def keys(
        self: Data[Any, Any, Any, Any, Base],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        return self._index().values()

    @overload
    def items(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, AnyIdx[KeyT2], Any, Any, Base],
    ) -> Iterable[tuple[KeyT2, ValT]]: ...

    @overload
    def items(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Any, Base],
    ) -> Iterable[tuple[tuple[*KeyTt2], ValT]]: ...

    def items(
        self: Data[Any, Any, Any, Any, Base],
    ) -> Iterable[tuple[Any, ValT]]:
        """Iterable over index keys."""
        return zip(self.keys(), self.values())

    @overload
    def get(
        self: Data[Any, Idx[()], Any, Any, Base],
        key: None = ...,
        default: ValTo = ...,
    ) -> ValT | ValTo: ...

    @overload
    def get(
        self: Data[ValT2, AnyIdx[KeyT2], Any, Any, Base],
        key: KeyT2 | tuple[KeyT2],
        default: ValTo,
    ) -> ValT | ValTo: ...

    @overload
    def get(
        self: Data[ValT2, AnyIdx[*KeyTt2], Any, Any, Base],
        key: tuple[*KeyTt2],
        default: ValTo,
    ) -> ValT | ValTo: ...

    def get(
        self: Data[Any, Any, Any, Any, Base],
        key: Hashable = None,
        default: ValTo = None,
    ) -> ValT | ValTo:
        """Get a record by key."""
        try:
            return (self[key] if key is not None else self).values()[0]
        except KeyError | IndexError:
            return default

    def __iter__(  # noqa: D105
        self: Data[Any, Any, Any, Any, Base],
    ) -> Iterator[ValT]:
        return iter(self.values())

    def __len__(self: Data[Any, Any, Any, Any, Base]) -> int:
        """Get the number of items in the dataset."""
        df = self._df()
        if df is not None:
            return len(df)

        select = self.select()
        assert select is not None
        count = self.root.connection.execute(
            sqla.select(sqla.func.count()).select_from(select)
        ).scalar()
        assert count is not None
        return count

    # Selection and filtering:

    # 1. Interface application, keep value type
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], CrudT2, SqlT2],
        key: Data[
            Literal[Not.changed],
            AnyIdx[*KeyTt3],
            CrudT2,
            SqlT3,
            Interface[ValT2, Idx[*KeyTt2], Idx[*KeyTt4], SqlT2, CrudT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt2, *KeyTt3],
        CrudT2,
        SqlT3,
        RootT,
        *CtxTt,
        Ctx[ValT2, IdxT, CrudT, SqlT],
        *CtxTt3,
    ]: ...

    # 2. Interface application
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], CrudT2, SqlT2],
        key: Data[
            ValT3,
            AnyIdx[*KeyTt3],
            CrudT2,
            SqlT3,
            Interface[ValT2, Idx[*KeyTt2], Idx[*KeyTt4], SqlT2, CrudT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt2, *KeyTt3],
        CrudT2,
        SqlT3,
        RootT,
        *CtxTt,
        Ctx[ValT2, IdxT, CrudT, SqlT],
        *CtxTt3,
    ]: ...

    # 3. Key list / slice filtering, scalar index type
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], RU],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, RU, SqlT, RootT, *CtxTt]: ...

    # 4. Key list / slice filtering
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], RU],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, RU, SqlT, RootT, *CtxTt]: ...

    # 5. Key list / slice filtering, scalar index type, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], R],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, R, SqlT, RootT, *CtxTt]: ...

    # 6. Key list / slice filtering, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], R],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, R, SqlT, RootT, *CtxTt]: ...

    # 7. Key selection
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt3, *KeyTt2]],
        key: tuple[*KeyTt3],
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        CrudT,
        SqlT,
        RootT,
        *CtxTt,
        Ctx[ValT, IdxT, CrudT, SqlT],
    ]: ...

    # 8. Key selection, scalar
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT3, *KeyTt2]],
        key: KeyT3,
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        CrudT,
        SqlT,
        RootT,
        *CtxTt,
        Ctx[ValT, IdxT, CrudT, SqlT],
    ]: ...

    # 9. Type selection / filtering
    @overload
    def __getitem__(
        self,
        key: type[ValT3],
    ) -> Data[ValT3, IdxT, CrudT, SqlT, RootT, *CtxTt]: ...

    def __getitem__(
        self,
        key: Data | list | slice | tuple[slice, ...] | Hashable | type,
    ) -> Data | ValT:
        """Expand the relational-computational graph."""
        match key:
            case Data():
                return copy_and_override(type(key), key, _context=self)
            case type() | UnionType():
                raise NotImplementedError()
            case list() | slice() | Hashable():
                if not isinstance(key, list | slice) and not has_type(
                    key, tuple[slice, ...]
                ):
                    key = [key]

                return copy_and_override(
                    Filter,
                    self,
                    _context=self,
                    _filter=cast(
                        list[Hashable] | slice | tuple[slice, ...],
                        key,
                    ),
                )

    # Comparison:

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, sqla.ColumnElement, Interface[ValT2]],
        other: Data[Any, Any, Any, sqla.ColumnElement, Interface[ValT2]],
    ) -> Filter[ValT2]: ...

    @overload
    def __eq__(
        self: Data[*tuple[Any, ...]],
        other: Any,
    ) -> bool: ...

    def __eq__(  # noqa: D105
        self,
        other: Any,
    ) -> Filter | bool:
        self_sql = self._sql()
        if not isinstance(self_sql, sqla.ColumnElement):
            return id(self) == id(other)

        if not isinstance(other, Data):
            return id(self) == id(other)

        other_sql = other._sql()
        if not isinstance(other_sql, sqla.ColumnElement):
            return id(self) == id(other)

        return copy_and_override(
            Filter,
            self,
            _context=self.root,
            _filter=self_sql == other_sql,
        )

    @overload
    def __neq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, sqla.ColumnElement, Interface[ValT2]],
        other: Data[Any, Any, Any, sqla.ColumnElement, Interface[ValT2]],
    ) -> Filter[ValT2]: ...

    @overload
    def __neq__(
        self: Data[*tuple[Any, ...]],
        other: Any,
    ) -> bool: ...

    def __neq__(  # noqa: D105
        self,
        other: Any,
    ) -> Filter | bool:
        eq = self == other
        if isinstance(eq, Filter):
            filt = eq._filter
            assert isinstance(filt, sqla.ColumnElement)
            return copy_and_override(Filter, self, _context=self.root, _filter=filt)

        return not eq

    def __lt__(  # noqa: D105
        self: Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
        other: OrdT | Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
    ) -> Filter[ValT2]:
        filt = (
            self._sql() < other._sql()
            if isinstance(other, Data)
            else self._sql() < other
        )
        return copy_and_override(
            Filter,
            self,
            _context=self.root,
            _filter=filt,
        )

    def __le__(  # noqa: D105
        self: Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
        other: OrdT | Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
    ) -> Filter[ValT2]:
        filt = (
            self._sql() <= other._sql()
            if isinstance(other, Data)
            else self._sql() <= other
        )
        return copy_and_override(
            Filter,
            self,
            _context=self.root,
            _filter=filt,
        )

    def __gt__(  # noqa: D105
        self: Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
        other: OrdT | Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
    ) -> Filter[ValT2]:
        filt = (
            self._sql() > other._sql()
            if isinstance(other, Data)
            else self._sql() > other
        )
        return copy_and_override(
            Filter,
            self,
            _context=self.root,
            _filter=filt,
        )

    def __ge__(  # noqa: D105
        self: Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
        other: OrdT | Data[OrdT, Any, Any, sqla.ColumnElement, Interface[ValT2]],
    ) -> Filter[ValT2]:
        filt = (
            self._sql() >= other._sql()
            if isinstance(other, Data)
            else self._sql() >= other
        )
        return copy_and_override(
            Filter,
            self,
            _context=self.root,
            _filter=filt,
        )

    def isin(
        self: Data[Any, Any, Any, sqla.ColumnElement, Interface[ValT2]],
        other: Iterable[ValT2] | slice,
    ) -> Filter[ValT2]:
        """Test values of this dataset for membership in the given iterable."""
        filt = (
            self._sql().between(other.start, other.stop)
            if isinstance(other, slice)
            else self._sql().in_(other)
        )
        return copy_and_override(
            Filter,
            self,
            _context=self.root,
            _filter=filt,
        )

    # Alignment:

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, None, RootT2],
        other: Data[tuple[*ValTt3], IdxT3, CrudT2, Any, RootT2],
    ) -> Alignment[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        CrudT2,
        None,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, Any, RootT2],
        other: Data[tuple[*ValTt3], IdxT3, CrudT2, None, RootT2],
    ) -> Alignment[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        CrudT2,
        None,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, SqlExpr, RootT2],
        other: Data[tuple[*ValTt3], IdxT3, CrudT2, SqlExpr, RootT2],
    ) -> Alignment[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        CrudT2,
        sqla.Select,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, CrudT2, None, RootT2],
        other: Data[ValT3, IdxT3, CrudT2, Any, RootT2],
    ) -> Alignment[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        CrudT2,
        None,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, CrudT2, Any, RootT2],
        other: Data[ValT3, IdxT3, CrudT2, None, RootT2],
    ) -> Alignment[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        CrudT2,
        None,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, CrudT2, SqlExpr, RootT2],
        other: Data[ValT3, IdxT3, CrudT2, SqlExpr, RootT2],
    ) -> Alignment[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        CrudT2,
        sqla.Select,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, None, RootT2],
        other: Data[ValT3, IdxT3, CrudT2, Any, RootT2],
    ) -> Alignment[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        CrudT2,
        None,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, Any, RootT2],
        other: Data[ValT3, IdxT3, CrudT2, None, RootT2],
    ) -> Alignment[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        CrudT2,
        None,
        RootT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, SqlExpr, RootT2],
        other: Data[ValT3, IdxT3, CrudT2, SqlExpr, RootT2],
    ) -> Alignment[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        CrudT2,
        sqla.Select,
        RootT2,
    ]: ...

    def __matmul__(
        self: Data[Any, Any, CrudT2, Any, RootT2],
        other: Data[Any, IdxT3, CrudT2, Any, RootT2],
    ) -> Alignment[
        tuple,
        IdxT | IdxT3,
        CrudT2,
        sqla.Select | None,
        RootT2,
    ]:
        """Align two datasets."""
        self_elem = self.elements if isinstance(self, Alignment) else (self,)
        other_elem = other.elements if isinstance(other, Alignment) else (other,)
        return Alignment[
            tuple[*self_elem, *other_elem],
            IdxT | IdxT3,
            CrudT2,
            sqla.Select | None,
            RootT2,
        ](
            self_elem + other_elem,
            _context=self.root,
        )

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
    ) -> DataBase[ArgT | SchemaT2, Any, CRUD, BackT2]:
        """Union two databases, right overriding left."""
        db = copy_and_override(
            DataBase[ArgT | SchemaT2, Any, CRUD, BackT2],
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
    ) -> DataBase[ArgT | SchemaT2, Any, CRUD, BackT2]:
        """Union two databases, left overriding right."""
        db = copy_and_override(
            DataBase[ArgT | SchemaT2, Any, CRUD, BackT2],
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
    ) -> DataBase[ArgT | SchemaT2, Any, CRUD, BackT2]:
        """Intersect two databases, right overriding left."""
        db = copy_and_override(
            DataBase[ArgT | SchemaT2, Any, CRUD, BackT2],
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

    _instance_map: dict[Hashable, RegT] = field(default_factory=dict)

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
class Filter(
    Data[
        Literal[Not.changed],
        Idx[()],
        R | U,
        sqla.Select,
        Interface[
            ArgT, Idx[*tuple[Any, ...]], Idx[()], sqla.FromClause | sqla.SelectBase, Any
        ],
    ]
):
    """Property definition for a model."""

    # Attributes:

    _filter: (
        sqla.ColumnElement[bool] | pl.Expr | slice | list[Hashable] | tuple[slice, ...]
    )

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
class Prop(Data[ValT, Idx[*KeyTt2, *KeyTt3], CrudT, SqlT, BaseT, RootT, *CtxTt]):
    """Property definition for a model."""

    # Attributes:

    init_after: ClassVar[set[type[Prop]]] = set()

    _owner: type[RootT] | None = None

    alias: str | None = None
    default: ValT | InputData[ValT, SqlT] | Literal[Not.defined] = Not.defined
    default_factory: Callable[[], ValT | InputData[ValT, SqlT]] | None = None
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
        self: Prop[*tuple[Any, ...]],
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


TupT = TypeVar("TupT", bound=tuple, covariant=True)
SelSqlT = TypeVar("SelSqlT", bound=sqla.Select | None, covariant=True, default=None)


@dataclass
class Alignment(Data[TupT, IdxT, CrudT, SelSqlT, RootT]):
    """Alignment of multiple props."""

    elements: tuple[Data[Any, IdxT, CrudT, Any, RootT], ...]


RecSqlT = TypeVar("RecSqlT", bound=sqla.CTE | None, covariant=True, default=None)


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


@dataclass
class Recursion(Data[ValT, Idx[*tuple[Any, ...]], R, RecSqlT, BaseT, RootT, *CtxTt]):
    """Combination of multiple props."""

    paths: tuple[Path[ValT, Any, Any, Any, BaseT, RootT], ...]
