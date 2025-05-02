"""Static schemas for universal relational databases."""

from __future__ import annotations

import inspect
import operator
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from functools import cache, partial, reduce
from inspect import get_annotations, getmodule
from types import ModuleType, NoneType, UnionType
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
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


@final
class KeepVal:
    """Singleton to allow keeping of context value type."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


ValT = TypeVar("ValT", covariant=True, default=Any)
ValT2 = TypeVar("ValT2")
ValT3 = TypeVar("ValT3")
ValT4 = TypeVar("ValT4")
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
KeyTt5 = TypeVarTuple("KeyTt5")

OrdT = TypeVar("OrdT", bound=Ordinal)

type ColTuple = tuple[sqla.ColumnElement, ...]
type SeriesTuple = tuple[pl.Series, ...]

type SqlExpr = (sqla.SelectBase | sqla.FromClause | sqla.ColumnElement | ColTuple)
type PlObj = (pl.DataFrame | pl.Series | SeriesTuple)

DataT = TypeVar(
    "DataT",
    bound=SqlExpr | PlObj,
    covariant=True,
    default=Any,
)
DataT2 = TypeVar(
    "DataT2",
    bound=SqlExpr | PlObj,
)
DataT3 = TypeVar(
    "DataT3",
    bound=SqlExpr | PlObj,
)


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


type FullIdx[*K] = Idx[*K] | SelfIdx[*K] | HashIdx[*K] | AutoIdx[AutoIndexable[*K]]

KeepIdxT = TypeVar(
    "KeepIdxT",
    bound=Idx,
    default=Idx[*tuple[Any, ...]],
)
SubIdxT = TypeVar(
    "SubIdxT",
    bound=Idx,
    default=Idx[()],
)
AddIdxT = TypeVar(
    "AddIdxT",
    bound=Idx,
    default=Idx[()],
)


class PassIdx(Generic[KeepIdxT, SubIdxT, AddIdxT]):
    """Pass-through index."""


type AnyIdx[*K] = FullIdx[*K] | PassIdx[Any, Any, Any]

IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound=AnyIdx,
    default=AnyIdx,
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
ArgDataT = TypeVar("ArgDataT", contravariant=True, bound=SqlExpr | PlObj, default=Any)


class Ctx(Generic[ArgT, ArgIdxT, ArgDataT, CrudT]):
    """Interface."""


CtxT = TypeVar("CtxT", bound=Ctx, default=Any, covariant=True)
CtxT2 = TypeVar("CtxT2", bound=Ctx)
CtxT3 = TypeVar("CtxT3", bound=Ctx)

CtxTt = TypeVarTuple("CtxTt", default=Unpack[tuple[Any, ...]])
CtxTt2 = TypeVarTuple("CtxTt2")
CtxTt3 = TypeVarTuple("CtxTt3")


class Base(Ctx[ArgT, Any, Any, CrudT], ABC):
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


type InputData[
    V,
    S: sqla.SelectBase
    | sqla.FromClause
    | sqla.ColumnElement
    | pl.DataFrame
    | pl.Series,
] = V | Iterable[V] | Mapping[Hashable, V] | pd.DataFrame | pl.DataFrame | S | Prop[V]

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
class Data(Generic[ValT, IdxT, CrudT, DataT, CtxT, *CtxTt]):
    """Property definition for a model."""

    # Core attributes:

    context: CtxT | Data[Any, Any, Any, Any, CtxT]
    typeref: TypeRef[Prop[ValT]] | None = None

    # Extension methods:

    def _name(self) -> str:
        """Name of the property."""
        return gen_str_hash(self)

    def _data(self) -> DataT:
        """Get SQL-side reference to this property."""
        assert get_typevar_map(self.resolved_type)[DataT] is NoneType
        return cast(DataT, None)

    def _value(self, data: Mapping[str, Any]) -> ValT:
        """Transform dict-like data (e.g. dataframe row) to declared value type."""
        raise NotImplementedError()

    def _index(
        self: Data[Any, FullIdx[*KeyTt2]],
    ) -> Align[tuple[*KeyTt2], SelfIdx[*KeyTt2], R, Any, Base]:
        """Get the index of this data."""
        raise NotImplementedError()

    def _sql_mutation(
        self: Data[Any, Any, CrudT2, *tuple[Any, ...]],
        input_data: InputData[ValT, DataT],
        mode: set[type[CrudT2]] = {C, U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        raise NotImplementedError()

    # Type:

    @cached_prop
    def resolved_type(self) -> SingleTypeDef[Prop[ValT]]:
        """Resolved type of this prop."""
        if self.typeref is None:
            return cast(SingleTypeDef[Prop[ValT]], type(self))

        return hint_to_typedef(
            self.typeref.hint,
            typevar_map=self.typeref.var_map,
            ctx_module=self.typeref.ctx_module,
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

    @cached_prop
    def root(self) -> CtxT:
        """Get the root of this property."""
        if isinstance(self.context, Data):
            return self.context.root

        return self.context

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if not isinstance(self.context, Data):
            return self._name()

        return self.context.fqn + "." + self._name()

    # Index:

    def _map_index_selectors(
        self, sel: list | slice | tuple[list | slice, ...]
    ) -> (
        Mapping[Data[Any, Any, Any, sqla.ColumnElement, CtxT], slice | Collection]
        | Mapping[Data[Any, Any, Any, pl.Series, CtxT], slice | Collection]
    ):
        # TODO: implement
        raise NotImplementedError()

    # SQL:

    @overload
    def select(self: Data[Any, Any, Any, PlObj, Any, *tuple[Any, ...]]) -> None: ...

    @overload
    def select(self) -> sqla.Select: ...

    def select(self) -> sqla.SelectBase | None:
        """Return select statement for this dataset."""
        sql = self._data()
        match sql:
            case pl.DataFrame() | pl.Series():
                return None
            case sqla.SelectBase():
                return sql
            case tuple():
                if has_type(sql, tuple[sqla.ColumnElement, ...]):
                    return sqla.select(*sql)
                else:
                    return None
            case _:
                return sqla.select(sql)

    @overload
    def select_str(self: Data[Any, Any, Any, PlObj, Any, *tuple[Any, ...]]) -> None: ...

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
    def query(self: Data[Any, Any, Any, PlObj, Any, *tuple[Any, ...]]) -> None: ...

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
        data = self._data()

        if isinstance(data, pl.DataFrame):
            return data

        if isinstance(data, pl.Series):
            return data.to_frame()

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
        self: Data[Any, FullIdx[KeyT2], Any, Any, Base],
    ) -> Sequence[KeyT2]: ...

    @overload
    def keys(
        self: Data[Any, FullIdx[*KeyTt2], Any, Any, Base],
    ) -> Sequence[tuple[*KeyTt2]]: ...

    def keys(
        self: Data[Any, Any, Any, Any, Base],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        return self._index().values()

    @overload
    def items(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, FullIdx[KeyT2], Any, Any, Base],
    ) -> Iterable[tuple[KeyT2, ValT]]: ...

    @overload
    def items(
        self: Data[Any, FullIdx[*KeyTt2], Any, Any, Base],
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
        self: Data[ValT2, FullIdx[KeyT2], Any, Any, Base],
        key: KeyT2 | tuple[KeyT2],
        default: ValTo,
    ) -> ValT | ValTo: ...

    @overload
    def get(
        self: Data[ValT2, FullIdx[*KeyTt2], Any, Any, Base],
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
        data = self._data()
        if isinstance(data, pl.DataFrame | pl.Series):
            return len(data)

        select = self.select()
        assert select is not None
        count = self.root.connection.execute(
            sqla.select(sqla.func.count()).select_from(select)
        ).scalar()
        assert count is not None
        return count

    # Application:

    # 1. Context application, altered index, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], CrudT2, DataT2],
        key: Data[
            KeepVal,
            PassIdx[Idx[*KeyTt2], Idx[*KeyTt4], Idx[*KeyTt3]],
            CrudT2,
            DataT3,
            Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], DataT2, CrudT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt2, *KeyTt3],
        CrudT2,
        DataT3,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], DataT, CrudT],
        *CtxTt3,
    ]: ...

    # 2. Context application, altered index, new value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], CrudT2, DataT2],
        key: Data[
            ValT3,
            PassIdx[Idx[*KeyTt2], Idx[*KeyTt4], Idx[*KeyTt3]],
            CrudT2,
            DataT3,
            Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], DataT2, CrudT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt2, *KeyTt3],
        CrudT2,
        DataT3,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], DataT, CrudT],
        *CtxTt3,
    ]: ...

    # 3. Context application, new index, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2], CrudT2, DataT2],
        key: Data[
            KeepVal,
            FullIdx[*KeyTt3],
            CrudT2,
            DataT3,
            Ctx[ValT2, Idx[*KeyTt2], DataT2, CrudT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt3],
        CrudT2,
        DataT3,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2], DataT, CrudT],
        *CtxTt3,
    ]: ...

    # 4. Context application, new index, new value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2], CrudT2, DataT2],
        key: Data[
            ValT3,
            FullIdx[*KeyTt3],
            CrudT2,
            DataT3,
            Ctx[ValT2, Idx[*KeyTt2], DataT2, CrudT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt3],
        CrudT2,
        DataT3,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2], DataT, CrudT],
        *CtxTt3,
    ]: ...

    # 5. Key list / slice filtering, scalar index type
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], RU],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, RU, DataT, CtxT, *CtxTt]: ...

    # 6. Key list / slice filtering
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], RU],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, RU, DataT, CtxT, *CtxTt]: ...

    # 7. Key list / slice filtering, scalar index type, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], R],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, R, DataT, CtxT, *CtxTt]: ...

    # 8. Key list / slice filtering, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], R],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, R, DataT, CtxT, *CtxTt]: ...

    # 9. Key selection
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt3, *KeyTt2]],
        key: tuple[*KeyTt3],
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        CrudT,
        DataT,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[*KeyTt3, *KeyTt2], DataT, CrudT],
    ]: ...

    # 10. Key selection, scalar
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT3, *KeyTt2]],
        key: KeyT3,
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        CrudT,
        DataT,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[KeyT3, *KeyTt2], DataT, CrudT],
    ]: ...

    # 11. Base type selection
    @overload
    def __getitem__(
        self: Base[ValT2, CrudT2],
        key: type[ValT2],
    ) -> Data[ValT2, IdxT, CrudT, DataT, CtxT, *CtxTt]: ...

    def __getitem__(
        self,
        key: Data | list | slice | tuple[slice, ...] | Hashable | type,
    ) -> Data | ValT:
        """Expand the relational-computational graph."""
        match key:
            case Data():
                return copy_and_override(type(key), key, context=self)
            case type() | UnionType():
                assert isinstance(self, Base)
                if isinstance(key, UnionType):
                    union_types = typedef_to_typeset(key)
                    alignment = reduce(
                        Data.__matmul__, (self.registry(t) for t in union_types)
                    )
                    return alignment._map_reduce_operator(operator.or_)

                return self.registry(key)
            case list() | slice() | Hashable():
                if not isinstance(key, list | slice) and not has_type(
                    key, tuple[slice, ...]
                ):
                    key = [key]

                keymap = self._map_index_selectors(key)

                return Filter.from_keymap(keymap)

    # Alignment:

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, CrudT2, SqlExpr, CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, CrudT2, SqlExpr, CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        CrudT2,
        sqla.Select,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, Any, CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, CrudT2, Any, CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        CrudT2,
        PlObj,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[tuple[*ValTt2], Any, CrudT2, SqlExpr, CtxT2],
        other: Data[ValT3, IdxT3, CrudT2, SqlExpr, CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        CrudT2,
        sqla.Select,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, CrudT2, Any, CtxT2],
        other: Data[ValT3, IdxT3, CrudT2, Any, CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        CrudT2,
        PlObj,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, SqlExpr, CtxT2],
        other: Data[ValT3, IdxT3, CrudT2, SqlExpr, CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        CrudT2,
        sqla.Select,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, CrudT2, Any, CtxT2],
        other: Data[ValT3, IdxT3, CrudT2, Any, CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        CrudT2,
        PlObj,
        CtxT2,
    ]: ...

    def __matmul__(
        self: Data[Any, Any, CrudT2, Any, CtxT2],
        other: Data[Any, IdxT3, CrudT2, Any, CtxT2],
    ) -> Data[
        tuple,
        IdxT | IdxT3,
        CrudT2,
        sqla.Select | PlObj,
        CtxT2,
    ]:
        """Align two datasets."""
        if is_subtype(self.value_typeform, tuple):
            assert isinstance(self, Align)
            self_data = self.data
            self_types = self.value_types
        else:
            self_data = (self,)
            self_types = (self.value_typeform,)

        if is_subtype(other.value_typeform, tuple):
            assert isinstance(other, Align)
            other_data = other.data
            other_types = other.value_types
        else:
            other_data = (other,)
            other_types = (other.value_typeform,)

        return Align[
            tuple[*self_types, *other_types],
            IdxT | IdxT3,
            CrudT2,
            sqla.Select | PlObj,
            CtxT2,
        ](
            self_data + other_data,
            context=self.root,
        )

    # Reduction:

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], AnyIdx[*KeyTt2], Any, SqlExpr],
        op: Callable[[ValT2, ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        sqla.ColumnElement,
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DataT, CrudT],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], AnyIdx[*KeyTt2], Any, Any],
        op: Callable[[ValT2, ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        pl.Series,
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DataT, CrudT],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], AnyIdx[*KeyTt2], Any, SqlExpr],
        op: Callable[[ValT2, ValT4], ValT3],
        right: ValT4,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        sqla.ColumnElement,
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DataT, CrudT],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], AnyIdx[*KeyTt2], Any, Any],
        op: Callable[[ValT2, ValT4], ValT3],
        right: ValT4,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        pl.Series,
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DataT, CrudT],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[ValT2, AnyIdx[*KeyTt2], Any, sqla.ColumnElement],
        op: Callable[[ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        sqla.ColumnElement,
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DataT, CrudT],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[ValT2, AnyIdx[*KeyTt2], Any, pl.Series],
        op: Callable[[ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        pl.Series,
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DataT, CrudT],
    ]: ...

    def _map_reduce_operator(
        self: Data,
        op: Callable[[Any, Any], Any] | Callable[[Any], Any],
        right: Any | Literal[Not.defined] = Not.defined,
    ) -> Data[
        Any,
        Any,
        R,
        sqla.ColumnElement | pl.Series,
        CtxT,
        Ctx[ValT, Any, DataT, CrudT],
    ]:
        """Create a scalar comparator for the given operation."""
        if right is not Not.defined:
            mapping = Map(
                context=Ctx(),
                func=lambda x: cast(Callable[[Any, Any], Any], op)(x, right),
                data_func=lambda data: cast(Callable[[Any, Any], Any], op)(data, right),
            )
            return self[mapping]

        if len(inspect.getfullargspec(op).args) == 1:
            op = cast(Callable[[Any], Any], op)
            mapping = Map(
                context=Ctx(),
                func=op,
                data_func=op,
            )
            return self[mapping]

        op = cast(Callable[[Any, Any], Any], op)
        reduction = Reduce(
            context=Ctx(),
            func=op,
            data_func=op,
        )
        assert issubclass(self.common_value_type, tuple)
        return self[reduction]

    # Comparison:

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, sqla.ColumnElement, CtxT2, *CtxTt2],
        other: Data[ValT3, IdxT3, Any, sqla.ColumnElement, CtxT2, *CtxTt2],
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        sqla.ColumnElement,
        CtxT2,
        *CtxTt2,
    ]: ...

    @overload
    def __eq__(
        self: Data[Any, Any, Any, Any, CtxT2, *CtxTt2],
        other: Data[ValT3, IdxT3, Any, Any, CtxT2, *CtxTt2],
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        pl.Series,
        CtxT2,
        *CtxTt2,
    ]: ...

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, AnyIdx[*KeyTt2], Any, sqla.ColumnElement],
        other: Any,
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        pl.Series,
        CtxT,
        *CtxTt,
    ]: ...

    @overload
    def __eq__(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Any],
        other: Any,
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        pl.Series,
        CtxT,
        *CtxTt,
    ]: ...

    def __eq__(  # noqa: D105 # pyright: ignore[reportIncompatibleMethodOverride]
        self: Data,
        other: Any,
    ) -> (
        Data[
            KeepVal,
            PassIdx,
            R,
            sqla.ColumnElement | pl.Series,
        ]
        | bool
    ):
        if not isinstance(other, Data):
            return self._map_reduce_operator(operator.eq, other)

        alignment = self @ other

        return Filter(
            context=self.context, bool_data=alignment._map_reduce_operator(operator.eq)
        )

    @overload
    def isin(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, AnyIdx[*KeyTt2], Any, sqla.ColumnElement],
        other: Collection[ValT2] | slice,
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        sqla.ColumnElement,
        CtxT,
        *CtxTt,
    ]: ...

    @overload
    def isin(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Any],
        other: Collection[ValT2] | slice,
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        pl.Series,
        CtxT,
        *CtxTt,
    ]: ...

    def isin(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Any],
        other: Collection[ValT2] | slice,
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        sqla.ColumnElement | pl.Series,
    ]:
        """Test values of this dataset for membership in the given iterable."""
        if isinstance(other, slice):
            mapping = Map[bool](
                context=Ctx(),
                func=lambda x: other.start <= x <= other.stop,
                data_func={
                    pl.Series: lambda x: other.start <= x <= other.stop,
                    sqla.ColumnElement: partial(
                        sqla.ColumnElement.between, cleft=other.start, cright=other.stop
                    ),
                },
            )
        else:
            mapping = Map[bool](
                context=Ctx(),
                func=lambda x: x in other,
                data_func={
                    pl.Series: lambda x: pl.Series.is_in(x, other),
                    sqla.ColumnElement: partial(sqla.ColumnElement.in_, other=other),
                },
            )

        return Filter(
            context=self.context,
            bool_data=mapping,
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


TupT = TypeVar("TupT", bound=tuple, covariant=True)


@dataclass
class Align(Data[TupT, IdxT, CrudT, DataT, CtxT]):
    """Alignment of multiple props."""

    data: tuple[Data[Any, IdxT, CrudT, Any, CtxT], ...]

    @cached_prop
    def value_types(self) -> tuple[SingleTypeDef[ValT] | UnionType, ...]:
        """Get the value types."""
        return tuple(d.value_typeform for d in self.data)


@dataclass(kw_only=True)
class Map(
    Data[
        ValT,
        PassIdx,
        R,
        DataT,
        Ctx[ArgT, Idx[*tuple[Any, ...]], ArgDataT, Any],
    ]
):
    """Apply a mapping function to a dataset."""

    func: Callable[[ArgT], ValT]
    data_func: (
        Callable[[ArgDataT], DataT]
        | Mapping[type[ArgDataT], Callable[[ArgDataT], DataT]]
    ) = field(default_factory=dict)


@dataclass(kw_only=True)
class Reduce(
    Data[
        ValT,
        PassIdx,
        R,
        DataT,
        Ctx[tuple[ArgT, ...], Idx[*tuple[Any, ...]], ArgDataT, Any],
    ]
):
    """Apply a mapping function to a dataset."""

    func: Callable[[ArgT, ArgT], ValT]
    data_func: (
        Callable[[ArgDataT, ArgDataT], DataT]
        | Mapping[type[ArgDataT], Callable[[ArgDataT, ArgDataT], DataT]]
    ) = field(default_factory=dict)


@dataclass(kw_only=True)
class Aggregate(
    Data[
        ValT,
        PassIdx[KeepIdxT, SubIdxT],
        R,
        DataT,
        Ctx[ArgT, Idx[*tuple[Any, ...]], ArgDataT, Any],
    ]
):
    """Apply a mapping function to a dataset."""

    func: Callable[[Iterable[ArgT]], ValT]
    data_func: (
        Callable[[ArgDataT], DataT]
        | Mapping[type[ArgDataT], Callable[[ArgDataT], DataT]]
    ) = field(default_factory=dict)

    keep_levels: type[KeepIdxT] | None = None
    agg_levels: type[SubIdxT] | None = None


RuTi = TypeVar("RuTi", bound=R | U, default=R)


@dataclass(kw_only=True)
class Filter(
    Data[
        KeepVal,
        PassIdx,
        RuTi,
        DataT,
        CtxT,
    ]
):
    """Filter a dataset."""

    bool_data: Data[bool, Any, Any, DataT, CtxT]

    @staticmethod
    @overload
    def from_keymap(
        keymap: Mapping[
            Data[Any, Any, Any, sqla.ColumnElement, CtxT], slice | Collection
        ],
    ) -> Filter[Any, sqla.ColumnElement, CtxT]: ...

    @staticmethod
    @overload
    def from_keymap(
        keymap: Mapping[Data[Any, Any, Any, pl.Series, CtxT], slice | Collection],
    ) -> Filter[Any, pl.Series, CtxT]: ...

    @staticmethod
    def from_keymap(
        keymap: (
            Mapping[Data[Any, Any, Any, sqla.ColumnElement, CtxT], slice | Collection]
            | Mapping[Data[Any, Any, Any, pl.Series, CtxT], slice | Collection]
        ),
    ) -> Filter[Any, sqla.ColumnElement | pl.Series, CtxT]:
        """Construct filter from index key map."""
        bool_data = reduce(
            operator.and_, (idx.isin(filt) for idx, filt in keymap.items())
        )

        return Filter(context=bool_data.root, bool_data=bool_data)


@dataclass(kw_only=True)
class Prop(Data[ValT, Idx[*KeyTt2, *KeyTt3], CrudT, DataT, BaseT, CtxT, *CtxTt]):
    """Property definition for a model."""

    # Attributes:

    init_after: ClassVar[set[type[Prop]]] = set()

    _owner: type[CtxT] | None = None

    alias: str | None = None
    default: ValT | InputData[ValT, DataT] | Literal[Not.defined] = Not.defined
    default_factory: Callable[[], ValT | InputData[ValT, DataT]] | None = None
    init: bool = True
    repr: bool = True
    hash: bool = True
    compare: bool = True

    # Extension methods:

    def _value_get(
        self, instance: CtxT
    ) -> ValT | Literal[Not.handled, Not.resolved, Not.defined]:
        """Get the scalar value of this property given an object instance."""
        return Not.handled

    def _value_set(
        self: Prop[Any, Any, C | U], instance: CtxT, value: Any
    ) -> None | Literal[Not.handled]:
        """Set the scalar value of this property given an object instance."""
        return Not.handled

    # Ownership:

    def __set_name__(self, owner: type[CtxT], name: str) -> None:  # noqa: D105
        if self.alias is None:
            self.alias = name

        if self._owner is None:
            self._owner = owner

        if self.context is None:
            self.context = owner

        if self.typeref is None:
            self.typeref = hint_to_typedef(
                get_annotations(owner)[name],
                typevar_map=self.owner_typeargs,
                ctx_module=self.owner_module,
            )

    @cached_prop
    def owner(self: Prop[*tuple[Any, ...], CtxT2]) -> type[CtxT2]:
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


@dataclass
class Path(Data[ValT, IdxT, CrudT, DataT, BaseT, CtxT, *CtxTt]):
    """Alignment of multiple props."""

    props: (
        tuple[Data[ValT, IdxT, CrudT, DataT, BaseT, CtxT],]
        | tuple[
            Data[Any, Any, CrudT, sqla.FromClause | None, BaseT, CtxT],
            *tuple[Prop[Any, Any, CrudT, sqla.FromClause | None, BaseT, Any], ...],
            Prop[ValT, Any, CrudT, DataT, BaseT, Any],
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


RecSqlT = TypeVar("RecSqlT", bound=sqla.CTE | None, covariant=True, default=None)


@dataclass
class Recursion(Data[ValT, Idx[*tuple[Any, ...]], R, RecSqlT, BaseT, CtxT, *CtxTt]):
    """Combination of multiple props."""

    paths: tuple[Path[ValT, Any, Any, Any, BaseT, CtxT], ...]
