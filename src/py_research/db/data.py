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
from inspect import getmodule
from types import NoneType, UnionType
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

SubShapeT = TypeVar("SubShapeT", bound="Shape | None", default=Any)


class Shape(Generic[SubShapeT]):
    """Base for frame data types."""


@final
class Col(Shape[None]):
    """Singleton to mark standard columnar data."""


@final
class Tab(Shape[Col]):
    """Singleton to mark standard tabular data (consisting only of columns)."""


@final
class Tabs(Shape[Tab]):
    """Singleton to mark stacked tabular data (multiple tables or cols)."""


ShapeT = TypeVar(
    "ShapeT",
    bound=Shape,
    covariant=True,
    default=Shape,
)
ShapeT2 = TypeVar(
    "ShapeT2",
    bound=Shape,
)
ShapeT3 = TypeVar(
    "ShapeT3",
    bound=Shape,
)


DxT = TypeVar("DxT", bound="Dx", covariant=True, default=Any)
DxT2 = TypeVar(
    "DxT2",
    bound="Dx",
)


class PL:
    """Polars engine."""


@final
class SQL(PL):
    """SQLA engine, supporting polars fallback."""


EngineT = TypeVar("EngineT", bound=PL, default=Any, covariant=True)
EngineT2 = TypeVar("EngineT2", bound=PL)


class Dx(Generic[EngineT, ShapeT]):
    """Data exchange type."""


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
    default=Any,
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

RwxT = TypeVar("RwxT", bound=CRUD, default=Any, contravariant=True)
RwxT2 = TypeVar("RwxT2", bound=CRUD)
RwxT3 = TypeVar("RwxT3", bound=CRUD)

ArgT = TypeVar("ArgT", contravariant=True, default=Any)
ArgIdxT = TypeVar("ArgIdxT", contravariant=True, bound=Idx, default=Any)


class Ctx(Generic[ArgT, ArgIdxT, DxT, RwxT]):
    """Data context."""


CtxT = TypeVar("CtxT", bound=Ctx, default=Any, covariant=True)
CtxT2 = TypeVar("CtxT2", bound=Ctx)
CtxT3 = TypeVar("CtxT3", bound=Ctx)

CtxTt = TypeVarTuple("CtxTt", default=Unpack[tuple[Any, ...]])
CtxTt2 = TypeVarTuple("CtxTt2")
CtxTt3 = TypeVarTuple("CtxTt3")


class Base(Ctx[ArgT, Any, Any, RwxT], ABC):
    """Base for retrieving/storing data."""

    @property
    @abstractmethod
    def connection(self) -> sqla.engine.Connection:
        """SQLAlchemy connection to the database."""
        ...

    @abstractmethod
    def registry[T: AutoIndexable](
        self: Base[T, Any], value_type: type[T]
    ) -> Registry[T, RwxT, Base[T, RwxT]]:
        """Get the registry for a type in this base."""
        ...


BaseT = TypeVar("BaseT", bound=Base, covariant=True, default=Any)


class Interface(Ctx[ArgT, ArgIdxT, DxT, RwxT]):
    """Data interface."""


type InputData[V, S] = V | Iterable[V] | Mapping[
    Hashable, V
] | pd.DataFrame | pl.DataFrame | S | Data[V]

Params = ParamSpec("Params")


def _get_prop_type(hint: SingleTypeDef | str) -> type[Data] | type[None]:
    """Resolve the prop typehint."""
    if has_type(hint, SingleTypeDef):
        base = get_origin(hint)
        if base is None or not issubclass(base, Data):
            return NoneType

        return base
    elif isinstance(hint, str):
        return _map_data_type_name(hint)
    else:
        return NoneType


@cache
def _data_type_name_map() -> dict[str, type[Data]]:
    return {cls.__name__: cls for cls in get_subclasses(Data) if cls is not Data}


def _map_data_type_name(name: str) -> type[Data | None]:
    """Map property type name to class."""
    name_map = _data_type_name_map()
    matches = [name_map[n] for n in name_map if name.startswith(n + "[")]
    return matches[0] if len(matches) == 1 else NoneType


class Frame(Generic[EngineT, ShapeT]):
    """Raw data."""

    @overload
    def __init__(self: Frame[SQL, Col], data: sqla.ColumnElement) -> None: ...

    @overload
    def __init__(self: Frame[SQL, Tab | Tabs], data: sqla.Select) -> None: ...

    @overload
    def __init__(
        self: Frame[PL, Col], data: pl.Series | sqla.ColumnElement
    ) -> None: ...

    @overload
    def __init__(
        self: Frame[PL, Tab | Tabs], data: pl.DataFrame | sqla.Select
    ) -> None: ...

    def __init__(self: Frame, data: Any = None) -> None:  # noqa: D107
        self._data = data

    @overload
    def get(
        self: Frame[SQL, Col],
    ) -> sqla.ColumnElement: ...

    @overload
    def get(
        self: Frame[SQL, Tab | Tabs],
    ) -> sqla.Select: ...

    @overload
    def get(self: Frame[Any, Col]) -> pl.Series | sqla.ColumnElement: ...

    @overload
    def get(
        self: Frame[Any, Tab | Tabs],
    ) -> pl.DataFrame | sqla.Select: ...

    @overload
    def get(
        self: Frame[Any, Any],
    ) -> pl.Series | pl.DataFrame | sqla.ColumnElement | sqla.Select: ...

    def get(self: Frame) -> Any:
        """Get the raw data."""
        return self._data


def coalescent_union(
    left_frame: Frame[EngineT2, ShapeT2],
    right_frame: Frame[EngineT2, ShapeT2],
    coalesce: Literal["left", "right"] = "left",
) -> Frame[EngineT2, ShapeT2]:
    """Union two data instances and coalesce their columns."""
    left, right = left_frame.get(), right_frame.get()

    if isinstance(left, pl.Series) and isinstance(right, pl.Series):
        if isinstance(left.dtype, pl.Boolean) and isinstance(right.dtype, pl.Boolean):
            return cast(Frame[EngineT2, ShapeT2], Frame(left | right))

        return cast(
            Frame[EngineT2, ShapeT2],
            Frame(
                pl.DataFrame([left.rename("left"), right.rename("right")]).select(
                    {
                        left.name: (
                            pl.coalesce("left", "right")
                            if coalesce == "left"
                            else pl.coalesce("right", "left")
                        )
                    }
                )[left.name]
            ),
        )

    if isinstance(left, sqla.ColumnElement) and isinstance(right, sqla.ColumnElement):
        if isinstance(left.type, sqla.Boolean) and isinstance(right.type, sqla.Boolean):
            return cast(Frame[EngineT2, ShapeT2], Frame(left | right))

        return cast(
            Frame[EngineT2, ShapeT2],
            Frame(
                sqla.func.coalesce(left, right)
                if coalesce == "left"
                else sqla.func.coalesce(right, left)
            ),
        )

    if isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame):
        all_cols = set(left.columns) | set(right.columns)

        return cast(
            Frame[EngineT2, ShapeT2],
            Frame(
                pl.concat(
                    [
                        left.select(pl.all().name.prefix("left")),
                        right.select(pl.all().name.prefix("right")),
                    ],
                    how="horizontal",
                ).select(
                    {
                        col: (
                            f"left.{col}"
                            if col not in right.columns
                            else (
                                f"right.{col}"
                                if col not in left.columns
                                else (
                                    pl.coalesce(f"left.{col}", f"right.{col}")
                                    if coalesce == "left"
                                    else pl.coalesce(f"right.{col}", f"left.{col}")
                                )
                            )
                        )
                        for col in all_cols
                    }
                )
            ),
        )

    if isinstance(left, sqla.Select) and isinstance(right, sqla.Select):
        all_cols = set(left.selected_columns.keys()) | set(
            right.selected_columns.keys()
        )

        left_froms = left.get_final_froms()
        right_froms = right.get_final_froms()
        assert len(left_froms) == 1 and len(right_froms) == 1
        assert left_froms[0] is right_froms[0]

        common_from = left_froms[0]

        return cast(
            Frame[EngineT2, ShapeT2],
            Frame(
                sqla.select(
                    *(
                        (
                            left.c[col]
                            if col not in right.selected_columns.keys()
                            else (
                                right.c[col]
                                if col not in left.selected_columns.keys()
                                else (
                                    sqla.func.coalesce(left.c[col], right.c[col])
                                    if coalesce == "left"
                                    else sqla.func.coalesce(right.c[col], left.c[col])
                                )
                            )
                        )
                        for col in all_cols
                    )
                ).select_from(common_from),
            ),
        )

    raise ValueError(
        "Incompatible data types for coalescent union: "
        f"{type(left_frame)}, {type(right_frame)}"
    )


def frame_isin(
    frame: Frame[EngineT2, Col],
    values: Collection | slice,
) -> Frame[EngineT2, Col]:
    """Check if the values are in the frame."""
    data = frame.get()
    series = None
    column = None

    if isinstance(values, slice):
        if isinstance(data, pl.Series):
            series = values.start <= data <= values.stop
        else:
            column = data.between(values.start, values.stop)
    else:
        if isinstance(data, pl.Series):
            series = data.is_in(values)
        else:
            column = data.in_(values)

    data = series if series is not None else column
    assert data is not None

    return cast(Frame[EngineT2, Col], Frame(data))


@dataclass(kw_only=True)
class Data(Generic[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt]):
    """Property definition for a model."""

    # Core attributes:

    context: CtxT | Data[Any, Any, Any, Any, CtxT]
    typeref: TypeRef[Data[ValT]] | None = None

    # Extension methods:

    def _name(self) -> str:
        """Name of the property."""
        return gen_str_hash(self)

    def _frame(
        self: Data[Any, Any, Any, Dx[EngineT2, ShapeT2]],
    ) -> Frame[EngineT2, ShapeT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()

    def _value(self, data: Mapping[str, Any]) -> ValT:
        """Transform dict-like data (e.g. dataframe row) to declared value type."""
        raise NotImplementedError()

    def _index(
        self: Data[Any, FullIdx[*KeyTt2] | PassIdx[Any, Any, Idx[*KeyTt2]]],
    ) -> (
        Data[
            tuple[*KeyTt2],
            SelfIdx[*KeyTt2],
            R,
            Dx[PL | SQL, Tab],
            CtxT,
            *CtxTt,
            Ctx[ValT, Any, DxT, RwxT],
        ]
        | None
    ):
        """Get the index of this data."""
        raise NotImplementedError()

    def _subframes(
        self: Data[
            tuple[ValT2, ...], AnyIdx[*KeyTt2], Any, Dx[EngineT2, Shape[ShapeT3]]
        ],
    ) -> tuple[
        Data[
            ValT2,
            Idx[*KeyTt2],
            RwxT,
            Dx[EngineT2, ShapeT3],
            CtxT,
            *CtxTt,
            Ctx[tuple[ValT2, ...], Idx[*KeyTt2], DxT, RwxT],
        ],
        ...,
    ]:
        raise NotImplementedError()

    def _mutation(
        self: Data[Any, Any, RwxT2, *tuple[Any, ...]],
        input_data: InputData[ValT, ShapeT],
        mode: set[type[RwxT2]] = {C, U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        raise NotImplementedError()

    # Type:

    @cached_prop
    def resolved_type(self) -> SingleTypeDef[Data[ValT]]:
        """Resolved type of this prop."""
        if self.typeref is None:
            return cast(SingleTypeDef[Data[ValT]], type(self))

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

    @cached_prop
    def index(self: Data[Any, AnyIdx[*KeyTt2], Any, Dx[EngineT2]]) -> Data[
        tuple[*KeyTt2],
        SelfIdx[*KeyTt2],
        R,
        Dx[EngineT2, Tab],
        CtxT,
        *CtxTt,
    ]:
        """Get the index of this data."""
        # TODO: implement, resolve keep-idx, sub-idx and add-idx config
        raise NotImplementedError()

    def _map_index_selectors(
        self, sel: list | slice | tuple[list | slice, ...]
    ) -> Mapping[Data[Any, Any, Any, Dx[SQL | PL, Col], CtxT], slice | Collection]:
        # TODO: implement
        raise NotImplementedError()

    # SQL:

    @overload
    def select(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, Dx[SQL]],
    ) -> sqla.Select: ...

    @overload
    def select(self: Data[Any, Any, Any, Dx[PL]]) -> None: ...

    def select(self: Data) -> sqla.SelectBase | None:
        """Return select statement for this dataset."""
        frame = self._frame().get()

        match frame:
            case sqla.Select():
                return frame
            case sqla.FromClause():
                return frame.select()
            case sqla.ColumnElement():
                return sqla.select(frame)
            case _:
                return None

    @overload
    def select_str(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, Dx[SQL]],
    ) -> str: ...

    @overload
    def select_str(
        self: Data[Any, Any, Any, Dx[PL]],
    ) -> None: ...

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
    def query(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, Dx[SQL]],
    ) -> sqla.Subquery: ...

    @overload
    def query(
        self: Data[Any, Any, Any, Dx[PL]],
    ) -> None: ...

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
        self: Data[Any, Any, Any, Dx, Base],
    ) -> pl.DataFrame:
        """Load dataset as dataframe."""
        frame = self._frame().get()

        if isinstance(frame, pl.DataFrame):
            return frame
        elif isinstance(frame, pl.Series):
            return frame.to_frame()

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
        return self.index.values()

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
        frame = self._frame().get()
        if isinstance(frame, pl.Series | pl.DataFrame):
            return len(frame)

        query = self.query()
        assert query is not None

        count = self.root.connection.execute(
            sqla.select(sqla.func.count()).select_from(query)
        ).scalar()
        assert count is not None

        return count

    # Application:

    # 1. Context application, altered index, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], RwxT2, Dx[EngineT2]],
        key: Data[
            KeepVal,
            PassIdx[Idx[*KeyTt2], Idx[*KeyTt4], Idx[*KeyTt3]],
            RwxT2,
            Dx[EngineT2, ShapeT3],
            Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], Any, Any],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt2, *KeyTt3],
        RwxT2,
        Dx[EngineT2, ShapeT3],
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], DxT, RwxT],
        *CtxTt3,
    ]: ...

    # 2. Context application, altered index, new value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], RwxT2, Dx[EngineT2]],
        key: Data[
            ValT3,
            PassIdx[Idx[*KeyTt2], Idx[*KeyTt4], Idx[*KeyTt3]],
            RwxT2,
            Dx[EngineT2, ShapeT3],
            Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], Any, Any],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt2, *KeyTt3],
        RwxT2,
        Dx[EngineT2, ShapeT3],
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], DxT, RwxT],
        *CtxTt3,
    ]: ...

    # 3. Context application, new index, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2], RwxT2, Dx[EngineT2]],
        key: Data[
            KeepVal,
            FullIdx[*KeyTt3],
            RwxT2,
            Dx[EngineT2, ShapeT3],
            Ctx[ValT2, Idx[*KeyTt2], Any, Any],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt3],
        RwxT2,
        Dx[EngineT2, ShapeT3],
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2], DxT, RwxT],
        *CtxTt3,
    ]: ...

    # 4. Context application, new index, new value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2], RwxT2, Dx[EngineT2]],
        key: Data[
            ValT3,
            FullIdx[*KeyTt3],
            RwxT2,
            Dx[EngineT2, ShapeT3],
            Ctx[ValT2, Idx[*KeyTt2], Any, Any],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt3],
        RwxT2,
        Dx[EngineT2, ShapeT3],
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2], DxT, RwxT],
        *CtxTt3,
    ]: ...

    # 5. Key list / slice filtering, scalar index type
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], RU],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, RU, DxT, CtxT, *CtxTt]: ...

    # 6. Key list / slice filtering
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], RU],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, RU, DxT, CtxT, *CtxTt]: ...

    # 7. Key list / slice filtering, scalar index type, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], R],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, R, DxT, CtxT, *CtxTt]: ...

    # 8. Key list / slice filtering, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], R],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, R, DxT, CtxT, *CtxTt]: ...

    # 9. Key selection
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt3, *KeyTt2]],
        key: tuple[*KeyTt3],
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        RwxT,
        DxT,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[*KeyTt3, *KeyTt2], DxT, RwxT],
    ]: ...

    # 10. Key selection, scalar
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT3, *KeyTt2]],
        key: KeyT3,
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        RwxT,
        DxT,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[KeyT3, *KeyTt2], DxT, RwxT],
    ]: ...

    # 11. Base type selection
    @overload
    def __getitem__(
        self: Base[ValT2, RwxT2],
        key: type[ValT2],
    ) -> Data[ValT2, IdxT, RwxT, DxT, CtxT, *CtxTt]: ...

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
                    return Reduce(
                        context=alignment,
                        func=operator.or_,
                        frame_func=partial(coalescent_union, coalesce="left"),
                    )

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
        self: Data[Any, Any, RwxT2, Dx[EngineT2, Col], CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, RwxT2, Dx[EngineT2, Tab], CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        RwxT2,
        Dx[EngineT2, Tab],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, RwxT2, Dx[EngineT2], CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, RwxT2, Dx[EngineT2], CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        RwxT2,
        Dx[EngineT2, Tabs],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[tuple[*ValTt2], Any, RwxT2, Dx[EngineT2, Tab], CtxT2],
        other: Data[ValT3, IdxT3, RwxT2, Dx[EngineT2, Col], CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        RwxT2,
        Dx[EngineT2, Tab],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, RwxT2, Dx[EngineT2], CtxT2],
        other: Data[ValT3, IdxT3, RwxT2, Dx[EngineT2], CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        RwxT2,
        Dx[EngineT2, Tabs],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, RwxT2, Dx[EngineT2, Col], CtxT2],
        other: Data[ValT3, IdxT3, RwxT2, Dx[EngineT2, Col], CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        RwxT2,
        Dx[EngineT2, Tab],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, RwxT2, Dx[EngineT2], CtxT2],
        other: Data[ValT3, IdxT3, RwxT2, Dx[EngineT2], CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        RwxT2,
        Dx[EngineT2, Tabs],
        CtxT2,
    ]: ...

    def __matmul__(
        self: Data[Any, Any, RwxT2, Dx[EngineT2], CtxT2],
        other: Data[Any, IdxT3, RwxT2, Dx[EngineT2], CtxT2],
    ) -> Data[
        tuple,
        IdxT | IdxT3,
        RwxT2,
        Dx[EngineT2, Tab | Tabs],
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
            RwxT2,
            Dx[EngineT2, Tab | Tabs],
            CtxT2,
        ](
            data=self_data + other_data,
            context=self.root,
        )

    # Reduction:

    @overload
    def _map_reduce_operator(
        self: Data[
            tuple[ValT2, ...], AnyIdx[*KeyTt2], Any, Dx[EngineT2, Shape[ShapeT3]]
        ],
        op: Callable[[ValT2, ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        Dx[EngineT2, ShapeT3],
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DxT, RwxT],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], AnyIdx[*KeyTt2], Any, Dx[EngineT2, ShapeT3]],
        op: Callable[[ValT2, ValT4], ValT3],
        right: ValT4,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        Dx[EngineT2, ShapeT3],
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DxT, RwxT],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[ValT2, AnyIdx[*KeyTt2], Any, Dx[EngineT2, ShapeT3]],
        op: Callable[[ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        R,
        Dx[EngineT2, ShapeT3],
        CtxT,
        Ctx[ValT, Idx[*KeyTt2], DxT, RwxT],
    ]: ...

    def _map_reduce_operator(
        self: Data,
        op: Callable[[Any, Any], Any] | Callable[[Any], Any],
        right: Any | Literal[Not.defined] = Not.defined,
    ) -> Data[
        Any,
        Any,
        R,
        Any,
        CtxT,
        Ctx[ValT, Any, DxT, RwxT],
    ]:
        """Create a scalar comparator for the given operation."""
        if right is not Not.defined:
            mapping = Map(
                context=Ctx(),
                func=lambda x: cast(Callable[[Any, Any], Any], op)(x, right),
                frame_func=lambda frame: cast(Callable[[Any, Any], Any], op)(
                    frame.get(), right
                ),
            )
            return cast(
                Data[
                    Any,
                    Any,
                    R,
                    Any,
                    CtxT,
                    Ctx[ValT, Any, DxT, RwxT],
                ],
                self[mapping],
            )

        if len(inspect.getfullargspec(op).args) == 1:
            op = cast(Callable[[Any], Any], op)
            mapping = Map(
                context=Ctx(),
                func=op,
                frame_func=lambda frame: op(frame.get()),
            )
            return cast(
                Data[
                    Any,
                    Any,
                    R,
                    Any,
                    CtxT,
                    Ctx[ValT, Any, DxT, RwxT],
                ],
                self[mapping],
            )

        op = cast(Callable[[Any, Any], Any], op)
        reduction = Reduce(
            context=Ctx(),
            func=op,
            frame_func=lambda left, right: op(left.get(), right.get()),
        )
        assert issubclass(self.common_value_type, tuple)
        return cast(
            Data[
                Any,
                Any,
                R,
                Any,
                CtxT,
                Ctx[ValT, Any, DxT, RwxT],
            ],
            self[reduction],
        )

    # Comparison:

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, Dx[EngineT2, Col], CtxT2, *CtxTt2],
        other: Data[ValT3, IdxT3, Any, Dx[EngineT2, Col], CtxT2, *CtxTt2],
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        DxT,
        CtxT,
        *CtxTt,
    ]: ...

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, AnyIdx[*KeyTt2], Any, Dx[Any, Col]],
        other: Any,
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        DxT,
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
            DxT,
            CtxT,
            *CtxTt,
        ]
        | bool
    ):
        if not isinstance(other, Data):
            return cast(
                Data[
                    KeepVal,
                    PassIdx,
                    R,
                    DxT,
                    CtxT,
                    *CtxTt,
                ],
                Filter(
                    context=self.context,
                    bool_data=self._map_reduce_operator(operator.eq, other),
                ),
            )

        alignment = self @ other

        return cast(
            Data[
                KeepVal,
                PassIdx,
                R,
                DxT,
                CtxT,
                *CtxTt,
            ],
            Filter(
                context=self.context,
                bool_data=alignment._map_reduce_operator(operator.eq),
            ),
        )

    def isin(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Dx[Any, Col]],
        other: Collection[ValT2] | slice,
    ) -> Data[
        KeepVal,
        PassIdx,
        R,
        DxT,
        CtxT,
        *CtxTt,
    ]:
        """Test values of this dataset for membership in the given iterable."""
        if isinstance(other, slice):
            mapping = Map[bool](
                context=Ctx(),
                func=lambda x: other.start <= x <= other.stop,
                frame_func=partial(frame_isin, values=other),
            )
        else:
            mapping = Map[bool](
                context=Ctx(),
                func=lambda x: x in other,
                frame_func=partial(frame_isin, values=other),
            )

        return cast(
            Data[
                KeepVal,
                PassIdx,
                R,
                DxT,
                CtxT,
                *CtxTt,
            ],
            Filter(
                context=self.context,
                bool_data=mapping,
            ),
        )

    # Index set operations:

    def __or__(
        self: Data[ValT2, AnyIdx[*KeyTt2], Any, DxT2, CtxT2],
        other: Data[ValT3, AnyIdx[*KeyTt2], Any, DxT2, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        PassIdx,
        R,
        DxT2,
        Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any, Any],
        CtxT2,
    ]:
        """Union two databases, right overriding left."""
        alignment = self @ other
        return cast(
            Data,
            Reduce(
                context=alignment,
                func=operator.or_,
                frame_func=partial(coalescent_union, coalesce="left"),
            ),
        )

    def __xor__(
        self: Data[ValT2, AnyIdx[*KeyTt2], Any, DxT2, CtxT2],
        other: Data[ValT3, AnyIdx[*KeyTt2], Any, DxT2, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        PassIdx,
        R,
        DxT2,
        Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any, Any],
        CtxT2,
    ]:
        """Union two databases, right overriding left."""
        alignment = self @ other
        return cast(
            Data,
            Reduce(
                context=alignment,
                func=operator.xor,
                frame_func=partial(coalescent_union, coalesce="right"),
            ),
        )

    def __lshift__(
        self: Data[ValT2, AnyIdx[*KeyTt2], Any, DxT2, CtxT2],
        other: Data[ValT3, AnyIdx[*KeyTt2], Any, DxT2, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        PassIdx,
        R,
        DxT2,
        Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any, Any],
        CtxT2,
    ]:
        """Union two databases, right overriding left."""
        alignment = Align(
            context=self.root,
            data=(self, other),
            join="left",
        )
        return cast(
            Data,
            Reduce(
                context=alignment,
                func=operator.and_,
                frame_func=partial(coalescent_union, coalesce="right"),
            ),
        )

    def __ior__(
        self: Data[Any, Any, C | U, Dx[Any, ShapeT2], Base],
        input_data: InputData[ValT, ShapeT2],
    ) -> Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        mutations = self._mutation(input_data, mode={C, U})

        for mutation in mutations:
            self.root.connection.execute(mutation)

        return cast(Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt], self)

    def __ixor__(
        self: Data[Any, Any, C, Dx[Any, ShapeT2], Base],
        input_data: InputData[ValT, ShapeT2],
    ) -> Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        mutations = self._mutation(input_data, mode={C})

        for mutation in mutations:
            self.root.connection.execute(mutation)

        return cast(Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt], self)

    def __ilshift__(
        self: Data[Any, Any, U, Dx[Any, ShapeT2], Base],
        input_data: InputData[ValT, ShapeT2],
    ) -> Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        mutations = self._mutation(input_data, mode={U})

        for mutation in mutations:
            self.root.connection.execute(mutation)

        return cast(Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt], self)

    def __setitem__(
        self: Data[Any, Any, C | U | D, Dx[Any, ShapeT2], Base],
        key: slice,
        input_data: InputData[ValT, ShapeT2],
    ) -> Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        assert key == slice(None, None, None)

        if input_data is not self:
            mutations = self._mutation(input_data, mode={C, U, D})

            for mutation in mutations:
                self.root.connection.execute(mutation)

        return cast(Data[ValT, IdxT, RwxT, DxT, CtxT, *CtxTt], self)


RegT = TypeVar("RegT", covariant=True, bound=AutoIndexable)


@dataclass(kw_only=True)
class Registry(Data[RegT, AutoIdx[RegT], RwxT, Dx[SQL, Tab], BaseT, None]):
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


@dataclass(kw_only=True)
class Align(Data[TupT, IdxT, RwxT, DxT, CtxT]):
    """Alignment of multiple props."""

    data: tuple[Data[Any, IdxT, RwxT, Any, CtxT], ...]
    join: Literal["left", "right", "full"] = "full"

    @cached_prop
    def value_types(self) -> tuple[SingleTypeDef[ValT] | UnionType, ...]:
        """Get the value types."""
        return tuple(d.value_typeform for d in self.data)


ArgShapeT = TypeVar("ArgShapeT", bound=Shape, covariant=True, default=Any)


@dataclass(kw_only=True)
class Map(
    Data[
        ValT,
        PassIdx,
        R,
        Dx[EngineT, ShapeT],
        Ctx[ArgT, Any, Dx[EngineT, ArgShapeT]],
    ],
    Generic[ArgT, ArgShapeT, ValT, ShapeT, EngineT],
):
    """Apply a mapping function to a dataset."""

    func: Callable[[ArgT], ValT]
    frame_func: Callable[[Frame[EngineT, ArgShapeT]], Frame[EngineT, ShapeT]] | None = (
        None
    )


@dataclass(kw_only=True)
class Reduce(
    Data[
        ValT,
        PassIdx,
        R,
        Dx[EngineT, ShapeT],
        Ctx[tuple[ArgT, ...], Any, Dx[EngineT, ArgShapeT]],
    ],
    Generic[ArgT, ArgShapeT, ValT, ShapeT, EngineT],
):
    """Apply a mapping function to a dataset."""

    func: Callable[[ArgT, ArgT], ValT]
    frame_func: (
        Callable[
            [Frame[EngineT, ArgShapeT], Frame[EngineT, ArgShapeT]],
            Frame[EngineT, ShapeT],
        ]
        | None
    ) = None


@dataclass(kw_only=True)
class Agg(
    Data[
        ValT,
        PassIdx[KeepIdxT, SubIdxT],
        R,
        Dx[EngineT, ShapeT],
        Ctx[ArgT, Any, Dx[EngineT, ArgShapeT]],
    ],
    Generic[ArgT, KeepIdxT, SubIdxT, ArgShapeT, ValT, ShapeT, EngineT],
):
    """Apply a mapping function to a dataset."""

    func: Callable[[Iterable[ArgT]], ValT]
    frame_func: Callable[[Frame[EngineT, ArgShapeT]], Frame[EngineT, ShapeT]] | None = (
        None
    )

    keep_levels: type[KeepIdxT] | None = None
    agg_levels: type[SubIdxT] | None = None


RuTi = TypeVar("RuTi", bound=R | U, default=R)


@dataclass(kw_only=True)
class Filter(
    Data[
        KeepVal,
        PassIdx,
        RuTi,
        Dx[EngineT, ShapeT],
        CtxT,
    ]
):
    """Filter a dataset."""

    bool_data: Data[bool, Any, Any, Dx[EngineT, ShapeT], CtxT]

    @staticmethod
    def from_keymap(
        keymap: Mapping[
            Data[Any, Any, Any, Dx[EngineT2, Col], CtxT], slice | Collection
        ],
    ) -> Filter[Any, EngineT2, Any, CtxT]:
        """Construct filter from index key map."""
        bool_data = reduce(
            operator.and_, (idx.isin(filt) for idx, filt in keymap.items())
        )

        return Filter(context=bool_data.root, bool_data=bool_data)
