"""Basic types for relational data expression."""

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
    Set,
)
from dataclasses import dataclass, field
from functools import partial, reduce
from inspect import getmodule
from types import UnionType
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    Unpack,
    cast,
    final,
    get_args,
    overload,
)

import networkx as nx
import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlparse
from typing_extensions import TypeVar, TypeVarTuple

from py_research.caching import cached_method, cached_prop
from py_research.data import copy_and_override
from py_research.hashing import gen_int_hash
from py_research.reflect.types import (
    SingleTypeDef,
    TypeAware,
    get_common_type,
    has_type,
    is_subtype,
    typedef_to_typeset,
)
from py_research.types import Not, Ordinal


@final
class Keep:
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

KeyT = TypeVar("KeyT", bound=Hashable, default=Any)
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


DxT = TypeVar(
    "DxT",
    bound=Shape | Keep,
    covariant=True,
    default=Shape | Keep,
)
DxT2 = TypeVar(
    "DxT2",
    bound=Shape | Keep,
)
DxT3 = TypeVar(
    "DxT3",
    bound=Shape | Keep,
)

SxT = TypeVar("SxT", bound=Shape, covariant=True, default=Shape)
SxT2 = TypeVar(
    "SxT2",
    bound=Shape,
)
SxT3 = TypeVar(
    "SxT3",
    bound=Shape,
)


class PL:
    """Polars engine."""


@final
class SQL(PL):
    """SQLA engine, supporting polars fallback."""


ExT = TypeVar("ExT", bound=PL, default=PL, covariant=True)
ExT2 = TypeVar("ExT2", bound=PL)


@final
@dataclass
class Idx(Generic[*KeyTt]):
    """Define the custom index type of a dataset."""

    components: tuple[Data[Any, Any, Col, SQL, R, Interface], ...]


@final
class SelfIdx(Generic[*KeyTt]):
    """Index by self."""


@final
class HashIdx(Generic[*KeyTt]):
    """Index by hash of self."""


class AutoIndexable(Protocol[*KeyTt]):
    """Base class for indexable objects."""

    @classmethod
    def _index_components(cls) -> tuple[Data[Any, Any, Col, SQL, R, Interface], ...]:
        """Get SQL columns for this auto-indexed type."""
        ...


AutoIdxT = TypeVar("AutoIdxT", bound=AutoIndexable, covariant=True)


@final
class AutoIdx(Generic[AutoIdxT]):
    """Index by custom value derived from self."""

    value_type: type[AutoIdxT]


type FullIdx[*K] = Idx[*K] | SelfIdx[*K] | HashIdx[*K] | AutoIdx[AutoIndexable[*K]]

FullIdxT = TypeVar(
    "FullIdxT",
    bound=FullIdx,
    default=FullIdx[*tuple[Any, ...]],
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
    covariant=True,
)


@dataclass
class ModIdx(Generic[SubIdxT, AddIdxT]):
    """Pass-through index."""

    reduction: SingleTypeDef[SubIdxT]
    expansion: AddIdxT


type KeepIdx = ModIdx[Idx[()], Idx[()]]
type Reduce[IdxT: Idx] = ModIdx[IdxT, Idx[()]]
type Expand[IdxT: Idx] = ModIdx[Idx[()], IdxT]


type AnyIdx[*K] = FullIdx[*K] | ModIdx[Any, Any]

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


@final
class FromChild:
    """Singleton to mark type as derived from child (e.g. the same)."""


ArgT = TypeVar("ArgT", contravariant=True, default=Any)
ArgIdxT = TypeVar("ArgIdxT", bound=Idx | FromChild, contravariant=True, default=Any)
ArgDxT = TypeVar("ArgDxT", bound=Shape | FromChild, contravariant=True, default=Any)


class Ctx(Generic[ArgT, ArgIdxT, ArgDxT]):
    """Data context."""


CtxT = TypeVar("CtxT", bound=Ctx, default=Any, covariant=True)
CtxT2 = TypeVar("CtxT2", bound=Ctx)
CtxT3 = TypeVar("CtxT3", bound=Ctx)

CtxTt = TypeVarTuple("CtxTt", default=Unpack[tuple[Any, ...]])
CtxTt2 = TypeVarTuple("CtxTt2")
CtxTt3 = TypeVarTuple("CtxTt3")


class Root(Ctx[ArgT, Any, Any], Generic[ArgT]):
    """Base for retrieving/storing data."""


RootT = TypeVar("RootT", bound=Root, covariant=True, default=Any)
RootT2 = TypeVar("RootT2", bound=Root)


class Base(Root[ArgT], Generic[ArgT, RwxT]):
    """Base for retrieving/storing data."""

    @property
    def connection(self: Root[SQL]) -> sqla.engine.Connection:
        """SQLAlchemy connection to the database."""
        ...

    def registry[T: AutoIndexable](
        self: Base[T], value_type: type[T]
    ) -> Registry[T, RwxT, Base[T, RwxT]]:
        """Get the registry for a type in this base."""
        ...


@dataclass
class Interface(Ctx[ArgT, ArgIdxT, ArgDxT]):
    """Data interface."""

    arg_type: SingleTypeDef[ArgT] | None = None


type InputFrame = (
    pl.DataFrame
    | pd.DataFrame
    | sqla.Select
    | sqla.FromClause
    | pl.Series
    | pd.Series
    | sqla.ColumnElement
)

type InputData[V, S, I] = Data[V] | V | Iterable[V] | Mapping[
    Hashable, V
] | S | Mapping[str, I]

Params = ParamSpec("Params")


class Frame(Generic[ExT, SxT]):
    """Raw data."""

    @overload
    def __init__(self: Frame[SQL, Col], data: sqla.ColumnElement) -> None: ...

    @overload
    def __init__(self: Frame[SQL, Tab], data: sqla.Select) -> None: ...

    @overload
    def __init__(self: Frame[SQL, Tabs], data: dict[str, sqla.Select]) -> None: ...

    @overload
    def __init__(
        self: Frame[PL, Col], data: pl.Series | sqla.ColumnElement
    ) -> None: ...

    @overload
    def __init__(self: Frame[PL, Tab], data: pl.DataFrame | sqla.Select) -> None: ...

    @overload
    def __init__(
        self: Frame[PL, Tabs], data: dict[str, pl.DataFrame] | dict[str, sqla.Select]
    ) -> None: ...

    def __init__(self: Frame, data: Any = None) -> None:  # noqa: D107
        self._data = data

    def cols(self) -> list[str]:
        """Get the column names of this frame."""
        match self._data:
            case pl.DataFrame():
                return self._data.columns
            case pl.Series():
                return [self._data.name if self._data.name is not None else "0"]
            case sqla.Select():
                return [
                    (c.key if c.key is not None else str(i))
                    for i, c in enumerate(self._data.selected_columns)
                ]
            case sqla.ColumnElement():
                return [self._data.key if self._data.key is not None else "0"]
            case dict() if has_type(self._data, dict[str, pl.DataFrame]):
                return [
                    f"{prefix}.{col}"
                    for prefix, df in self._data.items()
                    for col in df.columns
                ]
            case _:
                raise ValueError("Unsupported data type.")

    @overload
    def get(
        self: Frame[SQL, Col],
    ) -> sqla.ColumnElement: ...

    @overload
    def get(
        self: Frame[SQL, Tab],
    ) -> sqla.Select: ...

    @overload
    def get(
        self: Frame[SQL, Tabs],
    ) -> dict[str, sqla.Select]: ...

    @overload
    def get(self: Frame[PL, Col]) -> pl.Series | sqla.ColumnElement: ...

    @overload
    def get(
        self: Frame[PL, Tab],
    ) -> pl.DataFrame | sqla.Select: ...

    @overload
    def get(
        self: Frame[PL, Tabs],
    ) -> dict[str, pl.DataFrame] | dict[str, sqla.Select]: ...

    @overload
    def get(
        self: Frame[PL, Any],
    ) -> (
        pl.Series
        | pl.DataFrame
        | sqla.ColumnElement
        | sqla.Select
        | dict[str, pl.DataFrame]
        | dict[str, sqla.Select]
    ): ...

    def get(self: Frame) -> Any:
        """Get the raw data."""
        return self._data


def frame_coalesce(
    frame: Frame[ExT2, Shape[SxT2]],
    coalesce: Literal["left", "right"] = "left",
) -> Frame[ExT2, SxT2]:
    """Union two data instances and coalesce their columns."""
    data = frame.get()

    if isinstance(data, pl.DataFrame):
        if all(isinstance(d, pl.Boolean) for d in data.dtypes):
            return cast(
                Frame[ExT2, SxT2], Frame(reduce(operator.or_, data.iter_columns()))
            )

        return cast(
            Frame[ExT2, SxT2],
            Frame(
                data.select(
                    coalesced=(
                        pl.coalesce(*data.columns)
                        if coalesce == "left"
                        else pl.coalesce(*reversed(data.columns))
                    )
                )["coalesced"]
            ),
        )

    if isinstance(data, sqla.Select):
        if all(
            isinstance(c.type, sqla.types.Boolean)
            for c in data.selected_columns.values()
        ):
            return cast(
                Frame[ExT2, SxT2],
                Frame(reduce(operator.or_, data.selected_columns.values())),
            )

        return cast(
            Frame[ExT2, SxT2],
            Frame(
                (
                    sqla.func.coalesce(*data.selected_columns.values())
                    if coalesce == "left"
                    else sqla.func.coalesce(*reversed(data.selected_columns.values()))
                ).label("coalesced")
            ),
        )

    if has_type(data, dict[str, pl.DataFrame]):
        all_cols = reduce(set.union, (set(d.columns) for d in data.values()))

        return cast(
            Frame[ExT2, SxT2],
            Frame(
                pl.concat(
                    [
                        df.select(pl.all().name.prefix(prefix))
                        for prefix, df in data.items()
                    ],
                    how="horizontal",
                ).select(
                    *{
                        col: (
                            pl.coalesce(
                                *(
                                    f"{prefix}.{col}"
                                    for prefix in data.keys()
                                    if col in data[prefix].columns
                                )
                            )
                            if coalesce == "left"
                            else pl.coalesce(
                                *reversed(
                                    [
                                        f"{prefix}.{col}"
                                        for prefix in data.keys()
                                        if col in data[prefix].columns
                                    ]
                                )
                            )
                        )
                        for col in all_cols
                    }
                )
            ),
        )

    if has_type(data, dict[str, sqla.Select]):
        all_cols = reduce(
            set.union,
            (
                set(c.key for c in t.selected_columns.values() if c.key is not None)
                for t in data.values()
            ),
        )

        return cast(
            Frame[ExT2, SxT2],
            Frame(
                sqla.select(
                    *(
                        (
                            sqla.func.coalesce(
                                *(t.c[col] for t in data.values() if col in t.c)
                            )
                            if coalesce == "left"
                            else sqla.func.coalesce(
                                *reversed(
                                    [t.c[col] for t in data.values() if col in t.c]
                                )
                            )
                        ).label(col)
                        for col in all_cols
                    )
                ),
            ),
        )

    raise ValueError("Incompatible data type for coalescent union: " f"{type(data)}")


def frame_isin(
    frame: Frame[ExT2, Col],
    values: Collection | slice,
) -> Frame[ExT2, Col]:
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

    return cast(Frame[ExT2, Col], Frame(data))


class Node:
    """Base class for graphable objects."""


@dataclass(kw_only=True)
class Data(TypeAware[ValT], Generic[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt], ABC):
    """Property definition for a model."""

    # Core attributes:

    context: CtxT | Data[Any, Any, Any, Any, Any, CtxT]

    # Extension methods:

    @abstractmethod
    def _id(self) -> str:
        """Identity of the data object."""
        raise NotImplementedError()

    @abstractmethod
    def _index(
        self,
    ) -> IdxT:
        """Get the index of this data."""
        raise NotImplementedError()

    @abstractmethod
    def _frame(
        self: Data[Any, Any, SxT2, Any, Any, Root, *tuple[Any, ...]],
    ) -> Frame[PL, SxT2]:
        """Get SQL expression or Polars data."""
        raise NotImplementedError()

    def _mutation(
        self: Data[Any, Any, Any, Any, RwxT2],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[RwxT2]] = {C, U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this data SQL-side."""
        raise NotImplementedError()

    def graph(self: Data[Node]) -> nx.Graph:
        """Get the graph of this data."""
        raise NotImplementedError()

    # Type:

    @cached_prop
    def value_typeform(self) -> SingleTypeDef[ValT] | UnionType:
        """Target typeform of this prop."""
        return self.typeargs[ValT]

    @cached_prop
    def value_type_set(self) -> set[type[ValT]]:
        """Target types of this prop (>1 in case of union typeform)."""
        return typedef_to_typeset(self.value_typeform)

    @cached_prop
    def common_value_type(self) -> type:
        """Common base type of the target types."""
        return get_common_type(self.value_typeform)

    # Context:

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if not isinstance(self.context, Data):
            return self._id()

        return self.context.fqn + "." + self._id()

    @overload
    def parent(
        self: Data[
            ValT2,
            IdxT2,
            SxT2,
            ExT2,
            Any,
            CtxT2,
            *CtxTt2,
            Ctx[FromChild, FromChild, FromChild],
        ],
    ) -> Data[ValT2, IdxT2, SxT2, ExT2, R, CtxT2, *CtxTt2]: ...

    @overload
    def parent(
        self: Data[
            ValT2,
            IdxT2,
            Any,
            ExT2,
            Any,
            CtxT2,
            *CtxTt2,
            Ctx[FromChild, FromChild, SxT3],
        ],
    ) -> Data[ValT2, IdxT2, SxT3, ExT2, R, CtxT2, *CtxTt2]: ...

    @overload
    def parent(
        self: Data[
            ValT2,
            Any,
            SxT2,
            ExT2,
            Any,
            CtxT2,
            *CtxTt2,
            Ctx[FromChild, Idx[*KeyTt3], FromChild],
        ],
    ) -> Data[ValT2, Idx[*KeyTt3], SxT2, ExT2, R, CtxT2, *CtxTt2]: ...

    @overload
    def parent(
        self: Data[
            Any,
            IdxT2,
            SxT2,
            ExT2,
            Any,
            CtxT2,
            *CtxTt2,
            Ctx[ValT3, FromChild, FromChild],
        ],
    ) -> Data[ValT3, IdxT2, SxT2, ExT2, R, CtxT2, *CtxTt2]: ...

    @overload
    def parent(
        self: Data[
            Any,
            IdxT2,
            Any,
            ExT2,
            Any,
            CtxT2,
            *CtxTt2,
            Ctx[ValT3, FromChild, SxT3],
        ],
    ) -> Data[ValT3, IdxT2, SxT3, ExT2, R, CtxT2, *CtxTt2]: ...

    @overload
    def parent(
        self: Data[
            Any,
            Any,
            SxT2,
            ExT2,
            Any,
            CtxT2,
            *CtxTt2,
            Ctx[ValT3, Idx[*KeyTt3], FromChild],
        ],
    ) -> Data[ValT3, Idx[*KeyTt3], SxT2, ExT2, R, CtxT2, *CtxTt2]: ...

    @overload
    def parent(
        self: Data[
            Any,
            Any,
            Any,
            ExT2,
            Any,
            CtxT2,
            *CtxTt2,
            Ctx[ValT3, Idx[*KeyTt3], SxT3],
        ],
    ) -> Data[ValT3, Idx[*KeyTt3], SxT3, ExT2, R, CtxT2, *CtxTt2]: ...

    @overload
    def parent(
        self: Data[
            Any, Any, Any, ExT2, Any, Ctx[ValT3, Idx[*KeyTt3], SxT3], *tuple[()]
        ],
    ) -> None: ...

    @overload
    def parent(
        self: Data,
    ) -> Data | None: ...

    def parent(
        self: Data,
    ) -> Data | None:
        """Get the context of this property."""
        if isinstance(self.context, Data):
            return cast(Data, self.context)

        return None

    def root(self) -> CtxT:
        """Get the root of this property."""
        if isinstance(self.context, Data):
            return self.context.root()

        return self.context

    # Relational Identity:

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.typeref, self.context, self._id(), self._index()))

    # Index:

    def _idx_components(
        self: Data,
    ) -> tuple[Data[Any, IdxT, Col, ExT, R, CtxT, *tuple[Any, ...]], ...]:
        """Get the index components of this dataset."""
        index = self._index()

        full_idx: tuple[Data[Any, Any, Col, Any, R, CtxT, *tuple[Any, ...]], ...]
        match index:
            case Idx():
                full_idx = tuple(self[c] for c in index.components)
            case SelfIdx():
                assert self.typeargs[DxT] is Col
                full_idx = (
                    cast(Data[Any, Any, Col, Any, R, CtxT, *tuple[Any, ...]], self),
                )
            case HashIdx():
                hashed = cast(
                    Data[Any, Any, Col, Any, R, CtxT, *tuple[Any, ...]], self
                )[unstable_hash]
                full_idx = (hashed,)
            case AutoIdx():
                components = cast(
                    AutoIdx[AutoIndexable], index
                ).value_type._index_components()
                full_idx = tuple(self[c] for c in components)
            case ModIdx():
                parent = self.parent()
                assert parent is not None
                parent_idx = parent._idx_components()

                reduce_args = get_args(index.reduction)
                reduced_idx = parent_idx[: -len(reduce_args)]

                full_idx = reduced_idx + cast(Idx, index.expansion).components
            case _:
                raise ValueError(f"Unsupported index type: {type(index)}")

        return full_idx

    @overload
    def index(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Idx[()], Any, ExT2],
    ) -> None: ...

    @overload
    def index(self: Data[Any, AnyIdx[*KeyTt2], Any, ExT2]) -> Data[
        tuple[*KeyTt2],
        SelfIdx[*KeyTt2],
        Tab,
        ExT2,
        R,
        CtxT,
        *CtxTt,
    ]: ...

    def index(self: Data[Any, AnyIdx[*KeyTt2], Any, ExT2]) -> (
        Data[
            tuple[*KeyTt2],
            SelfIdx[*KeyTt2],
            Tab,
            ExT2,
            R,
            CtxT,
            *CtxTt,
        ]
        | None
    ):
        """Get the index of this data."""
        idx_comp = self._idx_components()

        if len(idx_comp) == 0:
            return None

        alignment = reduce(Data.__matmul__, idx_comp)
        return cast(
            Data[
                tuple[*KeyTt2],
                SelfIdx[*KeyTt2],
                Tab,
                ExT2,
                R,
                CtxT,
                *CtxTt,
            ],
            alignment,
        )

    def _map_index_filters(
        self, sel: list | slice | tuple[list | slice, ...]
    ) -> Mapping[Data[Any, IdxT, Col, ExT, R, CtxT, *tuple[Any, ...]], list | slice]:
        idx = self._idx_components()

        match sel:
            case list() | slice():
                return {idx[0]: sel}
            case tuple():
                return {i: s for i, s in zip(idx, sel)}

    # SQL:

    @overload
    def select(
        self: Data[Any, Any, Any, SQL, Any, Base],
    ) -> sqla.Select: ...

    @overload
    def select(self: Data[Any, Any, Any, PL]) -> sqla.Select | None: ...

    def select(self: Data[Any, Any, Shape]) -> sqla.Select | None:
        """Return select statement for this dataset."""
        frame = self._frame().get()

        match frame:
            case sqla.Select():
                return frame
            case sqla.ColumnElement():
                return sqla.select(frame)
            case dict() if has_type(frame, dict[str, sqla.Select]):
                return sqla.select(
                    *(
                        c.label(f"{prefix}.{c.key}")
                        for prefix, t in frame.items()
                        for c in t.selected_columns
                    )
                )
            case _:
                return None

    @overload
    def select_str(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, SQL, Any, Base],
    ) -> str: ...

    @overload
    def select_str(
        self: Data[Any, Any, Any, PL],
    ) -> None: ...

    def select_str(self) -> str | None:
        """Return select statement for this dataset."""
        select = self.select()
        if select is None:
            return None

        root = self.root()
        assert isinstance(root, Base)

        return sqlparse.format(
            str(select.compile(root.connection)),
            reindent=True,
            keyword_case="upper",
        )

    @overload
    def query(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, SQL],
    ) -> sqla.Subquery: ...

    @overload
    def query(
        self: Data[Any, Any, Any, PL],
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

    @overload
    def load(
        self: Data[Any, Any, Col, Any, Any, Root],
    ) -> pl.Series: ...

    @overload
    def load(
        self: Data[Any, Any, Tab, Any, Any, Root],
    ) -> pl.DataFrame: ...

    @overload
    def load(
        self: Data[Any, Any, Tabs, Any, Any, Root],
    ) -> dict[str, pl.DataFrame]: ...

    @overload
    def load(
        self: Data[Any, Any, Shape, Any, Any, Root],
    ) -> pl.Series | pl.DataFrame | dict[str, pl.DataFrame]: ...

    def load(
        self: Data[Any, Any, Any, PL, Any, Root],
    ) -> pl.Series | pl.DataFrame | dict[str, pl.DataFrame]:
        """Load dataset as dataframe."""
        frame = self._frame().get()

        if isinstance(frame, pl.Series | pl.DataFrame) or has_type(
            frame, dict[str, pl.DataFrame]
        ):
            return frame

        select = self.select()
        assert select is not None

        base = self.root()
        assert isinstance(base, Base)

        res = pl.read_database(
            select,
            base.connection,
        )

        if isinstance(frame, dict):
            return {
                k: res.select(
                    *(
                        pl.col(c).alias(c.split(".")[-1])
                        for c in res.columns
                        if c.startswith(f"{k}.")
                    )
                )
                for k in frame.keys()
            }

        return res

    # Collection interface:

    def values(
        self: Data[Any, Any, Shape, PL, Any, Root],
    ) -> Sequence[ValT]:
        """Iterable over this dataset's values."""
        data = self.load()

        match data:
            case pl.Series():
                return data.to_list()
            case pl.DataFrame():
                constructor = self.common_value_type
                return [constructor(d) for d in data.to_dicts()]
            case dict():
                return self.common_value_type(*data.values())

    @overload
    def keys(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, FullIdx[KeyT2], Any, Any, Any, Root],
    ) -> Sequence[KeyT2]: ...

    @overload
    def keys(
        self: Data[Any, FullIdx[*KeyTt2], Any, Any, Any, Root],
    ) -> Sequence[tuple[*KeyTt2]]: ...

    def keys(
        self: Data[Any, Any, Any, Any, Any, Root],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        idx = self.index()
        return idx.values() if idx is not None else [tuple()] * len(self)

    @overload
    def items(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, FullIdx[KeyT2], Any, Any, Any, Root],
    ) -> Iterable[tuple[KeyT2, ValT]]: ...

    @overload
    def items(
        self: Data[Any, FullIdx[*KeyTt2], Any, Any, Any, Root],
    ) -> Iterable[tuple[tuple[*KeyTt2], ValT]]: ...

    def items(
        self: Data[Any, Any, Any, Any, Any, Root],
    ) -> Iterable[tuple[Any, ValT]]:
        """Iterable over index keys."""
        return zip(self.keys(), self.values())

    @overload
    def get(
        self: Data[Any, Idx[()], Any, Any, Any, Root],
        key: None = ...,
        default: ValTo = ...,
    ) -> ValT | ValTo: ...

    @overload
    def get(
        self: Data[ValT2, FullIdx[KeyT2], Any, Any, Any, Root],
        key: KeyT2 | tuple[KeyT2],
        default: ValTo,
    ) -> ValT | ValTo: ...

    @overload
    def get(
        self: Data[ValT2, FullIdx[*KeyTt2], Any, Any, Any, Root],
        key: tuple[*KeyTt2],
        default: ValTo,
    ) -> ValT | ValTo: ...

    def get(
        self: Data[Any, Any, Any, Any, Any, Root],
        key: Hashable = None,
        default: ValTo = None,
    ) -> ValT | ValTo:
        """Get a record by key."""
        try:
            return (self[key] if key is not None else self).values()[0]
        except KeyError | IndexError:
            return default

    def __iter__(  # noqa: D105
        self: Data[Any, Any, Any, Any, Any, Root],
    ) -> Iterator[ValT]:
        return iter(self.values())

    def __len__(self: Data[Any, Any, Any, Any, Any, Root]) -> int:
        """Get the number of items in the dataset."""
        frame = self._frame().get()
        if isinstance(frame, pl.Series | pl.DataFrame):
            return len(frame)

        query = self.query()
        assert query is not None

        base = self.root()
        assert isinstance(base, Base)

        count = base.connection.execute(
            sqla.select(sqla.func.count()).select_from(query)
        ).scalar()
        assert count is not None

        return count

    # Application:

    # 1. Context application, altered index, kept value + DxT
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], DxT2, ExT2, RwxT2],
        key: Data[
            Keep,
            ModIdx[Idx[*KeyTt4], Idx[*KeyTt3]],
            Keep,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt2, *KeyTt3],
        DxT2,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], SxT2],
        *CtxTt3,
    ]: ...

    # 2. Context application, altered index, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], DxT2, ExT2, RwxT2],
        key: Data[
            Keep,
            ModIdx[Idx[*KeyTt4], Idx[*KeyTt3]],
            DxT3,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt2, *KeyTt3],
        DxT3,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], SxT2],
        *CtxTt3,
    ]: ...

    # 3. Context application, altered index, new value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2, *KeyTt4], DxT2, ExT2, RwxT2],
        key: Data[
            ValT3,
            ModIdx[Idx[*KeyTt4], Idx[*KeyTt3]],
            DxT3,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt2, *KeyTt3],
        DxT3,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2, *KeyTt4], SxT2],
        *CtxTt3,
    ]: ...

    # 4. Context application, no parent index, kept value + DxT
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[()], DxT2, ExT2, RwxT2],
        key: Data[
            Keep,
            ModIdx[Idx[*tuple[Any, ...]], Idx[*KeyTt3]],
            Keep,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[()], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt3],
        DxT2,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[()], SxT2],
        *CtxTt3,
    ]: ...

    # 5. Context application, no parent index, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[()], DxT2, ExT2, RwxT2],
        key: Data[
            Keep,
            ModIdx[Idx[*tuple[Any, ...]], Idx[*KeyTt3]],
            DxT3,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[()], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt3],
        DxT3,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[()], SxT2],
        *CtxTt3,
    ]: ...

    # 6. Context application, no parent index, new value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[()], DxT2, ExT2, RwxT2],
        key: Data[
            ValT3,
            ModIdx[Idx[*tuple[Any, ...]], Idx[*KeyTt3]],
            DxT3,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[()], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt3],
        DxT3,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[()], SxT2],
        *CtxTt3,
    ]: ...

    # 7. Context application, new index, kept value + DxT
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2], DxT2, ExT2, RwxT2],
        key: Data[
            Keep,
            FullIdx[*KeyTt3],
            Keep,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[*KeyTt2], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt3],
        DxT2,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2], SxT2],
        *CtxTt3,
    ]: ...

    # 8. Context application, new index, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2], DxT2, ExT2, RwxT2],
        key: Data[
            Keep,
            FullIdx[*KeyTt3],
            DxT3,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[*KeyTt2], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt3],
        DxT3,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2], SxT2],
        *CtxTt3,
    ]: ...

    # 9. Context application, new index, new value
    @overload
    def __getitem__(
        self: Data[ValT2, AnyIdx[*KeyTt2], DxT2, ExT2, RwxT2],
        key: Data[
            ValT3,
            FullIdx[*KeyTt3],
            DxT3,
            ExT2,
            RwxT2,
            Ctx[ValT2, Idx[*KeyTt2], SxT2],
            *CtxTt3,
        ],
    ) -> Data[
        ValT3,
        Idx[*KeyTt3],
        DxT3,
        ExT2,
        RwxT2,
        CtxT,
        *CtxTt,
        Ctx[ValT2, Idx[*KeyTt2], SxT2],
        *CtxTt3,
    ]: ...

    # 10. Key list / slice filtering, scalar index type
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], Any, Any, RU],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, DxT, ExT, RU, CtxT, *CtxTt]: ...

    # 11. Key list / slice filtering
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Any, RU],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, DxT, ExT, RU, CtxT, *CtxTt]: ...

    # 12. Key list / slice filtering, scalar index type, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT2], Any, Any, R],
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, DxT, ExT, R, CtxT, *CtxTt]: ...

    # 13. Key list / slice filtering, ro
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt2], Any, Any, R],
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, DxT, ExT, R, CtxT, *CtxTt]: ...

    # 14. Key selection, fully rooted
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt3], Any, Any, Any, Root],
        key: tuple[*KeyTt3],
    ) -> ValT: ...

    # 15. Key selection, fully rooted, scalar
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT3], Any, Any, Any, Root],
        key: KeyT3,
    ) -> ValT: ...

    # 16. Key selection
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[*KeyTt3, *KeyTt2], SxT2],
        key: tuple[*KeyTt3],
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        DxT,
        ExT,
        RwxT,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[*KeyTt3, *KeyTt2], SxT2],
    ]: ...

    # 17. Key selection, scalar
    @overload
    def __getitem__(
        self: Data[Any, AnyIdx[KeyT3, *KeyTt2], SxT2],
        key: KeyT3,
    ) -> Data[
        ValT,
        Idx[*KeyTt2],
        DxT,
        ExT,
        RwxT,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[KeyT3, *KeyTt2], SxT2],
    ]: ...

    # 18. Base type selection
    @overload
    def __getitem__(
        self: Base[ValT2],
        key: type[ValT2],
    ) -> Data[ValT2, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    def __getitem__(
        self: Data | Base,
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
                    return alignment[
                        Transform(
                            func=operator.or_,
                            frame_func=partial(frame_coalesce, coalesce="left"),
                            contract="value",
                        )
                    ]

                return self.registry(key)
            case list() | slice() | Hashable():
                assert isinstance(self, Data)

                rooted = False
                if not isinstance(key, list | slice) and not has_type(
                    key, tuple[slice, ...]
                ):
                    key = key if isinstance(key, tuple) else (key,)
                    if len(key) == len(self._idx_components()) and is_subtype(
                        self.typeargs[CtxT], Root
                    ):
                        rooted = True

                    data = self[KeySelect(cast(tuple, key))]

                    if rooted:
                        vals = cast(Data[Any, Any, Any, Any, Any, Root], data).values()
                        return vals[0]

                    return data

                keymap = self._map_index_filters(key)
                return self[Filter.from_keymap(keymap)]

    # Alignment:

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Col, ExT2, RwxT2, CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, Tab, ExT2, RwxT2, CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        Tab,
        ExT2,
        RwxT2,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, DxT2, ExT2, RwxT2, CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, DxT2, ExT2, RwxT2, CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        Tabs,
        ExT2,
        RwxT2,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[tuple[*ValTt2], Any, Tab, ExT2, RwxT2, CtxT2],
        other: Data[ValT3, IdxT3, Col, ExT2, RwxT2, CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        Tab,
        ExT2,
        RwxT2,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, DxT2, ExT2, RwxT2, CtxT2],
        other: Data[ValT3, IdxT3, DxT2, ExT2, RwxT2, CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        Tabs,
        ExT2,
        RwxT2,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, Col, ExT2, RwxT2, CtxT2],
        other: Data[ValT3, IdxT3, Col, ExT2, RwxT2, CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        Tab,
        ExT2,
        RwxT2,
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, DxT2, ExT2, RwxT2, CtxT2],
        other: Data[ValT3, IdxT3, DxT2, ExT2, RwxT2, CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        Tabs,
        ExT2,
        RwxT2,
        CtxT2,
    ]: ...

    def __matmul__(
        self: Data[Any, Any, Any, ExT2, RwxT2, CtxT2],
        other: Data[Any, IdxT3, Any, ExT2, RwxT2, CtxT2],
    ) -> Data[
        tuple,
        IdxT | IdxT3,
        Tab | Tabs,
        ExT2,
        RwxT2,
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
            Tab | Tabs,
            ExT2,
            RwxT2,
            CtxT2,
        ](
            data=self_data + other_data,
            context=self.root(),
        )

    # Reduction:

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], AnyIdx[*KeyTt2], Shape[SxT3], ExT2],
        op: Callable[[ValT2, ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        SxT3,
        ExT2,
        R,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[*KeyTt2], SxT3],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], AnyIdx[*KeyTt2], SxT3, ExT2],
        op: Callable[[ValT2, ValT4], ValT3],
        right: ValT4,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        SxT3,
        ExT2,
        R,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[*KeyTt2], SxT3],
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[ValT2, AnyIdx[*KeyTt2], SxT3, ExT2],
        op: Callable[[ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        Idx[*KeyTt2],
        SxT3,
        ExT2,
        R,
        CtxT,
        *CtxTt,
        Ctx[ValT, Idx[*KeyTt2], SxT3],
    ]: ...

    def _map_reduce_operator(
        self: Data,
        op: Callable[[Any, Any], Any] | Callable[[Any], Any],
        right: Any | Literal[Not.defined] = Not.defined,
    ) -> Data[
        Any,
        Any,
        Any,
        Any,
        R,
        CtxT,
        *CtxTt,
        Ctx[ValT, Any, Any],
    ]:
        """Create a scalar comparator for the given operation."""
        if right is not Not.defined:
            mapping = Transform(
                func=lambda x: cast(Callable[[Any, Any], Any], op)(x, right),
                frame_func=lambda frame: cast(Callable[[Any, Any], Any], op)(
                    frame.get(), right
                ),
            )
            return cast(
                Data,
                self[mapping],
            )

        if len(inspect.getfullargspec(op).args) == 1:
            op = cast(Callable[[Any], Any], op)
            mapping = Transform(
                func=op,
                frame_func=lambda frame: op(frame.get()),
            )
            return cast(
                Data,
                self[mapping],
            )

        op = cast(Callable[[Any, Any], Any], op)
        reduction = Transform(
            func=op,
            frame_func=lambda left, right: op(left.get(), right.get()),
            contract="value",
        )
        assert issubclass(self.common_value_type, tuple)
        return cast(
            Data,
            self[reduction],
        )

    # Comparison:

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[ValT2, Any, Col, ExT2, Any, CtxT2, *CtxTt2],
        other: Data[ValT3, IdxT3, Col, ExT2, Any, CtxT2, *CtxTt2],
    ) -> Data[
        bool,
        Any,
        Col,
        ExT2,
        R,
        CtxT2,
        *CtxTt2,
        Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any],
    ]: ...

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Col, ExT2, Any, CtxT2, *CtxTt2],
        other: Any,
    ) -> Data[
        bool,
        Any,
        Col,
        ExT2,
        R,
        CtxT2,
        *CtxTt2,
    ]: ...

    def __eq__(  # noqa: D105 # pyright: ignore[reportIncompatibleMethodOverride]
        self: Data[Any, Any, Col],
        other: Any,
    ) -> (
        Data[
            bool,
            Any,
            Col,
            Any,
            R,
            Any,
            *tuple[Any, ...],
        ]
        | bool
    ):
        if not isinstance(other, Data):
            return self._map_reduce_operator(operator.eq, other)

        alignment = self @ other
        return alignment._map_reduce_operator(operator.eq)

    def isin(
        self: Data[Any, Any, Col, ExT2, Any, CtxT2, *CtxTt2],
        other: Collection[ValT2] | slice,
    ) -> Data[
        bool,
        Any,
        Col,
        ExT2,
        R,
        CtxT2,
        *CtxTt2,
    ]:
        """Test values of this dataset for membership in the given iterable."""
        if isinstance(other, slice):
            mapping = Transform(
                func=lambda x: other.start <= x <= other.stop,
                frame_func=partial(frame_isin, values=other),
            )
        else:
            mapping = Transform(
                func=lambda x: x in other,
                frame_func=partial(frame_isin, values=other),
            )

        return cast(
            Data[
                bool,
                Any,
                Col,
                ExT2,
                R,
                CtxT2,
                *CtxTt2,
            ],
            self[mapping],
        )

    # Index set operations:

    def __or__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        ModIdx,
        DxT2,
        ExT2,
        R,
        Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any],
        CtxT2,
    ]:
        """Union two databases, right overriding left."""
        alignment = self @ other
        reduction = Transform(
            func=operator.or_,
            frame_func=partial(frame_coalesce, coalesce="left"),
            contract="value",
        )

        return cast(
            Data[
                ValT2 | ValT3,
                ModIdx,
                DxT2,
                ExT2,
                R,
                Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any],
                CtxT2,
            ],
            alignment[reduction],
        )

    def __xor__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        ModIdx,
        DxT2,
        ExT2,
        R,
        Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any],
        CtxT2,
    ]:
        """Union two databases, right overriding left."""
        alignment = self @ other
        return cast(
            Data[
                ValT2 | ValT3,
                ModIdx,
                DxT2,
                ExT2,
                R,
                Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any],
                CtxT2,
            ],
            alignment[
                Transform(
                    func=operator.xor,
                    frame_func=partial(frame_coalesce, coalesce="right"),
                    contract="value",
                )
            ],
        )

    def __lshift__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        ModIdx,
        DxT2,
        ExT2,
        R,
        Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any],
        CtxT2,
    ]:
        """Union two databases, right overriding left."""
        alignment = Align(
            context=self.root(),
            data=(self, other),
            join="left",
        )
        return cast(
            Data[
                ValT2 | ValT3,
                ModIdx,
                DxT2,
                ExT2,
                R,
                Ctx[tuple[ValT2, ValT3], Idx[*tuple[Any, ...]], Any],
                CtxT2,
            ],
            alignment[
                Transform(
                    func=operator.and_,
                    frame_func=partial(frame_coalesce, coalesce="right"),
                    contract="value",
                )
            ],
        )

    @overload
    def __ior__(
        self: Data[Any, Any, Col, SQL, C | U, Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Col, Any, C | U, Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tab, SQL, C | U, Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tab, Any, C | U, Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tabs, SQL, C | U, Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tabs, Any, C | U, Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    def __ior__(
        self: Data[Any, Any, Any, Any, C | U, Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        mutations = self._mutation(input_data, mode={C, U})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt], self)

    @overload
    def __ixor__(
        self: Data[Any, Any, Col, SQL, C, Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Col, Any, C, Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tab, SQL, C, Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tab, Any, C, Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tabs, SQL, C, Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tabs, Any, C, Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    def __ixor__(
        self: Data[Any, Any, Any, Any, C, Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        mutations = self._mutation(input_data, mode={C})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt], self)

    @overload
    def __ilshift__(
        self: Data[Any, Any, Col, SQL, U, Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Col, Any, U, Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tab, SQL, U, Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tab, Any, U, Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tabs, SQL, U, Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tabs, Any, U, Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    def __ilshift__(
        self: Data[Any, Any, Any, Any, U, Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        mutations = self._mutation(input_data, mode={U})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt], self)

    @overload
    def __setitem__(
        self: Data[Any, Any, Col, SQL, C | U | D, Base],
        key: slice,
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, Col, Any, C | U | D, Base],
        key: slice,
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, Tab, SQL, C | U | D, Base],
        key: slice,
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, Tab, Any, C | U | D, Base],
        key: slice,
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, Tabs, SQL, C | U | D, Base],
        key: slice,
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    @overload
    def __setitem__(
        self: Data[Any, Any, Tabs, Any, C | U | D, Base],
        key: slice,
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]: ...

    def __setitem__(
        self: Data[Any, Any, DxT2, Any, C | U | D, Base],
        key: slice,
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]:
        """Union two databases, right overriding left."""
        assert key == slice(None, None, None)

        if input_data is not self:
            mutations = self._mutation(input_data, mode={C, U, D})

            for mutation in mutations:
                self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt], self)


RegT = TypeVar("RegT", covariant=True, bound=AutoIndexable)


@dataclass(kw_only=True)
class Registry(Data[RegT, AutoIdx[RegT], Tab, SQL, RwxT, RootT, None], ABC):
    """Represent a base data type collection."""

    _instance_map: dict[Hashable, RegT] = field(default_factory=dict)

    def _id(self) -> str:
        ctx_module = getmodule(self.common_value_type)
        return (
            (
                ctx_module.__name__ + "." + self.common_value_type.__name__
                if ctx_module is not None
                else self.common_value_type.__name__
            )
            + "."
            + self._id()
        )


TupT = TypeVar("TupT", bound=tuple, covariant=True)


@dataclass(kw_only=True)
class Align(Data[TupT, IdxT, DxT, ExT, RwxT, CtxT, *CtxTt]):
    """Alignment of multiple props."""

    data: tuple[Data[Any, IdxT, Any, Any, RwxT, CtxT, *CtxTt], ...]
    join: Literal["left", "right", "outer", "inner"] = "outer"
    match_idx_on: Literal["value", "value+id", "value+id+path"] = "value"

    @cached_prop
    def value_types(self) -> tuple[SingleTypeDef[ValT] | UnionType, ...]:
        """Get the value types."""
        return tuple(d.value_typeform for d in self.data)

    def _id(self) -> str:
        """Name of the property."""
        # TODO: Implement this method for the Align class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> IdxT:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()


class Transform(
    Data[
        ValT,
        ModIdx[SubIdxT, AddIdxT],
        DxT,
        ExT,
        R,
        Interface[ArgT, Any, ArgDxT],
    ],
    Generic[ArgT, ArgDxT, ValT, SubIdxT, AddIdxT, DxT, ExT],
):
    """Apply a mapping function to a dataset."""

    @overload
    def __init__(
        self: Transform[ValT2, SxT2, ValT3, Idx[()], Idx[()], SxT3, ExT2],
        func: Callable[[ValT2], ValT3],
        frame_func: Callable[[Frame[ExT2, SxT2]], Frame[ExT2, SxT3]],
    ): ...

    @overload
    def __init__(
        self: Transform[ValT2, SxT2, ValT3, Idx[*KeyTt2], Idx[()], SxT3, ExT2],
        func: Callable[[Iterable[ValT2]], ValT3],
        frame_func: Callable[[Frame[ExT2, SxT2]], Frame[ExT2, SxT3]],
        contract: SingleTypeDef[Idx[*KeyTt2]],
    ): ...

    @overload
    def __init__(
        self: Transform[
            Iterable[ValT2], Shape[SxT3], ValT3, Idx[()], Idx[()], SxT3, ExT2
        ],
        func: Callable[[Iterable[ValT2]], ValT3] | Callable[[ValT2, ValT2], ValT3],
        frame_func: Callable[[Frame[ExT2, Shape[SxT3]]], Frame[ExT2, SxT3]],
        contract: Literal["value"],
    ): ...

    @overload
    def __init__(
        self: Transform[Iterable[ValT2], SxT2, ValT3, Idx[()], Idx[()], SxT3, ExT2],
        func: Callable[[Iterable[ValT2]], ValT3] | Callable[[ValT2, ValT2], ValT3],
        frame_func: Callable[[Frame[ExT2, SxT2], Frame[ExT2, SxT2]], Frame[ExT2, SxT3]],
        contract: Literal["value"],
    ): ...

    def __init__(self, *args, **kwargs):  # noqa: D107
        self.context = Interface()
        # TODO: Implement the constructor for the Transform class.

    def _id(self) -> str:
        """Name of the property."""
        # TODO: Implement this method for the Map class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> ModIdx[SubIdxT, AddIdxT]:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()


def unstable_frame_hash(
    frame: Frame[ExT, Col],
) -> Frame[ExT, Col]:
    """Get a hash of the frame."""
    data = frame.get()
    if isinstance(data, pl.Series):
        return cast(Frame[ExT, Col], Frame(data.hash()))

    return cast(Frame[ExT, Col], Frame(sqla.func.MD5(data)))


unstable_hash = Transform(
    func=hash,
    frame_func=unstable_frame_hash,
)


class Filter(
    Data[
        Keep,
        KeepIdx,
        Keep,
        ExT,
        R | U,
        Interface,
    ]
):
    """Filter a dataset."""

    def __init__(self, bool_data: Data[bool, Any, Col, ExT, Any]):  # noqa: D107
        self.context = Interface()
        self.bool_data = bool_data

    @staticmethod
    def from_keymap(
        keymap: Mapping[Data[Any, Any, Col, ExT2, Any], slice | Collection],
    ) -> Filter[ExT2]:
        """Construct filter from index key map."""
        bool_data = reduce(
            operator.and_, (idx.isin(filt) for idx, filt in keymap.items())
        )

        return Filter(bool_data=bool_data)

    def _id(self) -> str:
        """Name of the property."""
        # TODO: Implement this method for the Filter class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> KeepIdx:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()

    def _mutation(
        self: Data[Any, Any, Any, Any, RwxT2],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[RwxT2]] = {U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        raise NotImplementedError()


class KeySelect(
    Data[
        Keep,
        Reduce[SubIdxT],
        Keep,
        SQL,
        R | U,
        Interface,
    ]
):
    """Select a specific key value (prefix)."""

    def __init__(self: KeySelect[Idx[*KeyTt2]], key: tuple[*KeyTt2]):  # noqa: D107
        self.context = Interface()
        self.key = key

    def _id(self) -> str:
        """Name of the property."""
        # TODO: Implement this method for the Filter class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> Reduce[SubIdxT]:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()

    def _mutation(
        self: Data[Any, Any, Any, Any, RwxT2],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[RwxT2]] = {U},
    ) -> Sequence[sqla.Executable]:
        """Get mutation statements to set this property SQL-side."""
        raise NotImplementedError()
