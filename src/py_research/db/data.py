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
    overload,
    override,
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
    TypeRef,
    has_type,
    is_subtype,
    typedef_to_typeset,
)
from py_research.storage.storables import Realm
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
LoadT = TypeVar(
    "LoadT", bound=pl.Series | pl.DataFrame | dict[str, pl.DataFrame], default=Any
)
LoadT2 = TypeVar("LoadT2", bound=pl.Series | pl.DataFrame | dict[str, pl.DataFrame])


class Shape(Generic[SubShapeT, LoadT]):
    """Base for frame data types."""


@final
class Col(Shape[None, pl.Series]):
    """Singleton to mark standard columnar data."""


@final
class Tab(Shape[Col, pl.DataFrame]):
    """Singleton to mark standard tabular data (consisting only of columns)."""


@final
class Tabs(Shape[Tab, dict[str, pl.DataFrame]]):
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

DefT = TypeVar("DefT", contravariant=True, default=None)
DefT2 = TypeVar("DefT2")


class Def(Generic[DefT]):
    """Declarative definition type."""


type PL = Def[None]

type SQL = Def[sqla.Select | sqla.ColumnElement | dict[str, sqla.Select] | None]
"""SQL engine."""


class API(Def[DefT]):
    """Custom API engine."""


ExT = TypeVar("ExT", bound=PL, default=Any, covariant=True)
ExT2 = TypeVar("ExT2", bound=PL)


@final
@dataclass
class ExtIdx(Generic[*KeyTt]):
    """Define the custom index type of a dataset."""

    components: tuple[Data[Any, Any, Col, SQL, Acc[R, R], Interface], ...]


@final
class SelfIdx(Generic[*KeyTt]):
    """Index by self."""


HashKeyTt = TypeVarTuple("HashKeyTt", default=Unpack[tuple[int]])


@final
class HashIdx(Generic[*HashKeyTt]):
    """Index by hash of self."""


class Indexable(Protocol[*KeyTt]):
    """Base class for auto-indexable objects."""

    @classmethod
    def _index_components(
        cls,
    ) -> tuple[Data[Any, Any, Col, SQL, Acc[R, R], Interface], ...]:
        """Get components for this auto-indexed type."""
        ...


IdxblT = TypeVar(
    "IdxblT",
    bound=Indexable,
    covariant=True,
    default=Indexable[*tuple[Any, *tuple[Any, ...]]],
)
IdxblT2 = TypeVar(
    "IdxblT2",
    bound=Indexable,
)


@final
class MainIdx(Generic[IdxblT]):
    """Index by custom pk derived from provided value type."""

    value_type: type[IdxblT]


@final
@dataclass
class RichIdx(Generic[*KeyTt]):
    """Index by custom pk derived from value type itself."""

    components: tuple[Data[Any, Any, Col, SQL, Acc[R, R], Interface], ...]


type Idx[*K] = ExtIdx[*K] | SelfIdx[*K] | HashIdx[*K] | MainIdx[Indexable[*K]]

IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound=Idx,
    default=Any,
)
IdxT2 = TypeVar(
    "IdxT2",
    bound=Idx,
)
IdxT3 = TypeVar(
    "IdxT3",
    bound=Idx,
)

RdxT = TypeVar(
    "RdxT",
    covariant=True,
    bound=RichIdx | None,
    default=Any,
)


class RW:
    """Base class of all read-write permissions, which allows any operation."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class R(RW):
    """Singleton to allow reading of records."""


RwT = TypeVar("RwT", bound=RW, default=Any, covariant=True)
RwT2 = TypeVar("RwT2", bound=RW)


@final
class C(RW):
    """Singleton to allow creation of new records."""


@final
class U(RW):
    """Singleton to allow updating of records."""


@final
class D(RW):
    """Singleton to allow deletion of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


type RU = R | U

RuT = TypeVar("RuT", bound=RU, default=Any, contravariant=True)
RuT2 = TypeVar("RuT2", bound=RU)

type CRUD = C | R | U | D

CrudT = TypeVar("CrudT", bound=RW, default=Any, contravariant=True)
CrudT2 = TypeVar("CrudT2", bound=RW)
CrudT3 = TypeVar("CrudT3", bound=RW)


class Acc(Generic[CrudT, RwT]):
    """Access control."""


AccT = TypeVar("AccT", bound=Acc, default=Any, covariant=True)
AccT2 = TypeVar("AccT2", bound=Acc)


ArgT = TypeVar("ArgT", contravariant=True, default=Any)
ArgIdxT = TypeVar("ArgIdxT", bound=Idx, contravariant=True, default=Any)
ArgDxT = TypeVar("ArgDxT", bound=Shape, contravariant=True, default=Any)


class Ctx(Generic[ArgT, ArgIdxT, ArgDxT]):
    """Data context."""


CtxT = TypeVar("CtxT", bound=Ctx, default=Any, covariant=True)
CtxT2 = TypeVar("CtxT2", bound=Ctx)
CtxT3 = TypeVar("CtxT3", bound=Ctx)

ClsT = TypeVar("ClsT", bound=Indexable, contravariant=True, default=Any)

ColT = TypeVar("ColT", default=Any)
ColT2 = TypeVar("ColT2")

TabT = TypeVar("TabT", default=Any)
TabT2 = TypeVar("TabT2")

TabsT = TypeVar("TabsT", default=Any)
TabsT2 = TypeVar("TabsT2")


class Base(
    Ctx[None, Any, Any], Realm, Generic[ClsT, ExT, CrudT, ColT, TabT, TabsT], ABC
):
    """Base for retrieving/storing data."""

    @abstractmethod
    def get_registry(self, cls: type[ClsT]) -> TabT:
        """Get internal reference to registry for given class."""
        ...

    @abstractmethod
    def mutate_registry(
        self,
        cls: type[ClsT],
        input_data: pl.DataFrame | Iterator[pl.DataFrame] | TabT,
        mode: Set[type[CrudT]] = {C, U},
    ) -> None:
        """Mutate registry for given class."""
        ...

    @overload
    def compose(
        self,
        ref: ColT | TabT | TabsT,
        dataset: Data[Any, Any, Col, Any, Any, Interface[ClsT]],
    ) -> ColT | None: ...

    @overload
    def compose(
        self,
        ref: ColT | TabT | TabsT,
        dataset: Data[Any, Any, Tab, Any, Any, Interface[ClsT]],
    ) -> TabT | None: ...

    @overload
    def compose(
        self,
        ref: ColT | TabT | TabsT,
        dataset: Data[Any, Any, Tabs, Any, Any, Interface[ClsT]],
    ) -> TabsT | None: ...

    @abstractmethod
    def compose(
        self,
        ref: ColT | TabT | TabsT,
        dataset: Data[Any, Any, Any, Any, Any, Interface[ClsT]],
    ) -> ColT | TabT | TabsT | None:
        """Compose an existing col/table/tables reference with a nested dataset."""
        ...

    @overload
    def load(
        self,
        ref: ColT,
        count: int | Literal["all"] | None = ...,
        start_page: int = ...,
    ) -> pl.Series | Iterator[pl.Series]: ...

    @overload
    def load(
        self,
        ref: TabT,
        count: int | Literal["all"] | None = ...,
        start_page: int = ...,
    ) -> pl.DataFrame | Iterator[pl.DataFrame]: ...

    @overload
    def load(
        self,
        ref: TabsT,
        count: int | Literal["all"] | None = ...,
        start_page: int = ...,
    ) -> dict[str, pl.DataFrame] | Iterator[dict[str, pl.DataFrame]]: ...

    @abstractmethod
    def load(
        self,
        ref: ColT | TabT | TabsT,
        count: int | Literal["all"] | None = None,
        start_page: int = 0,
    ) -> (
        pl.Series
        | pl.DataFrame
        | dict[str, pl.DataFrame]
        | Iterator[pl.Series | pl.DataFrame | dict[str, pl.DataFrame]]
    ):
        """Load data for given col/table/tables reference."""
        ...


BaseT = TypeVar("BaseT", bound=Base, covariant=True, default=Any)
BaseT2 = TypeVar("BaseT2", bound=Base)


@dataclass
class Interface(Ctx[ArgT, ArgIdxT, ArgDxT]):
    """Data interface."""

    arg_type: SingleTypeDef[ArgT] | None = None
    arg_idx_type: SingleTypeDef[ArgIdxT] | None = None
    arg_dx_type: SingleTypeDef[ArgDxT] | None = None


type InputFrame = (
    pl.DataFrame
    | pd.DataFrame
    | sqla.Select
    | sqla.FromClause
    | pl.Series
    | pd.Series
    | sqla.ColumnElement
)

type InputData[V, S, I] = Data[V] | V | Iterable[V] | Mapping[Any, V] | S | Mapping[
    str, I
]

Params = ParamSpec("Params")


class Node(Protocol):
    """Base class for graphable objects."""


CdxT = TypeVar("CdxT", bound=int | slice | Iterable[int] | None, default=None)


@dataclass
class ColSel(Generic[CdxT]):
    """Select columns from a dataset with tuple values."""

    sel: CdxT = None  # pyright: ignore[reportAssignmentType]

    def __getitem__(self, key: CdxT) -> ColSel[CdxT]:
        return copy_and_override(ColSel, self, sel=key)


@dataclass(kw_only=True)
class Data(Generic[ValT, IdxT, DxT, ExT, AccT, CtxT, RdxT], ABC):
    """Base class for all data objects."""

    default_load_batchsize: ClassVar[int] = 1000

    # Core attributes:

    context: CtxT | Data[Any, Any, Any, Any, Any, CtxT]
    typeref: TypeRef[Data] = field(default_factory=TypeRef["Data"])

    def __post_init__(self) -> None:  # noqa: D105
        if self.typeref.hint is object:
            self.typeref.hint = type(self)

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
    def _define(
        self: (
            Data[Any, Any, Col, Def[DefT2], Any, Base[Any, Any, Any, DefT2]]
            | Data[Any, Any, Tab, Def[DefT2], Any, Base[Any, Any, Any, Any, DefT2]]
            | Data[
                Any, Any, Tabs, Def[DefT2], Any, Base[Any, Any, Any, Any, Any, DefT2]
            ]
        ),
        def_type: type[DefT2],
    ) -> DefT2:
        """Get declarative expression in language supported by base."""
        raise NotImplementedError()

    @abstractmethod
    def _compute(
        self: Data[Any, Any, Shape[Any, LoadT2], Any, Any, Base],
    ) -> LoadT2:
        """Get Polars data."""
        raise NotImplementedError()

    @abstractmethod
    def _mutate(
        self: Data[Any, Any, Any, Any, Acc[CrudT2]],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[CrudT2]] = {C, U},
    ) -> None:
        """Mutate underlying registries of this dataset."""
        raise NotImplementedError()

    # Type:

    @cached_prop
    def value_typeref(self) -> TypeRef[ValT]:
        """Target typeform of this prop."""
        return self.typeref.scalar_typevar_map[ValT]

    # Context:

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if not isinstance(self.context, Data):
            return self._id()

        return self.context.fqn + "." + self._id()

    def parent(
        self: Data,
    ) -> Data[Any, Any, Tab] | None:
        """Get the context of this property."""
        if isinstance(self.context, Data):
            return cast(Data[Any, Any, Tab], self.context)

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
        self: Data, rich: bool = False
    ) -> tuple[Data[Any, IdxT, Col, ExT, Acc[R, R], CtxT], ...]:
        """Get the index components of this dataset."""
        index = self._index()
        parent = self.parent()
        assert parent is not None
        parent_idx = parent._idx_components()

        full_idx: tuple[Data[Any, Any, Col, Any, Acc[R, R], CtxT], ...]

        if rich:
            rich_idx: type[RichIdx | None] = self.typeref.scalar_typevar_map[
                RdxT
            ].common_type
            assert not issubclass(rich_idx, type(None))
            full_idx = cast(
                tuple[Data[Any, IdxT, Col, ExT, Acc[R, R], CtxT], ...],
                rich_idx.components,
            )
        else:
            match index:
                case ExtIdx():
                    full_idx = tuple(self[c] for c in index.components)
                case SelfIdx():
                    assert self.typeref.typevar_map[DxT] is Col
                    full_idx = (cast(Data[Any, Any, Col, Any, Acc[R, R], CtxT], self),)
                case HashIdx():
                    hashed = cast(Data[Any, Any, Col, Any, Acc[R, R], CtxT], self)[
                        unstable_hash
                    ]
                    full_idx = (hashed,)
                case MainIdx():
                    components = cast(
                        MainIdx[Indexable], index
                    ).value_type._index_components()
                    full_idx = tuple(self[c] for c in components)
                case _:
                    raise ValueError(f"Unsupported index type: {type(index)}")

        return parent_idx + full_idx

    @overload
    def index(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, ExtIdx[()], Any, ExT2], rich: Literal[False] = ...
    ) -> None: ...

    @overload
    def index(
        self: Data[Any, Idx[*KeyTt2], Any, ExT2], rich: Literal[False] = ...
    ) -> Data[
        tuple[*KeyTt2],
        SelfIdx[*KeyTt2],
        Tab,
        ExT2,
        Acc[R, R],
        CtxT,
    ]: ...

    @overload
    def index(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, ExT2, Any, Any, RichIdx[()] | None],
        rich: Literal[True],
    ) -> None: ...

    @overload
    def index(
        self: Data[Any, Any, Any, ExT2, Any, Any, RichIdx[*KeyTt2] | None],
        rich: Literal[True],
    ) -> Data[
        tuple[*KeyTt2],
        SelfIdx[*KeyTt2],
        Tab,
        ExT2,
        Acc[R, R],
        CtxT,
    ]: ...

    def index(self: Data[Any, Any, Any, ExT2, Any, Any, Any], rich: bool = False) -> (
        Data[
            tuple,
            SelfIdx,
            Tab,
            ExT2,
            Acc[R, R],
            CtxT,
        ]
        | None
    ):
        """Get the index of this data."""
        idx_comp = self._idx_components(rich)

        if len(idx_comp) == 0:
            return None

        alignment = reduce(Data.__matmul__, idx_comp)
        return cast(
            Data[
                tuple,
                SelfIdx,
                Tab,
                ExT2,
                Acc[R, R],
                CtxT,
            ],
            alignment,
        )

    def _map_index_filters(
        self, sel: list | slice | tuple[list | slice, ...]
    ) -> Mapping[Data[Any, IdxT, Col, ExT, Acc[R, R], CtxT], list | slice]:
        idx = self._idx_components(rich=False)

        match sel:
            case list() | slice():
                return {idx[0]: sel}
            case tuple():
                return {i: s for i, s in zip(idx, sel)}

    def graph(self: Data[Node]) -> nx.Graph:
        """Get the graph of this data."""
        raise NotImplementedError()

    # SQL:

    @overload
    def select(
        self: Data[Any, Any, Any, SQL, Any, Base],
    ) -> sqla.Select: ...

    @overload
    def select(self: Data[Any, Any, Any, Any]) -> sqla.Select | None: ...

    def select(self: Data[Any, Any, Any]) -> sqla.Select | None:
        """Return select statement for this dataset."""
        return self._define(sqla.Select)

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
        self: Data[Any, Any, Col, SQL | API, Any, Base],
        count: Literal["all"],
        start_page: int = ...,
    ) -> pl.Series: ...

    @overload
    def load(
        self: Data[Any, Any, Tab, SQL | API, Any, Base],
        count: Literal["all"],
        start_page: int = ...,
    ) -> pl.DataFrame: ...

    @overload
    def load(
        self: Data[Any, Any, Tabs, SQL | API, Any, Base],
        count: Literal["all"],
        start_page: int = ...,
    ) -> dict[str, pl.DataFrame]: ...

    @overload
    def load(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Col, SQL | API, Any, Base],
        count: int | None = ...,
        start_page: int = ...,
    ) -> Iterator[pl.Series]: ...

    @overload
    def load(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Tab, SQL | API, Any, Base],
        count: int | None = ...,
        start_page: int = ...,
    ) -> Iterator[pl.DataFrame]: ...

    @overload
    def load(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Tabs, SQL | API, Any, Base],
        count: int | None = ...,
        start_page: int = ...,
    ) -> Iterator[dict[str, pl.DataFrame]]: ...

    @overload
    def load(
        self: Data[Any, Any, Col, PL, Any, Base],
        count: int | None = ...,
        start_page: int = ...,
    ) -> pl.Series: ...

    @overload
    def load(
        self: Data[Any, Any, Tab, PL, Any, Base],
        count: int | None = ...,
        start_page: int = ...,
    ) -> pl.DataFrame: ...

    @overload
    def load(
        self: Data[Any, Any, Tabs, PL, Any, Base],
        count: int | None = ...,
        start_page: int = ...,
    ) -> dict[str, pl.DataFrame]: ...

    @overload
    def load(
        self: Data[Any, Any, Shape, Any, Any, Base],
    ) -> (
        pl.Series
        | pl.DataFrame
        | dict[str, pl.DataFrame]
        | Iterator[pl.Series | pl.DataFrame | dict[str, pl.DataFrame]]
    ): ...

    def load(
        self: Data[Any, Any, Any, PL, Any, Base],
        count: int | Literal["all"] | None = None,
        start_page: int = 0,
    ) -> (
        pl.Series
        | pl.DataFrame
        | dict[str, pl.DataFrame]
        | Iterator[pl.Series | pl.DataFrame | dict[str, pl.DataFrame]]
    ):
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

        res = (
            pl.read_database(
                select,
                base.connection,
            )
            if count == "all"
            else pl.read_database(
                select,
                base.connection,
                iter_batches=True,
                batch_size=count if count is not None else self.default_load_batchsize,
            )
        )

        if isinstance(res, pl.DataFrame):
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

        yield from res

    # Collection interface:

    def values(
        self: Data[Any, Any, Shape, PL, Any, Base],
    ) -> Iterator[ValT]:
        """Iterable over this dataset's values."""
        data = self.load()

        if not isinstance(data, Iterator):
            data = [data]

        main_type = self.value_typeref.common_type
        item_types = [t.common_type for t in self.value_typeref.arg_typerefs]

        for d in data:
            match d:
                case pl.Series():
                    yield from d.to_list()
                case pl.DataFrame():
                    yield from (main_type(row) for row in d.to_dicts())
                case dict():
                    yield from (
                        main_type(
                            item_type(kwargs)
                            for item_type, kwargs in zip(item_types, row)
                        )
                        for row in zip(*(df.to_dicts() for df in d.values()))
                    )

    @overload
    def keys(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, Any, Any, Base, RichIdx[KeyT2]], rich: Literal[True]
    ) -> Iterator[KeyT2]: ...

    @overload
    def keys(
        self: Data[Any, Any, Any, Any, Any, Base, RichIdx[*KeyTt2]], rich: Literal[True]
    ) -> Iterator[tuple[*KeyTt2]]: ...

    @overload
    def keys(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Idx[KeyT2], Any, Any, Any, Base], rich: bool = ...
    ) -> Iterator[KeyT2]: ...

    @overload
    def keys(
        self: Data[Any, Idx[*KeyTt2], Any, Any, Any, Base], rich: bool = ...
    ) -> Iterator[tuple[*KeyTt2]]: ...

    def keys(
        self: (
            Data[Any, IdxT, Any, Any, Any, Base]
            | Data[Any, Any, Any, Any, Any, Base, RdxT]
        ),
        rich: bool = False,
    ) -> Iterator[Hashable]:
        """Iterable over index keys."""
        idx = self.index(rich)
        assert idx is not None
        return idx.values() if idx is not None else (tuple() for _ in range(len(self)))

    @overload
    def items(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Idx[KeyT2], Any, Any, Any, Base], rich: Literal[False] = ...
    ) -> Iterator[tuple[KeyT2, ValT]]: ...

    @overload
    def items(
        self: Data[Any, Idx[*KeyTt2], Any, Any, Any, Base], rich: Literal[False] = ...
    ) -> Iterator[tuple[tuple[*KeyTt2], ValT]]: ...

    @overload
    def items(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Any, Any, Any, Base, RichIdx[KeyT2]], rich: Literal[True]
    ) -> Iterator[tuple[KeyT2, ValT]]: ...

    @overload
    def items(
        self: Data[Any, Any, Any, Any, Any, Base, RichIdx[*KeyTt2]], rich: Literal[True]
    ) -> Iterator[tuple[tuple[*KeyTt2], ValT]]: ...

    def items(
        self: (
            Data[Any, IdxT, Any, Any, Any, Base]
            | Data[Any, Any, Any, Any, Any, Base, RdxT]
        ),
        rich: bool = False,
    ) -> Iterator[tuple[Any, ValT]]:
        """Iterator over index keys."""
        return zip(self.keys(rich), self.values())

    @overload
    def get(
        self: (
            Data[Any, ExtIdx[()], Any, Any, Any, Base]
            | Data[Any, Any, Any, Any, Any, Base, RichIdx[()]]
        ),
        key: None = ...,
        default: ValTo = ...,
    ) -> ValT | ValTo: ...

    @overload
    def get(
        self: (
            Data[ValT2, Idx[KeyT2], Any, Any, Any, Base]
            | Data[ValT2, Any, Any, Any, Any, Base, RichIdx[KeyT2]]
        ),
        key: KeyT2 | tuple[KeyT2],
        default: ValTo,
    ) -> ValT | ValTo: ...

    @overload
    def get(
        self: (
            Data[ValT2, Idx[*KeyTt2], Any, Any, Any, Base]
            | Data[ValT2, Any, Any, Any, Any, Base, RichIdx[*KeyTt2]]
        ),
        key: tuple[*KeyTt2],
        default: ValTo,
    ) -> ValT | ValTo: ...

    def get(
        self: Data[Any, Any, Any, Any, Any, Base, Any],
        key: Hashable = None,
        default: ValTo = None,
    ) -> ValT | ValTo:
        """Get a record by key."""
        try:
            return next((self[key] if key is not None else self).values())
        except KeyError | IndexError:
            return default

    def __iter__(
        self: Data[Any, Any, Any, Any, Any, Base],
    ) -> Iterator[ValT]:
        return iter(self.values())

    def __len__(self: Data[Any, Any, Any, Any, Any, Base]) -> int:
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

    # Context Application:

    # 1. Context application, kept value + DxT, rich index
    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[
            ValT2,
            Idx[*KeyTt2],
            DxT2,
            ExT2,
            Acc[Any, CrudT3 | RwT2],
            Any,
            RichIdx[*KeyTt4],
        ],
        key: Data[
            Keep,
            Idx[*KeyTt3],
            Keep,
            ExT2,
            Acc[CrudT3, RwT2],
            Ctx[ValT2, ExtIdx[*KeyTt2], SxT2],
            RichIdx[*KeyTt5],
        ],
    ) -> Data[
        ValT2,
        ExtIdx[*KeyTt2, *KeyTt3],
        DxT2,
        ExT2,
        Acc[CrudT3, RwT2],
        CtxT,
        RichIdx[*KeyTt4, *KeyTt5],
    ]: ...

    # 2. Context application, kept value, rich index
    @overload
    def __getitem__(
        self: Data[
            ValT2,
            Idx[*KeyTt2],
            DxT2,
            ExT2,
            Acc[Any, CrudT3 | RwT2],
            Any,
            RichIdx[*KeyTt4],
        ],
        key: Data[
            Keep,
            Idx[*KeyTt3],
            DxT3,
            ExT2,
            Acc[CrudT3, RwT2],
            Ctx[ValT2, ExtIdx[*KeyTt2], SxT2],
            RichIdx[*KeyTt5],
        ],
    ) -> Data[
        ValT2,
        ExtIdx[*KeyTt2, *KeyTt3],
        DxT3,
        ExT2,
        Acc[CrudT3, RwT2],
        CtxT,
        RichIdx[*KeyTt4, *KeyTt5],
    ]: ...

    # 3. Context application, new value, rich index
    @overload
    def __getitem__(
        self: Data[
            ValT2,
            Idx[*KeyTt2],
            DxT2,
            ExT2,
            Acc[Any, CrudT3 | RwT2],
            Any,
            RichIdx[*KeyTt4],
        ],
        key: Data[
            ValT3,
            Idx[*KeyTt3],
            DxT3,
            ExT2,
            Acc[CrudT3, RwT2],
            Ctx[ValT2, ExtIdx[*KeyTt2], SxT2],
            RichIdx[*KeyTt5],
        ],
    ) -> Data[
        ValT3,
        ExtIdx[*KeyTt2, *KeyTt3],
        DxT3,
        ExT2,
        Acc[CrudT3, RwT2],
        CtxT,
        RichIdx[*KeyTt4, *KeyTt5],
    ]: ...

    # 4. Context application, kept value + DxT
    @overload
    def __getitem__(
        self: Data[ValT2, Idx[*KeyTt2], DxT2, ExT2, Acc[Any, CrudT3 | RwT2]],
        key: Data[
            Keep,
            Idx[*KeyTt3],
            Keep,
            ExT2,
            Acc[CrudT3, RwT2],
            Ctx[ValT2, ExtIdx[*KeyTt2], SxT2],
        ],
    ) -> Data[
        ValT2, ExtIdx[*KeyTt2, *KeyTt3], DxT2, ExT2, Acc[CrudT3, RwT2], CtxT, None
    ]: ...

    # 5. Context application, kept value
    @overload
    def __getitem__(
        self: Data[ValT2, Idx[*KeyTt2], DxT2, ExT2, Acc[Any, CrudT3 | RwT2]],
        key: Data[
            Keep,
            Idx[*KeyTt3],
            DxT3,
            ExT2,
            Acc[CrudT3, RwT2],
            Ctx[ValT2, ExtIdx[*KeyTt2], SxT2],
        ],
    ) -> Data[
        ValT2, ExtIdx[*KeyTt2, *KeyTt3], DxT3, ExT2, Acc[CrudT3, RwT2], CtxT, None
    ]: ...

    # 6. Context application, new value
    @overload
    def __getitem__(
        self: Data[ValT2, Idx[*KeyTt2], DxT2, ExT2, Acc[Any, CrudT3 | RwT2]],
        key: Data[
            ValT3,
            Idx[*KeyTt3],
            DxT3,
            ExT2,
            Acc[CrudT3, RwT2],
            Ctx[ValT2, ExtIdx[*KeyTt2], SxT2],
        ],
    ) -> Data[
        ValT3, ExtIdx[*KeyTt2, *KeyTt3], DxT3, ExT2, Acc[CrudT3, RwT2], CtxT, None
    ]: ...

    # 7. Base type selection
    @overload
    def __getitem__(
        self: Base,
        key: type[IdxblT2],
    ) -> Data[IdxblT2, MainIdx[IdxblT2], DxT, ExT, AccT, CtxT, RdxT]: ...

    # 8. Key list / slice filtering, scalar index type
    @overload
    def __getitem__(
        self: (
            Data[Any, Idx[KeyT2], Any, Any, Acc[RuT2]]
            | Data[Any, Any, Any, Any, Acc[RuT2], Any, RichIdx[KeyT2]]
        ),
        key: list[KeyT2] | slice,
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[RuT2], CtxT, RdxT]: ...

    # 9. Key list / slice filtering
    @overload
    def __getitem__(
        self: (
            Data[Any, Idx[*KeyTt2], Any, Any, Acc[RuT2]]
            | Data[Any, Any, Any, Any, Acc[RuT2], Any, RichIdx[*KeyTt2]]
        ),
        key: list[tuple[*KeyTt2]] | tuple[slice, ...],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[RuT2], CtxT, RdxT]: ...

    # 10. Key selection
    @overload
    def __getitem__(
        self: (
            Data[Any, Idx[*KeyTt3, *KeyTt2], SxT2]
            | Data[Any, Any, Any, Any, Acc[RuT2], Any, RichIdx[*KeyTt3, *KeyTt2]]
        ),
        key: tuple[*KeyTt3],
    ) -> Data[ValT, ExtIdx[*KeyTt2], DxT, ExT, AccT, CtxT, RdxT]: ...

    # 11. Key selection, scalar
    @overload
    def __getitem__(
        self: (
            Data[Any, Idx[KeyT3, *KeyTt2], SxT2]
            | Data[Any, Any, Any, Any, Acc[RuT2], Any, RichIdx[KeyT3, *KeyTt2]]
        ),
        key: KeyT3,
    ) -> Data[ValT, ExtIdx[*KeyTt2], DxT, ExT, AccT, CtxT, RdxT]: ...

    # 12. Key selection, fully rooted
    @overload
    def __getitem__(
        self: (
            Data[Any, Idx[*KeyTt3], Any, Any, Any, Base]
            | Data[Any, Any, Any, Any, Acc[RuT2], Base, RichIdx[*KeyTt3]]
        ),
        key: tuple[*KeyTt3],
    ) -> ValT: ...

    # 13. Key selection, fully rooted, scalar
    @overload
    def __getitem__(
        self: (
            Data[Any, Idx[KeyT3], Any, Any, Any, Base]
            | Data[Any, Any, Any, Any, Acc[RuT2], Base, RichIdx[KeyT3]]
        ),
        key: KeyT3,
    ) -> ValT: ...

    # 14. Positional item (column) selection, multiple
    @overload
    def __getitem__(
        self: Data[tuple[*tuple[ValT2, ...]]],
        key: ColSel[slice | Iterable[int]],
    ) -> Data[tuple[ValT2, ...], IdxT, DxT, ExT, AccT, CtxT, RdxT]: ...

    # 15. Positional item (column) selection, single
    @overload
    def __getitem__(
        self: Data[tuple[*tuple[ValT2, ...]]],
        key: ColSel[int],
    ) -> Data[ValT2, IdxT, DxT, ExT, AccT, CtxT, RdxT]: ...

    def __getitem__(
        self: Data | Base,
        key: Data | list | slice | tuple[slice, ...] | Hashable | type | ColSel[Any],
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
                            reduce=True,
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
                        self.typeref.scalar_typevar_map[CtxT].typeform, Base
                    ):
                        rooted = True

                    data = self[KeySelect(cast(tuple, key))]

                    if rooted:
                        vals = cast(Data[Any, Any, Any, Any, Any, Base], data).values()
                        return next(vals)

                    return data

                keymap = self._map_index_filters(key)
                return self[Filter.from_keymap(keymap)]
            case ColSel(col_sel):
                assert isinstance(
                    self, Align
                ), "Column selection is only supported for aligned datasets."
                assert col_sel is not None

                col_sel = (
                    col_sel if isinstance(col_sel, Iterable | slice) else (col_sel,)
                )
                return self._get_subset(col_sel)

    # Alignment:

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Col, ExT2, Acc[CrudT2, RwT2], CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, Tab, ExT2, Acc[CrudT2, RwT2], CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        Tab,
        ExT2,
        Acc[CrudT2, RwT2],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, DxT2, ExT2, Acc[CrudT2, RwT2], CtxT2],
        other: Data[tuple[*ValTt3], IdxT3, DxT2, ExT2, Acc[CrudT2, RwT2], CtxT2],
    ) -> Data[
        tuple[ValT, *ValTt3],
        IdxT | IdxT3,
        Tabs,
        ExT2,
        Acc[CrudT2, RwT2],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[tuple[*ValTt2], Any, Tab, ExT2, Acc[CrudT2, RwT2], CtxT2],
        other: Data[ValT3, IdxT3, Col, ExT2, Acc[CrudT2, RwT2], CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        Tab,
        ExT2,
        Acc[CrudT2, RwT2],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[tuple[*ValTt2], Any, DxT2, ExT2, Acc[CrudT2, RwT2], CtxT2],
        other: Data[ValT3, IdxT3, DxT2, ExT2, Acc[CrudT2, RwT2], CtxT2],
    ) -> Data[
        tuple[*ValTt2, ValT3],
        IdxT | IdxT3,
        Tabs,
        ExT2,
        Acc[CrudT2, RwT2],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, Col, ExT2, Acc[CrudT2, RwT2], CtxT2],
        other: Data[ValT3, IdxT3, Col, ExT2, Acc[CrudT2, RwT2], CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        Tab,
        ExT2,
        Acc[CrudT2, RwT2],
        CtxT2,
    ]: ...

    @overload
    def __matmul__(
        self: Data[Any, Any, DxT2, ExT2, Acc[CrudT2, RwT2], CtxT2],
        other: Data[ValT3, IdxT3, DxT2, ExT2, Acc[CrudT2, RwT2], CtxT2],
    ) -> Data[
        tuple[ValT, ValT3],
        IdxT | IdxT3,
        Tabs,
        ExT2,
        Acc[CrudT2, RwT2],
        CtxT2,
    ]: ...

    def __matmul__(
        self: Data[Any, Any, Any, ExT2, Acc[CrudT2, RwT2], CtxT2],
        other: Data[Any, IdxT3, Any, ExT2, Acc[CrudT2, RwT2], CtxT2],
    ) -> Data[
        tuple,
        IdxT | IdxT3,
        Tab | Tabs,
        ExT2,
        Acc[CrudT2, RwT2],
        CtxT2,
    ]:
        """Align two datasets."""
        if is_subtype(self.value_typeref.typeform, tuple):
            assert isinstance(self, Align)
            self_data = self.data
            self_types = self.value_types
        else:
            self_data = (self,)
            self_types = (self.value_typeref,)

        if is_subtype(other.value_typeref.typeform, tuple):
            assert isinstance(other, Align)
            other_data = other.data
            other_types = other.value_types
        else:
            other_data = (other,)
            other_types = (other.value_typeref,)

        return Align[
            tuple[*self_types, *other_types],
            IdxT | IdxT3,
            Tab | Tabs,
            ExT2,
            Acc[CrudT2, RwT2],
            CtxT2,
        ](
            data=self_data + other_data,
            context=self.root(),
        )

    # Reduction:

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], Idx[*KeyTt2], Shape[SxT3], ExT2],
        op: Callable[[ValT2, ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        ExtIdx[*KeyTt2],
        SxT3,
        ExT2,
        Acc[R, R],
        CtxT,
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[tuple[ValT2, ...], Idx[*KeyTt2], SxT3, ExT2],
        op: Callable[[ValT2, ValT4], ValT3],
        right: ValT4,
    ) -> Data[
        ValT3,
        ExtIdx[*KeyTt2],
        SxT3,
        ExT2,
        Acc[R, R],
        CtxT,
    ]: ...

    @overload
    def _map_reduce_operator(
        self: Data[ValT2, Idx[*KeyTt2], SxT3, ExT2],
        op: Callable[[ValT2], ValT3],
        right: Literal[Not.defined] = ...,
    ) -> Data[
        ValT3,
        ExtIdx[*KeyTt2],
        SxT3,
        ExT2,
        Acc[R, R],
        CtxT,
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
        Acc[R, R],
        CtxT,
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
            reduce=True,
        )
        assert issubclass(self.value_typeref.common_type, tuple)
        return cast(
            Data,
            self[reduction],
        )

    # Comparison:

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[ValT2, Any, Col, ExT2, Any, CtxT2],
        other: Data[ValT3, IdxT3, Col, ExT2, Any, CtxT2],
    ) -> Data[
        bool,
        Any,
        Col,
        ExT2,
        Acc[R, R],
        CtxT2,
    ]: ...

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self: Data[Any, Any, Col, ExT2, Any, CtxT2],
        other: Any,
    ) -> Data[
        bool,
        Any,
        Col,
        ExT2,
        Acc[R, R],
        CtxT2,
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
            Acc[R, R],
            Any,
        ]
        | bool
    ):
        if not isinstance(other, Data):
            return self._map_reduce_operator(operator.eq, other)

        alignment = self @ other
        return alignment._map_reduce_operator(operator.eq)

    def isin(
        self: Data[Any, Any, Col, ExT2, Any, CtxT2],
        other: Collection[ValT2] | slice,
    ) -> Data[
        bool,
        Any,
        Col,
        ExT2,
        Acc[R, R],
        CtxT2,
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
                Acc[R, R],
                CtxT2,
            ],
            self[mapping],
        )

    # Index set operations:

    def __or__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        Any,
        DxT2,
        ExT2,
        Acc[R, R],
        CtxT2,
    ]:
        """Upsert / union with right preference."""
        alignment = self @ other
        return cast(
            Data[
                ValT2 | ValT3,
                Any,
                DxT2,
                ExT2,
                Acc[R, R],
                CtxT2,
            ],
            alignment[
                Transform(
                    func=operator.or_,
                    frame_func=partial(frame_coalesce, coalesce="right"),
                    reduce=True,
                )
            ],
        )

    def __and__(
        self: Data[Any, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT3,
        Any,
        DxT2,
        ExT2,
        Acc[R, R],
        CtxT2,
    ]:
        """Intersection with right preference."""
        alignment = Align(
            context=self.root(),
            data=(self, other),
            join="inner",
        )
        return cast(
            Data[
                ValT3,
                Any,
                DxT2,
                ExT2,
                Acc[R, R],
                CtxT2,
            ],
            alignment[
                Transform(
                    func=operator.and_,
                    frame_func=partial(frame_coalesce, coalesce="right"),
                    reduce=True,
                )
            ],
        )

    def __xor__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        Any,
        DxT2,
        ExT2,
        Acc[R, R],
        CtxT2,
    ]:
        """XOR."""
        raise NotImplementedError()

    def __lshift__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        Any,
        DxT2,
        ExT2,
        Acc[R, R],
        CtxT2,
    ]:
        """Take only left index, but update from right."""
        alignment = Align(
            context=self.root(),
            data=(self, other),
            join="left",
        )
        return cast(
            Data[
                ValT2 | ValT3,
                Any,
                DxT2,
                ExT2,
                Acc[R, R],
                CtxT2,
            ],
            alignment[
                Transform(
                    func=lambda _, x: x,
                    frame_func=partial(frame_coalesce, coalesce="right"),
                    reduce=True,
                )
            ],
        )

    def __rshift__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[ValT3, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2 | ValT3,
        Any,
        DxT2,
        ExT2,
        Acc[R, R],
        CtxT2,
    ]:
        """Take only right index, but update from left."""
        alignment = Align(
            context=self.root(),
            data=(self, other),
            join="right",
        )
        return cast(
            Data[
                ValT2 | ValT3,
                Any,
                DxT2,
                ExT2,
                Acc[R, R],
                CtxT2,
            ],
            alignment[
                Transform(
                    func=lambda _, x: x,
                    frame_func=partial(frame_coalesce, coalesce="left"),
                    reduce=True,
                )
            ],
        )

    def __floordiv__(
        self: Data[ValT2, Any, DxT2, ExT2, Any, CtxT2],
        other: Data[Any, Any, DxT2, ExT2, Any, CtxT2],
    ) -> Data[
        ValT2,
        Any,
        DxT2,
        ExT2,
        Acc[R, R],
        CtxT2,
    ]:
        """Remove overlapping elements of right from left."""
        raise NotImplementedError()

    @overload
    def __ipow__(
        self: Data[Any, Any, Col, SQL, Acc[C], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ipow__(
        self: Data[Any, Any, Col, Any, Acc[C], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ipow__(
        self: Data[Any, Any, Tab, SQL, Acc[C], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ipow__(
        self: Data[Any, Any, Tab, Any, Acc[C], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ipow__(
        self: Data[Any, Any, Tabs, SQL, Acc[C], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ipow__(
        self: Data[Any, Any, Tabs, Any, Acc[C], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ipow__(
        self: Data[Any, Any, Any, SQL, Acc[C], Base],
        input_data: InputData[ValT, Not, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ipow__(
        self: Data[Any, Any, Any, Any, Acc[C], Base],
        input_data: InputData[ValT, Not, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __ipow__(
        self: Data[Any, Any, Any, Any, Acc[C], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """Insert / union with left preference."""
        mutations = self._mutation(input_data, mode={C})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    @overload
    def __ilshift__(
        self: Data[Any, Any, Col, SQL, Acc[U], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Col, Any, Acc[U], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tab, SQL, Acc[U], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tab, Any, Acc[U], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tabs, SQL, Acc[U], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ilshift__(
        self: Data[Any, Any, Tabs, Any, Acc[U], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __ilshift__(
        self: Data[Any, Any, Any, Any, Acc[U], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """Update self from input, no insert."""
        mutations = self._mutation(input_data, mode={U})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    @overload
    def __ifloordiv__(
        self: Data[Any, Any, Col, SQL, Acc[D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ifloordiv__(
        self: Data[Any, Any, Col, Any, Acc[D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ifloordiv__(
        self: Data[Any, Any, Tab, SQL, Acc[D], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ifloordiv__(
        self: Data[Any, Any, Tab, Any, Acc[D], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ifloordiv__(
        self: Data[Any, Any, Tabs, SQL, Acc[D], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ifloordiv__(
        self: Data[Any, Any, Tabs, Any, Acc[D], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __ifloordiv__(
        self: Data[Any, Any, Any, Any, Acc[D], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """Remove overlapping elements of input from self."""
        mutations = self._mutation(input_data, mode={D})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    @overload
    def __ior__(
        self: Data[Any, Any, Col, SQL, Acc[C | U], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Col, Any, Acc[C | U], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tab, SQL, Acc[C | U], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tab, Any, Acc[C | U], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tabs, SQL, Acc[C | U], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Tabs, Any, Acc[C | U], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Any, SQL, Acc[C | U], Base],
        input_data: InputData[ValT, Not, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ior__(
        self: Data[Any, Any, Any, Any, Acc[C | U], Base],
        input_data: InputData[ValT, Not, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __ior__(
        self: Data[Any, Any, Any, Any, Acc[C | U], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """Upsert / union with right preference."""
        mutations = self._mutation(input_data, mode={C, U})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    @overload
    def __iand__(
        self: Data[Any, Any, Col, SQL, Acc[U | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __iand__(
        self: Data[Any, Any, Col, Any, Acc[U | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __iand__(
        self: Data[Any, Any, Tab, SQL, Acc[U | D], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __iand__(
        self: Data[Any, Any, Tab, Any, Acc[U | D], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __iand__(
        self: Data[Any, Any, Tabs, SQL, Acc[U | D], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __iand__(
        self: Data[Any, Any, Tabs, Any, Acc[U | D], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __iand__(
        self: Data[Any, Any, Any, SQL, Acc[U | D], Base],
        input_data: InputData[ValT, Not, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __iand__(
        self: Data[Any, Any, Any, Any, Acc[U | D], Base],
        input_data: InputData[ValT, Not, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __iand__(
        self: Data[Any, Any, Any, Any, Acc[U | D], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """Intersection with right preference."""
        mutations = self._mutation(input_data, mode={U, D})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    @overload
    def __ixor__(
        self: Data[Any, Any, Col, SQL, Acc[C | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Col, Any, Acc[C | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tab, SQL, Acc[C | D], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tab, Any, Acc[C | D], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tabs, SQL, Acc[C | D], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __ixor__(
        self: Data[Any, Any, Tabs, Any, Acc[C | D], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __ixor__(
        self: Data[Any, Any, Any, Any, Acc[C | D], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """XOR."""
        mutations = self._mutation(input_data, mode={C, D})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    @overload
    def __irshift__(
        self: Data[Any, Any, Col, SQL, Acc[C | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __irshift__(
        self: Data[Any, Any, Col, Any, Acc[C | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __irshift__(
        self: Data[Any, Any, Tab, SQL, Acc[C | D], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __irshift__(
        self: Data[Any, Any, Tab, Any, Acc[C | D], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __irshift__(
        self: Data[Any, Any, Tabs, SQL, Acc[C | D], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __irshift__(
        self: Data[Any, Any, Tabs, Any, Acc[C | D], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __irshift__(
        self: Data[Any, Any, Any, Any, Acc[C | D], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """Keep index of input and update from self."""
        mutations = self._mutation(input_data, mode={C, D})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    @overload
    def __imatmul__(
        self: Data[Any, Any, Col, SQL, Acc[C | U | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series | sqla.ColumnElement, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __imatmul__(
        self: Data[Any, Any, Col, Any, Acc[C | U | D], Base],
        input_data: InputData[ValT, pl.Series | pd.Series, Not],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __imatmul__(
        self: Data[Any, Any, Tab, SQL, Acc[C | U | D], Base],
        input_data: InputData[
            ValT,
            pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause,
            pl.Series | pd.Series | sqla.ColumnElement,
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __imatmul__(
        self: Data[Any, Any, Tab, Any, Acc[C | U | D], Base],
        input_data: InputData[ValT, pl.DataFrame | pd.DataFrame, pl.Series | pd.Series],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __imatmul__(
        self: Data[Any, Any, Tabs, SQL, Acc[C | U | D], Base],
        input_data: InputData[
            ValT, Not, pl.DataFrame | pd.DataFrame | sqla.Select | sqla.FromClause
        ],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    @overload
    def __imatmul__(
        self: Data[Any, Any, Tabs, Any, Acc[C | U | D], Base],
        input_data: InputData[ValT, Not, pl.DataFrame | pd.DataFrame],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]: ...

    def __imatmul__(
        self: Data[Any, Any, Any, Any, Acc[C | U | D], Base],
        input_data: InputData[ValT, Any, Any],
    ) -> Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT]:
        """Aligned assignment -> override left with right, including deletion."""
        mutations = self._mutation(input_data, mode={C, U, D})

        for mutation in mutations:
            self.root().connection.execute(mutation)

        return cast(Data[ValT, IdxT, DxT, ExT, Acc[CrudT], CtxT], self)

    def __setitem__(
        self,
        key: Any,
        input_data: Any,
    ) -> None:
        """Generic setitem."""
        pass

    def __delitem__(self, key: Any) -> None:
        """Generic delitem."""
        pass

    def describe(self) -> str:
        """Describe the data."""
        raise NotImplementedError()


@dataclass(kw_only=True)
class Registry(
    Data[IdxblT, MainIdx[IdxblT], Tab, ExT, Acc[CrudT, RwT], BaseT, RichIdx[IdxblT]],
    ABC,
):
    """Represent a base data type collection."""

    _instance_map: dict[Hashable, IdxblT] = field(default_factory=dict)

    @override
    def _id(self) -> str:
        ctx_module = getmodule(self.value_typeref.common_type)
        return (
            (
                ctx_module.__name__ + "." + self.value_typeref.common_type.__name__
                if ctx_module is not None
                else self.value_typeref.common_type.__name__
            )
            + "."
            + self._id()
        )


TupT = TypeVar("TupT", bound=tuple, covariant=True)


@dataclass(kw_only=True)
class Align(Data[TupT, IdxT, DxT, ExT, AccT, CtxT]):
    """Alignment of multiple props."""

    data: tuple[Data[Any, IdxT, Any, Any, AccT, CtxT], ...]
    join: Literal["left", "right", "outer", "inner"] = "outer"
    match_idx_on: Literal["value", "value+id", "value+id+path"] = "value"

    @cached_prop
    def value_types(self) -> tuple[SingleTypeDef[ValT] | UnionType, ...]:
        """Get the value types."""
        return tuple(d.value_typeref.typeform for d in self.data)

    @override
    def _id(self) -> str:
        # TODO: Implement this method for the Align class.
        raise NotImplementedError()

    @override
    def _index(
        self,
    ) -> IdxT:
        raise NotImplementedError()

    @override
    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        raise NotImplementedError()

    @override
    def _mutation(
        self: Align[Any, Any, Any, Any, Acc[CrudT2]],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[CrudT2]] = {R},
    ) -> Sequence[sqla.Executable]:
        raise NotImplementedError()

    def _get_subset(
        self, subset: Iterable[int] | slice
    ) -> Align[Any, IdxT, DxT, ExT, AccT, CtxT]:
        """Get a subset of the aligned data."""
        raise NotImplementedError()


class Transform(
    Data[
        ValT,
        IdxT,
        DxT,
        ExT,
        Acc[R, RwT],
        Interface[ArgT, ArgIdxT, ArgDxT],
    ],
    Generic[ArgT, ArgIdxT, ArgDxT, ValT, IdxT, DxT, ExT, RwT],
):
    """Apply a mapping function to a dataset."""

    @overload
    def __init__(
        self: Transform[ValT2, ExtIdx[()], SxT2, ValT3, ExtIdx[()], SxT3, ExT2],
        func: Callable[[ValT2], ValT3],
        frame_func: Callable[[Frame[ExT2, SxT2]], Frame[ExT2, SxT3]],
    ): ...

    @overload
    def __init__(
        self: Transform[
            Iterable[ValT2], ExtIdx[()], Shape[SxT3], ValT3, ExtIdx[()], SxT3, ExT2
        ],
        func: Callable[[Iterable[ValT2]], ValT3] | Callable[[ValT2, ValT2], ValT3],
        frame_func: Callable[[Frame[ExT2, Shape[SxT3]]], Frame[ExT2, SxT3]],
        reduce: Literal[True],
    ): ...

    @overload
    def __init__(
        self: Transform[
            Iterable[ValT2], ExtIdx[()], SxT2, ValT3, ExtIdx[()], SxT3, ExT2
        ],
        func: Callable[[Iterable[ValT2]], ValT3] | Callable[[ValT2, ValT2], ValT3],
        frame_func: Callable[[Frame[ExT2, SxT2], Frame[ExT2, SxT2]], Frame[ExT2, SxT3]],
        reduce: Literal[True],
    ): ...

    def __init__(self, *args, **kwargs):  # noqa: D107
        self.context = Interface()
        # TODO: Implement the constructor for the Transform class.

    @override
    def _id(self) -> str:
        # TODO: Implement this method for the Transform class.
        raise NotImplementedError()

    @override
    def _index(
        self,
    ) -> IdxT:
        raise NotImplementedError()

    @override
    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        raise NotImplementedError()

    @override
    def _mutation(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[R]] = {R},
    ) -> Sequence[sqla.Executable]:
        return []


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
        ExtIdx[()],
        Keep,
        ExT,
        Acc[RuT],
        Interface[Any, ExtIdx[()], Any],
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

    @override
    def _id(self) -> str:
        # TODO: Implement this method for the Filter class.
        raise NotImplementedError()

    @override
    def _index(
        self,
    ) -> ExtIdx[()]:
        raise NotImplementedError()

    @override
    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        raise NotImplementedError()

    @override
    def _mutation(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Filter[Any, RuT2],
        input_data: InputData[ValT, InputFrame, InputFrame],
        mode: Set[type[RuT2]] = {U},
    ) -> Sequence[sqla.Executable]:
        raise NotImplementedError()


class KeySelect(
    Data[
        Keep,
        ExtIdx[()],
        Keep,
        SQL,
        Acc[R | U],
        Interface[Any, ExtIdx[*KeyTt], Any],
    ]
):
    """Select a specific key value (prefix)."""

    def __init__(self: KeySelect[ExtIdx[*KeyTt2]], key: tuple[*KeyTt2]):  # noqa: D107
        self.context = Interface()
        self.key = key

    @override
    def _id(self) -> str:
        # TODO: Implement this method for the KeySelect class.
        raise NotImplementedError()

    @override
    def _index(
        self,
    ) -> ExtIdx[()]:
        raise NotImplementedError()

    @override
    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        raise NotImplementedError()

    @override
    def _mutation(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        input_data: InputData[Any, InputFrame, InputFrame],
        mode: Set[type[R]] = {R},
    ) -> Sequence[sqla.Executable]:
        return []
