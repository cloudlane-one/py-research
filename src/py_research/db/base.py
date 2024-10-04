"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import Field, dataclass, field
from datetime import date, datetime, time
from enum import Enum, auto
from functools import cached_property, partial, reduce
from inspect import get_annotations, getmodule
from io import BytesIO
from pathlib import Path
from secrets import token_hex
from types import GenericAlias, ModuleType, NoneType, UnionType, new_class
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
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)
from uuid import UUID, uuid4

import openpyxl
import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.dialects.mysql as mysql
import sqlalchemy.dialects.postgresql as postgresql
import sqlalchemy.orm as orm
import sqlalchemy.sql.type_api as sqla_types
import sqlalchemy.sql.visitors as sqla_visitors
import yarl
from bidict import bidict
from cloudpathlib import CloudPath
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from sqlalchemy_utils import UUIDType
from typing_extensions import TypeVar, TypeVarTuple
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.data import copy_and_override
from py_research.files import HttpFile
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    SingleTypeDef,
    get_lowest_common_base,
    has_type,
    is_subtype,
)

VT = TypeVar("VT", covariant=True)
VTI = TypeVar("VTI", default=Any)
VT2 = TypeVar("VT2")
VT3 = TypeVar("VT3")
VT4 = TypeVar("VT4")

WT = TypeVar("WT", bound="R", default="RW", covariant=True)

KT = TypeVar("KT", bound=Hashable, contravariant=True, default=Any)
KT2 = TypeVar("KT2", bound=Hashable)
KT3 = TypeVar("KT3", bound=Hashable)

RT = TypeVar("RT", covariant=True, bound="Record", default="Record")
RTS = TypeVar("RTS", covariant=True, bound="Record", default="Record")
RTT = TypeVarTuple("RTT")
RT2 = TypeVar("RT2", bound="Record")
RT3 = TypeVar("RT3", bound="Record")

IT = TypeVar(
    "IT",
    covariant=True,
    bound="Index",
    default="BaseIdx",
)
ITT = TypeVarTuple("ITT")

BT = TypeVar(
    "BT",
    bound="Backend | Record",
    default="Record",
    covariant=True,
)

PT = TypeVar("PT", covariant=True, bound="Record", default="Record")
PT2 = TypeVar("PT2", bound="Record")

TT = TypeVar("TT", covariant=True, bound=tuple | None, default=None)

MT = TypeVar("MT", bound="MutateIdx", covariant=True)
MT2 = TypeVar("MT2", bound="MutateIdx")

DT = TypeVar("DT", bound=pd.DataFrame | pl.DataFrame)
LT = TypeVar("LT", covariant=True, bound="Record | None", default="Record | None")
LT2 = TypeVar("LT2", bound="Record | None")

CT = TypeVar("CT", bound="ColIdx", covariant=True, default=Hashable)

Params = ParamSpec("Params")


type Backend = LiteralString | None

type Index = Hashable | BaseIdx | SingleIdx | Filt
type ColIdx = Hashable | SingleIdx
type MutateIdx = Hashable | BaseIdx

type IdxStart[Key: Hashable] = Key | tuple[Key, *tuple[Any, ...]]
type IdxEnd[Key: Hashable] = tuple[*tuple[Any, ...], Key]
type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]
type IdxTupStart[*IdxTup] = tuple[*IdxTup, *tuple[Any]]
type IdxTupStartEnd[*IdxTup, Key2: Hashable] = tuple[*IdxTup, *tuple[Any], Key2]

type RecordValue[Rec: Record] = Rec | Iterable[Rec] | Mapping[Any, Rec] | None


class BaseIdx:
    """Singleton to mark dataset index as default index."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


class SingleIdx:
    """Singleton to mark dataset index as a single value."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


class Filt(Generic[MT]):
    """Singleton to mark dataset index as filtered."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


class R:
    """Read-only flag."""


class RO(R):
    """Read-only flag."""


class RW(RO):
    """Read-write flag."""


type RecInput[Rec: Record, Key: MutateIdx] = pd.DataFrame | pl.DataFrame | Iterable[
    Rec
] | Mapping[Key, Rec] | sqla.Select | Rec

type PartialRec[Rec: Record] = Mapping[DB, Any]

type PartialRecInput[Rec: Record, Key: MutateIdx] = RecInput[Rec, Any] | Iterable[
    PartialRec[Rec]
] | Mapping[Key, PartialRec[Rec]] | PartialRec[Rec]

type ValInput[Val, Key: Hashable] = pd.Series | pl.Series | Mapping[
    Key, Val
] | sqla.Select[tuple[Key, Val]] | Val


def map_df_dtype(c: pd.Series | pl.Series) -> type | None:
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


def map_col_dtype(c: sqla.ColumnElement) -> type | None:
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


def props_from_data(
    data: pd.DataFrame | pl.DataFrame | sqla.Select,
    foreign_keys: Mapping[str, Col] | None = None,
) -> list[Prop]:
    """Extract prop definitions from dataframe or query."""
    foreign_keys = foreign_keys or {}

    def _gen_prop(
        name: str, data: pd.Series | pl.Series | sqla.ColumnElement
    ) -> Prop[Any, Any]:
        is_rel = name in foreign_keys
        value_type = (
            map_df_dtype(data)
            if isinstance(data, pd.Series | pl.Series)
            else map_col_dtype(data)
        ) or Any
        col = Col[value_type](
            primary_key=True,
            _name=name if not is_rel else f"fk_{name}",
            _type=PropType(Col[value_type]),
            record_set=DynRecord._db,
        )
        return (
            col
            if not is_rel
            else RelSet(
                on={col: foreign_keys[name]},
                _type=PropType(RelSet[cast(type, foreign_keys[name].value_type)]),
                _name=f"rel_{name}",
                parent_type=DynRecord,
            )
        )

    columns = (
        [data[col] for col in data.columns]
        if isinstance(data, pd.DataFrame | pl.DataFrame)
        else list(data.columns)
    )

    index_props = []
    if isinstance(data, pd.DataFrame):
        levels = {name: name for name in data.index.names}
        if any(lvl is None for lvl in levels.values()):
            levels = {i: f"index_{i}" for i in range(len(levels))}

        for level_name, prop_name in levels.items():
            index_props.append(
                _gen_prop(
                    prop_name, data.index.get_level_values(level_name).to_series()
                )
            )

    return [
        *index_props,
        *(_gen_prop(str(col.name), col) for col in columns),
    ]


def _remove_external_fk(table: sqla.Table):
    """Dirty vodoo to remove external FKs from existing table."""
    for c in table.columns.values():
        c.foreign_keys = set(
            fk
            for fk in c.foreign_keys
            if fk.constraint and fk.constraint.referred_table.schema == table.schema
        )

    table.foreign_keys = set(  # type: ignore[reportAttributeAccessIssue]
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


class State(Enum):
    """Demark load status."""

    undef = auto()


@dataclass(frozen=True)
class PropType(Generic[VT]):
    """Reference to a type."""

    hint: str | type[Prop[VT]] | None = None
    ctx: ModuleType | None = None
    typevar_map: dict[TypeVar, SingleTypeDef] = field(default_factory=dict)

    def prop_type(self) -> type[Prop]:
        """Resolve the property type reference."""
        hint = self.hint or Prop

        if isinstance(hint, type | GenericAlias):
            base = get_origin(hint)
            assert base is not None and issubclass(base, Prop)
            return base
        else:
            return (
                Col
                if "Col" in hint
                else RelSet if "RelSet" in hint else Rel if "Rel" in hint else Prop
            )

    def _to_typedef(
        self, hint: SingleTypeDef | UnionType | TypeVar | str | ForwardRef
    ) -> SingleTypeDef:
        typedef = hint

        if isinstance(typedef, str):
            typedef = eval(
                typedef, {**globals(), **(vars(self.ctx) if self.ctx else {})}
            )

        if isinstance(typedef, TypeVar):
            typedef = self.typevar_map.get(typedef) or typedef.__bound__ or object

        if isinstance(typedef, ForwardRef):
            typedef = typedef._evaluate(
                {**globals(), **(vars(self.ctx) if self.ctx else {})},
                {},
                recursive_guard=frozenset(),
            )

        if isinstance(typedef, UnionType):
            union_types: set[type] = {
                get_origin(union_arg) or union_arg for union_arg in get_args(typedef)
            }
            typedef = get_lowest_common_base(union_types)

        return cast(SingleTypeDef, typedef)

    def _to_type(self, hint: SingleTypeDef | UnionType | TypeVar) -> type:
        if isinstance(hint, type):
            return hint

        typedef = (
            self._to_typedef(hint) if isinstance(hint, UnionType | TypeVar) else hint
        )
        orig = get_origin(typedef)

        if orig is None or orig is Literal:
            return object

        assert isinstance(orig, type)
        return new_class(
            orig.__name__ + "_" + token_hex(5),
            (hint,),
            None,
            lambda ns: ns.update({"_src_mod": self.ctx}),
        )

    @cached_property
    def _generic_type(self) -> SingleTypeDef | UnionType:
        """Resolve the generic property type reference."""
        hint = self.hint or Prop
        generic = self._to_typedef(hint)
        assert is_subtype(generic, Prop)

        return generic

    @cached_property
    def _generic_args(self) -> tuple[SingleTypeDef | UnionType | TypeVar, ...]:
        """Resolve the generic property type reference."""
        args = get_args(self._generic_type)
        return tuple(self._to_typedef(hint) for hint in args)

    def value_type(self: PropType[VT2]) -> SingleTypeDef[VT2]:
        """Resolve the value type reference."""
        args = self._generic_args
        if len(args) == 0:
            return cast(type[VT2], object)

        arg = args[0]
        return cast(SingleTypeDef[VT2], self._to_typedef(arg))

    def record_type(self: PropType[Record | Iterable[Record]]) -> type[Record]:
        """Resolve the record type reference."""
        assert is_subtype(self._generic_type, RelSet)

        args = self._generic_args
        assert len(args) > 0

        recs = args[0]
        if isinstance(recs, TypeVar):
            recs = self._to_type(recs)

        rec_args = get_args(recs)
        rec_arg = recs

        if is_subtype(recs, Iterable):
            assert len(rec_args) >= 1
            rec_arg = rec_args[0]

        rec_type = self._to_type(rec_arg)
        assert issubclass(rec_type, Record)

        return rec_type

    def link_type(self: PropType[Record | Iterable[Record]]) -> type[Record | None]:
        """Resolve the record type reference."""
        assert is_subtype(self._generic_type, RelSet)
        args = self._generic_args
        rec = args[1]
        rec_type = self._to_type(rec)

        if not issubclass(rec_type, Record):
            return NoneType

        return rec_type


@dataclass(eq=False)
class Prop(Generic[VT, WT]):
    """Reference a property of a record."""

    _type: PropType[VT] = field(default_factory=PropType[VT])
    _name: str | None = None

    alias: str | None = None
    default: VT | None = None
    default_factory: Callable[[], VT] | None = None
    init: bool = True

    getter: Callable[[Record], VT] | None = None
    setter: Callable[[Record, VT], None] | None = None

    local: bool = False

    @property
    def name(self) -> str:
        """Property name."""
        if self.alias is not None:
            return self.alias

        assert self._name is not None
        return self._name

    def __set_name__(self, _, name: str) -> None:  # noqa: D105
        if self._name is None:
            self._name = name
        else:
            assert name == self._name

    def __hash__(self) -> int:
        """Hash the Prop."""
        return gen_int_hash(self)


type DirectLink[Rec: Record] = (
    Col[Any, Any, Any, Any, Any]
    | Iterable[Col[Any, Any, Any, Any, Any]]
    | dict[Col, Col[Rec, Any, Any, Any, Any]]
)

type BackLink[Rec: Record] = (RelSet[Any, Any, Any, Any, Record, Rec] | type[Rec])

type BiLink[Rec: Record, Rec2: Record] = (
    RelSet[Rec, Any, Any, Any, Record, Rec2]
    | tuple[
        RelSet[Any, Any, Any, Any, Record, Rec2],
        RelSet[Rec, Any, Any, Any, Record, Rec2],
    ]
    | type[Rec2]
)


@overload
def prop(
    *,
    default: VT2 | RT2 | None = ...,
    default_factory: Callable[[], VT2 | RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: bool = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: Literal[True],
    init: bool = ...,
) -> Any: ...


@overload
def prop(
    *,
    default: VT2 | RT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: bool = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Col[VT2]: ...


@overload
def prop(
    *,
    default: VT2 | RT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: bool | Literal["fk"] = ...,
    primary_key: bool | Literal["fk"] = ...,
    link_on: DirectLink[RT2],
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[RT2]: ...


@overload
def prop(
    *,
    default: VT2 | RT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal["fk"],
    primary_key: bool | Literal["fk"] = ...,
    link_on: DirectLink[RT2] | None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[RT2]: ...


@overload
def prop(
    *,
    default: VT2 | RT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: bool | Literal["fk"] = ...,
    primary_key: Literal["fk"],
    link_on: DirectLink[RT2] | None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[RT2]: ...


@overload
def prop(
    *,
    default: VT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], VT2],
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Prop[VT2, RO]: ...


@overload
def prop(
    *,
    default: VT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], VT2],
    setter: Callable[[Record, VT2], None],
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Prop[VT2, RW]: ...


@overload
def prop(
    *,
    default: VT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], VT2],
    setter: None = ...,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement],
    local: bool = ...,
    init: bool = ...,
) -> Col[VT2, RO]: ...


@overload
def prop(
    *,
    default: VT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], VT2],
    setter: Callable[[Record, VT2], None],
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement],
    local: bool = ...,
    init: bool = ...,
) -> Col[VT2, RW]: ...


@overload
def prop(
    *,
    default: VT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[True] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], VT2],
    setter: None = ...,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Col[VT2, RO]: ...


@overload
def prop(
    *,
    default: VT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], VT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[True] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], VT2],
    setter: Callable[[Record, VT2], None],
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Col[VT2, RW]: ...


@overload
def prop(  # type: ignore[reportOverlappingOverload]
    *,
    default: RT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: BackLink[RT2] | None = ...,
    link_via: BiLink[RT2, Link],
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> RelSet[RT2, None]: ...


@overload
def prop(
    *,
    default: RT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: BackLink[RT2] | None = ...,
    link_via: BiLink[RT2, RT3] | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> RelSet[RT2, RT3]: ...


@overload
def prop(
    *,
    default: RT2 | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    default_factory: Callable[[], RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: BackLink[RT2] | None = ...,
    link_via: BiLink[RT2, RT3] | None = ...,  # type: ignore[reportInvalidTypeVarUse]
    order_by: Mapping[Col[Any, Any, Any, Record, RT2 | RT3], int],
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> RelSet[RT2, RT3, int]: ...


@overload
def prop(
    *,
    default: RT2 | None = ...,
    default_factory: Callable[[], RT2 | VT3] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    link_on: None = ...,
    link_from: BackLink[RT2] | None = ...,
    link_via: BiLink[RT2, RT3] | None = ...,
    order_by: None = ...,
    map_by: Col[VT4, Any, Any, Record, RT2 | RT3],
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> RelSet[RT2, RT3, VT4]: ...


def prop(
    *,
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
    alias: str | None = None,
    index: bool | Literal["fk"] = False,
    primary_key: bool | Literal["fk"] = False,
    link_on: DirectLink[Record] | None = None,
    link_from: BackLink[Record] | None = None,
    link_via: BiLink[Record, Record] | None = None,
    order_by: Mapping[Col, int] | None = None,
    map_by: Col | None = None,
    getter: Callable[[Record], Any] | None = None,
    setter: Callable[[Record, Any], None] | None = None,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = None,
    local: bool = False,
    init: bool = True,
) -> Any:
    """Define a backlinking relation to another record."""
    if local:
        return Prop(
            default=default,
            default_factory=default_factory,
            alias=alias,
            init=init,
            _type=PropType(),
            local=True,
        )

    if any(
        a is not None for a in (link_on, link_from, link_via, order_by, map_by)
    ) or any(a == "fk" for a in (index, primary_key)):
        return RelSet(
            default=default,
            default_factory=default_factory,
            alias=alias,
            init=init,
            index=index == "fk",
            primary_key=primary_key == "fk",
            on=(
                link_on
                if link_on is not None
                else link_from if link_from is not None else link_via
            ),
            order_by=order_by,
            map_by=map_by,
            _type=PropType(),
        )

    if getter is not None and not (primary_key or index or sql_getter is not None):
        return Prop(
            default=default,
            default_factory=default_factory,
            alias=alias,
            init=init,
            getter=getter,
            setter=setter,
            _type=PropType(),
        )

    return Col(
        default=default,
        default_factory=default_factory,
        alias=alias,
        init=init,
        index=index is not False,
        primary_key=primary_key is not False,
        getter=getter,
        setter=setter,
        sql_getter=sql_getter,
        _type=PropType(),
    )


@dataclass
class RecordLinks(Generic[RT]):
    """Descriptor to access the link records of a record's rels."""

    rec: RT

    @overload
    def __getitem__(self, rel: RelSet[Record, RT, SingleIdx, RW, Record, PT]) -> PT: ...

    @overload
    def __getitem__(
        self, rel: RelSet[Record, RT, SingleIdx | None, RW, Record, PT]
    ) -> PT | None: ...

    @overload
    def __getitem__(
        self, rel: RelSet[Record, RT, KT, RW, Record, PT]
    ) -> dict[KT, PT]: ...

    @overload
    def __getitem__(
        self, rel: RelSet[Record, RT, BaseIdx, RW, Record, PT]
    ) -> list[PT]: ...

    def __getitem__(self, rel: RelSet[Record, RT, Any, RW, Record, PT]) -> RecordValue:
        """Return the model of the given relation, if any."""
        if rel.link_type is Record:
            return DynRecord()

        if rel.name not in self.rec._edge_dict:
            self.rec._edge_dict[rel.name] = self.rec._db._load_prop(
                rel, self.rec._index
            )

        return self.rec._edge_dict[rel.name]

    @overload
    def __setitem__(
        self, rel: RelSet[Record, RT, SingleIdx | None, RW, Record, PT], value: PT
    ) -> None: ...

    @overload
    def __setitem__(
        self, rel: RelSet[Record, RT, KT, RW, Record, PT], value: Mapping[KT, PT]
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        rel: RelSet[Record, RT, BaseIdx, RW, Record, PT],
        value: Mapping[Any, PT] | Iterable[PT],
    ): ...

    def __setitem__(
        self, rel: RelSet[Record, RT, Any, RW, Record, PT], value: RecordValue
    ) -> None:
        """Return the model of the given relation, if any."""
        self.rec._edge_dict[rel.name] = value


class RecordMeta(type):
    """Metaclass for record types."""

    _record_bases: list[type[Record]]
    _base_pks: dict[
        type[Record],
        dict[Col[Hashable], Col[Hashable]],
    ]

    _defined_props: dict[str, Col | RelSet]
    _defined_cols: dict[str, Col]
    _defined_rels: dict[str, RelSet]

    _is_record: bool
    _template: bool
    _src_mod: ModuleType | None

    def __init__(cls, name, bases, dct):
        """Initialize a new record type."""
        super().__init__(name, bases, dct)

        prop_defs = {
            name: PropType(hint, ctx=cls._src_mod)
            for name, hint in get_annotations(cls).items()
            if issubclass(get_origin(hint) or type, Prop)
            or (isinstance(hint, str) and ("Attr" in hint or "Rel" in hint))
        }

        for prop_name, prop_type in prop_defs.items():
            if prop_name not in cls.__dict__:
                setattr(
                    cls,
                    prop_name,
                    prop_type.prop_type()(_name=prop_name, _type=prop_type),
                )
            else:
                prop = cls.__dict__[prop_name]
                assert isinstance(prop, Prop)
                prop._type = prop_type

        cls._record_bases = []
        base_types: Iterable[type | GenericAlias] = (
            cls.__dict__["__orig_bases__"]
            if "__orig_bases__" in cls.__dict__
            else cls.__bases__
        )
        for c in base_types:
            orig = get_origin(c) if not isinstance(c, type) else c
            if orig is not c and hasattr(orig, "__parameters__"):
                typevar_map = dict(zip(getattr(orig, "__parameters__"), get_args(c)))
            else:
                typevar_map = {}

            if not isinstance(orig, RecordMeta) or orig._is_record:
                continue

            if orig._template:
                for prop_name, prop_set in orig._defined_props.items():
                    prop_defs[prop_name] = prop_set._type
                    if prop_name not in cls.__dict__:
                        if isinstance(prop_set, Col):
                            prop_set = copy_and_override(
                                prop_set,
                                type(prop_set),
                                _type=copy_and_override(
                                    prop_set._type,
                                    type(prop_set._type),
                                    typevar_map=typevar_map,
                                    ctx=cls._src_mod,
                                ),
                            )
                        else:
                            prop_set = copy_and_override(
                                prop_set,
                                type(prop_set),
                                parent_type=prop_set.parent_type,
                                _type=copy_and_override(
                                    prop_set._type,
                                    type(prop_set._type),
                                    typevar_map=typevar_map,
                                    ctx=cls._src_mod,
                                ),
                            )
                        setattr(cls, prop_name, prop_set)
            else:
                assert orig is c
                cls._record_bases.append(orig)

        cls._base_pks = (
            {
                base: {
                    Col[Hashable, Any, Record](
                        _name=pk.name,
                        primary_key=True,
                        _type=PropType(Col[pk.value_type, Any, Record]),
                        record_set=cast(type[Record], cls)._db,
                    ): pk
                    for pk in base._primary_keys.values()
                }
                for base in cls._record_bases
            }
            if not cls._is_record
            else {}
        )

        cls._defined_props = {name: getattr(cls, name) for name in prop_defs.keys()}

        cls._defined_cols = {
            name: col
            for name, col in cls._defined_props.items()
            if isinstance(col, Col)
        }

        cls._defined_rels = {
            name: rel
            for name, rel in cls._defined_props.items()
            if isinstance(rel, RelSet)
        }

    @property
    def _defined_rel_cols(cls) -> dict[str, Col[Hashable]]:
        return {
            a.name: a
            for rel in cls._defined_rels.values()
            if rel.direct_rel is True
            for a in rel.fk_map.keys()
        }

    @property
    def _primary_keys(cls: type[Record]) -> dict[str, Col[Hashable]]:
        base_pk_cols = {v.name: v for vs in cls._base_pks.values() for v in vs.keys()}
        defined_pks = {
            name: a for name, a in cls._defined_cols.items() if a.primary_key
        }

        if len(base_pk_cols) > 0 and len(defined_pks) > 0:
            assert set(defined_pks.keys()).issubset(
                set(base_pk_cols.keys())
            ), f"Primary keys of {cls} are ambiguous. "
            return base_pk_cols

        return {**base_pk_cols, **defined_pks}

    @property
    def _props(cls) -> dict[str, Col | RelSet]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._props for c in cls._record_bases),
            cls._defined_props,
        )

    @property
    def _cols(cls) -> dict[str, Col]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._cols for c in cls._record_bases),
            cls._defined_cols,
        )

    @property
    def _rels(cls) -> dict[str, RelSet]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._rels for c in cls._record_bases),
            cls._defined_rels,
        )

    @property
    def _rel_types(cls) -> set[type[Record]]:
        return {rel.record_type for rel in cls._rels.values()}


class RecDB(Generic[RT]):
    """Descriptor to access the database of a record."""

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[RT] | None
    ) -> DB[RT, BaseIdx, RW, RT, RT]:
        if instance is None:
            return DB()

        if "__db" not in instance.__dict__:
            instance.__dict__["__db"] = DB[RT, BaseIdx, RW, RT, RT]()

        return instance.__dict__["__db"]

    def __set__(  # noqa: D105
        self,
        instance: object,
        owner: type | type[RT] | None,
        value: DB[RT, BaseIdx, RW, RT, RT],
    ) -> None:
        instance.__dict__["__db"] = value


@dataclass_transform(kw_only_default=True, field_specifiers=(prop,), eq_default=False)
class Record(Generic[KT], metaclass=RecordMeta):
    """Schema for a record in a database."""

    _template: ClassVar[bool]
    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID: UUIDType(binary=False),  # Binary type causes issues with DuckDB
    }
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    _db: RecDB[Self] = RecDB[Self]()
    _edge_dict: dict[str, RecordValue] = prop(default_factory=dict, local=True)

    _is_record: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)
        cls._is_record = False
        if "_template" not in cls.__dict__:
            cls._template = False
        if "_src_mod" not in cls.__dict__:
            cls._src_mod = getmodule(cls)

        cls.__dataclass_fields__ = {
            **{
                name: Field(
                    p.default,
                    p.default_factory,  # type: ignore[reportArgumentType]
                    p.init,
                    repr=True,
                    hash=p.primary_key,
                    metadata={},
                    compare=True,
                    kw_only=True,
                )
                for name, p in cls._cols.items()
            },
            **{
                name: Field(
                    p.default,
                    p.default_factory,  # type: ignore[reportArgumentType]
                    p.init,
                    repr=True,
                    hash=False,
                    metadata={},
                    compare=True,
                    kw_only=True,
                )
                for name, p in cls._rels.items()
            },
        }

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init__()
        for name, value in kwargs.items():
            setattr(self, name, value)

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        if cls._table_name is not None:
            return cls._table_name

        fqn_parts = PyObjectRef.reference(cls).fqn.split(".")

        name = fqn_parts[-1]
        for part in reversed(fqn_parts[:-1]):
            name = part + "_" + name
            if len(name) > 40:
                break

        return name

    @classmethod
    def _sql_table_name(
        cls,
        subs: Mapping[type[Record], sqla.TableClause],
    ) -> str:
        """Return a SQLAlchemy table object for this schema."""
        sub = subs.get(cls)
        return sub.name if sub is not None else cls._default_table_name()

    @classmethod
    def _sql_cols(cls, registry: orm.registry) -> dict[str, sqla.Column]:
        """Columns of this record type's table."""
        table_cols = {
            **cls._defined_cols,
            **cls._defined_rel_cols,
            **cls._primary_keys,
        }

        return {
            name: sqla.Column(
                col.name,
                registry._resolve_type(col.value_type),
                primary_key=col.prop.primary_key,
                autoincrement=False,
                index=col.prop.index,
            )
            for name, col in table_cols.items()
        }

    @classmethod
    def _foreign_keys(
        cls, metadata: sqla.MetaData, subs: Mapping[type[Record], sqla.TableClause]
    ) -> list[sqla.ForeignKeyConstraint]:
        """Foreign key constraints for this record type's table."""
        fks: list[sqla.ForeignKeyConstraint] = []

        for rel in cls._defined_rels.values():
            if len(rel.fk_map) > 0:
                rel_table = rel.record_type._table(metadata, subs)
                fks.append(
                    sqla.ForeignKeyConstraint(
                        [col.name for col in rel.fk_map.keys()],
                        [rel_table.c[col.name] for col in rel.fk_map.values()],
                        name=f"{cls._sql_table_name(subs)}_{rel.name}_fk",
                    )
                )

        for base, pks in cls._base_pks.items():
            base_table = base._table(metadata, subs)

            fks.append(
                sqla.ForeignKeyConstraint(
                    [col.name for col in pks.keys()],
                    [base_table.c[col.name] for col in pks.values()],
                    name=(
                        cls._sql_table_name(subs)
                        + "_base_fk_"
                        + gen_str_hash(base._sql_table_name(subs), 5)
                    ),
                )
            )

        return fks

    @classmethod
    def _table(
        cls,
        metadata: sqla.MetaData,
        subs: Mapping[type[Record], sqla.TableClause],
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        table_name = cls._sql_table_name(subs)

        if table_name in metadata.tables:
            # Return the table object from metadata if it already exists.
            # This is necessary to avoid circular dependencies.
            return metadata.tables[table_name]

        registry = orm.registry(metadata=metadata, type_annotation_map=cls._type_map)
        sub = subs.get(cls)

        cols = cls._sql_cols(registry)

        # Create a partial SQLAlchemy table object from the class definition
        # without foreign keys to avoid circular dependencies.
        # This adds the table to the metadata.
        sqla.Table(
            table_name,
            registry.metadata,
            *cols.values(),
            schema=(sub.schema if sub is not None else None),
        )

        fks = cls._foreign_keys(metadata, subs)

        # Re-create the table object with foreign keys and return it.
        return sqla.Table(
            table_name,
            registry.metadata,
            *cols.values(),
            *fks,
            schema=(sub.schema if sub is not None else None),
            extend_existing=True,
        )

    @classmethod
    def _joined_table(
        cls,
        metadata: sqla.MetaData,
        subs: Mapping[type[Record], sqla.TableClause],
    ) -> sqla.Table | sqla.Join:
        """Recursively join all bases of this record to get the full data."""
        table = cls._table(metadata, subs)

        base_joins = [
            (base._joined_table(metadata, subs), pk_map)
            for base, pk_map in cls._base_pks.items()
        ]
        for target_table, pk_map in base_joins:
            table = table.join(
                target_table,
                reduce(
                    sqla.and_,
                    (
                        table.c[pk.name] == target_table.c[target_pk.name]
                        for pk, target_pk in pk_map.items()
                    ),
                ),
            )

        return table

    @classmethod
    def _backrels_to_rels(
        cls, target: type[RT2]
    ) -> set[RelSet[Self, Any, SingleIdx, Any, Self, RT2]]:
        """Get all direct relations from a target record type to this type."""
        rels: set[RelSet[Self, Any, SingleIdx, Any, Self, RT2]] = set()
        for rel in cls._rels.values():
            if issubclass(target, rel.fk_record_type):
                rel = cast(RelSet[Self, Any, Any, Any, Self, RT2], rel)
                rels.add(
                    RelSet[Self, Any, SingleIdx, Any, Self, RT2](
                        on=rel.on,
                        _type=PropType(RelSet[cls, Any, Any, Any, Any, target]),
                        parent_type=cast(type[RT2], rel.fk_record_type),
                    )
                )

        return rels

    @classmethod
    def _rel(cls, other: type[RT2]) -> RelSet[RT2, Any, BaseIdx, Any, Self, Self]:
        """Dynamically define a relation to another record type."""
        return RelSet[RT2, Any, Any, Any, Self, Self](
            on=other,
            _type=PropType(RelSet[other, Any, BaseIdx, Any, cls, cls]),
            parent_type=cls,
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:  # noqa: D105
        assert cls._default_table_name() is not None
        return sqla.table(cls._default_table_name())

    @classmethod  # type: ignore[reportArgumentType]
    def _copy(
        cls: Callable[Params, RT2],
        rec: Record[KT],
        idx: KT,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> RT2:
        rec_type = type(rec)
        target_props = rec_type._props

        rec_props = {p: getattr(rec, p.name) for p in target_props.values() if p.init}
        rec_kwargs = {p.name: v for p, v in rec_props.items()}

        if len(rec_type._primary_keys) > 1:
            assert isinstance(idx, Iterable)
            new_idx = dict(zip(rec_type._primary_keys, idx))
        else:
            new_idx = {next(iter(rec_type._primary_keys)): idx}

        new_kwargs = {**rec_kwargs, **kwargs, **new_idx}

        return cls(*args, **new_kwargs)

    def __hash__(self) -> int:
        """Hash the record."""
        return gen_int_hash(self)

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)

    @overload
    def _to_dict(
        self,
        name_keys: Literal[False] = ...,
        only_loaded: bool = ...,
        load: bool = ...,
        include: set[type[Prop]] | None = ...,
        with_links: bool = ...,
    ) -> dict[Col | RelSet, Any]: ...

    @overload
    def _to_dict(
        self,
        name_keys: Literal[True],
        only_loaded: bool = ...,
        load: bool = ...,
        include: set[type[Prop]] | None = ...,
        with_links: Literal[False] = ...,
    ) -> dict[str, Any]: ...

    def _to_dict(
        self,
        name_keys: bool = False,
        only_loaded: bool = True,
        load: bool = False,
        include: set[type[Prop]] | None = None,
        with_links: bool = False,
    ) -> dict[Col | RelSet, Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Prop], ...] = (
            tuple(include) if include is not None else (Prop,)
        )

        def getter(r, n):
            return r.__dict__.get(n, State.undef) if not load else getattr(r, n)

        vals = {
            p if not name_keys else p.name: getter(self, p.name)
            for p in type(self)._props.values()
            if isinstance(p, include_types)
        }

        if with_links and issubclass(RelSet, include_types) and not name_keys:
            vals = {
                **vals,
                **{
                    cast(
                        RelSet[Any, Any, Any, Any, Any, Record, Any, None], r
                    ).link_set: self._edge_dict[r.name]
                    for r in type(self)._rels.values()
                    if issubclass(r.link_type, Record) and r.name in self._edge_dict
                },
            }

        return cast(
            dict,
            (
                vals
                if not only_loaded
                else {k: v for k, v in vals.items() if v is not State.undef}
            ),
        )

    @property
    def _index(self: Record[KT2]) -> KT2:
        """Return the index of the record."""
        pks = type(self)._primary_keys
        if len(pks) == 1:
            return getattr(self, next(iter(pks)))
        return cast(KT2, tuple(getattr(self, pk) for pk in pks))

    @classmethod
    def _from_partial_dict(
        cls,
        data: Mapping[Col | RelSet, Any] | Mapping[str, Any],
        loader: Callable[[Col | RelSet, KT], Any] | None = None,
        name_keys: bool = False,
    ) -> Self:
        """Return the index contained in a dict representation of this record."""
        args = {
            p.name: data.get(p if not name_keys else p.name, State.undef)  # type: ignore[reportArgumentType]
            for p in cls._props.values()
        }
        return cls(**args, _loader=loader)

    @property
    def _links(self) -> RecordLinks[Self]:
        """Descriptor to access the link records of a record's rels."""
        return RecordLinks(self)

    def __repr__(self) -> str:  # noqa: D105
        return self._to_dict(name_keys=True, only_loaded=False).__repr__()


@dataclass
class RelTree(Generic[*RTT]):
    """Tree of relations starting from the same root."""

    rels: Iterable[RelSet[Record, Any, Any, Any]] = field(default_factory=set)

    def __post_init__(self) -> None:  # noqa: D105
        assert all(
            rel.root_type == self.root for rel in self.rels
        ), "Relations in set must all start from same root."
        self.targets = [rel.record_type for rel in self.rels]

    @cached_property
    def root(self) -> type[Record]:
        """Root record type of the set."""
        return list(self.rels)[-1].root_type

    @cached_property
    def dict(self) -> DataDict:
        """Tree representation of the relation set."""
        tree: DataDict = {}

        for rel in self.rels:
            subtree = tree
            if len(rel._rel_path) > 1:
                for ref in rel._rel_path[1:]:
                    if ref not in subtree:
                        subtree[ref] = {}
                    subtree = subtree[ref]

        return tree

    def prefix(
        self, prefix: type[Record] | RelSet[Any, Any, Any, Any, Any, Any, Any]
    ) -> Self:
        """Prefix all relations in the set with given relation."""
        rels = {rel.prefix(prefix) for rel in self.rels}
        return cast(Self, RelTree(rels))

    def __mul__(self, other: RelSet[RT] | RelTree) -> RelTree[*RTT, RT]:
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    def __rmul__(self, other: RelSet[RT] | RelTree) -> RelTree[RT, *RTT]:
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*other.rels, *self.rels])

    def __or__(
        self: RelTree[RT2], other: RelSet[RT3] | RelTree[RT3]
    ) -> RelTree[RT2 | RT3]:
        """Add more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    @property
    def types(self) -> tuple[*RTT]:
        """Return the record types in the relation tree."""
        return cast(tuple[*RTT], tuple(r.record_type for r in self.rels))


type AggMap[Rec: Record] = dict[Col[Rec, Any], Col | sqla.Function]


@dataclass(kw_only=True, frozen=True)
class Agg(Generic[RT]):
    """Define an aggregation map."""

    target: type[RT]
    map: AggMap[RT]


type Join = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


@dataclass(kw_only=True, eq=False)
class DB(
    Generic[RT, IT, WT, BT, RTS, TT],
):
    """Record dataset."""

    backend: BT = None
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

    overlay: str | None = None
    subs: dict[type[Record], sqla.TableClause] = field(default_factory=dict)

    validate_on_init: bool = True
    create_cross_fk: bool = True
    overlay_with_schemas: bool = True

    keys: Sequence[slice | list[Hashable] | Hashable] = field(default_factory=list)
    filters: list[sqla.ColumnElement[bool]] = field(default_factory=list)
    merges: RelTree = field(default_factory=RelTree)

    _record_types: Mapping[
        type[Record], Literal[True] | Require | str | sqla.TableClause
    ] = field(default_factory=dict)
    _metadata: sqla.MetaData = field(default_factory=sqla.MetaData)

    def __post_init__(self):  # noqa: D105
        if self.types is not None:
            types = self.types
            if isinstance(types, set):
                types = cast(
                    dict[type[Record], Literal[True]], {rec: True for rec in self.types}
                )

            self.subs = {
                **self.subs,
                **{
                    rec: (sub if isinstance(sub, sqla.TableClause) else sqla.table(sub))
                    for rec, sub in types.items()
                    if not isinstance(sub, Require) and sub is not True
                },
            }
            self._record_types = {**self._record_types, **types}

        if self.schema is not None:
            if isinstance(self.schema, Mapping):
                self.subs = {
                    **self.subs,
                    **{
                        rec: sqla.table(rec._default_table_name(), schema=schema_name)
                        for schema, schema_name in self.schema.items()
                        for rec in schema._record_types
                    },
                }
                schemas = self.schema
            else:
                schemas = {self.schema: True}

            self._record_types = cast(
                dict,
                {
                    **self._record_types,
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

        if self.overlay is not None and self.overlay_with_schemas:
            self._ensure_schema_exists(self.overlay)

    @cached_property
    def record_type(self) -> type[RT]:
        """Base record type."""
        return get_lowest_common_base(self._record_types.keys())

    @cached_property
    def backend_name(self) -> str:
        """Unique name of backend."""
        return self.backend if isinstance(self.backend, str) else token_hex(5)

    @cached_property
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

    @cached_property
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
            else (sqla.create_engine(f"duckdb:///:memory:{self.backend_name}"))
        )

    @cached_property
    def assoc_types(self) -> set[type[Record]]:
        """Set of all association tables in this DB."""
        assoc_types = set()
        for rec in self._record_types:
            pks = set([col.name for col in rec._primary_keys.values()])
            fks = set(
                [col.name for rel in rec._rels.values() for col in rel.fk_map.keys()]
            )
            if pks == fks:
                assoc_types.add(rec)

        return assoc_types

    @cached_property
    def relation_map(self) -> dict[type[Record], set[RelSet]]:
        """Maps all tables in this DB to their outgoing or incoming relations."""
        rels: dict[type[Record], set[RelSet]] = {
            table: set() for table in self._record_types
        }

        for rec in self._record_types:
            for rel in rec._rels.values():
                rels[rec].add(rel)
                rels[rel.record_type].add(rel)

        return rels

    def describe(self) -> dict[str, str | dict[str, str] | None]:
        """Return a description of this database.

        Returns:
            Mapping of table names to table descriptions.
        """
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
        }

    def to_graph(
        self: DB[Any, Any, Any, Any, Any, None], nodes: Sequence[type[Record]]
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
            .assign(table=n.record_type._default_table_name())
            for n in node_tables
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )

        directed_edges = reduce(set.union, (self.relation_map[n] for n in nodes))

        undirected_edges: dict[type[Record], set[tuple[RelSet, ...]]] = {
            t: set() for t in nodes
        }
        for n in nodes:
            for at in self.assoc_types:
                if len(at._rels) == 2:
                    left, right = (r for r in at._rels.values())
                    assert left is not None and right is not None
                    if left.record_type == n:
                        undirected_edges[n].add((left, right))
                    elif right.record_type == n:
                        undirected_edges[n].add((right, left))

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                *[
                    node_df.loc[
                        node_df["table"]
                        == str((rel.parent_type or Record)._default_table_name())
                    ]
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(rel.record_type._default_table_name())
                        ],
                        left_on=[c.name for c in rel.fk_map.keys()],
                        right_on=[c.name for c in rel.fk_map.values()],
                    )
                    .rename(columns={"node_id": "target"})[["source", "target"]]
                    .assign(ltr=",".join(c.name for c in rel.fk_map.keys()), rtl=None)
                    for rel in directed_edges
                ],
                *[
                    self[assoc_table]
                    .to_df(kind=pd.DataFrame)
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(left_rel.record_type._default_table_name())
                        ].dropna(axis="columns", how="all"),
                        left_on=[c.name for c in left_rel.fk_map.keys()],
                        right_on=[c.name for c in left_rel.fk_map.values()],
                        how="inner",
                    )
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(left_rel.record_type._default_table_name())
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
                                *(a for a in (left_rel.parent_type or Record)._cols),
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

    def validate(self) -> None:
        """Perform pre-defined schema validations."""
        types = {}

        if self.types is not None:
            types |= {
                rec: (isinstance(req, Require) and req.present) or req is True
                for rec, req in self._record_types.items()
            }

        if isinstance(self.schema, Mapping):
            types |= {
                rec: isinstance(req, Require) and req.present
                for schema, req in self.schema.items()
                for rec in schema._record_types
            }

        tables = {self._get_table(rec): required for rec, required in types.items()}

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

    def to_db(
        self: DB[Any, Any, Any, Any, Any, None],
        data: pd.DataFrame | pl.DataFrame | sqla.Select,
        fks: Mapping[str, Col] | None = None,
    ) -> DB[DynRecord, Any, RO, BT]:
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, pd.DataFrame | pl.DataFrame)
            else f"temp_{token_hex(5)}"
        )

        rec = dynamic_record_type(name, props=props_from_data(data, fks))
        ds = DB[DynRecord, BaseIdx, RW, BT, DynRecord](
            types={rec}, backend=self.backend, url=self.url
        )

        ds &= data

        return ds

    def _get_table(self, rec: type[Record], writable: bool = False) -> sqla.Table:
        if writable and self.overlay is not None and rec not in self.subs:
            # Create an empty overlay table for the record type
            self.subs[rec] = sqla.table(
                (
                    (self.overlay + "_" + rec._default_table_name())
                    if self.overlay_with_schemas
                    else rec._default_table_name()
                ),
                schema=self.overlay if self.overlay_with_schemas else None,
            )

        table_name = rec._sql_table_name(self.subs)

        if table_name in self._metadata.tables:
            return self._metadata.tables[table_name]

        new_metadata = sqla.MetaData()
        for t in self._metadata.tables.values():
            t.to_metadata(new_metadata)

        table = rec._table(new_metadata, self.subs)

        # Create any missing tables in the database.
        if self.backend is not None:
            new_tables = set(new_metadata.tables) - set(self._metadata.tables)
            if len(new_tables) > 0:
                new_metadata.create_all(self.engine, checkfirst=True)

        return table

    def _get_joined_table(self, rec: type[Record]) -> sqla.Table | sqla.Join:
        new_metadata = sqla.MetaData()
        for t in self._metadata.tables.values():
            t.to_metadata(new_metadata)

        table = rec._joined_table(new_metadata, self.subs)

        # Create any missing tables in the database.
        if self.backend is not None:
            new_tables = set(new_metadata.tables) - set(self._metadata.tables)
            if len(new_tables) > 0:
                new_metadata.create_all(self.engine, checkfirst=True)

        return table

    def _get_alias(self, rel: RelSet[Any, Any, Any, Any, Any, Any]) -> sqla.FromClause:
        """Get alias for a relation reference."""
        return self._get_joined_table(rel.record_type).alias(gen_str_hash(rel, 8))

    def _get_random_alias(self, rec: type[Record]) -> sqla.FromClause:
        """Get random alias for a type."""
        return self._get_joined_table(rec).alias(token_hex(4))

    def _replace_col(
        self,
        element: sqla_visitors.ExternallyTraversible,
        reflist: set[RelSet[Any, Any, Any, Any, Any]] = set(),
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if isinstance(element, Col):
            if isinstance(element.record_set, RelSet):
                reflist.add(element.record_set)

            if isinstance(self, RelSet) and isinstance(element.record_set, RelSet):
                element.record_set = element.record_set.prefix(self)

            table = (
                self._get_alias(element.record_set)
                if isinstance(element.record_set, RelSet)
                else self._get_table(element.record_type)
            )
            return table.c[element.name]

        return None

    def _parse_filter(
        self,
        key: sqla.ColumnElement[bool],
    ) -> tuple[sqla.ColumnElement[bool], RelTree]:
        """Parse filter argument and return SQL expression and join operations."""
        reflist: set[RelSet] = set()
        replace_func = partial(self._replace_col, reflist=reflist)
        filt = sqla_visitors.replacement_traverse(key, {}, replace=replace_func)
        merge = RelTree(reflist)

        return filt, merge

    def _parse_schema_items(
        self,
        element: sqla_visitors.ExternallyTraversible,
        **kw: Any,
    ) -> sqla.ColumnElement | sqla.FromClause | None:
        if isinstance(element, RelSet):
            return self._get_alias(element)
        elif isinstance(element, Col):
            table = (
                self._get_alias(element.record_set)
                if isinstance(element.record_set, RelSet)
                else self._get_table(element.parent_type)
            )
            return table.c[element.name]
        elif has_type(element, type[Record]):
            return self._get_table(element)

        return None

    def _parse_expr[CE: sqla.ClauseElement](self, expr: CE) -> CE:
        """Parse an expression in this database's context."""
        return cast(
            CE,
            sqla_visitors.replacement_traverse(
                expr, {}, replace=self._parse_schema_items
            ),
        )

    def _ensure_schema_exists(self, schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_schema(schema_name):
            with self.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name

    def _table_exists(self, sqla_table: sqla.Table) -> bool:
        """Check if a table exists in the database."""
        return sqla.inspect(self.engine).has_table(
            sqla_table.name, schema=sqla_table.schema
        )

    def _create_sqla_table(self, sqla_table: sqla.Table) -> None:
        """Create SQL-side table from Table class."""
        if not self.create_cross_fk:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            _remove_external_fk(sqla_table)

        sqla_table.create(self.engine)

    def _load_from_excel(self, record_types: list[type[Record]] | None = None) -> None:
        """Load all tables from Excel."""
        assert self.backend is not None
        assert self.backend_type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.url, Path | CloudPath | HttpFile)

        path = self.url.get() if isinstance(self.url, HttpFile) else self.url

        with open(path, "rb") as file:
            for rec in record_types or self._record_types:
                pl.read_excel(
                    file, sheet_name=rec._default_table_name()
                ).write_database(str(self._get_table(rec)), str(self.engine.url))

    def _save_to_excel(
        self, record_types: Iterable[type[Record]] | None = None
    ) -> None:
        """Save all (or selected) tables to Excel."""
        assert self.backend is not None
        assert self.backend_type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.url, Path | CloudPath | HttpFile)

        file = BytesIO() if isinstance(self.url, HttpFile) else self.url.open("wb")

        with ExcelWorkbook(file) as wb:
            for rec in record_types or self._record_types:
                pl.read_database(
                    f"SELECT * FROM {self._get_table(rec)}",
                    self.engine,
                ).write_excel(wb, worksheet=rec._default_table_name())

        if isinstance(self.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.url.set(file)

    def _delete_from_excel(self, record_types: Iterable[type[Record]]) -> None:
        """Delete selected table from Excel."""
        assert self.backend is not None
        assert self.backend_type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.url, Path | CloudPath | HttpFile)

        file = BytesIO() if isinstance(self.url, HttpFile) else self.url.open("wb")

        wb = openpyxl.load_workbook(file)
        for rec in record_types or self._record_types:
            del wb[rec._default_table_name()]

        if isinstance(self.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.url.set(file)

    @cached_property
    def _idx_cols(self: DB[Record, Any, Any, BT]) -> list[sqla.ColumnElement]:
        """Return the index columns."""
        return [
            *(
                col.label(f"{self.record_type._default_table_name()}.{col_name}")
                for col_name, col in self.root_table.columns.items()
                if self.record_type._cols[col_name] in self.idx
            ),
            *(
                col.label(f"{rel.path_str}.{col_name}")
                for rel in self.merges.rels
                for col_name, col in self._get_alias(rel).columns.items()
                if rel.record_type._cols[col_name] in self.idx
            ),
        ]

    @cached_property
    def idx(self) -> set[Col]:
        """Return the index cols."""
        path_idx = None
        if isinstance(self, RelSet):
            path_idx = self.path_idx
        elif isinstance(self, Col) and isinstance(self.record_set, RelSet):
            path_idx = self.record_set.path_idx

        if path_idx is not None:
            return {p for p in path_idx}

        if isinstance(self, DB):
            return set(self.record_type._primary_keys.values())

        return set()

    @cached_property
    def single_key(self) -> Hashable:
        """Return whether the set is a single record."""
        single_keys = [k for k in self.keys if not isinstance(k, list | slice)]
        if len(single_keys) > 0:
            return single_keys[0]

    @cached_property
    def root_table(self) -> sqla.FromClause:
        """Get the main table for the current selection."""
        return self._get_random_alias(self.record_type)

    def suffix(
        self, right: RelSet[RT2, Any, Any, Any, Any, Any, Any]
    ) -> RelSet[RT2, Record, Any, WT, BT, Record, Record]:
        """Prefix this prop with a relation or record type."""
        rel_path = right._rel_path[1:] if len(right._rel_path) > 1 else (right,)

        prefixed_rel = reduce(
            lambda r1, r2: copy_and_override(
                r2,
                RelSet,
                parent_type=r1.rec,
                backend=r1.backend,
                url=r1.url,
                keys=[*r2.keys, *r1.keys],
                filters=[*r2.filters, *r1.filters],
                merges=r2.merges * r1.merges,
            ),
            rel_path,
            cast(RelSet, self),
        )

        return cast(
            RelSet[RT2, Record, Any, WT, BT, Any, Record],
            prefixed_rel,
        )

    @cached_property
    def rec(self) -> type[RT]:
        """Reference props of the target record type."""
        return cast(
            type[RT],
            type(
                self.record_type.__name__ + "_" + token_hex(5),
                (self.record_type,),
                {"_rel": self},
            ),
        )

    # Overloads: attribute selection:

    # 1. DB-level type selection
    @overload
    def __getitem__(
        self: DB[Any, Any, Any, Any],
        key: type[RT2],
    ) -> DB[RT2, IT, WT, BT, RT2]: ...

    # 2. Top-level attribute selection, custom index
    @overload
    def __getitem__(  # type: ignore[reportOverlappingOverload]
        self: DB[RT2, KT2 | Filt[KT2], Any, Any],
        key: Col[VT2, Any, RT2],
    ) -> Col[VT2, WT, KT, BT, RT2]: ...

    # 3. Top-level attribute selection, base index
    @overload
    def __getitem__(
        self: DB[Record[KT2], Any, Any, Any, RT2],
        key: Col[VT2, Any, RT2],
    ) -> Col[VT2, WT, KT2, BT, RT2]: ...

    # 4. Nested attribute selection, custom index
    @overload
    def __getitem__(
        self: DB[Any, KT2 | Filt[KT2], Any, Any],
        key: Col[VT2, Any, Any],
    ) -> Col[VT2, WT, KT2 | IdxStart[KT2], BT, Record]: ...

    # 5. Nested attribute selection, base index
    @overload
    def __getitem__(
        self: DB[Record[KT2], Any, Any, Any, Any],
        key: Col[VT2, Any, Any],
    ) -> Col[VT2, WT, KT2 | IdxStart[KT2], BT, Record]: ...

    # Overloads: relation selection:

    # 6. Top-level relation selection, singular, base index
    @overload
    def __getitem__(  # type: ignore[reportOverlappingOverload]
        self: DB[Record[KT2], BaseIdx | Filt[BaseIdx], Any, Any, RT2],
        key: RelSet[RT3, LT2, SingleIdx | None, Any, Record, RT2],
    ) -> RelSet[RT3, LT2, KT2, WT, BT, RT2, RT3, None]: ...

    # 7. Top-level relation selection, singular, single index
    @overload
    def __getitem__(
        self: DB[RT2, SingleIdx, Any, Any],
        key: RelSet[RT3, LT2, SingleIdx, Any, Record, RT2],
    ) -> RelSet[RT3, LT2, SingleIdx, WT, BT, RT2, RT3, None]: ...

    # 8. Top-level relation selection, singular nullable, single index
    @overload
    def __getitem__(
        self: DB[RT2, SingleIdx, Any, Any],
        key: RelSet[RT3, LT2, SingleIdx | None, Any, Record, RT2],
    ) -> RelSet[RT3 | Scalar[None], LT2, SingleIdx, WT, BT, RT2, RT3, None]: ...

    # 9. Top-level relation selection, singular, custom index
    @overload
    def __getitem__(
        self: DB[RT2, KT2 | Filt[KT2], Any, Any],
        key: RelSet[RT3, LT2, SingleIdx | None, Any, Record, RT2],
    ) -> RelSet[RT3, LT2, KT2, WT, BT, RT2, RT3, None]: ...

    # 10. Top-level relation selection, base plural, base index
    @overload
    def __getitem__(
        self: DB[Record[KT2], BaseIdx | Filt[BaseIdx], Any, Any, RT2],
        key: RelSet[
            RT3,
            LT2,
            BaseIdx | Filt[BaseIdx],
            Any,
            Record,
            RT2,
            Record[KT3],
        ],
    ) -> RelSet[RT3, LT2, tuple[KT2, KT3], WT, BT, RT2, RT3, None]: ...

    # 11. Top-level relation selection, base plural, single index
    @overload
    def __getitem__(  # type: ignore[reportOverlappingOverload]
        self: DB[RT2, SingleIdx, Any, Any],
        key: RelSet[
            RT3,
            LT2,
            BaseIdx | Filt[BaseIdx],
            Any,
            Record,
            RT2,
            Record[KT3],
        ],
    ) -> RelSet[RT3, LT2, KT3, WT, BT, RT2, RT3, None]: ...

    # 12. Top-level relation selection, base plural, tuple index
    @overload
    def __getitem__(
        self: DB[Record[KT2], tuple[*ITT] | Filt[tuple[*ITT]], Any, Any, RT2],
        key: RelSet[
            RT3,
            LT2,
            BaseIdx | Filt[BaseIdx],
            Any,
            Record,
            RT2,
            Record[KT3],
        ],
    ) -> RelSet[
        RT3,
        LT2,
        tuple[*ITT, KT3] | tuple[KT2, KT3],
        WT,
        BT,
        RT2,
        RT3,
        None,
    ]: ...

    # 13. Top-level relation selection, base plural, custom index
    @overload
    def __getitem__(
        self: DB[Any, KT2 | Filt[KT2], Any, Any, RT2],
        key: RelSet[
            RT3,
            LT2,
            BaseIdx | Filt[BaseIdx],
            Any,
            Record,
            RT2,
            Record[KT3],
        ],
    ) -> RelSet[RT3, LT2, tuple[KT2, KT3], WT, BT, RT2, RT3, None]: ...

    # 14. Top-level relation selection, plural, base index
    @overload
    def __getitem__(
        self: DB[Record[KT2], BaseIdx | Filt[BaseIdx], Any, Any, RT2],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, RT2],
    ) -> RelSet[RT3, LT2, tuple[KT2, KT3], WT, BT, RT2, RT3, None]: ...

    # 15. Top-level relation selection, plural, single index
    @overload
    def __getitem__(  # type: ignore[reportOverlappingOverload]
        self: DB[RT2, SingleIdx],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, RT2],
    ) -> RelSet[RT3, LT2, KT3, WT, BT, RT2, RT3, None]: ...

    # 16. Top-level relation selection, plural, tuple index
    @overload
    def __getitem__(
        self: DB[Record[KT2], tuple[*ITT] | Filt[tuple[*ITT]], Any, Any, RT2],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, RT2],
    ) -> RelSet[
        RT3,
        LT2,
        tuple[*ITT, KT3] | tuple[KT2, KT3],
        WT,
        BT,
        RT2,
        RT3,
        None,
    ]: ...

    # 17. Top-level relation selection, plural, custom index
    @overload
    def __getitem__(
        self: DB[RT2, KT2 | Filt[KT2], Any, Any],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, RT2],
    ) -> RelSet[RT3, LT2, tuple[KT2, KT3], WT, BT, RT2, RT3, None]: ...

    # 18. Nested relation selection, singular, base index
    @overload
    def __getitem__(
        self: DB[Record[KT2], BaseIdx | Filt[BaseIdx], Any, Any, Any],
        key: RelSet[RT3, LT2, SingleIdx, Any, Record, PT2],
    ) -> RelSet[RT3, LT2, IdxStart[KT2], WT, BT, PT2, RT3, None]: ...

    # 19. Nested relation selection, singular, single index
    @overload
    def __getitem__(
        self: DB[Any, SingleIdx, Any, Any, Any],
        key: RelSet[RT3, LT2, SingleIdx, Any, Record, PT2],
    ) -> RelSet[RT3, LT2, Hashable | SingleIdx, WT, BT, PT2, RT3, None]: ...

    # 20. Nested relation selection, singular, tuple index
    @overload
    def __getitem__(
        self: DB[Any, tuple[*ITT] | Filt[tuple[*ITT]], Any, Any, Any],
        key: RelSet[RT3, LT2, SingleIdx, Any, Record, PT2],
    ) -> RelSet[RT3, LT2, IdxTupStart[*ITT], WT, BT, PT2, RT3, None]: ...

    # 21. Nested relation selection, singular, custom index
    @overload
    def __getitem__(
        self: DB[Any, KT2 | Filt[KT2], Any, Any, Any],
        key: RelSet[RT3, LT2, SingleIdx, Any, Record, PT2],
    ) -> RelSet[RT3, LT2, IdxStart[KT2], WT, BT, PT2, RT3, None]: ...

    # 22. Nested relation selection, base plural, base index
    @overload
    def __getitem__(
        self: DB[Record[KT2], BaseIdx | Filt[BaseIdx], Any, Any, Any],
        key: RelSet[RT3, LT2, BaseIdx | Filt[BaseIdx], Any, Record, PT2, Record[KT3]],
    ) -> RelSet[RT3, LT2, IdxStartEnd[KT2, KT3], WT, BT, PT2, RT3, None]: ...

    # 23. Nested relation selection, base plural, single index
    @overload
    def __getitem__(
        self: DB[Any, SingleIdx, Any, Any, Any],
        key: RelSet[RT3, LT2, BaseIdx | Filt[BaseIdx], Any, Record, PT2, Record[KT3]],
    ) -> RelSet[RT3, LT2, IdxEnd[KT3], WT, BT, PT2, RT3, None]: ...

    # 24. Nested relation selection, base plural, tuple index
    @overload
    def __getitem__(
        self: DB[Any, tuple[*ITT] | Filt[tuple[*ITT]], Any, Any, Any],
        key: RelSet[RT3, LT2, BaseIdx | Filt[BaseIdx], Any, Record, PT2, Record[KT3]],
    ) -> RelSet[RT3, LT2, IdxTupStartEnd[*ITT, KT3], WT, BT, PT2, RT3, None]: ...

    # 25. Nested relation selection, base plural, custom index
    @overload
    def __getitem__(
        self: DB[Any, KT2 | Filt[KT2], Any, Any, Any],
        key: RelSet[RT3, LT2, BaseIdx | Filt[BaseIdx], Any, Record, PT2, Record[KT3]],
    ) -> RelSet[RT3, LT2, IdxStartEnd[KT2, KT3], WT, BT, PT2, RT3, None]: ...

    # 26. Nested relation selection, plural, base index
    @overload
    def __getitem__(
        self: DB[Record[KT2], BaseIdx | Filt[BaseIdx], Any, Any, Any],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, PT2],
    ) -> RelSet[RT3, LT2, IdxStartEnd[KT2, KT3], WT, BT, PT2, RT3, None]: ...

    # 27. Nested relation selection, plural, single index
    @overload
    def __getitem__(
        self: DB[Any, SingleIdx],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, PT2],
    ) -> RelSet[RT3, LT2, IdxEnd[KT3], WT, BT, PT2, RT3, None]: ...

    # 28. Nested relation selection, plural, tuple index
    @overload
    def __getitem__(
        self: DB[Any, tuple[*ITT] | Filt[tuple[*ITT]]],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, PT2],
    ) -> RelSet[RT3, LT2, IdxTupStartEnd[*ITT, KT3], WT, BT, PT2, RT3, None]: ...

    # 29. Nested relation selection, plural, custom index
    @overload
    def __getitem__(
        self: DB[Any, KT2 | Filt[KT2]],
        key: RelSet[RT3, LT2, KT3 | Filt[KT3], Any, Record, PT2],
    ) -> RelSet[RT3, LT2, IdxStartEnd[KT2, KT3], WT, BT, PT2, RT3, None]: ...

    # 30. Default relation selection
    @overload
    def __getitem__(
        self: DB[Any, Any, Any, Any, Any],
        key: RelSet[RT3, LT2, Any, Any, Record, PT2],
    ) -> RelSet[RT3, LT2, Any, WT, BT, PT2, RT3, None]: ...

    # 31. RelSet: Merge selection, single index
    @overload
    def __getitem__(
        self: RelSet[RT2, LT2, SingleIdx, Any, Any, PT2, Any],
        key: RelTree[RT2, *RTT],
    ) -> RelSet[
        RT,
        LT2,
        BaseIdx,
        WT,
        BT,
        PT2,
        RT,
        tuple[RT, *RTT],
    ]: ...

    # 32. RelSet: Merge selection, default
    @overload
    def __getitem__(
        self: RelSet[RT2, LT2, Any, Any, Any, PT2, Any],
        key: RelTree[RT2, *RTT],
    ) -> RelSet[
        RT,
        LT2,
        IT,
        WT,
        BT,
        PT2,
        RT,
        tuple[RT, *RTT],
    ]: ...

    # 33. RelSet: Expression filtering, keep index
    @overload
    def __getitem__(
        self: RelSet[Any, LT2, MT2 | Filt[MT2], Any, Any, PT2],
        key: sqla.ColumnElement[bool],
    ) -> RelSet[RT, LT2, Filt[MT2], WT, BT, PT2, RT]: ...

    # 34. RelSet: List selection
    @overload
    def __getitem__(
        self: RelSet[Record[KT2], LT2, MT2 | Filt[MT2], Any, Any, PT2],
        key: Iterable[KT2 | MT2],
    ) -> RelSet[RT, LT2, Filt[MT2], WT, BT, PT2, RT]: ...

    # 35. RelSet: Slice selection
    @overload
    def __getitem__(
        self: RelSet[Any, LT2, MT2 | Filt[MT2], Any, Any, PT2],
        key: slice | tuple[slice, ...],
    ) -> RelSet[RT, LT2, Filt[MT2], WT, BT, PT2, RT]: ...

    # 36. RelSet: Index value selection, static
    @overload
    def __getitem__(
        self: RelSet[Record[KT2], LT2, KT3 | Index, Any, Record, PT2],
        key: KT2 | KT3,
    ) -> RelSet[RT, LT2, SingleIdx, WT, BT, PT2, RT]: ...

    # 37. RelSet: Index value selection
    @overload
    def __getitem__(
        self: RelSet[Record[KT2], LT2, KT3 | Index, Any, Backend, PT2],
        key: KT2 | KT3,
    ) -> RT: ...

    # 38. Merge selection, single index
    @overload
    def __getitem__(
        self: DB[Any, SingleIdx, Any, Any],
        key: RelTree[RT, *RTT],
    ) -> DB[RT, BaseIdx, WT, BT, RT, tuple[RT, *RTT]]: ...

    # 39. Merge selection, default
    @overload
    def __getitem__(
        self: DB[Any, Any, Any, Any],
        key: RelTree[RT, *RTT],
    ) -> DB[RT, IT, WT, BT, RT, tuple[RT, *RTT]]: ...

    # 40. Expression filtering, keep index
    @overload
    def __getitem__(
        self: DB[Any, MT2 | Filt[MT2], Any, Any], key: sqla.ColumnElement[bool]
    ) -> DB[RT, Filt[MT2], WT, BT]: ...

    # 41. List selection
    @overload
    def __getitem__(
        self: DB[Record[KT2], MT2 | Filt[MT2], Any, Any], key: Iterable[KT2 | MT2]
    ) -> DB[RT, Filt[MT2], WT, BT]: ...

    # 42. Slice selection
    @overload
    def __getitem__(
        self: DB[Any, MT2 | Filt[MT2], Any, Any], key: slice | tuple[slice, ...]
    ) -> DB[RT, Filt[MT2], WT, BT]: ...

    # 43. Index value selection, static
    @overload
    def __getitem__(
        self: DB[Record[KT2], KT3 | Index, Any, Record], key: KT2 | KT3
    ) -> DB[RT, SingleIdx, WT, BT]: ...

    # 44. Index value selection
    @overload
    def __getitem__(
        self: DB[Record[KT2], KT3 | Index, Any, Backend], key: KT2 | KT3
    ) -> RT: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self: DB[Record, Any, Any, Any, Any],
        key: (
            type[Record]
            | Col
            | DB
            | RelTree
            | sqla.ColumnElement[bool]
            | list[Hashable]
            | slice
            | tuple[slice, ...]
            | Hashable
        ),
    ) -> Col[Any, Any, Any, Any, Any] | DB[Record, Any, Any, Any, Any, Any] | Record:
        match key:
            case type():
                assert issubclass(
                    key,
                    self.record_type,
                )
                return copy_and_override(self, DB, types={key})
            case Col():
                if not issubclass(self.record_type, key.parent_type):
                    assert isinstance(key.record_set, RelSet)
                    return self.suffix(key.record_set)[key]

                return Col(
                    record_set=self,
                    _type=key._type,
                )
            case RelSet():
                return self.suffix(key)
            case RelTree():
                return copy_and_override(self, type(self), merges=self.merges * key)
            case sqla.ColumnElement():
                return copy_and_override(self, type(self), filters=[*self.filters, key])
            case list() | slice() | tuple() | Hashable():
                if not isinstance(key, list | slice) and not has_type(
                    key, tuple[slice, ...]
                ):
                    assert (
                        self.single_key is None or key == self.single_key
                    ), "Cannot select multiple single record keys"

                if isinstance(self.backend, type):
                    return copy_and_override(self, type(self), keys=[*self.keys, key])

                return list(iter(self))[0]

    def select(
        self,
        *,
        index_only: bool = False,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        selection_table = self.root_table
        assert selection_table is not None

        select = sqla.select(
            *(
                col.label(f"{self.record_type._default_table_name()}.{col_name}")
                for col_name, col in selection_table.columns.items()
                if not index_only or self.record_type._cols[col_name] in self.idx
            ),
            *(
                col.label(f"{rel.path_str}.{col_name}")
                for rel in self.merges.rels
                for col_name, col in self._get_alias(rel).columns.items()
                if not index_only or rel.record_type._cols[col_name] in self.idx
            ),
        ).select_from(selection_table)

        for join in self._joins():
            select = select.join(*join)

        for filt in self.filters:
            select = select.where(filt)

        return select

    @overload
    def to_df(
        self: DB[Record, Any, Any, Any, Any, tuple[*RTT]],
        kind: type[DT],
    ) -> tuple[DT, ...]: ...

    @overload
    def to_df(self: DB[Record, Any, Any, Any, Any, None], kind: type[DT]) -> DT: ...

    @overload
    def to_df(
        self: DB[Record, Any, Any, Any, Any, tuple[*RTT]],
        kind: None = ...,
    ) -> tuple[pl.DataFrame, ...]: ...

    @overload
    def to_df(
        self: DB[Record, Any, Any, Any, Any, None], kind: None = ...
    ) -> pl.DataFrame: ...

    @overload
    def to_df(
        self: DB[Record, Any, Any, Any, Any, Any], kind: None = ...
    ) -> pl.DataFrame | tuple[pl.DataFrame, ...]: ...

    def to_df(
        self: DB[Record, Any, Any, Any, Any, Any],
        kind: type[DT] | None = None,
    ) -> DT | tuple[DT, ...]:
        """Download selection."""
        select = self.select()

        idx_cols = [
            f"{rel.path_str}.{pk}"
            for rel in self.merges.rels
            for pk in rel.record_type._primary_keys
        ]

        main_cols = {
            col: col.lstrip(self.record_type._default_table_name() + ".")
            for col in select.columns.keys()
            if col.startswith(self.record_type._default_table_name())
        }

        extra_cols = {
            rel: {
                col: col.lstrip(rel.path_str + ".")
                for col in select.columns.keys()
                if col.startswith(rel.path_str)
            }
            for rel in self.merges.rels
        }

        merged_df = None
        if kind is pd.DataFrame:
            with self.engine.connect() as con:
                merged_df = pd.read_sql(select, con)
                merged_df = merged_df.set_index(idx_cols)
        else:
            merged_df = pl.read_database(str(select.compile(self.engine)), self.engine)

        main_df, *extra_dfs = cast(
            tuple[DT, ...],
            (
                merged_df[list(main_cols.keys())].rename(main_cols),
                *(
                    merged_df[list(cols.keys())].rename(cols)
                    for cols in extra_cols.values()
                ),
            ),
        )

        return main_df, *extra_dfs

    @overload
    def __iter__(self: DB[Any, Any, Any, Any, Any, None]) -> Iterator[RT]: ...

    @overload
    def __iter__(
        self: DB[Any, Any, Any, Any, Any, tuple]
    ) -> Iterator[tuple[RT, *tuple[Record, ...]]]: ...

    def __iter__(self) -> Iterator[RT | tuple[RT, *tuple[Record, ...]]]:  # noqa: D105
        dfs = self.to_df()
        if isinstance(dfs, pl.DataFrame):
            dfs = (dfs,)

        rec_types: tuple[type[Record], ...] = (self.record_type, *self.merges.types)
        idx_cols = [
            f"{rel.path_str}.{pk}"
            for rel in self.merges.rels
            for pk in rel.record_type._primary_keys
        ]

        loaded: dict[type[Record], dict[Hashable, Record]] = {r: {} for r in rec_types}

        def iterator() -> Generator[RT | tuple[RT, *tuple[Record, ...]]]:
            for rows in zip(*(df.iter_rows(named=True) for df in dfs)):
                main_row = rows[0]
                idx = tuple(main_row[i] for i in idx_cols)
                idx = idx[0] if len(idx) == 1 else idx

                rec_list = []
                for rec_type, row in zip(rec_types[1:], rows[1:]):
                    new_rec = self.record_type._from_partial_dict(row)
                    rec = loaded[rec_type].get(new_rec._index) or new_rec

                    rec_list.append(rec)

                yield tuple(rec_list) if len(rec_list) > 1 else rec_list[0]

        return iterator()

    def __imatmul__(
        self: DB[RT2, MT, RW, BT, RT2],
        other: DB[RT2, MT, Any, BT] | RecInput[RT2, MT],
    ) -> DB[RT2, MT, RW, BT, RT2]:
        """Aligned assignment."""
        self._mutate(other, mode="update")
        return self

    def __iand__(
        self: DB[RT2, MT, RW, BT, RT2],
        other: DB[RT2, MT, Any, BT] | RecInput[RT2, MT],
    ) -> DB[RT2, MT, RW, BT, RT2]:
        """Replacing assignment."""
        self._mutate(other, mode="replace")
        return self

    def __ior__(
        self: DB[RT2, MT, RW, BT, RT2],
        other: DB[RT2, MT, Any, BT] | RecInput[RT2, MT],
    ) -> DB[RT2, MT, RW, BT, RT2]:
        """Upserting assignment."""
        self._mutate(other, mode="upsert")
        return self

    def __iadd__(
        self: DB[RT2, MT, RW, BT, RT2],
        other: DB[RT2, MT, Any, BT] | RecInput[RT2, MT],
    ) -> DB[RT2, MT, RW, BT, RT2]:
        """Inserting assignment."""
        self._mutate(other, mode="insert")
        return self

    def __isub__(
        self: DB[RT2, MT, RW, BT, RT2],
        other: DB[RT2, MT, Any, BT] | Iterable[MT] | MT,
    ) -> DB[RT2, MT, RW, BT, RT2]:
        """Deletion."""
        raise NotImplementedError("Delete not supported yet.")

    @overload
    def __lshift__(
        self: DB[Any, KT, WT, BT],
        other: DB[RT, KT, Any, BT] | RecInput[RT, KT],
    ) -> list[KT]: ...

    @overload
    def __lshift__(
        self: DB[Record[KT2], Any, WT, BT],
        other: DB[RT, Any, Any, BT] | RecInput[RT, KT2],
    ) -> list[KT2]: ...

    def __lshift__(
        self: DB[Any, Any, WT, BT],
        other: DB[RT, Any, Any, BT] | RecInput[RT, Any],
    ) -> list:
        """Injection."""
        raise NotImplementedError("Inject not supported yet.")

    @overload
    def __rshift__(
        self: DB[Any, KT, RW, BT], other: KT | Iterable[KT]
    ) -> dict[KT, RT]: ...

    @overload
    def __rshift__(
        self: DB[Record[KT2], Any, RW, BT], other: KT2 | Iterable[KT2]
    ) -> dict[KT2, RT]: ...

    def __rshift__(
        self: DB[Any, Any, RW, BT], other: Hashable | Iterable[Hashable]
    ) -> dict[Any, RT]:
        """Extraction."""
        raise NotImplementedError("Extract not supported yet.")

    # 1. Type deletion
    @overload
    def __delitem__(self: DB[RT2, Any, RW, Any], key: type[RT2]) -> None: ...

    # 2. List deletion
    @overload
    def __delitem__(
        self: DB[
            Record[KT2],
            BaseIdx | Filt[BaseIdx] | KT | Filt[KT],
            RW,
            Any,
        ],
        key: Iterable[KT | KT2],
    ) -> None: ...

    # 3. Index value deletion
    @overload
    def __delitem__(
        self: DB[
            Record[KT2],
            BaseIdx | Filt[BaseIdx] | KT | Filt[KT],
            RW,
            Any,
        ],
        key: KT | KT2,
    ) -> None: ...

    # 4. Slice deletion
    @overload
    def __delitem__(
        self: DB[Any, Any, RW, Any], key: slice | tuple[slice, ...]
    ) -> None: ...

    # 5. Expression filter deletion
    @overload
    def __delitem__(
        self: DB[Any, Any, RW, Any], key: sqla.ColumnElement[bool]
    ) -> None: ...

    # Implementation:

    def __delitem__(  # noqa: D105
        self: DB[Record, Any, RW, Any, Any, Any],
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

        select = self.select()

        tables = {self._get_table(rec) for rec in self.record_type._record_bases}

        statements = []

        for table in tables:
            # Prepare delete statement.
            if self.engine.dialect.name in (
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
                                table.c[col.name] == select.c[col.name]
                                for col in table.primary_key.columns
                            ),
                        )
                    )
                )
            else:
                # Correlated update.
                raise NotImplementedError("Correlated update not supported yet.")

        # Execute delete statements.
        with self.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

    # def copy(
    #     self,
    #     backend: B2_opt = None, # type: ignore
    #     # overlay: str | bool = True,
    # ) -> RecordSet[Rec_def, Idx_def, B_nul, R_def, RM]:
    #     """Transfer the DB to a different backend (defaults to in-memory)."""
    #     other = copy_and_override(self, type(self), backend=backend)

    #     for rec in self.schema_types:
    #         self[rec].load(kind=pl.DataFrame).write_database(
    #             str(self._get_table(rec)), str(other.backend_url)
    #         )

    #     return cast(RecordSet[Rec_def, Idx_def, B_nul, R_def, RM], other)

    def extract(
        self: DB[Any, Any, Any, Any, Any, None],
        use_schema: bool | type[Schema] = False,
        aggs: Mapping[RelSet, Agg] | None = None,
    ) -> DB[Record, None, RW, BT]:
        """Extract a new database instance from the current selection."""
        # Get all rec types in the schema.
        rec_types = (
            use_schema._record_types
            if isinstance(use_schema, type)
            else (
                set(self._record_types.keys())
                if use_schema
                else ({self.record_type, *self.record_type._rel_types})
            )
        )

        # Get the entire subdag from this selection.
        all_paths_rels = {
            r
            for rel in self.record_type._rels.values()
            for r in rel._get_subdag(rec_types)
        }

        # Extract rel paths, which contain an aggregated rel.
        aggs_per_type: dict[type[Record], list[tuple[RelSet, Agg]]] = {}
        if aggs is not None:
            for rel, agg in aggs.items():
                for path_rel in all_paths_rels:
                    if rel in path_rel._rel_path:
                        aggs_per_type[rel.parent_type] = [
                            *aggs_per_type.get(rel.parent_type, []),
                            (rel, agg),
                        ]
                        all_paths_rels.remove(path_rel)

        replacements: dict[type[Record], sqla.Select] = {}
        for rec in rec_types:
            # For each table, create a union of all results from the direct routes.
            selects = [
                self[rel].select()
                for rel in all_paths_rels
                if issubclass(rec, rel.record_type)
            ]
            replacements[rec] = sqla.union(*selects).select()

        aggregations: dict[type[Record], sqla.Select] = {}
        for rec, rec_aggs in aggs_per_type.items():
            selects = []
            for rel, agg in rec_aggs:
                src_select = self[rel].select()
                selects.append(
                    sqla.select(
                        *[
                            (
                                src_select.c[sa.name]
                                if isinstance(sa, Col)
                                else sqla_visitors.replacement_traverse(
                                    sa,
                                    {},
                                    replace=lambda element, **kw: (
                                        src_select.c[element.name]
                                        if isinstance(element, Col)
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
        new_db = cast(
            DB[Record, None, RW, BT],
            copy_and_override(self, DB, overlay=f"temp_{token_hex(10)}"),
        )

        # Overlay the new tables onto the new database.
        for rec in rec_types:
            if rec in replacements:
                new_db[rec] &= replacements[rec]

        for rec, agg_select in aggregations.items():
            new_db[rec] &= agg_select

        return new_db

    def __or__(self, other: DB[RT, IT, Any, BT]) -> DB[RT, IT | IT]:
        """Union two datasets."""
        raise NotImplementedError("Union not supported yet.")

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        with self.engine.connect() as conn:
            res = conn.execute(
                sqla.select(sqla.func.count()).select_from(self.select().subquery())
            ).scalar()
            assert isinstance(res, int)
            return res

    def __contains__(self: DB[Any, Any, Any, Backend], key: Hashable) -> bool:
        """Check if a record is in the dataset."""
        return len(self[key]) > 0

    def __clause_element__(self) -> sqla.Subquery:
        """Return subquery for the current selection to be used inside SQL clauses."""
        return self.select().subquery()

    def _joins(self, _subtree: DataDict | None = None) -> list[Join]:
        """Extract join operations from the relation tree."""
        if self.root_table is None:
            return []

        joins = []
        _subtree = _subtree or self.merges.dict

        for rel, next_subtree in _subtree.items():
            parent = (
                self._get_alias(rel._parent_rel)
                if isinstance(rel._parent_rel, RelSet)
                else self.root_table
            )

            temp_alias_map = {
                rec: self._get_random_alias(rec) for rec in rel.inter_joins.keys()
            }

            joins.extend(
                (
                    temp_alias_map[rec],
                    reduce(
                        sqla.or_,
                        (
                            reduce(
                                sqla.and_,
                                (
                                    parent.c[lk.name] == temp_alias_map[rec].c[rk.name]
                                    for lk, rk in join_on.items()
                                ),
                            )
                            for join_on in joins
                        ),
                    ),
                )
                for rec, joins in rel.inter_joins.items()
            )

            target_table = self._get_alias(rel)

            joins.append(
                (
                    target_table,
                    reduce(
                        sqla.or_,
                        (
                            reduce(
                                sqla.and_,
                                (
                                    temp_alias_map[lk.parent_type].c[lk.name]
                                    == target_table.c[rk.name]
                                    for lk, rk in join_on.items()
                                ),
                            )
                            for join_on in rel.target_joins
                        ),
                    ),
                )
            )

            joins.extend(self._joins(next_subtree))

        return joins

    def _get_subdag(
        self,
        backlink_records: set[type[Record]] | None = None,
        _traversed: set[RelSet[Record, Any, Any, Any, Record]] | None = None,
    ) -> set[RelSet[Record, Any, Any, Any, Record]]:
        """Find all paths to the target record type."""
        backlink_records = backlink_records or set()
        _traversed = _traversed or set()

        # Get relations of the target type as next relations
        next_rels = set(self.record_type._rels.values())

        for backlink_record in backlink_records:
            next_rels |= backlink_record._backrels_to_rels(self.record_type)

        # Filter out already traversed relations
        next_rels = {rel for rel in next_rels if rel not in _traversed}

        # Add next relations to traversed set
        _traversed |= next_rels

        if isinstance(self, RelSet):
            # Prefix next relations with current relation
            next_rels = {rel.prefix(self) for rel in next_rels}

        # Return next relations + recurse
        return next_rels | {
            rel
            for next_rel in next_rels
            for rel in next_rel._get_subdag(backlink_records, _traversed)
        }

    def _gen_idx_match_expr(
        self,
        values: Sequence[slice | list[Hashable] | Hashable | sqla.ColumnElement],
    ) -> sqla.ColumnElement[bool] | None:
        """Generate SQL expression for matching index values."""
        if values == slice(None):
            return None

        exprs = [
            (
                idx.label(None).in_(val)
                if isinstance(val, list)
                else (
                    idx.label(None).between(val.start, val.stop)
                    if isinstance(val, slice)
                    else idx.label(None) == val
                )
            )
            for idx, val in zip(self.idx, values)
        ]

        if len(exprs) == 0:
            return None

        return reduce(sqla.and_, exprs)

    def _load_prop(
        self: DB[Any, Any, Any, Any, Any, None],
        p: Col | RelSet,
        parent_idx: Hashable,
    ) -> Any:
        base = self[p.record_type]
        rec = base[parent_idx]

        if isinstance(p, Col):
            return rec.to_df(pd.DataFrame)[p.name].iloc[0]

        return rec

    @staticmethod
    def _normalize_rel_data(
        rec_data: RecordValue[Record],
        parent_idx: Hashable,
        list_idx: bool,
        covered: set[int],
    ) -> dict[Any, Record]:
        """Convert record data to dictionary."""
        res = {}

        if isinstance(rec_data, Record) and id(rec_data) not in covered:
            res = {parent_idx: rec_data}
        elif isinstance(rec_data, Mapping):
            res = {
                (parent_idx, idx): rec
                for idx, rec in rec_data.items()
                if id(rec) not in covered
            }
        elif isinstance(rec_data, Iterable):
            res = {
                (parent_idx, idx if list_idx else rec._index): rec
                for idx, rec in enumerate(rec_data)
                if id(rec) not in covered
            }

        covered |= set(id(r) for r in res.values())
        return res

    @staticmethod
    def _get_record_rels(
        rec_data: dict[Any, dict[Col | RelSet, Any]],
        list_idx: bool,
        covered: set[int],
    ) -> dict[RelSet, dict[Any, Record]]:
        """Get relation data from record data."""
        rel_data: dict[RelSet, dict[Any, Record]] = {}

        for idx, rec in rec_data.items():
            for prop, prop_val in rec.items():
                if isinstance(prop, RelSet) and prop_val is not State.undef:
                    prop = cast(RelSet, prop)
                    rel_data[prop] = {
                        **rel_data.get(prop, {}),
                        **DB._normalize_rel_data(prop_val, idx, list_idx, covered),
                    }

        return rel_data

    @cached_property
    def _has_list_index(self) -> bool:
        return (
            isinstance(self, RelSet)
            and self.path_idx is not None
            and len(self.path_idx) == 1
            and is_subtype(self.path_idx[0].value_type, int)
        )

    def _mutate(  # noqa: C901, D105
        self: DB[RT2, Any, RW, BT, Any],
        value: DB[RT2, Any, Any, Any, Any] | PartialRecInput[RT2, Any],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
        covered: set[int] | None = None,
    ) -> None:
        covered = covered or set()

        record_data: dict[Any, dict[Col | RelSet, Any]] | None = None
        rel_data: dict[RelSet, dict[Any, Record]] | None = None
        df_data: pd.DataFrame | pl.DataFrame | None = None
        partial: bool = False

        match value:
            case Record():
                record_data = {value._index: value._to_dict(with_links=True)}
            case Mapping():
                if has_type(value, Mapping[Col | RelSet, Any]):
                    record_data = {
                        self.record_type._from_partial_dict(value)._index: {
                            p: v for p, v in value.items()
                        }
                    }
                    partial = True
                else:
                    assert has_type(value, Mapping[Any, Record | PartialRec])

                    record_data = {}
                    for idx, rec in value.items():
                        if isinstance(rec, Record):
                            rec_dict = rec._to_dict(with_links=True)
                        else:
                            rec_dict = {p: v for p, v in rec.items()}
                            partial = True
                        record_data[idx] = cast(dict[Col | RelSet, Any], rec_dict)
            case pd.DataFrame() | pl.DataFrame():
                df_data = value
            case Iterable():
                record_data = {}
                for idx, rec in enumerate(value):
                    if isinstance(rec, Record):
                        rec_dict = rec._to_dict(with_links=True)
                        rec_idx = rec._index
                    else:
                        assert has_type(rec, PartialRec)
                        rec_dict = {p: v for p, v in rec.items()}
                        rec_idx = self.record_type._from_partial_dict(rec)._index
                        partial = True

                    record_data[idx if self._has_list_index else rec_idx] = rec_dict
            case _:
                pass

        assert not (
            mode in ("upsert", "replace") and partial
        ), "Partial record input requires update mode."

        if record_data is not None:
            # Split record data into attribute and relation data.
            col_data = {
                idx: {p.name: v for p, v in rec.items() if isinstance(p, Col)}
                for idx, rec in record_data.items()
            }
            rel_data = self._get_record_rels(record_data, self._has_list_index, covered)

            # Transform attribute data into DataFrame.
            df_data = pd.DataFrame.from_dict(col_data, orient="index")

        if rel_data is not None:
            # Recurse into relation data.
            for r, r_data in rel_data.items():
                self[r]._mutate(r_data, mode="replace", covered=covered)

        # Load data into a temporary table.
        if df_data is not None:
            if isinstance(df_data, pd.DataFrame) and any(
                name is None for name in df_data.index.names
            ):
                assert isinstance(self, RelSet)
                idx_names = [vs.name for vs in self.idx]
                df_data.index.set_names(idx_names, inplace=True)

            value_table = sqla.table(
                f"{self.record_type._default_table_name()}_{token_hex(5)}"
            )

            if isinstance(df_data, pd.DataFrame):
                df_data.reset_index().to_sql(
                    value_table.name,
                    self.engine,
                    if_exists="replace",
                    index=False,
                )
            else:
                df_data.write_database(str(value_table), self.engine)

        elif isinstance(value, sqla.Select):
            value_table = value.subquery()
        elif isinstance(value, DB):
            value_table = value.select().subquery()
        else:
            raise ValueError("Could not parse input data.")

        cols_by_table = {
            self._get_table(rec): {
                a for a in self.record_type._cols.values() if a.parent_type is rec
            }
            for rec in self.record_type._record_bases
        }

        statements = []

        if mode == "replace":
            # Delete all records in the current selection.
            select = self.select()

            for table in cols_by_table:
                # Prepare delete statement.
                if self.engine.dialect.name in (
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
                                    table.c[col.name] == select.c[col.name]
                                    for col in table.primary_key.columns
                                ),
                            )
                        )
                    )
                else:
                    # Correlated update.
                    raise NotImplementedError("Correlated update not supported yet.")

        if mode in ("replace", "upsert", "insert"):
            # Construct the insert statements.

            assert len(self.filters) == 0, "Can only upsert into unfiltered datasets."

            for table, cols in cols_by_table.items():
                # Do an insert-from-select operation, which updates on conflict:
                statement = table.insert().from_select(
                    [a.name for a in cols],
                    value_table,
                )

                if mode in ("replace", "upsert"):
                    if isinstance(statement, postgresql.Insert):
                        # For Postgres / DuckDB, use: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#updating-using-the-excluded-insert-values
                        statement = statement.on_conflict_do_update(
                            index_elements=[
                                col.name for col in table.primary_key.columns
                            ],
                            set_=dict(statement.excluded),
                        )
                    elif isinstance(statement, mysql.Insert):
                        # For MySQL / MariaDB, use: https://docs.sqlalchemy.org/en/20/dialects/mysql.html#insert-on-duplicate-key-update-upsert
                        statement = statement.prefix_with("INSERT INTO")
                        statement = statement.on_duplicate_key_update(
                            **statement.inserted
                        )
                    else:
                        # For others, use CTE: https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.cte
                        raise NotImplementedError(
                            "Upsert not supported for this database dialect."
                        )

                statements.append(statement)
        else:
            # Construct the update statements.

            assert isinstance(self.root_table, sqla.Table), "Base must be a table."

            # Derive current select statement and join with value table, if exists.
            value_join_on = (
                sqla.text("TRUE")
                if self.single_key is not None
                else reduce(
                    sqla.and_,
                    (
                        self.root_table.c[idx_col.name] == idx_col
                        for idx_col in value_table.primary_key
                    ),
                )
            )
            select = self.select().join(
                value_table,
                value_join_on,
            )

            for table, cols in cols_by_table.items():
                col_names = {a.name for a in cols}
                values = {
                    c_name: c
                    for c_name, c in value_table.columns.items()
                    if c_name in col_names
                }

                # Prepare update statement.
                if self.engine.dialect.name in (
                    "postgres",
                    "postgresql",
                    "duckdb",
                    "mysql",
                    "mariadb",
                ):
                    # Update-from.
                    statements.append(
                        table.update()
                        .values(values)
                        .where(
                            reduce(
                                sqla.and_,
                                (
                                    table.c[col.name] == select.c[col.name]
                                    for col in table.primary_key.columns
                                ),
                            )
                        )
                    )
                else:
                    # Correlated update.
                    raise NotImplementedError("Correlated update not supported yet.")

        # Execute delete / insert / update statements.
        with self.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        if mode in ("replace", "upsert") and isinstance(self, RelSet):
            # Update incoming relations from parent records.
            if self.direct_rel is not True:
                if issubclass(self.fk_record_type, self.parent_type):
                    # Case: parent links directly to child (n -> 1)
                    for fk, pk in self.direct_rel.fk_map.items():
                        self.parent_set[fk] @= value_table.select().with_only_columns(
                            value_table.c[pk.name]
                        )
                else:
                    # Case: parent and child are linked via assoc table (n <--> m)
                    # Update link table with new child indexes.
                    assert issubclass(self.link_type, Record)
                    ls = self.link_set
                    ls &= value_table.select()

            # Note that the (1 <- n) case is already covered by updating
            # the child record directly, which includes all its foreign keys.

        # Drop the temporary table, if any.
        if isinstance(value_table, sqla.Table):
            cast(sqla.Table, value_table).drop(self.engine)

        return

    def __setitem__(
        self,
        key: Any,
        other: DB[Any, Any, Any, Any, Any, Any] | Col[Any, Any, Any, Any, Any],
    ) -> None:
        """Catchall setitem."""
        return


@dataclass(kw_only=True, eq=False)
class Col(  # type: ignore[reportIncompatibleVariableOverride]
    Prop[VTI, WT],
    sqla.ColumnClause[VTI],
    Generic[VTI, WT, CT, BT, RT],
):
    """Reference an attribute of a record."""

    def __post_init__(self) -> None:  # noqa: D105
        # Initialize fields required by SQLAlchemy superclass.
        self.table = None
        self.is_literal = False

    record_set: DB[RT, Any, WT, BT] = field(default=Record._db)

    index: bool = False
    primary_key: bool = False
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = None

    @cached_property
    def sql_type(self) -> sqla_types.TypeEngine:
        """Column key."""
        return sqla_types.to_instance(self.value_type)  # type: ignore[reportCallIssue]

    def all(self) -> sqla.CollectionAggregate[bool]:
        """Return a SQL ALL expression for this attribute."""
        return sqla.all_(self)

    def any(self) -> sqla.CollectionAggregate[bool]:
        """Return a SQL ANY expression for this attribute."""
        return sqla.any_(self)

    @cached_property
    def value_type(self) -> SingleTypeDef[VTI]:
        """Value type of the property."""
        return self._type.value_type()

    @cached_property
    def record_type(self) -> type[RT]:
        """Record type of the set."""
        return self.record_set.record_type

    @overload
    def __get__(
        self, instance: None, owner: type[RT2]
    ) -> Col[VTI, WT, CT, RT2, RT2]: ...

    @overload
    def __get__(
        self: Col[Any, Any, Any, RT2, Any], instance: RT2, owner: type[RT2]
    ) -> VTI: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[RT2] | None
    ) -> Col[Any, Any, Any, Any, Any] | VTI | Self:
        if isinstance(instance, Record):
            if self.getter is not None:
                value = self.getter(instance)
            else:
                value = instance.__dict__.get(self.name, State.undef)

            if value is State.undef and not self.local:
                value = instance._db._load_prop(
                    getattr(owner, self.name), instance._index
                )

            if value is State.undef:
                if self.default_factory is not None:
                    value = self.default_factory()
                elif self.default is not None:
                    value = self.default

            if value is State.undef:
                raise ValueError("Property value could not be fetched.")

            instance.__dict__[self.name] = value
            return value
        elif owner is not None and issubclass(owner, Record):
            rec_type = cast(type[RT2], owner)
            return copy_and_override(
                self,
                Col[VTI, WT, CT, RT, RT],
                record_set=cast(DB[RT, BaseIdx, WT, RT, RT], rec_type._db),
            )

        return self

    def __set__(self: Prop[VT2, RW], instance: Record, value: VT2 | State) -> None:
        """Set the value of the property."""
        parsed_value = value

        if has_type(value, RecInput):
            parsed_value = cast(RecInput[Record, Any], value)
            match parsed_value:
                case Record():
                    parsed_value = (
                        copy_and_override(
                            parsed_value, type(parsed_value), _db=instance._db
                        )
                        if parsed_value._db.backend_name != instance._db.backend_name
                        else parsed_value
                    )
                case pd.DataFrame() | pl.DataFrame():
                    pass
                case Mapping():
                    parsed_value = {
                        k: (
                            copy_and_override(v, type(v), _db=instance._db)
                            if v._db.backend_name != instance._db.backend_name
                            else v
                        )
                        for k, v in parsed_value.items()
                    }
                case Iterable():
                    parsed_value = [
                        (
                            copy_and_override(v, type(v), _db=instance._db)
                            if v._db.backend_name != instance._db.backend_name
                            else v
                        )
                        for v in parsed_value
                    ]
                case _:
                    pass

        if self.setter is not None and parsed_value is not State.undef:
            self.setter(instance, cast(VT2, parsed_value))

        instance.__dict__[self.name] = parsed_value

    # Plural selection
    @overload
    def __getitem__(
        self: Col[Any, Any, KT],
        key: Iterable[KT] | slice | tuple[slice, ...],
    ) -> Col[VTI, WT, KT, BT, RT]: ...

    # Single value selection
    @overload
    def __getitem__(
        self: Col[Any, Any, KT], key: KT
    ) -> Col[VTI, WT, SingleIdx, BT, RT]: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self: Col[Any, Any, KT],
        key: list[Hashable] | slice | tuple[slice, ...] | Hashable,
    ) -> Col[Any, Any, Any, Any, Any]:
        return copy_and_override(
            self, Col, record_set=self.record_set[cast(slice, key)]
        )

    def __setitem__(
        self,
        key: Any,
        other: Col[Any, Any, Any, Any, Any],
    ) -> None:
        """Catchall setitem."""
        return

    def select(self) -> sqla.Select:
        """Return select statement for this dataset."""
        selection_table = self.record_set.root_table
        assert selection_table is not None

        return sqla.select(
            *(selection_table.c[a.name] for a in self.idx),
            selection_table.c[self.name],
        ).select_from(selection_table)

    @overload
    def to_series(self: Col, kind: type[pd.Series]) -> pd.Series: ...

    @overload
    def to_series(self: Col, kind: type[pl.Series] = ...) -> pl.Series: ...

    def to_series(
        self,
        kind: type[pd.Series | pl.Series] = pl.Series,
    ) -> pd.Series | pl.Series:
        """Download selection."""
        select = self.select()
        engine = self.record_set.engine

        if kind is pd.Series:
            with engine.connect() as con:
                return pd.read_sql(select, con).set_index(
                    [c.key for c in self._idx_cols]
                )[self.name]

        return pl.read_database(str(select.compile(engine)), engine)[self.name]

    def __imatmul__(
        self: Col[VT, RW, KT, BT, RT],
        value: ValInput | Col[VT, Any, KT, BT],
    ) -> Col[VT, RW, KT, BT, RT]:
        """Do item-wise, broadcasting assignment on this value set."""
        match value:
            case pd.Series() | pl.Series():
                series = value
            case Mapping():
                series = pd.Series(dict(value))
            case Iterable():
                series = pd.Series(dict(enumerate(value)))
            case _:
                series = pd.Series({self.single_key or 0: value})

        df = series.rename(self.name).to_frame()

        self.record_set._mutate(df)
        return self


@dataclass(kw_only=True, eq=False)
class RelSet(
    Prop[RT | Iterable[RT], WT],
    DB[RT, IT, WT, BT, RTS, TT],
    Generic[RT, LT, IT, WT, BT, PT, RTS, TT],
):
    """Relation set class."""

    parent_type: type[PT] = Record

    index: bool = False
    primary_key: bool = False

    on: DirectLink[RT] | BackLink[RT] | BiLink[RT, Any] | None = None
    order_by: Mapping[Col[Any, Any, Any, Record], int] | None = None
    map_by: Col[Any, Any, Any, Record] | None = None

    @property
    def dyn_target_type(self) -> type[RT] | None:
        """Dynamic target type of the relation."""
        match self.on:
            case dict():
                return cast(type[RT], next(iter(self.on.values())).parent_type)
            case tuple():
                via_1 = self.on[1]
                assert isinstance(via_1, DB)
                return via_1.record_type
            case type() | DB() | Col() | Iterable() | None:
                return None
            case _:
                return None

    @overload
    def __get__(
        self: RelSet[Any, Any, Any, Any, Any, Any, Any],
        instance: None,
        owner: type[RT2],
    ) -> RelSet[RT, LT, IT, WT, RT2, RT2, RTS, TT]: ...

    @overload
    def __get__(
        self: RelSet[Any, Any, SingleIdx, Any, Any, Any, Any],
        instance: RT2,
        owner: type[RT2],
    ) -> RT: ...

    @overload
    def __get__(
        self: RelSet[Any, Any, SingleIdx, Any, Any, RT2, Any],
        instance: RT2,
        owner: type[RT2],
    ) -> RelSet[RT, LT, IT, WT, BT, RT2, RTS, TT]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[RT2] | None
    ) -> RelSet[Any, Any, Any, Any, Any, Any, Any, Any] | RT | Self:
        raise NotImplementedError()

    @cached_property
    def record_type(self) -> type[RT]:
        """Record type of the set."""
        return cast(type[RT], self._type.record_type())

    @staticmethod
    def _get_tag(rec_type: type[Record]) -> RelSet | None:
        """Retrieve relation-tag of a record type, if any."""
        try:
            rel = getattr(rec_type, "_rel")
            return rel if isinstance(rel, RelSet) else None
        except AttributeError:
            return None

    @cached_property
    def _parent_rel(self) -> RelSet[PT, Any, Any, WT, BT, Any, PT] | None:
        """Parent relation of this Rel."""
        return cast(
            RelSet[PT, Any, Any, WT, BT, Any, PT],
            (
                self._get_tag(self.parent_type)
                if issubclass(self.parent_type, Record)
                else None
            ),
        )

    @cached_property
    def _rel_path(self) -> tuple[type[Record], *tuple[RelSet, ...]]:
        """Path from base record type to this Rel."""
        if self._parent_rel is None:
            return (self.parent_type,)

        return cast(
            tuple[type[Record], *tuple[RelSet, ...]],
            (
                *self._parent_rel._rel_path,
                *([self] if isinstance(self, RelSet) else []),
            ),
        )

    @cached_property
    def root_type(self) -> type[Record]:
        """Root record type of the set."""
        return self._rel_path[0]

    @cached_property
    def root_table(self) -> sqla.FromClause:
        """Get the main table for the current selection."""
        return self._get_random_alias(self.root_type)

    def unbind(
        self,
    ) -> RelSet[RT, LT, IT, WT, Record, PT, RTS]:
        """Return backend-less version of this RelSet."""
        tmpl = cast(
            RelSet[RT, LT, IT, WT, Record, PT, RTS],
            self,
        )
        return copy_and_override(
            tmpl,
            type(tmpl),
            merges=RelTree(),
        )

    def prefix(
        self,
        left: type[PT] | RelSet[PT, Any, Any, Any, Any, Any, Any, Any],
    ) -> RelSet[RT, LT, IT, WT, BT, PT, RT, TT]:
        """Prefix this prop with a relation or record type."""
        current_root = self.root_type
        new_root = left if isinstance(left, RelSet) else left._rel(current_root)

        rel_path = self._rel_path[1:] if len(self._rel_path) > 1 else (self,)

        prefixed_rel = reduce(
            lambda r1, r2: copy_and_override(
                r2,
                RelSet,
                rel=r2.rel,  # type: ignore[reportArgumentType]
                parent_type=r1.rec,  # type: ignore[reportArgumentType]
                keys=[*r2.keys, *r1.keys],
                filters=[*r2.filters, *r1.filters],
                merges=r2.merges * r1.merges,
                backend=r1.backend,  # type: ignore[reportArgumentType]
                url=r1.url,
            ),
            rel_path,
            new_root,
        )

        return cast(
            RelSet[RT, LT, IT, WT, BT, PT, RT, TT],
            prefixed_rel,
        )

    @cached_property
    def path_idx(self) -> list[Col] | None:
        """Get the path index of the relation."""
        p = [
            a
            for rel in self._rel_path[1:]
            for a in (
                [rel.map_by]
                if rel.map_by is not None
                else rel.order_by.keys() if rel.order_by is not None else []
            )
        ]

        if len(p) == 0:
            return None

        return [
            *self.root_type._primary_keys.values(),
            *p,
        ]

    @cached_property
    def path_str(self) -> str:
        """String representation of the relation path."""
        prefix = (
            self.parent_type.__name__
            if len(self._rel_path) == 0
            else self._rel_path[-1].path_str
        )
        return f"{prefix}.{self.name}"

    @cached_property
    def parent_set(self) -> DB[PT, Any, WT, BT, PT]:
        """Parent set of this Rel."""
        if self._parent_rel is not None:
            tmpl = cast(RelSet[PT, Any, Any, WT, BT, Any, PT], self)
            return copy_and_override(
                tmpl,
                type(tmpl),
                on=self._parent_rel.on,
                parent_type=self._parent_rel.parent_type,
            )

        tmpl = cast(DB[PT, Any, WT, BT, PT], self)
        return copy_and_override(tmpl, type(tmpl))

    @property
    def link_type(self) -> type[LT]:
        """Get the link record type."""
        return cast(
            type[LT],
            self._type.link_type(),
        )

    @property
    def link_set(
        self: RelSet[Any, RT2, Any, Any, Any, Any, Any],
    ) -> DB[RT2, KT, WT, BT, RT2]:
        """Get the link set."""
        r = self.parent_type._rel(self.link_type)
        return self.parent_set[r]

    @cached_property
    def link(self: RelSet[Any, RT2, Any, Any, Any, Any]) -> type[RT2]:
        """Reference props of the link record type."""
        return (
            self.link_set.rec if self.link_set is not None else cast(type[RT2], Record)
        )

    @cached_property
    def fk_record_type(self) -> type[Record]:
        """Record type of the foreign key."""
        match self.on:
            case type():
                return self.on
            case RelSet():
                return self.on.parent_type
            case tuple():
                link = self.on[0]
                assert isinstance(link, DB)
                return link.parent_type
            case None if issubclass(self.link_type, Record):
                return self.link_type
            case dict() | Col() | Iterable() | None:
                return self.parent_type
            case _:
                raise ValueError("Invalid relation definition.")

    @cached_property
    def direct_rel(
        self,
    ) -> RelSet[PT, LT, SingleIdx, WT, BT, Record] | Literal[True]:
        """Direct rel."""
        on = self.on
        if on is None and issubclass(self.link_type, Record):
            on = self.link_type

        match on:
            case type():
                rels = [
                    r
                    for r in on._rels.values()
                    if issubclass(self.parent_type, r.record_type)
                    and r.direct_rel is True
                ]
                assert len(rels) == 1, "Direct relation must be unique."
                return cast(
                    RelSet[PT, LT, SingleIdx, WT, BT, Record],
                    rels[0],
                )
            case RelSet():
                return cast(
                    RelSet[PT, LT, SingleIdx, WT, BT, Record],
                    on,
                )
            case tuple():
                link = on[0]
                assert isinstance(link, RelSet)
                return cast(RelSet[PT, LT, SingleIdx, WT, BT, Record], link)
            case dict() | Col() | Iterable() | None:
                return True
            case _:
                raise ValueError("Invalid relation definition.")

    @cached_property
    def counter_rel(self) -> RelSet[PT, LT, Any, WT, BT, RT]:
        """Counter rel."""
        if self.direct_rel is not True and issubclass(
            self.direct_rel.parent_type, self.record_type
        ):
            return cast(RelSet[PT, LT, Any, WT, BT, RT], self.direct_rel)

        return cast(
            RelSet[PT, LT, Any, WT, BT, RT],
            self.record_type._rel(self.parent_type),
        )

    @cached_property
    def fk_map(
        self,
    ) -> bidict[Col[Hashable], Col[Hashable]]:
        """Map source foreign keys to target cols."""
        target = self.record_type
        on = self.on
        if on is None and issubclass(self.link_type, Record):
            on = self.link_type

        match on:
            case type() | RelSet() | tuple():
                return bidict()
            case dict():
                return bidict(
                    {
                        Col[Hashable](
                            _name=fk.name,
                            _type=PropType(Col[cast(type[Hashable], fk.value_type)]),
                            record_set=cast(RelSet, self.unbind()),
                        ): cast(Col[Hashable], pk)
                        for fk, pk in on.items()
                    }
                )
            case Col() | Iterable():
                cols = on if isinstance(on, Iterable) else [on]
                source_cols = [
                    Col[Hashable](
                        _name=col.name,
                        _type=PropType(Col[cast(type[Hashable], col.value_type)]),
                        record_set=cast(RelSet, self.unbind()),
                    )
                    for col in cols
                ]
                target_cols = target._primary_keys.values()
                fk_map = dict(zip(source_cols, target_cols))

                assert all(
                    is_subtype(
                        self.parent_type._defined_cols[fk_col.name].value_type,
                        pk_col.value_type,
                    )
                    for fk_col, pk_col in fk_map.items()
                    if pk_col.typedef is not None
                ), "Foreign key value types must match primary key value types."

                return bidict(fk_map)
            case None:
                return bidict(
                    {
                        Col[Hashable](
                            _name=f"{self.name}_{target_col.name}",
                            _type=PropType(Col[target_col.value_type, Any, Record]),
                            record_set=cast(RelSet, self.unbind()),
                        ): cast(Col[Hashable], target_col)
                        for target_col in target._primary_keys.values()
                    }
                )
            case _:
                return bidict()

    @cached_property
    def inter_joins(
        self,
    ) -> dict[
        type[Record],
        list[Mapping[Col[Hashable], Col[Hashable]]],
    ]:
        """Intermediate joins required by this rel."""
        on = self.on
        if on is None and issubclass(self.link_type, Record):
            on = self.link_type

        match on:
            case RelSet():
                # Relation is defined via other relation
                other_rel = on
                assert isinstance(
                    other_rel, DB
                ), "Back-reference must be an explicit relation"

                if issubclass(other_rel.record_type, self.parent_type):
                    # Supplied record type object is a backlinking relation
                    return {}
                else:
                    # Supplied record type object is a forward relation
                    # on a relation table
                    back_rels = [
                        rel
                        for rel in other_rel.parent_type._rels.values()
                        if issubclass(rel.record_type, self.parent_type)
                        and len(rel.fk_map) > 0
                    ]

                    return {
                        other_rel.parent_type: [
                            back_rel.fk_map.inverse for back_rel in back_rels
                        ]
                    }
            case type():
                if issubclass(on, self.record_type):
                    # Relation is defined via all direct backlinks of given record type.
                    return {}

                # Relation is defined via relation table
                back_rels = [
                    rel
                    for rel in on._rels.values()
                    if issubclass(rel.record_type, self.parent_type)
                    and len(rel.fk_map) > 0
                ]

                return {on: [back_rel.fk_map.inverse for back_rel in back_rels]}
            case tuple() if has_type(on, tuple[RelSet, RelSet]):
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                back, _ = on
                assert len(back.fk_map) > 0, "Back relation must be direct."
                assert issubclass(back.parent_type, Record)
                return {back.parent_type: [back.fk_map.inverse]}
            case _:
                # Relation is defined via foreign key attributes
                return {}

    @cached_property
    def target_joins(
        self,
    ) -> list[Mapping[Col[Hashable], Col[Hashable]]]:
        """Mappings of column keys to join the target on."""
        on = self.on
        if on is None and issubclass(self.link_type, Record):
            on = self.link_type

        match on:
            case RelSet():
                # Relation is defined via other relation or relation table
                other_rel = on
                assert (
                    len(other_rel.fk_map) > 0
                ), "Backref or forward-ref on relation table must be a direct relation"
                return [
                    (
                        other_rel.fk_map.inverse
                        if issubclass(other_rel.record_type, self.parent_type)
                        else other_rel.fk_map
                    )
                ]

            case type():
                if issubclass(on, self.record_type):
                    # Relation is defined via all direct backlinks of given record type.
                    back_rels = [
                        rel
                        for rel in on._rels.values()
                        if issubclass(rel.record_type, self.parent_type)
                        and len(rel.fk_map) > 0
                    ]
                    assert len(back_rels) > 0, "No direct backlinks found."
                    return [back_rel.fk_map.inverse for back_rel in back_rels]

                # Relation is defined via relation table
                fwd_rels = [
                    rel
                    for rel in on._rels.values()
                    if issubclass(rel.record_type, self.parent_type)
                    and len(rel.fk_map) > 0
                ]
                assert (
                    len(fwd_rels) > 0
                ), "No direct forward rels found on relation table."
                return [fwd_rel.fk_map for fwd_rel in fwd_rels]

            case tuple():
                assert has_type(on, tuple[RelSet, RelSet])
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                _, fwd = on
                assert len(fwd.fk_map) > 0, "Forward relation must be direct."
                return [fwd.fk_map]

            case _:
                # Relation is defined via foreign key attributes
                return [self.fk_map]


class Rel(RelSet[RT, None, SingleIdx, WT, BT, PT, RTS, TT]):
    """Singular relation."""


class RecUUID(Record[UUID]):
    """Record type with a default UUID primary key."""

    _template = True
    _id: Col[UUID] = prop(primary_key=True, default_factory=uuid4)


class RecHashed(Record[int]):
    """Record type with a default hashed primary key."""

    _template = True

    _id: Col[int] = prop(primary_key=True, init=False)

    def __post_init__(self) -> None:  # noqa: D105
        self._id = gen_int_hash(
            {a.name: getattr(self, a.name) for a in self._defined_cols.values()}
        )


class Scalar(Record[KT], Generic[VT2, KT]):
    """Dynamically defined record type."""

    _template = True

    _id: Col[KT] = prop(primary_key=True, default_factory=uuid4)
    _value: Col[VT2]


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls: type[Record], name: str) -> Col[Any, Any, Record]:
        """Get dynamic attribute by dynamic name."""
        return Col(_name=name, _type=PropType(Col[cls]), record_set=cls._db)

    def __getattr__(cls: type[Record], name: str) -> Col[Any, Any, Record]:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)

        return Col(_name=name, _type=PropType(Col[cls]), record_set=cls._db)


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


a = DynRecord


def dynamic_record_type(
    name: str, props: Iterable[Prop[Any, Any]] = []
) -> type[DynRecord]:
    """Create a dynamically defined record type."""
    return type(name, (DynRecord,), {p.name: p for p in props})


class Link(RecHashed, Generic[RT2, RT3]):
    """Automatically defined relation record type."""

    _template = True

    _from: RelSet[RT2]
    _to: RelSet[RT3]


class Schema:
    """Group multiple record types into a schema."""

    _record_types: set[type[Record]]
    _rel_record_types: set[type[Record]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        subclasses = get_subclasses(cls, max_level=1)
        cls._record_types = {s for s in subclasses if isinstance(s, Record)}
        cls._rel_record_types = {rr for r in cls._record_types for rr in r._rel_types}
        super().__init_subclass__()


@dataclass
class Require:
    """Mark schema or record type as required."""

    present: bool = True


type DataDict = dict[RelSet, DataDict]
