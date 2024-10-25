"""Static schemas for universal relational databases."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import MISSING, Field, dataclass, field
from datetime import date, datetime, time
from functools import cached_property, partial, reduce
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
import sqlalchemy.orm as orm
import sqlalchemy.sql.type_api as sqla_types
import sqlalchemy.sql.visitors as sqla_visitors
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

from py_research.data import copy_and_override
from py_research.enums import StrEnum
from py_research.files import HttpFile
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericProtocol,
    SingleTypeDef,
    extract_nullable_type,
    get_lowest_common_base,
    has_type,
    is_subtype,
)

ValT = TypeVar("ValT", covariant=True)
ValTi = TypeVar("ValTi", default=Any)
ValT2 = TypeVar("ValT2")
ValT3 = TypeVar("ValT3")
ValT4 = TypeVar("ValT4")

WriteT = TypeVar("WriteT", bound="R", default="RW", covariant=True)
WriteT2 = TypeVar("WriteT2", bound="R")
WriteT3 = TypeVar("WriteT3", bound="R")

PubT = TypeVar("PubT", bound="Public | Private", default="Public")

RecT = TypeVar("RecT", bound="Record", covariant=True, default="Record")
RecT2 = TypeVar("RecT2", bound="Record")
RecT3 = TypeVar("RecT3", bound="Record")
RecTt = TypeVarTuple("RecTt")


RelT = TypeVar("RelT", bound="Record | None", covariant=True, default="Record | None")
RelT2 = TypeVar("RelT2", bound="Record | None")
RelT3 = TypeVar("RelT3", bound="Record | None")


ParT = TypeVar("ParT", bound="Record", contravariant=True, default=Any)
ParT2 = TypeVar("ParT2", bound="Record")
ParT3 = TypeVar("ParT3", bound="Record")


LnT = TypeVar("LnT", bound="Record | None", covariant=True, default=None)
LnT2 = TypeVar("LnT2", bound="Record | None")
LnT3 = TypeVar("LnT3", bound="Record | None")


KeyT = TypeVar("KeyT", bound=Hashable, default=Any)
KeyT2 = TypeVar("KeyT2", bound=Hashable)
KeyT3 = TypeVar("KeyT3", bound=Hashable)
KeyTt = TypeVarTuple("KeyTt")

IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound="Hashable | BaseIdx",
    default="BaseIdx",
)
IdxT2 = TypeVar(
    "IdxT2",
    bound="Hashable | BaseIdx",
)
IdxT3 = TypeVar(
    "IdxT3",
    bound="Hashable | BaseIdx",
)

BackT = TypeVar(
    "BackT",
    bound="StatBackendID",
    covariant=True,
    default="Static",
)
BackT2 = TypeVar(
    "BackT2",
    bound="StatBackendID",
)
BackT3 = TypeVar(
    "BackT3",
    bound="StatBackendID",
)

SelT = TypeVar("SelT", covariant=True, bound="Record | tuple | None", default=None)
SelT2 = TypeVar("SelT2", bound="Record | tuple | None")
SelT3 = TypeVar("SelT3", bound="Record | tuple | None")

FiltT = TypeVar(
    "FiltT",
    bound="Full | Filtered | Singular | Nullable",
    covariant=True,
    default="Full",
)
FiltT2 = TypeVar("FiltT2", bound="Full | Filtered | Singular | Nullable")
FiltT3 = TypeVar("FiltT3", bound="Full | Filtered | Singular | Nullable")

SingleT2 = TypeVar("SingleT2", bound="Singular | Nullable")


RefT = TypeVar("RefT", bound="Singular | Nullable", covariant=True, default="Singular")

LinkT = TypeVar("LinkT", bound="Record", covariant=True, default="Link")


DfT = TypeVar("DfT", bound=pd.DataFrame | pl.DataFrame)


Params = ParamSpec("Params")


@final
class Undef:
    """Demark undefined status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class Keep:
    """Demark unchanged status."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


class R:
    """Base class for writability flags."""


@final
class RO(R):
    """Read-only flag."""


@final
class RW(R):
    """Read-write flag."""


class Local(StrEnum):
    """Local backend."""

    static = "local-stat"
    dynamic = "local-dyn"


type Dynamic = Literal[Local.dynamic]
type Static = Literal[Local.static]
type DynBackendID = LiteralString | Local
type StatBackendID = DynBackendID | Static


type RecordValue[Rec: Record] = Rec | Iterable[Rec] | Mapping[Any, Rec] | None


type RecInput[
    Rec: Record, Key: Hashable, RKey: Hashable
] = pd.DataFrame | pl.DataFrame | Iterable[Rec | RKey] | Mapping[
    Key, Rec | RKey
] | sqla.Select | Rec | RKey


type ValInput[Val, Key: Hashable] = pd.Series | pl.Series | Mapping[
    Key, Val
] | sqla.Select[tuple[Val] | tuple[Key, Val]] | Val


@final
class BaseIdx:
    """Singleton to mark dataset as having the record type's base index."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class Full:
    """Singleton to mark dataset as having full index."""


@final
class Filtered:
    """Singleton to mark dataset index as filtered."""


@final
class Singular:
    """Singleton to mark dataset index as a single value."""


@final
class Nullable:
    """Singleton to mark dataset index as a single or no value."""


type Index = Hashable | Full | Singular | Filtered
type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]


_pl_type_map: dict[type, pl.DataType | type] = {
    UUID: pl.String,
}


def _get_pl_schema(cols: Iterable[Col[Any, Any, Any, Any, Any, Any]]) -> pl.Schema:
    """Return the schema of the dataset."""
    return pl.Schema(
        {
            col.name: _pl_type_map.get(col.value_origin_type, col.value_origin_type)
            for col in cols
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
    fks: Mapping[str, Col[Any, Any, Static, Any, BaseIdx, Singular | Nullable]] = {},
) -> Prop[Any, Any]:
    is_rel = name in fks
    value_type = (
        _pd_to_py_dtype(data)
        if isinstance(data, pd.Series | pl.Series)
        else _sql_to_py_dtype(data)
    ) or Any
    attr = Attr[value_type](
        primary_key=pk, _name=name if not is_rel else f"fk_{name}", _type=value_type
    )
    return (
        attr
        if not is_rel
        else Rel(fks=fks[name], _type=fks[name].attr.value_type, _name=f"rel_{name}")
    )


def props_from_data(
    data: pd.DataFrame | pl.DataFrame | sqla.Select,
    foreign_keys: (
        Mapping[str, Col[Any, Any, Static, Any, BaseIdx, Singular | Nullable]] | None
    ) = None,
    primary_keys: list[str] | None = None,
) -> list[Prop]:
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


@dataclass
class PropType(Generic[ValT]):
    """Reference to a type."""

    hint: str | SingleTypeDef[Prop[ValT, Any]] | None = None
    ctx: ModuleType | None = None
    typevar_map: dict[TypeVar, SingleTypeDef] = field(default_factory=dict)

    def __set__(  # noqa: D105
        self, instance: object, value: PropType | str | type[Prop] | SingleTypeDef[ValT]
    ) -> None:
        if isinstance(value, PropType):
            self.hint = value.hint
            self.ctx = value.ctx
            self.typevar_map = value.typevar_map
        elif isinstance(value, str) or is_subtype(value, Prop):
            self.hint = value
        else:
            self.hint = cast(type[Prop[ValT, Any]], Prop[value])

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.hint, self.ctx, self.typevar_map))

    def prop_type(self) -> type[Prop] | type[None]:
        """Resolve the property type reference."""
        hint = self.hint

        if hint is None:
            return NoneType

        if has_type(hint, SingleTypeDef):
            base = get_origin(hint)
            if base is None or not issubclass(base, Prop):
                return NoneType

            return base
        elif isinstance(hint, str):
            return (
                Attr
                if "Attr" in hint
                else (
                    RelSet
                    if "RelSet" in hint
                    else Rel if "Rel" in hint else Prop if "Prop" in hint else NoneType
                )
            )
        else:
            return NoneType

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

    def value_type(self: PropType[ValT2]) -> SingleTypeDef[ValT2]:
        """Resolve the value type reference."""
        args = self._generic_args
        if len(args) == 0:
            return cast(type[ValT2], object)

        arg = args[0]
        return cast(SingleTypeDef[ValT2], self._to_typedef(arg))

    def record_type(
        self: PropType[Record | Iterable[Record | None] | None],
    ) -> type[Record]:
        """Resolve the record type reference."""
        assert is_subtype(self._generic_type, RelTable)

        args = self._generic_args
        assert len(args) > 0

        recs = args[0]
        if isinstance(recs, TypeVar):
            recs = self._to_type(recs)

        rec_args = get_args(recs)
        rec_arg = extract_nullable_type(recs)
        assert rec_arg is not None

        if is_subtype(recs, Iterable):
            assert len(rec_args) >= 1
            rec_arg = rec_args[0]

        rec_type = self._to_type(rec_arg)
        assert issubclass(rec_type, Record)

        return rec_type

    def link_type(
        self: PropType[Record | Iterable[Record | None] | None],
    ) -> type[Record | None]:
        """Resolve the record type reference."""
        assert is_subtype(self._generic_type, RelTable)
        args = self._generic_args
        rec = args[1]
        rec_type = self._to_type(rec)

        if not issubclass(rec_type, Record):
            return NoneType

        return rec_type


@dataclass(eq=False)
class Prop(Generic[ValT, WriteT]):
    """Record property."""

    _name: str | None = None
    _type: PropType[ValT] = field(default_factory=PropType[ValT])

    @property
    def name(self) -> str:
        """Property name."""
        assert self._name is not None
        return self._name

    @cached_property
    def value_type(self) -> SingleTypeDef[ValT]:
        """Value type of the property."""
        return self._type.value_type()

    @cached_property
    def value_origin_type(self) -> type:
        """Value type of the property."""
        return (
            self.value_type
            if isinstance(self.value_type, type)
            else get_origin(self.value_type) or object
        )

    def __set_name__(self, _, name: str) -> None:  # noqa: D105
        if self._name is None:
            self._name = name
        else:
            assert name == self._name


@final
class Public:
    """Demark public status of attribute."""


@final
class Private:
    """Demark private status of attribute."""


@dataclass(eq=False)
class Attr(Prop[ValT, WriteT], Generic[ValT, WriteT, PubT]):
    """Record attribute."""

    alias: str | None = None
    init: bool = True
    default: ValT | type[Undef] = Undef
    default_factory: Callable[[], ValT] | None = None

    getter: Callable[[Record], ValT] | None = None
    setter: Callable[[Record, ValT], None] | None = None
    pub_status: type[PubT] = Public

    index: bool = False
    primary_key: bool = False
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = None

    @property
    def name(self) -> str:
        """Property name."""
        if self.alias is not None:
            return self.alias

        return super().name

    @overload
    def __get__(
        self: Attr[Any, Any, Public], instance: None, owner: type[ParT2]
    ) -> Col[ValT, WriteT, Static, ParT2, BaseIdx]: ...

    @overload
    def __get__(self, instance: ParT2, owner: type[ParT2]) -> ValT: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[ParT2] | None
    ) -> Col[ValT, WriteT, Static, ParT2, BaseIdx] | ValT | Self:
        if owner is not None and issubclass(owner, Record):
            if isinstance(instance, Record):
                if (
                    self.pub_status is Public
                    and instance._connected
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

            if instance is None and self.pub_status is Public:
                return Col(
                    _table=owner._stat_table,
                    _attr=cast(Attr[ValT, WriteT, Public], self),
                )

        return self

    def __set__(
        self: Attr[ValT2, RW, Any], instance: Record, value: ValT2 | type[Keep]
    ) -> None:
        """Set the value of the column."""
        if value is Keep:
            return

        if self.setter is not None:
            self.setter(instance, cast(ValT2, value))
        else:
            instance.__dict__[self.name] = value

        if self.pub_status is Public and instance._connected:
            instance._table._mutate(instance)

        return


@dataclass(eq=False)
class AttrSet(Prop[ValT, WriteT], Generic[ValT, IdxT, WriteT]):
    """Record attribute set."""

    @overload
    def __get__(
        self, instance: None, owner: type[ParT2]
    ) -> Col[ValT, WriteT, Static, ParT2, IdxT, Full]: ...

    @overload
    def __get__(
        self, instance: ParT2, owner: type[ParT2]
    ) -> Col[ValT, WriteT, DynBackendID, ParT2, IdxT, Full]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[ParT2] | None
    ) -> Col[ValT, WriteT, Any, ParT2, IdxT, Full] | ValT | Self:
        if owner is not None and issubclass(owner, Record):
            rel_to_owner = Rel(_name="record", _type=owner)
            rec_type = dynamic_record_type(
                owner._default_table_name() + "_" + self.name, [rel_to_owner]
            )

            if isinstance(instance, Record):
                rel_table = RelTable[
                    Any, None, IdxT, WriteT, DynBackendID, Any, Any, Full, ParT2
                ](
                    _db=instance._db,
                    _parent_type=cast(type[ParT2], owner),
                    _rel_prop=BackRel(to=getattr(rec_type, "record")),
                )
                return Col[ValT, WriteT, DynBackendID, ParT2, IdxT, Full](
                    _table=rel_table,
                    _attr=self,
                )

            if instance is None:
                rel_table = RelTable[
                    Any, None, IdxT, WriteT, Static, Any, Any, Full, ParT2
                ](
                    _db=owner._stat_db,
                    _parent_type=cast(type[ParT2], owner),
                    _rel_prop=BackRel(to=getattr(rec_type, "record")),
                )
                return Col[ValT, WriteT, Static, ParT2, IdxT, Full](
                    _table=rel_table,
                    _attr=self,
                )

        return self

    def __set__(
        self: AttrSet[ValT2, Any, RW],
        instance: Record,
        value: ValInput[ValT2, Any] | Col[ValT2, Any, Any, Any, Any] | type[Keep],
    ) -> None:
        """Set the value of the column."""
        if value is Keep:
            return

        instance._table._mutate(instance)
        stat_col: Col[ValT2, RW, Static, Record, IdxT, Full] = getattr(
            type(instance), self.name
        )
        instance._table[stat_col] @= value
        return


type Fks[P: Record, T: Record] = (
    Col[Any, Any, Static, P, BaseIdx, Singular | Nullable]
    | dict[
        Col[Any, Any, Static, P, BaseIdx, Singular | Nullable],
        Col[Any, Any, Static, T, BaseIdx, Singular | Nullable],
    ]
    | list[Col[Any, Any, Static, P, BaseIdx, Singular | Nullable]]
)


@dataclass(kw_only=True, eq=False)
class Rel(
    Prop[RelT, WriteT],
):
    """Relational record set."""

    fks: Fks[Record, Record] | None = None

    index: bool = False
    primary_key: bool = False

    @cached_property
    def target_type(self: Rel[RecT2 | None, Any]) -> type[RecT2]:
        """Target record type of the relation."""
        return cast(type[RecT2], self._type.record_type())

    @cached_property
    def fk_map(
        self,
    ) -> bidict[
        Col[Hashable, Any, Static, Any, BaseIdx, Singular | Nullable],
        Col[Hashable, Any, Static, Any, BaseIdx, Singular | Nullable],
    ]:
        """Map source foreign keys to target cols."""
        match self.fks:
            case dict():
                return bidict(
                    {
                        copy_and_override(
                            type(fk),
                            fk,
                            _table=fk._table,
                            _attr=copy_and_override(
                                type(fk.attr),
                                fk.attr,
                                index=fk.index or self.index,
                                primary_key=fk.primary_key or self.primary_key,
                            ),
                        ): pk
                        for fk, pk in self.fks.items()
                    }
                )
            case Col() | list():
                fks = self.fks if isinstance(self.fks, list) else [self.fks]
                fks = [
                    copy_and_override(
                        type(fk),
                        fk,
                        _table=fk._table,
                        _attr=copy_and_override(
                            type(fk.attr),
                            fk.attr,
                            index=fk.index or self.index,
                            primary_key=fk.primary_key or self.primary_key,
                        ),
                    )
                    for fk in fks
                ]

                pks = self.target_type._pk_cols.values()

                return bidict(dict(zip(fks, pks)))
            case None:
                return bidict()

    @overload
    def __get__(
        self: Rel[RecT2, WriteT],
        instance: None,
        owner: type[ParT2],
    ) -> RelTable[
        RecT2, None, BaseIdx, WriteT, Static, RecT2, None, Singular, ParT2
    ]: ...

    @overload
    def __get__(
        self: Rel[RecT2 | None, WriteT],
        instance: None,
        owner: type[ParT2],
    ) -> RelTable[
        RecT2, None, BaseIdx, WriteT, Static, RecT2, None, Nullable, ParT2
    ]: ...

    @overload
    def __get__(
        self: Rel[RecT2, WriteT],
        instance: ParT2,
        owner: type[ParT2],
    ) -> RecT2: ...

    @overload
    def __get__(
        self: Rel[RecT2 | None, WriteT],
        instance: ParT2,
        owner: type[ParT2],
    ) -> RecT2 | None: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self: Rel[RecT2 | None, Any],
        instance: object | None,
        owner: type | type[ParT2] | None,
    ) -> (
        RelTable[
            RecT2,
            None,
            BaseIdx,
            WriteT,
            Static,
            RecT2,
            None,
            Singular | Nullable,
            ParT2,
        ]
        | Record
        | None
        | Rel[Any, Any]
    ):
        if owner is not None and issubclass(owner, Record):
            if isinstance(instance, Record):
                self_ref = cast(
                    RelTable[
                        RecT2,
                        None,
                        BaseIdx,
                        WriteT,
                        Static,
                        RecT2,
                        None,
                        Singular | Nullable,
                        ParT2,
                    ],
                    getattr(owner, self.name),
                )
                return instance._db[type(instance)][self_ref].get(instance._index)

            if instance is None:
                return RelTable(
                    _db=owner._stat_db,
                    _parent_type=owner,
                    _rel_prop=self,
                )

        return self

    def __set__(  # noqa: D105
        self: Rel[RecT2, RW],
        instance: ParT2,
        value: (
            RelTable[RecT2, Any, Any, Any, Any, ParT2, RecT2, Singular | Nullable]
            | RecT2
            | Hashable
            | type[Keep]
        ),
    ) -> None:
        if value is Keep:
            return

        owner = type(instance)
        stat_rel: RelTable[
            RecT2, None, BaseIdx, RW, Static, RecT2, None, Singular, ParT2
        ] = getattr(owner, self.name)
        instance._table[stat_rel]._mutate(value)
        instance._update_dict()


@dataclass(kw_only=True)
class RelSet(
    Prop[Iterable[RecT], WriteT],
    Generic[RecT, LnT, IdxT, WriteT],
):
    """Relational record set."""

    map_by: Col[Any, Any, Static, Any] | None = None

    @cached_property
    def target_type(self) -> type[RecT]:
        """Get the record type."""
        return cast(type[RecT], self._type.record_type())

    @property
    def link_type(self) -> type[LnT]:
        """Get the link record type."""
        return cast(
            type[LnT],
            self._type.link_type(),
        )

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[ParT2],
    ) -> RelTable[RecT, LnT, IdxT, WriteT, Static, RecT, None, Full, ParT2]: ...

    @overload
    def __get__(
        self,
        instance: ParT2,
        owner: type[ParT2],
    ) -> RelTable[RecT, LnT, IdxT, WriteT, DynBackendID, RecT, Record, Full, ParT2]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105 # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        instance: object | None,
        owner: type | type[ParT2] | None,
    ) -> (
        RelTable[RecT, LnT, IdxT, WriteT, Any, RecT, Record | None, Full, ParT2]
        | Record
        | None
        | Self
    ):
        if owner is not None and issubclass(owner, Record):
            if isinstance(instance, Record):
                return RelTable[
                    RecT, LnT, IdxT, WriteT, Any, RecT, Record | None, Full, ParT2
                ](
                    _db=instance._db,
                    _parent_type=cast(type[ParT2], owner),
                    _rel_prop=self,
                )

            if instance is None:
                return RelTable(
                    _db=owner._stat_db,
                    _parent_type=owner,
                    _rel_prop=self,
                )

        return self

    def __set__(  # noqa: D105
        self,
        instance: ParT2,
        value: (
            RelTable[RecT2, LnT, IdxT, Any, Any, ParT2, RecT2]
            | RecInput[RecT2, KeyT2 | KeyT2, KeyT2]
            | type[Keep]
        ),
    ) -> None:
        if value is Keep:
            return

        stat_rel: RelTable[RecT, LnT, IdxT, RW, Static, RecT, None, Full, ParT2] = (
            getattr(type(instance), self.name)
        )
        instance._table[stat_rel]._mutate(value)


type BackLink[P: Record, T: Record] = (
    RelTable[P, None, BaseIdx, R, Static, P, None, Singular | Nullable, T]
)


@dataclass(kw_only=True)
class BackRel(
    RelSet[RecT, None, IdxT, WriteT],
):
    """Relational record set."""

    to: BackLink[Record, RecT]
    map_by: Col[Any, Any, Static, RecT] | None = None


class RecordMeta(type):
    """Metaclass for record types."""

    _record_superclasses: list[type[Record]]
    _class_props: dict[str, Prop[Any, Any]]

    _is_root_class: bool = False
    _template: bool = False
    _src_mod: ModuleType | None = None
    _derivate: bool = False

    def __init__(cls, name, bases, dct):
        """Initialize a new record type."""
        super().__init__(name, bases, dct)

        if "_src_mod" not in cls.__dict__:
            cls._src_mod = getmodule(cls if not cls._derivate else bases[0])

        prop_defs = {
            name: pt
            for name, hint in get_annotations(cls).items()
            if issubclass((pt := PropType(hint, ctx=cls._src_mod)).prop_type(), Prop)
            or (
                isinstance(hint, str)
                and ("Attr" in hint or "Col" in hint or "Rel" in hint)
            )
        }

        for prop_name, prop_type in prop_defs.items():
            if prop_name not in cls.__dict__:
                pt = prop_type.prop_type()
                assert issubclass(pt, Attr | RelTable)
                setattr(
                    cls,
                    prop_name,
                    pt(_name=prop_name, _type=prop_type),
                )
            else:
                prop = cls.__dict__[prop_name]
                assert isinstance(prop, Prop)
                prop._type = prop_type

        cls._record_superclasses = []
        super_types: Iterable[type | GenericProtocol] = (
            cls.__dict__["__orig_bases__"]
            if "__orig_bases__" in cls.__dict__
            else cls.__bases__
        )
        for c in super_types:
            orig = get_origin(c) if not isinstance(c, type) else c
            if orig is not c and hasattr(orig, "__parameters__"):
                typevar_map = dict(zip(getattr(orig, "__parameters__"), get_args(c)))
            else:
                typevar_map = {}

            if not isinstance(orig, RecordMeta) or orig._is_root_class:
                continue

            if orig._template or cls._derivate:
                for prop_name, prop in orig._class_props.items():
                    prop_defs[prop_name] = prop._type
                    if prop_name not in cls.__dict__:
                        prop = copy_and_override(
                            type(prop),
                            prop,
                            _type=copy_and_override(
                                type(prop._type),
                                prop._type,
                                typevar_map=typevar_map,
                                ctx=cls._src_mod,
                            ),
                        )
                        setattr(cls, prop_name, prop)
            else:
                assert orig is c
                cls._record_superclasses.append(orig)

        cls._class_props = (
            {name: getattr(cls, name) for name in prop_defs.keys()}
            if not cls._is_root_class
            else {}
        )

    @property
    def _props(
        cls: type[RecT2],  # pyright: ignore[reportGeneralTypeIssues]
    ) -> dict[str, Prop[Any, Any]]:
        """The statically defined properties of this record type."""
        return reduce(
            lambda x, y: {**x, **y},
            (c._props for c in cls._record_superclasses),
            cls._class_props,
        )

    @property
    def _attrs(
        cls: type[RecT2],  # pyright: ignore[reportGeneralTypeIssues]
    ) -> dict[str, Attr[Any, Any]]:
        return {k: c for k, c in cls._props.items() if isinstance(c, Attr)}

    @property
    def _attr_sets(
        cls: type[RecT2],  # pyright: ignore[reportGeneralTypeIssues]
    ) -> dict[str, AttrSet[Any, Any, Any]]:
        return {k: c for k, c in cls._props.items() if isinstance(c, AttrSet)}

    @property
    def _rels(
        cls: type[RecT2],  # pyright: ignore[reportGeneralTypeIssues]
    ) -> dict[str, Rel[Any, Any]]:
        return {k: r for k, r in cls._props.items() if isinstance(r, Rel)}

    @property
    def _rel_sets(
        cls: type[RecT2],  # pyright: ignore[reportGeneralTypeIssues]
    ) -> dict[str, RelSet[Any, Any, Any, Any]]:
        return {k: r for k, r in cls._props.items() if isinstance(r, RelSet)}

    @property
    def _rel_types(
        cls: type[RecT2],  # pyright: ignore[reportGeneralTypeIssues]
    ) -> set[type[Record]]:
        return {rel.target_type for rel in cls._rels.values()} | {
            rel_set.target_type for rel_set in cls._rel_sets.values()
        }

    @property
    def _stat_db(cls) -> DB[Any, Static]:
        return DB[RO, Static](types={cls}, b_id=Local.static)

    @property
    def _stat_table(
        cls: type[RecT2],  # pyright: ignore[reportGeneralTypeIssues]
    ) -> Table[RecT2, Any, Static, None, Any, RecT2]:
        return cls._stat_db[cls]


@dataclass_transform(
    kw_only_default=True, field_specifiers=(Attr, Rel, RelSet), eq_default=False
)
class Record(Generic[KeyT], metaclass=RecordMeta):
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
                    p.default,
                    p.default_factory,  # pyright: ignore[reportArgumentType]
                    p.init,
                    hash=isinstance(p, Col) and p.primary_key,
                    repr=True,
                    metadata={},
                    compare=True,
                    kw_only=True,
                )
                for name, p in cls._attrs.items()
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
                for name, p in cls._attr_sets.items()
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
                for name, p in cls._rels.items()
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
    def _backrels_to_rels(
        cls, target: type[RecT2]
    ) -> set[RelTable[Self, Any, Singular, Any, Static, Self, Any, Any, RecT2]]:
        """Get all direct relations from a target record type to this type."""
        rels: set[RelTable[Self, Any, Singular, Any, Static, Self, Any, Any, RecT2]] = (
            set()
        )
        for rel in cls._rels.values():
            if isinstance(rel, BackRel):
                rels.add(
                    RelTable[Self, Any, Singular, Any, Static, Self, Any, Any, RecT2](
                        _parent_type=target,
                        _rel_prop=cast(Rel[Self], rel.to.rel_prop),
                    )
                )

        return rels

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

    _db: Attr[DB[RW, Any], RW, Private] = Attr(
        pub_status=Private, default_factory=lambda: DB(b_id=Local.dynamic)
    )
    _connected: Attr[bool, RW, Private] = Attr(pub_status=Private, default=False)
    _root: Attr[bool, RW, Private] = Attr(pub_status=Private, default=True)
    _index: Attr[KeyT, RW, Private] = Attr(pub_status=Private, init=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record instance."""
        super().__init__()

        cls = type(self)

        attrs = {name: val for name, val in kwargs.items() if name in cls._attrs}
        direct_rels = {name: val for name, val in kwargs.items() if name in cls._rels}
        attr_sets = {
            name: val for name, val in kwargs.items() if name in cls._attr_sets
        }
        indirect_rels = {
            name: val for name, val in kwargs.items() if name in cls._rel_sets
        }

        # First set all attributes.
        for name, value in attrs.items():
            setattr(self, name, value)

        # Then set all direct relations.
        for name, value in direct_rels.items():
            setattr(self, name, value)

        # Then all attribute sets.
        for name, value in attr_sets.items():
            setattr(self, name, value)

        # Finally set all indirect relations.
        for name, value in indirect_rels.items():
            setattr(self, name, value)

        self.__post_init__()

        pks = self._table._pk_cols
        if len(pks) == 1:
            self._index = getattr(self, next(iter(pks)))
        else:
            self._index = cast(KeyT, tuple(getattr(self, pk) for pk in pks))

        return

    @cached_property
    def _table(self) -> Table[Self, RW, Any, Any, Singular, Self]:
        if not self._connected:
            self._db[type(self)] |= self
            self._connected = True
        return self._db[type(self)][self._index]

    def __post_init__(self) -> None:  # noqa: D105
        pass

    def __hash__(self) -> int:
        """Identify the record by database and id."""
        return gen_int_hash((self._db if self._connected else None, self._index))

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)

    @overload
    def _to_dict(
        self,
        name_keys: Literal[False] = ...,
        include: set[type[Prop]] | None = ...,
        with_fks: bool = ...,
    ) -> dict[Prop, Any]: ...

    @overload
    def _to_dict(
        self,
        name_keys: Literal[True],
        include: set[type[Prop]] | None = ...,
        with_fks: bool = ...,
    ) -> dict[str, Any]: ...

    def _to_dict(
        self,
        name_keys: bool = True,
        include: set[type[Prop]] | None = None,
        with_fks: bool = True,
    ) -> dict[Prop, Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Prop], ...] = (
            tuple(include) if include is not None else (Attr,)
        )

        vals = {
            p if not name_keys else p.name: getattr(self, p.name)
            for p in (type(self)._props if with_fks else type(self)._props).values()
            if isinstance(p, include_types)
        }

        return cast(dict, vals)

    @classmethod
    def _is_complete_dict(
        cls,
        data: Mapping[Prop[Any, Any], Any] | Mapping[str, Any],
    ) -> bool:
        """Check if dict data contains all required info for record type."""
        data = {(p if isinstance(p, str) else p.name): v for p, v in data.items()}
        return all(
            c.attr.name in data
            or c.attr.getter is None
            or c.attr.default is Undef
            or c.attr.default_factory is None
            for c in cls._stat_table._data_cols.values()
            if c.attr.init is not False
        ) and all(
            r.name in data or all(fk.name in data for fk in r.fk_map.keys())
            for r in cls._rels.values()
        )

    def _update_dict(self) -> None:
        table = self._db[type(self)][self._index]
        df = table.to_df()
        rec_dict = list(df.iter_rows(named=True))[0]
        self.__dict__.update(rec_dict)


@dataclass(frozen=True)
class RelTree(Generic[*RecTt]):
    """Tree of relations starting from the same root."""

    rels: Iterable[RelTable[Record, Any, Any, Any, Any, Any, Any, Any, Any]] = field(
        default_factory=set
    )

    def __post_init__(self) -> None:  # noqa: D105
        assert all(
            rel._root_set == self.root_set for rel in self.rels
        ), "Relations in set must all start from same root."

    @cached_property
    def root_set(self) -> Table[Record, Any, Any]:
        """Root record type of the set."""
        return list(self.rels)[-1]._root_set

    @cached_property
    def all_rels(self) -> set[RelTable[Record, Any, Any, Any, Any, Any, Any, Any, Any]]:
        """All relations in the tree, including sub-merges."""
        return {rel for rel in self.rels} | {
            sub_rel._prefix(rel)
            for rel in self.rels
            for sub_rel in rel._rel_tree.all_rels
        }

    def prefix(self, prefix: Table[Any, Any, Any, Any, Any]) -> Self:
        """Prefix all relations in the set with given relation."""
        rels = {rel._prefix(prefix) for rel in self.rels}
        return cast(Self, RelTree(rels))

    def __mul__(
        self, other: RelTable[RecT2, Any, Any, Any, Any, Any, Any, Any, Any] | RelTree
    ) -> RelTree[*RecTt, RecT2]:
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    def __rmul__(
        self, other: RelTable[RecT2, Any, Any, Any, Any, Any, Any, Any, Any] | RelTree
    ) -> RelTree[RecT2, *RecTt]:
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*other.rels, *self.rels])

    def __or__(
        self: RelTree[RecT2], other: RelTable[RecT3] | RelTree[RecT3]
    ) -> RelTree[RecT2 | RecT3]:
        """Add more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    @property
    def types(self) -> tuple[*RecTt]:
        """Return the record types in the relation tree."""
        return cast(tuple[*RecTt], tuple(r.target for r in self.rels))


type AggMap[Rec: Record] = dict[Col[Rec, Any], Col | sqla.Function]


@dataclass(kw_only=True, frozen=True)
class Agg(Generic[RecT]):
    """Define an aggregation map."""

    target: type[RecT]
    map: AggMap[RecT]


type JoinDict = dict[RelTable[Any, None, Any, Any, Any, Any, Any, Any, Any], JoinDict]
type SqlJoin = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


@dataclass(kw_only=True, eq=False)
class Table(
    Generic[RecT, WriteT, BackT, SelT, FiltT, RelT],
):
    """Record dataset."""

    @cached_property
    def target(self) -> type[RecT]:
        """Traget record type."""
        return self._target_type

    @cached_property
    def rec(self) -> type[RecT]:
        """Reference props of the target record type."""
        return cast(
            type[RecT],
            type(
                self.target.__name__ + "_" + token_hex(5),
                (self.target,),
                {
                    "_rel": self,
                    "_derivate": True,
                    "_src_mod": getmodule(self.target),
                },
            ),
        )

    def execute[
        *T
    ](
        self,
        stmt: sqla.Select[tuple[*T]] | sqla.Insert | sqla.Update | sqla.Delete,
    ) -> sqla.Result[tuple[*T]]:
        """Execute a SQL statement in this database's context."""
        stmt = self._parse_expr(stmt)
        with self._db.engine.begin() as conn:
            return conn.execute(self._parse_expr(stmt))

    # Overloads: attribute selection:

    # 1. DB-level type selection
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Any],
        key: type[RecT3],
    ) -> Table[RecT3, WriteT, BackT, SelT, FiltT, RecT3]: ...

    # 2. Top-level attribute selection, rel parent, base-idx col
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, KeyT2, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, RecT2, BaseIdx, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, KeyT2, FiltT2]: ...

    # 3. Top-level attribute selection, rel parent, plural col, both tuples
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, tuple, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, RecT2, tuple, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple, FiltT2]: ...

    # 4. Top-level attribute selection, rel parent, plural col, left tuple
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, tuple[*KeyTt], Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, RecT2, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple[*KeyTt, KeyT3], FiltT2]: ...

    # 5. Top-level attribute selection, rel parent, plural col, right tuple
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, KeyT2, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, RecT2, tuple[*KeyTt], FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple[KeyT2, *KeyTt], FiltT2]: ...

    # 6. Top-level attribute selection, rel parent, plural col, no tuple
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, KeyT2, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, RecT2, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple[KeyT2, KeyT3], FiltT2]: ...

    # 7. Top-level attribute selection, base parent, singular col
    @overload
    def __getitem__(
        self: Table[RecT2, Any, Any, Any, Any, Record[KeyT2]],
        key: Col[ValT3, WriteT3, Static, RecT2, Any, SingleT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, BaseIdx, SingleT2]: ...

    # 8. Top-level attribute selection, base parent, plural col, both tuples
    @overload
    def __getitem__(
        self: Table[RecT2, Any, Any, Any, Any, Record[tuple]],
        key: Col[ValT3, WriteT3, Static, RecT2, tuple, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple, FiltT2]: ...

    # 9. Top-level attribute selection, base parent, plural col, left tuple
    @overload
    def __getitem__(
        self: Table[RecT2, Any, Any, Any, Any, Record[tuple[*KeyTt]]],
        key: Col[ValT3, WriteT3, Static, RecT2, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple[*KeyTt, KeyT3], FiltT2]: ...

    # 10. Top-level attribute selection, base parent, plural col, right tuple
    @overload
    def __getitem__(
        self: Table[RecT2, Any, Any, Any, Any, Record[KeyT2]],
        key: Col[ValT3, WriteT3, Static, RecT2, tuple[*KeyTt], FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple[KeyT2, *KeyTt], FiltT2]: ...

    # 11. Top-level attribute selection, base parent, plural col, no tuple
    @overload
    def __getitem__(
        self: Table[RecT2, Any, Any, Any, Any, Record[KeyT2]],
        key: Col[ValT3, WriteT3, Static, RecT2, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, RecT2, tuple[KeyT2, KeyT3], FiltT2]: ...

    # 12. Nested attribute selection, rel parent, plural col, both tuples or base
    @overload
    def __getitem__(
        self: RelTable[Any, Any, tuple, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, Any, tuple | BaseIdx, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, tuple, FiltT2]: ...

    # 13. Nested attribute selection, rel parent, plural col, left tuple
    @overload
    def __getitem__(
        self: RelTable[Any, Any, tuple, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, Any, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, IdxStartEnd[Any, KeyT3], FiltT2]: ...

    # 14. Nested attribute selection, rel parent, plural col, right base or tuple
    @overload
    def __getitem__(
        self: RelTable[Any, Any, KeyT2, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, Any, tuple | BaseIdx, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, KeyT2 | IdxStartEnd[KeyT2, Any], FiltT2]: ...

    # 15. Nested attribute selection, rel parent, plural col, no tuple
    @overload
    def __getitem__(
        self: RelTable[Any, Any, KeyT2, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, Any, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, IdxStartEnd[KeyT2, KeyT3], FiltT2]: ...

    # 16. Nested attribute selection, base parent, singular col
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Record[KeyT2]],
        key: Col[ValT3, WriteT3, Static, Any, Any, SingleT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, BaseIdx, SingleT2]: ...

    # 17. Nested attribute selection, base parent, plural col, both tuples
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Record[tuple]],
        key: Col[ValT3, WriteT3, Static, Any, tuple, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, tuple, FiltT2]: ...

    # 18. Nested attribute selection, base parent, plural col, left tuple
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Record[tuple]],
        key: Col[ValT3, WriteT3, Static, Any, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, IdxStartEnd[Any, KeyT3], FiltT2]: ...

    # 19. Nested attribute selection, base parent, plural col, right tuple
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Record[KeyT2]],
        key: Col[ValT3, WriteT3, Static, Any, tuple, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, IdxStartEnd[KeyT2, Any], FiltT2]: ...

    # 20. Nested attribute selection, base parent, plural col, no tuple
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Record[KeyT2]],
        key: Col[ValT3, WriteT3, Static, Any, KeyT3, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, IdxStartEnd[KeyT2, KeyT3], FiltT2]: ...

    # 21. Default attribute selection
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Any],
        key: Col[ValT3, WriteT3, Static, Any, Any, FiltT2],
    ) -> Col[ValT3, WriteT3, BackT, Any, Any, FiltT2]: ...

    # Overloads: relation selection:

    # 22. Top-level relation selection, rel parent, left base key, right base key
    @overload
    def __getitem__(
        self: RelTable[
            RecT2, Any, BaseIdx, Any, Any, Record[KeyT2], Record | None, Any
        ],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            RecT2,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 23. Top-level relation selection, rel parent, left base key, right tuple
    @overload
    def __getitem__(
        self: RelTable[
            RecT2, Any, BaseIdx, Any, Any, Record[KeyT2], Record | None, Any
        ],
        key: RelTable[
            RecT3, LnT3, tuple[*KeyTt], WriteT3, Static, RecT3, SelT3, FiltT3, RecT2
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, *KeyTt],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 24. Top-level relation selection, rel parent, left base key, right single key
    @overload
    def __getitem__(
        self: RelTable[
            RecT2, Any, BaseIdx, Any, Any, Record[KeyT2], Record | None, Any
        ],
        key: RelTable[RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, RecT2],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 25. Top-level relation selection, rel parent, left tuple, right base key
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, tuple[*KeyTt], Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            RecT2,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[*KeyTt, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 26. Top-level relation selection, rel parent, left tuple, right tuple
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, tuple, Any, Any, Any, Record | None, Any],
        key: RelTable[RecT3, LnT3, tuple, WriteT3, Static, RecT3, SelT3, FiltT3, RecT2],
    ) -> RelTable[RecT3, LnT3, tuple, WriteT3, BackT, RecT3, SelT3, FiltT3, RecT2]: ...

    # 27. Top-level relation selection, rel parent, left tuple, right single key
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, tuple[*KeyTt], Any, Any, Any, Record | None, Any],
        key: RelTable[RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, RecT2],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[*KeyTt, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 28. Top-level relation selection, rel parent, left single key, right base key
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, KeyT2, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            RecT2,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 29. Top-level relation selection, rel parent, left single key, right tuple
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, KeyT2, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3, LnT3, tuple[*KeyTt], WriteT3, Static, RecT3, SelT3, FiltT3, RecT2
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, *KeyTt],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 30. Top-level relation selection, rel parent, left single key, right single key
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, KeyT2, Any, Any, Any, Record | None, Any],
        key: RelTable[RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, RecT2],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 31. Nested relation selection, rel parent, left base key, right base key
    @overload
    def __getitem__(
        self: RelTable[Any, Any, BaseIdx, Any, Any, Record[KeyT2], Record | None, Any],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            Record,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 32. Nested relation selection, rel parent, left base key, right tuple
    @overload
    def __getitem__(
        self: RelTable[Any, Any, BaseIdx, Any, Any, Record[KeyT2], Record | None, Any],
        key: RelTable[
            RecT3,
            LnT3,
            tuple[*KeyTt],
            WriteT3,
            Static,
            RecT3,
            SelT3,
            FiltT3,
            Record,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[KeyT2, Any],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 33. Nested relation selection, rel parent, left base key, right single key
    @overload
    def __getitem__(
        self: RelTable[Any, Any, BaseIdx, Any, Any, Record[KeyT2], Record | None, Any],
        key: RelTable[
            RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, Record
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 34. Nested relation selection, rel parent, left tuple, right base key
    @overload
    def __getitem__(
        self: RelTable[Any, Any, tuple, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            Record,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[Any, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 35. Nested relation selection, rel parent, left tuple, right tuple
    @overload
    def __getitem__(
        self: RelTable[Any, Any, tuple, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3, LnT3, tuple, WriteT3, Static, RecT3, SelT3, FiltT3, Record
        ],
    ) -> RelTable[RecT3, LnT3, tuple, WriteT3, BackT, RecT3, SelT3, FiltT3, Record]: ...

    # 36. Nested relation selection, rel parent, left tuple, right single key
    @overload
    def __getitem__(
        self: RelTable[Any, Any, tuple, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, Record
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[Any, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 37. Nested relation selection, rel parent, left single key, right base key
    @overload
    def __getitem__(
        self: RelTable[Any, Any, KeyT2, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            Record,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 38. Nested relation selection, rel parent, left single key, right tuple
    @overload
    def __getitem__(
        self: RelTable[Any, Any, KeyT2, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3, LnT3, tuple, WriteT3, Static, RecT3, SelT3, FiltT3, Record
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[KeyT2, Any],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 39. Nested relation selection, rel parent, left single key, right single key
    @overload
    def __getitem__(
        self: RelTable[Any, Any, KeyT2, Any, Any, Any, Record | None, Any],
        key: RelTable[
            RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, Record
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        IdxStartEnd[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Record,
    ]: ...

    # 40. Top-level relation selection, base parent, right base key
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Record | None, Any, RecT2],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            RecT2,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 41. Top-level relation selection, base parent, right tuple
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Record | None, Any, RecT2],
        key: RelTable[
            RecT3, LnT3, tuple[*KeyTt], WriteT3, Static, RecT3, SelT3, FiltT3, RecT2
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, *KeyTt],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 42. Top-level relation selection, base parent, right single key
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Record | None, Any, RecT2],
        key: RelTable[RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, RecT2],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        RecT2,
    ]: ...

    # 43. Nested relation selection, base parent, right base key
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Record | None, Any, Any],
        key: RelTable[
            RecT3,
            LnT3,
            BaseIdx,
            WriteT3,
            Static,
            Record[KeyT3],
            SelT3,
            FiltT3,
            Any,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Any,
    ]: ...

    # 44. Nested relation selection, base parent, right tuple
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Record | None, Any, Any],
        key: RelTable[
            RecT3,
            LnT3,
            tuple[*KeyTt],
            WriteT3,
            Static,
            RecT3,
            SelT3,
            FiltT3,
            Any,
        ],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, *KeyTt],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Any,
    ]: ...

    # 45. Nested relation selection, base parent, right single key
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Record | None, Any, Any],
        key: RelTable[RecT3, LnT3, KeyT3, WriteT3, Static, RecT3, SelT3, FiltT3, Any],
    ) -> RelTable[
        RecT3,
        LnT3,
        tuple[KeyT2, KeyT3],
        WriteT3,
        BackT,
        RecT3,
        SelT3,
        FiltT3,
        Any,
    ]: ...

    # 46. Default relation selection
    @overload
    def __getitem__(
        self: Table[Any, Any, Any, Any, Any, Any],
        key: RelTable[
            RecT3,
            LnT3,
            Any,
            WriteT3,
            Static,
            RecT3,
            SelT3,
            FiltT3,
            Any,
        ],
    ) -> RelTable[RecT3, LnT3, tuple, WriteT3, BackT, RecT3, SelT3, FiltT3, Record]: ...

    # Index filtering and selection:

    # 47. RelSet: Merge selection
    @overload
    def __getitem__(
        self: RelTable[RecT2, Any, KeyT2, Any, Any, Any, Record | None],
        key: RelTree[RecT2, *RecTt],
    ) -> RelTable[
        RecT2, LnT, IdxT, WriteT, BackT, RecT2, tuple[RecT2, *RecTt], FiltT, ParT
    ]: ...

    # 48. RelSet: Expression / key list / slice filtering
    @overload
    def __getitem__(
        self: RelTable[
            Record[KeyT2], Any, KeyT3 | BaseIdx, Any, Any, Any, Any, Full | Filtered
        ],
        key: (
            sqla.ColumnElement[bool]
            | Iterable[KeyT2 | KeyT3]
            | slice
            | tuple[slice, ...]
        ),
    ) -> RelTable[RecT, LnT, IdxT, WriteT, BackT, RecT, SelT, Filtered, ParT]: ...

    # 49. RelSet: Index value selection, no record loading
    @overload
    def __getitem__(
        self: RelTable[
            Record[KeyT2], Any, KeyT3 | BaseIdx, Any, Any, Any, None, Full | Filtered
        ],
        key: KeyT2 | KeyT3,
    ) -> RelTable[RecT, LnT, IdxT, WriteT, BackT, RecT, SelT, Singular, ParT]: ...

    # 50. RelSet: Index value selection, record loading
    @overload
    def __getitem__(
        self: RelTable[
            Record[KeyT2], Any, KeyT3 | BaseIdx, Any, Any, Any, Record, Full | Filtered
        ],
        key: KeyT2 | KeyT3,
    ) -> RecT: ...

    # 51. Merge selection
    @overload
    def __getitem__(
        self: Table[RecT2, Any, Any, Record | None, Any],
        key: RelTree[RecT2, *RecTt],
    ) -> Table[RecT2, WriteT, BackT, tuple[RecT2, *RecTt], FiltT, RecT2]: ...

    # 52. Expression / key list / slice filtering
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Any, Full | Filtered],
        key: sqla.ColumnElement[bool] | Iterable[KeyT2] | slice | tuple[slice, ...],
    ) -> Table[RecT, WriteT, BackT, SelT, Filtered, RecT]: ...

    # 53. Index value selection, no record loading
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, None, Full | Filtered],
        key: KeyT2,
    ) -> Table[RecT, WriteT, BackT, SelT, Singular, RecT]: ...

    # 54. Index value selection, record loading
    @overload
    def __getitem__(
        self: Table[Record[KeyT2], Any, Any, Record, Full | Filtered],
        key: KeyT2,
    ) -> RecT: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self: Table[Record, Any, Any, Any, Any, Any],
        key: (
            type[Record]
            | Col[Any, Any, Static, Any, Any, Any]
            | RelTable[Any, Any, Any, Any, Static, Any, Any, Any, Any]
            | RelTree
            | sqla.ColumnElement[bool]
            | list[Hashable]
            | slice
            | tuple[slice, ...]
            | Hashable
        ),
    ) -> (
        Col[Any, Any, Any, Any, Any, Any]
        | Table[Record, Any, Any, Any, Any, Any]
        | Record
    ):
        match key:
            case type():
                assert issubclass(
                    key,
                    self.target,
                )
                return copy_and_override(Table[key], self, _target_type=key)
            case Col():
                if isinstance(key._table, RelTable) and (
                    not isinstance(self, RelTable)
                    or (key.db_table._to_static() != self._to_static())
                ):
                    return self._suffix(key._table)[key]

                return Col(_table=self, _attr=key.attr)
            case RelTable():
                return self._suffix(key)
            case RelTree():
                return copy_and_override(
                    type(self),
                    self,
                    _target_type=self.target,
                    _sel_merge=self._sel_merge * key.prefix(self),
                )
            case sqla.ColumnElement():
                return copy_and_override(
                    type(self),
                    self,
                    _target_type=self.target,
                    _sql_filt=[*self._sql_filt, key],
                )
            case list() | slice() | tuple() | Hashable():
                if not isinstance(key, list | slice) and not has_type(
                    key, tuple[slice, ...]
                ):
                    assert (
                        self._single_key is None or key == self._single_key
                    ), "Cannot select multiple single record keys"

                key_set = copy_and_override(
                    type(self),
                    self,
                    _target_type=self.target,
                    _key_filt=[*self._key_filt, key],
                )

                if not is_subtype(self._sel_type, Record):
                    return key_set

                try:
                    return list(iter(key_set))[0]
                except IndexError as e:
                    raise KeyError(key) from e

    @overload
    def get(
        self: Table[Any, Any, Any, Any, Singular | Nullable],
        *,
        key: None = ...,
        default: ValT2,
    ) -> RecT | ValT2: ...

    @overload
    def get(
        self: Table[Any, Any, Any, Any, Singular | Nullable],
        key: None = ...,
        default: None = ...,
    ) -> RecT | None: ...

    @overload
    def get(
        self: RelTable[
            Record[KeyT2], Any, KeyT3 | BaseIdx, Any, Any, Any, Record, Full | Filtered
        ],
        key: KeyT2 | KeyT3,
        default: ValT2,
    ) -> RecT | ValT2: ...

    @overload
    def get(
        self: RelTable[
            Record[KeyT2], Any, KeyT3 | BaseIdx, Any, Any, Any, Record, Full | Filtered
        ],
        key: KeyT2 | KeyT3,
        default: None = ...,
    ) -> RecT | None: ...

    @overload
    def get(
        self: Table[Record[KeyT2], Any, Any, Any],
        key: KeyT2,
        default: ValT2,
    ) -> RecT | ValT2: ...

    @overload
    def get(
        self: Table[Record[KeyT2], Any, Any, Any],
        key: KeyT2,
        default: None = ...,
    ) -> RecT | None: ...

    def get(
        self: Table[Any, Any, Any, Any, Any],
        key: Hashable | None = None,
        default: ValT2 | None = None,
    ) -> Record | ValT2 | None:
        """Get a record by key."""
        try:
            return (self[key] if key is not None else self).values()[0]
        except KeyError | IndexError:
            return default

    def select(
        self,
        *,
        index_only: bool = False,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        return (
            sqla.select(self._sql_query)
            if not index_only
            else sqla.select(*(col for col in self._idx_cols.values())).select_from(
                self._sql_query
            )
        )

    @overload
    def to_df(
        self: Table[Any, Any, Any, tuple, Any],
        kind: type[DfT],
        index_only: Literal[False] = ...,
    ) -> tuple[DfT, ...]: ...

    @overload
    def to_df(
        self: Table[Any, Any, Any, Record | None, Any],
        kind: type[DfT],
        index_only: bool = ...,
    ) -> DfT: ...

    @overload
    def to_df(
        self: Table[Any, Any, Any, tuple, Any],
        kind: None = ...,
        index_only: Literal[False] = ...,
    ) -> tuple[pl.DataFrame, ...]: ...

    @overload
    def to_df(
        self: Table[Any, Any, Any, Record | None, Any],
        kind: None = ...,
        index_only: bool = ...,
    ) -> pl.DataFrame: ...

    @overload
    def to_df(
        self: Table[Any, Any, Any, Any, Any],
        kind: None = ...,
        index_only: Literal[False] = ...,
    ) -> pl.DataFrame | tuple[pl.DataFrame, ...]: ...

    def to_df(
        self: Table[Record, Any, Any, Any, Any],
        kind: type[DfT] | None = None,
        index_only: bool = False,
    ) -> DfT | tuple[DfT, ...]:
        """Download selection."""
        select = self.select(index_only=index_only)

        idx_cols = list(self._idx_cols.keys())

        merged_df = None
        if kind is pd.DataFrame:
            with self._db.engine.connect() as con:
                merged_df = pd.read_sql(select, con)
                merged_df = merged_df.set_index(idx_cols)
        else:
            merged_df = pl.read_database(
                str(select.compile(self._db.engine)), self._db.engine
            )

        if index_only:
            return cast(DfT, merged_df)

        main_prefix = self.target._default_table_name() + "."
        main_cols = {
            col: col[len(main_prefix) :]
            for col in select.columns.keys()
            if col.startswith(main_prefix)
        }

        extra_cols = {
            rel: {
                col: col[len(rel._path_str) + 1 :]
                for col in select.columns.keys()
                if col.startswith(rel._path_str)
            }
            for rel in self._sel_merge.rels
        }

        main_df, *extra_dfs = cast(
            tuple[DfT, ...],
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
    def keys(
        self: RelTable[Any, Any, KeyT3, Any, Any, Any, Any, Any],
    ) -> Sequence[KeyT3]: ...

    @overload
    def keys(
        self: Table[Record[KeyT2], Any, Any, Any, Any],
    ) -> Sequence[KeyT2]: ...

    def keys(
        self: Table[Any, Any, Any, Any, Any],
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        df = self.to_df(index_only=True)
        if len(self._idx_cols) == 1:
            return [tup[0] for tup in df.iter_rows()]

        return list(df.iter_rows())

    @overload
    def values(self: Table[Record, Any, Any, Record | None, Any]) -> list[RecT]: ...

    @overload
    def values(
        self: Table[Record, Any, Any, tuple[*RecTt], Any]
    ) -> list[tuple[RecT, *tuple[Record, ...]]]: ...

    def values(  # noqa: D102
        self: Table[Record, Any, Any, Any, Any],
    ) -> list[RecT2] | list[tuple[RecT2, *tuple[Record, ...]]]:
        dfs = self.to_df()
        if isinstance(dfs, pl.DataFrame):
            dfs = (dfs,)

        rec_types: list[type[Record]] = [self.target, *self._sel_merge.types]

        valid_caches = {rt: self._db._get_valid_cache_set(rt) for rt in rec_types}
        instance_maps = {rt: self._db._get_instance_map(rt) for rt in rec_types}

        recs = []
        for rows in zip(*(df.iter_rows(named=True) for df in dfs)):
            rows = cast(tuple[dict[str, Any], ...], rows)

            rec_list = []
            for rec_type, row in zip(rec_types, rows):
                new_rec = self.target(**row)

                if new_rec._index in valid_caches[rec_type]:
                    rec = instance_maps[rec_type][new_rec._index]
                else:
                    rec = new_rec
                    rec._db = self._db
                    valid_caches[rec_type].add(rec._index)
                    instance_maps[rec_type][rec._index] = rec

                rec_list.append(rec)

            recs.append(tuple(rec_list) if len(rec_list) > 1 else rec_list[0])

        return recs

    @overload
    def __iter__(
        self: Table[RecT2, Any, Any, Record | None, Any]
    ) -> Iterator[RecT2]: ...

    @overload
    def __iter__(
        self: Table[RecT2, Any, Any, tuple[*RecTt], Any]
    ) -> Iterator[tuple[RecT2, *tuple[Record, ...]]]: ...

    def __iter__(  # noqa: D105
        self: Table[RecT2, Any, Any, Any, Any],
    ) -> Iterator[RecT2] | Iterator[tuple[RecT2, *tuple[Record, ...]]]:
        return iter(self.values())

    @overload
    def __imatmul__(
        self: RelTable[Record[KeyT2], Any, KeyT3, RW, Any, Any, Record | None, Any],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT3, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, FiltT, RecT]: ...

    @overload
    def __imatmul__(
        self: Table[Record[KeyT2], RW, Any, Record | None, Any],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, FiltT, RecT]: ...

    def __imatmul__(
        self: Table[Any, RW, Any, Record | None, Any],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, Any, Any],
    ) -> Table[RecT, WriteT, BackT, SelT, FiltT, RecT]:
        """Aligned assignment."""
        self._mutate(other, mode="update")
        return cast(Table[RecT, WriteT, BackT, SelT, FiltT, RecT], self)

    @overload
    def __iand__(
        self: RelTable[Record[KeyT2], Any, KeyT3, RW, Any, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT3, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    @overload
    def __iand__(
        self: Table[Record[KeyT2], RW, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    def __iand__(
        self: Table[Record[KeyT2], RW, Any, Any, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]:
        """Replacing assignment."""
        self._mutate(other, mode="replace")
        return cast(Table[RecT, WriteT, BackT, SelT, Full, RecT], self)

    @overload
    def __ior__(
        self: RelTable[Record[KeyT2], Any, KeyT3, RW, Any, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT3, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    @overload
    def __ior__(
        self: Table[Record[KeyT2], RW, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    def __ior__(
        self: Table[Record[KeyT2], RW, Any, Any, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]:
        """Upserting assignment."""
        self._mutate(other, mode="upsert")
        return cast(Table[RecT, WriteT, BackT, SelT, Full, RecT], self)

    @overload
    def __iadd__(
        self: RelTable[Record[KeyT2], Any, KeyT3, RW, Any, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT3, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    @overload
    def __iadd__(
        self: Table[Record[KeyT2], RW, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    def __iadd__(
        self: Table[Record[KeyT2], RW, Any, Any, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]:
        """Inserting assignment."""
        self._mutate(other, mode="insert")
        return cast(Table[RecT, WriteT, BackT, SelT, Full, RecT], self)

    @overload
    def __isub__(
        self: RelTable[Record[KeyT2], Any, KeyT3, RW, Any, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT3, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    @overload
    def __isub__(
        self: Table[Record[KeyT2], RW, Any, Record | None, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]: ...

    def __isub__(
        self: Table[Record[KeyT2], RW, Any, Any, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> Table[RecT, WriteT, BackT, SelT, Full, RecT]:
        """Deletion."""
        raise NotImplementedError("Delete not supported yet.")

    def __lshift__(
        self: Table[Record[KeyT2], RW, Any, Any, Full],
        other: Table[RecT, Any, Any, Any, Any] | RecInput[RecT, KeyT2, KeyT2],
    ) -> list[KeyT2]:
        """Injection."""
        raise NotImplementedError("Inject not supported yet.")

    def __rshift__(
        self: Table[Record[KeyT2], RW, Any, Any, Full], other: KeyT2 | Iterable[KeyT2]
    ) -> dict[KeyT2, RecT]:
        """Extraction."""
        raise NotImplementedError("Extract not supported yet.")

    # 1. Type deletion
    @overload
    def __delitem__(self: Table[Any, RW, Any, Any, Any], key: type[RecT2]) -> None: ...

    # 2. Filter deletion
    @overload
    def __delitem__(
        self: Table[
            Record[KeyT2],
            RW,
            Any,
            Any,
            Any,
        ],
        key: (
            Iterable[KeyT2]
            | KeyT2
            | slice
            | tuple[slice, ...]
            | sqla.ColumnElement[bool]
        ),
    ) -> None: ...

    # Implementation:

    def __delitem__(  # noqa: D105
        self: Table[Record, RW, Any, Any, Any],
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

        tables = {
            self._db[rec]._get_sql_base_table(mode="upsert")
            for rec in self.target._record_superclasses
        }

        statements = []

        for table in tables:
            # Prepare delete statement.
            if self._db.engine.dialect.name in (
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
                                col == self[self._pk_cols[col.name]]
                                for col in table.primary_key.columns
                            ),
                        )
                    )
                )
            else:
                # Correlated update.
                raise NotImplementedError("Correlated update not supported yet.")

        # Execute delete statements.
        with self._db.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        if self._db.backend_type == "excel-file":
            self._save_to_excel()

    @overload
    def extract(
        *,
        self: Table[Record, Any, Any, Any, Any],
        aggs: Mapping[RelTable, Agg] | None = ...,
        to_backend: None = ...,
        overlay_type: OverlayType = ...,
    ) -> DB[RW, BackT]: ...

    @overload
    def extract(
        *,
        self: Table[Record, Any, Any, Any, Any],
        aggs: Mapping[RelTable, Agg] | None = ...,
        to_backend: Backend[BackT2],
        overlay_type: OverlayType = ...,
    ) -> DB[RW, BackT2]: ...

    def extract(
        self: Table[
            Record,
            Any,
            Any,
            Any,
            Any,
        ],
        aggs: Mapping[RelTable, Agg] | None = None,
        to_backend: Backend[BackT2] | None = None,
        overlay_type: OverlayType = "name_prefix",
    ) -> DB[RW, BackT | BackT2]:
        """Extract a new database instance from the current selection."""
        # Get all rec types in the schema.
        rec_types = {self.target, *self.target._rel_types}

        # Get the entire subdag of this target type.
        all_paths_rels = {
            r
            for rel in self.target._rels.values()
            for r in self[rel]._get_subdag(rec_types)
        }

        # Extract rel paths, which contain an aggregated rel.
        aggs_per_type: dict[type[Record], list[tuple[RelTable, Agg]]] = {}
        if aggs is not None:
            for rel, agg in aggs.items():
                for path_rel in all_paths_rels:
                    if path_rel._is_ancestor(rel):
                        aggs_per_type[rel.parent] = [
                            *aggs_per_type.get(rel.parent, []),
                            (rel, agg),
                        ]
                        all_paths_rels.remove(path_rel)

        replacements: dict[type[Record], sqla.Select] = {}
        for rec in rec_types:
            # For each table, create a union of all results from the direct routes.
            selects = [
                self[rel].select()
                for rel in all_paths_rels
                if issubclass(rec, rel.target)
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
                                self[sa]
                                if isinstance(sa, Col)
                                else sqla_visitors.replacement_traverse(
                                    sa,
                                    {},
                                    replace=lambda element, **kw: (
                                        self[element]
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
        overlay_db = copy_and_override(
            DB[RW, BackT],
            self._db,
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
        if to_backend is not None:
            other_db = copy_and_override(
                DB[RW, BackT2],
                overlay_db,
                b_id=to_backend.b_id,
                url=to_backend.url,
                _def_types={},
                _metadata=sqla.MetaData(),
                _instance_map={},
            )

            for rec in set(replacements) | set(aggregations):
                other_db[rec] &= overlay_db[rec].to_df()

            overlay_db = other_db

        return overlay_db

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        with self._db.engine.connect() as conn:
            res = conn.execute(
                sqla.select(sqla.func.count()).select_from(self.select().subquery())
            ).scalar()
            assert isinstance(res, int)
            return res

    def __contains__(
        self: Table[RecT2, Any, Any, Any, Any], key: Hashable | RecT2
    ) -> bool:
        """Check if a record is in the dataset."""
        return len(self[key._index if isinstance(key, Record) else key]) > 0

    def __clause_element__(self) -> sqla.Subquery:
        """Return subquery for the current selection to be used inside SQL clauses."""
        return self._sql_query

    def __hash__(self: Table[Any, Any, Any, Any, Any]) -> int:
        """Hash the RecSet."""
        return gen_int_hash(
            (self._db, self.target, self._key_filt, self._filters, self._sel_merge)
        )

    def __eq__(self, other: object) -> bool:
        """Hash the Prop."""
        return hash(self) == hash(other)

    _target_type: type[RecT]
    _db: DB[WriteT | RW, BackT] = field(default_factory=lambda: DB[WriteT, BackT]())

    _key_filt: Sequence[slice | tuple[slice, ...] | list[Hashable] | Hashable] = field(
        default_factory=list
    )
    _sql_filt: list[sqla.ColumnElement[bool]] = field(default_factory=list)

    _sel_merge: RelTree = field(default_factory=RelTree)
    _sel_type: type[SelT] = NoneType

    @property
    def _base_rels(
        self,
    ) -> dict[str, RelTable[Any, Any, Any, Any, BackT, Any, Any, Any, RecT]]:
        """The relations of this record type without superclasses."""
        return {
            name: self._db[
                cast(
                    RelTable[Any, Any, Any, Any, Static, Any, Any, Any, RecT],
                    getattr(self.target, name),
                )
            ]
            for name, rel in self.target._class_props.items()
            if isinstance(rel, Rel | RelSet)
        }

    @property
    def _base_fk_cols(
        self,
    ) -> dict[str, Col[Hashable, Any, BackT, RecT, BaseIdx, Singular | Nullable]]:
        """The foreign key columns of this record type without superclasses."""
        return {
            c.attr.name: c
            for rel in self._base_rels.values()
            for c in rel._fk_map.keys()
        }

    @property
    def _pk_cols(
        self,
    ) -> dict[str, Col[Hashable, Any, BackT, RecT, BaseIdx, Singular | Nullable]]:
        """The primary key columns of this record type."""
        return {
            name: self._db[
                cast(
                    Col[Any, Any, Static, RecT, BaseIdx, Singular | Nullable],
                    getattr(self.target, name),
                )
            ]
            for name, a in self.target._props.items()
            if isinstance(a, Attr) and a.primary_key
        }

    @property
    def _base_cols(
        self,
    ) -> dict[str, Col[Any, Any, BackT, RecT, BaseIdx, Singular | Nullable]]:
        """The columns of this record type without superclasses."""
        return (
            self._base_fk_cols
            | self._pk_cols
            | {
                k: self._db[
                    cast(
                        Col[Any, Any, Static, RecT, BaseIdx, Singular | Nullable],
                        getattr(self.target, a.name),
                    )
                ]
                for k, a in self.target._class_props.items()
                if isinstance(a, Attr) and a.pub_status is Public
            }
        )

    @property
    def _fk_cols(
        self,
    ) -> Mapping[str, Col[Hashable, Any, BackT, Record, BaseIdx, Singular | Nullable]]:
        """The foreign key columns of this record type."""
        return reduce(
            lambda x, y: {**x, **y},
            (self._db[c]._base_fk_cols for c in self.target._record_superclasses),
            self._base_fk_cols,
        )

    @property
    def _target_cols(
        self,
    ) -> dict[str, Col[Any, Any, BackT, RecT, BaseIdx, Singular | Nullable]]:
        return {
            k: self._db[
                cast(
                    Col[Any, Any, Static, RecT, BaseIdx, Singular | Nullable],
                    getattr(self.target, a.name),
                )
            ]
            for k, a in self.target._attrs.items()
            if a.pub_status is Public
        }

    @property
    def _rel_cols(self) -> dict[str, Col[Any, Any, BackT, RecT, Any, Full]]:
        return {
            k: self._db[
                cast(
                    Col[Any, Any, Static, RecT, Any, Full], getattr(self.target, a.name)
                )
            ]
            for k, a in self.target._attr_sets.items()
        }

    @property
    def _rel_tables(
        self,
    ) -> dict[str, RelTable[Any, Any, Any, Any, BackT, Any, Any, Any, RecT]]:
        return {
            k: self._db[
                cast(
                    RelTable[Any, Any, Any, Any, Static, Any, Any, Any, RecT],
                    getattr(self.target, r.name),
                )
            ]
            for k, r in self.target._props.items()
            if isinstance(r, Rel | RelSet)
        }

    @property
    def _data_cols(
        self,
    ) -> dict[str, Col[Any, Any, BackT, RecT, BaseIdx, Singular | Nullable]]:
        return {k: c for k, c in self._target_cols.items() if k not in self._fk_cols}

    @cached_property
    def _single_key(self) -> Hashable | None:
        """Return the single selected key, if it exists."""
        single_keys = [
            k
            for k in self._key_filt
            if not isinstance(k, list | slice) and not has_type(k, tuple[slice, ...])
        ]
        if len(single_keys) > 0:
            return single_keys[0]

        return None

    @cached_property
    def _filter_merge(self) -> tuple[list[sqla.ColumnElement[bool]], RelTree]:
        """Get the SQL filters for this table."""
        key_filt = (
            [
                (
                    idx.in_(val)
                    if isinstance(val, list)
                    else (
                        idx.between(val.start, val.stop)
                        if isinstance(val, slice)
                        else idx == val
                    )
                )
                for idx, val in zip(self._idx_cols.values(), self._key_filt)
            ]
            if len(self._key_filt) > 0
            else []
        )

        return self._parse_filters(
            [
                *self._sql_filt,
                *key_filt,
            ]
        )

    @cached_property
    def _rel_tree(self) -> RelTree:
        """Get the relation merges for this table."""
        return (
            self._filter_merge[1]
            | self._sel_merge
            | RelTree([col.rel_table for col in self._rel_cols.values()])
        )

    @cached_property
    def _filters(self) -> list[sqla.ColumnElement[bool]]:
        """Get the SQL filters for this table."""
        return self._filter_merge[0] + [
            f for rel in self._rel_tree.all_rels for f in rel._filters
        ]

    @cached_property
    def _join_dict(self) -> JoinDict:
        """Dict representation of the relation tree."""
        tree: JoinDict = {}

        for rel in self._rel_tree.all_rels:
            subtree = tree
            if len(rel._join_path) > 1:
                for ref in rel._join_path[1:]:
                    if ref not in subtree:
                        subtree[ref] = {}
                    subtree = subtree[ref]

        return tree

    @cached_property
    def _sql_base_cols(self) -> dict[str, sqla.Column]:
        """Columns of this record type's table."""
        registry = orm.registry(
            metadata=self._db._metadata, type_annotation_map=self.target._type_map
        )

        return {
            name: sqla.Column(
                col.name,
                registry._resolve_type(
                    col.attr.value_type  # pyright: ignore[reportArgumentType]
                ),
                primary_key=col.attr.primary_key,
                autoincrement=False,
                index=col.index,
                nullable=has_type(None, col.attr.value_type),
            )
            for name, col in self._base_cols.items()
        }

    @cached_property
    def _sql_base_fks(self) -> list[sqla.ForeignKeyConstraint]:
        fks: list[sqla.ForeignKeyConstraint] = []

        for rt in self._base_rels.values():
            if isinstance(rt.rel_prop, Rel):
                rel_table = rt._get_sql_base_table()
                fks.append(
                    sqla.ForeignKeyConstraint(
                        [col.name for col in rt._fk_map.keys()],
                        [rel_table.c[col.name] for col in rt._fk_map.values()],
                        name=f"{self.target._get_table_name(self._db._subs)}_{rt.rel_prop.name}_fk",
                    )
                )

        for superclass in self.target._record_superclasses:
            base_table = self._db[superclass]._get_sql_base_table()

            fks.append(
                sqla.ForeignKeyConstraint(
                    [pk_name for pk_name in self._pk_cols],
                    [base_table.c[pk_name] for pk_name in self._pk_cols],
                    name=(
                        self.target._get_table_name(self._db._subs)
                        + "_base_fk_"
                        + gen_str_hash(superclass._get_table_name(self._db._subs), 5)
                    ),
                )
            )

        return fks

    @cached_property
    def _query_cols(
        self,
    ) -> Mapping[str, Col[Any, Any, BackT, Any, BaseIdx, Singular | Nullable]]:
        """Return the index cols."""
        return {
            self._gen_col_fqn(col_name): col
            for col_name, col in self._target_cols.items()
        } | {
            label: col
            for rel in self._rel_tree.all_rels
            for label, col in rel._target_cols.items()
        }

    @cached_property
    def _idx_cols(
        self,
    ) -> Mapping[str, Col[Hashable, Any, BackT, Any, BaseIdx, Singular | Nullable]]:
        """Return the index cols."""
        return {
            self._gen_col_fqn(col_name): col for col_name, col in self._pk_cols.items()
        }

    @cached_property
    def _sql_table(
        self,
    ) -> sqla.FromClause:
        """Recursively join all bases of this record to get the full data."""
        base_table = self._get_sql_base_table("read")

        table = base_table
        cols = {col.name: col for col in base_table.columns}
        for superclass in self.target._record_superclasses:
            superclass_table = self._db[superclass]._sql_table
            cols |= {col.key: col for col in superclass_table.columns}

            table = table.join(
                superclass_table,
                reduce(
                    sqla.and_,
                    (
                        base_table.c[pk_name] == superclass_table.c[pk_name]
                        for pk_name in self._pk_cols
                    ),
                ),
            )

        return (
            sqla.select(*(col.label(col_name) for col_name, col in cols.items()))
            .select_from(table)
            .subquery()
        )

    @property
    def _sql_select(self) -> sqla.Select:
        """Get select for this recset with stable alias name."""
        select = sqla.select(
            *(
                self._sql_table.c[col.name].label(label)
                for label, col in self._query_cols.items()
            )
        ).select_from(self._sql_table)

        for join in self._gen_joins():
            select = select.join(*join)

        for filt in self._filters:
            select = select.where(filt)

        return select

    @cached_property
    def _sql_query(self) -> sqla.Subquery:
        """Get select for this recset with stable alias name."""
        return self._sql_select.subquery(
            self.target._get_table_name(self._db._subs) + "." + hex(hash(self))[:6]
        )

    def _to_static(
        self,
    ) -> Table[RecT, WriteT, Static, SelT, FiltT, RecT]:
        """Return backend-less version of this RelSet."""
        return copy_and_override(
            Table[RecT, WriteT, Static, SelT, FiltT, RecT],
            self,
            _db=DB(b_id=Local.static, types={self.target}),
            _target_type=self.target,
        )

    def _gen_idx_value_map(self, idx: Any, base: bool = False) -> dict[str, Hashable]:
        idx_names = list(
            self._idx_cols.keys()
            if not base
            else self._db[self.target]._idx_cols.keys()
        )

        if len(idx_names) == 1:
            return {idx_names[0]: idx}

        assert isinstance(idx, tuple) and len(idx) == len(idx_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(idx_names, idx)}

    def _get_subdag(
        self,
        backlink_records: set[type[Record]] | None = None,
        _traversed: set[RelTable[Record, Any, Any, Any, Static]] | None = None,
    ) -> set[RelTable[Record, Any, Any, Any, Static, Any, Any, Any, Any]]:
        """Find all paths to the target record type."""
        backlink_records = backlink_records or set()
        _traversed = _traversed or set()

        # Get relations of the target type as next relations
        next_rels = set(self._to_static()._rel_tables.values())

        for backlink_record in backlink_records:
            next_rels |= backlink_record._backrels_to_rels(self.target)

        # Filter out already traversed relations
        next_rels = {rel for rel in next_rels if rel not in _traversed}

        # Add next relations to traversed set
        _traversed |= next_rels

        next_rels = {rel._prefix(self._to_static()) for rel in next_rels}

        # Return next relations + recurse
        return next_rels | {
            rel
            for next_rel in next_rels
            for rel in next_rel._get_subdag(backlink_records, _traversed)
        }

    def _gen_joins(
        self,
        _subtree: JoinDict | None = None,
        _parent: Table[Any, Any, Any, Any, Any, Any] | None = None,
    ) -> list[SqlJoin]:
        """Extract join operations from the relation tree."""
        joins: list[SqlJoin] = []
        _subtree = _subtree if _subtree is not None else self._join_dict
        _parent = _parent if _parent is not None else self

        for target, next_subtree in _subtree.items():
            joins.append(
                (
                    target._sql_query,
                    reduce(
                        sqla.and_,
                        (
                            (
                                _parent[fk] == target[pk]
                                for fk, pk in target._fk_map.items()
                            )
                            if isinstance(target.direct_rel, Rel)
                            else (
                                target[fk] == _parent[pk]
                                for fk, pk in target._fk_map.items()
                            )
                        ),
                    ),
                )
            )

            joins.extend(self._gen_joins(next_subtree, target))

        return joins

    def _suffix(
        self, right: RelTable[RecT2, Any, Any, Any, Any, Any, Any]
    ) -> RelTable[RecT2, Record, Any, WriteT, BackT, Record, Record]:
        """Suffix this with a reltable."""
        return copy_and_override(
            RelTable[RecT2, Record, Any, WriteT, BackT, Any, Record],
            right,
            _db=self._db,
            _parent_type=self.rec,
            _rel_prop=right.rel_prop,
        )

    def _visit_col(
        self,
        element: sqla_visitors.ExternallyTraversible,
        reflist: set[RelTable[Any, Any, Any, Any, Any]] = set(),
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if isinstance(element, Col):
            if isinstance(element.db_table, RelTable):
                if (
                    not isinstance(self, RelTable)
                    or element.db_table._to_static() != self._to_static()
                ):
                    element._table = element.db_table._prefix(self)

                reflist.add(element.db_table)

            return element

        return None

    def _parse_filters(
        self,
        filt: Iterable[sqla.ColumnElement[bool]],
    ) -> tuple[list[sqla.ColumnElement[bool]], RelTree]:
        """Parse filter argument and return SQL expression and join operations."""
        reflist: set[RelTable] = set()
        replace_func = partial(self._visit_col, reflist=reflist, render=False)
        parsed_filt = [
            sqla_visitors.replacement_traverse(f, {}, replace=replace_func)
            for f in filt
        ]
        merge = RelTree(reflist)

        return parsed_filt, merge

    def _gen_col_fqn(self, col_name: str) -> str:
        """Return the fully qualified name of a column."""
        return f"{self.target._default_table_name()}.{col_name}"

    def _parse_schema_items(
        self,
        element: sqla_visitors.ExternallyTraversible,
        **kw: Any,
    ) -> Any | None:
        if isinstance(element, RelTable) and element._to_static() != self._to_static():
            return element._prefix(self)
        elif (
            isinstance(element, Col)
            and isinstance(element.db_table, RelTable)
            and element.db_table._to_static() != self._to_static()
        ):
            return copy_and_override(
                type(element),
                element,
                _table=element.db_table._prefix(self),
                _attr=element._attr,
            )
        elif has_type(element, type[Record]):
            return self._db[element]

        return None

    def _parse_expr[CE: sqla.ClauseElement](self, expr: CE) -> CE:
        """Parse an expression in this database's context."""
        return cast(
            CE,
            sqla_visitors.replacement_traverse(
                expr, {}, replace=self._parse_schema_items
            ),
        )

    def _get_sql_base_table(
        self,
        mode: Literal["read", "replace", "upsert"] = "read",
        without_auto_fks: bool = False,
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        orig_table: sqla.Table | None = None

        if (
            mode != "read"
            and self._db.write_to_overlay is not None
            and self.target not in self._db._subs
        ):
            orig_table = self._get_sql_base_table("read")

            # Create an empty overlay table for the record type
            self._db._subs[self.target] = sqla.table(
                (
                    (
                        self._db.write_to_overlay
                        + "_"
                        + self.target._default_table_name()
                    )
                    if self._db.overlay_type == "name_prefix"
                    else self.target._default_table_name()
                ),
                schema=(
                    self._db.write_to_overlay
                    if self._db.overlay_type == "db_schema"
                    else None
                ),
            )

        table_name = self.target._get_table_name(self._db._subs)

        if not without_auto_fks and table_name in self._db._metadata.tables:
            # Return the table object from metadata if it already exists.
            # This is necessary to avoid circular dependencies.
            return self._db._metadata.tables[table_name]

        sub = self._db._subs.get(self.target)

        cols = self._sql_base_cols
        if without_auto_fks:
            cols = {
                name: col for name, col in cols.items() if name in self.target._attrs
            }

        # Create a partial SQLAlchemy table object from the class definition
        # without foreign keys to avoid circular dependencies.
        # This adds the table to the metadata.
        sqla.Table(
            table_name,
            self._db._metadata,
            *cols.values(),
            schema=(sub.schema if sub is not None else None),
        )

        fks = self._sql_base_fks
        if without_auto_fks:
            fks = [
                fk
                for fk in fks
                if not any(
                    c.name in self._base_fk_cols and c.name not in self._base_cols
                    for c in fk.columns
                )
            ]

        # Re-create the table object with foreign keys and return it.
        table = sqla.Table(
            table_name,
            self._db._metadata,
            *cols.values(),
            *fks,
            schema=(sub.schema if sub is not None else None),
            extend_existing=True,
        )

        self._create_sqla_table(table)

        if orig_table is not None and mode == "upsert":
            with self._db.engine.connect() as conn:
                conn.execute(
                    sqla.insert(table).from_select(
                        orig_table.columns.keys(), orig_table.select()
                    )
                )

        return table

    def _gen_upload_table(
        self,
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        metadata = sqla.MetaData()
        registry = orm.registry(
            metadata=self._db._metadata, type_annotation_map=self.target._type_map
        )

        upload_cols = self._target_cols | dict(self._idx_cols)

        cols = [
            sqla.Column(
                col_name,
                registry._resolve_type(
                    col.attr.value_type  # pyright: ignore[reportArgumentType]
                ),
                primary_key=col.attr.primary_key,
                autoincrement=False,
                index=col.attr.index,
                nullable=has_type(None, col.attr.value_type),
            )
            for col_name, col in upload_cols.items()
        ]

        table_name = self.target._default_table_name() + "_" + token_hex(5)
        table = sqla.Table(
            table_name,
            metadata,
            *cols,
        )

        return table

    def _create_sqla_table(self, sqla_table: sqla.Table) -> None:
        """Create SQL-side table from Table class."""
        if self._db.remove_cross_fks:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            _remove_cross_fk(sqla_table)

        sqla_table.create(self._db.engine, checkfirst=True)

    def _load_from_excel(self) -> None:
        """Load all tables from Excel."""
        assert isinstance(self._db.url, Path | CloudPath | HttpFile)
        path = (
            self._db.url.get() if isinstance(self._db.url, HttpFile) else self._db.url
        )

        recs: list[type[Record]] = [
            self.target,
            *self.target._record_superclasses,
        ]
        with open(path, "rb") as file:
            for rec in recs:
                table = self._db[rec]._get_sql_base_table("replace")

                with self._db.engine.connect() as conn:
                    conn.execute(table.delete())

                pl.read_excel(
                    file, sheet_name=rec._get_table_name(self._db._subs)
                ).write_database(
                    str(table),
                    str(self._db.engine.url),
                    if_table_exists="append",
                )

    def _save_to_excel(self) -> None:
        """Save all (or selected) tables to Excel."""
        assert isinstance(self._db.url, Path | CloudPath | HttpFile)
        file = (
            self._db.url.get() if isinstance(self._db.url, HttpFile) else self._db.url
        )

        recs: list[type[Record]] = [
            self.target,
            *self.target._record_superclasses,
        ]
        with ExcelWorkbook(file) as wb:
            for rec in recs:
                pl.read_database(
                    str(self._db[rec]._get_sql_base_table().select()),
                    self._db.engine,
                ).write_excel(wb, worksheet=rec._get_table_name(self._db._subs))

        if isinstance(self._db.url, HttpFile):
            assert isinstance(file, BytesIO)
            self._db.url.set(file)

    def _df_to_table(
        self,
        df: pd.DataFrame | pl.DataFrame,
        pks: list[str] | None = None,
    ) -> sqla.Table:
        if isinstance(df, pd.DataFrame) and any(
            name is None for name in df.index.names
        ):
            idx_names = list(self._idx_cols.keys())
            df.index.set_names(idx_names, inplace=True)

        if isinstance(df, pd.DataFrame):
            pks = pks or list(df.index.names)
        pks = pks or []

        value_table = self._gen_upload_table()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DuckDBEngineWarning)
            if isinstance(df, pd.DataFrame):
                df.reset_index().to_sql(
                    value_table.name,
                    self._db.engine,
                    if_exists="replace",
                    index=False,
                )
            else:
                df.write_database(
                    str(value_table), self._db.engine, if_table_exists="append"
                )

        return value_table

    def _records_to_df(self, records: dict[Any, Record]) -> pl.DataFrame:
        col_data = [
            {
                **self._gen_idx_value_map(idx),
                **{
                    p.name: getattr(rec, p.name)
                    for p in type(rec)._stat_table._target_cols.values()
                },
            }
            for idx, rec in records.items()
        ]

        rec_types = {type(rec) for rec in records.values()}
        cols = set(self._idx_cols.values()) | reduce(
            set.union, (set(rec._stat_table._target_cols.values()) for rec in rec_types)
        )
        # Transform attribute data into DataFrame.
        return pl.DataFrame(col_data, schema=_get_pl_schema(cols))

    def _mutate(
        self: Table[RecT2, RW, Any, Any, Any],
        value: Table[RecT2, Any, Any, Any, Any] | RecInput[RecT2, Hashable, Hashable],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        record_ids: dict[Hashable, Hashable] | None = None
        valid_caches = self._db._get_valid_cache_set(self.target)

        match value:
            case sqla.Select():
                self._mutate_from_sql(value.subquery(), mode)
                valid_caches.clear()
            case Table():
                value = cast(Table[RecT2, Any, Any, Any, Any], value)
                if value._db != self._db:
                    remote_db = value if isinstance(value, DB) else value.extract()
                    for s in remote_db._def_types:
                        if remote_db.b_id == self._db.b_id:
                            self._db[s]._mutate_from_sql(
                                remote_db[s]._sql_query, "upsert"
                            )
                        else:
                            value_table = self._df_to_table(
                                remote_db[s].to_df(),
                                pks=list(remote_db[s]._idx_cols.keys()),
                            )
                            self._db[s]._mutate_from_sql(
                                value_table,
                                "upsert",
                            )
                            value_table.drop(self._db.engine)

                self._mutate_from_sql(value.select().subquery(), mode)
                valid_caches -= set(value.keys())
            case pd.DataFrame() | pl.DataFrame():
                value_table = self._df_to_table(value, pks=list(self._idx_cols.keys()))
                self._mutate_from_sql(
                    value_table,
                    mode,
                )
                value_table.drop(self._db.engine)

                base_idx_cols = [c.name for c in self._pk_cols.values()]
                base_idx_keys = set(
                    value[base_idx_cols].iter_rows()
                    if isinstance(value, pl.DataFrame)
                    else value[base_idx_cols].itertuples(index=False)
                )

                valid_caches -= base_idx_keys
            case Record():
                self._mutate_from_records({value._index: value}, mode)
                valid_caches -= {value._index}
            case Iterable():
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

                self._mutate_from_records(
                    records,
                    mode,
                )
                valid_caches -= {rec._index for rec in records.values()}

                if len(record_ids) > 0:
                    if isinstance(self, RelTable):
                        self._mutate_from_rec_ids(record_ids, mode)
                        valid_caches -= set(record_ids.values())
                    elif mode == "insert":
                        raise TypeError("Cannot insert record ids into a non-relset.")
            case Hashable():
                assert isinstance(
                    self, RelTable
                ), "Can only update relation sets with record ids."
                self._mutate_from_rec_ids({value: value}, mode)
                valid_caches -= {value}

        return

    def _mutate_from_records(
        self: Table[RecT2, RW, Any, Any, Any],
        records: dict[Hashable, Record],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        db_grouped = {
            db: dict(recs)
            for db, recs in groupby(
                sorted(
                    records.items(), key=lambda x: x[1]._connected and x[1]._db.b_id
                ),
                lambda x: None if not x[1]._connected else x[1]._db,
            )
        }

        unconnected_records = db_grouped.get(None, {})
        local_records = db_grouped.get(self._db, {})

        remote_records = {
            db: recs
            for db, recs in db_grouped.items()
            if db is not None and db != self._db
        }

        if unconnected_records:
            df_data = self._records_to_df(unconnected_records)
            value_table = self._df_to_table(df_data, pks=list(self._idx_cols.keys()))
            self._mutate_from_sql(
                value_table,
                mode,
            )
            value_table.drop(self._db.engine)

        if local_records and isinstance(self, RelTable):
            # Only update relations for records already existing in this db.
            self._mutate_from_rec_ids(
                {idx: rec._index for idx, rec in local_records.items()}, mode
            )

        for db, recs in remote_records.items():
            rec_ids = [rec._index for rec in recs.values()]
            remote_set = db[self.target][rec_ids]

            remote_db = (
                db if all(rec._root for rec in recs.values()) else remote_set.extract()
            )
            for s in remote_db._def_types:
                if remote_db.b_id == self._db.b_id:
                    self._db[s]._mutate_from_sql(remote_db[s]._sql_query, "upsert")
                else:
                    value_table = self._df_to_table(
                        remote_db[s].to_df(),
                        pks=list(remote_db[s]._idx_cols.keys()),
                    )
                    self._db[s]._mutate_from_sql(
                        value_table,
                        "upsert",
                    )
                    value_table.drop(self._db.engine)

            self._mutate_from_sql(remote_set.select().subquery(), mode)

        return

    def _mutate_from_sql(
        self: Table[RecT2, RW, Any, Any, Any],
        value_table: sqla.FromClause,
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        base_recs: list[type[Record]] = [
            self.target,
            *self.target._record_superclasses,
        ]
        cols_by_table = {
            self._db[rec]._get_sql_base_table(
                "upsert" if mode in ("update", "insert", "upsert") else "replace"
            ): {
                label: col
                for label, col in self._query_cols.items()
                if col.db_table.target is rec
            }
            for rec in base_recs
        }

        statements: list[sqla.Executable] = []

        if mode == "replace":
            # Delete all records in the current selection.
            for table in cols_by_table:
                # Prepare delete statement.
                if self._db.engine.dialect.name in (
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
                                    col == self._sql_query.corresponding_column(col)
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

            assert len(self._filters) == 0, "Can only upsert into unfiltered datasets."

            for table, cols in cols_by_table.items():
                # Do an insert-from-select operation, which updates on conflict:
                if mode == "upsert":
                    if self._db.engine.dialect.name in (
                        "postgres",
                        "postgresql",
                        "duckdb",
                    ):
                        # For Postgres / DuckDB, use: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#updating-using-the-excluded-insert-values
                        statement = postgresql.Insert(table).from_select(
                            [col.name for col in cols.values()],
                            sqla.select(
                                *(
                                    value_table.c[label].label(col.name)
                                    for label, col in cols.items()
                                )
                            ).select_from(value_table),
                        )
                        statement = statement.on_conflict_do_update(
                            index_elements=[
                                col.name for col in table.primary_key.columns
                            ],
                            set_={
                                c.name: statement.excluded[c.name]
                                for c in cols.values()
                                if c.name not in table.primary_key.columns
                            },
                        )
                    elif self._db.engine.dialect.name in (
                        "mysql",
                        "mariadb",
                    ):
                        # For MySQL / MariaDB, use: https://docs.sqlalchemy.org/en/20/dialects/mysql.html#insert-on-duplicate-key-update-upsert
                        statement = (
                            mysql.Insert(table)
                            .from_select(
                                [c.name for c in cols.values()],
                                sqla.select(
                                    *(
                                        value_table.c[label].label(col.name)
                                        for label, col in cols.items()
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
                        [c.name for c in cols.values()],
                        value_table,
                    )

                statements.append(statement)
        else:
            # Construct the update statements.

            # Derive current select statement and join with value table, if exists.
            value_join_on = reduce(
                sqla.and_,
                (
                    self._sql_query.corresponding_column(idx_col) == idx_col
                    for idx_col in value_table.primary_key
                ),
            )
            select = self._sql_query.join(
                value_table,
                value_join_on,
            )

            for table, cols in cols_by_table.items():
                col_names = {c.name for c in cols.values()}
                values = {
                    c_name: c
                    for c_name, c in value_table.columns.items()
                    if c_name in col_names
                }

                # Prepare update statement.
                if self._db.engine.dialect.name in (
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
        with self._db.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        if self._db.backend_type == "excel-file":
            self._save_to_excel()

        return

    def __setitem__(
        self,
        key: Any,
        other: Table[Any, Any, Any, Any, Any] | Col[Any, Any, Any, Any, Any],
    ) -> None:
        """Catchall setitem."""
        return


@dataclass(kw_only=True, eq=False)
class Col(
    sqla.ColumnClause[ValTi],
    Generic[ValTi, WriteT, BackT, RecT, IdxT, FiltT],
):
    """Reference an attribute of a record."""

    _table: Table[RecT, WriteT, BackT, Any, Any]
    _attr: Attr[ValTi, WriteT] | AttrSet[ValTi, IdxT, WriteT]

    def __post_init__(self) -> None:  # noqa: D105
        # Initialize fields required by SQLAlchemy superclass.
        self.table = self.db_table._sql_query
        self.name = self.attr.name
        self.type = sqla_types.to_instance(
            self.attr.value_type  # pyright: ignore[reportArgumentType,reportCallIssue]
        )
        self.is_literal = False

    @cached_property
    def db_table(self) -> Table[RecT, WriteT, BackT, Any, Any]:
        """Table reference."""
        return self._table

    @cached_property
    def rel_table(
        self: Col[Any, Any, Any, Any, Any, Full | Filtered]
    ) -> RelTable[DynRecord, None, IdxT, WriteT, BackT, Any, Any, Any, RecT]:
        """Table reference."""
        assert isinstance(self._table, RelTable)
        return self._table

    @cached_property
    def attr(
        self: Col[Any, Any, Any, Any, Any, Singular | Nullable]
    ) -> Attr[ValTi, WriteT]:
        """Attribute reference."""
        assert isinstance(self._attr, Attr)
        return self._attr

    @cached_property
    def attr_set(
        self: Col[Any, Any, Any, Any, Any, Full | Filtered]
    ) -> AttrSet[ValTi, IdxT, WriteT]:
        """Attribute reference."""
        assert isinstance(self._attr, AttrSet)
        return self._attr

    @overload
    def __getitem__(
        self: Col[Any, Any, Any, Any, KeyT2],
        key: Iterable[KeyT2] | slice | tuple[slice, ...] | KeyT2,
    ) -> Col[ValTi, WriteT, BackT, ParT]: ...

    @overload
    def __getitem__(
        self: Col[Any, Any, Any, Record[KeyT2], Any],
        key: Iterable[KeyT2] | slice | tuple[slice, ...] | KeyT2,
    ) -> Col[ValTi, WriteT, BackT, ParT]: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self: Col[Any, Any, Any, Any, Any],
        key: list[Hashable] | slice | tuple[slice, ...] | Hashable,
    ) -> Col[Any, Any, Any, Any, Any]:
        return copy_and_override(
            Col, self, _table=self._table[cast(slice, key)], _attr=self.attr
        )

    def __setitem__(
        self,
        key: Any,
        other: Col[Any, Any, Any, Any, Any, Any],
    ) -> None:
        """Catchall setitem."""
        return

    def select(self) -> sqla.Select:
        """Return select statement for this column."""
        return sqla.select(self).select_from(self.db_table._sql_query)

    @overload
    def to_series(self, kind: type[pd.Series]) -> pd.Series: ...

    @overload
    def to_series(self, kind: type[pl.Series] = ...) -> pl.Series: ...

    def to_series(
        self,
        kind: type[pd.Series | pl.Series] = pl.Series,
    ) -> pd.Series | pl.Series:
        """Download selection."""
        select = self.select()
        engine = self.db_table._db.engine

        if kind is pd.Series:
            with engine.connect() as con:
                return pd.read_sql(select, con).set_index(
                    [c.key for c in self._idx_cols]
                )[self.name]

        return pl.read_database(str(select.compile(engine)), engine)[self.name]

    def __imatmul__(
        self: Col[Any, RW, Any, Any, Any, Any],
        value: ValInput[ValTi, Any] | Col[ValTi, Any, Any, Any, Any, Any],
    ) -> Col[ValTi, RW, BackT, ParT, IdxT, FiltT]:
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

        self.db_table._mutate(df)
        return self

    def keys(
        self: Col,
    ) -> Sequence[Hashable]:
        """Iterable over index keys."""
        return self.db_table.keys()

    def values(  # noqa: D102
        self,
    ) -> list[ValTi]:
        return self.to_series().to_list()

    def __iter__(  # noqa: D105
        self,
    ) -> Iterator[ValTi]:
        return iter(self.values())

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return len(self._table)

    def __hash__(self) -> int:
        """Hash the Col."""
        return gen_int_hash((self._table, self.attr))


@dataclass(kw_only=True, eq=False)
class RelTable(
    Table[RecT, WriteT, BackT, SelT, FiltT, RelT],
    Generic[RecT, LnT, IdxT, WriteT, BackT, RelT, SelT, FiltT, ParT],
):
    """Relational record set."""

    @cached_property
    def parent(
        self: RelTable[Any, Any, Any, Any, Any, Any, Any, Any, ParT2]
    ) -> type[ParT2]:
        """Parent record type."""
        return self._parent_type

    @cached_property
    def rel_prop(
        self: RelTable[Any, Any, Any, Any, Any, Any, Any, Any, Any]
    ) -> (
        Rel[RecT | None, WriteT]
        | BackRel[RecT, Any, WriteT]
        | RelSet[RecT, LnT, Any, WriteT]
    ):
        """Relation of this table."""
        return self._rel_prop

    @cached_property
    def direct_rel(
        self: RelTable[Any, None, Any, Any, Any, Any, Any, Any, Any]
    ) -> Rel[RecT | None, WriteT] | BackRel[RecT, Any, WriteT]:
        """Relation of this table."""
        assert isinstance(self._rel_prop, Rel | BackRel)
        return self._rel_prop

    @cached_property
    def counter_rel(
        self,
    ) -> (
        Rel[ParT | None, WriteT]
        | BackRel[ParT, Any, WriteT]
        | RelSet[ParT, LnT, Any, WriteT]
    ):
        """Counter relation of this table's relation prop."""
        match self.rel_prop:
            case Rel():
                rels_to_parent = [
                    r
                    for r in self._rel_tables.values()
                    if issubclass(self.parent, r.target)
                    and isinstance(r.rel_prop, BackRel)
                ]
                if len(rels_to_parent) == 1:
                    return cast(BackRel[ParT, Any, WriteT], rels_to_parent[0].rel_prop)
                else:
                    return BackRel[ParT, Any, WriteT](
                        _type=BackRel[self.parent],
                        to=getattr(self.target, self.rel_prop.name),
                    )
            case BackRel():
                return cast(Rel[ParT | None, WriteT], self.rel_prop.to.direct_rel)
            case RelSet():
                rels_to_parent = [
                    r
                    for r in self._rel_tables.values()
                    if issubclass(self.parent, r.target) and r.link is self.link
                ]
                if len(rels_to_parent) == 1:
                    return cast(
                        RelSet[ParT, LnT, Any, WriteT], rels_to_parent[0].rel_prop
                    )
                else:
                    return RelSet(_type=RelSet[self.parent, self.link])

    @cached_property
    def target(self) -> type[RecT]:
        """Traget record type."""
        return self.rel_prop.target_type

    @cached_property
    def link(self) -> type[LnT]:
        """Link record type."""
        return cast(
            type[LnT],
            (
                self.rel_prop.link_type
                if isinstance(self.rel_prop, RelSet)
                else NoneType
            ),
        )

    @cached_property
    def parent_table(self) -> Table[ParT, WriteT, BackT, Any, Any, ParT]:
        """Parent set of this Rel."""
        if self._parent_rel is not None:
            tmpl = cast(RelTable[ParT, Any, Any, WriteT, BackT, ParT, Any, Any], self)
            return copy_and_override(
                type(tmpl),
                tmpl,
                _parent_type=self._parent_rel.parent,
                _rel_prop=self._parent_rel.rel_prop,
            )

        return copy_and_override(
            Table[ParT, WriteT, BackT, Any, Any, ParT],
            self,
            _target_type=self.parent,
        )

    @property
    def link_table(
        self: RelTable[Any, RecT2, Any, Any, Any, Any, Any, Any],
    ) -> RelTable[RecT2, None, IdxT, WriteT, BackT, RecT2, SelT, FiltT, ParT]:
        """Get the link set."""
        link_to_parent = self._target_path[0]
        parent_to_link_rel = link_to_parent.counter_rel

        return copy_and_override(
            RelTable[RecT2, None, IdxT, WriteT, BackT, RecT2, SelT, FiltT, ParT],
            self,
            _parent_type=self.parent,
            _rel_prop=parent_to_link_rel,
        )

    @cached_property
    def ln(self: RelTable[Any, RecT2, Any, Any, Any, Any]) -> type[RecT2]:
        """Reference props of the link record type."""
        return (
            self.link_table.rec
            if self.link_table is not None
            else cast(type[RecT2], Record)
        )

    def __hash__(self) -> int:
        """Hash the RelSet."""
        return gen_int_hash((Table.__hash__(self), self.rel_prop))

    _parent_type: type[ParT]
    _rel_prop: (
        Rel[RecT | None, WriteT]
        | BackRel[RecT, Any, WriteT]
        | RelSet[RecT, LnT, Any, WriteT]
    )

    _target_type: type[RecT] = field(init=False)

    @cached_property
    def _fk_map(
        self,
    ) -> bidict[
        Col[Hashable, Any, BackT, Any, BaseIdx, Singular | Nullable],
        Col[Hashable, Any, BackT, Any, BaseIdx, Singular | Nullable],
    ]:
        """Map source foreign keys to target cols."""
        match self.rel_prop:
            case Rel():
                if self.rel_prop.fk_map is not None:
                    return bidict(
                        {
                            self[fk]: self._db[pk]
                            for fk, pk in self.rel_prop.fk_map.items()
                        }
                    )
                else:
                    return bidict(
                        {
                            Col[
                                Hashable, Any, BackT, Any, BaseIdx, Singular | Nullable
                            ](
                                _table=self,
                                _attr=Attr[Hashable](
                                    _name=f"{self.rel_prop.name}_{target_col.attr.name}",
                                    _type=target_col.attr.value_type,
                                    init=False,
                                    index=self.rel_prop.index,
                                    primary_key=self.rel_prop.primary_key,
                                ),
                            ): self._db[
                                target_col
                            ]
                            for target_col in self.target._stat_table._pk_cols.values()
                        }
                    )
            case BackRel():
                return bidict(
                    {
                        self._db[fk]: self._db[pk]
                        for fk, pk in self.rel_prop.to._fk_map.items()
                    }
                )
            case RelSet():
                return bidict()

    @cached_property
    def _filters(self) -> list[sqla.ColumnElement[bool]]:
        """Get the SQL filters for this table."""
        return super()._filters + self.parent_table._filters

    @cached_property
    def _rel_tree(self) -> RelTree:
        """Full relation tree for this table."""
        return super()._rel_tree | self.parent_table._rel_tree * self

    @cached_property
    def _parent_rel(self) -> RelTable[ParT, Any, Any, WriteT, BackT, Any, ParT] | None:
        """Parent relation of this Rel."""
        try:
            rel = getattr(self.parent, "_rel")

            if not isinstance(rel, RelTable):
                return None

            return cast(
                RelTable[ParT, Any, Any, WriteT, BackT, Any, ParT],
                rel,
            )
        except AttributeError:
            return None

    @property
    def _target_path(
        self,
    ) -> tuple[RelTable[Any, None, Any, Any, Any, Any, Any, Any, Any], ...]:
        if self.link is None:
            return (cast(RelTable[Any, None], self),)
        else:
            assert issubclass(self.link, Record)
            backrels = [
                r
                for r in self.link._rels.values()
                if issubclass(self.parent, r.target_type)
            ]
            assert len(backrels) == 1, "Backrel on link record must be unique."
            fwrels = [
                r
                for r in self.link._rels.values()
                if issubclass(self.target, r.target_type)
            ]
            assert len(fwrels) == 1, "Forward ref on link record must be unique."
            return getattr(self.link, backrels[0].name), getattr(
                self.link, fwrels[0].name
            )

    @cached_property
    def _join_path(
        self,
    ) -> tuple[
        Table[Record, WriteT, BackT],
        *tuple[RelTable[Any, None, Any, Any, Any, Any, Any, Any, Any], ...],
    ]:
        """Path from base record type to this Rel."""
        if self._parent_rel is None:
            return (self.parent_table,)

        return (
            *self._parent_rel._join_path,
            *self._target_path,
        )

    @cached_property
    def _path_idx(
        self,
    ) -> Mapping[
        Col[Any, Any, Any, Any, Any, Any], Table[Any, Any, Any, Any, Any, Any]
    ]:
        """Get the path index of the relation."""
        return {
            **(
                self.parent_table._path_idx
                if isinstance(self.parent_table, RelTable)
                else {
                    col: self.parent_table
                    for col in self.parent_table._pk_cols.values()
                }
            ),
            **{
                col: self
                for col in (
                    [self.rel_prop.map_by]
                    if isinstance(self.rel_prop, RelSet)
                    and self.rel_prop.map_by is not None
                    else (
                        self._pk_cols.values()
                        if isinstance(self.rel_prop, RelSet)
                        else []
                    )
                )
            },
        }

    @cached_property
    def _has_list_index(self) -> bool:
        return len(self._path_idx) == 1 and is_subtype(
            list(self._path_idx)[0].value_type, int
        )

    @cached_property
    def _path_str(self) -> str:
        """String representation of the relation path."""
        prefix = (
            self.parent.__name__
            if self._parent_rel is None
            else self._parent_rel._path_str
        )
        return f"{prefix}.{self.rel_prop.name}"

    @cached_property
    def _root_set(self) -> Table[Record, WriteT, BackT]:
        """Root record type of the set."""
        return self._join_path[0]

    @cached_property
    def _idx_cols(
        self,
    ) -> Mapping[str, Col[Hashable, Any, BackT, RecT, BaseIdx, Singular | Nullable]]:
        """Return the index cols."""
        return {
            rel._gen_col_fqn(col.name): self[col] for col, rel in self._path_idx.items()
        }

    @cached_property
    def _sql_query(self) -> sqla.Subquery:
        """Get select for this relset with stable alias name."""
        return self._sql_select.subquery(self._path_str + "." + hex(hash(self))[:6])

    def _is_ancestor(self, other: Table[RecT2, Any, Any, Any, Any]) -> bool:
        """Check if this table is an ancestor of another."""
        return (
            other == self._parent_rel
            or self._parent_rel is not None
            and self._parent_rel._is_ancestor(other)
        )

    def _gen_col_fqn(self, col_name: str) -> str:
        """Return the fully qualified name of a column."""
        return f"{self._path_str}.{col_name}"

    def _prefix(
        self,
        left: Table[ParT, WriteT, BackT2, Any, Any],
    ) -> RelTable[RecT, LnT, IdxT, WriteT, BackT2, RecT, SelT, FiltT, ParT]:
        """Prefix this reltable with a reltable or record type."""
        return copy_and_override(
            RelTable[RecT, LnT, IdxT, WriteT, BackT2, RecT, SelT, FiltT, ParT],
            self,
            _db=left._db,
            _parent_type=left.rec,
            _rel_prop=self.rel_prop,
        )

    def _to_static(
        self,
    ) -> RelTable[RecT, LnT, IdxT, WriteT, Static, RecT, SelT, FiltT, ParT]:
        """Return backend-less version of this RelSet."""
        return copy_and_override(
            RelTable[RecT, LnT, IdxT, WriteT, Static, RecT, SelT, FiltT, ParT],
            self,
            _db=DB(b_id=Local.static, types={self.target}),
            _parent_type=self.parent,
            _rel_prop=self.rel_prop,
        )

    def _gen_fk_value_map(self, val: Hashable) -> dict[str, Hashable]:
        fk_names = [col.name for col in self._fk_map.keys()]

        if len(fk_names) == 1:
            return {fk_names[0]: val}

        assert isinstance(val, tuple) and len(val) == len(fk_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(fk_names, val)}

    def _indexes_to_df(self, indexes: dict[Hashable, Hashable]) -> pl.DataFrame:
        col_data = [
            {
                **self._gen_idx_value_map(idx),
                **self._gen_idx_value_map(base_idx, base=True),
            }
            for idx, base_idx in indexes.items()
        ]

        idx_cols = set(self._idx_cols.values()) | set(self._pk_cols.values())
        # Transform attribute data into DataFrame.
        return pl.DataFrame(col_data, schema=_get_pl_schema(idx_cols))

    def _mutate_from_rec_ids(  # noqa: C901
        self: RelTable[RecT2, Any, Any, RW, BackT, Any, Any, Any],
        indexes: dict[Hashable, Hashable],
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
    ) -> None:
        idx_df = self._indexes_to_df(indexes)
        self._mutate_from_sql(
            self._df_to_table(idx_df, pks=list(self._idx_cols.keys())),
            mode,
            rel_only=True,
        )
        return

    def _mutate_from_sql(  # noqa: C901
        self: RelTable[RecT2, Any, Any, RW, Any, Any, Any, Any],
        value_table: sqla.FromClause,
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
        rel_only: bool = False,
    ) -> None:
        if not rel_only:
            Table._mutate_from_sql(self, value_table, mode)

        # Update relations with parent records.
        if isinstance(self.rel_prop, Rel):
            # Case: parent links directly to child (n -> 1)
            idx_cols = [value_table.c[col_name] for col_name in self._idx_cols]
            fk_cols = [
                value_table.c[pk.name].label(fk.name) for fk, pk in self._fk_map.items()
            ]
            self.parent_table._mutate_from_sql(
                sqla.select(*idx_cols, *fk_cols).select_from(value_table).subquery(),
                "update",
            )
        elif issubclass(self.link, Record):
            # Case: parent and child are linked via assoc table (n <--> m)
            # Update link table with new child indexes.
            idx_cols = [value_table.c[col_fqn] for col_fqn in self._idx_cols]
            fk_cols = [
                value_table.c[pk.name].label(fk.name)
                for fk, pk in self._target_path[1]._fk_map.items()
            ]
            link_cols = [
                value_table.c[col_fqn].label(col.name)
                for col_fqn, col in self._idx_cols.items()
                if col.name in self.link_table._target_cols
            ]
            self.link_table._mutate_from_sql(
                sqla.select(*idx_cols, *fk_cols, *link_cols)
                .select_from(value_table)
                .subquery(),
                mode,
            )
        else:
            # The (1 <- n) case is already covered by updating
            # the child record directly, which includes all its foreign keys.
            pass

        return


@dataclass(kw_only=True, eq=False)
class Backend(Generic[BackT]):
    """Data backend."""

    b_id: BackT = Local.dynamic
    """Unique name to identify this database's backend by."""

    url: sqla.URL | CloudPath | HttpFile | Path | None = None
    """Connection URL or path."""


type OverlayType = Literal["name_prefix", "db_schema"]


@dataclass(kw_only=True, eq=False)
class DB(Table[Record, WriteT, BackT, None, Full], Backend[BackT]):
    """Database."""

    db: DB[WriteT, BackT] = field(init=False, repr=False)

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
        self.db = self

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

        self.target = self.target or get_lowest_common_base(self._def_types.keys())

        if self.validate_on_init:
            self.validate()

        if self.write_to_overlay is not None and self.overlay_type == "db_schema":
            self._ensure_schema_exists(self.write_to_overlay)

        if self.backend_type == "excel-file":
            for rec in self._def_types:
                self[rec]._load_from_excel()

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
            else (sqla.create_engine(f"duckdb:///:memory:{self.b_id}"))
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
        self: DB[RW, Any],
        data: pd.DataFrame | pl.DataFrame | sqla.Select,
        fks: (
            Mapping[str, Col[Any, Any, Static, Any, BaseIdx, Singular | Nullable]]
            | None
        ) = None,
    ) -> Table[DynRecord, R, BackT, Any, Any, DynRecord]:
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, pd.DataFrame | pl.DataFrame)
            else f"temp_{token_hex(5)}"
        )

        rec = dynamic_record_type(name, props=props_from_data(data, fks))
        ds = Table[DynRecord, RW, BackT](_target_type=rec, _db=self)

        ds &= data

        return ds

    def to_graph(
        self, nodes: Sequence[type[Record]]
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
            .assign(table=n.target._default_table_name())
            for n in node_tables
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )

        directed_edges = reduce(
            set.union, (set(self[n]._rel_tables.values()) for n in nodes)
        )

        undirected_edges: dict[type[Record], set[tuple[Rel, ...]]] = {
            t: set() for t in nodes
        }
        for n in nodes:
            for at in self._assoc_types:
                if len(at._rels) == 2:
                    left, right = (r for r in at._rels.values())
                    assert left is not None and right is not None
                    if left.target_type == n:
                        undirected_edges[n].add((left, right))
                    elif right.target_type == n:
                        undirected_edges[n].add((right, left))

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                *[
                    node_df.loc[
                        node_df["table"]
                        == str((rt.parent or Record)._default_table_name())
                    ]
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(rt.rel_prop.target_type._default_table_name())
                        ],
                        left_on=[c.name for c in rt._fk_map.keys()],
                        right_on=[c.name for c in rt._fk_map.values()],
                    )
                    .rename(columns={"node_id": "target"})[["source", "target"]]
                    .assign(
                        ltr=",".join(c.name for c in rt._fk_map.keys()),
                        rtl=None,
                    )
                    for rt in directed_edges
                ],
                *[
                    self[assoc_table]
                    .to_df(kind=pd.DataFrame)
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(left_rel.target_type._default_table_name())
                        ].dropna(axis="columns", how="all"),
                        left_on=[c.name for c in left_rel.fk_map.keys()],
                        right_on=[c.name for c in left_rel.fk_map.values()],
                        how="inner",
                    )
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(left_rel.target_type._default_table_name())
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
                                *(a for a in self[assoc_table]._target_cols),
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

    def __hash__(self) -> int:
        """Hash the DB."""
        return gen_int_hash(
            (
                self.b_id,
                self.url,
                self._subs,
            )
        )

    @cached_property
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

    @cached_property
    def _assoc_types(self) -> set[type[Record]]:
        """Set of all association tables in this DB."""
        assoc_types = set()
        for rec in self._def_types:
            assoc_table = self[rec]
            pks = set([col.name for col in assoc_table._pk_cols.values()])
            fks = set(
                [col.name for rel in rec._rels.values() for col in rel.fk_map.keys()]
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

    def _ensure_schema_exists(self, schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_schema(schema_name):
            with self.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name


class RecUUID(Record[UUID]):
    """Record type with a default UUID primary key."""

    _template = True
    _id: Attr[UUID] = Attr(primary_key=True, default_factory=uuid4)


class RecHashed(Record[int]):
    """Record type with a default hashed primary key."""

    _template = True

    _id: Attr[int] = Attr(primary_key=True, init=False)

    def __post_init__(self) -> None:  # noqa: D105
        self._id = gen_int_hash(
            {
                a.name: getattr(self, a.name)
                for a in type(self)._stat_table._base_cols.values()
            }
        )


class Scalar(Record[KeyT], Generic[ValT2, KeyT]):
    """Dynamically defined record type."""

    _template = True

    _id: Attr[KeyT] = Attr(primary_key=True, default_factory=uuid4)
    _value: Col[ValT2]


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls: type[Record], name: str) -> Col[Any, Any, Any, Record]:
        """Get dynamic attribute by dynamic name."""
        return Col(_attr=Attr(_name=name), _table=cls._stat_table)

    def __getattr__(cls: type[Record], name: str) -> Col[Any, Any, Any, Record]:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)

        return Col(_attr=Attr(_name=name), _table=cls._stat_table)


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


a = DynRecord


def dynamic_record_type(
    name: str, props: Iterable[Prop[Any, Any]] = []
) -> type[DynRecord]:
    """Create a dynamically defined record type."""
    return type(
        name,
        (DynRecord,),
        {
            **{p.name: p for p in props},
            "__annotations__": {p.name: p._type.hint for p in props},
        },
    )


class Link(RecHashed, Generic[RecT2, RecT3]):
    """Automatically defined relation record type."""

    _template = True

    _from: RelTable[RecT2]
    _to: RelTable[RecT3]


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
