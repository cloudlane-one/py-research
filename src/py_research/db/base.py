"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from datetime import date, datetime, time
from functools import cached_property, partial, reduce
from inspect import get_annotations, getmodule
from io import BytesIO
from itertools import zip_longest
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
    Protocol,
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

DataFrame = pd.DataFrame | pl.DataFrame
Series = pd.Series | pl.Series

Name = TypeVar("Name", bound=LiteralString)

B_def = TypeVar("B_def", bound="Backend", default="Backend", covariant=True)
B_nul = TypeVar(
    "B_nul", bound="Backend | None", default="Backend | None", covariant=True
)
B2_opt = TypeVar("B2_opt", bound="Backend | None")

Key = TypeVar("Key", bound=Hashable)
Key2 = TypeVar("Key2", bound=Hashable)
Key3 = TypeVar("Key3", bound=Hashable)
Key4 = TypeVar("Key4", bound=Hashable)
Key_def = TypeVar("Key_def", contravariant=True, bound=Hashable, default=Any)

type Index = Hashable | BaseIdx | SingleIdx | FilteredIdx
Idx = TypeVar("Idx", bound=Index)
Idx_cov = TypeVar("Idx_cov", covariant=True, bound=Index)
Idx_def = TypeVar(
    "Idx_def",
    covariant=True,
    bound=Index,
    default=Index,
)

type KeyIdx = Hashable | SingleIdx
KeyIdx_def = TypeVar("KeyIdx_def", bound=KeyIdx, default=KeyIdx, covariant=True)

type InsertIdx = Hashable | BaseIdx
InsIdx = TypeVar("InsIdx", bound=InsertIdx, covariant=True)

IdxTup = TypeVarTuple("IdxTup")

type IdxStart[Key: Hashable] = Key | tuple[Key, *tuple[Any, ...]]
type IdxEnd[Key: Hashable] = tuple[*tuple[Any, ...], Key]
type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]
type IdxTupStart[*IdxTup] = tuple[*IdxTup, *tuple[Any]]
type IdxTupStartEnd[*IdxTup, Key2: Hashable] = tuple[*IdxTup, *tuple[Any], Key2]

Val = TypeVar("Val")
Val2 = TypeVar("Val2")
Val3 = TypeVar("Val3")
Val_cov = TypeVar("Val_cov", covariant=True)
Val_def = TypeVar("Val_def", covariant=True, default=Any)
Val_defi = TypeVar("Val_defi", default=Any)

PVal = TypeVar("PVal", bound="Prop")

Rec = TypeVar("Rec", bound="Record")
Rec2 = TypeVar("Rec2", bound="Record")
Rec3 = TypeVar("Rec3", bound="Record")
Rec_cov = TypeVar("Rec_cov", covariant=True, bound="Record")
Rec_def = TypeVar("Rec_def", covariant=True, bound="Record", default="Record")
Rec2_def = TypeVar("Rec2_def", covariant=True, bound="Record", default="Record")
Rec3_nul = TypeVar(
    "Rec3_nul", covariant=True, bound="Record | None", default="Record | None"
)
Rec3_opt = TypeVar("Rec3_opt", covariant=True, bound="Record | None")
Rec4_def = TypeVar("Rec4_def", covariant=True, bound="Record", default="Record")
Rec2_nul = TypeVar(
    "Rec2_nul", bound="Record | None", default="Record | None", covariant=True
)

type RecordValue[Rec: Record] = Rec | Iterable[Rec] | Mapping[Any, Rec] | None
Recs_cov = TypeVar("Recs_cov", bound=RecordValue, covariant=True)

RM = TypeVar("RM", covariant=True, bound=tuple | None, default=None)

RelTup = TypeVarTuple("RelTup")

P = TypeVar("P", bound="Prop")
P_nul = TypeVar(
    "P_nul", bound="Prop[Any, R] | None", covariant=True, default="Prop[Any, R] | None"
)

R_def = TypeVar("R_def", bound="R", default="R", covariant=True)
RWT = TypeVar("RWT", bound="R", default="RW", covariant=True)

Df = TypeVar("Df", bound=DataFrame)
Dl = TypeVar("Dl", bound="DataFrame | Record")

Params = ParamSpec("Params")
DC = TypeVar("DC", bound="DataclassInstance")


class DataclassInstance(Protocol):
    """Protocol for dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def copy_and_override(
    obj: DC,
    _init: Callable[Params, DC] | None = None,
    *args: Params.args,
    **kwargs: Params.kwargs,
) -> DC:
    """Re-construct a dataclass instance with all its init params + override.

    Warning:
        Does not work for kw_only dataclasses and InitVars (yet).
    """
    target_fields = set(fields(cast(type, _init)))
    obj_fields = {
        f: getattr(obj, f.name) for f in fields(obj) if f.init and f in target_fields
    }

    obj_args = [
        v
        for f, v in obj_fields.items()
        if not f.kw_only
        and f.default is MISSING
        and f.default_factory is MISSING
        and f.name not in kwargs
    ]
    obj_kwargs = {f.name: v for f, v in obj_fields.items() if f not in obj_args}

    new_args = [
        v1 if v1 is not MISSING else v2
        for v1, v2 in zip_longest(args, obj_args, fillvalue=MISSING)
        if v2 is not MISSING
    ]
    new_kwargs = {**obj_kwargs, **kwargs}
    constr_func = _init or type(obj)
    return constr_func(*new_args, **new_kwargs)  # type: ignore


class BaseIdx:
    """Singleton to mark dataset index as default index."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


class SingleIdx:
    """Singleton to mark dataset index as a single value."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


class FilteredIdx(Generic[Idx_cov]):
    """Singleton to mark dataset index as filtered."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


@dataclass(frozen=True, kw_only=True)
class Backend(Generic[Name]):
    """SQL backend for DB."""

    name: Name
    """Unique name to identify this backend by."""

    url: sqla.URL | CloudPath | HttpFile | Path | None = None
    """Connection URL or path."""

    @cached_property
    def type(
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
            if (self.type == "sql-connection" or self.type == "sqlite-file")
            else (sqla.create_engine(f"duckdb:///:memory:{self.name}"))
        )


class R:
    """Read-only flag."""


class RO(R):
    """Read-only flag."""


class RW(RO):
    """Read-write flag."""


type RecInput[Rec: Record, Key: InsertIdx] = DataFrame | Iterable[Rec] | Mapping[
    Key, Rec
] | sqla.Select | Rec

type PartialRec[Rec: Record] = Mapping[Set[Rec, Any], Any]

type PartialRecInput[Rec: Record, Key: InsertIdx] = RecInput[Rec, Any] | Iterable[
    PartialRec[Rec]
] | Mapping[Key, PartialRec[Rec]] | PartialRec[Rec]

type ValInput[Val, Key: Hashable] = Series | Mapping[Key, Val] | sqla.Select[
    tuple[Key, Val]
] | Val


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
    data: DataFrame | sqla.Select, foreign_keys: Mapping[str, ValueSet] | None = None
) -> list[Prop]:
    """Extract prop definitions from dataframe or query."""
    if isinstance(data, pd.DataFrame) and len(data.index.names) > 1:
        raise NotImplementedError("Multi-index not supported yet.")

    foreign_keys = foreign_keys or {}

    def _gen_prop(name: str, data: Series | sqla.ColumnElement) -> Prop:
        is_rel = name in foreign_keys
        value_type = (
            map_df_dtype(data) if isinstance(data, Series) else map_col_dtype(data)
        ) or Any
        attr = ValueSet[value_type, Any, None, Any, RW](
            prop=Attr(
                primary_key=True,
                _name=name if not is_rel else f"fk_{name}",
                _type=PropType(Attr[value_type]),
            ),
            record_set=DynRecord._set(),
        )
        return (
            attr.prop
            if not is_rel
            else Rel(on={attr: foreign_keys[name]}, _type=PropType(Rel[Record]))
        )

    columns = (
        [data[col] for col in data.columns]
        if isinstance(data, DataFrame)
        else list(data.columns)
    )

    return [
        *(
            (
                _gen_prop(level, data.index.get_level_values(level).to_series())
                for level in data.index.names
            )
            if isinstance(data, pd.DataFrame)
            else []
        ),
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

    table.foreign_keys = set(  # type: ignore
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


class Unloaded:
    """Singleton to mark unfetched values."""


@dataclass(frozen=True)
class PropType(Generic[P_nul]):
    """Reference to a type."""

    hint: str | type[P_nul] | None = None
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
            return Attr if "Attr" in hint else Rel if "Rel" in hint else Prop

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

        return typedef  # type: ignore

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

    def value_type(self: PropType[Prop[Val, Any]]) -> SingleTypeDef[Val]:
        """Resolve the value type reference."""
        args = self._generic_args
        if len(args) == 0:
            return cast(type[Val], object)

        arg = args[0]
        return cast(SingleTypeDef[Val], self._to_typedef(arg))

    def record_type(self: PropType[Rel[Any, Any, Any, Any]]) -> type[Record]:
        """Resolve the record type reference."""
        assert is_subtype(self._generic_type, Rel)

        args = self._generic_args
        assert len(args) > 0

        recs = args[0]
        if isinstance(recs, TypeVar):
            recs = self._to_type(recs)

        rec_args = get_args(recs)
        rec_arg = recs

        if is_subtype(recs, Mapping):
            assert len(rec_args) >= 2
            rec_arg = rec_args[1]
        elif is_subtype(recs, Iterable):
            assert len(rec_args) >= 1
            rec_arg = rec_args[0]

        rec_type = self._to_type(rec_arg)
        assert issubclass(rec_type, Record)

        return rec_type

    def link_type(self: PropType[Rel[Any, Any, Any, Any]]) -> type[Record | None]:
        """Resolve the record type reference."""
        assert is_subtype(self._generic_type, Rel)
        args = self._generic_args
        rec = args[1]
        rec_type = self._to_type(rec)

        if not issubclass(rec_type, Record):
            return NoneType

        return rec_type


@dataclass(eq=False)
class Prop(Generic[Val_cov, RWT]):
    """Reference a property of a record."""

    _type: PropType[Self]
    _name: str | None = None

    alias: str | None = None
    default: Val_cov | None = None
    default_factory: Callable[[], Val_cov] | None = None
    init: bool = True

    getter: Callable[[Record], Val_cov] | None = None
    setter: Callable[[Record, Val_cov], None] | None = None

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

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Val_cov: ...

    @overload
    def __get__(
        self, instance: None, owner: type[Rec]
    ) -> Set[Val_cov, Rec, None, Any, Prop[Val_cov, RWT]]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[Rec] | None
    ) -> Val_cov | Set[Val_cov, Rec, None, Any, Prop[Val_cov, RWT]] | Self:
        if isinstance(instance, Record):
            if self.getter is not None:
                value = self.getter(instance)
            else:
                value = instance.__dict__[self.name]

            if isinstance(value, Unloaded) and instance._loader is not None:
                try:
                    value = instance._loader(getattr(owner, self.name), instance._index)
                except KeyError:
                    pass

            if isinstance(value, Unloaded):
                if self.default_factory is not None:
                    value = self.default_factory()
                elif self.default is not None:
                    value = self.default

            if isinstance(value, Unloaded):
                raise ValueError("Property value could not fetched.")

            instance.__dict__[self.name] = value
            return value
        elif owner is not None and issubclass(owner, Record):
            rec_type = cast(type[Rec], owner)
            if isinstance(self, Rel):
                return RelSet(
                    prop=self,  # type: ignore
                    backend=None,
                    parent_type=rec_type,
                )
            if isinstance(self, Attr):
                return ValueSet(
                    prop=self,
                    backend=None,
                    record_set=rec_type._set(),
                )
            return Set(prop=self, backend=None)

        return self

    def __set__(self: Prop[Val, RW], instance: Record, value: Val | Unloaded) -> None:
        """Set the value of the property."""
        if self.setter is not None and not isinstance(value, Unloaded):
            self.setter(instance, value)
        instance.__dict__[self.name] = value


@dataclass(eq=False)
class Attr(Prop[Val_cov, RWT]):
    """Define an attribute of a record."""

    index: bool = False
    primary_key: bool = False

    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = None

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Val_cov: ...

    @overload
    def __get__(
        self, instance: None, owner: type[Rec]
    ) -> ValueSet[Val_cov, Rec, None, Any, RWT]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105 # type: ignore
        self, instance: object | None, owner: type | type[Rec] | None
    ) -> Val_cov | Set[Val_cov, Rec, None, Any, Prop[Val_cov, RWT]] | Self:
        return super().__get__(instance, owner)


@dataclass(eq=False)
class Rel(Prop[Recs_cov, RWT], Generic[Recs_cov, Rec2_nul, RWT, Rec_def]):
    """Define a relation to another record."""

    index: bool = False
    primary_key: bool = False

    on: DirectLink[Rec_def] | BackLink[Rec_def] | BiLink[Rec_def, Any] | None = None
    order_by: Mapping[ValueSet[Any, Any, None], int] | None = None
    map_by: ValueSet[Any, Any, None] | None = None
    collection: Callable[[Any], Recs_cov] | None = None

    @property
    def dyn_target_type(self) -> type[Rec_def] | None:
        """Dynamic target type of the relation."""
        match self.on:
            case dict():
                return cast(type[Rec_def], next(iter(self.on.values())).parent_type)
            case tuple():
                via_1 = self.on[1]
                assert isinstance(via_1, RecordSet)
                return via_1.record_type
            case type() | RecordSet() | ValueSet() | Iterable() | None:
                return None

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Recs_cov: ...

    @overload
    def __get__(
        self: Rel[Mapping[Key, Rec2], Rec3_opt, RWT],
        instance: None,
        owner: type[Rec],
    ) -> RelSet[Rec2, Key, None, RWT, Rec, Rec3_opt, Rec2, None]: ...

    @overload
    def __get__(
        self: Rel[Sequence[Rec2], Rec3_opt, RWT],
        instance: None,
        owner: type[Rec],
    ) -> RelSet[Rec2, int, None, RWT, Rec, Rec3_opt, Rec2, None]: ...

    @overload
    def __get__(
        self: Rel[Iterable[Rec2], Rec3_opt, RWT],
        instance: None,
        owner: type[Rec],
    ) -> RelSet[Rec2, BaseIdx, None, RWT, Rec, Rec3_opt, Rec2, None]: ...

    @overload
    def __get__(
        self: Rel[Rec2, Rec3_opt, RWT],
        instance: None,
        owner: type[Rec],
    ) -> RelSet[Rec2, SingleIdx, None, RWT, Rec, Rec3_opt, Rec2, None]: ...

    @overload
    def __get__(
        self: Rel[Rec2 | None, Rec3_opt, RWT],
        instance: None,
        owner: type[Rec],
    ) -> RelSet[Rec2, SingleIdx | None, None, RWT, Rec, Rec3_opt, Rec2, None]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105 # type: ignore
        self, instance: object | None, owner: type | type[Rec2] | None
    ) -> Recs_cov | Set[Recs_cov, Rec2, None, Any, Prop[Recs_cov, RWT]] | Self:
        return super().__get__(instance, owner)


type DirectLink[Rec: Record] = (
    ValueSet | Iterable[ValueSet] | dict[ValueSet, ValueSet[Rec, Any]]
)

type BackLink[Rec: Record] = (RelSet[Any, Any, None, Any, Rec] | type[Rec])

type BiLink[Rec: Record, Rec2: Record] = (
    RelSet[Rec, Any, None, Any, Rec2]
    | tuple[RelSet[Any, Any, None, Any, Rec2], RelSet[Rec, Any, None, Any, Rec2]]
    | type[Rec2]
)


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: bool = ...,
    collection: None = ...,
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
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: bool = ...,
    collection: None = ...,
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
) -> Attr[Val]: ...


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: bool | Literal["fk"] = ...,
    primary_key: bool | Literal["fk"] = ...,
    collection: None = ...,
    link_on: DirectLink[Rec],
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[Rec]: ...


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal["fk"],
    primary_key: bool | Literal["fk"] = ...,
    collection: None = ...,
    link_on: DirectLink[Rec] | None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[Rec]: ...


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: bool | Literal["fk"] = ...,
    primary_key: Literal["fk"],
    collection: None = ...,
    link_on: DirectLink[Rec] | None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[Rec]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Prop[Val, RO]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: Callable[[Record, Val], None],
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Prop[Val, RW]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: None = ...,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement],
    local: bool = ...,
    init: bool = ...,
) -> Attr[Val, RO]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: Callable[[Record, Val], None],
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement],
    local: bool = ...,
    init: bool = ...,
) -> Attr[Val, RW]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[True] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: None = ...,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Attr[Val, RO]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[True] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: Callable[[Record, Val], None],
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Attr[Val, RW]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,  # type: ignore
    order_by: Mapping[ValueSet[Any, Any, None, Rec | Rec2], int] | None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[list[Rec], Rec2]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: Callable[[Iterable[Rec]], Recs_cov],
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,  # type: ignore
    order_by: Mapping[ValueSet[Any, Any, None, Rec | Rec2], int] | None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[Recs_cov, Rec2]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,
    order_by: None = ...,
    map_by: ValueSet[Val3, Any, None, Rec | Rec2],
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[dict[Val3, Rec], Rec2]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: Callable[[Mapping[Val3, Rec]], Recs_cov],
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,
    order_by: None = ...,
    map_by: ValueSet[Val3, Any, None, Rec | Rec2],
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
    local: bool = ...,
    init: bool = ...,
) -> Rel[Recs_cov, Rec2]: ...


def prop(
    *,
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
    alias: str | None = None,
    index: bool | Literal["fk"] = False,
    primary_key: bool | Literal["fk"] = False,
    collection: Callable[[Any], Any] | None = None,
    link_on: DirectLink[Record] | None = None,
    link_from: BackLink[Record] | None = None,
    link_via: BiLink[Record, Record] | None = None,
    order_by: Mapping[ValueSet[Any, Any, None], int] | None = None,
    map_by: ValueSet[Any, Any, None] | None = None,
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
        )

    if any(
        a is not None for a in (link_on, link_from, link_via, order_by, map_by)
    ) or any(a == "fk" for a in (index, primary_key)):
        return Rel(
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
            collection=collection,
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

    return Attr(
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
class RecordLinks(Generic[Rec]):
    """Descriptor to access the link records of a record's rels."""

    rec: Rec

    @overload
    def __getitem__(
        self, rel: RelSet[Record, SingleIdx, None, RW, Rec, Rec2]
    ) -> Rec2: ...

    @overload
    def __getitem__(
        self, rel: RelSet[Record, SingleIdx | None, None, RW, Rec, Rec2]
    ) -> Rec2 | None: ...

    @overload
    def __getitem__(
        self, rel: RelSet[Record, Key, None, RW, Rec, Rec2]
    ) -> dict[Key, Rec2]: ...

    @overload
    def __getitem__(
        self, rel: RelSet[Record, BaseIdx, None, RW, Rec, Rec2]
    ) -> list[Rec2]: ...

    def __getitem__(self, rel: RelSet[Record, Any, None, RW, Rec, Rec2]) -> RecordValue:
        """Return the model of the given relation, if any."""
        if rel.link_type is Record:
            return DynRecord()

        if self.rec._loader is not None and rel.prop.name not in self.rec._edge_dict:
            self.rec._edge_dict[rel.prop.name] = self.rec._loader(rel, self.rec._index)

        return self.rec._edge_dict[rel.prop.name]

    @overload
    def __setitem__(
        self, rel: RelSet[Record, SingleIdx | None, None, RW, Rec, Rec2], value: Rec2
    ) -> None: ...

    @overload
    def __setitem__(
        self, rel: RelSet[Record, Key, None, RW, Rec, Rec2], value: Mapping[Key, Rec2]
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        rel: RelSet[Record, BaseIdx, None, RW, Rec, Rec2],
        value: Mapping[Any, Rec2] | Iterable[Rec2],
    ): ...

    def __setitem__(
        self, rel: RelSet[Record, Any, None, RW, Rec, Rec2], value: RecordValue
    ) -> None:
        """Return the model of the given relation, if any."""
        self.rec._edge_dict[rel.prop.name] = value


class RecordMeta(type):
    """Metaclass for record types."""

    _record_bases: list[type[Record]]
    _base_pks: dict[
        type[Record],
        dict[ValueSet[Hashable, Any, None], ValueSet[Hashable, Any, None]],
    ]

    _defined_props: dict[str, Set[Any, Any, None, Any, Prop]]
    _defined_attrs: dict[str, ValueSet[Any, Any, None]]
    _defined_rels: dict[str, RelSet[Record, Any, None]]

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
                    prop_defs[prop_name] = prop_set.prop._type
                    if prop_name not in cls.__dict__:
                        prop_set = copy_and_override(
                            prop_set,
                            type(prop_set),
                            prop=copy_and_override(
                                prop_set.prop,
                                type(prop_set.prop),
                                _type=copy_and_override(
                                    prop_set.prop._type,
                                    type(prop_set.prop._type),
                                    typevar_map=typevar_map,
                                    ctx=cls._src_mod,
                                ),
                            ),
                        )
                        setattr(cls, prop_name, prop_set.prop)
            else:
                assert orig is c
                cls._record_bases.append(orig)

        cls._base_pks = (
            {
                base: {
                    ValueSet(
                        prop=Attr(
                            _name=pk.name,
                            primary_key=True,
                            _type=PropType(Attr[pk.value_type]),
                        ),
                        record_set=cast(type[Record], cls)._set(),
                    ): pk
                    for pk in base._primary_keys.values()
                }
                for base in cls._record_bases
            }
            if not cls._is_record
            else {}
        )

        cls._defined_props = {name: getattr(cls, name) for name in prop_defs.keys()}

        cls._defined_attrs = {
            name: attr
            for name, attr in cls._defined_props.items()
            if isinstance(attr, ValueSet)
        }

        cls._defined_rels = {
            name: rel
            for name, rel in cls._defined_props.items()
            if isinstance(rel, RelSet)
        }

    @property
    def _defined_rel_attrs(cls) -> dict[str, ValueSet[Hashable, Any, None]]:
        return {
            a.prop.name: a
            for rel in cls._defined_rels.values()
            if rel.direct_rel is True
            for a in rel.fk_map.keys()
        }

    @property
    def _primary_keys(cls: type[Record]) -> dict[str, ValueSet[Hashable, Any, None]]:
        base_pk_attrs = {v.name: v for vs in cls._base_pks.values() for v in vs.keys()}

        if len(base_pk_attrs) > 0:
            assert all(not a.primary_key for a in cls._defined_attrs.values())
            return base_pk_attrs

        return {name: a for name, a in cls._defined_attrs.items() if a.prop.primary_key}

    @property
    def _props(cls) -> dict[str, Set[Any, Any, None, Any, Prop]]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._props for c in cls._record_bases),
            cls._defined_props,
        )

    @property
    def _attrs(cls) -> dict[str, ValueSet[Hashable, Any, None]]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._attrs for c in cls._record_bases),
            cls._defined_attrs,
        )

    @property
    def _rels(cls) -> dict[str, RelSet[Record, Any, None]]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._rels for c in cls._record_bases),
            cls._defined_rels,
        )

    @property
    def _rel_types(cls) -> set[type[Record]]:
        return {rel.record_type for rel in cls._rels.values()}


@dataclass_transform(kw_only_default=True, field_specifiers=(prop,))
class Record(Generic[Key_def], metaclass=RecordMeta):
    """Schema for a record in a database."""

    _template: ClassVar[bool]
    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID: UUIDType(binary=False),  # Binary type causes issues with DuckDB
    }

    _loader: Callable[[Set, Key_def], Any] | None = None
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
    def _columns(cls, registry: orm.registry) -> list[sqla.Column]:
        """Columns of this record type's table."""
        table_attrs = (
            set(cls._defined_attrs.values())
            | set(cls._defined_rel_attrs.values())
            | set(cls._primary_keys.values())
        )

        return [
            sqla.Column(
                attr.prop.name,
                registry._resolve_type(attr.value_type),
                primary_key=attr.prop.primary_key,
                autoincrement=False,
                index=attr.prop.index,
            )
            for attr in table_attrs
        ]

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
                        [attr.prop.name for attr in rel.fk_map.keys()],
                        [rel_table.c[attr.prop.name] for attr in rel.fk_map.values()],
                        name=f"{cls._sql_table_name(subs)}_{rel.prop.name}_fk",
                    )
                )

        for base, pks in cls._base_pks.items():
            base_table = base._table(metadata, subs)

            fks.append(
                sqla.ForeignKeyConstraint(
                    [attr.name for attr in pks.keys()],
                    [base_table.c[attr.name] for attr in pks.values()],
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

        # Create a partial SQLAlchemy table object from the class definition
        # without foreign keys to avoid circular dependencies.
        # This adds the table to the metadata.
        sqla.Table(
            table_name,
            registry.metadata,
            *cls._columns(registry),
            schema=(sub.schema if sub is not None else None),
        )

        # Re-create the table object with foreign keys and return it.
        return sqla.Table(
            table_name,
            registry.metadata,
            *cls._columns(registry),
            *cls._foreign_keys(metadata, subs),
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
        cls, target: type[Rec]
    ) -> set[RelSet[Self, SingleIdx, None, Any, Rec]]:
        """Get all direct relations from a target record type to this type."""
        return {
            RelSet[Self, SingleIdx, None, Any, Rec](
                prop=Rel(on=rel.prop.on, _type=PropType(Rel[list[cls], Any, Any, cls])),
                parent_type=cast(type[Rec], rel.fk_record_type),
            )
            for rel in cls._rels.values()
            if issubclass(target, rel.fk_record_type)
        }

    @overload
    @classmethod
    def _rel(
        cls, other: type[Rec], index: None = ...
    ) -> RelSet[Rec, BaseIdx, None, R, Self]: ...

    @overload
    @classmethod
    def _rel(
        cls, other: type[Rec], index: type[Key]
    ) -> RelSet[Rec, SingleIdx, None, R, Self]: ...

    @classmethod
    def _rel(
        cls, other: type[Rec], index: type[Key] | None = None
    ) -> RelSet[Rec, Any, None, R, Self]:
        """Dynamically define a relation to another record type."""
        return RelSet[Rec, Any, None, R, Self](
            prop=Rel(on=other, _type=PropType(Rel[list[other], Any, Any, other])),
            parent_type=cls,
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:  # noqa: D105
        assert cls._default_table_name() is not None
        return sqla.table(cls._default_table_name())

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
        include: tuple[type[Prop], ...] = ...,
        with_links: bool = ...,
    ) -> dict[Set[Self, Any], Any]: ...

    @overload
    def _to_dict(
        self,
        name_keys: Literal[True],
        only_loaded: bool = ...,
        load: bool = ...,
        include: tuple[type[Prop], ...] = ...,
        with_links: Literal[False] = ...,
    ) -> dict[str, Any]: ...

    def _to_dict(
        self,
        name_keys: bool = False,
        only_loaded: bool = True,
        load: bool = False,
        include: tuple[type[Prop], ...] = (Attr, Rel),
        with_links: bool = False,
    ) -> dict[Set[Self, Any], Any] | dict[str, Any]:
        """Convert the record to a dictionary."""

        def getter(r, n):
            return r.__dict__[n] if not load else getattr(r, n)

        vals = {
            p if not name_keys else p.name: getter(self, p.name)
            for p in type(self)._props.values()
            if isinstance(p, include)
        }

        if with_links and Rel in include and not name_keys:
            vals = {
                **vals,
                **{
                    cast(
                        RelSet[Any, Any, Any, Any, Any, Record, Any, None], r
                    ).link_set: self._edge_dict[r.prop.name]
                    for r in type(self)._rels.values()
                    if issubclass(r.link_type, Record)
                    and r.prop.name in self._edge_dict
                },
            }

        return cast(
            dict,
            (
                vals
                if not only_loaded
                else {k: v for k, v in vals.items() if not isinstance(v, Unloaded)}
            ),
        )

    @property
    def _index(self: Record[Key]) -> Key:
        """Return the index of the record."""
        pks = type(self)._primary_keys
        if len(pks) == 1:
            return getattr(self, next(iter(pks)))
        return cast(Key, tuple(getattr(self, pk) for pk in pks))

    @classmethod
    def _index_from_dict(cls: type[Record[Key]], data: Mapping[Set, Any]) -> Key | None:
        """Return the index contained in a dict representation of this record."""
        pks = cls._primary_keys

        values = tuple(data.get(pk) for pk in pks.values())
        if any(v is None for v in values):
            return None

        return cast(Key, values if len(values) == 1 else values[0])

    @property
    def _links(self) -> RecordLinks[Self]:
        """Descriptor to access the link records of a record's rels."""
        return RecordLinks(self)

    @classmethod
    def _set(cls) -> RecordSet[Self, Key_def, None, Any, None, Any]:
        return RecordSet(item_type=cls)


@dataclass
class RelTree(Generic[*RelTup]):
    """Tree of relations starting from the same root."""

    rels: Iterable[RelSet] = field(default_factory=set)

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

    def __rmul__(self, other: RelSet[Rec] | RelTree) -> RelTree[*RelTup, Rec]:
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    def __or__(
        self: RelTree[Rec], other: RelSet[Rec2] | RelTree[Rec2]
    ) -> RelTree[Rec | Rec2]:
        """Add more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])


@dataclass(kw_only=True, eq=False)
class Set(Generic[Val_def, KeyIdx_def, B_nul, Rec2_nul, P_nul]):
    """Reference a property of a record."""

    prop: P_nul = None
    item_type: type[Val_def] | None = None

    keys: Sequence[slice | list[Hashable] | Hashable] = field(default_factory=list)

    schema_types: set[type[Record]] = field(default_factory=set)
    overlay: str | None = None
    subs: dict[type[Record], sqla.TableClause] = field(default_factory=dict)
    create_cross_fk: bool = True
    overlay_with_schemas: bool = True

    backend: B_nul = None

    @cached_property
    def value_type(self) -> SingleTypeDef[Val_def]:
        """Value type of the property."""
        if self.item_type is not None:
            return self.item_type

        assert self.prop is not None
        return self.prop._type.value_type()

    @cached_property
    def record_type(self) -> type[Record]:
        """Record type of the set."""
        raise NotImplementedError()

    @cached_property
    def parent_type(self) -> type[Rec2_nul]:
        """Parent record type of the set."""
        raise NotImplementedError()

    @cached_property
    def idx(self) -> set[ValueSet]:
        """Return the index attrs."""
        path_idx = None
        if isinstance(self, RelSet):
            path_idx = self.path_idx
        elif isinstance(self, ValueSet) and isinstance(self.record_set, RelSet):
            path_idx = self.record_set.path_idx

        if path_idx is not None:
            return {p for p in path_idx}

        if isinstance(self, RecordSet):
            return set(self.record_type._primary_keys.values())

        return set()

    @cached_property
    def metadata(self) -> sqla.MetaData:
        """Metadata object for this DB instance."""
        return sqla.MetaData()

    @cached_property
    def db(self: Set[Any, Any, B_def]) -> DB[B_def]:
        """Return the database object."""
        return DB(backend=self.backend)

    def execute[
        *T
    ](
        self: Set[Any, Any, Backend],
        stmt: sqla.Select[tuple[*T]] | sqla.Insert | sqla.Update | sqla.Delete,
    ) -> sqla.Result[tuple[*T]]:
        """Execute a SQL statement in this database's context."""
        stmt = self._parse_expr(stmt)
        with self.backend.engine.begin() as conn:
            return conn.execute(self._parse_expr(stmt))

    def dataset(
        self: Set[Any, Any, B_def],
        data: DataFrame | sqla.Select,
        foreign_keys: Mapping[str, ValueSet] | None = None,
    ) -> RecordSet[DynRecord, Any, B_def, RO]:
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, DataFrame)
            else f"temp_{token_hex(5)}"
        )

        rec = dynamic_record_type(name, props=props_from_data(data, foreign_keys))
        self._get_table(rec, writable=True)
        ds = RecordSet[DynRecord, BaseIdx, B_def, RW](
            item_type=rec, backend=self.backend
        )
        ds &= data

        return ds  # type: ignore

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(self)

    def __setitem__(self, key: Any, other: Set) -> None:
        """Catchall setitem."""
        raise NotImplementedError()

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

        table = rec._table(self.metadata, self.subs)

        # Create any missing tables in the database.
        if self.backend is not None:
            self.metadata.create_all(self.backend.engine)

        return table

    def _get_joined_table(self, rec: type[Record]) -> sqla.Table | sqla.Join:
        table = rec._joined_table(self.metadata, self.subs)

        # Create any missing tables in the database.
        if self.backend is not None:
            self.metadata.create_all(self.backend.engine)

        return table

    def _get_alias(self, rel: RelSet[Any, Any, Any, Any, Any, Any]) -> sqla.FromClause:
        """Get alias for a relation reference."""
        return self._get_joined_table(rel.record_type).alias(gen_str_hash(rel, 8))

    def _get_random_alias(self, rec: type[Record]) -> sqla.FromClause:
        """Get random alias for a type."""
        return self._get_joined_table(rec).alias(token_hex(4))

    def _replace_attr(
        self,
        element: sqla_visitors.ExternallyTraversible,
        reflist: set[RelSet[Any, Any, Any, Any, Any]] = set(),
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if isinstance(element, ValueSet):
            if isinstance(element.record_set, RelSet):
                reflist.add(element.record_set)

            if isinstance(self, RelSet):
                element.record_set = element.record_set.prefix(self)

            table = (
                self._get_alias(element.record_set)
                if isinstance(element.record_set, RelSet)
                else self._get_table(element.record_set.record_type)
            )
            return table.c[element.prop.name]

        return None

    def _parse_filter(
        self,
        key: sqla.ColumnElement[bool],
    ) -> tuple[sqla.ColumnElement[bool], RelTree]:
        """Parse filter argument and return SQL expression and join operations."""
        reflist: set[RelSet] = set()
        replace_func = partial(self._replace_attr, reflist=reflist)
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
        elif isinstance(element, ValueSet):
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

    def _ensure_schema_exists(self: Set[Any, Any, Backend], schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.backend.engine).has_schema(schema_name):
            with self.backend.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name

    def _table_exists(self: Set[Any, Any, Backend], sqla_table: sqla.Table) -> bool:
        """Check if a table exists in the database."""
        return sqla.inspect(self.backend.engine).has_table(
            sqla_table.name, schema=sqla_table.schema
        )

    def _create_sqla_table(
        self: Set[Any, Any, Backend], sqla_table: sqla.Table
    ) -> None:
        """Create SQL-side table from Table class."""
        if not self.create_cross_fk:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            _remove_external_fk(sqla_table)

        sqla_table.create(self.backend.engine)

    def _load_from_excel(
        self: Set[Any, Any, Backend], record_types: list[type[Record]] | None = None
    ) -> None:
        """Load all tables from Excel."""
        assert self.backend is not None
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        path = (
            self.backend.url.get()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url
        )

        with open(path, "rb") as file:
            for rec in record_types or self.schema_types:
                pl.read_excel(
                    file, sheet_name=rec._default_table_name()
                ).write_database(
                    str(self._get_table(rec)), str(self.backend.engine.url)
                )

    def _save_to_excel(
        self: Set[Any, Any, Backend], record_types: Iterable[type[Record]] | None = None
    ) -> None:
        """Save all (or selected) tables to Excel."""
        assert self.backend is not None
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        file = (
            BytesIO()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url.open("wb")
        )

        with ExcelWorkbook(file) as wb:
            for rec in record_types or self.schema_types:
                pl.read_database(
                    f"SELECT * FROM {self._get_table(rec)}",
                    self.backend.engine,
                ).write_excel(wb, worksheet=rec._default_table_name())

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)

    def _delete_from_excel(
        self: Set[Any, Any, Backend], record_types: Iterable[type[Record]]
    ) -> None:
        """Delete selected table from Excel."""
        assert self.backend is not None
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        file = (
            BytesIO()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url.open("wb")
        )

        wb = openpyxl.load_workbook(file)
        for rec in record_types or self.schema_types:
            del wb[rec._default_table_name()]

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)

        raise TypeError("Invalid property reference.")


@dataclass(kw_only=True, eq=False)
class ValueSet(
    Set[Val_defi, KeyIdx_def, B_nul, Rec_def, Attr[Val_defi, R_def]],
    sqla.ColumnClause[Val_defi],
):
    """Reference an attribute of a record."""

    def __post_init__(self) -> None:  # noqa: D105
        # Initialize fields required by SQLAlchemy superclass.
        self.table = None
        self.is_literal = False

    # Replace fields of SQLAlchemy superclass with properties:

    @property
    def name(self) -> str:
        """Column key."""
        assert Prop.name.fget is not None
        return Prop.name.fget(self)

    @property
    def key(self) -> str:
        """Column key."""
        return self.name

    @cached_property
    def sql_type(self) -> sqla_types.TypeEngine:
        """Column key."""
        return sqla_types.to_instance(self.value_type)  # type: ignore

    def all(self) -> sqla.CollectionAggregate[bool]:
        """Return a SQL ALL expression for this attribute."""
        return sqla.all_(self)

    def any(self) -> sqla.CollectionAggregate[bool]:
        """Return a SQL ANY expression for this attribute."""
        return sqla.any_(self)

    # Value set interface:

    record_set: RecordSet[
        Rec_def, KeyIdx_def | BaseIdx | FilteredIdx[BaseIdx], B_nul, R_def
    ] = field(default_factory=Record._set)

    @cached_property
    def record_type(self) -> type[Rec_def]:
        """Record type of the set."""
        return self.record_set.record_type

    @cached_property
    def parent_type(self) -> type[Rec_def]:
        """Parent record type of the set."""
        return self.record_set.record_type

    # Plural selection
    @overload
    def __getitem__(
        self: ValueSet[Any, Key],
        key: Iterable[Key_def] | slice | tuple[slice, ...],
    ) -> ValueSet[Val_def, Key, B_nul, Rec_def]: ...

    # Single value selection
    @overload
    def __getitem__(
        self: ValueSet[Any, Key], key: Key
    ) -> ValueSet[Val_def, SingleIdx, B_nul, Rec_def]: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self: ValueSet,
        key: list[Hashable] | slice | tuple[slice, ...] | Hashable,
    ) -> ValueSet:
        return copy_and_override(self, ValueSet, keys=[key])

    def select(self) -> sqla.Select:
        """Return select statement for this dataset."""
        selection_table = self.record_set.root_table
        assert selection_table is not None

        return sqla.select(
            *(selection_table.c[a.prop.name] for a in self.idx),
            selection_table.c[self.prop.name],
        ).select_from(selection_table)

    @overload
    def load(  # type: ignore
        self: ValueSet[Val_cov, SingleIdx, Backend],
        kind: type[Val_cov] = ...,
    ) -> Val_cov: ...

    @overload
    def load(self: ValueSet[Any, Any, Backend], kind: type[Series]) -> Series: ...

    @overload
    def load(
        self: ValueSet[Val_cov, Key, Backend], kind: type[Val_cov] = ...
    ) -> dict[Key, Val_cov]: ...

    def load(
        self: ValueSet[Any, Any, Backend],
        kind: type[Val_cov | Series] = Record,
    ) -> Val_cov | Series | dict[Any, Val_cov]:
        """Download selection."""
        select = self.select()

        if kind is pd.Series:
            with self.engine.connect() as con:
                return pd.read_sql(select, con).set_index(
                    [c.key for c in self._idx_cols]
                )[self.prop.name]
        elif kind is pl.Series:
            return pl.read_database(select, self.engine)[self.prop.name]

        with self.engine.connect() as con:
            return (
                pd.read_sql(select, con)
                .set_index([c.key for c in self._idx_cols])[self.prop.name]
                .to_dict()
            )

    def __imatmul__(
        self: ValueSet[Val_cov, KeyIdx_def, B_def, Rec_def, R_def],
        value: ValInput | ValueSet[Val_cov, Key_def, B_def],
    ) -> ValueSet[Val_cov, KeyIdx_def, B_def, Rec_def, R_def]:
        """Do item-wise, broadcasting assignment on this dataset."""
        # Load data into a temporary table.
        value_df = None
        if isinstance(value, ValueSet):
            value_set = value
        elif isinstance(value, sqla.Select):
            value_set = self._db.dataset(value)
        else:
            value_df = (
                value.to_frame()
                if isinstance(value, Series)
                else (
                    pd.DataFrame.from_records(value)
                    if isinstance(value, Mapping)
                    else pd.DataFrame({self.prop.name: [value]})
                )
            )
            value_set = self._db.dataset(value_df)

        # Derive current select statement and join with value table, if exists.
        select = self.select()
        if value_set is not None:
            select = select.join(
                value_set.root_table,
                reduce(
                    sqla.and_,
                    (
                        self.root_table.c[idx_col.name] == idx_col
                        for idx_col in value_set.root_table.primary_key
                    ),
                ),
            )

        assert isinstance(self.root_table, sqla.Table), "Base must be a table."

        if self.engine.dialect.name in (
            "postgres",
            "postgresql",
            "duckdb",
            "mysql",
            "mariadb",
        ):
            # Update-from.
            statement = (
                self.root_table.update()
                .values(
                    {c_name: c for c_name, c in value_set.root_table.columns.items()}
                )
                .where(
                    reduce(
                        sqla.and_,
                        (
                            self.root_table.c[col.name] == select.c[col.name]
                            for col in self.root_table.primary_key.columns
                        ),
                    )
                )
            )

            with self.engine.begin() as con:
                con.execute(statement)
        else:
            # Correlated update.
            raise NotImplementedError("Correlated update not supported yet.")

        # Drop the temporary table, if any.
        if value_set is not None and value_set is not value:
            cast(sqla.Table, value_set.root_table).drop(self.engine)

        return self


type AggMap[Rec: Record] = dict[ValueSet[Rec, Any], ValueSet | sqla.Function]


@dataclass(kw_only=True, frozen=True)
class Agg(Generic[Rec]):
    """Define an aggregation map."""

    target: type[Rec]
    map: AggMap[Rec]


type Join = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


@dataclass(kw_only=True, eq=False)
class RecordSet(
    Set[
        RecordValue[Rec_cov],
        Rec2_nul,
        B_nul,
        Rec2_nul,
        Rel[RecordValue[Rec_cov], Record, R_def, Record] | None,
    ],
    Generic[Rec_cov, Idx_def, B_nul, R_def, Rec2_nul, RM],
):
    """Dataset."""

    parent_type: type[Rec2_nul] = type(None)  # type: ignore
    filters: list[sqla.ColumnElement[bool]] = field(default_factory=list)
    merges: RelTree = field(default_factory=RelTree)

    @staticmethod
    def _get_tag(rec_type: type[Record]) -> RelSet | None:
        """Retrieve relation-tag of a record type, if any."""
        try:
            rel = getattr(rec_type, "_rel")
            return rel if isinstance(rel, RelSet) else None
        except AttributeError:
            return None

    @cached_property
    def _parent(self) -> RelSet | None:
        """Parent relation of this Rel."""
        return (
            self._get_tag(self.parent_type)
            if issubclass(self.parent_type, Record)
            else None
        )

    @cached_property
    def _rel_path(self) -> tuple[type[Record], *tuple[RelSet, ...]] | tuple[()]:
        """Path from base record type to this Rel."""
        if not issubclass(self.parent_type, Record):
            return tuple()

        if self._parent is None:
            return (self.parent_type,)

        return cast(
            tuple[type[Record], *tuple[RelSet, ...]],
            (
                *self._parent._rel_path,
                *([self] if isinstance(self, RelSet) else []),
            ),
        )

    @cached_property
    def _idx_cols(self: RecordSet[Record, Any, B_def]) -> list[sqla.ColumnElement]:
        """Return the index columns."""
        return [
            *(
                col.label(f"{self.record_type._default_table_name()}.{col_name}")
                for col_name, col in self.root_table.columns.items()
                if self.record_type._attrs[col_name] in self.idx
            ),
            *(
                col.label(f"{rel.path_str}.{col_name}")
                for rel in self.merges.rels
                for col_name, col in self._get_alias(rel).columns.items()
                if rel.record_type._attrs[col_name] in self.idx
            ),
        ]

    @cached_property
    def record_type(self) -> type[Rec_cov]:
        """Record type of the set."""
        if self.prop is not None:
            return cast(type[Rec_cov], self.prop._type.record_type())

        assert self.item_type is not None
        return cast(type[Rec_cov], self.item_type)

    @cached_property
    def root_type(self) -> type[Record]:
        """Root record type of the set."""
        if len(self._rel_path) > 0:
            return self._rel_path[0]

        return self.record_type

    @cached_property
    def root_table(self: RecordSet[Any, Any, Backend]) -> sqla.FromClause:
        """Get the main table for the current selection."""
        return self._get_random_alias(self.root_type)

    def prefix(
        self: RecordSet[Any, Any, Any, Any, Rec2 | None, Any],
        left: type[Rec2] | RelSet[Rec2, Any, Any, Any, Any, Any, Any, Any],
    ) -> RelSet[Rec_cov, Idx_def, B_nul, R_def, Rec2, Any, Rec_cov, RM]:
        """Prefix this prop with a relation or record type."""
        current_root = self.root_type
        new_root = left if isinstance(left, RelSet) else left._rel(current_root)

        rel_path = self._rel_path[1:] if len(self._rel_path) > 1 else (self,)

        prefixed_rel = reduce(
            lambda r1, r2: copy_and_override(
                r2,
                RelSet,
                parent_type=r1.rec,
                keys=[*r2.keys, *r1.keys],
                filters=[*r2.filters, *r1.filters],
                merges=r2.merges * r1.merges,
            ),
            rel_path,
            new_root,
        )

        return cast(
            RelSet[Rec_cov, Idx_def, B_nul, R_def, Rec2, Any, Rec_cov, RM],
            prefixed_rel,
        )

    def suffix(
        self, left: RelSet[Rec, Any, Any, Any, Rec_cov, Any, Any]
    ) -> RelSet[Rec, Any, B_nul, R_def, Record, Any, Record]:
        """Prefix this prop with a relation or record type."""
        rel_path = left._rel_path[1:] if len(left._rel_path) > 1 else (left,)

        prefixed_rel = reduce(
            lambda r1, r2: copy_and_override(
                r2,
                RelSet,
                parent_type=r1.rec,
                keys=[*r2.keys, *r1.keys],
                filters=[*r2.filters, *r1.filters],
                merges=r2.merges * r1.merges,
            ),
            rel_path,
            cast(RelSet, self),
        )

        return cast(
            RelSet[Rec, Any, B_nul, R_def, Record, Any, Record],
            prefixed_rel,
        )

    @cached_property
    def rec(self) -> type[Rec_cov]:
        """Reference props of the target record type."""
        return cast(
            type[Rec_cov], type(token_hex(5), (self.record_type,), {"_rel": self})
        )

    # Overloads: attribute selection:

    # 1. DB-level type selection
    @overload
    def __getitem__(  # type: ignore
        self,
        key: type[Rec],
    ) -> RecordSet[Rec, Idx_def, B_nul, R_def, Rec2_nul, None]: ...

    # 2. Top-level attribute selection, custom index
    @overload
    def __getitem__(
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: ValueSet[Val, Rec_cov],
    ) -> ValueSet[Val, Key, B_nul, Rec_cov, R_def]: ...

    # 3. Top-level attribute selection, base index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], Any],
        key: ValueSet[Val, Rec_cov],
    ) -> ValueSet[Val, Key2, B_nul, Rec_cov, R_def]: ...

    # 4. Nested attribute selection, custom index
    @overload
    def __getitem__(
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: ValueSet[Val, Any],
    ) -> ValueSet[Val, Key | IdxStart[Key], B_nul, Record, R_def]: ...

    # 5. Nested attribute selection, base index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], Any],
        key: ValueSet[Val, Any],
    ) -> ValueSet[Val, Key2 | IdxStart[Key2], B_nul, Record, R_def]: ...

    # Overloads: relation selection:

    # 6. Top-level relation selection, singular, base index
    @overload
    def __getitem__(  # type: ignore
        self: RecordSet[Record[Key2], BaseIdx | FilteredIdx[BaseIdx]],
        key: RelSet[Rec2, SingleIdx | None, None, Any, Rec_cov, Rec3_nul],
    ) -> RelSet[Rec2, Key2, B_nul, R_def, Rec_cov, Rec3_nul, Rec2, None]: ...

    # 7. Top-level relation selection, singular, single index
    @overload
    def __getitem__(  # type: ignore
        self: RecordSet[Any, SingleIdx],
        key: RelSet[Rec2, SingleIdx, None, Any, Rec_cov, Rec3_nul],
    ) -> RelSet[Rec2, SingleIdx, B_nul, R_def, Rec_cov, Rec3_nul, Rec2, None]: ...

    # 8. Top-level relation selection, singular nullable, single index
    @overload
    def __getitem__(
        self: RecordSet[Any, SingleIdx],
        key: RelSet[Rec2, SingleIdx | None, None, Any, Rec_cov, Rec3_nul],
    ) -> RelSet[
        Rec2 | Scalar[None], SingleIdx, B_nul, R_def, Rec_cov, Rec3_nul, Rec2, None
    ]: ...

    # 9. Top-level relation selection, singular, custom index
    @overload
    def __getitem__(  # type: ignore
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: RelSet[Rec2, SingleIdx | None, None, Any, Rec_cov, Rec3_nul],
    ) -> RelSet[Rec2, Key, B_nul, R_def, Rec_cov, Rec3_nul, Rec2, None]: ...

    # 10. Top-level relation selection, base plural, base index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], BaseIdx | FilteredIdx[BaseIdx]],
        key: RelSet[
            Rec2,
            BaseIdx | FilteredIdx[BaseIdx],
            None,
            Any,
            Rec_cov,
            Rec3_opt,
            Record[Key4],
        ],
    ) -> RelSet[
        Rec2, tuple[Key2, Key4], B_nul, R_def, Rec_cov, Rec3_opt, Rec2, None
    ]: ...

    # 11. Top-level relation selection, base plural, single index
    @overload
    def __getitem__(  # type: ignore
        self: RecordSet[Any, SingleIdx],
        key: RelSet[
            Rec2,
            BaseIdx | FilteredIdx[BaseIdx],
            None,
            Any,
            Rec_cov,
            Rec3_opt,
            Record[Key4],
        ],
    ) -> RelSet[Rec2, Key4, B_nul, R_def, Rec_cov, Rec3_opt, Rec2, None]: ...

    # 12. Top-level relation selection, base plural, tuple index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], tuple[*IdxTup] | FilteredIdx[tuple[*IdxTup]]],
        key: RelSet[
            Rec2,
            BaseIdx | FilteredIdx[BaseIdx],
            None,
            Any,
            Rec_cov,
            Rec3_opt,
            Record[Key4],
        ],
    ) -> RelSet[
        Rec2,
        tuple[*IdxTup, Key4] | tuple[Key2, Key4],
        B_nul,
        R_def,
        Rec_cov,
        Rec3_opt,
        Rec2,
        None,
    ]: ...

    # 13. Top-level relation selection, base plural, custom index
    @overload
    def __getitem__(
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: RelSet[
            Rec2,
            BaseIdx | FilteredIdx[BaseIdx],
            None,
            Any,
            Rec_cov,
            Rec3_opt,
            Record[Key4],
        ],
    ) -> RelSet[
        Rec2, tuple[Key, Key4], B_nul, R_def, Rec_cov, Rec3_opt, Rec2, None
    ]: ...

    # 14. Top-level relation selection, plural, base index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], BaseIdx | FilteredIdx[BaseIdx]],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec_cov, Rec3_opt],
    ) -> RelSet[
        Rec2, tuple[Key2, Key3], B_nul, R_def, Rec_cov, Rec3_opt, Rec2, None
    ]: ...

    # 15. Top-level relation selection, plural, single index
    @overload
    def __getitem__(  # type: ignore
        self: RecordSet[Any, SingleIdx],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec_cov, Rec3_opt],
    ) -> RelSet[Rec2, Key3, B_nul, R_def, Rec_cov, Rec3_opt, Rec2, None]: ...

    # 16. Top-level relation selection, plural, tuple index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], tuple[*IdxTup] | FilteredIdx[tuple[*IdxTup]]],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec_cov, Rec3_opt],
    ) -> RelSet[
        Rec2,
        tuple[*IdxTup, Key3] | tuple[Key2, Key3],
        B_nul,
        R_def,
        Rec_cov,
        Rec3_opt,
        Rec2,
        None,
    ]: ...

    # 17. Top-level relation selection, plural, custom index
    @overload
    def __getitem__(
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec_cov, Rec3_opt],
    ) -> RelSet[
        Rec2, tuple[Key, Key3], B_nul, R_def, Rec_cov, Rec3_opt, Rec2, None
    ]: ...

    # 18. Nested relation selection, singular, base index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], BaseIdx | FilteredIdx[BaseIdx]],
        key: RelSet[Rec2, SingleIdx, None, Any, Rec, Rec3_opt],
    ) -> RelSet[Rec2, IdxStart[Key2], B_nul, R_def, Rec, Rec3_opt, Rec2, None]: ...

    # 19. Nested relation selection, singular, single index
    @overload
    def __getitem__(
        self: RecordSet[Any, SingleIdx],
        key: RelSet[Rec2, SingleIdx, None, Any, Rec, Rec3_opt],
    ) -> RelSet[
        Rec2, Hashable | SingleIdx, B_nul, R_def, Rec, Rec3_opt, Rec2, None
    ]: ...

    # 20. Nested relation selection, singular, tuple index
    @overload
    def __getitem__(
        self: RecordSet[Any, tuple[*IdxTup] | FilteredIdx[tuple[*IdxTup]]],
        key: RelSet[Rec2, SingleIdx, None, Any, Rec, Rec3_opt],
    ) -> RelSet[
        Rec2, IdxTupStart[*IdxTup], B_nul, R_def, Rec, Rec3_opt, Rec2, None
    ]: ...

    # 21. Nested relation selection, singular, custom index
    @overload
    def __getitem__(
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: RelSet[Rec2, SingleIdx, None, Any, Rec, Rec3_opt],
    ) -> RelSet[Rec2, IdxStart[Key], B_nul, R_def, Rec, Rec3_opt, Rec2, None]: ...

    # 22. Nested relation selection, base plural, base index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], BaseIdx | FilteredIdx[BaseIdx]],
        key: RelSet[
            Rec2, BaseIdx | FilteredIdx[BaseIdx], None, Any, Rec, Rec3_opt, Record[Key4]
        ],
    ) -> RelSet[
        Rec2, IdxStartEnd[Key2, Key4], B_nul, R_def, Rec, Rec3_opt, Rec2, None
    ]: ...

    # 23. Nested relation selection, base plural, single index
    @overload
    def __getitem__(
        self: RecordSet[Any, SingleIdx],
        key: RelSet[
            Rec2, BaseIdx | FilteredIdx[BaseIdx], None, Any, Rec, Rec3_opt, Record[Key4]
        ],
    ) -> RelSet[Rec2, IdxEnd[Key4], B_nul, R_def, Rec, Rec3_opt, Rec2, None]: ...

    # 24. Nested relation selection, base plural, tuple index
    @overload
    def __getitem__(
        self: RecordSet[Any, tuple[*IdxTup] | FilteredIdx[tuple[*IdxTup]]],
        key: RelSet[
            Rec2, BaseIdx | FilteredIdx[BaseIdx], None, Any, Rec, Rec3_opt, Record[Key4]
        ],
    ) -> RelSet[
        Rec2, IdxTupStartEnd[*IdxTup, Key4], B_nul, R_def, Rec, Rec3_opt, Rec2, None
    ]: ...

    # 25. Nested relation selection, base plural, custom index
    @overload
    def __getitem__(
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: RelSet[
            Rec2, BaseIdx | FilteredIdx[BaseIdx], None, Any, Rec, Rec3_opt, Record[Key4]
        ],
    ) -> RelSet[
        Rec2, IdxStartEnd[Key, Key4], B_nul, R_def, Rec, Rec3_opt, Rec2, None
    ]: ...

    # 26. Nested relation selection, plural, base index
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], BaseIdx | FilteredIdx[BaseIdx]],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec, Rec3_nul],
    ) -> RelSet[
        Rec2, IdxStartEnd[Key2, Key3], B_nul, R_def, Rec, Rec3_nul, Rec2, None
    ]: ...

    # 27. Nested relation selection, plural, single index
    @overload
    def __getitem__(
        self: RecordSet[Any, SingleIdx],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec, Rec3_nul],
    ) -> RelSet[Rec2, IdxEnd[Key3], B_nul, R_def, Rec, Rec3_nul, Rec2, None]: ...

    # 28. Nested relation selection, plural, tuple index
    @overload
    def __getitem__(
        self: RecordSet[Any, tuple[*IdxTup] | FilteredIdx[tuple[*IdxTup]]],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec, Rec3_nul],
    ) -> RelSet[
        Rec2, IdxTupStartEnd[*IdxTup, Key3], B_nul, R_def, Rec, Rec3_nul, Rec2, None
    ]: ...

    # 29. Nested relation selection, plural, custom index
    @overload
    def __getitem__(
        self: RecordSet[Any, Key | FilteredIdx[Key]],
        key: RelSet[Rec2, Key3 | FilteredIdx[Key3], None, Any, Rec, Rec3_nul],
    ) -> RelSet[
        Rec2, IdxStartEnd[Key, Key3], B_nul, R_def, Rec, Rec3_nul, Rec2, None
    ]: ...

    # 30. Default relation selection
    @overload
    def __getitem__(
        self: RecordSet,
        key: RelSet[Rec2, Any, None, Any, Rec, Rec3_nul],
    ) -> RelSet[Rec2, Any, B_nul, R_def, Rec, Rec3_nul, Rec2, None]: ...

    # 31. RelSet: Merge selection, single index
    @overload
    def __getitem__(
        self: RelSet[Any, SingleIdx, Any, Any, Rec2, Rec3, Any, None],
        key: RelTree[Rec_cov, *RelTup],
    ) -> RelSet[
        Rec_cov,
        BaseIdx,
        B_nul,
        R_def,
        Rec2,
        Rec3,
        Rec_cov,
        tuple[Rec_cov, *RelTup],
    ]: ...

    # 32. RelSet: Merge selection, default
    @overload
    def __getitem__(
        self: RelSet[Any, Any, Any, Any, Rec2, Rec3, Any, None],
        key: RelTree[Rec_cov, *RelTup],
    ) -> RelSet[
        Rec_cov,
        Idx_def,
        B_nul,
        R_def,
        Rec2,
        Rec3,
        Rec_cov,
        tuple[Rec_cov, *RelTup],
    ]: ...

    # 33. RelSet: Expression filtering, keep index
    @overload
    def __getitem__(
        self: RelSet[Any, Any, Any, Any, Rec2, Rec3],
        key: sqla.ColumnElement[bool],
    ) -> RelSet[Rec_cov, FilteredIdx[Idx_def], B_nul, R_def, Rec2, Rec3, Rec_cov]: ...

    # 34. RelSet: List selection
    @overload
    def __getitem__(  # type: ignore
        self: RelSet[Record[Key2], Key | Index, Any, Any, Rec2_def, Rec3],
        key: Iterable[Key | Key2],
    ) -> RelSet[
        Rec_cov, FilteredIdx[Idx_def], B_nul, R_def, Rec2_def, Rec3, Rec_cov
    ]: ...

    # 35. RelSet: Slice selection
    @overload
    def __getitem__(
        self: RelSet[Any, Any, Any, Any, Rec2, Rec3],
        key: slice | tuple[slice, ...],
    ) -> RelSet[Rec_cov, FilteredIdx[Idx_def], B_nul, R_def, Rec2, Rec3, Rec_cov]: ...

    # 36. RelSet: Index value selection
    @overload
    def __getitem__(
        self: RelSet[Record[Key2], Key | Index, Any, Any, Rec2, Rec3],
        key: Key | Key2,
    ) -> RelSet[Rec_cov, SingleIdx, B_nul, R_def, Rec2, Rec3, Rec_cov]: ...

    # 37. Merge selection, single index
    @overload
    def __getitem__(
        self: RecordSet[Any, SingleIdx, Any, Any, Any, None],
        key: RelTree[Rec_cov, *RelTup],
    ) -> RecordSet[
        Rec_cov, BaseIdx, B_nul, R_def, Rec2_nul, tuple[Rec_cov, *RelTup]
    ]: ...

    # 38. Merge selection, default
    @overload
    def __getitem__(
        self: RecordSet[Any, Any, Any, Any, Any, None],
        key: RelTree[Rec_cov, *RelTup],
    ) -> RecordSet[
        Rec_cov, Idx_def, B_nul, R_def, Rec2_nul, tuple[Rec_cov, *RelTup]
    ]: ...

    # 39. Expression filtering, keep index
    @overload
    def __getitem__(
        self: RecordSet, key: sqla.ColumnElement[bool]
    ) -> RecordSet[Rec_cov, FilteredIdx[Idx_def], B_nul, R_def, Rec2_nul]: ...

    # 40. List selection
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], Key | Index], key: Iterable[Key | Key2]
    ) -> RecordSet[Rec_cov, FilteredIdx[Idx_def], B_nul, R_def, Rec2_nul]: ...

    # 41. Slice selection
    @overload
    def __getitem__(
        self: RecordSet[Any], key: slice | tuple[slice, ...]
    ) -> RecordSet[Rec_cov, FilteredIdx[Idx_def], B_nul, R_def, Rec2_nul]: ...

    # 42. Index value selection
    @overload
    def __getitem__(
        self: RecordSet[Record[Key2], Key | Index], key: Key | Key2
    ) -> RecordSet[Rec_cov, SingleIdx, B_nul, R_def, Rec2_nul]: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self: RecordSet[Any, Any, Any, Any, Any, Any],
        key: (
            type[Record]
            | ValueSet
            | RecordSet
            | RelTree
            | sqla.ColumnElement[bool]
            | list[Hashable]
            | slice
            | tuple[slice, ...]
            | Hashable
        ),
    ) -> ValueSet[Any, Any, Any, Any, Any] | RecordSet[Any, Any, Any, Any, Any, Any]:
        match key:
            case type():
                assert issubclass(
                    key,
                    self.record_type,
                )
                return copy_and_override(self, type(self), item_type=key)
            case ValueSet():
                if not issubclass(self.record_type, key.record_set.record_type):
                    assert isinstance(key.record_set, RelSet)
                    return self.suffix(key.record_set)[key]

                return ValueSet(
                    record_set=self,
                    prop=key.prop,
                    item_type=key.item_type,
                    backend=self.backend,
                )
            case RelSet():
                return self.suffix(key)
            case RelTree():
                return copy_and_override(self, type(self), merges=self.merges * key)
            case sqla.ColumnElement():
                return copy_and_override(self, type(self), filters=[*self.filters, key])
            case list() | slice() | tuple() | Hashable():
                return copy_and_override(self, type(self), keys=[*self.keys, key])

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
                if not index_only or self.record_type._attrs[col_name] in self.idx
            ),
            *(
                col.label(f"{rel.path_str}.{col_name}")
                for rel in self.merges.rels
                for col_name, col in self._get_alias(rel).columns.items()
                if not index_only or rel.record_type._attrs[col_name] in self.idx
            ),
        ).select_from(selection_table)

        for join in self._joins():
            select = select.join(*join)

        for filt in self.filters:
            select = select.where(filt)

        return select

    @overload
    def load(  # type: ignore
        self: RecordSet[Rec_cov, SingleIdx, Backend, Any, Any, tuple[*RelTup]],
        kind: type[Record] = ...,
    ) -> tuple[Rec_cov, *RelTup]: ...

    @overload
    def load(  # type: ignore
        self: RecordSet[Rec_cov, SingleIdx, Backend],
        kind: type[Record] = ...,
    ) -> Rec_cov: ...

    @overload
    def load(
        self: RecordSet[Record, Any, Backend, Any, Any, tuple[*RelTup]],
        kind: type[Df],
    ) -> tuple[Df, ...]: ...

    @overload
    def load(self: RecordSet[Record, Any, Backend], kind: type[Df]) -> Df: ...

    @overload
    def load(
        self: RecordSet[Record, Key, Backend, Any, Any, tuple[*RelTup]],
        kind: type[Record] = ...,
    ) -> dict[Key, tuple[Rec_cov, *RelTup]]: ...

    @overload
    def load(
        self: RecordSet[Record[Key2], Any, Backend, Any, Any, tuple[*RelTup]],
        kind: type[Record] = ...,
    ) -> dict[Key2, tuple[Rec_cov, *RelTup]]: ...

    @overload
    def load(
        self: RecordSet[Rec_cov, Key, Backend], kind: type[Record] = ...
    ) -> dict[Key, Rec_cov]: ...

    @overload
    def load(
        self: RecordSet[Record[Key2], Any, Backend], kind: type[Record] = ...
    ) -> dict[Key2, Rec_cov]: ...

    def load(
        self: RecordSet[Record, Any, Backend, Any, Any, Any],
        kind: type[Record | Df] = Record,
    ) -> (
        Rec_cov
        | tuple[Rec_cov, *tuple[Any, ...]]
        | Df
        | tuple[Df, ...]
        | dict[Any, Rec_cov]
        | dict[Any, tuple[Rec_cov, *tuple[Any, ...]]]
    ):
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
            if rel.path_str is not None
        }

        merged_df = None
        if kind is pd.DataFrame:
            with self.backend.engine.connect() as con:
                merged_df = pd.read_sql(select, con)
                merged_df = merged_df.set_index(idx_cols)
        else:
            merged_df = pl.read_database(select, self.backend.engine)

        if issubclass(kind, Record):
            assert isinstance(merged_df, pl.DataFrame)

            rec_types = {
                self.parent_type: main_cols,
                **{rel.record_type: cols for rel, cols in extra_cols.items()},
            }

            loaded: dict[type[Record], dict[Hashable, Record]] = {
                r: {} for r in rec_types
            }
            records: dict[Any, Record | tuple[Record, ...]] = {}

            for row in merged_df.iter_rows(named=True):
                idx = tuple(row[i] for i in idx_cols)
                idx = idx[0] if len(idx) == 1 else idx

                rec_list = []
                for rec_type, cols in rec_types.items():
                    rec_data: dict[Set, Any] = {
                        getattr(rec_type, attr): row[col] for col, attr in cols.items()
                    }
                    rec_idx = self.record_type._index_from_dict(rec_data)
                    rec = loaded[rec_type].get(rec_idx) or self.parent_type(
                        _loader=self._load_prop,
                        **{
                            p.prop.name: v
                            for p, v in rec_data.items()
                            if p.prop is not None
                        },
                        **{r: Unloaded() for r in self.record_type._rels},
                    )

                    rec_list.append(rec)

                records[idx] = tuple(rec_list) if len(rec_list) > 1 else rec_list[0]

            return cast(
                dict[Any, Rec_cov] | dict[Any, tuple[Rec_cov, *tuple[Any, ...]]],
                records,
            )

        main_df, *extra_dfs = cast(
            tuple[Df, ...],
            (
                merged_df[list(main_cols.keys())].rename(main_cols),
                *(
                    merged_df[list(cols.keys())].rename(cols)
                    for cols in extra_cols.values()
                ),
            ),
        )

        return main_df, *extra_dfs

    def __imatmul__(
        self: RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None],
        other: RecordSet[Rec_cov, InsIdx, B_def] | RecInput[Rec_cov, InsIdx],
    ) -> RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None]:
        """Aligned assignment."""
        return self._mutate(other, mode="update")

    def __iand__(
        self: RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None],
        other: RecordSet[Rec_cov, InsIdx, B_def] | RecInput[Rec_cov, InsIdx],
    ) -> RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None]:
        """Replacing assignment."""
        return self._mutate(other, mode="replace")

    def __ior__(
        self: RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None],
        other: RecordSet[Rec_cov, InsIdx, B_def] | RecInput[Rec_cov, InsIdx],
    ) -> RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None]:
        """Upserting assignment."""
        return self._mutate(other, mode="upsert")

    def __iadd__(
        self: RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None],
        other: RecordSet[Rec_cov, InsIdx, B_def] | RecInput[Rec_cov, InsIdx],
    ) -> RecordSet[Any, InsIdx, B_def, RW, Rec2_nul, None]:
        """Inserting assignment."""
        return self._mutate(other, mode="insert")

    def __isub__(
        self: RecordSet[Any, InsIdx, B_def, RWT],
        other: RecordSet[Rec_cov, InsIdx, B_def] | Iterable[InsIdx] | InsIdx,
    ) -> RecordSet[Any, InsIdx, B_def, RWT, Rec2_nul, RM]:
        """Deletion."""
        raise NotImplementedError("Delete not supported yet.")

    @overload
    def __lshift__(
        self: RecordSet[Any, Key, B_def, RWT],
        other: RecordSet[Rec_cov, Key, B_def] | RecInput[Rec_cov, Key],
    ) -> list[Key]: ...

    @overload
    def __lshift__(
        self: RecordSet[Record[Key2], Any, B_def, RWT],
        other: RecordSet[Rec_cov, Any, B_def] | RecInput[Rec_cov, Key2],
    ) -> list[Key2]: ...

    def __lshift__(
        self: RecordSet[Any, Any, B_def, RWT],
        other: RecordSet[Rec_cov, Any, B_def] | RecInput[Rec_cov, Any],
    ) -> list:
        """Injection."""
        raise NotImplementedError("Inject not supported yet.")

    @overload
    def __rshift__(
        self: RecordSet[Any, Key, B_def, RW], other: Key | Iterable[Key]
    ) -> dict[Key, Rec_cov]: ...

    @overload
    def __rshift__(
        self: RecordSet[Record[Key2], Any, B_def, RW], other: Key2 | Iterable[Key2]
    ) -> dict[Key2, Rec_cov]: ...

    def __rshift__(
        self: RecordSet[Any, Any, B_def, RW], other: Hashable | Iterable[Hashable]
    ) -> dict[Any, Rec_cov]:
        """Extraction."""
        raise NotImplementedError("Extract not supported yet.")

    # 1. Type deletion
    @overload
    def __delitem__(self: RecordSet[Rec, Any, Backend, RW], key: type[Rec]) -> None: ...

    # 2. List deletion
    @overload
    def __delitem__(
        self: RecordSet[
            Record[Key2],
            BaseIdx | FilteredIdx[BaseIdx] | Key | FilteredIdx[Key],
            Backend,
            RW,
        ],
        key: Iterable[Key | Key2],
    ) -> None: ...

    # 3. Index value deletion
    @overload
    def __delitem__(
        self: RecordSet[
            Record[Key2],
            BaseIdx | FilteredIdx[BaseIdx] | Key | FilteredIdx[Key],
            Backend,
            RW,
        ],
        key: Key | Key2,
    ) -> None: ...

    # 4. Slice deletion
    @overload
    def __delitem__(
        self: RecordSet[Any, Any, Backend, RW], key: slice | tuple[slice, ...]
    ) -> None: ...

    # 5. Expression filter deletion
    @overload
    def __delitem__(
        self: RecordSet[Any, Any, Backend, RW], key: sqla.ColumnElement[bool]
    ) -> None: ...

    # Implementation:

    def __delitem__(  # noqa: D105
        self: RecordSet[Record, Any, Backend, RW, Any, Any],
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
            del self[key][:]  # type: ignore

        select = self.select()

        tables = {self._get_table(rec) for rec in self.record_type._record_bases}

        statements = []

        for table in tables:
            # Prepare delete statement.
            if self.backend.engine.dialect.name in (
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
        with self.backend.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

    def copy(
        self: RecordSet[Rec_cov, Idx_def, Backend],
        backend: B_def,
        # overlay: str | bool = True,
    ) -> RecordSet[Rec_cov, Idx_def, B_def, R_def, Rec2_nul, RM]:
        """Transfer the DB to a different backend (defaults to in-memory)."""
        other = copy_and_override(self, type(self), backend=backend)

        for rec in self.schema_types:
            self[rec].load(kind=pl.DataFrame).write_database(
                str(self._get_table(rec)), str(other.backend.url)
            )

        return cast(RecordSet[Rec_cov, Idx_def, B_def, R_def, Rec2_nul, RM], other)

    def extract(
        self: RecordSet[Record, Any, B_def],
        use_schema: bool | type[Schema] = False,
        aggs: Mapping[RelSet, Agg] | None = None,
    ) -> DB[B_def]:
        """Extract a new database instance from the current selection."""
        # Get all rec types in the schema.
        rec_types = (
            use_schema._record_types
            if isinstance(use_schema, type)
            else (
                self.schema_types
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
                                if isinstance(sa, ValueSet)
                                else sqla_visitors.replacement_traverse(
                                    sa,
                                    {},
                                    replace=lambda element, **kw: (
                                        src_select.c[element.name]
                                        if isinstance(element, ValueSet)
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
        new_db = DB(backend=self.backend, overlay=f"temp_{token_hex(10)}")

        # Overlay the new tables onto the new database.
        for rec in rec_types:
            if rec in replacements:
                new_db[rec] &= replacements[rec]

        for rec, agg_select in aggregations.items():
            new_db[rec] &= agg_select

        return new_db

    def __or__(
        self, other: RecordSet[Rec_cov, Idx, B_nul]
    ) -> RecordSet[Rec_cov, Idx_def | Idx]:
        """Union two datasets."""
        raise NotImplementedError("Union not supported yet.")

    def __len__(self: RecordSet[Any, Any, Backend]) -> int:
        """Return the number of records in the dataset."""
        with self.backend.engine.connect() as conn:
            res = conn.execute(
                sqla.select(sqla.func.count()).select_from(self.select().subquery())
            ).scalar()
            assert isinstance(res, int)
            return res

    def __contains__(self: RecordSet[Any, Any, Backend], key: Hashable) -> bool:
        """Check if a record is in the dataset."""
        return len(self[key]) > 0

    def __clause_element__(self: RecordSet[Any, Any, Backend]) -> sqla.Subquery:
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
                self._get_alias(rel._parent)
                if isinstance(rel._parent, RelSet)
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
        _traversed: set[RelSet] | None = None,
    ) -> set[RelSet]:
        """Find all paths to the target record type."""
        assert issubclass(self.parent_type, Record)

        backlink_records = backlink_records or set()
        _traversed = _traversed or set()

        # Get relations of the target type as next relations
        next_rels = set(self.parent_type._rels.values())

        for backlink_record in backlink_records:
            next_rels |= backlink_record._backrels_to_rels(self.parent_type)

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
        self: RecordSet[Any, Any, Backend],
        p: Set[Val, Any, None],
        parent_idx: Hashable,
    ) -> Val:
        base = self.db[p.record_type]
        base_record = base[parent_idx]

        if isinstance(p, ValueSet):
            return getattr(base_record.load(), p.name)
        elif isinstance(p, RecordSet):
            recs = base_record[p].load()
            recs_type = p.record_type

            if (
                isinstance(recs, dict)
                and not issubclass(recs_type, Mapping)
                and issubclass(recs_type, Iterable)
            ):
                recs = list(recs.values())

            if p.prop is not None and p.prop.collection is not None:
                recs = p.prop.collection(recs)

            return cast(Val, recs)

        raise TypeError("Invalid property reference.")

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
        rec_data: dict[Any, dict[Set, Any]], list_idx: bool, covered: set[int]
    ) -> dict[RelSet, dict[Any, Record]]:
        """Get relation data from record data."""
        rel_data: dict[RelSet, dict[Any, Record]] = {}

        for idx, rec in rec_data.items():
            for prop, prop_val in rec.items():
                if isinstance(prop, RelSet) and not isinstance(prop_val, Unloaded):
                    prop = cast(RelSet, prop)
                    rel_data[prop] = {
                        **rel_data.get(prop, {}),
                        **RecordSet._normalize_rel_data(
                            prop_val, idx, list_idx, covered
                        ),
                    }

        return rel_data

    def _mutate(  # noqa: C901, D105
        self: RecordSet[Any, Any, B_def, RW, Any, None],
        value: RecordSet | PartialRecInput[Rec_cov, Any] | ValInput,
        mode: Literal["update", "upsert", "replace", "insert"] = "update",
        covered: set[int] | None = None,
    ) -> RecordSet[Rec_cov, Idx_def, B_def, RW, Rec2_nul, None]:
        assert issubclass(self.parent_type, Record), "Record type must be defined."

        covered = covered or set()

        record_data: dict[Any, dict[Set, Any]] | None = None
        rel_data: dict[RelSet, dict[Any, Record]] | None = None
        df_data: DataFrame | None = None
        partial: bool = False

        list_idx = (
            isinstance(self, RelSet)
            and self.path_idx is not None
            and len(self.path_idx) == 1
            and is_subtype(self.path_idx[0].value_type, int)
        )

        if isinstance(value, Record):
            record_data = {value._index: value._to_dict(with_links=True)}
        elif isinstance(value, Mapping):
            if has_type(value, Mapping[Set, Any]):
                record_data = {
                    self.parent_type._index_from_dict(value): {
                        p: v for p, v in value.items()
                    }
                }
                partial = True
            else:
                assert has_type(value, Mapping[Any, PartialRec[Record]])

                record_data = {}
                for idx, rec in value.items():
                    if isinstance(rec, Record):
                        rec_dict = rec._to_dict(with_links=True)
                    else:
                        rec_dict = {p: v for p, v in rec.items()}
                        partial = True
                    record_data[idx] = rec_dict

        elif isinstance(value, DataFrame):
            df_data = value
        elif isinstance(value, Series):
            df_data = value.rename("_value").to_frame()
        elif isinstance(value, Iterable):
            record_data = {}
            for idx, rec in enumerate(value):
                if isinstance(rec, Record):
                    rec_dict = rec._to_dict(with_links=True)
                    rec_idx = rec._index
                else:
                    rec_dict = {p: v for p, v in rec.items()}
                    rec_idx = self.parent_type._index_from_dict(rec)
                    partial = True

                record_data[idx if list_idx else rec_idx] = rec_dict

        assert not (
            mode in ("upsert", "replace") and partial
        ), "Partial record input requires update mode."

        if record_data is not None:
            # Split record data into attribute and relation data.
            attr_data = {
                idx: {p.name: v for p, v in rec.items() if isinstance(p, ValueSet)}
                for idx, rec in record_data.items()
            }
            rel_data = self._get_record_rels(record_data, list_idx, covered)

            # Transform attribute data into DataFrame.
            df_data = pd.DataFrame.from_records(attr_data)

        if rel_data is not None:
            # Recurse into relation data.
            for r, r_data in rel_data.items():
                self[r]._mutate(r_data, mode="replace", covered=covered)

        # Load data into a temporary table.
        if df_data is not None:
            value_set = self.dataset(df_data)
        elif isinstance(value, sqla.Select):
            value_set = self.dataset(value)
        else:
            assert isinstance(value, RecordSet)
            value_set = value

        attrs_by_table = {
            self._get_table(rec): {
                a for a in self.parent_type._attrs.values() if a.parent_type is rec
            }
            for rec in self.parent_type._record_bases
        }

        statements = []

        if mode == "replace":
            # Delete all records in the current selection.
            select = self.select()

            for table in attrs_by_table:
                # Prepare delete statement.
                if self.backend.engine.dialect.name in (
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

            for table, attrs in attrs_by_table.items():
                # Do an insert-from-select operation, which updates on conflict:
                statement = table.insert().from_select(
                    [a.name for a in attrs],
                    value_set.select().subquery(),
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
            select = self.select().join(
                value_set.root_table,
                reduce(
                    sqla.and_,
                    (
                        self.root_table.c[idx_col.name] == idx_col
                        for idx_col in value_set.root_table.primary_key
                    ),
                ),
            )

            for table, attrs in attrs_by_table.items():
                attr_names = {a.name for a in attrs}
                values = {
                    c_name: c
                    for c_name, c in value_set.root_table.columns.items()
                    if c_name in attr_names
                }

                # Prepare update statement.
                if self.backend.engine.dialect.name in (
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
        with self.backend.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        # Drop the temporary table, if any.
        if value_set is not None:
            cast(sqla.Table, value_set.root_table).drop(self.backend.engine)

        if mode in ("replace", "upsert") and isinstance(self, RelSet):
            # Update incoming relations from parent records.
            if self.direct_rel is not True:
                if issubclass(self.fk_record_type, self.parent_type):
                    # Case: parent links directly to child (n -> 1)
                    for fk, pk in self.direct_rel.fk_map.items():
                        self.parent_set[fk] @= value_set[pk]
                else:
                    # Case: parent and child are linked via assoc table (n <--> m)
                    # Update link table with new child indexes.
                    assert issubclass(self.link_type, Record)
                    ls = self.link_set
                    ls &= value_set.select()

            # Note that the (1 <- n) case is already covered by updating
            # the child record directly, which includes all its foreign keys.

        return self  # type: ignore


@dataclass(kw_only=True, eq=False)
class RelSet(
    RecordSet[Rec_def, Idx_def, B_nul, R_def, Rec2_def, RM],
    Generic[Rec_def, Idx_def, B_nul, R_def, Rec2_def, Rec3_nul, Rec4_def, RM],
):
    """Relation set class."""

    prop: Rel[RecordValue[Rec_def], Rec3_nul, R_def, Rec_def]  # type: ignore

    def unbind(
        self,
    ) -> RelSet[Rec_def, Idx_def, None, R_def, Rec2_def, Rec3_nul, Rec4_def, None]:
        """Return backend-less version of this RelSet."""
        tmpl = cast(
            RelSet[Rec_def, Idx_def, None, R_def, Rec2_def, Rec3_nul, Rec4_def, None],
            self,
        )
        return copy_and_override(tmpl, type(tmpl), backend=None, merges=RelTree())

    @cached_property
    def path_idx(self) -> list[ValueSet] | None:
        """Get the path index of the relation."""
        if not isinstance(self, RelSet):
            return None

        p = [
            a
            for rel in self._rel_path[1:]
            if rel.prop is not None
            for a in (
                [rel.prop.map_by]
                if rel.prop.map_by is not None
                else rel.prop.order_by.keys() if rel.prop.order_by is not None else []
            )
        ]

        if len(p) == 0:
            return None

        return [
            *self.root_type._primary_keys.values(),
            *p,
        ]

    @cached_property
    def path_str(self) -> str | None:
        """String representation of the relation path."""
        if self.prop is None or not isinstance(self, RelSet):
            return None

        prefix = (
            self.parent_type.__name__
            if len(self._rel_path) == 0
            else self._rel_path[-1].path_str
        )
        return f"{prefix}.{self.prop.name}"

    @cached_property
    def parent_set(self) -> RecordSet[Rec2_def, Any, B_nul, R_def]:
        """Parent set of this Rel."""
        tmpl = (
            cast(RelSet[Rec2_def, Any, B_nul, R_def], self)
            if self._parent is not None
            else cast(RecordSet[Rec2_def, Any, B_nul, R_def], self)
        )

        if self._parent is not None:
            return copy_and_override(
                tmpl,
                type(tmpl),
                item_type=cast(type[Rec2_def], self._parent.item_type),
                prop=cast(Rel[Any, Any, R_def, Rec2_def], self._parent.prop),
                parent_type=self._parent.parent_type,
            )

        return copy_and_override(tmpl, type(tmpl))

    @property
    def link_type(self) -> type[Rec3_nul]:
        """Get the link record type."""
        return cast(
            type[Rec3_nul],
            (self.prop._type.link_type() if self.prop is not None else NoneType),
        )

    @property
    def link_set(
        self: RelSet[Any, Any, B_nul, R_def, Rec2_def, Rec3, Any, None],
    ) -> RecordSet[Rec3, Any, B_nul, R_def, Rec2_def, None]:
        """Get the link set."""
        r = self.parent_type._rel(self.link_type)
        return self.parent_set[r]

    @cached_property
    def link(
        self: RelSet[Any, Any, B_nul, R_def, Rec2_def, Rec3, Any, None]
    ) -> type[Rec3]:
        """Reference props of the link record type."""
        return (
            self.link_set.rec if self.link_set is not None else cast(type[Rec3], Record)
        )

    @cached_property
    def fk_record_type(self) -> type[Record]:
        """Record type of the foreign key."""
        match self.prop.on:
            case type():
                return self.prop.on
            case RelSet():
                return self.prop.on.parent_type
            case tuple():
                link = self.prop.on[0]
                assert isinstance(link, RecordSet)
                return link.parent_type
            case None if issubclass(self.link_type, Record):
                return self.link_type
            case dict() | ValueSet() | Iterable() | None:
                return self.parent_type

    @cached_property
    def direct_rel(
        self,
    ) -> RelSet[Rec2_def, SingleIdx, B_nul, R_def, Record] | Literal[True]:
        """Direct rel."""
        on = self.prop.on
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
                    RelSet[Rec2_def, SingleIdx, B_nul, R_def, Record],
                    rels[0],
                )
            case RelSet():
                return cast(
                    RelSet[Rec2_def, SingleIdx, B_nul, R_def, Record],
                    on,
                )
            case tuple():
                link = on[0]
                assert isinstance(link, RelSet)
                return cast(RelSet[Rec2_def, SingleIdx, B_nul, R_def, Record], link)
            case dict() | ValueSet() | Iterable() | None:
                return True

    @cached_property
    def counter_rel(self) -> RelSet[Rec2_def, Any, B_nul, R_def, Rec_def]:
        """Counter rel."""
        if self.direct_rel is not True and issubclass(
            self.direct_rel.parent_type, self.record_type
        ):
            return cast(RelSet[Rec2_def, Any, B_nul, R_def, Rec_def], self.direct_rel)

        return cast(
            RelSet[Rec2_def, Any, B_nul, R_def, Rec_def],
            self.record_type._rel(self.parent_type),
        )

    @cached_property
    def fk_map(
        self,
    ) -> bidict[ValueSet[Hashable, Any, None], ValueSet[Hashable, Any, None]]:
        """Map source foreign keys to target attrs."""
        target = self.record_type
        on = self.prop.on
        if on is None and issubclass(self.link_type, Record):
            on = self.link_type

        match on:
            case type() | RecordSet() | tuple():
                return bidict()
            case dict():
                return bidict(
                    {
                        ValueSet[Hashable, Any, None](
                            prop=Attr(
                                _name=fk.name, _type=PropType(Attr[fk.value_type])
                            ),
                            record_set=self.unbind(),
                        ): cast(ValueSet[Hashable, Any, None], pk)
                        for fk, pk in on.items()
                    }
                )
            case ValueSet() | Iterable():
                attrs = on if isinstance(on, Iterable) else [on]
                source_attrs = [
                    ValueSet[Hashable, Any, None](
                        prop=Attr(
                            _name=attr.name, _type=PropType(Attr[attr.value_type])
                        ),
                        record_set=self.unbind(),
                    )
                    for attr in attrs
                ]
                target_attrs = target._primary_keys.values()
                fk_map = dict(zip(source_attrs, target_attrs))

                assert all(
                    is_subtype(
                        self.parent_type._defined_attrs[fk_attr.name].value_type,
                        pk_attr.value_type,
                    )
                    for fk_attr, pk_attr in fk_map.items()
                    if pk_attr.typedef is not None
                ), "Foreign key value types must match primary key value types."

                return bidict(fk_map)
            case None:
                return bidict(
                    {
                        ValueSet[Hashable, Any, None](
                            prop=Attr(
                                _name=f"{self.prop.name}_{target_attr.prop.name}",
                                _type=PropType(Attr[target_attr.value_type]),
                            ),
                            record_set=self.unbind(),
                        ): cast(ValueSet[Hashable, Any, None], target_attr)
                        for target_attr in target._primary_keys.values()
                    }
                )

    @cached_property
    def inter_joins(
        self,
    ) -> dict[
        type[Record],
        list[Mapping[ValueSet[Hashable, Any, None], ValueSet[Hashable, Any, None]]],
    ]:
        """Intermediate joins required by this rel."""
        on = self.prop.on
        if on is None and issubclass(self.link_type, Record):
            on = self.link_type

        match on:
            case RecordSet():
                # Relation is defined via other relation
                other_rel = on
                assert isinstance(
                    other_rel, RecordSet
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
    ) -> list[Mapping[ValueSet[Hashable, Any, None], ValueSet[Hashable, Any, None]]]:
        """Mappings of column keys to join the target on."""
        on = self.prop.on
        if on is None and issubclass(self.link_type, Record):
            on = self.link_type

        match on:
            case RecordSet():
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


@dataclass(kw_only=True, eq=False)
class DB(RecordSet[Record, BaseIdx, B_def, RWT, None, None]):
    """Database class."""

    backend: B_def = Backend(name="default")
    item_type: type[Record] = Record  # type: ignore

    schema: (
        type[Schema]
        | Mapping[
            type[Schema],
            Require | str,
        ]
        | None
    ) = None
    records: (
        Mapping[
            type[Record],
            Require | str | sqla.TableClause,
        ]
        | None
    ) = None
    validate_on_init: bool = True

    def __post_init__(self):  # noqa: D105
        if self.records is not None:
            self.subs = {
                **self.subs,
                **{
                    rec: (sub if isinstance(sub, sqla.TableClause) else sqla.table(sub))
                    for rec, sub in self.records.items()
                    if not isinstance(sub, Require)
                },
            }
            self.schema_types |= set(self.records.keys())

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
                schemas = (
                    set(self.schema.keys())
                    if isinstance(self.schema, Mapping)
                    else set([self.schema])
                )
            else:
                schemas = {self.schema}

            self.schema_types |= {
                rec for schema in schemas for rec in schema._record_types
            }

        if self.validate_on_init:
            self.validate()

        if self.overlay is not None and self.overlay_with_schemas:
            self._ensure_schema_exists(self.overlay)

    @cached_property
    def assoc_types(self) -> set[type[Record]]:
        """Set of all association tables in this DB."""
        assoc_types = set()
        for rec in self.schema_types:
            pks = set([attr.name for attr in rec._primary_keys.values()])
            fks = set(
                [attr.name for rel in rec._rels.values() for attr in rel.fk_map.keys()]
            )
            if pks == fks:
                assoc_types.add(rec)

        return assoc_types

    @cached_property
    def relation_map(self) -> dict[type[Record], set[RelSet]]:
        """Maps all tables in this DB to their outgoing or incoming relations."""
        rels: dict[type[Record], set[RelSet]] = {
            table: set() for table in self.schema_types
        }

        for rec in self.schema_types:
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
        if self.schema is not None:
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
                asdict(self.backend)
                if self.backend is not None and self.backend.type != "in-memory"
                else None
            ),
        }

    def to_graph(
        self, nodes: Sequence[type[Rec_cov]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        node_tables = [self[n] for n in nodes]

        # Concat all node tables into one.
        node_dfs = [
            n.load(kind=pd.DataFrame)
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
                    .load(kind=pd.DataFrame)
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
                                *(a for a in (left_rel.parent_type or Record)._attrs),
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

        if self.records is not None:
            types |= {
                rec: isinstance(req, Require) and req.present
                for rec, req in self.records.items()
            }

        if isinstance(self.schema, Mapping):
            types |= {
                rec: isinstance(req, Require) and req.present
                for schema, req in self.schema.items()
                for rec in schema._record_types
            }

        tables = {self._get_table(rec): required for rec, required in types.items()}

        inspector = sqla.inspect(self.backend.engine)

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


class RecUUID(Record[UUID]):
    """Record type with a default UUID primary key."""

    _template = True
    _id: Attr[UUID] = prop(primary_key=True, default_factory=uuid4)


class RecHashed(Record[int]):
    """Record type with a default hashed primary key."""

    _template = True

    _id: Attr[int] = prop(primary_key=True, init=False)

    def __post_init__(self) -> None:  # noqa: D105
        self._id = gen_int_hash(
            {
                a.prop.name: getattr(self, a.prop.name)
                for a in self._defined_attrs.values()
            }
        )

    @classmethod
    def _index_from_dict(cls, data: Mapping[Set, Any]) -> int:
        """Return the index contained in a dict representation of this record."""
        return gen_int_hash(
            {a.prop.name: v for a, v in data.items() if isinstance(a, ValueSet)}
        )


class Scalar(Record[Key_def], Generic[Val, Key_def]):
    """Dynamically defined record type."""

    _template = True

    _id: Attr[Key_def] = prop(primary_key=True, default_factory=uuid4)
    _value: Attr[Val]


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls: type[Record], name: str) -> ValueSet:
        """Get dynamic attribute by dynamic name."""
        return ValueSet(
            prop=Attr(_name=name, _type=PropType(Attr[cls])), record_set=cls._set()
        )

    def __getattr__(cls: type[Record], name: str) -> ValueSet:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)

        return ValueSet(
            prop=Attr(_name=name, _type=PropType(Attr[cls])), record_set=cls._set()
        )


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""


a = DynRecord


def dynamic_record_type(name: str, props: Iterable[Prop] = []) -> type[DynRecord]:
    """Create a dynamically defined record type."""
    return type(name, (DynRecord,), {p.name: p for p in props})


class Link(RecHashed, Generic[Rec, Rec2]):
    """Automatically defined relation record type."""

    _template = True

    _from: Rel[Rec]
    _to: Rel[Rec2]


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
