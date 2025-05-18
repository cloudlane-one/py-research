"""Basis for relational data records."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Self, cast, overload

import sqlalchemy as sqla
import sqlalchemy.types as sqla_type
from typing_extensions import TypeVar

from py_research.caching import cached_prop
from py_research.hashing import gen_int_hash
from py_research.reflect.types import TypeRef
from py_research.types import UUID4, Not

from .data import (
    PL,
    AutoIndexable,
    Base,
    Col,
    Ctx,
    Data,
    Expand,
    Frame,
    Idx,
    Interface,
    KeyTt,
    ModIdx,
    R,
    RuTi,
    RwxT,
    Tab,
    U,
    ValT,
    ValT2,
)
from .models import Model, Prop

RecT = TypeVar("RecT", bound="Record", contravariant=True, default=Any)


@dataclass(kw_only=True, eq=False)
class Attr(
    Prop[ValT, Idx[()], RuTi, PL, Col, RecT],
    Data[ValT, Expand[Idx[()]], Col, PL, RwxT, Interface[RecT, Any, Tab]],
):
    """Single-value attribute or column."""

    data: Data[ValT, Expand[Idx[()]], Col, PL, RuTi, Interface[RecT, Any, Tab]] = field(
        init=False
    )
    context: Interface[RecT] | Data[Any, Any, Any, Any, Any, Interface[RecT]] = (
        Interface()
    )
    getter: Callable[[RecT], ValT | Literal[Not.resolved]] = field(init=False)

    def __post_init__(self) -> None:  # noqa: D105
        self.data = self
        self.getter = self._attr_getter
        self.setter = self._attr_setter

    def __set_name__(self, owner: type[RecT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)
        self.context = Interface(owner)

    def _attr_getter(self, instance: RecT) -> ValT | Literal[Not.resolved]:
        """Get the value of this attribute on an instance."""
        if self.name not in instance.__dict__:
            return Not.resolved

        return instance.__dict__[self.name]

    def _attr_setter(self: Attr[ValT2], instance: RecT, value: ValT2) -> None:
        """Set the value of this attribute on an instance."""
        instance.__dict__[self.name] = value

    def _index(
        self,
    ) -> Expand[Idx[()]]:
        """Get the index of this data."""
        return ModIdx(reduction=Idx[()], expansion=Idx(tuple()))

    def _frame(
        self: Data[Any, Any, Col, Any, Any, Base, Ctx[Any, Any, Tab]],
    ) -> Frame[PL, Col]:
        """Get SQL-side reference to this property."""
        parent = self.parent()
        assert parent is not None
        parent_df = parent.load()

        return Frame(parent_df[self._name()])


TupT = TypeVar("TupT", bound=tuple[Any, ...], default=Any)


@dataclass(kw_only=True, eq=False)
class AttrTuple(
    Prop[TupT, Idx[()], R, PL, Col | Tab, RecT],
):
    """Index property for a record type."""

    data: Data[TupT, Expand[Idx[()]], Col | Tab, PL, R, Interface[RecT, Any, Tab]] = (
        field(init=False)
    )
    getter: Callable[[RecT], TupT | Literal[Not.resolved]] = field(init=False)

    attrs: tuple[Attr[Any, Any, RecT], ...]

    def __post_init__(self) -> None:  # noqa: D105
        """Get the value of this attribute."""
        self.context = Interface()
        self.getter = self._attr_tuple_getter

        assert len(self.attrs) > 0
        data = (
            reduce(Data.__matmul__, self.attrs)
            if len(self.attrs) > 1
            else self.attrs[0].data
        )
        self.data = cast(
            Data[TupT, Expand[Idx[()]], Col | Tab, PL, R, Interface[RecT, Any, Tab]],
            data,
        )

    def _attr_tuple_getter(self, instance: RecT) -> TupT | Literal[Not.resolved]:
        """Get the value of this attribute tuple."""
        val = tuple(attr.__get__(instance, type(instance)) for attr in self.attrs)
        return cast(TupT, val[0] if len(val) == 1 else val)

    def gen_value_map(self, val: Any) -> dict[str, Hashable]:
        """Generate a key-value map for this key."""
        idx_names = [attr.name for attr in self.attrs]

        if len(idx_names) == 1:
            return {idx_names[0]: val}

        assert isinstance(val, tuple) and len(val) == len(idx_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(idx_names, val)}

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self.context, self.attrs))


@dataclass(eq=False)
class Key(AttrTuple[TupT, RecT]):
    """Index property for a record type."""

    attrs: tuple[Attr[Any, Any, RecT], ...] = ()

    def __set_name__(self, owner: type[RecT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if len(self.attrs) == 0:
            self.attrs = (
                Attr(
                    alias=self.name,
                    typeref=TypeRef(Attr[Hashable]),
                    context=Interface(owner),
                ),
            )
            setattr(owner, f"{name}_attr", self.attrs[0])


TgtT = TypeVar("TgtT", bound="Record", covariant=True)


@dataclass(eq=False)
class ForeignKey(AttrTuple[TupT, RecT], Generic[TgtT, TupT, RecT]):
    """Index property for a record type."""

    attrs: tuple[Attr[Any, Any, RecT], ...] = field(init=False)

    attr_map: dict[Attr[Any, Any, RecT], Attr[Any, Any, TgtT]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:  # noqa: D105
        self.attrs = tuple(self.attr_map.keys())

    def __set_name__(self, owner: type[RecT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if self.attr_map is None:
            pk_attrs = self.target._primary_key().attrs

            col_map = {
                Attr(
                    alias=f"{self.name}_{pk_attr.name}_fk",
                    typeref=TypeRef(Attr[pk_attr.common_value_type]),
                    context=Interface(owner),
                ): pk_attr
                for pk_attr in pk_attrs
            }

            self.columns = tuple(col_map.keys())
            for col in col_map.keys():
                setattr(owner, col.name, col)

    @property
    def target(self) -> type[TgtT]:
        """Return the target type for this foreign key."""
        target_type = self.typeargs[TgtT]
        assert isinstance(target_type, type)
        return cast(type[TgtT], target_type)


class Record(Model, AutoIndexable[*KeyTt]):
    """Schema for a table in a database."""

    _root_class = True

    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID4: sqla.types.CHAR(36),
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)

        for superclass in cls._model_superclasses:
            if superclass in cls.__bases__ and issubclass(superclass, Record):
                super_pk = superclass._primary_key()
                col_map = {
                    Attr[Hashable, Any, Self](
                        alias=pk_attr.name,
                        typeref=pk_attr.typeref,
                        context=Interface(cls),
                    ): pk_attr
                    for pk_attr in super_pk.attrs
                }
                for col in col_map.keys():
                    setattr(cls, col.name, col)

                pk = Key[cast(type[tuple], super_pk.typeargs[TupT]), cls](
                    attrs=tuple(col_map.keys())
                )
                setattr(cls, pk.name, pk)

                fk = ForeignKey[
                    superclass, cast(type[tuple], super_pk.typeargs[TupT]), cls
                ](attr_map=col_map)
                setattr(cls, fk.name, fk)

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        name = cls._table_name or cls._fqn
        fqn_parts = name.split(".")

        name = fqn_parts[-1]
        for part in reversed(fqn_parts[:-1]):
            name = part + "_" + name
            if len(name) > 40:
                break

        return name

    @classmethod
    def _attrs(cls) -> set[Attr]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Attr)
        }

    @classmethod
    def _keys(cls) -> set[Key]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Key)
        }

    @classmethod
    def _primary_key(cls) -> Key:
        """Primary key of this record type's table."""
        return cls.__dict__["_pk"]

    @classmethod
    def _foreign_keys(cls) -> set[ForeignKey]:
        """Columns of this record type's table."""
        return {
            prop
            for prop in cls._get_class_props().values()
            if isinstance(prop, ForeignKey)
        }

    _published: bool = False
    _base: Attr[Base] = Attr(default_factory=Base)

    _pk: Key[tuple[*KeyTt], Self]

    @cached_prop
    def _whereclause(self) -> sqla.ColumnElement[bool]:
        cols = {
            col.name: sql_col
            for col, sql_col in self._base._map_cols(type(self)).items()
        }
        pk_map = type(self)._pk.gen_value_map(self._pk)

        return sqla.and_(
            *(cols[col] == val for col, val in pk_map.items()),
        )

    def _load_dict(self) -> None:
        if not self._published or self._pk in self._base._get_valid_cache_set(
            type(self)
        ):
            return

        query = self._table.sql.select().where(self._whereclause)

        with self._base.engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()

        if row is not None:
            self.__dict__.update(row._asdict())

    def _save_dict(self) -> None:
        df = records_to_df([self])
        self._table.mutate(self._base.table({type(self)}, df))
        self._base._get_valid_cache_set(type(self)).remove(self._pk)

    def __hash__(self) -> int:
        """Identify the record by database and id."""
        return gen_int_hash((self._base if self._published else None, self._pk))

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)


ColT = TypeVar("ColT")


@dataclass(kw_only=True)
class Column(
    Attr[ColT, RwxT, RecT],
    sqla.SQLColumnExpression[ColT],
    Generic[ColT, RwxT, RecT, BackT],
):
    """Column property for a table model."""

    _table: Table[RecT] | None = None

    sql_default: sqla.ColumnElement[ColT] | None = None

    @cached_prop
    def sql_type(self) -> sqla_type.TypeEngine:
        """Return the SQL type of this column."""
        return sqla_type.to_instance(self.common_value_type)

    @property
    def sql_col(self) -> sqla.ColumnClause[ColT]:
        """Return the SQL column of this property."""
        table = (
            self._table
            if self._table is not None
            else (
                self._owner._table
                if self._owner is not None and issubclass(self._owner, Record)
                else None
            )
        )
        col = sqla.column(
            self.name,
            self.sql_type,
            _selectable=table.sql if table else None,
        )
        setattr(col, "_prop", self)
        return col

    def __clause_element__(self) -> sqla.ColumnClause[ColT]:
        """Return the column clause element."""
        return self.sql_col

    @cached_prop
    def sql_comparator(self) -> sqla_type.TypeEngine.Comparator:
        """Return the comparator for this column."""
        return self.sql_type.comparator_factory(self.sql_col)

    if not TYPE_CHECKING:

        def __getattr__(self, key: str) -> Any:
            """Support rich comparison methods."""
            return getattr(self.sql_comparator, key)

    def label(self, name: str | None) -> sqla.Label[ColT]:
        """Label this column."""
        return self.sql_col.label(name)

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> ColT: ...

    @overload
    def __get__(
        self, instance: None, owner: type[RecT]
    ) -> Column[ColT, RwxT, RecT, Symbolic]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride] (TODO: report pyright bug)
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this column."""
        super_get = Attr.__get__(self, instance, owner)
        if super_get is not Ungot():
            return super_get

        assert instance is not None

        if isinstance(instance, Record):
            instance._load_dict()

        if self.name not in instance.__dict__:
            if self.default is not Not.defined:
                instance.__dict__[self.name] = self.default
            else:
                assert self.default_factory is not None
                instance.__dict__[self.name] = self.default_factory()

        return instance.__dict__[self.name]

    def __set__(self: Column[Any, U, Any], instance: Model, value: ColT) -> None:
        """Set the value of this attribute."""
        super_set = Attr.__set__(self, instance, value)
        if super_set is Unset():
            return

        instance.__dict__[self.name] = value

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self._owner, self._table))
