"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from copy import copy
from dataclasses import MISSING, Field, dataclass, field
from functools import cmp_to_key, reduce
from inspect import get_annotations, getmodule
from types import ModuleType, NoneType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    cast,
    dataclass_transform,
    get_origin,
    overload,
)

import polars as pl
import sqlalchemy as sqla
import sqlalchemy.types as sqla_type
from typing_extensions import TypeVar

from py_research.caching import cached_prop
from py_research.data import Params, copy_and_override
from py_research.hashing import gen_int_hash
from py_research.reflect.types import GenericAlias, TypeRef, get_typevar_map
from py_research.types import UUID4, DataclassInstance, Not

from .data import (
    AddIdxT,
    Data,
    Dx,
    EngineT,
    FullIdxT,
    Idx,
    InputData,
    Interface,
    PassIdx,
    RwxT,
    ShapeT,
    Tab,
    U,
    ValT,
    ValT2,
    _get_data_type,
)

OwnT = TypeVar("OwnT", bound="Model", contravariant=True, default=Any)
OwnT2 = TypeVar("OwnT2", bound="Model")


@dataclass(kw_only=True)
class Prop(
    Data[
        ValT,
        PassIdx[Idx[*tuple[Any, ...]], Idx[()], AddIdxT],
        RwxT,
        Dx[EngineT, ShapeT],
        Interface[OwnT, Any, Tab],
    ]
):
    """Property definition for a model."""

    init_after: ClassVar[set[type[Prop]]] = set()

    context: (
        Interface[OwnT, Any, Tab] | Data[Any, Any, Any, Any, Interface[OwnT, Any, Tab]]
    ) = field(default_factory=Interface)

    alias: str | None = None
    default: ValT | InputData[ValT, ShapeT] | Literal[Not.defined] = Not.defined
    default_factory: Callable[[], ValT | InputData[ValT, ShapeT]] | None = None
    init: bool = True
    repr: bool = True
    hash: bool = True
    compare: bool = True

    # Extension methods:

    def _value_get(
        self, instance: OwnT
    ) -> ValT | Literal[Not.handled, Not.resolved, Not.defined]:
        """Get the scalar value of this property given an object instance."""
        return Not.handled

    def _value_set(
        self: Prop[Any, Any, U], instance: OwnT, value: Any
    ) -> None | Literal[Not.handled]:
        """Set the scalar value of this property given an object instance."""
        return Not.handled

    # Name and Ownership:

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        if self.alias is None:
            self.alias = name

        if self.typeref is None:
            typeargs = get_typevar_map(owner) | {OwnT: owner}

            self.typeref = TypeRef(
                get_annotations(owner)[name],
                var_map=typeargs,
                ctx_module=owner._src_mod or getmodule(owner),
            )

    def _name(self) -> str:
        """Name of the property."""
        assert self.alias is not None
        return self.alias

    @property
    def name(self) -> str:
        """Name of the property."""
        return self._name()

    # Descriptor read/write:

    @overload
    def __get__(
        self, instance: None, owner: type[OwnT2]
    ) -> Prop[ValT, AddIdxT, RwxT, EngineT, ShapeT, OwnT2]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(self, instance: Any, owner: type | None) -> Any:  # noqa: D105
        if owner is None or not issubclass(owner, Model) or instance is None:
            return self

        res = self._value_get(instance)
        assert res is not Not.handled
        return res

    @overload
    def __set__(  # noqa: D105
        self: Prop[ValT2, Any, U], instance: OwnT, value: ValT2
    ) -> None: ...

    @overload
    def __set__(  # noqa: D105
        self: Prop[Any, Any, U], instance: Any, value: Any
    ) -> None: ...

    def __set__(  # noqa: D105
        self: Prop[Any, Any, U], instance: Any, value: Any
    ) -> None:
        if isinstance(instance, Model):
            res = self._value_set(instance, value)
            assert res is not Not.handled


class ModelMeta(type):
    """Metaclass for record types."""

    _root_class: bool = False
    _template: bool = False
    _derivate: bool = False

    _src_mod: ModuleType | None = None
    _model_superclasses: list[type[Model]]
    __class_props: dict[str, Prop] | None

    def __init__(cls, name, bases, dct):
        """Initialize a new record type."""
        super().__init__(name, bases, dct)

        if "_src_mod" not in dct:
            cls._src_mod = getmodule(cls if not cls._derivate else bases[0])

        # Copy all supplied data instances to get independent objects.
        for prop_name, prop in cls.__dict__.items():
            if isinstance(prop, Prop):
                setattr(cls, prop_name, copy(prop))

        cls.__class_props = None
        cls._get_class_props()

    @property
    def _super_types(cls) -> Iterable[type | GenericAlias]:
        return (
            cls.__dict__["__orig_bases__"]
            if "__orig_bases__" in cls.__dict__
            else cls.__bases__
        )

    def _get_class_props(cls) -> dict[str, Prop]:
        if cls.__class_props is not None:
            return cls.__class_props

        # Construct list of Record superclasses and apply template superclasses.
        cls._model_superclasses = []

        props = {}
        typevar_map = get_typevar_map(cls)

        for name, anno in get_annotations(cls).items():
            if name not in cls.__dict__:
                prop_type = _get_data_type(anno, bound=Prop)
                prop = prop_type() if prop_type is not NoneType else Attr(alias=name)

                props[name] = prop
                setattr(cls, name, prop)

        for c in cls._super_types:
            # Get proper origin class of generic supertype.
            orig = get_origin(c) if not isinstance(c, type) else c

            if orig is None:
                continue

            # Handle typevar substitutions.
            typevar_map = get_typevar_map(c, subs=typevar_map)

            # Skip root Record class and non-Record classes.
            if not isinstance(orig, ModelMeta) or orig._root_class:
                continue

            # Apply template classes.
            if orig._template or cls._derivate:
                # Pre-collect all template props
                # to prepend them in order.
                orig_props = {}
                for prop_name, super_prop in orig._get_class_props().items():
                    if prop_name not in props:
                        prop = copy_and_override(type(super_prop), super_prop)
                        setattr(cls, prop_name, prop)
                        orig_props[prop_name] = prop

                props = {**orig_props, **props}  # Prepend template props.
            else:
                assert orig is c  # Must be concrete class, not a generic
                cls._model_superclasses.append(cast(type[Model], orig))

        cls.__class_props = props
        return props

    @property
    def _props(cls) -> dict[str, Prop]:
        """The statically defined properties of this record type."""
        return reduce(
            lambda x, y: {**x, **y},
            (c._props for c in cls._model_superclasses),
            cls._get_class_props(),
        )

    @property
    def __dataclass_fields__(cls) -> dict[str, Field]:  # noqa: D105
        return {
            p._name(): Field(
                p.default if p.default is not Not.defined else MISSING,
                (
                    p.default_factory if p.default_factory is not None else MISSING
                ),  # pyright: ignore[reportArgumentType]
                p.init,
                metadata={},
                repr=p.repr,
                hash=p.hash,
                compare=p.compare,
                kw_only=True,
            )
            for p in cls._props.values()
        }


ModT = TypeVar("ModT", bound="Model", covariant=True)


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Prop,),
    eq_default=False,
)
class Model(DataclassInstance, Generic[FullIdxT], metaclass=ModelMeta):
    """Schema for a record in a database."""

    _template: ClassVar[bool]
    _derivate: ClassVar[bool] = False
    _root_class: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)
        cls._root_class = False
        if "_template" not in cls.__dict__:
            cls._template = False

    @classmethod  # pyright: ignore[reportArgumentType]
    def _from_existing(
        cls: Callable[Params, ModT],
        rec: ModT,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ModT:
        return copy_and_override(
            cls,
            rec,
            *(arg if arg is not Not.defined else MISSING for arg in args),
            **{k: v if v is not Not.defined else MISSING for k, v in kwargs.items()},
        )  # pyright: ignore[reportCallIssue]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record instance."""
        super().__init__()

        cls = type(self)
        props = cls._props
        prop_types = {
            _get_data_type(
                p.typeref.hint if p.typeref is not None else type(p), bound=Prop
            )
            for p in props.values()
        }

        init_sequence = list(p for p in prop_types if p is not NoneType)
        init_sequence = sorted(
            init_sequence,
            key=cmp_to_key(
                lambda x, y: (
                    -1
                    if any(issubclass(x, p) for p in y.init_after)
                    else 1 if any(issubclass(y, p) for p in x.init_after) else 0
                )
            ),
        )

        for prop_type in init_sequence:
            props = {k: v for k, v in props.items() if isinstance(v, prop_type)}
            for prop_name in props:
                if prop_name in kwargs:
                    setattr(self, prop_name, kwargs[prop_name])

        self.__post_init__()

    def __post_init__(self) -> None:  # noqa: D105
        pass

    @overload
    def _to_dict(
        self,
        keys: Literal["names", "fqns"] = ...,
        include: set[type[Prop]] | None = ...,
    ) -> dict[str, Any]: ...

    @overload
    def _to_dict(
        self,
        keys: Literal["instances"],
        include: set[type[Prop]] | None = ...,
    ) -> dict[Prop, Any]: ...

    def _to_dict(
        self,
        keys: Literal["instances", "names", "fqns"] = "names",
        include: set[type[Prop]] | None = None,
    ) -> dict[Prop, Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Prop], ...] = (
            tuple(include) if include is not None else (Prop,)
        )

        vals = {
            (
                p if keys == "instances" else p.name if keys == "names" else p.fqn
            ): getattr(self, p.name)
            for p in type(self)._props.values()
            if isinstance(p, include_types) and p.name is not None
        }

        return cast(dict, vals)

    @classmethod
    def _is_complete_dict(
        cls,
        data: Mapping[Prop, Any] | Mapping[str, Any],
    ) -> bool:
        """Check if dict data contains all required info for record type."""
        in_data = {(p if isinstance(p, str) else p.name): v for p, v in data.items()}
        return all(
            (
                a.name in in_data
                or a.default is not Not.defined
                or a.default_factory is not None
            )
            for a in cls._props.values()
            if a.init is not False and a.name is not None
        )

    def __repr__(self) -> str:
        """Return a string representation of the record."""
        return f"{type(self).__name__}({repr(self._to_dict())})"


AttrT = TypeVar("AttrT")


@dataclass(kw_only=True, eq=False)
class Attr(Prop[AttrT, CtxT, CrudT], Generic[AttrT, CrudT, CtxT]):
    """Single-value attribute or column."""

    default: AttrT | type[Undef] = Undef
    default_factory: Callable[[], AttrT] | None = None

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> AttrT: ...

    @overload
    def __get__(
        self, instance: None, owner: type[ModT]
    ) -> Attr[AttrT, CrudT, ModT]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride] (pyright bug?)
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this attribute."""
        super_get = Prop.__get__(self, instance, owner)
        if super_get is not Ungot():
            return super_get

        assert instance is not None

        if self.name not in instance.__dict__:
            if self.default is not Undef:
                instance.__dict__[self.name] = self.default
            else:
                assert self.default_factory is not None
                instance.__dict__[self.name] = self.default_factory()

        return instance.__dict__[self.name]

    def __set__(self: Attr[Any, U, Any], instance: Model, value: AttrT) -> None:
        """Set the value of this attribute."""
        super_set = Prop.__set__(self, instance, value)
        if super_set is not Unset():
            return

        instance.__dict__[self.name] = value

    @property
    def _default(self) -> AttrT | type[Undef]:
        """Default value of the property."""
        return self._default

    @property
    def _default_factory(self) -> Callable[[], AttrT] | None:
        """Default value factory of the property."""
        return self.default_factory


def get_pl_schema(
    col_map: Mapping[str, Column],
) -> dict[str, pl.DataType | type | None]:
    """Return the schema of the dataset."""
    exact_matches = {
        name: (pl_type_map.get(col.value_typeform), col)
        for name, col in col_map.items()
    }
    matches = {
        name: (
            (match, col.value_typeform)
            if match is not None
            else (pl_type_map.get(col.common_value_type), col.value_typeform)
        )
        for name, (match, col) in exact_matches.items()
    }

    return {
        name: (
            match if isinstance(match, pl.DataType | type | None) else match(match_type)
        )
        for name, (match, match_type) in matches.items()
    }


def records_to_df(
    records: Iterable[dict[str, Any] | Record],
) -> pl.DataFrame:
    """Convert values to a DataFrame."""
    model_types: set[type[Record]] = {
        type(val) for val in records if isinstance(val, Record)
    }
    df_data = [
        (
            val._to_dict(keys="names" if len(model_types) == 1 else "fqns")
            if isinstance(val, Record)
            else val
        )
        for val in records
    ]

    return pl.DataFrame(
        df_data,
        schema=get_pl_schema(
            {
                col.name if len(model_types) == 1 else col.fqn: col
                for mod in model_types
                for col in mod._cols()
            }
        ),
    )


@dataclass
class Table(Generic[RecT, BackT, RwxT]):
    """Table matching a table model, may be filtered."""

    base: Base[Any, BackT]
    rec_types: set[type[RecT]]
    content: (
        str | sqla.FromClause | pd.DataFrame | pl.DataFrame | Iterable[Record] | None
    ) = None

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this table based on its record types."""
        return "|".join(rec._fqn for rec in self.rec_types)

    @cached_prop
    def sql(self) -> sqla.FromClause:
        """This table's from clause."""
        match self.content:
            case str():
                return sqla.table(self.content)
            case sqla.FromClause():
                return self.content
            case pd.DataFrame() | pl.DataFrame():
                return self._upload_df(self.content)
            case Iterable():
                return self._upload_df(records_to_df(self.content))
            case None:
                return self._get_base_union()

    def df(self: Table[Any, DynBackendID, Any]) -> pl.DataFrame:
        """Return the table as a DataFrame."""
        return pl.read_database(self.sql, self.base.connection)

    def mutate(
        self: Table[Any, Any, CRUD],
        input_table: Table[RecT2, BackT, Any],
        mode: MutationMode = "update",
    ) -> None:
        """Mutate given table."""
        tables = {rec: table for rec, (table, _) in self._base_table_map.items()}
        input_sql = input_table.sql

        statements: list[sqla.Executable] = []

        if mode in ("replace", "delete"):
            # Delete current records first.
            statements += [
                safe_delete(table, input_sql, self.base.engine)
                for table in tables.values()
            ]

        if mode != "delete":
            for base_type, table in tables.items():
                # Rename input columns to match base table columns.
                input_sql = input_sql.select().with_only_columns(
                    *(
                        input_sql.c[col.name].label(name)
                        for name, col in self._base_col_map[base_type].items()
                    )
                )

                if mode == "update":
                    statements.append(safe_update(table, input_sql, self.base.engine))
                else:
                    statements.append(
                        safe_insert(
                            table,
                            input_sql,
                            self.base.engine,
                            upsert=(mode == "upsert"),
                        )
                    )

        # Execute delete / insert / update statements.
        con = self.base.connection
        for statement in statements:
            con.execute(statement)

        if self.base.backend_type == "excel-file":
            self.base._save_to_excel(
                # {self.record_type, *self.record_type._record_superclasses}
            )

    def validate(
        self, inspector: sqla.Inspector | None = None, required: bool = False
    ) -> None:
        """Perform pre-defined schema validations."""
        inspector = inspector or sqla.inspect(self.base.engine)

        for table, _ in self._base_table_map.values():
            validate_sql_table(inspector, table, required)

    def __clause_element__(self) -> sqla.FromClause:
        """Return the table clause element."""
        return self.sql

    @cached_prop
    def _base_table_map(
        self,
    ) -> dict[type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]]:
        """Return the base SQLAlchemy table objects for all record types."""
        base_table_map: dict[
            type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]
        ] = {}
        for rec in self.rec_types:
            super_recs = (r for r in rec._model_superclasses if issubclass(r, Record))
            for base_rec in (rec, *super_recs):
                if base_rec not in base_table_map:
                    table = self.base._get_base_table(rec)
                    join_on = (
                        None
                        if base_rec is rec
                        else reduce(
                            sqla.and_,
                            (
                                pk_sel == pk_super
                                for pk_sel, pk_super in zip(
                                    self.base._map_pk(rec).columns,
                                    self.base._map_pk(base_rec).columns,
                                )
                            ),
                        )
                    )
                    base_table_map[base_rec] = (table, join_on)

        return dict(
            sorted(
                base_table_map.items(),
                key=cmp_to_key(
                    lambda x, y: (
                        -1
                        if issubclass(y[0], x[0])
                        else 1 if issubclass(x[0], y[0]) else 0
                    )
                ),
            )
        )

    @cached_prop
    def _base_col_map(self) -> dict[type[Record], dict[str, sqla.Column]]:
        """Return the column set for all record types."""
        name_attr = attrgetter("name" if len(self.rec_types) == 1 else "fqn")
        return {
            rec: {
                name_attr(col): sql_col
                for col, sql_col in self.base._map_cols(rec).items()
            }
            for rec in self._base_table_map.keys()
        }

    def _get_base_union(self) -> sqla.FromClause:
        """Recursively join all bases of this record to get the full data."""
        sel_tables = {
            rec: table
            for rec, (table, join_on) in self._base_table_map.items()
            if join_on is None
        }

        def _union(
            union_stage: tuple[type[Record] | None, sqla.Select],
            next_union: tuple[type[Record], sqla.Join | sqla.Table],
        ) -> tuple[type[Record] | None, sqla.Select]:
            left_rec, left_table = union_stage
            right_rec, right_table = next_union

            right_table = right_table.select().with_only_columns(
                *(
                    col.label(name)
                    for name, col in self._base_col_map[right_rec].items()
                )
            )

            if left_rec is not None and len(
                set(left_rec._pk.columns) & set(right_rec._pk.columns)  # type: ignore
            ):
                return left_rec, coalescent_join_sql(
                    left_table.alias(),
                    right_table.alias(),
                    reduce(
                        sqla.and_,
                        (
                            left_pk == right_pk
                            for left_pk, right_pk in zip(
                                self.base._map_pk(left_rec).columns,
                                self.base._map_pk(right_rec).columns,
                            )
                        ),
                    ),
                )
            else:
                return None, left_table.union(right_table).select()

        if len(sel_tables) > 1:
            rec_query_items = list(sel_tables.items())
            first_rec, first_select = rec_query_items[0]
            first: tuple[type[Record] | None, sqla.Select] = (
                first_rec,
                first_select.select().with_only_columns(
                    *(
                        col.label(name)
                        for name, col in self._base_col_map[first_rec].items()
                    )
                ),
            )

            _, base_union = reduce(_union, rec_query_items[1:], first)
        else:
            base_union = list(sel_tables.values())[0]

        base_join = base_union
        for base_table, join_on in self._base_table_map.values():
            if join_on is not None:
                base_join = base_join.join(base_table, join_on)

        res = base_join.subquery() if isinstance(base_join, sqla.Select) else base_join
        setattr(res, "_tab", self)
        return res

    def _get_upload_table(self) -> sqla.Table:
        cols = {
            name: col.copy()
            for cols in self._base_col_map.values()
            for name, col in cols.items()
        }
        for name, col in cols.items():
            col.name = name

        return sqla.Table(
            f"upload/{self.fqn.replace('.', '_')}/{token_hex(5)}",
            self.base._metadata,
            *cols.values(),
        )

    def _upload_df(self, df: pd.DataFrame | pl.DataFrame) -> sqla.Table:
        table = self._get_upload_table()

        if isinstance(df, pl.DataFrame):
            df.write_database(
                str(table),
                self.base.connection,
                if_table_exists="replace",
            )
        else:
            df.reset_index().to_sql(
                table.name,
                self.base.connection,
                if_exists="replace",
                index=False,
            )

        return table


ColT = TypeVar("ColT")

KeyT = TypeVar("KeyT", bound=Hashable, default=Any)


@dataclass(kw_only=True)
class Column(
    Attr[ColT, RwxT, OwnT],
    sqla.SQLColumnExpression[ColT],
    Generic[ColT, RwxT, OwnT, BackT],
):
    """Column property for a table model."""

    _table: Table[OwnT] | None = None

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


TupT = TypeVar("TupT")


@dataclass
class ColTuple(Prop[TupT, OwnT, R]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...]

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> TupT: ...

    @overload
    def __get__(self, instance: None, owner: type[RecT]) -> ColTuple[TupT, RecT]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this attribute."""
        val = tuple(col.__get__(instance, owner) for col in self.columns)
        return val[0] if len(val) == 1 else val

    def __set__(self, instance: Any, value: Any) -> None:
        """Set the value of this attribute."""
        raise AttributeError("Cannot set an index.")

    def gen_value_map(self, val: Any) -> dict[str, Hashable]:
        """Generate a key-value map for this key."""
        idx_names = [c.name for c in self.columns]

        if len(idx_names) == 1:
            return {idx_names[0]: val}

        assert isinstance(val, tuple) and len(val) == len(idx_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(idx_names, val)}

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self.context, self.columns))


@dataclass(eq=False)
class Key(ColTuple[KeyT, OwnT]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...] = ()

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if len(self.columns) == 0:
            self.columns = (
                Column(
                    _name=self.name,
                    _type=Column[self.common_value_type],
                    _owner=self.context,
                ),
            )
            setattr(owner, f"{name}_col", self.columns[0])


TgtT = TypeVar("TgtT", bound="Record", covariant=True)


@dataclass(eq=False)
class ForeignKey(ColTuple[KeyT, OwnT], Generic[TgtT, KeyT, OwnT]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...] = field(init=False)

    column_map: dict[Column[Hashable, Any, OwnT], Column[Hashable, Any, TgtT]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:  # noqa: D105
        self.columns = (
            tuple(self.column_map.keys()) if self.column_map is not None else ()
        )

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if self.column_map is None:
            pk_cols = self.target._pk.columns

            col_map = {
                Column(
                    _name=f"{self.name}_{pk_col.name}_fk",
                    _type=Column[pk_col.common_value_type],
                    _owner=self.context,
                ): pk_col
                for pk_col in pk_cols
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


class RecordTable:
    """Table, which a record belongs to."""

    @overload
    def __get__(
        self, instance: None, owner: type[RecT2]
    ) -> Table[RecT2, Symbolic, R]: ...

    @overload
    def __get__(
        self, instance: RecT2, owner: type[RecT2]
    ) -> Table[RecT2, DynBackendID, Any]: ...

    def __get__(  # noqa: D105
        self, instance: Record | None, owner: type[Record]
    ) -> Table[Record, Any, Any]:
        return (
            symbol_base.table(owner)
            if instance is None
            else instance._base.table(owner)
        )


class Record(Model, Generic[KeyT]):
    """Schema for a table in a database."""

    _root_class = True

    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID4: sqla.types.CHAR(36),
    }

    _table = RecordTable()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)

        for superclass in cls._model_superclasses:
            if superclass in cls.__bases__ and issubclass(superclass, Record):
                super_pk = superclass._pk  # type: ignore
                col_map = {
                    Column(
                        _name=pk_col.name,
                        _type=Column[pk_col.common_target_type],
                        _owner=cls,
                    ): pk_col
                    for pk_col in super_pk.columns
                }
                for col in col_map.keys():
                    setattr(cls, col.name, col)

                pk = Key[super_pk.typeargs[KeyT], cls](columns=tuple(col_map.keys()))
                setattr(cls, pk.name, pk)

                fk = ForeignKey[superclass, super_pk.typeargs[KeyT], cls](
                    column_map=col_map
                )
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
    def _cols(cls) -> set[Column]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Column)
        }

    @classmethod
    def _keys(cls) -> set[Key]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Key)
        }

    @classmethod
    def _fks(cls) -> set[ForeignKey]:
        """Columns of this record type's table."""
        return {
            prop
            for prop in cls._get_class_props().values()
            if isinstance(prop, ForeignKey)
        }

    @classmethod
    def _pl_schema(cls) -> dict[str, pl.DataType | type | None]:
        """Return the schema of the dataset."""
        return get_pl_schema(
            {name: col for name, col in cls._props.items() if isinstance(col, Column)}
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:
        """Return the table clause element."""
        return cls._table._base_table_map[cls][0]

    _published: bool = False
    _base: Attr[Base[Any, DynBackendID]] = Attr(default_factory=Base)

    _pk: Key[KeyT, Self]

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
