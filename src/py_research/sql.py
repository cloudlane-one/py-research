"""Connect to an SQL server and perform cached analysis queries."""

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import pandas as pd
import sqlalchemy as sqla
import sqlalchemy.orm as orm
from pandas.api.types import (
    is_datetime64_dtype,
    is_int64_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.util import hash_pandas_object
from typing_extensions import Self


def _hash_df(df: pd.DataFrame | pd.Series) -> str:
    return hex(abs(sum(hash_pandas_object(df))))[2:12]


Params = ParamSpec("Params")


def _get_sql_ctx() -> "DB":
    sql_con = active_con.get()

    if sql_con is None:
        raise (RuntimeError("`SQLContext` object not supplied via context."))

    return sql_con


SchemaBase = orm.DeclarativeBase


default_type_map = {
    str: sqla.types.String().with_variant(
        sqla.types.String(50), "oracle"
    ),  # Avoid oracle error when VARCHAR has no size parameter
}


V = TypeVar("V")
S = TypeVar("S", bound=SchemaBase)
S_cov = TypeVar("S_cov", bound=SchemaBase, covariant=True)
S_contra = TypeVar("S_contra", bound=SchemaBase, contravariant=True)
V_contra = TypeVar("V_contra", contravariant=True)


class ColRef(
    orm.InstrumentedAttribute[V], Generic[V, S_cov]
):  # pylint: disable=W0223:abstract-method
    """Reference a column by scheme type, name and value type."""


class Col(orm.MappedColumn[V]):  # pylint: disable=W0223:abstract-method
    """Define table column within a table schema."""

    if TYPE_CHECKING:

        @overload
        def __get__(self, instance: None, owner: type[S]) -> ColRef[V, S]:
            ...

        @overload
        def __get__(self, instance: object, owner: type) -> V:
            ...

        def __get__(  # noqa: D105
            self, instance: object | None, owner: type[S]
        ) -> ColRef[V, S] | V:
            ...

    @classmethod
    def cast_sqla_attr(cls, attr: orm.MappedColumn) -> Self:
        """Cast a sqlalchemy attr object to this type."""
        return cast(Self, attr)


def _wrap_mapped_col(
    func: Callable[Params, orm.MappedColumn[V]]
) -> Callable[Params, Col[V]]:
    @wraps(func)
    def inner(*args: Params.args, **kwargs: Params.kwargs) -> Col[V]:
        return Col.cast_sqla_attr(func(*args, **kwargs))

    return inner


col = _wrap_mapped_col(orm.mapped_column)
Relation = sqla.ForeignKey


class Data(Protocol[S_contra]):
    """SQL table-shaped data."""

    @property
    def columns(self) -> Mapping[str, sqla.ColumnElement]:
        """Return all columns within this data."""
        ...

    def __clause_element__(self) -> sqla.FromClause:  # noqa: D105
        ...

    def __getitem__(  # noqa: D105
        self, ref: str | ColRef[V, S_contra]
    ) -> sqla.ColumnElement[V]:
        ...

    @property
    def name(self) -> str:
        """Data object's name, if any."""
        ...


@dataclass
class Table(Data[S_contra]):
    """Reference to a manifested SQL table."""

    sqla_table: sqla.Table

    @property
    def columns(self) -> dict[str, sqla.Column]:  # noqa: D102
        return dict(self.__clause_element__().columns)

    def __getitem__(  # noqa: D105
        self, ref: str | ColRef[V, S_contra]
    ) -> sqla.Column[V]:
        return self.columns[ref if isinstance(ref, str) else ref.key]

    def __clause_element__(self) -> sqla.Table:  # noqa: D105
        return self.sqla_table

    @property
    def name(self) -> str:  # noqa: D102
        return self.sqla_table.name


def _map_df_dtype(c: pd.Series) -> sqla.types.TypeEngine:
    if is_datetime64_dtype(c):
        return sqla.types.DATETIME()
    elif is_int64_dtype(c):
        return sqla.types.INTEGER()
    elif is_numeric_dtype(c):
        return sqla.types.FLOAT()
    elif is_string_dtype(c):
        max_len = c.str.len().max()
        if max_len < 16:
            return sqla.types.CHAR(max_len)
        elif max_len < 256:
            return sqla.types.VARCHAR(max_len)

    return sqla.types.BLOB()


def _cols_from_df(df: pd.DataFrame) -> dict[str, sqla.Column]:
    return {
        **{
            level: sqla.Column(
                level,
                _map_df_dtype(df.index.get_level_values(level).to_series()),
                primary_key=True,
            )
            for level in df.index.names
        },
        **{
            str(name): sqla.Column(str(col.name), _map_df_dtype(col))
            for name, col in df.items()
        },
    }


@dataclass
class Query(Data[S_contra]):
    """SQL data defined by a query."""

    sel: sqla.Select
    schema: type[S_contra] | None = None
    name: str | None = None

    @property
    def columns(self) -> dict[str, sqla.ColumnElement]:  # noqa: D102
        return dict(self.__clause_element__().columns)

    def __clause_element__(self) -> sqla.Subquery:  # noqa: D105
        return self.sel.subquery()

    def __getitem__(  # noqa: D105
        self, ref: str | ColRef[V, S_contra]
    ) -> sqla.ColumnElement[V]:
        return self.columns[ref if isinstance(ref, str) else ref.key]


SelFunc: TypeAlias = Callable[Params, sqla.Select | sqla.Table]
BoundSelFunc: TypeAlias = Callable[[], sqla.Select | sqla.Table]


@dataclass
class DeferredQuery(Data[S_contra]):
    """SQL data defined by a query-returning PYthon function."""

    func: BoundSelFunc
    schema: type[S_contra] | None = None

    @property
    def columns(self) -> dict[str, sqla.ColumnElement]:  # noqa: D102
        return dict(self.__clause_element__().columns)

    def __clause_element__(self) -> sqla.Subquery | sqla.Table:  # noqa: D105
        res = self.func()
        return res.subquery() if isinstance(res, sqla.Select) else res

    def __getitem__(  # noqa: D105
        self, ref: str | ColRef[V, S_contra]
    ) -> sqla.ColumnElement[V]:
        return self.columns[ref if isinstance(ref, str) else ref.key]

    @property
    def name(self) -> str | None:  # noqa: D102
        return (
            self.func.func.__name__
            if isinstance(self.func, partial)
            else self.func.__name__
            if isinstance(self.func, Callable)
            else None
        )


DS = TypeVar("DS", bound="DBSchema")
DS_cov = TypeVar("DS_cov", bound="DBSchema", covariant=True)
DS_contra = TypeVar("DS_contra", bound="DBSchema", contravariant=True)


@dataclass
class Schema(Generic[S_cov]):
    """SQL schema defining multiple related tables."""

    schema_def: type[S_cov] | None = None
    default: bool = False

    def __copy_to_namespace(self):
        self.metadata = sqla.MetaData(schema=self.name)
        if self.schema_def is not None:
            for table in self.schema_def.metadata.tables.values():
                table.to_metadata(self.metadata, schema=None)  # type: ignore

    def __post_init__(self):  # noqa: D105
        self.name = None
        self.__copy_to_namespace()

    def __set_name__(self, _: type, name: str):  # noqa: D105
        self.name = name if name != "__default__" else None
        self.__copy_to_namespace()

    def __get__(  # noqa: D105
        self, instance: Any, owner: type[DS]
    ) -> "SchemaRef[S_cov, DS]":
        return SchemaRef(self, owner)

    @property
    def tables(self) -> dict[str, Table]:
        """Tables defined by this schema."""
        return {name: Table(table) for name, table in self.metadata.tables.items()}

    def __getitem__(self, table: type[S] | str) -> Table[S]:  # noqa: D105
        # TODO: Find a way to limit table to type[S_cov] without losing type info,
        # maybe via decorating the schema classes.
        return Table(
            self.metadata.tables[
                ".".join(
                    [
                        *([self.name] if self.name else []),
                        table if isinstance(table, str) else table.__tablename__,
                    ]
                )
            ]
        )


@dataclass
class SchemaRef(Generic[S_cov, DS_cov]):
    """Reference to a SQL schema including its parent DB schema."""

    schema: Schema[S_cov]
    db_schema: type[DS_cov]


class DBSchema(Generic[S_cov]):
    """Schema for an entire SQL database."""

    __default__: Schema[S_cov] = Schema()

    @classmethod
    def schema_dict(cls) -> dict[str | None, Schema | None]:
        """Return dict with all sub-schemas."""
        return {
            None: cls.__default__.schema,  # type: ignore
            **{
                (name if name != "__default__" else None): s
                for name, s in cls.__dict__.items()
                if isinstance(s, Schema) and name != "__default__"
            },
        }

    @classmethod
    def default(cls) -> Schema[S_cov]:
        """Return default schema."""
        return cls.__default__.schema  # type: ignore

    @classmethod
    def defaults(cls) -> dict[type[SchemaBase], Schema]:
        """Return mapping of schema types to default schema names."""
        defaults = {}
        for s in cls.schema_dict().values():
            if (
                s is not None
                and s.schema_def is not None
                and (s.schema_def not in defaults.keys() or s.default)
            ):
                defaults[s.schema_def] = s
        return defaults


S2 = TypeVar("S2", bound=SchemaBase)


@dataclass
class DB(Generic[S_cov, DS_contra]):
    """Active connection to a SQL server with schema."""

    url: str | sqla.URL
    schema: type[DBSchema[S_cov]] = DBSchema
    tag: str = datetime.today().strftime("YYYY-MM-dd")
    validate: bool = True

    def __post_init__(self):  # noqa: D105
        self.default_meta = self.schema.default().metadata
        self.__token = None

        self.engine = sqla.create_engine(self.url)

        if self.validate:
            # Validation.

            inspector = sqla.inspect(self.engine)

            for schema_name, schema in self.schema.schema_dict().items():
                if schema_name is not None:
                    assert inspector.has_schema(schema_name)

                meta = schema.metadata if schema is not None else sqla.MetaData()

                for table in meta.tables.values():
                    assert inspector.has_table(table.name, schema_name)

                    db_columns = {
                        c["name"]: c
                        for c in inspector.get_columns(table.name, schema_name)
                    }
                    for column in table.columns:
                        assert column.name in db_columns

                        db_col = db_columns[column.name]
                        assert isinstance(db_col["type"], type(column.type))
                        assert (
                            db_col["nullable"] == column.nullable
                            or column.nullable is None
                        )

                    db_pk = inspector.get_pk_constraint(table.name, schema_name)
                    if (
                        len(db_pk["constrained_columns"]) > 0
                    ):  # Allow source tbales without pk
                        assert set(db_pk["constrained_columns"]) == set(
                            table.primary_key.columns.keys()
                        )

                    db_fks = inspector.get_foreign_keys(table.name, schema_name)
                    for fk in table.foreign_key_constraints:
                        matches = [
                            (
                                set(db_fk["constrained_columns"])
                                == set(fk.column_keys),
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

    @overload
    def __getitem__(self, key: SchemaRef[S2, DS_contra]) -> Schema[S2]:
        ...

    @overload
    def __getitem__(self, key: None) -> Schema[S_cov]:
        ...

    def __getitem__(  # noqa: D105
        self, key: SchemaRef[S2, DS_contra] | None
    ) -> Schema[S2] | Schema[S_cov]:
        return self.schema.default() if key is None else key.schema  # type: ignore

    def activate(self) -> None:
        """Set this instance as the current SQL context."""
        self.__token = active_con.set(self)

    def __enter__(self):  # noqa: D105
        self.activate()
        return self

    def __exit__(self, *_):  # noqa: D105
        if self.__token is not None:
            active_con.reset(self.__token)

    def _table_from_df(
        self,
        df: pd.DataFrame,
        schema: type[S_cov] | None = None,
        name: str | None = None,
    ) -> Table[S_cov]:
        table_name = f"{name or 'df'}_{self.tag}_{_hash_df(df)}"

        if table_name not in self.default_meta.tables and not sqla.inspect(
            self.engine
        ).has_table(table_name, schema=None):
            cols = (
                {
                    c.key: sqla.Column(c.key, c.type, primary_key=c.primary_key)
                    for c in schema.__table__.columns
                }
                if schema is not None
                else _cols_from_df(df)
            )
            table = Table(sqla.Table(table_name, self.default_meta, *cols.values()))
            table.sqla_table.create(self.engine)
            df.reset_index()[list(cols.keys())].to_sql(
                table_name, self.engine, if_exists="append", index=False
            )

        return (
            Table(self.default_meta.tables[table_name])
            if table_name in self.default_meta.tables
            else Table(
                sqla.Table(table_name, self.default_meta, autoload_with=self.engine)
            )
        )

    def _table_from_query(
        self,
        q: Query[S] | DeferredQuery[S],
        name: str | None = None,
    ) -> Table[S_cov]:
        table_name = f"{name or q.name or 'cached'}_{self.tag}"

        if table_name in self.default_meta.tables:
            return Table(self.default_meta.tables[table_name])
        elif sqla.inspect(self.engine).has_table(table_name, schema=None):
            return Table(
                sqla.Table(
                    table_name,
                    self.default_meta,
                    autoload_with=self.engine,
                )
            )
        else:
            sel_res = q.__clause_element__()
            table = Table(
                sqla.Table(
                    table_name,
                    self.default_meta,
                    *(
                        sqla.Column(name, col.type, primary_key=col.primary_key)
                        for name, col in sel_res.columns.items()
                    ),
                )
            )

            with self.engine.begin() as con:
                con.execute(
                    sqla.insert(table).from_select(
                        sel_res.exported_columns.keys(), sel_res
                    )
                )

            return table

    def to_table(
        self,
        src: pd.DataFrame | Query[S] | DeferredQuery[S],
        schema: type[S_cov] | None = None,
        name: str | None = None,
    ) -> Table[S_cov]:
        """Transfer dataframe or sql query results to manifested table."""
        match src:
            case pd.DataFrame():
                return self._table_from_df(src, schema, name)
            case Query() | DeferredQuery():
                return self._table_from_query(src, name)

    def to_df(self, q: Data) -> pd.DataFrame:
        """Transfer manifested table or query results to local dataframe."""
        sel = sqla.select(q)
        with self.engine.connect() as con:
            return pd.read_sql(sel, con)


active_con: ContextVar[DB | None] = ContextVar("active_sql_con", default=None)


QueryFunc: TypeAlias = Callable[Params, Query[S_cov]]
DefQueryFunc: TypeAlias = Callable[Params, DeferredQuery[S_cov]]


@overload
def query(func: SelFunc[Params]) -> QueryFunc[Params, S]:
    ...


@overload
def query(
    *, defer: Literal[True] = ..., schema: type[S] = ...  # type: ignore
) -> Callable[[SelFunc[Params]], DefQueryFunc[Params, S]]:
    ...


@overload
def query(
    *, defer: Literal[True] = ..., schema: None = ...
) -> Callable[[SelFunc[Params]], DefQueryFunc[Params, Any]]:
    ...


@overload
def query(
    *, defer: Literal[False] = ..., schema: type[S] = ...  # type: ignore
) -> Callable[[SelFunc[Params]], QueryFunc[Params, S]]:
    ...


@overload
def query(
    *, defer: Literal[False] = ..., schema: None = ...
) -> Callable[[SelFunc[Params]], QueryFunc[Params, Any]]:
    ...


def query(
    func: SelFunc[Params] | None = None,
    *,
    defer: bool | None = False,
    schema: type[S] | None = None,
) -> (
    QueryFunc[Params, S]
    | Callable[[SelFunc[Params]], QueryFunc[Params, S] | DefQueryFunc[Params, S]]
):
    """Transform :py:obj:`sqla.Selectable`-returning function into query."""

    def inner(
        func: SelFunc[Params],
    ) -> QueryFunc[Params, S] | DefQueryFunc[Params, S]:
        @wraps(func)
        def inner_inner(
            *args: Params.args, **kwargs: Params.kwargs
        ) -> Query[S] | DeferredQuery[S]:
            res = func(*args, **kwargs)
            return (
                Query(res if isinstance(res, sqla.Select) else sqla.select(res), schema)
                if not defer
                else DeferredQuery(partial(func, *args, **kwargs), schema)
            )

        return inner_inner  # type: ignore

    return inner(func) if func is not None else inner  # type: ignore


def default_table(schema: type[S]) -> DeferredQuery[S]:
    """Return default table for given table schema."""

    def inner() -> sqla.Table:
        ctx = _get_sql_ctx()

        valid_keys = list(
            filter(lambda s: issubclass(schema, s), ctx.schema.defaults().keys())
        )
        if len(valid_keys) == 0:
            raise KeyError(schema)

        meta = ctx.schema.defaults()[valid_keys[0]].metadata
        return meta.tables[
            ".".join(
                [
                    *([meta.schema] if meta.schema else []),
                    schema.__tablename__,
                ]
            )
        ]

    return DeferredQuery(inner, schema)
