"""Connect to an SQL server, validate its schema and perform cached analysis queries."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, partial
from typing import (
    Any,
    Generic,
    Literal,
    LiteralString,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
    overload,
)

import pandas as pd
import sqlalchemy as sqla
import sqlalchemy.orm as orm
from pandas.api.types import (
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from typing_extensions import Self

from py_research.hashing import gen_str_hash
from py_research.reflect.runtime import get_all_subclasses

N = TypeVar("N", bound=LiteralString)

V = TypeVar("V")

Params = ParamSpec("Params")

S = TypeVar("S", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")
S_contrav = TypeVar("S_contrav", contravariant=True, bound="Schema")
S2 = TypeVar("S2", bound="Schema")


T = TypeVar("T", bound="Table")


def _map_df_dtype(c: pd.Series) -> sqla.types.TypeEngine:
    if is_datetime64_dtype(c):
        return sqla.types.DATETIME()
    elif is_integer_dtype(c):
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
    if len(df.index.names) > 1:
        raise NotImplementedError("Multi-index not supported yet.")

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


def _map_foreignkey_schema(
    _: sqla.Table,
    to_schema: str | None,
    constraint: sqla.ForeignKeyConstraint,
    referred_schema: str | None,
    schema_dict: "dict[str | None, type[Schema]]",
) -> str | None:
    assert to_schema in schema_dict

    for schema_name, schema in schema_dict.items():
        if schema is not None:
            for schema_class in get_all_subclasses(schema):
                if (
                    issubclass(schema_class, Table)
                    and schema_class._sqla_table is constraint.referred_table
                ):
                    return schema_name

    return referred_schema


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


class ColRef(sqla.ColumnClause[V], Generic[V, T]):
    """Reference a column by table schema, name and value type."""

    def __init__(  # noqa: D107
        self, table: type[T], name: str, value_type: type[V] | None
    ) -> None:
        self.table_schema = table
        self.name = name
        self.value_type = value_type

        super().__init__(
            self.name,
            type_=self.table_schema._sqla_table.c[self.name].type,
        )


@dataclass
class Col(Generic[V]):
    """Define table column within a table schema."""

    name: str | None = None

    def __setname__(self, _, name: str) -> None:  # noqa: D105
        self.name = name

    def __get__(  # noqa: D105 # type: ignore
        self, instance: None, table: type[T]
    ) -> ColRef[V, T]:
        if instance is not None:
            raise NotImplementedError("Instance-based column access not supported.")

        assert self.name is not None
        value_type: type[V] = get_args(table._type_hints[self.name])[0]

        return ColRef(table, self.name, value_type)


class Schema:
    """Base class for declarative SQL schemas."""

    _type_map: dict[type, sqla.types.TypeEngine] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
    }

    @classmethod
    def _tables(cls) -> set[type["Table"]]:
        """Return all tables defined by subclasses."""
        subclasses = get_all_subclasses(cls)
        return {s for s in subclasses if isinstance(s, Table)}


class Table(Schema):
    """Protocol for table schemas."""

    _table: str
    _type_hints: dict[str, type]
    _sqla_table: sqla.Table

    def __clause_element__(self) -> sqla.TableClause:  # noqa: D105
        assert self._table is not None
        return sqla.table(self._table)

    def __init_subclass__(cls) -> None:  # noqa: D105
        cls._type_hints = get_type_hints(cls)

        registry = orm.registry(type_annotation_map=cls._type_map)

        cls._sqla_table = sqla.Table(
            cls._table,
            registry.metadata,
            *(
                sqla.Column(name, registry._resolve_type(cls._type_hints[name]))
                for name, col in cls.__dict__.items()
                if isinstance(col, Col)
            ),
        )

        return super().__init_subclass__()


@dataclass(frozen=True)
class Backend(Generic[N]):
    """SQL backend for DB."""

    name: N
    """Unique name to identify this backend by."""

    url: str | sqla.URL
    """Connection URL."""


SchemaValidation: TypeAlias = Literal["compatible-if-present", "all-present"]


@dataclass(frozen=True)
class DBCol(Generic[N, V]):
    """SQL column or table."""

    db: "DB[N, Any, Any]"
    select: sqla.Select[tuple[V]]
    name: str

    def __clause_element__(self) -> sqla.Subquery:  # noqa: D105
        return self.select.subquery()

    def to_series(self) -> pd.Series:
        """Download selected column data as series.

        Returns:
            Series containing the data.
        """
        with self.db.engine.connect() as con:
            return pd.read_sql(self.select, con)[self.select.columns[0].name]

    def __getitem__(self, key: slice) -> Self:  # noqa: D105
        assert key == slice(None)
        return self

    def __setitem__(self, key: slice, value: pd.Series) -> None:  # noqa: D105
        assert key == slice(None)
        raise NotImplementedError("Setting SQL data not supported yet.")


@dataclass(frozen=True)
class DBTable(DBCol[N, T], Generic[N, T]):
    """SQL table."""

    schema: type[T]

    @property
    def columns(self) -> dict[str, sqla.ColumnElement]:  # noqa: D102
        return dict(self.select.columns)

    @overload
    def __getitem__(self, ref: slice) -> Self: ...

    @overload
    def __getitem__(self, ref: str | ColRef[V, T]) -> DBCol[N, V]: ...

    @overload
    def __getitem__(self, ref: sqla.ColumnElement[bool]) -> "DBTable[N, T]": ...

    def __getitem__(  # noqa: D105
        self, ref: slice | str | ColRef[V, T] | sqla.ColumnElement[bool]
    ) -> "Self | DBCol[N, V] | DBTable[N, T]":
        match ref:
            case slice():
                return super().__getitem__(ref)
            case str() | ColRef():
                name = ref if isinstance(ref, str) else ref.name
                return cast(
                    DBCol[N, V],
                    DBCol(
                        db=self.db, select=sqla.select(self.columns[name]), name=name
                    ),
                )
            case sqla.ColumnElement():
                return DBTable(
                    db=self.db,
                    select=self.select.where(ref),
                    name=self.name,
                    schema=self.schema,
                )

    def to_df(self) -> pd.DataFrame:
        """Download table selection to dataframe.

        Returns:
            Dataframe containing the data.
        """
        with self.db.engine.connect() as con:
            return pd.read_sql(self.select, con)


@dataclass
class DB(Generic[N, S_cov, S_contrav]):
    """Active connection to a SQL server."""

    backend: Backend[N]
    schema: type[S_cov | S_contrav] | Mapping[type[S_cov | S_contrav], str] | None = (
        None
    )
    overlay: bool | str = False
    validations: Mapping[type[S_cov], SchemaValidation] = {}

    @cached_property
    def schema_dict(self) -> dict[str | None, type[S_cov | S_contrav]]:
        """Return dict with all schemas."""
        return (
            {name: s for s, name in self.schema.items()}
            if isinstance(self.schema, Mapping)
            else {None: self.schema} if self.schema is not None else {}
        )

    @cached_property
    def metadata(self) -> sqla.MetaData:
        """Return metadata object containing all this DB's tables."""
        metadata = sqla.MetaData()
        schema_dict = self.schema_dict

        for schema_name, schema in schema_dict.items():
            for table in schema._tables():
                table._sqla_table.to_metadata(
                    metadata,
                    schema=schema_name or sqla.schema.SchemaConst.RETAIN_SCHEMA,
                    referred_schema_fn=partial(
                        _map_foreignkey_schema, schema_dict=schema_dict
                    ),
                )

        return metadata

    @cached_property
    def engine(self) -> sqla.engine.Engine:
        """Return engine for this DB."""
        return sqla.create_engine(self.backend.url)

    def validate(self) -> None:
        """Perform pre-defined schema validations."""
        tables = {
            t: v
            for s, v in self.validations.items()
            for t in self[s].metadata.tables.values()
        }

        inspector = sqla.inspect(self.engine)

        for table, validation in tables.items():
            has_table = inspector.has_table(table.name, table.schema)

            if not has_table and validation == "compatible-if-present":
                continue

            assert has_table

            db_columns = {
                c["name"]: c for c in inspector.get_columns(table.name, table.schema)
            }
            for column in table.columns:
                assert column.name in db_columns

                db_col = db_columns[column.name]
                assert isinstance(db_col["type"], type(column.type))
                assert db_col["nullable"] == column.nullable or column.nullable is None

            db_pk = inspector.get_pk_constraint(table.name, table.schema)
            if len(db_pk["constrained_columns"]) > 0:  # Allow source tbales without pk
                assert set(db_pk["constrained_columns"]) == set(
                    table.primary_key.columns.keys()
                )

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

    def __post_init__(self):  # noqa: D105
        self.validate()

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self: "DB[N, Any, T]", key: type[T]) -> "DBTable[N, T]": ...

    @overload
    def __getitem__(self: "DB[N, S, Any]", key: type[S]) -> "DB[N, S, S]": ...

    def __getitem__(  # noqa: D105
        self, key: "slice | type[T] | type[S]"
    ) -> "Self | DBTable[N, T] | DB[N, S, S]":
        if isinstance(key, slice):
            return self

        if issubclass(key, Table):
            return DBTable(
                db=self,
                select=sqla.select(self.metadata.tables[key._table]),
                name=key._table,
                schema=key,
            )

        if issubclass(key, Schema):
            return DB(
                self.backend, key, overlay=self.overlay, validations=self.validations
            )

    def to_table(
        self,
        df: pd.DataFrame,
        schema: type[T] | None = None,
        create_external_fk: bool = True,
    ) -> DBTable[N, T]:
        """Transfer dataframe or sql query results to manifested SQL table.

        Args:
            df: Source / definition of data.
            schema: Schema of the table to create.
            create_external_fk:
                Whether to create foreign keys to external tables SQL-side
                (may cause permission issues).

        Returns:
            Reference to manifested SQL table.
        """
        table_name = f"df_{gen_str_hash(df)}"

        sqla_table = (
            sqla.Table(table_name, self.metadata, *_cols_from_df(df).values())
            if schema is None
            else schema._sqla_table
        )
        if not create_external_fk:
            _remove_external_fk(sqla_table)

        if table_name not in self.metadata.tables and not sqla.inspect(
            self.engine
        ).has_table(table_name, schema=None):
            sqla_table.create(self.engine)
            df.reset_index()[list(sqla_table.c.keys())].to_sql(
                table_name, self.engine, if_exists="append", index=False
            )

        return DBTable(
            db=self,
            select=sqla.select(sqla_table),
            name=table_name,
            schema=schema or Table,
        )
