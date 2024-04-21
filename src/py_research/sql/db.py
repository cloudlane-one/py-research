"""Abstract Python interface for SQL databases."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, partial
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    LiteralString,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import yarl
from cloudpathlib import CloudPath
from typing_extensions import Self
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.files import HttpFile
from py_research.sql.schema import ColRef, Schema, T, Table, V

from .utils import map_foreignkey_schema, remove_external_fk

N = TypeVar("N", bound=LiteralString)

Params = ParamSpec("Params")

S = TypeVar("S", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")

S_contrav = TypeVar("S_contrav", contravariant=True, bound="Schema")
S2 = TypeVar("S2", bound="Schema")


@dataclass(frozen=True)
class Backend(Generic[N]):
    """SQL backend for DB."""

    name: N
    """Unique name to identify this backend by."""

    url: sqla.URL | CloudPath | HttpFile | Path
    """Connection URL or path."""

    @cached_property
    def type(self) -> Literal["sql-connection", "sqlite-file", "excel-file"]:
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
    validations: Mapping[type[S_cov], SchemaValidation] = {}
    substitutions: Mapping[type[S_cov], str | sqla.TableClause] = {}

    create_cross_fk: bool = True

    @cached_property
    def schema_dict(self) -> dict[str | None, type[S_cov | S_contrav]]:
        """Return dict with all schemas."""
        return (
            {name: s for s, name in self.schema.items()}
            if isinstance(self.schema, Mapping)
            else {None: self.schema} if self.schema is not None else {}
        )

    @cached_property
    def meta(self) -> sqla.MetaData:
        """Return metadata object."""
        return sqla.MetaData()

    @cached_property
    def subs(self) -> dict[type[Table], sqla.Table]:
        """Return substitutions object."""
        return {
            table: (
                sqla.Table(name=sub.name, metadata=self.meta, schema=sub.schema)
                if isinstance(sub, sqla.TableClause)
                else sqla.Table(name=sub, metadata=self.meta)
            )
            for table, sub in self.substitutions.items()
            if issubclass(table, Table)
        }

    @cached_property
    def table_map(self) -> dict[type[Table], sqla.Table]:
        """Return table_map object containing all this DB's tables."""
        meta = self.meta
        schema_dict = self.schema_dict

        table_map = {}

        for schema_name, schema in schema_dict.items():
            for table in schema._tables():
                sub = self.subs.get(table)
                table_map[table] = table._sqla_table(meta, self.subs).to_metadata(
                    meta,
                    schema=(
                        sub.schema
                        if sub is not None
                        else schema_name or sqla.schema.SchemaConst.RETAIN_SCHEMA
                    ),  # type: ignore
                    referred_schema_fn=partial(
                        map_foreignkey_schema, schema_dict=schema_dict
                    ),
                    name=sub.name if sub is not None else None,
                )

        return table_map

    @cached_property
    def engine(self) -> sqla.engine.Engine:
        """Return engine for this DB."""
        return (
            sqla.create_engine(
                self.backend.url
                if isinstance(self.backend.url, sqla.URL)
                else str(self.backend.url)
            )
            if self.backend.type == "sql-connection"
            or self.backend.type == "sqlite-file"
            else sqla.create_engine("duckdb:///:memory:")
        )

    def validate(self) -> None:
        """Perform pre-defined schema validations."""
        tables = {
            t: v
            for s, v in self.validations.items()
            for t in self[s].table_map.values()
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
            self._pre_gettable(key)
            return DBTable(
                db=self,
                select=sqla.select(self.table_map[key]),
                name=key._default_name,
                schema=key,
            )

        if issubclass(key, Schema):
            return DB(
                self.backend,
                key,
                validations=self.validations,
                substitutions=self.substitutions,
            )

    def __setitem__(
        self,
        key: type[T],
        value: pd.DataFrame | sqla.Select,
    ) -> DBTable[N, T]:
        """Transfer dataframe or sql query results to SQL table.

        Args:
            key: Schema of the table to upload to.
            value: Source / definition of data.
            create_external_fk:
                Whether to create foreign keys to external tables SQL-side
                (may cause permission issues).

        Returns:
            Reference to manifested SQL table.
        """
        sqla_table = self._ensure_table_exists(key)

        if isinstance(value, pd.DataFrame):
            value.reset_index()[list(sqla_table.c.keys())].to_sql(
                key._default_name, self.engine, if_exists="replace", index=False
            )
        else:
            with self.engine.begin() as con:
                con.execute(
                    sqla.insert(sqla_table).from_select(
                        value.selected_columns.keys(), value
                    )
                )

        self._post_settable(key, value)

        return DBTable(
            db=self,
            select=sqla.select(sqla_table),
            name=key._default_name,
            schema=key or Table,
        )

    def _pre_gettable(self, table: type[Table]) -> None:
        """Perform pre-getitem hooks."""
        self._load_from_excel([table])

    def _post_settable(
        self, table: type[Table], value: pd.DataFrame | sqla.Select
    ) -> None:
        """Perform pre-setitem hooks."""
        self._save_to_excel([table])

    def _create_sqla_table(self, sqla_table: sqla.Table) -> None:
        """Construct SQLA table from Table class."""
        if not self.create_cross_fk:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            remove_external_fk(sqla_table)

        sqla_table.create(self.engine)

    def _ensure_table_exists(self, table: type[Table]) -> sqla.Table:
        """Ensure that the table exists in the database."""
        sqla_table = self.table_map.get(table)
        if sqla_table is None:
            if table in self.substitutions:
                sqla_table = self.subs[table]
            else:
                sqla_table = table._sqla_table(self.meta, subs=self.subs)

        if not sqla.inspect(self.engine).has_table(
            sqla_table.name, schema=sqla_table.schema
        ):
            self._create_sqla_table(sqla_table)

        return sqla_table

    def _load_from_excel(self, tables: list[type[Table]] | None = None) -> None:
        """Load all tables from Excel."""
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        path = (
            self.backend.url.get()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url
        )

        with open(path, "rb") as file:
            for table in tables or self.table_map.keys():
                sqla_table = self._ensure_table_exists(table)
                pl.read_excel(file, sheet_name=table._default_name).write_database(
                    str(sqla_table), str(self.engine.url)
                )

    def _save_to_excel(self, tables: list[type[Table]] | None) -> None:
        """Save all tables to Excel."""
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        file = (
            BytesIO()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url.open("wb")
        )

        with ExcelWorkbook(file) as wb:
            for table in tables or self.table_map.keys():
                pl.read_database(f"SELECT * FROM {table}", self.engine).write_excel(
                    wb, worksheet=table._default_name
                )

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)
