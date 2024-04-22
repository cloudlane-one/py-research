"""Abstract Python interface for SQL databases."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, partial, reduce
from io import BytesIO
from pathlib import Path
from secrets import token_hex
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
from py_research.hashing import gen_str_hash
from py_research.sql.schema import ColRef, DataFrame, Rel, Schema, T, Table, V

from .utils import cols_from_df, map_foreignkey_schema, remove_external_fk

N = TypeVar("N", bound=LiteralString)
N2 = TypeVar("N2", bound=LiteralString)

Params = ParamSpec("Params")

S = TypeVar("S", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")

S_contrav = TypeVar("S_contrav", contravariant=True, bound="Schema")
S2 = TypeVar("S2", bound="Schema")

D = TypeVar("D", bound="DataFrame")


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
    sources: set["DBTable[N, Any]"] = set()

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
        self,
        ref: slice | str | ColRef[V, Table] | sqla.ColumnElement[bool],
    ) -> "Self | DBCol[N, V] | DBTable[N, Any]":
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

    def to_df(self, kind: type[D] = pd.DataFrame) -> D:
        """Download table selection to dataframe.

        Returns:
            Dataframe containing the data.
        """
        if kind is pl.DataFrame:
            return cast(D, pl.read_database(self.select, self.db.engine))
        else:
            with self.db.engine.connect() as con:
                return cast(D, pd.read_sql(self.select, con))

    def extend(
        self, data: "DataFrame | sqla.Select | DBTable[N, T]"
    ) -> "DBTable[N, T]":
        """Extend the table with new data and return new table."""
        raise NotImplementedError()


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
        """Dict with all schemas."""
        return (
            {name: s for s, name in self.schema.items()}
            if isinstance(self.schema, Mapping)
            else {None: self.schema} if self.schema is not None else {}
        )

    @cached_property
    def assoc_tables(self) -> set[type[Table]]:
        """Set of all association tables in this DB."""
        return {t for s in self.schema_dict.values() for t in s._assoc_tables}

    @cached_property
    def meta(self) -> sqla.MetaData:
        """Metadata object for this DB instance."""
        return sqla.MetaData()

    @cached_property
    def subs(self) -> dict[type[Table], sqla.Table]:
        """Parsed substitutions map."""
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
        """Maps all table classes in this DB to their SQLA tables."""
        meta = self.meta
        schema_dict = self.schema_dict

        table_map = {}

        for schema_name, schema in schema_dict.items():
            for table in schema._tables:
                sub = self.subs.get(table)
                table_map[table] = table._sqla_table(meta, self.subs).to_metadata(
                    meta,
                    schema=(
                        sub.schema
                        if sub is not None
                        else schema_name or sqla.schema.SchemaConst.RETAIN_SCHEMA
                    ),  # type: ignore # None is supported but not in the stubs
                    referred_schema_fn=partial(
                        map_foreignkey_schema, schema_dict=schema_dict
                    ),
                    name=sub.name if sub is not None else None,
                )

        return table_map

    @cached_property
    def relation_map(self) -> dict[type[Table], set[Rel[Table, Table, Any]]]:
        """Maps all tables in this DB to their outgoing or incoming relations."""
        rels: dict[type[Table], set[Rel]] = {
            table: set() for table in self.table_map.keys()
        }

        for table in self.table_map.keys():
            for rel in table._relations:
                rels[table].add(rel)
                rels[rel.target].add(rel)

        return rels

    @cached_property
    def engine(self) -> sqla.engine.Engine:
        """SQLA Engine for this DB."""
        # Create engine based on backend type
        # For Excel-backends, use duckdb in-memory engine
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
        # Get all tables that need to be validated
        tables = {
            t: v
            for s, v in self.validations.items()
            for t in self[s].table_map.values()
        }

        inspector = sqla.inspect(self.engine)

        # Iterate over all tables and perform validations for each
        for table, validation in tables.items():
            has_table = inspector.has_table(table.name, table.schema)

            if not has_table and validation == "compatible-if-present":
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

    def __post_init__(self):  # noqa: D105
        self.validate()

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self: "DB[N, Any, T]", key: type[T]) -> "DBTable[N, T]": ...

    @overload
    def __getitem__(self: "DB[N, S, Any]", key: type[S]) -> "DB[N, S, S]": ...

    def __getitem__(  # noqa: D105
        self, key: slice | type[T] | type[S]
    ) -> "Self | DBTable[N, T] | DB[N, S, S]":
        if isinstance(key, slice):
            assert (
                key.start is None and key.stop is None and key.step is None
            ), "Only empty slices are supported as keys for DB objects."
            return self

        if issubclass(key, Table):
            # Return a single table.
            self._load_from_excel([key])
            return DBTable(
                db=self,
                select=sqla.select(self.table_map[key]),
                name=key._default_name,
                schema=key,
            )

        if issubclass(key, Schema):
            # Return a new DB instance bound to a sub-schema.
            return DB(
                self.backend,
                key,
                validations=self.validations,
                substitutions=self.substitutions,
            )

    @overload
    def __setitem__(
        self: "DB[N, Any, T]",
        key: slice,
        value: "DB[N, S_cov, Any] | dict[type[T] | str, DataFrame | sqla.Select]",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[N, Any, T]",
        key: type[T],
        value: "DBTable[N, T] | DataFrame | sqla.Select",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[N, S, Any]",
        key: type[S],
        value: "DB[N, S, Any] | dict[type[S] | str, DataFrame | sqla.Select]",
    ) -> None: ...

    def __setitem__(  # noqa: D105
        self,
        key: slice | type[Schema],
        value: "DBTable[N, Any] | DataFrame | sqla.Select | DB[N, Any, Any] | dict[type[Schema] | str, DataFrame | sqla.Select]",  # noqa: E501
    ) -> None:
        table_dict: dict[type[Table], DBTable[N, Any] | DataFrame | sqla.Select] = {}
        if isinstance(key, slice):
            assert (
                key.start is None and key.stop is None and key.step is None
            ), "Only empty slices are supported as keys for DB objects."
            assert isinstance(
                value, DB | dict
            ), "Value must be a DB object or a dictionary."
            table_dict = {k: value[k] for k in self.table_map.keys()}
        elif issubclass(key, Schema):
            assert isinstance(value, DB) or isinstance(
                value, dict
            ), "Value must be a DB object or a dictionary."
            table_dict = {k: value[k] for k in key._tables}
        elif issubclass(key, Table):
            assert (
                isinstance(value, DBTable)
                or isinstance(value, pd.DataFrame)
                or isinstance(value, sqla.Select)
            ), "Value must be a DBTable, DataFrame or SQLA Select object."
            table_dict = {key: value}

        for table, value in table_dict.items():
            self._set_table(
                self._ensure_table_exists(self._table_from_schema(table)), value
            )

        self._save_to_excel(table_dict.keys())

    def to_temp_table(
        self, data: DataFrame | sqla.Select, schema: type[T] | None = None
    ) -> DBTable[N, T]:
        """Create a temporary table from a DataFrame or SQL query."""
        table_name = (
            f"df_{gen_str_hash(data, 10)}"
            if isinstance(data, DataFrame)
            else f"query_{token_hex(5)}"
        )

        sqla_table = (
            sqla.Table(table_name, self.meta, *cols_from_df(data).values())
            if isinstance(data, DataFrame)
            else (
                sqla.Table(
                    table_name,
                    self.meta,
                    *(
                        sqla.Column(name, col.type, primary_key=col.primary_key)
                        for name, col in data.selected_columns.items()
                    ),
                )
                if schema is None
                else self._table_from_schema(schema, sub_name=table_name)
            )
        )

        self._set_table(self._ensure_table_exists(sqla_table), data)

        return DBTable(
            db=self,
            select=sqla.select(sqla_table),
            name=table_name,
            schema=schema or Table,
        )

    def cast(
        self, schema: type[S2], validation: SchemaValidation = "compatible-if-present"
    ) -> "DB[N, S2, S2]":
        """Cast the DB to a different schema."""
        return DB(
            self.backend,
            schema,
            validations={schema: validation},
            substitutions=self.substitutions,
        )

    def transfer(self, backend: Backend[N2]) -> "DB[N2, S_cov, S_contrav]":
        """Transfer the DB to a different backend."""
        other_db = DB(
            backend,
            self.schema,
            validations=self.validations,
            substitutions=self.substitutions,
            create_cross_fk=self.create_cross_fk,
        )

        if self.backend.type == "excel-file":
            self._load_from_excel()

        for sqla_table in self.table_map.values():
            pl.read_database(
                f"SELECT * FROM {sqla_table}", other_db.engine
            ).write_database(str(sqla_table), str(other_db.backend.url))

        return other_db

    def to_graph(
        self: "DB[N, T, Any]", nodes: Sequence[type[T]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        node_tables = [self[n] for n in nodes]

        # Concat all node tables into one.
        node_dfs = [
            n.to_df().reset_index().assign(table=n.sources.pop().name)
            for n in node_tables
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )

        directed_edges = reduce(set.union, (self.relation_map[n] for n in nodes))

        undirected_edges: dict[type[Table], set[tuple[Rel[Table, Table, Any], ...]]] = {
            t: set() for t in nodes
        }
        for n in nodes:
            for at in self.assoc_tables:
                if len(at._relations) == 2:
                    left, right = at._relations
                    if left.target == n:
                        undirected_edges[n].add((left, right))
                    elif right.target == n:
                        undirected_edges[n].add((right, left))

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                *[
                    node_df.loc[
                        node_df["table"] == str((rel._source or Table)._default_name)
                    ]
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[node_df["table"] == str(rel.target._default_name)],
                        left_on=[c.name for c in rel.cols],
                        right_on=[
                            c.name
                            for c in (
                                rel.foreign_key.values()
                                if isinstance(rel.foreign_key, dict)
                                else rel.target._primary_keys
                            )
                        ],
                    )
                    .rename(columns={"node_id": "target"})[["source", "target"]]
                    .assign(ltr=",".join(c.name for c in rel.cols), rtl=None)
                    for rel in directed_edges
                ],
                *[
                    self[str(assoc_table)]
                    .to_df()
                    .merge(
                        node_df.loc[
                            node_df["table"] == str(left_rel.target._default_name)
                        ].dropna(axis="columns", how="all"),
                        left_on=[c.name for c in left_rel.cols],
                        right_on=[
                            c.name
                            for c in (
                                left_rel.foreign_key.values()
                                if isinstance(left_rel.foreign_key, dict)
                                else left_rel.target._primary_keys
                            )
                        ],
                        how="inner",
                    )
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"] == str(left_rel.target._default_name)
                        ].dropna(axis="columns", how="all"),
                        left_on=[c.name for c in right_rel.cols],
                        right_on=[
                            c.name
                            for c in (
                                right_rel.foreign_key.values()
                                if isinstance(right_rel.foreign_key, dict)
                                else right_rel.target._primary_keys
                            )
                        ],
                        how="inner",
                    )
                    .rename(columns={"node_id": "target"})[
                        list(
                            {
                                "source",
                                "target",
                                *(left_rel._source or Table)._columns.keys(),
                            }
                        )
                    ]
                    .assign(
                        ltr=",".join(c.name for c in right_rel.cols),
                        rtl=",".join(c.name for c in left_rel.cols),
                    )
                    for assoc_table, rels in undirected_edges.items()
                    for left_rel, right_rel in rels
                ],
            ],
            ignore_index=True,
        )

        return node_df, edge_df

    def _schema_name(self, schema: type[Table]) -> str | None:
        """Return the schema name for a schema class."""
        for name, s in self.schema_dict.items():
            if issubclass(schema, s):
                return name

        return None

    def _table_from_schema(
        self, schema: type[T], sub_name: str | None = None
    ) -> sqla.Table:
        """Create a SQLA table from a schema class."""
        sqla_table = self.table_map.get(schema)
        if sqla_table is None:
            if schema in self.substitutions:
                sqla_table = self.subs[schema]
            else:
                sqla_table = schema._sqla_table(
                    self.meta,
                    subs=self.subs,
                    name=sub_name,
                    schema_name=self._schema_name(schema),
                )

        return sqla_table

    def _ensure_table_exists(self, sqla_table: sqla.Table) -> sqla.Table:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_table(
            sqla_table.name, schema=sqla_table.schema
        ):
            self._create_sqla_table(sqla_table)

        return sqla_table

    def _create_sqla_table(self, sqla_table: sqla.Table) -> None:
        """Create SQL-side table from Table class."""
        if not self.create_cross_fk:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            remove_external_fk(sqla_table)

        sqla_table.create(self.engine)

    def _set_table(
        self,
        sqla_table: sqla.Table,
        value: DBTable[N, T] | DataFrame | sqla.Select,
    ) -> None:
        """Set a table in the database to a new value."""
        if isinstance(value, pd.DataFrame):
            # Upload dataframe to SQL
            value.reset_index()[list(sqla_table.c.keys())].to_sql(
                str(sqla_table),
                self.engine,
                if_exists="replace",
                index=False,
            )
        elif isinstance(value, pl.DataFrame):
            # Upload dataframe to SQL
            value[list(sqla_table.c.keys())].write_database(
                str(sqla_table),
                str(self.engine.url),
                if_table_exists="replace",
            )
        else:
            select = value if isinstance(value, sqla.Select) else value.select

            # Transfer table or query results to SQL table
            with self.engine.begin() as con:
                con.execute(
                    sqla.insert(sqla_table).from_select(
                        select.selected_columns.keys(), select
                    )
                )

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
                pl.read_excel(file, sheet_name=table._default_name).write_database(
                    str(self.table_map[table]), str(self.engine.url)
                )

    def _save_to_excel(self, tables: Iterable[type[Table]] | None) -> None:
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
                pl.read_database(
                    f"SELECT * FROM {self.table_map[table]}", self.engine
                ).write_excel(wb, worksheet=table._default_name)

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)
