"""Abstract Python interface for SQL databases."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, reduce
from io import BytesIO
from pathlib import Path
from secrets import token_hex
from typing import (
    Any,
    Generic,
    Literal,
    LiteralString,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    cast,
    overload,
)

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import yarl
from bidict import bidict
from cloudpathlib import CloudPath
from pandas.api.types import (
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from typing_extensions import Self
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.files import HttpFile
from py_research.hashing import gen_str_hash
from py_research.reflect.types import has_type
from py_research.sql.schema import (
    Attr,
    AttrRef,
    Idx,
    Prop,
    Rec,
    Rec2,
    Record,
    Rel,
    RelRef,
    Required,
    Sch,
    Schema,
    Val,
)

DataFrame: TypeAlias = pd.DataFrame | pl.DataFrame
Series: TypeAlias = pd.Series | pl.Series

Name = TypeVar("Name", bound=LiteralString)
Name2 = TypeVar("Name2", bound=LiteralString)

DBS_contrav = TypeVar("DBS_contrav", contravariant=True, bound=Record | Schema)
Rec_cov = TypeVar("Rec_cov", covariant=True, bound=Record)

Idx_cov = TypeVar("Idx_cov", covariant=True, bound=Hashable)
Idx2 = TypeVar("Idx2", bound=Hashable)
Idx3 = TypeVar("Idx3", bound=Hashable)
IdxTup = TypeVarTuple("IdxTup")

Df = TypeVar("Df", bound=DataFrame)


def map_df_dtype(c: pd.Series | pl.Series) -> sqla.types.TypeEngine:  # noqa: C901
    """Map pandas dtype to sqlalchemy type."""
    if isinstance(c, pd.Series):
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
    else:
        if c.dtype.is_temporal():
            return sqla.types.DATE()
        elif c.dtype.is_integer():
            return sqla.types.INTEGER()
        elif c.dtype.is_float():
            return sqla.types.FLOAT()
        elif c.dtype.is_(pl.String):
            max_len = c.str.len_bytes().max()
            assert max_len is not None
            if (max_len) < 16:  # type: ignore
                return sqla.types.CHAR(max_len)  # type: ignore
            elif max_len < 256:  # type: ignore
                return sqla.types.VARCHAR(max_len)  # type: ignore

    return sqla.types.BLOB()


def cols_from_df(df: DataFrame) -> dict[str, sqla.Column]:
    """Create columns from DataFrame."""
    if isinstance(df, pd.DataFrame) and len(df.index.names) > 1:
        raise NotImplementedError("Multi-index not supported yet.")

    return {
        **(
            {
                level: sqla.Column(
                    level,
                    map_df_dtype(df.index.get_level_values(level).to_series()),
                    primary_key=True,
                )
                for level in df.index.names
            }
            if isinstance(df, pd.DataFrame)
            else {}
        ),
        **{
            str(df[col].name): sqla.Column(str(df[col].name), map_df_dtype(df[col]))
            for col in df.columns
        },
    }


def remove_external_fk(table: sqla.Table):
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


@dataclass(frozen=True)
class Backend(Generic[Name]):
    """SQL backend for DB."""

    name: Name
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


IdxStart: TypeAlias = Idx | tuple[Idx, *tuple[Any, ...]]
IdxStartEnd: TypeAlias = tuple[Idx, *tuple[Any, ...], Idx2]
IdxTupEnd: TypeAlias = tuple[*IdxTup, *tuple[Any], Idx2]
DfInput: TypeAlias = DataFrame | Iterable[Val] | Mapping[Idx, Val]
SeriesInput: TypeAlias = Series | Iterable[Val] | Mapping[Idx, Val]


@dataclass
class DB(Generic[Name, DBS_contrav, Rec_cov, Idx]):
    """Active connection to a SQL server."""

    backend: Backend[Name]
    schema: (
        type[DBS_contrav]
        | Required[DBS_contrav]
        | set[type[DBS_contrav] | Required[DBS_contrav]]
        | Mapping[
            type[DBS_contrav] | Required[DBS_contrav],
            str | sqla.TableClause,
        ]
        | None
    ) = None
    index: AttrRef[Any, Idx] | None = None

    create_cross_fk: bool = True
    validate_schema: bool = True

    selection: sqla.Select[tuple[Rec_cov]] | None = None

    @cached_property
    def meta(self) -> sqla.MetaData:
        """Metadata object for this DB instance."""
        return sqla.MetaData()

    @cached_property
    def subs(self) -> Mapping[type[Record], sqla.TableClause]:
        """Substitutions for tables in this DB."""
        if has_type(
            self.schema,
            Mapping[type[Record] | Required[Record], str | sqla.TableClause],
        ):
            return {
                (rec.type if isinstance(rec, Required) else rec): (
                    sub if isinstance(sub, sqla.TableClause) else sqla.table(sub)
                )
                for rec, sub in self.schema.items()
            }

        if has_type(self.schema, Mapping[type[Schema] | Required[Schema], str]):
            return {
                rec: sqla.table(rec._table_name, schema=schema_name)
                for schema, schema_name in self.schema.items()
                for rec in (
                    schema.type if isinstance(schema, Required) else schema
                )._record_types
            }

        return {}

    @cached_property
    def table_map(self) -> bidict[type[Record], sqla.Table]:
        """Maps all table classes in this DB to their SQLA tables."""
        meta = self.meta
        recs = self.subs.keys()

        if len(recs) == 0:
            assert isinstance(self.schema, type | set)
            types = self.schema if isinstance(self.schema, set) else set([self.schema])
            types = {rec.type if isinstance(rec, Required) else rec for rec in types}
            recs = {rec for rec in types if issubclass(rec, Record)} | {
                rec
                for schema in types
                if issubclass(schema, Schema)
                for rec in schema._record_types
            }

        return bidict({rec: rec._sqla_table(meta, self.subs) for rec in recs})

    @cached_property
    def assoc_types(self) -> set[type[Record]]:
        """Set of all association tables in this DB."""
        assoc_types = set()
        for rec in self.table_map.keys():
            pks = set([attr.name for attr in rec._primary_keys()])
            fks = set([attr.name for rel in rec._rels for attr in rel.fk_map.keys()])
            if pks in fks:
                assoc_types.add(rec)

        return assoc_types

    @cached_property
    def relation_map(self) -> dict[type[Record], set[Rel]]:
        """Maps all tables in this DB to their outgoing or incoming relations."""
        rels: dict[type[Record], set[Rel]] = {
            table: set() for table in self.table_map.keys()
        }

        for rec in self.table_map.keys():
            for rel in rec._rels:
                rels[rec].add(rel)
                rels[rel.target_type].add(rel)

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
        if not self.validate_schema:
            return

        types = None

        if has_type(
            self.schema,
            Mapping[type[Record] | Required[Record], str | sqla.TableClause],
        ):
            types = {
                (rec.type if isinstance(rec, Required) else rec): isinstance(
                    rec, Required
                )
                for rec in self.schema.keys()
            }
        elif has_type(self.schema, Mapping[type[Schema] | Required[Schema], str]):
            types = {
                rec: isinstance(schema, Required)
                for schema, schema_name in self.schema.items()
                for rec in (
                    schema.type if isinstance(schema, Required) else schema
                )._record_types
            }
        elif (
            has_type(self.schema, set[type[Schema] | Required[Schema]])
            or has_type(self.schema, type[Schema])
            or has_type(self.schema, Required[Schema])
        ):
            schema_items = (
                self.schema if isinstance(self.schema, set) else {self.schema}
            )
            types = {
                rec: isinstance(self.schema, Required)
                for schema in schema_items
                for rec in (
                    schema.type if isinstance(schema, Required) else schema
                )._record_types
            }
        elif (
            has_type(self.schema, set[type[Record] | Required[Record]])
            or has_type(self.schema, type[Record])
            or has_type(self.schema, Required[Record])
        ):
            recs = self.schema if isinstance(self.schema, set) else {self.schema}
            types = {
                rec.type if isinstance(rec, Required) else rec: isinstance(
                    rec, Required
                )
                for rec in recs
            }

        assert types is not None
        tables = {self.table_map[rec]: required for rec, required in types.items()}

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

    def __post_init__(self):  # noqa: D105
        self.validate()

    @overload
    def __getitem__(
        self: "DB[Name, Rec, Rec, Idx]", key: type[Rec]
    ) -> "DB[Name, DBS_contrav, Rec, Idx]": ...

    @overload
    def __getitem__(
        self: "DB[Name, Sch, Rec_cov, Idx]", key: type[Sch]
    ) -> "DB[Name, Sch, Rec_cov, Idx]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "DB[Name, DBS_contrav, Record[Any, Idx2], None]",
        key: AttrRef[Rec_cov, Val],
    ) -> "DB[Name, DBS_contrav, Record[Val, None], Idx2]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "DB[Name, DBS_contrav, Any, Idx]", key: AttrRef[Rec_cov, Val]
    ) -> "DB[Name, DBS_contrav, Record[Val, None], Idx]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Record[Any, Idx2], None]", key: AttrRef[Any, Val]
    ) -> "DB[Name, DBS_contrav, Record[Val, None], IdxStart[Idx2]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Any, Idx]", key: AttrRef[Any, Val]
    ) -> "DB[Name, DBS_contrav, Record[Val, None], IdxStart[Idx]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Record[Any, Idx2], None]",
        key: RelRef[Rec_cov, Rec2, Idx3],
    ) -> "DB[Name, DBS_contrav, Rec2, tuple[Idx2, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Rec_cov, tuple[*IdxTup]]",
        key: RelRef[Rec_cov, Rec2, Idx3],
    ) -> "DB[Name, DBS_contrav, Rec2, tuple[*IdxTup, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Rec_cov, Idx]",
        key: RelRef[Rec_cov, Rec2, Idx3],
    ) -> "DB[Name, DBS_contrav, Rec2, tuple[Idx, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Any, tuple[*IdxTup]]", key: RelRef[Any, Rec2, Idx3]
    ) -> "DB[Name, DBS_contrav, Rec2, IdxTupEnd[*IdxTup, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Record[Any, Idx2], None]",
        key: RelRef[Any, Rec2, Idx3],
    ) -> "DB[Name, DBS_contrav, Rec2, IdxStartEnd[Idx2, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Any, Idx]", key: RelRef[Any, Rec2, Idx3]
    ) -> "DB[Name, DBS_contrav, Rec2, IdxStartEnd[Idx, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Idx2], None]", key: Iterable[Idx2]
    ) -> "DB[Name, DBS_contrav, Rec_cov, None]": ...

    @overload
    def __getitem__(self, key: Iterable[Idx]) -> Self: ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Idx2], None]", key: Idx2
    ) -> Val: ...

    @overload
    def __getitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Any], Idx]", key: Idx
    ) -> Val: ...

    @overload
    def __getitem__(self, key: sqla.ColumnElement[bool]) -> Self: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(  # noqa: D105
        self,
        key: (
            type[Schema]
            | type[Record]
            | AttrRef[Any, Val]
            | RelRef
            | Iterable[Hashable]
            | Hashable
            | slice
        ),
    ) -> "DB | Val":
        raise NotImplementedError()

    @overload
    def __setitem__(
        self: "DB[Name, Rec, Rec, Idx]",
        key: type[Rec],
        value: "DB[Name, DBS_contrav, Rec, Idx | None] | DfInput[Rec, Idx]",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, Sch, Any, Idx]",
        key: type[Sch],
        value: "DB[Name, Sch, Any, Idx | None] | Mapping[Sch, DfInput[Sch, Idx]]",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Idx2], None]",
        key: AttrRef[Rec_cov, Val],
        value: "DB[Name, Record[Val, Idx2], Any, None] | DB[Name, Record[Val, Any], Any, Idx2] | SeriesInput[Val, Idx2]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Any], Idx]",
        key: AttrRef[Rec_cov, Val],
        value: "DB[Name, Record[Val, Idx], Any, None] | DB[Name, Record[Val, Any], Any, Idx] | SeriesInput[Val, Idx]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Idx2], None]",
        key: AttrRef[Any, Val],
        value: "DB[Name, Record[Val, IdxStart[Idx2]], Any, None] | DB[Name, Record[Val, Any], Any, IdxStart[Idx2]] | SeriesInput[Val, IdxStart[Idx2]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Any], Idx]",
        key: AttrRef[Any, Val],
        value: "DB[Name, Record[Val, IdxStart[Idx]], Any, None] | DB[Name, Record[Val, Any], Any, IdxStart[Idx]] | SeriesInput[Val, IdxStart[Idx]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Any, Idx2], None]",
        key: RelRef[Rec_cov, Rec2, Idx3],
        value: "DB[Name, Rec2, Rec2, tuple[Idx2, Idx3]] | DfInput[Rec2, tuple[Idx2, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Rec_cov, Idx]",
        key: RelRef[Rec_cov, Rec2, Idx3],
        value: "DB[Name, Rec2, Rec2, tuple[Idx, Idx3]] | DfInput[Rec2, tuple[Idx, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Rec_cov, tuple[*IdxTup]]",
        key: RelRef[Rec_cov, Rec2, Idx3],
        value: "DB[Name, Rec2, Rec2, tuple[*IdxTup, Idx3]] | DfInput[Rec2, tuple[*IdxTup, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Any, Idx2], None]",
        key: RelRef[Any, Rec2, Idx3],
        value: "DB[Name, Rec2, Rec2, IdxStartEnd[Idx2, Idx3]] | DfInput[Rec2, IdxStartEnd[Idx2, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Rec_cov, Idx]",
        key: RelRef[Any, Rec2, Idx3],
        value: "DB[Name, Rec2, Rec2, IdxStartEnd[Idx, Idx3]] | DfInput[Rec2, IdxStartEnd[Idx, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Rec_cov, tuple[*IdxTup]]",
        key: RelRef[Any, Rec2, Idx3],
        value: "DB[Name, Rec2, Rec2, IdxTupEnd[*IdxTup, Idx3]] | DfInput[Rec2, IdxTupEnd[*IdxTup, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Idx2], Idx_cov]",
        key: Iterable[Idx2],
        value: "DB[Name, Any, Rec_cov, Idx2 | None] | Iterable[Rec_cov] | DataFrame",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Any], Idx]",
        key: Iterable[Idx],
        value: "DB[Name, Any, Rec_cov, Idx | None] | Iterable[Rec_cov] | DataFrame",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Idx2], None]", key: Idx2, value: Val
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DB[Name, DBS_contrav, Record[Val, Any], Idx]", key: Idx, value: Val
    ) -> None: ...

    @overload
    def __setitem__(
        self, key: sqla.ColumnElement[bool], value: Self | Iterable[Rec_cov] | DataFrame
    ) -> None: ...

    @overload
    def __setitem__(
        self, key: slice, value: Self | Iterable[Rec_cov] | DataFrame
    ) -> None: ...

    def __setitem__(  # noqa: D105
        self,
        key: (
            type[Schema]
            | type[Record]
            | AttrRef[Any, Val]
            | RelRef
            | Iterable[Hashable]
            | Hashable
            | slice
        ),
        value: "DB | DfInput[Rec_cov, Any] | SeriesInput[Val, Any] | Val",
    ) -> None:
        raise NotImplementedError()

    def __clause_element__(self) -> sqla.Subquery:
        """Return subquery for the current selection to be used inside SQL clauses."""
        if self.selection is None:
            raise ValueError("Only a selected DB can be used as a clause element.")
        return self.selection.subquery()

    def to_temp_table(
        self, data: DataFrame | sqla.Select, schema: type[T] | None = None
    ) -> DBTable[Name, T]:
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

    def cast(
        self, schema: type[S2], validation: SchemaValidation = "compatible-if-present"
    ) -> "DB[Name, S2, S2]":
        """Cast the DB to a different schema."""
        return DB(
            self.backend,
            schema,
            validations={schema: validation},
            substitutions=self.substitutions,
        )

    def transfer(self, backend: Backend[Name2]) -> "DB[Name2, S_cov, S_contrav]":
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
        self: "DB[Name, T, Any]", nodes: Sequence[type[T]]
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
            for at in self.assoc_types:
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
        value: DBTable[Name, T] | DataFrame | sqla.Select,
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
