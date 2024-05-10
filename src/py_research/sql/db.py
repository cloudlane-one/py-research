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
    AttrRef,
    Idx,
    Rec,
    Rec2,
    Rec_cov,
    Record,
    Rel,
    RelRef,
    Required,
    Schema,
    Val,
)

DataFrame: TypeAlias = pd.DataFrame | pl.DataFrame
Series: TypeAlias = pd.Series | pl.Series

Name = TypeVar("Name", bound=LiteralString)
Name2 = TypeVar("Name2", bound=LiteralString)

DBS_contrav = TypeVar("DBS_contrav", contravariant=True, bound=Record | Schema)

Idx_cov = TypeVar("Idx_cov", covariant=True, bound=Hashable)
Idx2 = TypeVar("Idx2", bound=Hashable)
Idx3 = TypeVar("Idx3", bound=Hashable)
IdxTup = TypeVarTuple("IdxTup")

Df = TypeVar("Df", bound=DataFrame)


IdxStart: TypeAlias = Idx | tuple[Idx, *tuple[Any, ...]]
IdxStartEnd: TypeAlias = tuple[Idx, *tuple[Any, ...], Idx2]
IdxTupEnd: TypeAlias = tuple[*IdxTup, *tuple[Any], Idx2]

RecInput: TypeAlias = (
    DataFrame | Iterable[Rec] | Mapping[Idx, Rec] | sqla.Select[tuple[Rec]] | Rec
)
ValInput: TypeAlias = (
    Series | Iterable[Val] | Mapping[Idx, Val] | sqla.Select[tuple[Val]] | Val
)


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


@dataclass
class DataBase(Generic[Name, DBS_contrav]):
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

    validate_on_init: bool = True
    create_cross_fk: bool = True

    def __post_init__(self):  # noqa: D105
        if self.validate_on_init:
            self.validate()

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
    def record_types(self) -> set[type[Record]]:
        """Return all record types as per the defined schema."""
        recs = self.subs.keys()

        if len(recs) == 0:
            # No records retrieved from substitutions,
            # so schema must be defined via set of types.
            assert isinstance(self.schema, type | set)
            types = self.schema if isinstance(self.schema, set) else set([self.schema])
            types = {rec.type if isinstance(rec, Required) else rec for rec in types}
            recs = {rec for rec in types if issubclass(rec, Record)} | {
                rec
                for schema in types
                if issubclass(schema, Schema)
                for rec in schema._record_types
            }

        return set(recs)

    @cached_property
    def assoc_types(self) -> set[type[Record]]:
        """Set of all association tables in this DB."""
        assoc_types = set()
        for rec in self.record_types:
            pks = set([attr.name for attr in rec._primary_keys()])
            fks = set([attr.name for rel in rec._rels for attr in rel.fk_map.keys()])
            if pks in fks:
                assoc_types.add(rec)

        return assoc_types

    @cached_property
    def relation_map(self) -> dict[type[Record], set[Rel]]:
        """Maps all tables in this DB to their outgoing or incoming relations."""
        rels: dict[type[Record], set[Rel]] = {
            table: set() for table in self.record_types
        }

        for rec in self.record_types:
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
        tables = {
            rec._sqla_table(self.meta, self.subs): required
            for rec, required in types.items()
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

    def __getitem__(self, key: type[Rec]) -> "DataSet[Name, Rec, None]":
        """Return the dataset for given record type."""
        return DataSet(self, sqla.select(key._sqla_table(self.meta, self.subs)), key)

    def __setitem__(  # noqa: D105
        self,
        key: type[Rec],
        value: "DataSet[Name, Rec, Idx | None] | RecInput[Rec, Idx] | sqla.Select[tuple[Rec]]",  # noqa: E501
    ) -> None:
        sqla_table = self._ensure_table_exists(key._sqla_table(self.meta, self.subs))
        DataSet(self, sqla.select(sqla_table), key)[:] = value

    def dataset(
        self, data: RecInput[Rec, Any], record_type: type[Rec] | None = None
    ) -> "DataSet[Name, Rec, None]":
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        table_name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, DataFrame)
            else f"temp_{token_hex(5)}"
        )

        sqla_table = None

        if record_type is not None:
            sqla_table = record_type._sqla_table(self.meta, self.subs)
        elif isinstance(data, DataFrame):
            sqla_table = sqla.Table(table_name, self.meta, *cols_from_df(data).values())
        elif isinstance(data, Record):
            sqla_table = data._sqla_table(self.meta, self.subs)
        elif has_type(data, Iterable[Record]):
            sqla_table = next(data.__iter__())._sqla_table(self.meta, self.subs)
        elif has_type(data, Mapping[Any, Record]):
            sqla_table = next(data.values().__iter__())._sqla_table(
                self.meta, self.subs
            )
        elif isinstance(data, sqla.Select):
            sqla_table = sqla.Table(
                table_name,
                self.meta,
                *(
                    sqla.Column(name, col.type, primary_key=col.primary_key)
                    for name, col in data.selected_columns.items()
                ),
            )
        else:
            raise TypeError("Unsupported type given as value")

        sqla_table = self._ensure_table_exists(sqla_table)

        ds = DataSet(self, sqla.select(sqla_table), record_type)
        ds[:] = data
        return ds

    def transfer(self, backend: Backend[Name2]) -> "DataBase[Name2, DBS_contrav]":
        """Transfer the DB to a different backend."""
        other_db = DataBase(
            backend,
            self.schema,
            create_cross_fk=self.create_cross_fk,
            validate_on_init=False,
        )

        if self.backend.type == "excel-file":
            self._load_from_excel()

        for rec in self.record_types:
            sqla_table = rec._sqla_table(self.meta, self.subs)
            pl.read_database(
                f"SELECT * FROM {sqla_table}", other_db.engine
            ).write_database(str(sqla_table), str(other_db.backend.url))

        return other_db

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
            n.to_df().reset_index().assign(table=n.record_type._default_table_name())
            for n in node_tables
            if n.record_type is not None
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )

        directed_edges = reduce(set.union, (self.relation_map[n] for n in nodes))

        undirected_edges: dict[type[Record], set[tuple[Rel, ...]]] = {
            t: set() for t in nodes
        }
        for n in nodes:
            for at in self.assoc_types:
                if len(at._rels) == 2:
                    left, right = at._rels
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
                        == str((rel._record_type or Record)._default_table_name())
                    ]
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"]
                            == str(rel.target_type._default_table_name())
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
                    .to_df()
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
                                *(left_rel._record_type or Record)._all_attrs().keys(),
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

    def _load_from_excel(self, record_types: list[type[Record]] | None = None) -> None:
        """Load all tables from Excel."""
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        path = (
            self.backend.url.get()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url
        )

        with open(path, "rb") as file:
            for rec in record_types or self.record_types:
                pl.read_excel(
                    file, sheet_name=rec._default_table_name()
                ).write_database(
                    str(rec._sqla_table(self.meta, self.subs)), str(self.engine.url)
                )

    def _save_to_excel(self, record_types: Iterable[type[Record]] | None) -> None:
        """Save all tables to Excel."""
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        file = (
            BytesIO()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url.open("wb")
        )

        with ExcelWorkbook(file) as wb:
            for rec in record_types or self.record_types:
                pl.read_database(
                    f"SELECT * FROM {rec._sqla_table(self.meta, self.subs)}",
                    self.engine,
                ).write_excel(wb, worksheet=rec._default_table_name())

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)


@dataclass
class DataSet(Generic[Name, Rec_cov, Idx_cov]):
    """Dataset selection."""

    db: DataBase[Name, Any]
    selection: sqla.Select[tuple[Rec_cov]]
    record_type: type[Rec_cov] | None = None
    index: AttrRef[Any, Idx_cov] | None = None

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Idx2], None]",
        key: AttrRef[Rec_cov, Val],
    ) -> "DataSet[Name, Record[Val, None], Idx2]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Rec, Idx]", key: AttrRef[Rec, Val]
    ) -> "DataSet[Name, Record[Val, None], Idx]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Idx2], None]", key: AttrRef[Any, Val]
    ) -> "DataSet[Name, Record[Val, None], IdxStart[Idx2]]": ...

    @overload
    def __getitem__(
        self, key: AttrRef[Any, Val]
    ) -> "DataSet[Name, Record[Val, None], IdxStart[Idx_cov]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Idx2], None]",
        key: RelRef[Rec_cov, Rec2, Idx3],
    ) -> "DataSet[Name, Rec2, tuple[Idx2, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Rec_cov, tuple[*IdxTup]]",
        key: RelRef[Rec_cov, Rec2, Idx3],
    ) -> "DataSet[Name, Rec2, tuple[*IdxTup, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Rec_cov, Idx]",
        key: RelRef[Rec_cov, Rec2, Idx3],
    ) -> "DataSet[Name, Rec2, tuple[Idx, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Any, tuple[*IdxTup]]", key: RelRef[Any, Rec2, Idx3]
    ) -> "DataSet[Name, Rec2, IdxTupEnd[*IdxTup, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Idx2], None]",
        key: RelRef[Any, Rec2, Idx3],
    ) -> "DataSet[Name, Rec2, IdxStartEnd[Idx2, Idx3]]": ...

    @overload
    def __getitem__(
        self, key: RelRef[Any, Rec2, Idx3]
    ) -> "DataSet[Name, Rec2, IdxStartEnd[Idx_cov, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Idx2], None]", key: Iterable[Idx2]
    ) -> "DataSet[Name, Rec_cov, None]": ...

    @overload
    def __getitem__(self, key: Iterable[Idx_cov]) -> Self: ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Idx2], None]", key: Idx2
    ) -> Val: ...

    @overload
    def __getitem__(self: "DataSet[Name, Record[Val, Any], Idx]", key: Idx) -> Val: ...

    @overload
    def __getitem__(self, key: sqla.ColumnElement[bool]) -> Self: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(  # noqa: D105
        self,
        key: AttrRef[Any, Val] | RelRef | Iterable[Hashable] | Hashable | slice,
    ) -> "DataSet | Val":
        raise NotImplementedError()

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Idx2], None]",
        key: AttrRef[Rec_cov, Val],
        value: "DataSet[Name, Record[Val, Idx2], None] | DataSet[Name, Record[Val, Any], Idx2] | ValInput[Val, Idx2]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Any], Idx]",
        key: AttrRef[Rec_cov, Val],
        value: "DataSet[Name, Record[Val, Idx], None] | DataSet[Name, Record[Val, Any], Idx] | ValInput[Val, Idx]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Idx2], None]",
        key: AttrRef[Any, Val],
        value: "DataSet[Name, Record[Val, IdxStart[Idx2]], None] | DataSet[Name, Record[Val, Any], IdxStart[Idx2]] | ValInput[Val, IdxStart[Idx2]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Any], Idx]",
        key: AttrRef[Any, Val],
        value: "DataSet[Name, Record[Val, IdxStart[Idx]], None] | DataSet[Name, Record[Val, Any], IdxStart[Idx]] | ValInput[Val, IdxStart[Idx]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Idx2], None]",
        key: RelRef[Rec_cov, Rec2, Idx3],
        value: "DataSet[Name, Rec2, tuple[Idx2, Idx3]] | RecInput[Rec2, tuple[Idx2, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: RelRef[Rec_cov, Rec2, Idx3],
        value: "DataSet[Name, Rec2, tuple[Idx_cov, Idx3]] | RecInput[Rec2, tuple[Idx_cov, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Rec_cov, tuple[*IdxTup]]",
        key: RelRef[Rec_cov, Rec2, Idx3],
        value: "DataSet[Name, Rec2, tuple[*IdxTup, Idx3]] | RecInput[Rec2, tuple[*IdxTup, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Idx2], None]",
        key: RelRef[Any, Rec2, Idx3],
        value: "DataSet[Name, Rec2, IdxStartEnd[Idx2, Idx3]] | RecInput[Rec2, IdxStartEnd[Idx2, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: RelRef[Any, Rec2, Idx3],
        value: "DataSet[Name, Rec2, IdxStartEnd[Idx_cov, Idx3]] | RecInput[Rec2, IdxStartEnd[Idx_cov, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Any, tuple[*IdxTup]]",
        key: RelRef[Any, Rec2, Idx3],
        value: "DataSet[Name, Rec2, IdxTupEnd[*IdxTup, Idx3]] | RecInput[Rec2, IdxTupEnd[*IdxTup, Idx3]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Idx2], Any]",
        key: Iterable[Idx2],
        value: "DataSet[Name, Rec_cov, Idx2 | None] | Iterable[Rec_cov] | DataFrame",
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: Iterable[Idx_cov],
        value: "DataSet[Name, Rec_cov, Idx | None] | Iterable[Rec_cov] | DataFrame",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Idx2], None]", key: Idx2, value: Val
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Any], Idx]", key: Idx, value: Val
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
        key: AttrRef[Any, Val] | RelRef | Iterable[Hashable] | Hashable | slice,
        value: "DataSet | RecInput[Rec_cov, Any] | ValInput[Val, Any] | Val",
    ) -> None:
        # if isinstance(value, pd.DataFrame):
        #     # Upload dataframe to SQL
        #     value.reset_index()[list(sqla_table.c.keys())].to_sql(
        #         str(sqla_table),
        #         self.engine,
        #         if_exists="replace",
        #         index=False,
        #     )
        # elif isinstance(value, pl.DataFrame):
        #     # Upload dataframe to SQL
        #     value[list(sqla_table.c.keys())].write_database(
        #         str(sqla_table),
        #         str(self.engine.url),
        #         if_table_exists="replace",
        #     )
        # elif isinstance(value, Iterable):
        #     pass
        # else:
        #     select = value if isinstance(value, sqla.Select) else value.selection

        #     # Transfer table or query results to SQL table
        #     with self.engine.begin() as con:
        #         con.execute(
        #             sqla.insert(sqla_table).from_select(
        #                 select.selected_columns.keys(), select
        #             )
        #         )
        raise NotImplementedError()

    def __clause_element__(self) -> sqla.Subquery:
        """Return subquery for the current selection to be used inside SQL clauses."""
        if self.selection is None:
            raise ValueError("Only a selected DB can be used as a clause element.")
        return self.selection.subquery()

    def to_df(self, kind: type[Df] = pd.DataFrame) -> Df:
        """Download table selection to dataframe.

        Returns:
            Dataframe containing the data.
        """
        if kind is pl.DataFrame:
            return cast(Df, pl.read_database(self.selection, self.db.engine))
        else:
            with self.db.engine.connect() as con:
                return cast(Df, pd.read_sql(self.selection, con))
