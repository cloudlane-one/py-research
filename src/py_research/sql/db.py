"""Abstract Python interface for SQL databases."""

from collections.abc import Hashable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass, field
from functools import cache, cached_property, partial, reduce
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

import openpyxl
import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.dialects.mysql as mysql
import sqlalchemy.dialects.postgresql as postgresql
import sqlalchemy.sql.visitors as sqla_visitors
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
    Agg,
    Attr,
    BaseIdx,
    Idx,
    Key,
    MergeTup,
    Prop,
    Rec,
    Rec2,
    Rec_cov,
    Record,
    Rel,
    RelMerge,
    RelTree,
    Require,
    Schema,
    SingleIdx,
    Val,
)

DataFrame: TypeAlias = pd.DataFrame | pl.DataFrame
Series: TypeAlias = pd.Series | pl.Series

Name = TypeVar("Name", bound=LiteralString)
Name2 = TypeVar("Name2", bound=LiteralString)


DBS_contrav = TypeVar("DBS_contrav", contravariant=True, bound=Record | Schema)

Idx_cov = TypeVar("Idx_cov", covariant=True, bound=Hashable | BaseIdx)
Key2 = TypeVar("Key2", bound=Hashable)
Key3 = TypeVar("Key3", bound=Hashable)
IdxTup = TypeVarTuple("IdxTup")

Df = TypeVar("Df", bound=DataFrame)
Dl = TypeVar("Dl", bound=DataFrame | Record)


IdxStart: TypeAlias = Key | tuple[Key, *tuple[Any, ...]]
IdxStartEnd: TypeAlias = tuple[Key, *tuple[Any, ...], Key2]
IdxTupStartEnd: TypeAlias = tuple[*IdxTup, *tuple[Any], Key2]

RecInput: TypeAlias = (
    DataFrame | Iterable[Rec] | Mapping[Key, Rec] | sqla.Select[tuple[Key, Rec]] | Rec
)
ValInput: TypeAlias = Series | Mapping[Key, Val] | sqla.Select[tuple[Key, Val]] | Val


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


@dataclass(frozen=True)
class DataBase(Generic[Name]):
    """Active connection to a SQL server."""

    backend: Backend[Name]
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
    create_cross_fk: bool = True

    overlay: str | None = None
    _overlay_subs: dict[type[Record], sqla.TableClause] = field(default_factory=dict)

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
        subs = {}

        if self.records is not None:
            subs = {
                rec: (sub if isinstance(sub, sqla.TableClause) else sqla.table(sub))
                for rec, sub in self.records.items()
                if not isinstance(sub, Require)
            }

        if isinstance(self.schema, Mapping):
            subs = {
                rec: sqla.table(rec._table_name, schema=schema_name)
                for schema, schema_name in self.schema.items()
                for rec in schema._record_types
            }

        return {**subs, **self._overlay_subs}

    @cached_property
    def record_types(self) -> set[type[Record]]:
        """Return all record types as per the defined schema."""
        if self.schema is None:
            return set()

        types = (
            set(self.schema.keys())
            if isinstance(self.schema, Mapping)
            else set([self.schema])
        )
        recs = {
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
            pks = set([attr.name for attr in rec._indexes()])
            fks = set([attr.name for rel in rec._rels() for attr in rel.fk_map.keys()])
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
            for rel in rec._rels():
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
        if self.backend.type == "excel-file":
            self._load_from_excel([key])
        table = self._get_table(key)
        return DataSet(self, table, selection=key)

    def __setitem__(  # noqa: D105
        self,
        key: type[Rec],
        value: "DataSet[Name, Rec, Key] | RecInput[Rec, Key] | sqla.Select[tuple[Rec]]",  # noqa: E501
    ) -> None:
        if self.backend.type == "excel-file":
            self._load_from_excel([key])

        if key not in self._overlay_subs:
            self._overlay_subs[key] = self._get_table(key)

        table = self._ensure_table_exists(self._get_table(key))
        cols = list(table.c.keys())
        table_fqn = str(table)

        if isinstance(value, pd.DataFrame):
            value.reset_index()[cols].to_sql(
                table_fqn,
                self.engine,
                if_exists="replace",
                index=False,
            )
        elif isinstance(value, pl.DataFrame):
            value[cols].write_database(
                table_fqn,
                str(self.engine.url),
                if_table_exists="replace",
            )
        elif isinstance(value, Iterable):
            raise NotImplementedError(
                "Assignment via Iterable of records not implemented yet."
            )
        elif isinstance(value, Record):
            raise NotImplementedError(
                "Assignment via single record instance not implemented yet."
            )
        else:
            select = value if isinstance(value, sqla.Select) else value.select()

            # Transfer table or query results to SQL table
            with self.engine.begin() as con:
                con.execute(sqla.delete(table))
                con.execute(sqla.insert(table).from_select(table.c.keys(), select))

        if self.backend.type == "excel-file":
            self._save_to_excel([key])

    def __delitem__(self, key: type[Rec]) -> None:  # noqa: D105
        if key not in self._overlay_subs:
            self._overlay_subs[key] = self._get_table(key)

        table = self._ensure_table_exists(self._get_table(key))
        with self.engine.begin() as con:
            table.drop(con)

        if self.backend.type == "excel-file":
            self._delete_from_excel([key])

    def __or__(self, other: "DataBase[Name]") -> "DataBase[Name]":
        """Merge two databases."""
        new_db = DataBase(
            self.backend,
            create_cross_fk=self.create_cross_fk,
            validate_on_init=False,
            overlay=token_hex(5),
        )

        for rec in self.record_types | other.record_types:
            new_db[rec] = self[rec] | other[rec]

        return new_db

    def dataset(
        self,
        data: RecInput[Rec, Any],
        record_type: type[Rec] | None = None,
        foreign_keys: Mapping[str, Attr] | None = None,
    ) -> "DataSet[Name, Rec, None]":
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        table_name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, DataFrame)
            else f"temp_{token_hex(5)}"
        )

        table = None

        if record_type is not None:
            table = self._get_table(record_type)
        elif isinstance(data, DataFrame):
            table = sqla.Table(
                table_name,
                self.meta,
                *self._cols_from_df(data, foreign_keys=foreign_keys).values(),
            )
        elif isinstance(data, Record):
            table = self._get_table(data)
        elif has_type(data, Iterable[Record]):
            table = self._get_table(next(data.__iter__()))
        elif has_type(data, Mapping[Any, Record]):
            table = self._get_table(next(data.values().__iter__()))
        elif isinstance(data, sqla.Select):
            table = sqla.Table(
                table_name,
                self.meta,
                *(
                    sqla.Column(name, col.type, primary_key=col.primary_key)
                    for name, col in data.selected_columns.items()
                ),
            )
        else:
            raise TypeError("Unsupported type given as value")

        table = self._ensure_table_exists(table)

        ds = DataSet(self, table, selection=record_type)
        ds[:] = data
        return ds

    def transfer(self, backend: Backend[Name2]) -> "DataBase[Name2]":
        """Transfer the DB to a different backend."""
        other_db = DataBase(
            backend,
            self.schema,
            create_cross_fk=self.create_cross_fk,
            validate_on_init=False,
        )

        for rec in self.record_types:
            ds = self[rec]
            ds.load(kind=pl.DataFrame).write_database(
                str(ds.base), str(other_db.backend.url)
            )

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
            n.load(kind=pd.DataFrame)
            .reset_index()
            .assign(table=n.selection._default_table_name())
            for n in node_tables
            if isinstance(n.selection, type)
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
                if len(at._rels()) == 2:
                    left, right = (r for r in at._rels())
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
                    .load(kind=pd.DataFrame)
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
                                *(
                                    a.name
                                    for a in (left_rel._record_type or Record)._attrs()
                                ),
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

    def _cols_from_df(
        self, df: DataFrame, foreign_keys: Mapping[str, Attr] | None = None
    ) -> dict[str, sqla.Column]:
        """Create columns from DataFrame."""
        if isinstance(df, pd.DataFrame) and len(df.index.names) > 1:
            raise NotImplementedError("Multi-index not supported yet.")

        fks = foreign_keys or {}

        def gen_fk(col: str) -> list[sqla.ForeignKey]:
            return (
                [sqla.ForeignKey(self._get_table(fks[col].rec_type).c[fks[col].name])]
                if col in fks
                else []
            )

        return {
            **(
                {
                    level: sqla.Column(
                        level,
                        map_df_dtype(df.index.get_level_values(level).to_series()),
                        *gen_fk(level),
                        primary_key=True,
                    )
                    for level in df.index.names
                }
                if isinstance(df, pd.DataFrame)
                else {}
            ),
            **{
                str(df[col].name): sqla.Column(
                    str(df[col].name), map_df_dtype(df[col]), *gen_fk(col)
                )
                for col in df.columns
            },
        }

    @cache
    def _get_table(self, rec: type[Record]) -> sqla.Table:
        return rec._table(self.meta, self.subs)

    @cache
    def _get_joined_table(self, rec: type[Record]) -> sqla.Table | sqla.Join:
        return rec._joined_table(self.meta, self.subs)

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
                ).write_database(str(self._get_table(rec)), str(self.engine.url))

    def _save_to_excel(
        self, record_types: Iterable[type[Record]] | None = None
    ) -> None:
        """Save all (or selected) tables to Excel."""
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
                    f"SELECT * FROM {self._get_table(rec)}",
                    self.engine,
                ).write_excel(wb, worksheet=rec._default_table_name())

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)

    def _delete_from_excel(self, record_types: Iterable[type[Record]]) -> None:
        """Delete selected table from Excel."""
        assert self.backend.type == "excel-file", "Backend must be an Excel file."
        assert isinstance(self.backend.url, Path | CloudPath | HttpFile)

        file = (
            BytesIO()
            if isinstance(self.backend.url, HttpFile)
            else self.backend.url.open("wb")
        )

        wb = openpyxl.load_workbook(file)
        for rec in record_types or self.record_types:
            del wb[rec._default_table_name()]

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)


Union: TypeAlias = sqla.FromClause
Join: TypeAlias = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


@dataclass(frozen=True)
class DataSet(Generic[Name, Rec_cov, Idx_cov]):
    """Dataset selection."""

    db: DataBase[Name]
    base: sqla.FromClause
    selection: type[Rec_cov] | Prop[Rec_cov, Any] | None = None
    merges: RelMerge = RelMerge()
    filters: list[sqla.ColumnElement[bool]] = field(default_factory=list)
    keys: Sequence[slice | list[Hashable] | Hashable | sqla.ColumnElement] = field(
        default_factory=list
    )

    @cached_property
    def record_type(self) -> type[Rec_cov] | None:
        """Record type of this dataset."""
        return cast(
            type[Rec_cov] | None,
            (
                self.selection.record_type
                if isinstance(self.selection, Prop)
                else self.selection if isinstance(self.selection, type) else None
            ),
        )

    @cached_property
    def base_set(self) -> "DataSet[Name, Rec_cov, Idx_cov]":
        """Base dataset for this dataset."""
        return DataSet(self.db, self.base)

    @cached_property
    def main_idx(self) -> set[Attr[Rec_cov, Any]] | None:
        """Main index of this dataset."""
        match self.selection:
            case type():
                return self.selection._indexes()
            case Rel():
                return self.selection.target_type._indexes()
            case Prop():
                return self.selection.record_type._indexes()
            case None:
                return None

    @cached_property
    def path_idx(self) -> list[Attr] | None:
        """Optional, alternative index of this dataset based on merge path."""
        path = self.selection.path if isinstance(self.selection, Prop) else []

        if any(rel.index_by is None for rel in path):
            return None

        return [rel.index_by for rel in path if rel.index_by is not None]

    # Overloads: attribute selection:

    # 1. Top-level attribute selection, relational index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Idx]",
        key: Attr[Rec_cov, Val],
    ) -> "DataSet[Name, Record[Val, Key2], Idx]": ...

    # 2. Nested attribute selection, base index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], BaseIdx]", key: Attr[Any, Val]
    ) -> "DataSet[Name, Record[Val, Any], IdxStart[Key2]]": ...

    # 3. Nested attribute selection, relational index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]", key: Attr[Any, Val]
    ) -> "DataSet[Name, Record[Val, Any], IdxStart[Key | Key2]]": ...

    # Overloads: relation selection:

    # 4. Top-level relation selection, singular
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: Rel[Rec_cov, Rec2, Rec2],
    ) -> "DataSet[Name, Rec2, Key | Key2]": ...

    # 5. Top-level relation selection, indexed, tuple case
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup] | BaseIdx]",
        key: Rel[Rec_cov, Mapping[Key3, Rec2], Rec2],
    ) -> "DataSet[Name, Rec2, tuple[*IdxTup, Key3] | tuple[Key2, Key3]]": ...

    # 6. Top-level relation selection, indexed
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]",
        key: Rel[Rec_cov, Mapping[Key3, Rec2], Rec2],
    ) -> "DataSet[Name, Rec2, tuple[Key | Key2, Key3]]": ...

    # 7. Top-level relation selection, no index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: Rel[Rec_cov, Iterable, Rec2],
    ) -> "DataSet[Name, Rec2, BaseIdx]": ...

    # 8. Nested relation selection, tuple case
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup]]",
        key: Rel[Any, Mapping[Key3, Rec2], Rec2],
    ) -> (
        "DataSet[Name, Rec2, IdxTupStartEnd[*IdxTup, Key3] | IdxStartEnd[Key2, Key3]]"
    ): ...

    # 9. Nested relation selection
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]",
        key: Rel[Any, Mapping[Key3, Rec2], Rec2],
    ) -> "DataSet[Name, Rec2, IdxStartEnd[Key | Key2, Key3]]": ...

    # 10. List selection, keep index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Key2], Key]", key: list[Key | Key2]
    ) -> "DataSet[Name, Rec_cov, Key]": ...

    # 11. Index value selection
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Key2], Key]", key: Key | Key2
    ) -> "DataSet[Name, Rec_cov, SingleIdx]": ...

    # 12. Slice selection, keep index
    @overload
    def __getitem__(self, key: slice | tuple[slice, ...]) -> Self: ...

    # 13. Expression filtering, keep index
    @overload
    def __getitem__(self, key: sqla.ColumnElement[bool] | pd.Series[bool]) -> Self: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self,
        key: (
            Attr[Any, Val]
            | Rel
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice, ...]
            | sqla.ColumnElement[bool]
            | Series
        ),
    ) -> "DataSet":
        filt, join = (
            self._parse_filter(key)
            if isinstance(key, sqla.ColumnElement | pd.Series)
            else (None, None)
        )

        keys = None
        if isinstance(key, tuple):
            # Selection by tuple of index values.
            keys = list(key)
        elif isinstance(key, list | slice) and not isinstance(
            key, sqla.ColumnElement | pd.Series
        ):
            # Selection by index value list, slice or single value.
            keys = [key]

        return DataSet(
            self.db,
            self.base.join(*join) if isinstance(join, tuple) else self.base,
            (
                self.selection >> key
                if isinstance(self.selection, type | Rel) and isinstance(key, Prop)
                else self.selection
            ),
            self.merges + join if isinstance(join, RelMerge) else self.merges,
            self.filters + [filt] if filt is not None else self.filters,
            keys if keys is not None else self.keys,
        )

    @overload
    def select(self, merge: None = ...) -> sqla.Select[tuple[Rec_cov]]: ...

    @overload
    def select(self, merge: Rel[Any, Any, Rec]) -> sqla.Select[tuple[Rec_cov, Rec]]: ...

    @overload
    def select(
        self, merge: RelMerge[*MergeTup]
    ) -> sqla.Select[tuple[Rec_cov, *MergeTup]]: ...

    def select(
        self,
        merge: Rel | RelMerge | None = None,
        index_only: bool = False,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        abs_merge = self._parse_merge(merge)

        selection_table = (
            self._get_alias(self.selection)
            if isinstance(self.selection, Rel)
            else (
                self._get_alias(self.selection.parent)
                if isinstance(self.selection, Prop)
                and self.selection.parent is not None
                else (
                    self.db._get_joined_table(self.record_type)
                    if self.record_type is not None
                    else self.base
                )
            )
        )

        select = sqla.select(
            *(
                col
                for col in selection_table.columns
                if not index_only or col.primary_key
            ),
            *(
                col.label(f"{rel.path_str}.{col_name}")
                for rel in abs_merge.rels
                for col_name, col in self._get_alias(rel).columns.items()
                if not index_only or col.primary_key
            ),
        ).select_from(self.base)

        for join in self._get_joins():
            select = select.join(*join)

        for filt in self.filters:
            select = select.where(filt)

        return select

    @overload
    def load(
        self: "DataSet[Name, Rec_cov, SingleIdx]",
        merge: None = ...,
        kind: type[Record] = ...,
    ) -> Rec_cov: ...

    @overload
    def load(
        self: "DataSet[Name, Rec_cov, SingleIdx]",
        merge: Rel[Any, Any, Rec],
        kind: type[Record] = ...,
    ) -> tuple[Rec_cov, Rec]: ...

    @overload
    def load(
        self: "DataSet[Name, Rec_cov, SingleIdx]",
        merge: RelMerge[*MergeTup],
        kind: type[Record] = ...,
    ) -> tuple[Rec_cov, *MergeTup]: ...

    @overload
    def load(self, merge: None = ..., kind: type[Df] = ...) -> Df: ...  # type: ignore

    @overload
    def load(self, merge: Rel | RelMerge, kind: type[Df]) -> tuple[Df, ...]: ...

    @overload
    def load(
        self, merge: None = ..., kind: type[Record] = ...
    ) -> Sequence[Rec_cov]: ...

    @overload
    def load(
        self, merge: Rel[Any, Any, Rec], kind: type[Record] = ...
    ) -> Sequence[tuple[Rec_cov, Rec]]: ...

    @overload
    def load(
        self, merge: RelMerge[*MergeTup], kind: type[Record] = ...
    ) -> Sequence[tuple[Rec_cov, *MergeTup]]: ...

    def load(
        self,
        merge: Rel | RelMerge | None = None,
        kind: type[Record | Df] = Record,
    ) -> (
        Rec_cov
        | tuple[Rec_cov, *tuple[Any, ...]]
        | Df
        | tuple[Df, ...]
        | Sequence[Rec_cov | tuple[Rec_cov, *tuple[Any, ...]]]
    ):
        """Download table selection."""
        if issubclass(kind, Record):
            raise NotImplementedError("Downloading record instances not supported yet.")

        abs_merge = self._parse_merge(merge)
        select = self.select(merge)

        df = None
        if kind is pl.DataFrame:
            df = pl.read_database(select, self.db.engine)
        else:
            with self.db.engine.connect() as con:
                df = pd.read_sql(select, con)

        return cast(
            tuple[Df, ...],
            tuple(
                df[list(col for col in df.columns if col.startswith(relref.path_str))]
                for relref in abs_merge.rels
            ),
        )

    # 1. Top-level attribute assignment
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Key2], Key | BaseIdx]",
        key: Attr[Rec_cov, Val],
        value: "DataSet[Name, Record[Val, Key | Key2], BaseIdx] | DataSet[Name, Record[Val, Any], Key | Key2] | ValInput[Val, Key | Key2]",  # noqa: E501
    ) -> None: ...

    # 2. Nested attribute assignment
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Key2], Key | BaseIdx]",
        key: Attr[Any, Val],
        value: "DataSet[Name, Record[Val, IdxStart[Key | Key2]], BaseIdx] | DataSet[Name, Record[Val, Any], IdxStart[Key | Key2]] | ValInput[Val, IdxStart[Key | Key2]]",  # noqa: E501
    ) -> None: ...

    # 3. Top-level relation assignment, singular
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]",
        key: Rel[Rec_cov, Rec2, Rec2],
        value: "DataSet[Name, Rec2, Key | Key2] | RecInput[Rec2, Key | Key2]",
    ) -> None: ...

    # 4. Top-level relation assignment, indexed, tuple case
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup] | BaseIdx]",
        key: Rel[Rec_cov, Mapping[Key3, Rec2], Rec2],
        value: "DataSet[Name, Rec2, tuple[*IdxTup, Key3] | tuple[Key2, Key3]] | RecInput[Rec2, tuple[*IdxTup, Key3] | tuple[Key2, Key3]]",  # noqa: E501
    ) -> None: ...

    # 5. Top-level relation assignment, indexed
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: Rel[Rec_cov, Mapping[Key3, Rec2], Rec2],
        value: "DataSet[Name, Rec2, tuple[Key | Key2, Key3]] | RecInput[Rec2, tuple[Key | Key2, Key3]]",  # noqa: E501
    ) -> None: ...

    # 6. Top-level relation assignment, no index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Any]",
        key: Rel[Rec_cov, Iterable[Rec2], Rec2],
        value: "DataSet[Name, Rec2, BaseIdx] | RecInput[Rec2, Any]",
    ) -> None: ...

    # 7. Nested relation assignment, tuple case
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup] | BaseIdx]",
        key: Rel[Any, Mapping[Key3, Rec2], Rec2],
        value: "DataSet[Name, Rec2, IdxTupStartEnd[*IdxTup, Key3] | IdxStartEnd[Key2, Key3]] | RecInput[Rec2, IdxTupStartEnd[*IdxTup, Key3] | IdxStartEnd[Key2, Key3]]",  # noqa: E501
    ) -> None: ...

    # 8. Nested relation assignment
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: Rel[Any, Mapping[Key3, Rec2], Rec2],
        value: "DataSet[Name, Rec2, IdxStartEnd[Key | Key2, Key3]] | RecInput[Rec2, IdxStartEnd[Key | Key2, Key3]]",  # noqa: E501
    ) -> None: ...

    # 9. List assignment with base index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], BaseIdx]",
        key: list[Key2],
        value: "DataSet[Name, Rec_cov, Key2 | BaseIdx] | Iterable[Rec_cov] | DataFrame",  # noqa: E501
    ) -> None: ...

    # 10. List assignment with relational index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]",
        key: list[Key] | list[Key2],
        value: "DataSet[Name, Rec_cov, Key | Key2 | BaseIdx] | Iterable[Rec_cov] | DataFrame",  # noqa: E501
    ) -> None: ...

    # 11. Single value assignment with base index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Key2], BaseIdx]",
        key: Key2,
        value: Rec_cov | Val,
    ) -> None: ...

    # 12. Single value assignment with relational index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Key2], Key]",
        key: Key | Key2,
        value: Rec_cov | Val,
    ) -> None: ...

    # 13. Slice assignment
    @overload
    def __setitem__(
        self,
        key: slice | tuple[slice, ...],
        value: Self | Iterable[Rec_cov] | DataFrame,
    ) -> None: ...

    # 14. Filter assignment with base index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], BaseIdx]",
        key: sqla.ColumnElement[bool] | pd.Series[bool],
        value: "DataSet[Name, Rec_cov, Key2] | RecInput[Rec_cov, Key2]",
    ) -> None: ...

    # 15. Filter assignment with relational index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]",
        key: sqla.ColumnElement[bool] | pd.Series[bool],
        value: "DataSet[Name, Rec_cov, Key | Key2] | RecInput[Rec_cov, Key | Key2]",
    ) -> None: ...

    def __setitem__(  # noqa: D105
        self,
        key: (
            Attr
            | Rel
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice | tuple[slice, ...], ...]
            | sqla.ColumnElement[bool]
            | pd.Series[bool]
        ),
        value: "DataSet | RecInput[Rec_cov, Any] | ValInput",
    ) -> None:
        if isinstance(key, list):
            # Ensure that index is alignable.

            if isinstance(value, Sized):
                assert len(value) == len(key), "Length mismatch."

            if isinstance(value, pd.DataFrame):
                assert set(value.index.to_list()) == set(key), "Index mismatch."
            elif isinstance(value, Mapping):
                assert set(value.keys()) == set(key), "Index mismatch."

        cast(Self, self[key])._set(value)  # type: ignore

    def __ilshift__(
        self: "DataSet[Name, Record[Any, Key2], BaseIdx]",
        value: "DataSet[Name, Rec_cov, Key2 | BaseIdx] | RecInput[Rec_cov, Key2]",
    ) -> None:
        """Merge update into unfiltered dataset."""
        self._set(value, insert=True)

    def __or__(
        self, other: "DataSet[Name, Rec_cov, Idx]"
    ) -> "DataSet[Name, Rec_cov, Idx_cov | Idx]":
        """Union two datasets."""
        return DataSet(self.db, sqla.union(self.select(), other.select()).subquery())

    def extract(
        self,
        use_schema: bool | type[Schema] = False,
        aggs: Mapping[Rel, Agg] | None = None,
    ) -> DataBase[Name]:
        """Extract a new database instance from the current selection."""
        assert self.record_type is not None, "Record type must be defined."

        # Get all rec types in the schema.
        schemas = (
            {use_schema}
            if isinstance(use_schema, type)
            else (
                (
                    set(self.db.schema.keys())
                    if isinstance(self.db.schema, Mapping)
                    else {self.db.schema} if self.db.schema is not None else set()
                )
                if use_schema
                else set()
            )
        )
        rec_types = (
            {rec for schema in schemas for rec in schema._record_types}
            if schemas
            else ({self.record_type, *self.record_type._rel_types()})
        )

        # Get the entire subdag from this selection.
        all_paths_rels = {
            r for rel in self.record_type._rels() for r in rel.get_subdag(rec_types)
        }

        # Extract rel paths, which contain an aggregated rel.
        aggs_per_type: dict[type[Record], list[tuple[Rel, Agg]]] = {}
        if aggs is not None:
            for rel, agg in aggs.items():
                for path_rel in all_paths_rels:
                    if rel in path_rel.path:
                        aggs_per_type[rel.record_type] = [
                            *aggs_per_type.get(rel.record_type, []),
                            (rel, agg),
                        ]
                        all_paths_rels.remove(path_rel)

        replacements: dict[type[Record], sqla.Select] = {}
        for rec in rec_types:
            # For each table, create a union of all results from the direct routes.
            selects = [
                self[rel].select()
                for rel in all_paths_rels
                if issubclass(rec, rel.target_type)
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
                                if isinstance(sa, Attr)
                                else sqla_visitors.replacement_traverse(
                                    sa,
                                    {},
                                    replace=lambda element, **kw: (
                                        src_select.c[element.name]
                                        if isinstance(element, Attr)
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
        new_db = DataBase(self.db.backend, overlay=f"temp_{token_hex(10)}")

        # Overlay the new tables onto the new database.
        for rec in rec_types:
            if rec in replacements:
                new_db[rec] = replacements[rec]
            else:
                new_db[rec] = pd.DataFrame()  # Empty table.

        for rec, agg_select in aggregations.items():
            new_db[rec] = agg_select

        return new_db

    def __clause_element__(self) -> sqla.Subquery:
        """Return subquery for the current selection to be used inside SQL clauses."""
        return self.select().subquery()

    def _get_alias(self, rel: Rel) -> sqla.FromClause:
        """Get alias for a relation reference."""
        return self.db._get_joined_table(rel.target_type).alias(gen_str_hash(rel, 8))

    def _get_random_alias(self, rec: type[Record]) -> sqla.FromClause:
        """Get random alias for a type."""
        return self.db._get_joined_table(rec).alias(token_hex(4))

    def _get_joins(self, _subtree: RelTree | None = None) -> list[Join]:
        """Extract join operations from the relation tree."""
        _subtree = _subtree or self.merges.rel_tree
        joins = []

        for rel, next_subtree in _subtree.items():
            parent = (
                self._get_alias(rel.parent) if rel.parent is not None else self.base
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
                                    temp_alias_map[lk.record_type].c[lk.name]
                                    == target_table.c[rk.name]
                                    for lk, rk in join_on.items()
                                ),
                            )
                            for join_on in rel.joins
                        ),
                    ),
                )
            )

            joins.extend(self._get_joins(next_subtree))

        return joins

    def _parse_merge(self, merge: Rel | RelMerge | None) -> RelMerge:
        """Parse merge argument and prefix with current selection."""
        merge = (
            merge
            if isinstance(merge, RelMerge)
            else RelMerge({merge}) if merge is not None else RelMerge()
        )
        return (
            self.selection >> merge
            if isinstance(self.selection, Rel)
            else (
                self.selection.path[-1] >> merge
                if isinstance(self.selection, Prop)
                else merge
            )
        )

    def _gen_idx_match_expr(
        self,
        values: Sequence[slice | list[Hashable] | Hashable | sqla.ColumnElement],
    ) -> sqla.ColumnElement[bool] | None:
        if values == slice(None):
            return None

        idx_attrs = (
            self.main_idx
            if self.main_idx is not None and len(values) == len(self.main_idx)
            else (
                self.path_idx
                if self.path_idx is not None and len(values) == len(self.path_idx)
                else []
            )
        )

        exprs = [
            (
                idx.in_(val)
                if isinstance(val, list)
                else (
                    idx.between(val.start, val.stop)
                    if isinstance(val, slice)
                    else idx == val
                )
            )
            for idx, val in zip(idx_attrs, values)
        ]

        if len(exprs) == 0:
            return None

        return reduce(sqla.and_, exprs)

    def _replace_attr(
        self,
        element: sqla_visitors.ExternallyTraversible,
        reflist: set[Rel] = set(),
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if isinstance(element, Attr):
            if element.parent is not None:
                reflist.add(element.parent)

            if isinstance(self.selection, Rel):
                element = self.selection >> element
            elif isinstance(self.selection, Prop) and self.selection.parent is not None:
                element = self.selection.parent >> element

            table = (
                self._get_alias(element.parent)
                if element.parent is not None
                else self.db._get_table(element.record_type)
            )
            return table.c[element.name]

        return None

    def _parse_filter(
        self,
        key: sqla.ColumnElement[bool] | pd.Series,
    ) -> tuple[sqla.ColumnElement[bool], RelMerge | Join]:
        filt = None
        join = None
        merge = RelMerge()

        match key:
            case sqla.ColumnElement():
                # Filtering via SQL expression.
                reflist = set()
                replace_func = partial(self._replace_attr, reflist=reflist)
                filt = sqla_visitors.replacement_traverse(key, {}, replace=replace_func)
                merge += RelMerge(reflist)
            case pd.Series():
                # Filtering via boolean Series.

                series_name = token_hex(8)
                uploaded = self.db.dataset(key.rename(series_name).to_frame())
                assert all(
                    n1 == n2 for n1, n2 in zip(key.index.names, self.base.primary_key)
                )

                filt = uploaded.base.c[series_name] == True  # noqa: E712
                join = (
                    uploaded.base,
                    reduce(
                        sqla.and_,
                        (
                            self.base.c[pk] == uploaded.base.c[pk]
                            for pk in key.index.names
                        ),
                    ),
                )

        return filt, join or merge

    def _set(  # noqa: D105
        self,
        value: "DataSet | RecInput[Rec_cov, Any] | ValInput",
        insert: bool = False,
    ) -> None:
        if (
            has_type(value, Record)
            or has_type(value, Iterable[Record])
            or has_type(value, Mapping[Any, Record])
        ):
            raise NotImplementedError(
                "Inserting / updating via record instances not supported yet."
            )

        # Load data into a temporary table, if vectorized.
        upload_df = (
            value
            if isinstance(value, DataFrame)
            else (
                (value if isinstance(value, pd.Series) else pd.Series(dict(value)))
                .rename("_value")
                .to_frame()
                if isinstance(value, Mapping | pd.Series)
                else None
            )
        )
        value_set = (
            self.db.dataset(upload_df)
            if upload_df is not None
            else (
                self.db.dataset(value)
                if isinstance(value, sqla.Select)
                else value if isinstance(value, DataSet) else None
            )
        )

        # Construct the insert / update statement.
        if insert:
            assert (
                isinstance(self.selection, Record)
                and isinstance(self.base, sqla.Table)
                and len(self.filters) == 0
            ), "Can only upsert into unfiltered base datasets."
            assert value_set is not None

            # Do an insert-from-select operation, which updates on conflict:
            statement = self.base.insert().from_select(
                [col.name for col in self.base.columns],
                value_set.select().subquery(),
            )

            if isinstance(statement, postgresql.Insert):
                # For Postgres / DuckDB, use: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#updating-using-the-excluded-insert-values
                statement = statement.on_conflict_do_update(
                    index_elements=[col.name for col in self.base.primary_key.columns],
                    set_=dict(statement.excluded),
                )
            elif isinstance(statement, mysql.Insert):
                # For MySQL / MariaDB, use: https://docs.sqlalchemy.org/en/20/dialects/mysql.html#insert-on-duplicate-key-update-upsert
                statement = statement.prefix_with("INSERT INTO")
                statement = statement.on_duplicate_key_update(**statement.inserted)
            else:
                # For others, use CTE: https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.cte
                raise NotImplementedError(
                    "Upsert not supported for this database dialect."
                )
        else:
            assert isinstance(self.base, sqla.Table), "Base must be a table."

            # Derive current select statement and join with value table, if exists.
            select = self.select()
            if value_set is not None:
                select = select.join(
                    value_set.base,
                    reduce(
                        sqla.and_,
                        (
                            idx_col == self.base.c[idx_col.name]
                            for idx_col in value_set.base.primary_key
                        ),
                    ),
                )

            selected_attr = (
                self.selection.name if isinstance(self.selection, Attr) else None
            )

            # Prepare update statement.
            if self.db.engine.dialect.name in (
                "postgres",
                "postgresql",
                "duckdb",
                "mysql",
                "mariadb",
            ):
                # Update-from.
                statement = (
                    self.base.update()
                    .values(
                        {col.name: col for col in value_set.base.columns}
                        if value_set is not None
                        else {selected_attr: value}
                    )
                    .where(
                        reduce(
                            sqla.and_,
                            (
                                self.base.c[col.name] == select.c[col.name]
                                for col in self.base.primary_key.columns
                            ),
                        )
                    )
                )
            else:
                # Correlated update.
                raise NotImplementedError("Correlated update not supported yet.")

        # Execute insert / update statement.
        with self.db.engine.begin() as con:
            con.execute(statement)

        # Drop the temporary table, if any.
        if value_set is not None:
            cast(sqla.Table, value_set.base).drop(self.db.engine)
