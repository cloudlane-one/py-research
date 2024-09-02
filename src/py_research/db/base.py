"""Base class definitions for universal database interface."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence, Sized
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from functools import cached_property, partial, reduce
from io import BytesIO
from pathlib import Path
from secrets import token_hex
from typing import Any, Generic, Literal, LiteralString, Self, cast, overload

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
    is_bool_dtype,
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from typing_extensions import TypeVar, TypeVarTuple
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.files import HttpFile
from py_research.hashing import gen_str_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.types import has_type

from .schema import (
    Agg,
    Attr,
    AttrRef,
    BaseIdx,
    DynRecord,
    Key,
    PropRef,
    Rec,
    Rec2,
    Rec_cov,
    Record,
    RecordValue,
    Rel,
    RelDict,
    RelRef,
    RelTree,
    RelTup,
    Require,
    Scalar,
    Schema,
    SingleIdx,
    TypeDef,
    Unloaded,
    Val,
    Val_cov,
    dynamic_record_type,
)

DataFrame = pd.DataFrame | pl.DataFrame
Series = pd.Series | pl.Series


Name = TypeVar("Name", bound=LiteralString)
Name2 = TypeVar("Name2", bound=LiteralString)

RecMerge = TypeVar("RecMerge", covariant=True, bound=tuple | None, default=None)

Idx = TypeVar("Idx", bound="Hashable | BaseIdx")
Idx_cov = TypeVar("Idx_cov", covariant=True, bound=Hashable | BaseIdx | SingleIdx)
Key2 = TypeVar("Key2", bound=Hashable)
Key3 = TypeVar("Key3", bound=Hashable)
Key4 = TypeVar("Key4", bound=Hashable)
IdxTup = TypeVarTuple("IdxTup")

Df = TypeVar("Df", bound=DataFrame)
Dl = TypeVar("Dl", bound=DataFrame | Record)


type IdxStart[Key: Hashable] = Key | tuple[Key, *tuple[Any, ...]]
type IdxEnd[Key: Hashable] = tuple[*tuple[Any, ...], Key]
type IdxStartEnd[Key: Hashable, Key2: Hashable] = tuple[Key, *tuple[Any, ...], Key2]
type IdxTupStartEnd[*IdxTup, Key2: Hashable] = tuple[*IdxTup, *tuple[Any], Key2]

type RecInput[Rec: Record, Key: Hashable] = DataFrame | Iterable[Rec] | Mapping[
    Key, Rec
] | sqla.Select[tuple[Key, Rec]] | Rec

type PartialRec[Rec: Record] = Mapping[PropRef[Rec, Any], Any]

type PartialRecInput[Rec: Record, Key: Hashable] = RecInput[Rec, Any] | Iterable[
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
    data: DataFrame | sqla.Select, foreign_keys: Mapping[str, AttrRef] | None = None
) -> list[PropRef]:
    """Extract prop definitions from dataframe or query."""
    if isinstance(data, pd.DataFrame) and len(data.index.names) > 1:
        raise NotImplementedError("Multi-index not supported yet.")

    foreign_keys = foreign_keys or {}

    def _gen_prop(name: str, data: Series | sqla.ColumnElement) -> PropRef:
        is_rel = name in foreign_keys
        value_type = (
            map_df_dtype(data) if isinstance(data, Series) else map_col_dtype(data)
        ) or Any
        attr = AttrRef(
            primary_key=True,
            _name=name if not is_rel else f"fk_{name}",
            prop_type=TypeDef(Attr[value_type]),
            record_type=DynRecord,
        )
        return (
            attr
            if not is_rel
            else RelRef(
                on={attr: foreign_keys[name]},
                prop_type=TypeDef(Rel[DynRecord]),
                record_type=DynRecord,
            )
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


@dataclass
class DataBase(Generic[Name]):
    """Active connection to a (in-memory) SQL server."""

    backend: Backend[Name] = field(default_factory=lambda: Backend("default"))  # type: ignore
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
    overlay_with_schemas: bool = True

    overlay: str | None = None
    _overlay_subs: dict[type[Record], sqla.TableClause] = field(default_factory=dict)

    def __post_init__(self):  # noqa: D105
        if self.validate_on_init:
            self.validate()

        if self.overlay is not None and self.overlay_with_schemas:
            self._ensure_schema_exists(self.overlay)

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
                asdict(self.backend) if self.backend.type != "in-memory" else None
            ),
        }

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
            pks = set([attr.name for attr in rec._primary_keys.values()])
            fks = set(
                [attr.name for rel in rec._rels.values() for attr in rel.fk_map.keys()]
            )
            if pks in fks:
                assoc_types.add(rec)

        return assoc_types

    @cached_property
    def relation_map(self) -> dict[type[Record], set[RelRef]]:
        """Maps all tables in this DB to their outgoing or incoming relations."""
        rels: dict[type[Record], set[RelRef]] = {
            table: set() for table in self.record_types
        }

        for rec in self.record_types:
            for rel in rec._rels.values():
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
            else (
                sqla.create_engine("duckdb:///:memory:")
                if self.backend.name == "default"
                else sqla.create_engine("sqlite:///:memory:")
            )
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

    def _parse_schema_items(
        self,
        element: sqla_visitors.ExternallyTraversible,
        **kw: Any,
    ) -> sqla.ColumnElement | sqla.FromClause | None:
        if isinstance(element, RelRef):
            return self._get_alias(element)
        elif isinstance(element, AttrRef):
            table = (
                self._get_alias(element.parent)
                if element.parent is not None
                else self._get_table(element.record_type)
            )
            return table.c[element.name]
        elif has_type(element, type[Record]):
            return self._get_table(element)

        return None

    def _parse_expr[
        CE: sqla.ClauseElement
    ](self, expr: CE,) -> CE:
        """Parse an expression in this database's context."""
        return cast(
            CE,
            sqla_visitors.replacement_traverse(
                expr, {}, replace=self._parse_schema_items
            ),
        )

    def execute[
        *T
    ](
        self, stmt: sqla.Select[tuple[*T]] | sqla.Insert | sqla.Update | sqla.Delete
    ) -> sqla.Result[tuple[*T]]:
        """Execute a SQL statement in this database's context."""
        stmt = self._parse_expr(stmt)
        with self.engine.begin() as conn:
            return conn.execute(self._parse_expr(stmt))

    def __getitem__(self, key: type[Rec]) -> DataSet[Name, Rec, BaseIdx]:
        """Return the dataset for given record type."""
        if self.backend.type == "excel-file":
            self._load_from_excel([key])

        return DataSet(self, key)

    def __setitem__(  # noqa: D105
        self,
        key: type[Rec],
        value: (
            DataSet[Name, Rec, Key | BaseIdx | None]
            | RecInput[Rec, Key]
            | sqla.Select[tuple[Rec]]
        ),
    ) -> None:
        if self.backend.type == "excel-file":
            self._load_from_excel([key])

        table = self._get_table(key, writable=True)
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
        table = self._get_table(key, writable=True)
        with self.engine.begin() as con:
            table.drop(con)

        if self.backend.type == "excel-file":
            self._delete_from_excel([key])

    def __or__(self, other: DataBase[Name]) -> DataBase[Name]:
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
        data: DataFrame | sqla.Select,
        foreign_keys: Mapping[str, AttrRef] | None = None,
    ) -> DataSet[Name, Record, None]:
        """Create a temporary dataset instance from a DataFrame or SQL query."""
        name = (
            f"temp_df_{gen_str_hash(data, 10)}"
            if isinstance(data, DataFrame)
            else f"temp_{token_hex(5)}"
        )

        rec = dynamic_record_type(name, props=props_from_data(data, foreign_keys))
        self._get_table(rec, writable=True)
        ds = DataSet(self, rec)
        ds[:] = data

        return ds  # type: ignore

    def copy(
        self, backend: Backend[Name2] = Backend("main"), overlay: str | bool = True
    ) -> DataBase[Name2]:
        """Transfer the DB to a different backend (defaults to in-memory)."""
        other_db = DataBase(
            backend,
            self.schema,
            create_cross_fk=self.create_cross_fk,
            overlay=(
                (self.overlay or token_hex())
                if overlay is True
                else overlay if isinstance(overlay, str) else None
            ),
            overlay_with_schemas=self.overlay_with_schemas,
            validate_on_init=False,
        )

        for rec in self.record_types:
            self[rec].load(kind=pl.DataFrame).write_database(
                str(self._get_table(rec)), str(other_db.backend.url)
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
            .assign(table=n.base._default_table_name())
            for n in node_tables
            if isinstance(n.base, type)
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )

        directed_edges = reduce(set.union, (self.relation_map[n] for n in nodes))

        undirected_edges: dict[type[Record], set[tuple[RelRef, ...]]] = {
            t: set() for t in nodes
        }
        for n in nodes:
            for at in self.assoc_types:
                if len(at._rels) == 2:
                    left, right = (r for r in at._rels.values())
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
                        == str((rel.record_type or Record)._default_table_name())
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
                                    for a in (left_rel.record_type or Record)._attrs
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

    def _get_table(self, rec: type[Record], writable: bool = False) -> sqla.Table:
        if writable and self.overlay is not None and rec not in self._overlay_subs:
            # Create an empty overlay table for the record type
            self._overlay_subs[rec] = sqla.table(
                (
                    (self.overlay + "_" + rec._default_table_name())
                    if self.overlay_with_schemas
                    else rec._default_table_name()
                ),
                schema=self.overlay if self.overlay_with_schemas else None,
            )

        table = rec._table(self.meta, self.subs)

        # Create any missing tables in the database.
        self.meta.create_all(self.engine)

        return table

    def _get_joined_table(self, rec: type[Record]) -> sqla.Table | sqla.Join:
        table = rec._joined_table(self.meta, self.subs)

        # Create any missing tables in the database.
        self.meta.create_all(self.engine)

        return table

    def _get_alias(self, rel: RelRef) -> sqla.FromClause:
        """Get alias for a relation reference."""
        return self._get_joined_table(rel.target_type).alias(gen_str_hash(rel, 8))

    def _get_random_alias(self, rec: type[Record]) -> sqla.FromClause:
        """Get random alias for a type."""
        return self._get_joined_table(rec).alias(token_hex(4))

    def _ensure_schema_exists(self, schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_schema(schema_name):
            with self.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name

    def _table_exists(self, sqla_table: sqla.Table) -> bool:
        """Check if a table exists in the database."""
        return sqla.inspect(self.engine).has_table(
            sqla_table.name, schema=sqla_table.schema
        )

    def _create_sqla_table(self, sqla_table: sqla.Table) -> None:
        """Create SQL-side table from Table class."""
        if not self.create_cross_fk:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            sqla_table = sqla_table.to_metadata(sqla.MetaData())  # temporary metadata
            _remove_external_fk(sqla_table)

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


type Join = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


@dataclass(frozen=True)
class DataSet(Generic[Name, Rec_cov, Idx_cov, RecMerge]):
    """Dataset attached to a database."""

    db: DataBase[Name]
    base: type[Rec_cov] | PropRef[Rec_cov, Any] | set[DataSet[Name, Rec_cov, Idx_cov]]
    extensions: RelTree = field(default_factory=RelTree)
    filters: list[sqla.ColumnElement[bool]] = field(default_factory=list)
    keys: Sequence[slice | list[Hashable] | Hashable | sqla.ColumnElement] = field(
        default_factory=list
    )

    @cached_property
    def record_type(self) -> type[Rec_cov]:
        """Record type of this dataset."""
        return cast(
            type[Rec_cov],
            (
                self.base.record_type
                if isinstance(self.base, PropRef)
                else (
                    self.base
                    if isinstance(self.base, type)
                    else [r.record_type for r in self.base][0]
                )
            ),
        )

    @cached_property
    def main_idx(self) -> set[AttrRef[Rec_cov, Any]] | None:
        """Main index of this dataset."""
        return set(self.record_type._primary_keys.values())

    @cached_property
    def path_idx(self) -> list[AttrRef] | None:
        """Optional, alternative index of this dataset based on merge path."""
        return self.base.path_idx if isinstance(self.base, PropRef) else None

    @cached_property
    def base_table(self) -> sqla.FromClause:
        """Get the main table for the current selection."""
        return (
            sqla.union(*(sqla.select(ds.base_table) for ds in self.base)).subquery()
            if isinstance(self.base, set)
            else (
                self.db._get_alias(self.base)
                if isinstance(self.base, RelRef)
                else (
                    self.db._get_alias(self.base.parent)
                    if isinstance(self.base, PropRef) and self.base.parent is not None
                    else (self.db._get_joined_table(self.record_type))
                )
            )
        )

    def joins(self, _subtree: RelDict | None = None) -> list[Join]:
        """Extract join operations from the relation tree."""
        joins = []
        _subtree = _subtree or self.extensions.dict

        for rel, next_subtree in _subtree.items():
            parent = (
                self.db._get_alias(rel.parent)
                if rel.parent is not None
                else self.base_table
            )

            temp_alias_map = {
                rec: self.db._get_random_alias(rec) for rec in rel.inter_joins.keys()
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

            target_table = self.db._get_alias(rel)

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

            joins.extend(self.joins(next_subtree))

        return joins

    def _parse_merge_tree(self, merge: RelRef | RelTree | None) -> RelTree:
        """Parse merge argument and prefix with current selection."""
        merge = (
            merge
            if isinstance(merge, RelTree)
            else RelTree({merge}) if merge is not None else RelTree()
        )
        return (
            self.base >> merge
            if isinstance(self.base, RelRef)
            else (
                self.base.path[-1] >> merge if isinstance(self.base, PropRef) else merge
            )
        )

    def _gen_idx_match_expr(
        self,
        values: Sequence[slice | list[Hashable] | Hashable | sqla.ColumnElement],
    ) -> sqla.ColumnElement[bool] | None:
        """Generate SQL expression for matching index values."""
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
                idx.label(None).in_(val)
                if isinstance(val, list)
                else (
                    idx.label(None).between(val.start, val.stop)
                    if isinstance(val, slice)
                    else idx.label(None) == val
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
        reflist: set[RelRef] = set(),
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if isinstance(element, AttrRef):
            if element.parent is not None:
                reflist.add(element.parent)

            if isinstance(self.base, RelRef):
                element = self.base >> element
            elif isinstance(self.base, PropRef) and self.base.parent is not None:
                element = self.base.parent >> element

            table = (
                self.db._get_alias(element.parent)
                if element.parent is not None
                else self.db._get_table(element.record_type)
            )
            return table.c[element.name]

        return None

    def _parse_filter(
        self,
        key: sqla.ColumnElement[bool] | pd.Series,
    ) -> tuple[sqla.ColumnElement[bool], RelTree]:
        """Parse filter argument and return SQL expression and join operations."""
        filt = None
        merge = RelTree()

        match key:
            case sqla.ColumnElement():
                # Filtering via SQL expression.
                reflist = set()
                replace_func = partial(self._replace_attr, reflist=reflist)
                filt = sqla_visitors.replacement_traverse(key, {}, replace=replace_func)
                merge += RelTree(reflist)
            case pd.Series():
                # Filtering via boolean Series.
                assert all(
                    n1 == n2.name
                    for n1, n2 in zip(
                        key.index.names, self.record_type._primary_keys.values()
                    )
                )

                series_name = token_hex(8)
                uploaded = self.db.dataset(
                    key.rename(series_name).to_frame(),
                    foreign_keys={
                        pk: getattr(self.record_type, pk) for pk in key.index.names
                    },
                )

                filt = (
                    self.db._get_table(uploaded.record_type).c[  # noqa: E712
                        series_name
                    ]
                    == True
                )
                merge += RelRef(
                    on=uploaded.record_type,
                    record_type=self.record_type,
                    prop_type=TypeDef(Rel[uploaded.record_type]),
                )

        return filt, merge

    @cached_property
    def base_set(self) -> DataSet[Name, Rec_cov, Idx_cov]:
        """Base dataset for this dataset."""
        return DataSet(self.db, self.record_type)

    # Overloads: attribute selection:

    # 1. Top-level attribute selection
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], Any],
        key: AttrRef[Rec_cov, Val_cov],
    ) -> DataSet[Name, Scalar[Val_cov, Key2], Idx_cov]: ...

    # 2. Nested attribute selection, relational index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], Any], key: AttrRef[Any, Val_cov]
    ) -> DataSet[Name, Scalar[Val_cov], Idx_cov | Key2 | IdxStart[Key | Key2]]: ...

    # Overloads: relation selection:

    # 3. Top-level relation selection, singular, base index
    @overload
    def __getitem__(  # type: ignore
        self: DataSet[Any, Record[Key2], BaseIdx],
        key: RelRef[Rec_cov, Rec2 | None, Rec2],
    ) -> DataSet[Name, Rec2, Key2]: ...

    # 4. Top-level relation selection, singular, single index
    @overload
    def __getitem__(  # type: ignore
        self: DataSet[Any, Any, SingleIdx],
        key: RelRef[Rec_cov, Rec2, Rec2],
    ) -> DataSet[Name, Rec2, SingleIdx]: ...

    # 5. Top-level relation selection, singular nullable, single index
    @overload
    def __getitem__(  # type: ignore
        self: DataSet[Any, Any, SingleIdx],
        key: RelRef[Rec_cov, Rec2 | None, Rec2],
    ) -> DataSet[Name, Rec2 | Scalar[None], SingleIdx]: ...

    # 6. Top-level relation selection, singular, custom index
    @overload
    def __getitem__(
        self: DataSet[Any, Any, Key],
        key: RelRef[Rec_cov, Rec2 | None, Rec2],
    ) -> DataSet[Name, Rec2, Key]: ...

    # 7. Top-level relation selection, mapping, base index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], BaseIdx],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
    ) -> DataSet[Name, Rec2, tuple[Key2, Key3]]: ...

    # 8. Top-level relation selection, mapping, single index
    @overload
    def __getitem__(
        self: DataSet[Any, Any, SingleIdx],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
    ) -> DataSet[Name, Rec2, Key3]: ...

    # 9. Top-level relation selection, mapping, tuple index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], tuple[*IdxTup]],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
    ) -> DataSet[Name, Rec2, tuple[*IdxTup, Key3] | tuple[Key2, Key3]]: ...

    # 10. Top-level relation selection, mapping, custom index
    @overload
    def __getitem__(
        self: DataSet[Any, Any, Key],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
    ) -> DataSet[Name, Rec2, tuple[Key, Key3]]: ...

    # 11. Top-level relation selection, iterable, base index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], BaseIdx],
        key: RelRef[Rec_cov, Iterable[Rec2], Record[Key4]],
    ) -> DataSet[Name, Rec2, tuple[Key2, Key4]]: ...

    # 12. Top-level relation selection, iterable, single index
    @overload
    def __getitem__(
        self: DataSet[Any, Any, SingleIdx],
        key: RelRef[Rec_cov, Iterable[Rec2], Record[Key4]],
    ) -> DataSet[Name, Rec2, Key4]: ...

    # 13. Top-level relation selection, iterable, tuple index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], tuple[*IdxTup]],
        key: RelRef[Rec_cov, Iterable[Rec2], Record[Key4]],
    ) -> DataSet[Name, Rec2, tuple[*IdxTup, Key4] | tuple[Key2, Key4]]: ...

    # 14. Top-level relation selection, iterable, custom index
    @overload
    def __getitem__(
        self: DataSet[Any, Any, Key],
        key: RelRef[Rec_cov, Iterable[Rec2], Record[Key4]],
    ) -> DataSet[Name, Rec2, tuple[Key, Key4]]: ...

    # 15. Nested relation selection, base index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], BaseIdx],
        key: RelRef[
            Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2] | None, Record[Key4]
        ],
    ) -> DataSet[Name, Rec2, IdxStartEnd[Key2, Key3 | Key4]]: ...

    # 16. Nested relation selection, single index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], SingleIdx],
        key: RelRef[Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2], Record[Key4]],
    ) -> DataSet[Name, Rec2, IdxEnd[Key3 | Key4]]: ...

    # 17. Nested relation selection, tuple index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], tuple[*IdxTup]],
        key: RelRef[
            Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2] | None, Record[Key4]
        ],
    ) -> DataSet[
        Name,
        Rec2,
        IdxTupStartEnd[*IdxTup, Key3 | Key4] | IdxStartEnd[Key2, Key3 | Key4],
    ]: ...

    # 18. Nested relation selection, custom index
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], Key],
        key: RelRef[
            Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2] | None, Record[Key4]
        ],
    ) -> DataSet[Name, Rec2, IdxStartEnd[Key | Key2, Key3 | Key4]]: ...

    # 19. Default relation selection
    @overload
    def __getitem__(
        self: DataSet[Any, Any, Any, Any],
        key: RelRef[Any, Any, Rec2],
    ) -> DataSet[Name, Rec2, Any]: ...

    # 20. List selection
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], Key], key: Iterable[Key | Key2]
    ) -> DataSet[Name, Rec_cov, Key]: ...

    # 21. Merge selection
    @overload
    def __getitem__(
        self: DataSet[Any, Any, Key], key: RelTree[Rec_cov, *RelTup]
    ) -> DataSet[Name, Rec_cov, Key, tuple[*RelTup]]: ...

    # 22. Merge selection, single index
    @overload
    def __getitem__(
        self: DataSet[Any, Any, SingleIdx], key: RelTree[Rec_cov, *RelTup]
    ) -> DataSet[Name, Rec_cov, BaseIdx, tuple[*RelTup]]: ...

    # 23. List selection from record (value) tuple, keep index
    @overload
    def __getitem__(  # type: ignore
        self: DataSet[Any, Any, Any, tuple[Record, ...]], key: Iterable[Hashable]
    ) -> DataSet[Name, Rec_cov, Idx_cov]: ...

    # 24. Index value selection
    @overload
    def __getitem__(
        self: DataSet[Any, Record[Key2], BaseIdx | Key], key: Key | Key2
    ) -> DataSet[Name, Rec_cov, SingleIdx]: ...

    # 25. Index value selection from record (value) tuple
    @overload
    def __getitem__(
        self: DataSet[Any, Any, Any, tuple[Record, ...]], key: Hashable
    ) -> DataSet[Name, Rec_cov, SingleIdx]: ...

    # 26. Slice selection, keep index
    @overload
    def __getitem__(self, key: slice | tuple[slice, ...]) -> Self: ...

    # 27. Expression filtering, keep index
    @overload
    def __getitem__(self, key: sqla.ColumnElement[bool] | pd.Series) -> Self: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self,
        key: (
            AttrRef[Any, Val_cov]
            | RelRef
            | RelTree
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice, ...]
            | sqla.ColumnElement[bool]
            | Series
        ),
    ) -> DataSet[Any, Any, Any, Any]:
        ext = None
        filt = None

        if isinstance(key, sqla.ColumnElement | pd.Series):
            filt, ext = self._parse_filter(key)

        if isinstance(key, RelTree):
            ext = self._parse_merge_tree(key)

        keys = None
        if isinstance(key, tuple):
            # Selection by tuple of index values.
            keys = list(key)

        if isinstance(key, list | slice) and not isinstance(
            key, sqla.ColumnElement | pd.Series
        ):
            # Selection by index value list, slice or single value.
            keys = [key]

        return DataSet(
            self.db,
            (
                self.base >> key
                if isinstance(self.base, type | RelRef) and isinstance(key, PropRef)
                else self.base
            ),
            (self.extensions + ext if isinstance(ext, RelTree) else self.extensions),
            self.filters + [filt] if filt is not None else self.filters,
            keys if keys is not None else self.keys,
        )

    def select(
        self,
        *,
        index_only: bool = False,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        selection_table = self.base_table

        select = sqla.select(
            *(
                col.label(f"{self.record_type._default_table_name()}.{col_name}")
                for col_name, col in selection_table.columns.items()
                if not index_only or col.primary_key
            ),
            *(
                col.label(f"{rel.path_str}.{col_name}")
                for rel in self.extensions.rels
                for col_name, col in self.db._get_alias(rel).columns.items()
                if not index_only or col.primary_key
            ),
        ).select_from(selection_table)

        for join in self.joins():
            select = select.join(*join)

        for filt in self.filters:
            select = select.where(filt)

        return select

    def _load_prop(self, prop: PropRef[Record[Key], Val], parent_idx: Key) -> Val:
        base = self.db[prop.record_type]
        base_record = base[parent_idx]

        if isinstance(prop, AttrRef):
            return getattr(base_record.load(), prop.name)
        elif isinstance(prop, RelRef):
            recs = base_record[prop].load()
            recs_type = prop.prop_type.value_type()

            if (
                isinstance(recs, dict)
                and not issubclass(recs_type, Mapping)
                and issubclass(recs_type, Iterable)
            ):
                recs = list(recs.values())

            if prop.collection is not None:
                recs = prop.collection(recs)

            return cast(Val, recs)

        raise TypeError("Invalid property reference.")

    @overload
    def load(
        self: DataSet[Name, Rec_cov, SingleIdx],
        kind: type[Record] = ...,
    ) -> Rec_cov: ...

    @overload
    def load(
        self: DataSet[Name, Rec_cov, SingleIdx, tuple[*RelTup]],
        kind: type[Record] = ...,
    ) -> tuple[Rec_cov, *RelTup]: ...

    @overload
    def load(
        self: DataSet[Name, Any, Any, tuple[*RelTup]], kind: type[Df]
    ) -> tuple[Df, ...]: ...

    @overload
    def load(self, kind: type[Df]) -> Df: ...

    @overload
    def load(
        self: DataSet[Name, Any, Hashable | BaseIdx, tuple[*RelTup]],
        kind: type[Record] = ...,
    ) -> dict[Hashable, tuple[Rec_cov, *RelTup]]: ...

    @overload
    def load(
        self: DataSet[Name, Rec_cov, Key], kind: type[Record] = ...
    ) -> dict[Key, Rec_cov]: ...

    @overload
    def load(self, kind: type[Record] = ...) -> list[Rec_cov]: ...

    def load(
        self: DataSet[Any, Record, Any, Any],
        kind: type[Record | Df] = Record,
    ) -> (
        Rec_cov
        | tuple[Rec_cov, *tuple[Any, ...]]
        | Df
        | tuple[Df, ...]
        | dict[Any, Rec_cov]
        | dict[Any, tuple[Rec_cov, *tuple[Any, ...]]]
        | list[Rec_cov]
    ):
        """Download selection."""
        select = self.select()

        idx_cols = [
            f"{rel.path_str}.{pk}"
            for rel in self.extensions.rels
            for pk in rel.target_type._primary_keys
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
            for rel in self.extensions.rels
        }

        merged_df = None
        if kind is pd.DataFrame:
            with self.db.engine.connect() as con:
                merged_df = pd.read_sql(select, con)
                merged_df = merged_df.set_index(idx_cols)
        else:
            merged_df = pl.read_database(select, self.db.engine)

        if issubclass(kind, Record):
            assert isinstance(merged_df, pl.DataFrame)

            rec_types = {
                self.record_type: main_cols,
                **{rel.target_type: cols for rel, cols in extra_cols.items()},
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
                    rec_data: dict[PropRef, Any] = {
                        getattr(rec_type, attr): row[col] for col, attr in cols.items()
                    }
                    rec_idx = self.record_type._index_from_dict(rec_data)
                    rec = loaded[rec_type].get(rec_idx) or self.record_type(
                        _loader=self._load_prop,
                        **{p.name: v for p, v in rec_data.items()},  # type: ignore
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

    # 1. Top-level attribute assignment, base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: AttrRef[Rec_cov, Val_cov],
        value: (
            DataSet[Name, Scalar[Val_cov, Key2], BaseIdx | SingleIdx]
            | DataSet[Name, Scalar[Val_cov], Key2 | SingleIdx]
            | ValInput[Val_cov, Key2]
        ),
    ) -> None: ...

    # 2. Top-level attribute assignment, custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, Key],
        key: AttrRef[Rec_cov, Val_cov],
        value: (
            DataSet[Name, Scalar[Val_cov, Key], BaseIdx | SingleIdx]
            | DataSet[Name, Scalar[Val_cov], Key | SingleIdx]
            | ValInput[Val_cov, Key]
        ),
    ) -> None: ...

    # 3. Top-level attribute assignment, single index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, SingleIdx],
        key: AttrRef[Rec_cov, Val_cov],
        value: DataSet[Name, Scalar[Val_cov], SingleIdx] | Val_cov,
    ) -> None: ...

    # 4. Nested attribute assignment, base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: AttrRef[Any, Val_cov],
        value: (
            DataSet[Name, Scalar[Val_cov, IdxStart[Key2]], BaseIdx]
            | DataSet[Name, Scalar[Val_cov], IdxStart[Key2] | SingleIdx]
            | ValInput[Val_cov, IdxStart[Key2]]
        ),
    ) -> None: ...

    # 5. Nested attribute assignment, custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, Key],
        key: AttrRef[Any, Val_cov],
        value: (
            DataSet[Name, Scalar[Val_cov, IdxStart[Key]], BaseIdx]
            | DataSet[Name, Scalar[Val_cov], IdxStart[Key] | SingleIdx]
            | ValInput[Val_cov, IdxStart[Key]]
        ),
    ) -> None: ...

    # 6. Nested attribute assignment, single index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, SingleIdx],
        key: AttrRef[Any, Val_cov],
        value: DataSet[Name, Scalar[Val_cov, Any], Any] | Val_cov,
    ) -> None: ...

    # 7. Top-level relation assignment, singular, base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: RelRef[Rec_cov, Rec2, Rec2],
        value: DataSet[Name, Rec2, Key2 | SingleIdx] | PartialRecInput[Rec2, Key2],
    ) -> None: ...

    # 8. Top-level relation assignment, singular, custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, Key],
        key: RelRef[Rec_cov, Rec2, Rec2],
        value: DataSet[Name, Rec2, Key | SingleIdx] | PartialRecInput[Rec2, Key],
    ) -> None: ...

    # 9. Top-level relation assignment, singular, single index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, SingleIdx],
        key: RelRef[Rec_cov, Rec2, Rec2],
        value: DataSet[Name, Rec2, SingleIdx] | Rec2 | PartialRec[Rec2],
    ) -> None: ...

    # 10. Top-level relation assignment, singular nullable, single index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, SingleIdx],
        key: RelRef[Rec_cov, Rec2 | None, Rec2],
        value: (
            DataSet[Name, Rec2 | Scalar[None], SingleIdx]
            | Rec2
            | PartialRec[Rec2]
            | None
        ),
    ) -> None: ...

    # 11. Top-level relation assignment, mapping, base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
        value: (
            DataSet[Name, Rec2, tuple[Key2, Key3] | SingleIdx]
            | PartialRecInput[Rec2, tuple[Key2, Key3]]
        ),
    ) -> None: ...

    # 12. Top-level relation assignment, mapping, tuple index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], tuple[*IdxTup]],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
        value: (
            DataSet[Name, Rec2, tuple[*IdxTup, Key3] | SingleIdx]
            | PartialRecInput[Rec2, tuple[*IdxTup, Key3]]
        ),
    ) -> None: ...

    # 13. Top-level relation assignment, mapping, custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, Key],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
        value: (
            DataSet[Name, Rec2, tuple[Key2, Key3] | SingleIdx]
            | PartialRecInput[Rec2, tuple[Key2, Key3]]
        ),
    ) -> None: ...

    # 14. Top-level relation assignment, mapping, single index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, SingleIdx],
        key: RelRef[Rec_cov, Mapping[Key3, Rec2], Rec2],
        value: DataSet[Name, Rec2, Key3 | SingleIdx] | PartialRecInput[Rec2, Key3],
    ) -> None: ...

    # 15. Top-level relation assignment, iterable, base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: RelRef[Rec_cov, Iterable[Rec2], Record[Key4]],
        value: (
            DataSet[Name, Rec2, tuple[Key2, Key4] | SingleIdx]
            | PartialRecInput[Rec2, tuple[Key2, Key4]]
        ),
    ) -> None: ...

    # 16. Top-level relation assignment, iterable, custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, Key],
        key: RelRef[Rec_cov, Iterable[Rec2], Record[Key4]],
        value: (
            DataSet[Name, Rec2, tuple[Key, Key4] | SingleIdx]
            | PartialRecInput[Rec2, tuple[Key, Key4]]
        ),
    ) -> None: ...

    # 17. Top-level relation assignment, iterable, single index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, SingleIdx],
        key: RelRef[Rec_cov, Iterable[Rec2], Record[Key4]],
        value: (
            DataSet[Name, Rec2, Key4 | BaseIdx | SingleIdx]
            | PartialRecInput[Rec2, Key4]
        ),
    ) -> None: ...

    # 18. Nested relation assignment, base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: RelRef[
            Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2] | None, Record[Key4]
        ],
        value: (
            DataSet[Name, Rec2, IdxStartEnd[Key2, Key3 | Key4] | SingleIdx]
            | PartialRecInput[Rec2, IdxStartEnd[Key2, Key3 | Key4]]
            | None
        ),
    ) -> None: ...

    # 19. Nested relation assignment, single index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, SingleIdx],
        key: RelRef[
            Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2] | None, Record[Key4]
        ],
        value: (
            DataSet[Name, Rec2, IdxEnd[Key3 | Key4] | BaseIdx | SingleIdx]
            | PartialRecInput[Rec2, IdxEnd[Key3 | Key4]]
            | None
        ),
    ) -> None: ...

    # 20. Nested relation assignment, tuple index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, tuple[*IdxTup]],
        key: RelRef[
            Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2] | None, Record[Key4]
        ],
        value: (
            DataSet[Name, Rec2, IdxTupStartEnd[*IdxTup, Key3 | Key4] | SingleIdx]
            | PartialRecInput[Rec2, IdxTupStartEnd[*IdxTup, Key3 | Key4]]
            | None
        ),
    ) -> None: ...

    # 21. Nested relation assignment, custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, Key],
        key: RelRef[
            Any, Rec2 | Iterable[Rec2] | Mapping[Key3, Rec2] | None, Record[Key4]
        ],
        value: (
            DataSet[Name, Rec2, IdxStartEnd[Key, Key3 | Key4] | SingleIdx]
            | PartialRecInput[Rec2, IdxStartEnd[Key, Key3 | Key4]]
            | None
        ),
    ) -> None: ...

    # 22. Merge assignment
    @overload
    def __setitem__(
        self,
        key: RelTree[Rec_cov, *RelTup],
        value: (
            DataSet[Name, Rec_cov, Any, tuple[*RelTup]]
            | tuple[PartialRecInput[Rec_cov, Any], *tuple[PartialRecInput, ...]]
        ),
    ) -> None: ...

    # 23. List assignment with base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: list[Key2],
        value: (
            DataSet[Name, Rec_cov, Key2 | BaseIdx | SingleIdx]
            | PartialRecInput[Rec_cov, int]
        ),
    ) -> None: ...

    # 24. List assignment with custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Any, Key],
        key: list[Key] | list[Key2],
        value: DataSet[Name, Rec_cov, Key | SingleIdx] | PartialRecInput[Rec_cov, int],
    ) -> None: ...

    # 25. Single value assignment with base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2] | Scalar[Val_cov], BaseIdx],
        key: Key2,
        value: Rec_cov | Val_cov,
    ) -> None: ...

    # 26. Single value assignment with custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2] | Scalar[Val_cov], Key],
        key: Key | Key2,
        value: Rec_cov | Val_cov,
    ) -> None: ...

    # 27. Slice assignment
    @overload
    def __setitem__(
        self,
        key: slice | tuple[slice, ...],
        value: (
            DataSet[Name, Rec_cov, Idx_cov | SingleIdx] | PartialRecInput[Rec_cov, int]
        ),
    ) -> None: ...

    # 28. Filter assignment with base index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], BaseIdx],
        key: sqla.ColumnElement[bool] | pd.Series,
        value: (
            DataSet[Name, Rec_cov, Key2 | SingleIdx] | PartialRecInput[Rec_cov, Key2]
        ),
    ) -> None: ...

    # 29. Filter assignment with custom index
    @overload
    def __setitem__(
        self: DataSet[Name, Record[Key2], Key],
        key: sqla.ColumnElement[bool] | pd.Series,
        value: (
            DataSet[Name, Rec_cov, Key | Key2 | SingleIdx]
            | PartialRecInput[Rec_cov, Key | Key2]
        ),
    ) -> None: ...

    def __setitem__(  # noqa: D105
        self,
        key: (
            AttrRef
            | RelRef
            | RelTree
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice | tuple[slice, ...], ...]
            | sqla.ColumnElement[bool]
            | pd.Series
        ),
        value: DataSet | PartialRecInput | ValInput | tuple[PartialRecInput, ...],
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
        self: DataSet[Name, Record[Key2], BaseIdx],
        value: DataSet[Name, Rec_cov, Key2 | BaseIdx] | RecInput[Rec_cov, Key2],
    ) -> DataSet[Name, Record[Key2], BaseIdx]:
        """Merge update into unfiltered dataset."""
        return self._set(value, mode="upsert")

    # 1. List deletion
    @overload
    def __delitem__(
        self: DataSet[Any, Record[Key2], Key], key: Iterable[Key | Key2]
    ) -> None: ...

    # 2. Index value deletion
    @overload
    def __delitem__(
        self: DataSet[Any, Record[Key2], BaseIdx | Key], key: Key | Key2
    ) -> None: ...

    # 3. Slice deletion
    @overload
    def __delitem__(self, key: slice | tuple[slice, ...]) -> None: ...

    # 4. Expression filter deletion
    @overload
    def __delitem__(self, key: sqla.ColumnElement[bool] | pd.Series) -> None: ...

    # Implementation:

    def __delitem__(  # noqa: D105
        self,
        key: (
            list[Hashable]
            | Hashable
            | slice
            | tuple[slice, ...]
            | sqla.ColumnElement[bool]
            | Series
        ),
    ) -> None:
        if not isinstance(key, slice) or key != slice(None):
            del self[key][:]  # type: ignore

        select = self.select()

        tables = {self.db._get_table(rec) for rec in self.record_type._record_bases}

        statements = []

        for table in tables:
            # Prepare delete statement.
            if self.db.engine.dialect.name in (
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
        with self.db.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

    def __or__(
        self, other: DataSet[Name, Rec_cov, Idx]
    ) -> DataSet[Name, Rec_cov, Idx_cov | Idx]:
        """Union two datasets."""
        return DataSet(self.db, {self, other})  # type: ignore

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        with self.db.engine.connect() as conn:
            res = conn.execute(
                sqla.select(sqla.func.count()).select_from(self.select().subquery())
            ).scalar()
            assert isinstance(res, int)
            return res

    def extract(
        self,
        use_schema: bool | type[Schema] = False,
        aggs: Mapping[RelRef, Agg] | None = None,
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
            else ({self.record_type, *self.record_type._rel_types})
        )

        # Get the entire subdag from this selection.
        all_paths_rels = {
            r
            for rel in self.record_type._rels.values()
            for r in rel.get_subdag(rec_types)
        }

        # Extract rel paths, which contain an aggregated rel.
        aggs_per_type: dict[type[Record], list[tuple[RelRef, Agg]]] = {}
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
                                if isinstance(sa, AttrRef)
                                else sqla_visitors.replacement_traverse(
                                    sa,
                                    {},
                                    replace=lambda element, **kw: (
                                        src_select.c[element.name]
                                        if isinstance(element, AttrRef)
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

    def _get_record_rels(
        self, rec_data: dict[Any, dict[PropRef, Any]], list_idx: bool, covered: set[int]
    ) -> dict[RelRef, dict[Any, Record]]:
        """Get relation data from record data."""
        rel_data: dict[RelRef, dict[Any, Record]] = {}

        for idx, rec in rec_data.items():
            for prop, prop_val in rec.items():
                if isinstance(prop, RelRef) and not isinstance(prop_val, Unloaded):
                    rel_data[prop] = {
                        **rel_data.get(prop, {}),
                        **self._normalize_rel_data(prop_val, idx, list_idx, covered),
                    }

        return rel_data

    def _set(  # noqa: C901, D105
        self,
        value: DataSet | PartialRecInput[Rec_cov, Any] | ValInput,
        mode: Literal["update", "upsert", "replace"] = "update",
        covered: set[int] | None = None,
    ) -> Self:
        covered = covered or set()

        record_data: dict[Any, dict[PropRef, Any]] | None = None
        rel_data: dict[RelRef[Rec_cov, Any, Any], dict[Any, Record]] | None = None
        df_data: DataFrame | None = None
        partial: bool = False

        list_idx = (
            self.path_idx is not None
            and len(self.path_idx) == 1
            and issubclass(self.path_idx[0].prop_type.value_type(), int)
        )

        if isinstance(value, Record):
            record_data = {value._index: value._to_dict()}
        elif isinstance(value, Mapping):
            if has_type(value, Mapping[PropRef, Any]):
                record_data = {
                    self.record_type._index_from_dict(value): {
                        p: v for p, v in value.items()
                    }
                }
                partial = True
            else:
                assert has_type(value, Mapping[Any, PartialRec[Record]])

                record_data = {}
                for idx, rec in value.items():
                    if isinstance(rec, Record):
                        rec_dict = rec._to_dict()
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
                    rec_dict = rec._to_dict()
                    rec_idx = rec._index
                else:
                    rec_dict = {p: v for p, v in rec.items()}
                    rec_idx = self.record_type._index_from_dict(rec)
                    partial = True

                record_data[idx if list_idx else rec_idx] = rec_dict

        assert not (
            mode in ("upsert", "replace") and partial
        ), "Partial record input requires update mode."

        if record_data is not None:
            # Split record data into attribute and relation data.
            attr_data = {
                idx: {p.name: v for p, v in rec.items() if isinstance(p, AttrRef)}
                for idx, rec in record_data.items()
            }
            rel_data = self._get_record_rels(record_data, list_idx, covered)

            # Transform attribute data into DataFrame.
            df_data = pd.DataFrame.from_records(attr_data)

        if rel_data is not None:
            # Recurse into relation data.
            for r, r_data in rel_data.items():
                self[r]._set(r_data, mode="replace", covered=covered)

        # Load data into a temporary table.
        if df_data is not None:
            value_set = self.db.dataset(df_data)
        elif isinstance(value, sqla.Select):
            value_set = self.db.dataset(value)
        else:
            assert isinstance(value, DataSet)
            value_set = value

        attrs_by_table = {
            self.db._get_table(rec): {
                a for a in self.record_type._attrs.values() if a.record_type is rec
            }
            for rec in self.record_type._record_bases
        }

        statements = []

        if mode == "replace":
            # Delete all records in the current selection.
            select = self.select()

            for table in attrs_by_table:
                # Prepare delete statement.
                if self.db.engine.dialect.name in (
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

        if mode in ("replace", "upsert"):
            # Construct the insert statements.

            assert (
                isinstance(self.base, Record) and len(self.filters) == 0
            ), "Can only upsert into unfiltered base datasets."
            assert value_set is not None

            for table, attrs in attrs_by_table.items():
                # Do an insert-from-select operation, which updates on conflict:
                statement = table.insert().from_select(
                    [a.name for a in attrs],
                    value_set.select().subquery(),
                )

                if isinstance(statement, postgresql.Insert):
                    # For Postgres / DuckDB, use: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#updating-using-the-excluded-insert-values
                    statement = statement.on_conflict_do_update(
                        index_elements=[col.name for col in table.primary_key.columns],
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

                statements.append(statement)
        else:
            # Construct the update statements.

            assert isinstance(self.base_table, sqla.Table), "Base must be a table."

            # Derive current select statement and join with value table, if exists.
            select = self.select()
            if value_set is not None:
                select = select.join(
                    value_set.base_table,
                    reduce(
                        sqla.and_,
                        (
                            self.base_table.c[idx_col.name] == idx_col
                            for idx_col in value_set.base_table.primary_key
                        ),
                    ),
                )

            selected_attr = self.base.name if isinstance(self.base, AttrRef) else None

            for table, attrs in attrs_by_table.items():
                values = (
                    {
                        a.name: value_set.base_table.c[a.name]
                        for a in (attrs & set(value_set.record_type._attrs.values()))
                    }
                    if value_set is not None
                    else {selected_attr: value}
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
        with self.db.engine.begin() as con:
            for statement in statements:
                con.execute(statement)

        # Drop the temporary table, if any.
        if value_set is not None:
            cast(sqla.Table, value_set.base_table).drop(self.db.engine)

        if mode in ("replace", "upsert") and isinstance(self.base, RelRef):
            # Update incoming relations from parent records.
            parent_set = DataSet(
                self.db,
                (
                    self.base.parent
                    if self.base.parent is not None
                    else self.base.record_type
                ),
                self.extensions,
                self.filters,
                self.keys,
            )

            if issubclass(self.base.fk_record_type, self.base.record_type):
                # Case: parent links directly to child (n -> 1)
                # Somehow get list of all updated child record indexes.
                # Update parent records with new child indexes.
                raise NotImplementedError()
            elif self.base.link_rel is not None:
                # Case: parent and child are linked via assoc table (n <--> m)
                # Update link table with new child indexes.
                raise NotImplementedError()

            # Note that the (1 <- n) case is already covered by updating
            # the child record directly, which includes all its foreign keys.

        return self
