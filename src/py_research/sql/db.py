"""Abstract Python interface for SQL databases."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property, reduce
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
    Keyless,
    MergeTup,
    PropRef,
    Rec,
    Rec2,
    Rec_cov,
    Record,
    Rel,
    RelMerge,
    RelRef,
    RelTree,
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
Dl = TypeVar("Dl", bound=DataFrame | Record)


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


@dataclass(frozen=True)
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
            fks = set(
                [
                    attr.name
                    for rels in rec._rels.values()
                    for rel in rels
                    for attr in rel.fk_map.keys()
                ]
            )
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
            for rel_list in rec._rels.values():
                for rel in rel_list:
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
        table = self._get_table(key)
        return DataSet(self, table, key)

    def __setitem__(  # noqa: D105
        self,
        key: type[Rec],
        value: "DataSet[Name, Rec, Idx | None] | RecInput[Rec, Idx] | sqla.Select[tuple[Rec]]",  # noqa: E501
    ) -> None:
        table = self._ensure_table_exists(self._get_table(key))
        DataSet(self, table, key)[:] = value

    def dataset(
        self, data: RecInput[Rec, Any], record_type: type[Rec] | None = None
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
            table = sqla.Table(table_name, self.meta, *cols_from_df(data).values())
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

        ds = DataSet(self, table, record_type)
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
            sqla_table = self._get_table(rec)
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
            n.load().reset_index().assign(table=n.record_type._default_table_name())
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
                    left, right = (v[0] for v in at._rels.values())
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
                    .load()
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

    @cache
    def _get_table(self, rec: type[Record]) -> sqla.Table:
        return rec._table(self.meta, self.subs)

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
                    f"SELECT * FROM {self._get_table(rec)}",
                    self.engine,
                ).write_excel(wb, worksheet=rec._default_table_name())

        if isinstance(self.backend.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.backend.url.set(file)


class DefaultIdx:
    """Singleton to make dataset index via default index."""


@dataclass(frozen=True)
class DataSet(Generic[Name, Rec_cov, Idx_cov]):
    """Dataset selection."""

    db: DataBase[Name, Any]
    base_table: sqla.Table
    selection: type[Rec_cov] | PropRef[Rec_cov, Any] | None = None
    merges: RelMerge = RelMerge()
    filters: list[sqla.ColumnElement[bool]] = field(default_factory=list)
    index: AttrRef[Record, Any] | Iterable[AttrRef[Record, Any]] | DefaultIdx | None = (
        None
    )
    attr: AttrRef[Rec_cov, Any] | None = None

    @cached_property
    def record_type(self) -> type[Rec_cov] | None:
        """Record type of this dataset."""
        return self.selection if isinstance(self.selection, type) else None

    def _get_alias(
        self, path: list[RelRef], rel: Rel, target: type[Record]
    ) -> sqla.FromClause:
        """Get alias for a relation."""
        return self.db._get_table(rel.target_type).alias(
            gen_str_hash((path, rel, target), 8)
        )

    def _joins_from_tree(
        self, tree: RelTree, parents: list[sqla.FromClause]
    ) -> list[tuple[sqla.FromClause, sqla.ColumnElement[bool]]]:
        """Extract join operations from a relation tree."""
        joins = []

        for relref, subtree in tree.items():
            parents = []
            for rel in relref.rels:
                fj, sj = rel.joins
                table = self._get_alias(relref.path, rel, fj.rec)
                joins.append(
                    (
                        table,
                        reduce(
                            sqla.or_,
                            (
                                reduce(
                                    sqla.and_,
                                    (
                                        parent.c[fk.name] == table.c[tk.name]
                                        for fk, tk in fj.on.items()
                                    ),
                                )
                                for parent in parents
                            ),
                        ),
                    )
                )
                if sj is not None:
                    target_table = self._get_alias(relref.path, rel, sj.rec)
                    joins.append(
                        (
                            table,
                            reduce(
                                sqla.and_,
                                (
                                    table.c[fk.name] == target_table.c[tk.name]
                                    for fk, tk in fj.on.items()
                                ),
                            ),
                        )
                    )
                    parents.append(target_table)
                else:
                    parents.append(table)

            joins.extend(self._joins_from_tree(subtree, parents))

        return joins

    @cached_property
    def joins(self) -> list[tuple[sqla.FromClause, sqla.ColumnElement[bool]]]:
        """List of all join operations required to construct this tree."""
        return self._joins_from_tree(self.merges.rel_tree, parents=[self.base_table])

    @cached_property
    def active_idx(self) -> list[list[AttrRef[Record, Any]]]:
        """Optional, additional index of this dataset."""
        path = self.selection.path if isinstance(self.selection, PropRef) else []

        if isinstance(self.index, DefaultIdx):
            path_idx = (
                [
                    relref.rec_type._primary_keys()
                    for relref in path
                    if relref.idx is not None
                ]
                if not any(relref.idx is Keyless for relref in path)
                else []
            )
            if len(path_idx) > 0:
                return path_idx
            else:
                assert isinstance(self.selection, type)
                return [self.selection._primary_keys()]
        elif self.index is not None:
            return [
                [self.index] if isinstance(self.index, AttrRef) else list(self.index)
            ]
        else:
            return []

    @cached_property
    def idx_type(self) -> list[list[type]]:
        """Type hint for current index."""
        return [[idx_ii.val_type for idx_ii in idx_i] for idx_i in self.active_idx]

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Idx2], DefaultIdx]",
        key: AttrRef[Rec_cov, Val],
    ) -> "DataSet[Name, Record[Val, None], Idx2]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Rec, Idx]", key: AttrRef[Rec, Val]
    ) -> "DataSet[Name, Record[Val, None], Idx]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Idx2], DefaultIdx]", key: AttrRef[Any, Val]
    ) -> "DataSet[Name, Record[Val, None], IdxStart[Idx2]]": ...

    @overload
    def __getitem__(
        self, key: AttrRef[Any, Val]
    ) -> "DataSet[Name, Record[Val, None], IdxStart[Idx_cov]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Rec, Idx]",
        key: RelRef[Rec, Rec2, None],
    ) -> "DataSet[Name, Rec2, Idx]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Rec, Any]",
        key: RelRef[Rec, Rec2, Keyless],
    ) -> "DataSet[Name, Rec2, DefaultIdx]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Idx2], DefaultIdx]",
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
        self: "DataSet[Name, Record[Any, Idx2], DefaultIdx]",
        key: RelRef[Any, Rec2, Idx3],
    ) -> "DataSet[Name, Rec2, IdxStartEnd[Idx2, Idx3]]": ...

    @overload
    def __getitem__(
        self, key: RelRef[Any, Rec2, Idx3]
    ) -> "DataSet[Name, Rec2, IdxStartEnd[Idx_cov, Idx3]]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Idx2], DefaultIdx]", key: list[Idx2]
    ) -> "DataSet[Name, Rec_cov, DefaultIdx]": ...

    @overload
    def __getitem__(self, key: list[Idx_cov]) -> Self: ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Idx2], DefaultIdx]", key: Idx2
    ) -> "DataSet[Name, Rec_cov, None]": ...

    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Any], Idx]", key: Idx
    ) -> "DataSet[Name, Rec_cov, None]": ...

    @overload
    def __getitem__(
        self, key: slice | tuple[slice | tuple[slice, ...], ...]
    ) -> Self: ...

    @overload
    def __getitem__(self, key: sqla.ColumnElement[bool]) -> Self: ...

    def __getitem__(  # noqa: D105
        self,
        key: (
            AttrRef[Any, Val]
            | RelRef
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice | tuple[slice, ...], ...]
            | sqla.ColumnElement[bool]
        ),
    ) -> "DataSet":
        rec, path, filt, attr, idx = self._parse_key(key)

        return DataSet(
            self.db,
            self.base_table,
            rec,
            path,
            filt,
            idx,
            attr,
        )

    @overload
    def load(self, merge: None = ..., kind: type[Dl] = ...) -> Dl: ...  # type: ignore

    @overload
    def load(self, merge: RelRef | RelMerge, kind: type[Df]) -> tuple[Df, ...]: ...

    @overload
    def load(
        self, merge: RelRef[Any, Rec, Any], kind: type[Record] = ...
    ) -> Sequence[tuple[Rec_cov, Rec]]: ...

    @overload
    def load(
        self, merge: RelMerge[*MergeTup], kind: type[Record] = ...
    ) -> Sequence[tuple[Rec_cov, *MergeTup]]: ...

    def load(
        self,
        merge: RelRef | RelMerge | None = None,
        kind: type[Dl] = Record,
    ) -> Dl | tuple[Dl, ...] | Sequence[tuple]:
        """Download table selection."""
        if issubclass(kind, Record):
            raise NotImplementedError("Downloading record instances not supported yet.")

        merge = (
            merge
            if isinstance(merge, RelMerge)
            else RelMerge([merge]) if merge is not None else RelMerge()
        )
        abs_merge = (
            self.selection / merge
            if isinstance(self.selection, RelRef)
            else (
                self.selection.path[-1] / merge
                if isinstance(self.selection, PropRef)
                else merge
            )
        )
        full_merge = self.merges + abs_merge

        select = sqla.select(
            *(
                col.label(f"{relref.path_str}.{name}")
                for relref in abs_merge.rels
                for rel in relref.rels
                for target in [j.rec for j in rel.joins if j is not None]
                for name, col in self._get_alias(
                    relref.path, rel, target
                ).columns.items()
            )
        )
        for join in self._joins_from_tree(
            full_merge.rel_tree, parents=[self.base_table]
        ):
            select = select.join(*join)

        for filt in self.filters:
            select = select.where(filt)

        if kind is pl.DataFrame:
            return cast(Dl, pl.read_database(select, self.db.engine))
        else:
            with self.db.engine.connect() as con:
                return cast(Dl, pd.read_sql(select, con))

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Idx2], DefaultIdx]",
        key: AttrRef[Rec_cov, Val],
        value: "DataSet[Name, Record[Val, Idx2], DefaultIdx] | DataSet[Name, Record[Val, Any], Idx2] | ValInput[Val, Idx2]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Any], Idx]",
        key: AttrRef[Rec_cov, Val],
        value: "DataSet[Name, Record[Val, Idx], DefaultIdx] | DataSet[Name, Record[Val, Any], Idx] | ValInput[Val, Idx]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Idx2], DefaultIdx]",
        key: AttrRef[Any, Val],
        value: "DataSet[Name, Record[Val, IdxStart[Idx2]], DefaultIdx] | DataSet[Name, Record[Val, Any], IdxStart[Idx2]] | ValInput[Val, IdxStart[Idx2]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Any], Idx]",
        key: AttrRef[Any, Val],
        value: "DataSet[Name, Record[Val, IdxStart[Idx]], DefaultIdx] | DataSet[Name, Record[Val, Any], IdxStart[Idx]] | ValInput[Val, IdxStart[Idx]]",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Rec, Idx]",
        key: RelRef[Rec, Rec2, None],
        value: "DataSet[Name, Rec2, Idx] | RecInput[Rec2, Idx]",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Rec, Any]",
        key: RelRef[Rec, Rec2, Keyless],
        value: "DataSet[Name, Rec2, DefaultIdx] | RecInput[Rec2, Any]",
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Idx2], DefaultIdx]",
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
        self: "DataSet[Name, Record[Any, Idx2], DefaultIdx]",
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
        self: "DataSet[Name, Record[Any, Idx2], DefaultIdx]",
        key: list[Idx2],
        value: "DataSet[Name, Rec_cov, Idx2 | DefaultIdx] | Iterable[Rec_cov] | DataFrame",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: list[Idx_cov],
        value: "DataSet[Name, Rec_cov, Idx | DefaultIdx] | Iterable[Rec_cov] | DataFrame",  # noqa: E501
    ) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Idx2], None]", key: Idx2, value: Rec_cov | Val
    ) -> None: ...

    @overload
    def __setitem__(self: "DataSet[Name, Rec, Idx]", key: Idx, value: Rec) -> None: ...

    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Any], Idx]", key: Idx, value: Val
    ) -> None: ...

    @overload
    def __setitem__(
        self, key: sqla.ColumnElement[bool], value: Self | RecInput[Rec_cov, Any]
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: slice | tuple[slice | tuple[slice, ...], ...],
        value: Self | Iterable[Rec_cov] | DataFrame,
    ) -> None: ...

    def __setitem__(  # noqa: D105
        self,
        key: (
            AttrRef
            | RelRef
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice | tuple[slice, ...], ...]
            | sqla.ColumnElement[bool]
        ),
        value: "DataSet | RecInput[Rec_cov, Any] | ValInput",
    ) -> None:
        rec, path, filt, attr, idx = self._parse_key(key)

        if len(path) == 0 and len(filt) == 0:
            assert rec == self.record_type, "no path, so we should be at the base table"
            assert idx is not None, "no filters, so we should not have a single-record"
            if attr is not None:
                raise NotImplementedError("Attribute assignment not implemented yet")

            cols = list(self.base_table.c.keys())
            table_fqn = str(self.base_table)
            if isinstance(value, pd.DataFrame):
                value.reset_index()[cols].to_sql(
                    table_fqn,
                    self.db.engine,
                    if_exists="replace",
                    index=False,
                )
            elif isinstance(value, pl.DataFrame):
                value[cols].write_database(
                    table_fqn,
                    str(self.db.engine.url),
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
                select = (
                    value
                    if isinstance(value, sqla.Select)
                    else sqla.select(value.base_table)
                )

                # Transfer table or query results to SQL table
                with self.db.engine.begin() as con:
                    con.execute(sqla.delete(self.base_table))
                    con.execute(
                        sqla.insert(self.base_table).from_select(
                            self.base_table.c.keys(), select
                        )
                    )
        else:
            raise NotImplementedError(
                "Assignment outside top-level or with filters not supported yet."
            )

    def __clause_element__(self) -> sqla.Subquery:
        """Return subquery for the current selection to be used inside SQL clauses."""
        # if self.selection is None:
        #     raise ValueError("Only a selected DB can be used as a clause element.")
        # return self.selection.subquery()
        raise NotImplementedError()

    def _gen_idx_match_expr(
        self, values: Sequence[Sequence[slice | list[Hashable] | Hashable]]
    ) -> sqla.ColumnElement[bool] | None:
        if values == slice(None):
            return None

        exprs = []

        assert len(values) == len(self.active_idx)
        for idx, val in zip(self.active_idx, values):
            assert len(idx) == len(val)
            for i, v in zip(idx, val):
                exprs.append(
                    i.in_(v)
                    if isinstance(v, list)
                    else i.between(v.start, v.stop) if isinstance(v, slice) else i == v
                )

        return reduce(sqla.and_, exprs)

    def _is_singl_idx_type(
        self, values: Sequence[Sequence[slice | list[Hashable] | Hashable]]
    ) -> bool:
        return all(
            isinstance(v, i)
            for val, idx in zip(values, self.idx_type)
            for v, i in zip(val, idx)
        )

    def _parse_key(self, key: Any) -> tuple:
        sel = self.selection
        filt = self.filters
        idx = self.index

        match key:
            case PropRef():
                # Property selection.
                if isinstance(sel, type):
                    assert key.rec_type == sel
                    sel = key
                elif isinstance(sel, RelRef):
                    sel /= key
            case tuple():
                # Selection by tuple of index values.
                values = [list(v) if isinstance(v, tuple) else [v] for v in key]
                expr = self._gen_idx_match_expr(values)
                if expr is not None:
                    filt.append(expr)
                if self._is_singl_idx_type(values):
                    idx = None
            case list() | slice():
                # Selection by list or slice of index values.
                expr = self._gen_idx_match_expr([[key]])
                if expr is not None:
                    filt.append(expr)
            case Hashable():
                # Selection by single index value.
                expr = self._gen_idx_match_expr([[key]])
                if expr is not None:
                    filt.append(expr)
                idx = None
            case sqla.ColumnElement():
                # Filtering via SQL expression.
                filt.append(key)
            case _:
                raise TypeError("Unsupported key type.")

        return sel, filt, idx
