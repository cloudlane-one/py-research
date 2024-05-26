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

import pandas as pd
import polars as pl
import sqlalchemy as sqla
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
    AttrRef,
    BaseIdx,
    FilteredIdx,
    Idx,
    Key,
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
class DataBase(Generic[Name]):
    """Active connection to a SQL server."""

    backend: Backend[Name]
    schema: (
        type[Record | Schema]
        | Required[Record | Schema]
        | set[type[Record | Schema] | Required[Record | Schema]]
        | Mapping[
            type[Record] | Required[Record],
            str | sqla.TableClause,
        ]
        | Mapping[
            type[Schema] | Required[Schema],
            str,
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

        if has_type(
            self.schema,
            Mapping[type[Record] | Required[Record], str | sqla.TableClause],
        ):
            subs = {
                (rec.type if isinstance(rec, Required) else rec): (
                    sub if isinstance(sub, sqla.TableClause) else sqla.table(sub)
                )
                for rec, sub in self.schema.items()
            }

        if has_type(self.schema, Mapping[type[Schema] | Required[Schema], str]):
            subs = {
                rec: sqla.table(rec._table_name, schema=schema_name)
                for schema, schema_name in self.schema.items()
                for rec in (
                    schema.type if isinstance(schema, Required) else schema
                )._record_types
            }

        return {**subs, **self._overlay_subs}

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
        return DataSet(self, table, selection=key)

    def __setitem__(  # noqa: D105
        self,
        key: type[Rec],
        value: "DataSet[Name, Rec, Key] | RecInput[Rec, Key] | sqla.Select[tuple[Rec]]",  # noqa: E501
    ) -> None:
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
            select = (
                value
                if isinstance(value, sqla.Select)
                else sqla.select(value.base_table)
            )

            # Transfer table or query results to SQL table
            with self.engine.begin() as con:
                con.execute(sqla.delete(table))
                con.execute(sqla.insert(table).from_select(table.c.keys(), select))

    def __delitem__(self, key: type[Rec]) -> None:  # noqa: D105
        if key not in self._overlay_subs:
            self._overlay_subs[key] = self._get_table(key)

        table = self._ensure_table_exists(self._get_table(key))
        with self.engine.begin() as con:
            table.drop(con)

    def dataset(
        self,
        data: RecInput[Rec, Any],
        record_type: type[Rec] | None = None,
        foreign_keys: Mapping[str, AttrRef] | None = None,
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

        if self.backend.type == "excel-file":
            self._load_from_excel()

        for rec in self.record_types:
            ds = self[rec]
            ds.load(kind=pl.DataFrame).write_database(
                str(ds.base_table), str(other_db.backend.url)
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

    def _cols_from_df(
        self, df: DataFrame, foreign_keys: Mapping[str, AttrRef] | None = None
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


Join: TypeAlias = tuple[sqla.FromClause, sqla.ColumnElement[bool]]


@dataclass(frozen=True)
class DataSet(Generic[Name, Rec_cov, Idx_cov]):
    """Dataset selection."""

    db: DataBase[Name]
    base_table: sqla.Table
    base_joins: list[Join] = field(default_factory=list)
    selection: type[Rec_cov] | PropRef[Rec_cov, Any] | None = None
    merges: RelMerge = RelMerge()
    filters: list[sqla.ColumnElement[bool]] = field(default_factory=list)
    keys: Sequence[slice | list[Hashable] | Hashable | sqla.ColumnElement] = field(
        default_factory=list
    )

    @cached_property
    def base_set(self) -> "DataSet[Name, Rec_cov, Idx_cov]":
        """Base dataset for this dataset."""
        return DataSet(self.db, self.base_table)

    def _get_alias(self, relref: RelRef) -> sqla.FromClause:
        """Get alias for a relation reference."""
        return self.db._get_table(relref.val_type).alias(gen_str_hash(relref, 8))

    def _get_random_alias(self, rec: type[Record]) -> sqla.FromClause:
        """Get random alias for a type."""
        return self.db._get_table(rec).alias(token_hex(4))

    def _joins_from_tree(self, tree: RelTree) -> list[Join]:
        """Extract join operations from a relation tree."""
        joins = []

        for r, subtree in tree.items():
            parent = (
                self._get_alias(r.parent) if r.parent is not None else self.base_table
            )

            inter_alias_map = {
                rec: self._get_random_alias(rec) for rec in r.inter_joins.keys()
            }

            joins.extend(
                (
                    inter_alias_map[rec],
                    reduce(
                        sqla.or_,
                        (
                            reduce(
                                sqla.and_,
                                (
                                    parent.c[lk.name] == inter_alias_map[rec].c[rk.name]
                                    for lk, rk in inter_join_on.items()
                                ),
                            )
                            for inter_join_on in inter_join_ons
                        ),
                    ),
                )
                for rec, inter_join_ons in r.inter_joins.items()
            )

            target_table = self._get_alias(r)

            joins.append(
                (
                    target_table,
                    reduce(
                        sqla.or_,
                        (
                            reduce(
                                sqla.and_,
                                (
                                    inter_alias_map[lk.rec_type].c[lk.name]
                                    == target_table.c[rk.name]
                                    for lk, rk in join_on.items()
                                ),
                            )
                            for join_on in r.join_ons
                        ),
                    ),
                )
            )

            joins.extend(self._joins_from_tree(subtree))

        return joins

    def joins(self, extra_merges: RelMerge = RelMerge()) -> list[Join]:
        """List of all join operations required to construct this tree."""
        return [
            *self.base_joins,
            *self._joins_from_tree((self.merges + extra_merges).rel_tree),
        ]

    @cached_property
    def main_idx(self) -> list[AttrRef[Rec_cov, Any]] | None:
        """Main index of this dataset."""
        match self.selection:
            case type():
                return self.selection._primary_keys()
            case RelRef():
                assert issubclass(self.selection.val_type, Record)
                return self.selection.val_type._primary_keys()
            case PropRef():
                return self.selection.rec_type._primary_keys()
            case None:
                return None

    @cached_property
    def path_idx(self) -> list[AttrRef] | None:
        """Optional, alternative index of this dataset based on merge path."""
        path = self.selection.path if isinstance(self.selection, PropRef) else []

        if any(relref.idx_attr is None for relref in path):
            return None

        return [relref.idx_attr for relref in path if relref.idx_attr is not None]

    # Overloads: attribute selection:

    # 1. Top-level attribute selection, relational index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Idx]",
        key: AttrRef[Rec_cov, Val],
    ) -> "DataSet[Name, Record[Val, Key2], Idx]": ...

    # 2. Nested attribute selection, base index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], BaseIdx]", key: AttrRef[Any, Val]
    ) -> "DataSet[Name, Record[Val, Any], IdxStart[Key2]]": ...

    # 3. Nested attribute selection, relational index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]", key: AttrRef[Any, Val]
    ) -> "DataSet[Name, Record[Val, Any], IdxStart[Key | Key2]]": ...

    # Overloads: relation selection:

    # 4. Top-level relation selection, singular
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: RelRef[Rec_cov, Rec2, SingleIdx],
    ) -> "DataSet[Name, Rec2, Key | Key2]": ...

    # 5. Top-level relation selection, no index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: RelRef[Rec_cov, Rec2, BaseIdx],
    ) -> "DataSet[Name, Rec2, FilteredIdx]": ...

    # 6. Top-level relation selection, indexed, tuple case
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup] | BaseIdx]",
        key: RelRef[Rec_cov, Rec2, Key3],
    ) -> "DataSet[Name, Rec2, tuple[*IdxTup, Key3] | tuple[Key2, Key3]]": ...

    # 7. Top-level relation selection, indexed
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]",
        key: RelRef[Rec_cov, Rec2, Key3],
    ) -> "DataSet[Name, Rec2, tuple[Key | Key2, Key3]]": ...

    # 8. Nested relation selection, tuple case
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup]]",
        key: RelRef[Any, Rec2, Key3],  # noqa: E501
    ) -> (
        "DataSet[Name, Rec2, IdxTupStartEnd[*IdxTup, Key3] | IdxStartEnd[Key2, Key3]]"
    ): ...

    # 9. Nested relation selection
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]", key: RelRef[Any, Rec2, Key3]
    ) -> "DataSet[Name, Rec2, IdxStartEnd[Key | Key2, Key3]]": ...

    # Overloads: index list selection:

    # 10. List selection, mark base index as filtered
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Key2], BaseIdx]", key: list[Key2]
    ) -> "DataSet[Name, Rec_cov, FilteredIdx]": ...

    # 11. List selection, keep index
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Key2], Key]", key: list[Key | Key2]
    ) -> "DataSet[Name, Rec_cov, Key]": ...

    # 12. Index value selection:
    @overload
    def __getitem__(
        self: "DataSet[Name, Record[Val, Key2], Key]", key: Key | Key2
    ) -> "DataSet[Name, Rec_cov, SingleIdx]": ...

    # Overloads: index slice selection:

    # 13. Slice selection, mark base index as filtered
    @overload
    def __getitem__(
        self: "DataSet[Name, Rec_cov, BaseIdx]",
        key: slice | tuple[slice, ...],
    ) -> "DataSet[Name, Rec_cov, FilteredIdx]": ...

    # 14. Slice selection, keep index
    @overload
    def __getitem__(self, key: slice | tuple[slice, ...]) -> Self: ...

    # Overloads: filtering:

    # 15. Expression filtering, mark base index as filtered
    @overload
    def __getitem__(
        self: "DataSet[Name, Rec_cov, BaseIdx]",
        key: sqla.ColumnElement[bool] | pd.Series[bool],
    ) -> "DataSet[Name, Rec_cov, FilteredIdx]": ...

    # 16. Expression filtering, keep index
    @overload
    def __getitem__(self, key: sqla.ColumnElement[bool] | pd.Series[bool]) -> Self: ...

    # Implementation:

    def __getitem__(  # noqa: D105
        self,
        key: (
            AttrRef[Any, Val]
            | RelRef
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
            self.base_table,
            self.base_joins + [join] if isinstance(join, tuple) else self.base_joins,
            (
                self.selection >> key
                if isinstance(self.selection, type | RelRef)
                and isinstance(key, PropRef)
                else self.selection
            ),
            self.merges + join if isinstance(join, RelMerge) else self.merges,
            self.filters + [filt] if filt is not None else self.filters,
            keys if keys is not None else self.keys,
        )

    def _parse_merge(self, merge: RelRef | RelMerge | None) -> RelMerge:
        """Parse merge argument and prefix with current selection."""
        merge = (
            merge
            if isinstance(merge, RelMerge)
            else RelMerge([merge]) if merge is not None else RelMerge()
        )
        return (
            self.selection >> merge
            if isinstance(self.selection, RelRef)
            else (
                self.selection.path[-1] >> merge
                if isinstance(self.selection, PropRef)
                else merge
            )
        )

    @overload
    def select(self, merge: None = ...) -> sqla.Select[tuple[Rec_cov]]: ...

    @overload
    def select(
        self, merge: RelRef[Any, Rec, Any]
    ) -> sqla.Select[tuple[Rec_cov, Rec]]: ...

    @overload
    def select(
        self, merge: RelMerge[*MergeTup]
    ) -> sqla.Select[tuple[Rec_cov, *MergeTup]]: ...

    def select(
        self,
        merge: RelRef | RelMerge | None = None,
        index_only: bool = False,
    ) -> sqla.Select:
        """Return select statement for this dataset."""
        abs_merge = self._parse_merge(merge)

        select = sqla.select(
            *(
                (
                    col.label(f"{relref.path_str}.{col_name}")
                    for relref in abs_merge.rels
                    for col_name, col in self._get_alias(relref).columns.items()
                )
                if not index_only
                else (
                    (
                        self._get_alias(relref)
                        .c[attr.name]
                        .label(f"{relref.path_str}.{attr.name}")
                        for relref in abs_merge.rels
                        for attr in idx_attrs
                    )
                    if (idx_attrs := self.main_idx) is not None
                    else (
                        self._get_alias(relref)
                        .c[pk.name]
                        .label(f"{relref.path_str}.{pk.name}")
                        for relref in abs_merge.rels
                        for pk in self.base_table.primary_key.columns
                    )
                )
            )
        ).select_from(self.base_table)

        for join in self.joins(extra_merges=abs_merge):
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
        merge: RelRef[Any, Rec, Any],
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
    def load(self, merge: RelRef | RelMerge, kind: type[Df]) -> tuple[Df, ...]: ...

    @overload
    def load(
        self, merge: None = ..., kind: type[Record] = ...
    ) -> Sequence[Rec_cov]: ...

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

    def _set(  # noqa: D105
        self,
        value: "DataSet | RecInput[Rec_cov, Any] | ValInput",
    ) -> None:
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

        # Derive current select statement and join with value table, if exists.
        select = self.select()
        if value_set is not None:
            select = select.join(
                value_set.base_table,
                reduce(
                    sqla.and_,
                    (
                        idx_col == self.base_table.c[idx_col.name]
                        for idx_col in value_set.base_table.primary_key.columns
                    ),
                ),
            )

        single_attr = (
            self.selection.name if isinstance(self.selection, AttrRef) else None
        )

        # Prepare update statement.
        if self.db.engine.dialect.name in ("postgresql", "duckdb", "mysql", "mariadb"):
            # Update-from.
            update_stmt = (
                self.base_table.update()
                .values(
                    {col.name: col for col in value_set.base_table.columns}
                    if value_set is not None
                    else {single_attr: value}
                )
                .where(
                    reduce(
                        sqla.and_,
                        (
                            self.base_table.c[col.name] == select.c[col.name]
                            for col in self.base_table.primary_key.columns
                        ),
                    )
                )
            )
        else:
            # Correlated update.
            raise NotImplementedError("Correlated update not supported yet.")

        # Execute update statement.
        with self.db.engine.begin() as con:
            con.execute(update_stmt)

        # 4. Drop the temporary table, if any.
        if value_set is not None:
            value_set.base_table.drop(self.db.engine)

    # 1. Top-level attribute assignment
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Key2], Key | BaseIdx]",
        key: AttrRef[Rec_cov, Val],
        value: "DataSet[Name, Record[Val, Key | Key2], BaseIdx] | DataSet[Name, Record[Val, Any], Key | Key2] | ValInput[Val, Key | Key2]",  # noqa: E501
    ) -> None: ...

    # 2. Nested attribute assignment
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Val, Key2], Key | BaseIdx]",
        key: AttrRef[Any, Val],
        value: "DataSet[Name, Record[Val, IdxStart[Key | Key2]], BaseIdx] | DataSet[Name, Record[Val, Any], IdxStart[Key | Key2]] | ValInput[Val, IdxStart[Key | Key2]]",  # noqa: E501
    ) -> None: ...

    # 3. Top-level relation assignment, singular
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key]",
        key: RelRef[Rec_cov, Rec2, SingleIdx],
        value: "DataSet[Name, Rec2, Key | Key2] | RecInput[Rec2, Key | Key2]",
    ) -> None: ...

    # 4. Top-level relation assignment, no index
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Any]",
        key: RelRef[Rec_cov, Rec2, BaseIdx],
        value: "DataSet[Name, Rec2, BaseIdx] | RecInput[Rec2, Any]",
    ) -> None: ...

    # 5. Top-level relation assignment, indexed, tuple case
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup] | BaseIdx]",
        key: RelRef[Rec_cov, Rec2, Key3],
        value: "DataSet[Name, Rec2, tuple[*IdxTup, Key3] | tuple[Key2, Key3]] | RecInput[Rec2, tuple[*IdxTup, Key3] | tuple[Key2, Key3]]",  # noqa: E501
    ) -> None: ...

    # 6. Top-level relation assignment, indexed
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: RelRef[Rec_cov, Rec2, Key3],
        value: "DataSet[Name, Rec2, tuple[Key | Key2, Key3]] | RecInput[Rec2, tuple[Key | Key2, Key3]]",  # noqa: E501
    ) -> None: ...

    # 7. Nested relation assignment, tuple case
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], tuple[*IdxTup] | BaseIdx]",
        key: RelRef[Any, Rec2, Key3],
        value: "DataSet[Name, Rec2, IdxTupStartEnd[*IdxTup, Key3] | IdxStartEnd[Key2, Key3]] | RecInput[Rec2, IdxTupStartEnd[*IdxTup, Key3] | IdxStartEnd[Key2, Key3]]",  # noqa: E501
    ) -> None: ...

    # 8. Nested relation assignment
    @overload
    def __setitem__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        key: RelRef[Any, Rec2, Key3],
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
            AttrRef
            | RelRef
            | list[Hashable]
            | Hashable
            | slice
            | tuple[slice | tuple[slice, ...], ...]
            | sqla.ColumnElement[bool]
            | pd.Series[bool]
        ),
        value: "DataSet | RecInput[Rec_cov, Any] | ValInput",
    ) -> None:
        if (
            has_type(value, Record)
            or has_type(value, Iterable[Record])
            or has_type(value, Mapping[Any, Record])
        ):
            raise NotImplementedError("Inserting record instances not supported yet.")

        if isinstance(key, list):
            # Ensure that index is alignable.

            if isinstance(value, Sized):
                assert len(value) == len(key), "Length mismatch."

            if isinstance(value, pd.DataFrame):
                assert set(value.index.to_list()) == set(key), "Index mismatch."
            elif isinstance(value, Mapping):
                assert set(value.keys()) == set(key), "Index mismatch."

        cast(Self, self[key])._set(value)  # type: ignore

    @overload
    def __ilshift__(
        self: "DataSet[Name, Rec_cov, FilteredIdx | Key]",
        value: None,  # Insertion not allowed for filtered or relational indexes.
    ) -> None: ...

    @overload
    def __ilshift__(
        self: "DataSet[Name, Record[Any, Key2], Key | BaseIdx]",
        value: "DataSet[Name, Rec_cov, Key | BaseIdx] | RecInput[Rec_cov, Key | Key2]",
    ) -> None: ...

    @overload
    def __ilshift__(
        self: "DataSet[Name, Record[Val, Key2], Key | BaseIdx]",
        value: "DataSet[Name, Record[Val, Key | Key2], BaseIdx] | DataSet[Name, Record[Val, Any], Key | Key2] | ValInput[Val, Key | Key2]",  # noqa: E501
    ) -> None: ...

    def __ilshift__(
        self,
        value: "DataSet | RecInput[Rec_cov, Any] | ValInput | None",
    ) -> None:
        """Merge update into unfiltered dataset."""
        # Do an insert-from-select operation, which updates on conflict.
        # For Postgres / DuckDB, use: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#updating-using-the-excluded-insert-values
        # For MySQL / MariaDB, use: https://docs.sqlalchemy.org/en/20/dialects/mysql.html#insert-on-duplicate-key-update-upsert
        # For others, use CTE: https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.cte
        raise NotImplementedError()

    def extract(
        self, aggs: Mapping[RelRef[Any, Rec, Any], Agg[Rec, Any]] | None = None
    ) -> DataBase[Name]:
        """Extract a new database from the current selection."""
        # For now implement non-recursively.
        # This means that given another table in the schema,
        # joins along all direct routes from this selection to the table
        # are performed. Their results are then unioned and distincted.
        # Empty results are filtered out.
        # Functionality to whitelist certain recursive loops can be added later.
        raise NotImplementedError()

    def __clause_element__(self) -> sqla.Subquery:
        """Return subquery for the current selection to be used inside SQL clauses."""
        return self.select().subquery()

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

    def _replace_attrref(
        self,
        element: sqla_visitors.ExternallyTraversible,
        reflist: list[RelRef] = [],
        **kw: Any,
    ) -> sqla.ColumnElement | None:
        if isinstance(element, AttrRef):
            if element.parent is not None:
                reflist.append(element.parent)
            return self.db._get_table(element.rec_type).c[element.name]

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
                reflist = []
                replace_func = partial(self._replace_attrref, reflist=reflist)
                filt = sqla_visitors.replacement_traverse(key, {}, replace=replace_func)
                merge += RelMerge(reflist)
            case pd.Series():
                # Filtering via boolean Series.

                series_name = token_hex(8)
                uploaded = self.db.dataset(key.rename(series_name).to_frame())
                assert all(
                    n1 == n2
                    for n1, n2 in zip(
                        key.index.names, self.base_table.primary_key.columns
                    )
                )

                filt = uploaded.base_table.c[series_name] == True  # noqa: E712
                join = (
                    uploaded.base_table,
                    reduce(
                        sqla.and_,
                        (
                            self.base_table.c[pk] == uploaded.base_table.c[pk]
                            for pk in key.index.names
                        ),
                    ),
                )

        return filt, join or merge
