"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Generator, Hashable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cmp_to_key, reduce
from io import BytesIO
from operator import attrgetter
from pathlib import Path
from secrets import token_hex
from sqlite3 import PARSE_DECLTYPES
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    LiteralString,
    Self,
    cast,
    final,
    overload,
)

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.orm as orm
import sqlalchemy.sql.type_api as sqla_type
import yarl
from cloudpathlib import CloudPath
from typing_extensions import TypeVar
from xlsxwriter import Workbook as ExcelWorkbook

from py_research.caching import cached_method, cached_prop
from py_research.data import copy_and_override
from py_research.files import HttpFile
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.types import has_type
from py_research.types import UUID4, Not.defined

from .props import CRU, CRUD, Attr, C, RwxT, D, Model, Prop, R, U, Ungot, Unset
from .sql_utils import coalescent_join_sql, safe_delete, safe_insert, safe_update
from .utils import pl_type_map, register_sqlite_adapters, remove_cross_fk

register_sqlite_adapters()

OwnT = TypeVar("OwnT", bound="Record", contravariant=True, default=Any)

RecT = TypeVar("RecT", bound="Record", covariant=True, default=Any)
RecT2 = TypeVar("RecT2", bound="Record")

type TableInput[T: Record] = pd.DataFrame | pl.DataFrame | Iterable[T] | sqla.FromClause

sqla.schema.ColumnCollectionMixin

@final
class Symbolic:
    """Local backend."""


type DynBackendID = LiteralString | None


BackT = TypeVar(
    "BackT",
    bound=DynBackendID | Symbolic,
    covariant=True,
    default=Symbolic,
)

DbT = TypeVar(
    "DbT",
    bound=DynBackendID,
    covariant=True,
    default=None,
)


type OverlayType = Literal["transaction", "name_prefix", "db_schema"]


type MutationMode = Literal["update", "upsert", "replace", "insert", "delete"]


@dataclass
class Require:
    """Mark schema or record type as required."""

    present: bool = True


def get_pl_schema(
    col_map: Mapping[str, Column]
) -> dict[str, pl.DataType | type | None]:
    """Return the schema of the dataset."""
    exact_matches = {
        name: (pl_type_map.get(col.value_typeform), col)
        for name, col in col_map.items()
    }
    matches = {
        name: (
            (match, col.value_typeform)
            if match is not None
            else (pl_type_map.get(col.common_value_type), col.value_typeform)
        )
        for name, (match, col) in exact_matches.items()
    }

    return {
        name: (
            match if isinstance(match, pl.DataType | type | None) else match(match_type)
        )
        for name, (match, match_type) in matches.items()
    }


def records_to_df(
    records: Iterable[dict[str, Any] | Record],
) -> pl.DataFrame:
    """Convert values to a DataFrame."""
    model_types: set[type[Record]] = {
        type(val) for val in records if isinstance(val, Record)
    }
    df_data = [
        (
            val._to_dict(keys="names" if len(model_types) == 1 else "fqns")
            if isinstance(val, Record)
            else val
        )
        for val in records
    ]

    return pl.DataFrame(
        df_data,
        schema=get_pl_schema(
            {
                col.name if len(model_types) == 1 else col.fqn: col
                for mod in model_types
                for col in mod._cols()
            }
        ),
    )


def validate_sql_table(
    con: sqla.Engine | sqla.Inspector, table: sqla.Table, required: bool = False
) -> None:
    """Perform pre-defined schema validations."""
    inspector = con if isinstance(con, sqla.Inspector) else sqla.inspect(con)
    has_table = inspector.has_table(table.name, table.schema)

    if not has_table and not required:
        return

    # Check if table exists
    assert has_table

    db_columns = {c["name"]: c for c in inspector.get_columns(table.name, table.schema)}
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
                (db_fk["referred_table"].lower() == fk.referred_table.name.lower()),
                set(db_fk["referred_columns"])
                == set(f.column.name for f in fk.elements),
            )
            for db_fk in db_fks
        ]

        assert any(all(m) for m in matches)


BaseT = TypeVar("BaseT", bound="Record", contravariant=True)
BaseT2 = TypeVar("BaseT2", bound="Record")


@dataclass(eq=False)
class Base(Generic[BaseT, BackT, RwxT]):
    """Database connection."""

    backend: BackT = None  # pyright: ignore[reportAssignmentType]
    """Unique name to identify this database's backend by."""
    url: sqla.URL | CloudPath | HttpFile | Path | None = None
    """Connection URL or path."""

    schema: (
        type[BaseT]
        | Mapping[
            type[BaseT],
            Literal[True] | Require | str | sqla.TableClause,
        ]
        | set[type[BaseT]]
        | None
    ) = None
    validate_on_init: bool = False

    overlay: Literal[True] | str | None = None
    overlay_type: OverlayType = "transaction"

    remove_cross_fks: bool = False

    _rec_types: dict[type[Record], Literal[True] | Require | str | sqla.TableClause] = (
        field(default_factory=dict)
    )
    _subs: dict[type[Record], sqla.TableClause] = field(default_factory=dict)

    _metadata: sqla.MetaData = field(default_factory=sqla.MetaData)
    _valid_caches: dict[type[Record], set[Hashable]] = field(default_factory=dict)
    _instance_map: dict[type[Record], dict[Hashable, Record]] = field(
        default_factory=dict
    )

    _db_id: str | None = None
    _overlay_name: str | None = None
    _transaction: sqla.Transaction | None = None

    def __post_init__(self):  # noqa: D105
        tables: dict[type[Record], Literal[True] | Require | str | sqla.TableClause] = {
            tab: tab_def
            for tab, tab_def in self._schema_map.items()
            if issubclass(tab, Record)
        }

        # Handle Record classes in schema argument.
        self._subs = {
            **self._subs,
            **{
                tab: (sub if isinstance(sub, sqla.TableClause) else sqla.table(sub))
                for tab, sub in tables.items()
                if not isinstance(sub, Require | bool)
            },
        }
        self._rec_types |= tables

        self._overlay_name = (
            f"{self.db_id}_overlay_{token_hex(4)}"
            if self.overlay is True
            else self.overlay
        )

        if self._overlay_name is not None and self.overlay_type == "db_schema":
            self._ensure_sql_schema_exists(self._overlay_name)

        if self._overlay_name is not None and self.overlay_type == "transaction":
            self._transaction = sqla.Transaction(self.engine.connect())

        if not isinstance(self.backend, Symbolic) and self.validate_on_init:
            cast(Base[Any, DynBackendID, Any], self).validate()

    @property
    def db_id(self) -> str:
        """Return the unique database ID."""
        db_id = self._db_id
        if db_id is None:
            if isinstance(self.backend, Symbolic):
                db_id = "symbolic"
            elif self.backend is None:
                db_id = token_hex(4)
            else:
                db_id = self.backend + (
                    gen_str_hash(self.url) if self.url is not None else ""
                )

        self._db_id = db_id
        return db_id

    @cached_prop
    def backend_type(
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

    @cached_prop
    def engine(self) -> sqla.engine.Engine:
        """SQLA Engine for this DB."""
        # Create engine based on backend type
        # For Excel-backends, use sqlite in-memory engine
        return sqla.create_engine(
            (
                (self.url if isinstance(self.url, sqla.URL) else str(self.url))
                if self.backend_type in ("sql-connection", "sqlite-file")
                else f"sqlite:///file:{self.db_id}?uri=true&mode=memory&cache=shared"
            ),
        )

    @cached_prop
    def df_engine(self) -> sqla.engine.Engine:
        """SQLA Engine for this DB."""
        # Create engine based on backend type
        # For Excel-backends, use sqlite in-memory engine
        return sqla.create_engine(
            (
                (self.url if isinstance(self.url, sqla.URL) else str(self.url))
                if self.backend_type in ("sql-connection", "sqlite-file")
                else f"sqlite:///file:{self.db_id}?uri=true&mode=memory&cache=shared"
            ),
            connect_args=(
                {"detect_types": PARSE_DECLTYPES}
                if self.backend_type in ("sqlite-file", "in-memory", "excel-file")
                else {}
            ),
        )

    @property
    def connection(self) -> sqla.engine.Connection:
        """SQLA Connection for this DB."""
        return (
            self._transaction.connection
            if self._transaction is not None
            else self.engine.connect()
        )

    @overload
    def table(
        self: Base[BaseT2, DynBackendID, CRU],
        rec_types: type[BaseT2] | set[type[BaseT2]],
        input_data: TableInput[BaseT2],
    ) -> Table[BaseT2, BackT, R]: ...

    @overload
    def table(
        self: Base[BaseT2, Any, Any],
        rec_types: type[BaseT2] | set[type[BaseT2]],
        input_data: None = ...,
    ) -> Table[BaseT2, BackT, RwxT]: ...

    def table(
        self,
        rec_types: type[RecT2] | set[type[RecT2]],
        input_data: TableInput[RecT2] | None = None,
    ) -> Table[RecT2, BackT, Any]:
        """Create a table from input data."""
        return Table(
            self, rec_types if isinstance(rec_types, set) else {rec_types}, input_data
        )

    def validate(self: Base[Any, DynBackendID, Any]) -> None:
        """Perform pre-defined schema validations."""
        recs = {rec: isinstance(req, Require) for rec, req in self._rec_types.items()}
        inspector = sqla.inspect(self.engine)

        # Iterate over all tables and perform validations for each
        for rec, req in recs.items():
            self.table({rec}).validate(inspector, required=req)

    def commit(self) -> None:
        """Commit the transaction."""
        if self._transaction is not None:
            self._transaction.commit()
            self._transaction = None
        # Note: Commiting overlays of other types than "transaction"
        # is not supported yet.

    @contextmanager
    def edit(
        self: Base[BaseT, BackT, C | U | D], overlay_type: OverlayType = "transaction"
    ) -> Generator[Base[BaseT, BackT, RwxT]]:
        """Context manager to create temp overlay of base and auto-commit on exit."""
        assert self.overlay is None, "Cannot edit base with already active overlay."
        edit_base = copy_and_override(
            Base[BaseT, BackT, RwxT], self, overlay=True, backend=self.backend
        )

        try:
            yield edit_base
        finally:
            edit_base.commit()

    @cached_prop
    def _schema_map(
        self,
    ) -> Mapping[type[BaseT], Literal[True] | Require | str | sqla.TableClause]:
        return (
            {self.schema: True}
            if isinstance(self.schema, type)
            else (
                cast(
                    dict[type[BaseT], Literal[True]],
                    {rec: True for rec in self.schema},
                )
                if isinstance(self.schema, set)
                else self.schema if self.schema is not None else {}
            )
        )

    @cached_method
    def _get_base_table_name(
        self,
        rec_type: type[Record],
    ) -> str:
        """Return the name of the table for this class."""
        sub = self._subs.get(rec_type)
        return sub.name if sub is not None else rec_type._default_table_name()

    def _get_base_table(
        self,
        rec_type: type[Record],
        mode: Literal["read"] | MutationMode = "read",
    ) -> sqla.Table:
        """Return the base SQLAlchemy table object for this data's record type."""
        orig_table: sqla.Table | None = None

        if (
            mode != "read"
            and self._overlay_name is not None
            and rec_type not in self._subs
        ):
            orig_table = self._get_base_table(rec_type, "read")

            # Create an empty overlay table for the record type
            self._subs[rec_type] = sqla.table(
                (
                    (self._overlay_name + "/" + rec_type._default_table_name())
                    if self.overlay_type == "name_prefix"
                    else rec_type._default_table_name()
                ),
                schema=(
                    self._overlay_name if self.overlay_type == "db_schema" else None
                ),
            )

        table_name = self._get_base_table_name(rec_type)

        if table_name in self._metadata.tables:
            # Return the table object from metadata if it already exists.
            # This is necessary to avoid circular dependencies.
            return self._metadata.tables[table_name]

        sub = self._subs.get(rec_type)

        # Create a partial SQLAlchemy table object from the class definition
        # without foreign keys to avoid circular dependencies.
        # This adds the table to the metadata.
        sqla.Table(
            table_name,
            self._metadata,
            *self._map_cols(rec_type).values(),
            self._map_pk(rec_type),
            *self._map_keys(rec_type).values(),
            schema=(sub.schema if sub is not None else None),
        )

        # Re-create the table object with foreign keys and return it.
        table = sqla.Table(
            table_name,
            self._metadata,
            *self._map_fks(rec_type).values(),
            schema=(sub.schema if sub is not None else None),
            extend_existing=True,
        )

        self._create_sql_table(table)

        if orig_table is not None and mode == "upsert":
            with self.engine.begin() as conn:
                conn.execute(
                    sqla.insert(table).from_select(
                        orig_table.columns.keys(), orig_table.select()
                    )
                )

        if self.backend_type == "excel-file":
            self._load_base_table_from_excel(table)

        return table

    @cached_method
    def _map_cols(self, rec_type: type[Record]) -> dict[Column, sqla.Column]:
        """Columns of this record type's table."""
        registry = orm.registry(
            metadata=self._metadata,
            type_annotation_map=rec_type._type_map,
        )

        return {
            col: sqla.Column(
                col.name,
                registry._resolve_type(
                    col.value_typeform  # pyright: ignore[reportArgumentType]
                ),
                autoincrement=False,
                nullable=has_type(None, col.value_typeform),
            )
            for col in rec_type._cols()
        }

    @cached_method
    def _map_keys(self, rec_type: type[Record]) -> dict[Key, sqla.Index]:
        """Columns of this record type's table."""
        return {
            k: sqla.Index(
                f"{self._get_base_table_name(rec_type)}_key_{k.name}",
                *(col.name for col in k.columns),
                unique=True,
            )
            for k in rec_type._keys()
        }

    @cached_method
    def _map_pk(self, rec_type: type[Record]) -> sqla.PrimaryKeyConstraint:
        """Columns of this record type's table."""
        pk = rec_type._pk  # type: ignore
        return sqla.PrimaryKeyConstraint(
            *[col.name for col in pk.columns],
        )

    @cached_method
    def _map_fks(
        self, rec_type: type[Record]
    ) -> dict[ForeignKey, sqla.ForeignKeyConstraint]:
        fks: dict[ForeignKey, sqla.ForeignKeyConstraint] = {}

        for fk in rec_type._fks():
            target_type = cast(type[Record], fk.target)
            target_table = self._get_base_table(target_type)

            fks[fk] = sqla.ForeignKeyConstraint(
                [col.name for col in fk.column_map.keys()],
                [target_table.c[col.name] for col in fk.column_map.values()],
                name=f"{self._get_base_table_name(rec_type)}_fk_{fk.name}_{target_type._fqn}",
            )

        return fks

    def _get_valid_cache_set(self, rec_type: type[Record]) -> set[Hashable]:
        """Get the valid cache set for a record type."""
        if rec_type not in self._valid_caches:
            self._valid_caches[rec_type] = set()

        return self._valid_caches[rec_type]

    def _get_instance_map(self, rec_type: type[Record]) -> dict[Hashable, Record]:
        """Get the instance map for a record type."""
        if rec_type not in self._instance_map:
            self._instance_map[rec_type] = {}

        return self._instance_map[rec_type]

    def _load_base_table_from_excel(self, table: sqla.Table) -> None:
        """Load all tables from Excel."""
        assert isinstance(self.url, Path | CloudPath | HttpFile)
        path = self.url.get() if isinstance(self.url, HttpFile) else self.url

        if isinstance(path, Path) and not path.exists():
            return

        with self.engine.begin() as conn:
            conn.execute(table.delete().where(sqla.true()))

        with open(path, "rb") as file:
            df = pl.read_excel(file, sheet_name=table.name)

        df.write_database(
            str(table),
            self.df_engine,
            if_table_exists="append",
        )

    def _create_sql_table(self, table: sqla.Table) -> None:
        """Create SQL-side table from Table class."""
        if self.remove_cross_fks:
            # Create a temporary copy of the table object and remove external FKs.
            # That way, local metadata will retain info on the FKs
            # (for automatic joins) but the FKs won't be created in the DB.
            table = table.to_metadata(sqla.MetaData())  # temporary metadata
            remove_cross_fk(table)

        table.create(self.engine, checkfirst=True)

    def _ensure_sql_schema_exists(self, schema_name: str) -> str:
        """Ensure that the table exists in the database, then return it."""
        if not sqla.inspect(self.engine).has_schema(schema_name):
            with self.engine.begin() as conn:
                conn.execute(sqla.schema.CreateSchema(schema_name))

        return schema_name

    def _save_to_excel(self) -> None:
        """Save all (or selected) tables to Excel."""
        assert isinstance(self.url, Path | CloudPath | HttpFile)
        file = self.url.get() if isinstance(self.url, HttpFile) else self.url

        with ExcelWorkbook(file) as wb:
            for table in self._metadata.tables.values():
                pl.read_database(
                    table.select(),
                    self.engine.connect(),
                ).write_excel(wb, worksheet=table.name)

        if isinstance(self.url, HttpFile):
            assert isinstance(file, BytesIO)
            self.url.set(file)

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(
            (
                self.db_id,
                self.url,
                self._subs,
            )
        )


symbol_base = Base[Any, Symbolic](backend=Symbolic())
"""Base for all symbolic props."""


@dataclass
class Table(Generic[RecT, BackT, RwxT]):
    """Table matching a table model, may be filtered."""

    base: Base[Any, BackT]
    rec_types: set[type[RecT]]
    content: (
        str | sqla.FromClause | pd.DataFrame | pl.DataFrame | Iterable[Record] | None
    ) = None

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this table based on its record types."""
        return "|".join(rec._fqn for rec in self.rec_types)

    @cached_prop
    def sql(self) -> sqla.FromClause:
        """This table's from clause."""
        match self.content:
            case str():
                return sqla.table(self.content)
            case sqla.FromClause():
                return self.content
            case pd.DataFrame() | pl.DataFrame():
                return self._upload_df(self.content)
            case Iterable():
                return self._upload_df(records_to_df(self.content))
            case None:
                return self._get_base_union()

    def df(self: Table[Any, DynBackendID, Any]) -> pl.DataFrame:
        """Return the table as a DataFrame."""
        return pl.read_database(self.sql, self.base.connection)

    def mutate(
        self: Table[Any, Any, CRUD],
        input_table: Table[RecT2, BackT, Any],
        mode: MutationMode = "update",
    ) -> None:
        """Mutate given table."""
        tables = {rec: table for rec, (table, _) in self._base_table_map.items()}
        input_sql = input_table.sql

        statements: list[sqla.Executable] = []

        if mode in ("replace", "delete"):
            # Delete current records first.
            statements += [
                safe_delete(table, input_sql, self.base.engine)
                for table in tables.values()
            ]

        if mode != "delete":
            for base_type, table in tables.items():
                # Rename input columns to match base table columns.
                input_sql = input_sql.select().with_only_columns(
                    *(
                        input_sql.c[col.name].label(name)
                        for name, col in self._base_col_map[base_type].items()
                    )
                )

                if mode == "update":
                    statements.append(safe_update(table, input_sql, self.base.engine))
                else:
                    statements.append(
                        safe_insert(
                            table,
                            input_sql,
                            self.base.engine,
                            upsert=(mode == "upsert"),
                        )
                    )

        # Execute delete / insert / update statements.
        con = self.base.connection
        for statement in statements:
            con.execute(statement)

        if self.base.backend_type == "excel-file":
            self.base._save_to_excel(
                # {self.record_type, *self.record_type._record_superclasses}
            )

    def validate(
        self, inspector: sqla.Inspector | None = None, required: bool = False
    ) -> None:
        """Perform pre-defined schema validations."""
        inspector = inspector or sqla.inspect(self.base.engine)

        for table, _ in self._base_table_map.values():
            validate_sql_table(inspector, table, required)

    def __clause_element__(self) -> sqla.FromClause:
        """Return the table clause element."""
        return self.sql

    @cached_prop
    def _base_table_map(
        self,
    ) -> dict[type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]]:
        """Return the base SQLAlchemy table objects for all record types."""
        base_table_map: dict[
            type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]
        ] = {}
        for rec in self.rec_types:
            super_recs = (r for r in rec._model_superclasses if issubclass(r, Record))
            for base_rec in (rec, *super_recs):
                if base_rec not in base_table_map:
                    table = self.base._get_base_table(rec)
                    join_on = (
                        None
                        if base_rec is rec
                        else reduce(
                            sqla.and_,
                            (
                                pk_sel == pk_super
                                for pk_sel, pk_super in zip(
                                    self.base._map_pk(rec).columns,
                                    self.base._map_pk(base_rec).columns,
                                )
                            ),
                        )
                    )
                    base_table_map[base_rec] = (table, join_on)

        return dict(
            sorted(
                base_table_map.items(),
                key=cmp_to_key(
                    lambda x, y: (
                        -1
                        if issubclass(y[0], x[0])
                        else 1 if issubclass(x[0], y[0]) else 0
                    )
                ),
            )
        )

    @cached_prop
    def _base_col_map(self) -> dict[type[Record], dict[str, sqla.Column]]:
        """Return the column set for all record types."""
        name_attr = attrgetter("name" if len(self.rec_types) == 1 else "fqn")
        return {
            rec: {
                name_attr(col): sql_col
                for col, sql_col in self.base._map_cols(rec).items()
            }
            for rec in self._base_table_map.keys()
        }

    def _get_base_union(self) -> sqla.FromClause:
        """Recursively join all bases of this record to get the full data."""
        sel_tables = {
            rec: table
            for rec, (table, join_on) in self._base_table_map.items()
            if join_on is None
        }

        def _union(
            union_stage: tuple[type[Record] | None, sqla.Select],
            next_union: tuple[type[Record], sqla.Join | sqla.Table],
        ) -> tuple[type[Record] | None, sqla.Select]:
            left_rec, left_table = union_stage
            right_rec, right_table = next_union

            right_table = right_table.select().with_only_columns(
                *(
                    col.label(name)
                    for name, col in self._base_col_map[right_rec].items()
                )
            )

            if left_rec is not None and len(
                set(left_rec._pk.columns) & set(right_rec._pk.columns)  # type: ignore
            ):
                return left_rec, coalescent_join_sql(
                    left_table.alias(),
                    right_table.alias(),
                    reduce(
                        sqla.and_,
                        (
                            left_pk == right_pk
                            for left_pk, right_pk in zip(
                                self.base._map_pk(left_rec).columns,
                                self.base._map_pk(right_rec).columns,
                            )
                        ),
                    ),
                )
            else:
                return None, left_table.union(right_table).select()

        if len(sel_tables) > 1:
            rec_query_items = list(sel_tables.items())
            first_rec, first_select = rec_query_items[0]
            first: tuple[type[Record] | None, sqla.Select] = (
                first_rec,
                first_select.select().with_only_columns(
                    *(
                        col.label(name)
                        for name, col in self._base_col_map[first_rec].items()
                    )
                ),
            )

            _, base_union = reduce(_union, rec_query_items[1:], first)
        else:
            base_union = list(sel_tables.values())[0]

        base_join = base_union
        for base_table, join_on in self._base_table_map.values():
            if join_on is not None:
                base_join = base_join.join(base_table, join_on)

        res = base_join.subquery() if isinstance(base_join, sqla.Select) else base_join
        setattr(res, "_tab", self)
        return res

    def _get_upload_table(self) -> sqla.Table:
        cols = {
            name: col.copy()
            for cols in self._base_col_map.values()
            for name, col in cols.items()
        }
        for name, col in cols.items():
            col.name = name

        return sqla.Table(
            f"upload/{self.fqn.replace('.', '_')}/{token_hex(5)}",
            self.base._metadata,
            *cols.values(),
        )

    def _upload_df(self, df: pd.DataFrame | pl.DataFrame) -> sqla.Table:
        table = self._get_upload_table()

        if isinstance(df, pl.DataFrame):
            df.write_database(
                str(table),
                self.base.connection,
                if_table_exists="replace",
            )
        else:
            df.reset_index().to_sql(
                table.name,
                self.base.connection,
                if_exists="replace",
                index=False,
            )

        return table


ColT = TypeVar("ColT")

KeyT = TypeVar("KeyT", bound=Hashable, default=Any)


@dataclass(kw_only=True)
class Column(
    Attr[ColT, RwxT, OwnT],
    sqla.SQLColumnExpression[ColT],
    Generic[ColT, RwxT, OwnT, BackT],
):
    """Column property for a table model."""

    _table: Table[OwnT] | None = None

    sql_default: sqla.ColumnElement[ColT] | None = None

    @cached_prop
    def sql_type(self) -> sqla_type.TypeEngine:
        """Return the SQL type of this column."""
        return sqla_type.to_instance(self.common_value_type)

    @property
    def sql_col(self) -> sqla.ColumnClause[ColT]:
        """Return the SQL column of this property."""
        table = (
            self._table
            if self._table is not None
            else (
                self._owner._table
                if self._owner is not None and issubclass(self._owner, Record)
                else None
            )
        )
        col = sqla.column(
            self.name,
            self.sql_type,
            _selectable=table.sql if table else None,
        )
        setattr(col, "_prop", self)
        return col

    def __clause_element__(self) -> sqla.ColumnClause[ColT]:
        """Return the column clause element."""
        return self.sql_col

    @cached_prop
    def sql_comparator(self) -> sqla_type.TypeEngine.Comparator:
        """Return the comparator for this column."""
        return self.sql_type.comparator_factory(self.sql_col)

    if not TYPE_CHECKING:

        def __getattr__(self, key: str) -> Any:
            """Support rich comparison methods."""
            return getattr(self.sql_comparator, key)

    def label(self, name: str | None) -> sqla.Label[ColT]:
        """Label this column."""
        return self.sql_col.label(name)

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> ColT: ...

    @overload
    def __get__(
        self, instance: None, owner: type[RecT]
    ) -> Column[ColT, RwxT, RecT, Symbolic]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride] (TODO: report pyright bug)
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this column."""
        super_get = Attr.__get__(self, instance, owner)
        if super_get is not Ungot():
            return super_get

        assert instance is not None

        if isinstance(instance, Record):
            instance._load_dict()

        if self.name not in instance.__dict__:
            if self.default is not Not.defined:
                instance.__dict__[self.name] = self.default
            else:
                assert self.default_factory is not None
                instance.__dict__[self.name] = self.default_factory()

        return instance.__dict__[self.name]

    def __set__(self: Column[Any, U, Any], instance: Model, value: ColT) -> None:
        """Set the value of this attribute."""
        super_set = Attr.__set__(self, instance, value)
        if super_set is Unset():
            return

        instance.__dict__[self.name] = value

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self._owner, self._table))


TupT = TypeVar("TupT")


@dataclass
class ColTuple(Prop[TupT, OwnT, R]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...]

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> TupT: ...

    @overload
    def __get__(self, instance: None, owner: type[RecT]) -> ColTuple[TupT, RecT]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this attribute."""
        val = tuple(col.__get__(instance, owner) for col in self.columns)
        return val[0] if len(val) == 1 else val

    def __set__(self, instance: Any, value: Any) -> None:
        """Set the value of this attribute."""
        raise AttributeError("Cannot set an index.")

    def gen_value_map(self, val: Any) -> dict[str, Hashable]:
        """Generate a key-value map for this key."""
        idx_names = [c.name for c in self.columns]

        if len(idx_names) == 1:
            return {idx_names[0]: val}

        assert isinstance(val, tuple) and len(val) == len(idx_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(idx_names, val)}

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self.context, self.columns))


@dataclass(eq=False)
class Key(ColTuple[KeyT, OwnT]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...] = ()

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if len(self.columns) == 0:
            self.columns = (
                Column(
                    _name=self.name,
                    _type=Column[self.common_value_type],
                    _owner=self.context,
                ),
            )
            setattr(owner, f"{name}_col", self.columns[0])


TgtT = TypeVar("TgtT", bound="Record", covariant=True)


@dataclass(eq=False)
class ForeignKey(ColTuple[KeyT, OwnT], Generic[TgtT, KeyT, OwnT]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...] = field(init=False)

    column_map: dict[Column[Hashable, Any, OwnT], Column[Hashable, Any, TgtT]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:  # noqa: D105
        self.columns = (
            tuple(self.column_map.keys()) if self.column_map is not None else ()
        )

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if self.column_map is None:
            pk_cols = self.target._pk.columns

            col_map = {
                Column(
                    _name=f"{self.name}_{pk_col.name}_fk",
                    _type=Column[pk_col.common_value_type],
                    _owner=self.context,
                ): pk_col
                for pk_col in pk_cols
            }

            self.columns = tuple(col_map.keys())
            for col in col_map.keys():
                setattr(owner, col.name, col)

    @property
    def target(self) -> type[TgtT]:
        """Return the target type for this foreign key."""
        target_type = self.typeargs[TgtT]
        assert isinstance(target_type, type)
        return cast(type[TgtT], target_type)


class RecordTable:
    """Table, which a record belongs to."""

    @overload
    def __get__(
        self, instance: None, owner: type[RecT2]
    ) -> Table[RecT2, Symbolic, R]: ...

    @overload
    def __get__(
        self, instance: RecT2, owner: type[RecT2]
    ) -> Table[RecT2, DynBackendID, Any]: ...

    def __get__(  # noqa: D105
        self, instance: Record | None, owner: type[Record]
    ) -> Table[Record, Any, Any]:
        return (
            symbol_base.table(owner)
            if instance is None
            else instance._base.table(owner)
        )


class Record(Model, Generic[KeyT]):
    """Schema for a table in a database."""

    _root_class = True

    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID4: sqla.types.CHAR(36),
    }

    _table = RecordTable()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)

        for superclass in cls._model_superclasses:
            if superclass in cls.__bases__ and issubclass(superclass, Record):
                super_pk = superclass._pk  # type: ignore
                col_map = {
                    Column(
                        _name=pk_col.name,
                        _type=Column[pk_col.common_target_type],
                        _owner=cls,
                    ): pk_col
                    for pk_col in super_pk.columns
                }
                for col in col_map.keys():
                    setattr(cls, col.name, col)

                pk = Key[super_pk.typeargs[KeyT], cls](columns=tuple(col_map.keys()))
                setattr(cls, pk.name, pk)

                fk = ForeignKey[superclass, super_pk.typeargs[KeyT], cls](
                    column_map=col_map
                )
                setattr(cls, fk.name, fk)

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        name = cls._table_name or cls._fqn
        fqn_parts = name.split(".")

        name = fqn_parts[-1]
        for part in reversed(fqn_parts[:-1]):
            name = part + "_" + name
            if len(name) > 40:
                break

        return name

    @classmethod
    def _cols(cls) -> set[Column]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Column)
        }

    @classmethod
    def _keys(cls) -> set[Key]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Key)
        }

    @classmethod
    def _fks(cls) -> set[ForeignKey]:
        """Columns of this record type's table."""
        return {
            prop
            for prop in cls._get_class_props().values()
            if isinstance(prop, ForeignKey)
        }

    @classmethod
    def _pl_schema(cls) -> dict[str, pl.DataType | type | None]:
        """Return the schema of the dataset."""
        return get_pl_schema(
            {name: col for name, col in cls._props.items() if isinstance(col, Column)}
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:
        """Return the table clause element."""
        return cls._table._base_table_map[cls][0]

    _published: bool = False
    _base: Attr[Base[Any, DynBackendID]] = Attr(default_factory=Base)

    _pk: Key[KeyT, Self]

    @cached_prop
    def _whereclause(self) -> sqla.ColumnElement[bool]:
        cols = {
            col.name: sql_col
            for col, sql_col in self._base._map_cols(type(self)).items()
        }
        pk_map = type(self)._pk.gen_value_map(self._pk)

        return sqla.and_(
            *(cols[col] == val for col, val in pk_map.items()),
        )

    def _load_dict(self) -> None:
        if not self._published or self._pk in self._base._get_valid_cache_set(
            type(self)
        ):
            return

        query = self._table.sql.select().where(self._whereclause)

        with self._base.engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()

        if row is not None:
            self.__dict__.update(row._asdict())

    def _save_dict(self) -> None:
        df = records_to_df([self])
        self._table.mutate(self._base.table({type(self)}, df))
        self._base._get_valid_cache_set(type(self)).remove(self._pk)

    def __hash__(self) -> int:
        """Identify the record by database and id."""
        return gen_int_hash((self._base if self._published else None, self._pk))

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)
