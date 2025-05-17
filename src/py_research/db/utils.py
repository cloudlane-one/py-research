"""Utility functions for working with dataframes and SQL databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, time, timedelta
from functools import reduce
from types import UnionType
from typing import Any, Literal, get_args

import pandas as pd
import polars as pl
import sqlalchemy as sqla
import sqlalchemy.dialects.mysql as mysql
import sqlalchemy.dialects.postgresql as postgresql
import sqlalchemy.dialects.sqlite as sqlite
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from py_research.reflect.types import SingleTypeDef, TypeRef, get_common_type
from py_research.types import UUID4

pl_type_map: dict[
    Any, pl.DataType | type | Callable[[SingleTypeDef | UnionType], pl.DataType]
] = {
    **{
        t: t
        for t in (
            int,
            float,
            complex,
            str,
            bool,
            datetime,
            date,
            time,
            timedelta,
            bytes,
        )
    },
    Literal: lambda t: pl.Enum([str(a) for a in get_args(t)]),
    UUID4: pl.String,
}


def pd_to_py_dtype(c: pd.Series | pl.Series) -> type | None:
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


def get_pl_schema(
    cols: Mapping[str, TypeRef],
) -> dict[str, pl.DataType | type | None]:
    """Return the schema of the dataset."""
    exact_matches = {
        name: (pl_type_map.get(get_common_type(typ.typeform)), typ)
        for name, typ in cols.items()
    }
    matches = {
        name: (
            (match, typ.base_type)
            if match is not None
            else (pl_type_map.get(get_common_type(typ.typeform)), typ.typeform)
        )
        for name, (match, typ) in exact_matches.items()
    }

    return {
        name: (
            match if isinstance(match, pl.DataType | type | None) else match(match_type)
        )
        for name, (match, match_type) in matches.items()
    }


def sql_to_py_dtype(c: sqla.ColumnElement) -> type | None:
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


def remove_cross_fk(table: sqla.Table):
    """Dirty vodoo to remove external FKs from existing table."""
    for c in table.columns.values():
        c.foreign_keys = set(
            fk
            for fk in c.foreign_keys
            if fk.constraint and fk.constraint.referred_table.schema == table.schema
        )

    table.foreign_keys = set(  # pyright: ignore[reportAttributeAccessIssue]
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


def adapt_date_iso(val: date) -> str:
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()


def adapt_datetime_iso(val: datetime) -> str:
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()


def adapt_datetime_epoch(val: datetime) -> int:
    """Adapt datetime.datetime to Unix timestamp."""
    return int(val.timestamp())


def convert_date(val: bytes) -> date:
    """Convert ISO 8601 date to datetime.date object."""
    return date.fromisoformat(val.decode())


def convert_datetime(val: bytes) -> datetime:
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.fromisoformat(val.decode())


def convert_timestamp(val: bytes) -> datetime:
    """Convert Unix epoch timestamp to datetime.datetime object."""
    return datetime.fromtimestamp(int(val))


def register_sqlite_adapters() -> None:
    """Register SQLite adapters and converters."""
    sqlite3.register_adapter(date, adapt_date_iso)
    sqlite3.register_adapter(datetime, adapt_datetime_iso)
    sqlite3.register_adapter(datetime, adapt_datetime_epoch)

    sqlite3.register_converter("date", convert_date)
    sqlite3.register_converter("datetime", convert_datetime)
    sqlite3.register_converter("timestamp", convert_timestamp)


type JoinConditionSql = Callable[
    [sqla.FromClause | sqla.Select, sqla.FromClause | sqla.Select],
    sqla.ColumnElement[bool],
]
type JoinConditionPl = Callable[[pl.DataFrame, pl.DataFrame], pl.Expr]

type RelJoinSql = tuple[sqla.FromClause, JoinConditionSql]
type RelJoinPl = tuple[pl.DataFrame, JoinConditionPl]


def recursive_join_sql(
    fromclause: sqla.FromClause,
    join_loops: Iterable[list[RelJoinSql]],
) -> sqla.CTE:
    """Recursively join and union a fromclause via loops.

    Args:
        cols: Columns to select.
        fromclause: Initial fromclause.
        join_loops: Loops of joins. Must start and end in the given fromclause.
    """
    base_cte = sqla.select(fromclause).cte(recursive=True)

    unions: list[sqla.Select] = []
    for join_loop in join_loops:
        sel = sqla.select(*join_loop[-1][0].columns).select_from(base_cte)
        for join in join_loop:
            sel = sel.join(join[0], join[1](sel, join[0]))
        unions.append(sel)

    return base_cte.union(*unions)


def recursive_join_pl(
    fromclause: pl.DataFrame,
    join_loops: Iterable[list[RelJoinPl]],
) -> pl.DataFrame:
    """Recursively join and union a fromclause via loops.

    Args:
        cols: Columns to select.
        fromclause: Initial fromclause.
        join_loops: Loops of joins. Must start and end in the given fromclause.
    """
    union = fromclause
    last_len = 0

    while last_len < len(union):
        last_len = len(union)

        for join_loop in join_loops:
            sel = union

            for join in join_loop:
                sel = sel.join(join[0], join[1](sel, join[0]))
            sel = sel.with_columns(join_loop[-1][0].columns)

            union = union.vstack(sel).unique()

    return union


def safe_delete(
    table: sqla.Table, input_sql: sqla.FromClause | sqla.Select, engine: sqla.Engine
) -> sqla.Delete:
    """Generate an idempotent deletion statement."""
    if engine.dialect.name in (
        "postgres",
        "postgresql",
        "duckdb",
        "mysql",
        "mariadb",
    ):
        # Delete-from.
        return table.delete().where(
            reduce(
                sqla.and_,
                (pk == input_sql.c[pk.name] for pk in table.primary_key.columns),
            )
        )
    elif engine.dialect.name in ("sqlite",):
        return table.delete().where(
            sqla.column("rowid").in_(
                sqla.select(sqla.column("rowid"))
                .select_from(table)
                .join(
                    (
                        input_sql
                        if isinstance(input_sql, sqla.FromClause)
                        else input_sql.subquery()
                    ),
                    reduce(
                        sqla.and_,
                        (
                            pk == input_sql.c[pk.name]
                            for pk in table.primary_key.columns
                        ),
                    ),
                )
            )
        )
    else:
        raise NotImplementedError("Replacement not supported for this dialect.")


def safe_insert(
    table: sqla.Table,
    input_sql: sqla.FromClause | sqla.Select,
    engine: sqla.Engine,
    upsert: bool = False,
) -> sqla.Insert:
    """Generate an idempotent insert statement wiht optional upsert."""
    insert_cols = list(set(table.columns.keys()) & set(input_sql.columns.keys()))

    if engine.dialect.name in (
        "postgres",
        "postgresql",
        "duckdb",
        "sqlite",
    ):
        if engine.dialect.name in (
            "postgres",
            "postgresql",
            "duckdb",
        ):
            # For Postgres / DuckDB, use: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#updating-using-the-excluded-insert-values
            statement = postgresql.Insert(table)
        else:
            # For SQLite, use: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#updating-using-the-excluded-insert-values
            statement = sqlite.Insert(table)

        statement = statement.from_select(
            insert_cols,
            input_sql.select().where(sqla.text("true")),
        )

        if upsert:
            return statement.on_conflict_do_update(
                index_elements=[col.name for col in table.primary_key.columns],
                set_={
                    name: col
                    for name, col in statement.excluded.items()
                    if name not in table.primary_key.columns
                },
            )
        return statement.on_conflict_do_nothing()

    elif engine.dialect.name in (
        "mysql",
        "mariadb",
    ):
        # For MySQL / MariaDB, use: https://docs.sqlalchemy.org/en/20/dialects/mysql.html#insert-on-duplicate-key-update-upsert
        statement = (
            mysql.Insert(table)
            .from_select(
                insert_cols,
                input_sql,
            )
            .prefix_with("INSERT INTO")
        )
        if upsert:
            return statement.on_duplicate_key_update(**statement.inserted)
        return statement.prefix_with("INSERT IGNORE INTO")
    else:
        # For others, use CTE: https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.cte
        raise NotImplementedError("Upsert not supported for this database dialect.")


def safe_update(
    table: sqla.Table, input_sql: sqla.FromClause | sqla.Select, engine: sqla.Engine
) -> sqla.Update:
    """Generate an idempotent update statement."""
    if engine.dialect.name in (
        "postgres",
        "postgresql",
        "duckdb",
        "mysql",
        "mariadb",
        "sqlite",
    ):
        # Update-from.
        return (
            table.update()
            .values(dict(input_sql.columns))
            .where(
                reduce(
                    sqla.and_,
                    (pk == input_sql.c[pk.name] for pk in table.primary_key.columns),
                ),
            )
        )
    else:
        # Correlated update.
        raise NotImplementedError("Correlated update not supported yet.")


def split_df_by_prefixes(
    df: pl.DataFrame,
    prefixes: Iterable[str],
    sep: str = ".",
) -> dict[str, pl.DataFrame]:
    """Split a dataframe into multiple dataframes by column prefixes.

    Args:
        df: The dataframe to split.
        prefixes: The prefixes to split by.
        sep: The separator between the prefix and the rest of the column name.

    Returns:
        A dictionary of dataframes, where the keys are the prefixes and the values
        are the corresponding dataframes.
    """
    return {
        prefix: df.select(
            *(pl.col(c) for c in df.columns if c.startswith(prefix + sep))
        )
        for prefix in prefixes
    }
