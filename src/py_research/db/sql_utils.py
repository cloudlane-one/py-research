"""Utility functions for querying and mutating SQL databases."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce
from typing import Literal

import polars as pl
import sqlalchemy as sqla
import sqlalchemy.dialects.mysql as mysql
import sqlalchemy.dialects.postgresql as postgresql
import sqlalchemy.dialects.sqlite as sqlite

type JoinConditionSql = Callable[
    [sqla.FromClause | sqla.Select, sqla.FromClause | sqla.Select],
    sqla.ColumnElement[bool],
]
type JoinConditionPl = Callable[[pl.DataFrame, pl.DataFrame], pl.Expr]

type RelJoinSql = tuple[sqla.FromClause, JoinConditionSql]
type RelJoinPl = tuple[pl.DataFrame, JoinConditionPl]


def coalescent_join_sql(
    left: sqla.Select | sqla.FromClause,
    right: sqla.Select | sqla.FromClause,
    on: sqla.ColumnElement[bool],
    join: Literal["left", "right", "full"] = "full",
    coalesce: Literal["left", "right"] = "left",
) -> sqla.Select:
    """Join two selects and coalesce their columns."""
    left_table = left if isinstance(left, sqla.FromClause) else left.subquery()
    right_table = right if isinstance(right, sqla.FromClause) else right.subquery()

    cols = [
        sqla.func.coalesce(
            *(
                (
                    left_table.c.get(name, sqla.null()),
                    right_table.c.get(name, sqla.null()),
                )
                if coalesce == "left"
                else (
                    right_table.c.get(name, sqla.null()),
                    left_table.c.get(name, sqla.null()),
                )
            )
        )
        for name in set(left_table.columns.keys()) | set(right_table.columns.keys())
    ]

    return (
        sqla.select(*cols)
        .select_from(left_table)
        .join(right_table.alias(), on, isouter=True)
        if join == "left"
        else (
            sqla.select(*cols)
            .select_from(left_table)
            .join(right_table.alias(), on, full=True)
            if join == "full"
            else sqla.select(*cols)
            .select_from(right_table)
            .join(left_table.alias(), on, isouter=True)
        )
    )


def coalescent_join_pl(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: pl.Expr,
    join: Literal["left", "right", "full"] = "full",
    coalesce: Literal["left", "right"] = "left",
) -> pl.DataFrame:
    """Join two selects and coalesce their columns."""
    cols = [
        pl.coalesce(
            *(
                (
                    name if name in left.columns else None,
                    name if name in right.columns else None,
                )
                if coalesce == "left"
                else (
                    name if name in right.columns else None,
                    name if name in left.columns else None,
                )
            )
        )
        for name in set(left.columns) | set(right.columns)
    ]

    return left.join(right, on, how=join).with_columns(*cols)


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
