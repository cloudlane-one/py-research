"""SQL utility functions."""

import pandas as pd
import polars as pl
import sqlalchemy as sqla
from pandas.api.types import (
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from .schema import DataFrame, Schema


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


def map_foreignkey_schema(
    _: sqla.Table,
    to_schema: str | None,
    constraint: sqla.ForeignKeyConstraint,
    referred_schema: str | None,
    schema_dict: "dict[str | None, type[Schema]]",
) -> str | None:
    """Map foreign key schema to schema in schema_dict."""
    assert to_schema in schema_dict

    for schema_name, schema in schema_dict.items():
        if schema is not None:
            for table in schema._tables:
                if table._sqla_table is constraint.referred_table:
                    return schema_name

    return referred_schema


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
