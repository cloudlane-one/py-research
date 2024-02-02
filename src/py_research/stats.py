"""Helper functions for statistical evalutation of (dataframe-based) data."""

from collections.abc import Hashable

import numpy as np
import pandas as pd


def dist_table(
    df: pd.DataFrame,
    category_cols: str | list[str],
    id_cols: str | list[str] | None = None,
    value_col: str | None = None,
    domains: dict[str, list[Hashable] | np.ndarray | pd.Index] = {},
) -> pd.Series:
    """Return a frequency table of the distribution of unique entities.

    Entities are identified by ``id_cols``. Distribution is presented over
    unique categories in ``category_cols``.

    Args:
        df: Dataframe to evaluate.
        category_cols: Columns to evaluate distribution over.
        id_cols: Columns to identify entities by.
        value_col: Unique values per entity to sum up.
        domains:
            Force the distribution to be evaluated over these domains,
            filling missing values with 0.

    Returns:
        Series of the distribution's values (count or sum) given the categories
        in the index.
    """
    id_cols = (
        [id_cols]
        if isinstance(id_cols, str)
        else id_cols if id_cols is not None else [n or "index" for n in df.index.names]
    )
    category_cols = [
        *([category_cols] if isinstance(category_cols, str) else category_cols)
    ]

    counts = (
        df.reset_index()[
            [*category_cols, *id_cols, *([value_col] if value_col else [])]
        ]
        .groupby(by=category_cols, group_keys=True)
        .apply(
            lambda df: (
                len(df.drop_duplicates())
                if value_col is None
                else df.drop_duplicates(subset=id_cols)[value_col].sum()
            )
        )
        .rename("value")
    )

    if len(domains) > 0:
        if len(category_cols) == 1:
            col = category_cols[0]
            domain = domains.get(col)

            if domain is not None:
                counts = counts.reindex(
                    domain,
                    fill_value=0,
                ).rename_axis(index=col)
        else:
            count_df = counts.to_frame().reset_index()

            for col, domain in domains.items():
                assert col in category_cols, f"Unknown category column: {col}"
                other_cat = set(category_cols) - {col}

                count_df = (
                    count_df.groupby(list(other_cat), group_keys=True)
                    .apply(
                        lambda df: df.set_index(col)
                        .drop(columns=[*list(other_cat)])
                        .reindex(
                            domain,
                            fill_value=0,
                        )
                        .rename_axis(index=col)
                    )
                    .reset_index()
                )

            counts = count_df.set_index(category_cols)["value"]

    return counts
