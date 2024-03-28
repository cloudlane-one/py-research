"""Helper functions for statistical evalutation of (dataframe-based) data."""

from collections.abc import Hashable

import numpy as np
import pandas as pd


def _union_children_on_parents(
    df: pd.DataFrame, name_col: str, parent_col: str
) -> pd.DataFrame:
    """Add union of all their children's ids to each parent node, recursively."""
    tree = df[[name_col, parent_col]].drop_duplicates()

    child_ids = df
    for _ in range(20):
        df = (
            pd.concat(
                [
                    df,
                    (
                        child_ids.drop(columns=[name_col])
                        .rename(columns={parent_col: name_col})
                        .merge(tree, on=name_col, how="left")
                    ),
                ]
            )
            .drop_duplicates()
            .dropna(subset=[name_col])
        )

        child_ids = df.loc[df[name_col].isin(child_ids[parent_col].dropna().unique())]
        if child_ids.empty:
            break

    return df


def dist_table(
    df: pd.DataFrame,
    category_cols: str | list[str],
    id_cols: str | list[str] | None = None,
    value_col: str | None = None,
    domains: dict[str, list[Hashable] | np.ndarray | pd.Index] = {},
    category_parent_cols: str | dict[str, str] | None = None,
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
        category_parent_cols:
            If category values are discrete and hierarchical, you may supply
            a parent column for each category column. This will be used to
            aggregate the distribution over the parent categories.

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

    if category_parent_cols is not None:
        category_parent_cols = (
            category_parent_cols
            if isinstance(category_parent_cols, dict)
            else {category_cols[0]: category_parent_cols}
        )
        df = pd.concat(
            [
                _union_children_on_parents(df, name_col=col, parent_col=parent_col)
                for col, parent_col in category_parent_cols.items()
            ]
        ).drop_duplicates()

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
