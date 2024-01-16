"""Helper functions for statistical evalutation of (dataframe-based) data."""

import pandas as pd


def dist_table(
    df: pd.DataFrame, id_cols: str | list[str] | None, category_cols: str | list[str]
) -> pd.Series:
    """Return a frequency table of the distribution of unique entities.

    Entities are identified by ``id_cols``. Distribution is presented over
    unique categories in ``category_cols``.

    Args:
        df: Dataframe to evaluate.
        id_cols: Columns to identify entities by.
        category_cols: Columns to evaluate distribution over.

    Returns:
        Frequency table of the distribution of unique entities.
    """
    id_cols = (
        [id_cols]
        if isinstance(id_cols, str)
        else id_cols
        if id_cols is not None
        else [n or "index" for n in df.index.names]
    )
    category_cols = [
        *([category_cols] if isinstance(category_cols, str) else category_cols)
    ]

    counts = (
        df.reset_index()[[*category_cols, *id_cols]]
        .groupby(by=category_cols, group_keys=True)
        .apply(lambda df: len(df.drop_duplicates()))
        .rename("freq")
    )

    return counts
