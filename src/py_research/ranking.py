"""Functions for ranking entities by the nature and quantity of their publications."""
from typing import Literal

import numpy as np
import pandas as pd

from py_research.intl import get_localization

RankMode = Literal["ascending", "descending"]


def create_rank_series(
    rank_by: pd.Series | list[pd.Series] | pd.DataFrame,
    rank_mode: RankMode = "descending",
    exclude: pd.Series | None = None,
    name: str = "rank",
) -> pd.Series:
    """Create a series of ranks."""
    data_table = pd.DataFrame(rank_by)
    filtered_data = (
        data_table.loc[~exclude]  # pylint: disable=E1130:invalid-unary-operand-type
        if exclude is not None
        else data_table
    )

    ranks = pd.Series(
        np.arange(1, len(data_table) + 1),
        index=filtered_data.sort_values(
            by=list(filtered_data.columns), ascending=(rank_mode == "ascending")
        ).index,
        name=name,
    )

    if ranks.empty:
        return ranks

    rank_by_cols = list(data_table.columns)
    for _, g in data_table.groupby(
        by=(rank_by_cols[0] if len(rank_by_cols) == 1 else rank_by_cols)
    ):
        ranks.loc[g.index] = min(ranks.loc[g.index])

    return data_table.join(ranks, how="left")[name]


def create_ranking_filter(
    rank_by: pd.Series,
    cutoff_rank: int,
    sort_order: Literal["ascending", "descending"] = "descending",
    always_include: pd.Series | None = None,
    pre_filter: pd.Series | None = None,
    post_filter: pd.Series | None = None,
    count_always_included: bool = False,
) -> pd.Series:
    """Create a filter based on ranking."""
    loc = get_localization()

    desc = loc.text(
        ", ".join(
            s
            for s in [
                (
                    "up to "
                    + str(cutoff_rank)
                    + " highest-placed"
                    + (
                        " " + str(pre_filter.name)
                        if pre_filter is not None
                        and str(pre_filter.name)
                        and len(str(pre_filter.name)) <= 20
                        else ""
                    )
                    + (f" according to '{rank_by.name}' ({sort_order})")
                    + (
                        " only including " + str(pre_filter.name)
                        if pre_filter is not None
                        and pre_filter.name is not None
                        and len(str(pre_filter.name)) > 20
                        else ""
                    )
                ),
                (
                    ("always including " + str(always_include.name))
                    if always_include is not None and always_include.name is not None
                    else ""
                ),
            ]
            if s
        )
    )

    rank_by_filter = np.full(len(rank_by), True)

    if pre_filter is not None:
        # Apply a custom filter, if given
        rank_by_filter &= pre_filter

    rank_series = create_rank_series(
        rank_by.loc[rank_by_filter],
        sort_order,
        exclude=(always_include if not count_always_included else None),
    )
    rank_by_filter &= rank_series <= cutoff_rank

    if always_include is not None:
        # Apply a custom filter, if given
        rank_by_filter |= always_include

    if post_filter is not None:
        # Apply a custom filter, if given
        rank_by_filter &= post_filter

    return pd.Series(rank_by_filter, index=rank_by.index, name=desc)
