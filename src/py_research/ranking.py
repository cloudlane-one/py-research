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
    rank_only: pd.Series | None = None,
    show_only: pd.Series | None = None,
    show_always: pd.Series | None = None,
    count_always_shown: bool = False,
) -> pd.Series:
    """Create a filter based on ranking."""
    loc = get_localization()

    def _filter_explanation(
        cutoff: int,
        rank_by: str,
        sort_order: str,
        rank_only: str | None = None,
        show_only: str | None = None,
        show_always: str | None = None,
    ) -> str:
        return "; ".join(
            s
            for s in [
                (
                    f"{cutoff} highest-ranked"
                    + (
                        f" {loc.label(rank_only, 'col_title')}"
                        if rank_only is not None and len(rank_only) <= 20
                        else ""
                    )
                    + f" according to '{loc.label(rank_by, 'col_title')}'"
                    + f"({sort_order})"
                    + (
                        f", only ranking {loc.label(rank_only, 'col_title')}"
                        if rank_only is not None and len(rank_only) > 20
                        else ""
                    )
                ),
                (
                    f"only showing {loc.label(show_only, 'col_title')}"
                    if show_only is not None
                    else ""
                ),
                (
                    f"always showing {loc.label(show_always, 'col_title')}"
                    if show_always is not None
                    else ""
                ),
            ]
            if s
        )

    desc = loc.message(
        _filter_explanation,
        cutoff_rank,
        str(rank_by.name),
        sort_order,
        rank_only=str(rank_only.name)
        if rank_only is not None and rank_only.name is not None
        else None,
        show_only=str(show_only.name)
        if show_only is not None and show_only.name is not None
        else None,
        show_always=str(show_always.name)
        if show_always is not None and show_always.name is not None
        else None,
    )

    rank_by_filter = np.full(len(rank_by), True)

    if rank_only is not None:
        # Apply a custom filter, if given
        rank_by_filter &= rank_only

    rank_series = create_rank_series(
        rank_by.loc[rank_by_filter],
        sort_order,
        exclude=(show_always if not count_always_shown else None),
    )
    rank_by_filter &= rank_series <= cutoff_rank

    if show_always is not None:
        # Apply a custom filter, if given
        rank_by_filter |= show_always

    if show_only is not None:
        # Apply a custom filter, if given
        rank_by_filter &= show_only

    return pd.Series(rank_by_filter, index=rank_by.index, name=desc)
