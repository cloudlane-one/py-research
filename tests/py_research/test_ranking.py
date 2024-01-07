"""Test ranking module."""

import pandas as pd

from py_research.ranking import create_rank_series, create_ranking_filter


def test_create_rank_series():
    """Test create_rank_series."""
    # Test with a Series
    rank_by = pd.Series([1, 2, 3, 4, 5])
    exclude = pd.Series([False, False, True, False, False])
    rank_series = create_rank_series(rank_by, exclude=exclude)
    assert rank_series.equals(pd.Series([1.0, 2.0, None, 3.0, 4.0], name="rank"))

    # Test with a DataFrame
    rank_by = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    exclude = pd.Series([False, False, True, False, False])
    rank_series = create_rank_series(rank_by, exclude=exclude)
    assert rank_series.equals(pd.Series([1.5, 2.5, None, 3.5, 4.5], name="rank"))

    # Test with ascending rank mode
    rank_series = create_rank_series(rank_by, rank_mode="ascending", exclude=exclude)
    assert rank_series.equals(pd.Series([3.5, 2.5, None, 1.5, 0.5], name="rank"))

    # Test with no exclusion
    rank_series = create_rank_series(rank_by, rank_mode="ascending")
    assert rank_series.equals(pd.Series([4.5, 3.5, 2.5, 1.5, 0.5], name="rank"))


def test_create_ranking_filter():
    """Test create_ranking_filter."""
    # Test with a Series
    rank_by = pd.Series([1, 2, 3, 4, 5])
    cutoff_rank = 3
    filter_series = create_ranking_filter(rank_by, cutoff_rank)
    assert filter_series.equals(pd.Series([False, False, True, True, True]))

    # Test with ascending sort order
    filter_series = create_ranking_filter(rank_by, cutoff_rank, sort_order="ascending")
    assert filter_series.equals(pd.Series([True, True, True, False, False]))

    # Test with rank_only
    rank_only = pd.Series([False, False, True, True, True])
    filter_series = create_ranking_filter(rank_by, cutoff_rank, rank_only=rank_only)
    assert filter_series.equals(pd.Series([False, False, True, False, False]))

    # Test with show_only
    show_only = pd.Series([False, False, True, True, True])
    filter_series = create_ranking_filter(rank_by, cutoff_rank, show_only=show_only)
    assert filter_series.equals(pd.Series([False, False, True, True, True]))

    # Test with show_always
    show_always = pd.Series([True, False, False, False, False])
    filter_series = create_ranking_filter(rank_by, cutoff_rank, show_always=show_always)
    assert filter_series.equals(pd.Series([True, False, True, True, True]))

    # Test with count_always_shown
    filter_series = create_ranking_filter(
        rank_by, cutoff_rank, show_always=show_always, count_always_shown=True
    )
    assert filter_series.equals(pd.Series([True, False, True, True, False]))
