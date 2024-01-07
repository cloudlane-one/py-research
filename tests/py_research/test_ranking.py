"""Test ranking module."""

import pandas as pd

from py_research.ranking import create_rank_series, create_ranking_filter


def test_create_rank_series():
    """Test create_rank_series."""
    # Test with a Series
    rank_by = pd.Series([1, 2, 3, 4, 5])
    exclude = pd.Series([False, False, True, False, False])
    rank_series = create_rank_series(rank_by, exclude=exclude)
    pd.testing.assert_series_equal(
        rank_series, pd.Series([4, 3, None, 2, 1], name="rank"), check_dtype=False
    )

    # Test with a DataFrame
    rank_by = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    exclude = pd.Series([False, False, True, False, False])
    rank_series = create_rank_series(rank_by, exclude=exclude)
    pd.testing.assert_series_equal(
        rank_series, pd.Series([4, 3, None, 2, 1], name="rank"), check_dtype=False
    )

    # Test with ascending rank mode
    rank_series = create_rank_series(rank_by, rank_mode="ascending", exclude=exclude)
    pd.testing.assert_series_equal(
        rank_series, pd.Series([1, 2, None, 3, 4], name="rank"), check_dtype=False
    )

    # Test with no exclusion
    rank_series = create_rank_series(rank_by, rank_mode="ascending")
    pd.testing.assert_series_equal(
        rank_series, pd.Series([1, 2, 3, 4, 5], name="rank"), check_dtype=False
    )


def test_create_ranking_filter():
    """Test create_ranking_filter."""
    # Test with a Series
    rank_by = pd.Series([1, 2, 3, 4, 5])
    cutoff_rank = 3
    filter_series = create_ranking_filter(rank_by, cutoff_rank)
    pd.testing.assert_series_equal(
        filter_series, pd.Series([False, False, True, True, True]), check_names=False
    )

    # Test with ascending sort order
    filter_series = create_ranking_filter(rank_by, cutoff_rank, sort_order="ascending")
    pd.testing.assert_series_equal(
        filter_series, pd.Series([True, True, True, False, False]), check_names=False
    )

    # Test with rank_only
    rank_only = pd.Series([False, False, True, True, True])
    filter_series = create_ranking_filter(rank_by, cutoff_rank, rank_only=rank_only)
    pd.testing.assert_series_equal(
        filter_series, pd.Series([False, False, True, True, True]), check_names=False
    )

    # Test with show_only
    show_only = pd.Series([False, False, True, True, True])
    filter_series = create_ranking_filter(rank_by, cutoff_rank, show_only=show_only)
    pd.testing.assert_series_equal(
        filter_series, pd.Series([False, False, True, True, True]), check_names=False
    )

    # Test with show_always
    show_always = pd.Series([True, False, False, False, False])
    filter_series = create_ranking_filter(rank_by, cutoff_rank, show_always=show_always)
    pd.testing.assert_series_equal(
        filter_series, pd.Series([True, False, True, True, True]), check_names=False
    )

    # Test with count_always_shown
    filter_series = create_ranking_filter(
        rank_by, cutoff_rank, show_always=show_always, count_always_shown=True
    )
    pd.testing.assert_series_equal(
        filter_series, pd.Series([True, False, True, True, True]), check_names=False
    )
