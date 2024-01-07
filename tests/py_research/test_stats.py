"""Test stats module."""

import pandas as pd
from py_research.stats import dist_table


def test_dist_table():
    """Test dist_table."""
    # Test with a DataFrame
    df = pd.DataFrame(
        {
            "id": ["A", "A", "B", "B", "B", "C"],
            "category": ["X", "Y", "X", "Y", "X", "Y"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    result = dist_table(df, "id", "category")
    expected = pd.Series(
        [2, 3], index=pd.Index(["X", "Y"], name="category"), name="freq"
    )
    pd.testing.assert_series_equal(result, expected)

    # Test with multiple id_cols and category_cols
    df["sub_id"] = ["P", "Q", "P", "Q", "Q", "Q"]
    df["sub_category"] = ["U", "V", "U", "V", "U", "V"]
    result = dist_table(df, ["id", "sub_id"], ["category", "sub_category"])
    expected = pd.Series(
        [3, 3],
        index=pd.MultiIndex.from_tuples(
            [
                ("X", "U"),
                ("Y", "V"),
            ],
            names=["category", "sub_category"],
        ),
        name="freq",
    )
    pd.testing.assert_series_equal(result, expected)

    # Test with None id_cols
    df = df.set_index(["id", "sub_id"])
    result = dist_table(df, None, ["category", "sub_category"])
    pd.testing.assert_series_equal(result, expected)
