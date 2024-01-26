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
    result = dist_table(df, "category", "id")
    expected = pd.Series(
        [2, 3], index=pd.Index(["X", "Y"], name="category"), name="value"
    )
    pd.testing.assert_series_equal(result, expected)

    # Test with one domain
    result = dist_table(df, "category", "id", domains={"category": ["X", "Y", "Z"]})
    expected = pd.Series(
        [2, 3, 0], index=pd.Index(["X", "Y", "Z"], name="category"), name="value"
    )
    pd.testing.assert_series_equal(result, expected)

    # Test with value summation
    result = dist_table(df, "category", "id", value_col="value")
    expected = pd.Series(
        [4, 12], index=pd.Index(["X", "Y"], name="category"), name="value"
    )
    pd.testing.assert_series_equal(result, expected)

    # Test with multiple id_cols and category_cols
    df["sub_id"] = ["P", "Q", "P", "Q", "Q", "Q"]
    df["sub_category"] = ["U", "V", "U", "V", "U", "V"]
    result = dist_table(
        df,
        ["category", "sub_category"],
        ["id", "sub_id"],
    )
    expected = pd.Series(
        [3, 3],
        index=pd.MultiIndex.from_tuples(
            [
                ("X", "U"),
                ("Y", "V"),
            ],
            names=["category", "sub_category"],
        ),
        name="value",
    )
    pd.testing.assert_series_equal(result, expected)

    # Test with None id_cols
    df = df.set_index(["id", "sub_id"])
    result = dist_table(df, ["category", "sub_category"])
    pd.testing.assert_series_equal(result, expected)

    # Test with multiple domains
    result = dist_table(
        df,
        ["category", "sub_category"],
        ["id", "sub_id"],
        domains={"category": ["X", "Y", "Z"], "sub_category": ["U", "V", "W"]},
    )
    expected = pd.Series(
        [3, 0, 0, 0, 3, 0, 0, 0, 0],
        index=pd.MultiIndex.from_product(
            [["X", "Y", "Z"], ["U", "V", "W"]],
            names=["category", "sub_category"],
        ),
        name="value",
    )
    pd.testing.assert_series_equal(result, expected)
