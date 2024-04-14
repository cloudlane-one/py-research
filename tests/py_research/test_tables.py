"""Test tables module."""

import pandas as pd
from py_research.tables import ResultTable, TableColors, TableStyle


def test_result_table():
    """Test ResultTable."""
    # Create a DataFrame
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Create a ResultTable
    table = ResultTable(df)

    # Test the render method
    rendered = table.to_html()
    assert isinstance(rendered, str)
    assert "A" in rendered
    assert "B" in rendered
    assert "C" in rendered

    # Test the styles attribute
    style = TableStyle(cols=["A"], css={"font-weight": "bold"})
    table.styles.append(style)
    rendered = table.to_html()
    assert "font-weight: bold" in rendered

    # Test the labels attribute
    table.labels = {"A": "Alpha", "B": "Beta", "C": "Gamma"}
    rendered = table.to_html()
    assert "Alpha" in rendered
    assert "Beta" in rendered
    assert "Gamma" in rendered

    # Test widths attribute
    table.widths = {"A": "10rem", "B": "100px", "C": "auto"}
    rendered = table.to_html()
    assert "width: 10rem" in rendered
    assert "width: 100px" in rendered
    assert "width: auto" in rendered

    # Test relative widths attribute
    table.widths = {"A": 1, "B": 2, "C": 2}
    rendered = table.to_html()
    assert "width: 20.0%" in rendered
    assert "width: 40.0%" in rendered
    assert "width: 40.0%" in rendered

    # Test the title attribute
    table.title = "Test Table"
    rendered = table.to_html()
    assert "Test Table" in rendered

    # Test the table_colors attribute
    table.table_colors = TableColors(header_even="#123456")
    rendered = table.to_html()
    assert "background-color: #123456" in rendered

    # Test the table_styles attribute
    table.table_css = {"th": {"color": "#654321"}}
    rendered = table.to_html()
    assert "color: #654321 !important" in rendered


def test_result_table_multi_df():
    """Test ResultTable."""
    # Create a DataFrame
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
            "D": [10, 11, 12],
            "E": [13, 14, 15],
        }
    ).set_index(["D", "E"])

    # Create a different DataFrame
    df2 = pd.DataFrame(
        {
            "A": [10, 20, 30],
            "B": [40, 50, 60],
            "C": [70, 80, 90],
            "D": [11, 12, 13],
            "E": [14, 15, 16],
        }
    ).set_index(["D", "E"])

    # Merge the two DataFrames
    merge_df = pd.concat({"df1": df, "df2": df2}, axis="columns")

    # Create a ResultTable
    table = ResultTable(merge_df, show_index=True)
    table.styles.append(TableStyle(cols=["E"], hide="cols"))

    # Test the render method
    rendered = table.to_html()
    assert isinstance(rendered, str)
    assert "df1" in rendered
    assert "df2" in rendered
    assert "A" in rendered
    assert "B" in rendered
    assert "C" in rendered
    assert "D" in rendered
    assert "E" not in rendered

    # Test the styles attribute with a col filter
    style = TableStyle(cols=["A"], css={"font-weight": "bold"})
    table.styles.append(style)
    rendered = table.to_html()
    assert "font-weight: bold" in rendered

    # Test the styles attribute with a row filter
    style = TableStyle(rows=(merge_df[("df1", "A")] < 3), css={"color": "blue"})
    table.styles.append(style)
    rendered = table.to_html()
    assert "color: blue" in rendered

    # Test the styles attribute with an exact col
    style = TableStyle(dfs="df1", cols="A", css={"color": "red"})
    table.styles.append(style)
    rendered = table.to_html()
    assert "color: red" in rendered

    # Test the labels attribute
    table.labels = {"A": "Alpha", "B": "Beta", "C": "Gamma"}
    rendered = table.to_html()
    assert "df1" in rendered
    assert "df2" in rendered
    assert "Alpha" in rendered
    assert "Beta" in rendered
    assert "Gamma" in rendered

    # Test widths attribute
    table.widths = {"A": "10rem", "B": "100px", "C": "auto"}
    rendered = table.to_html()
    assert "width: 10rem" in rendered
    assert "width: 100px" in rendered
    assert "width: auto" in rendered

    # Test the title attribute
    table.title = "Test Table"
    rendered = table.to_html()
    assert "Test Table" in rendered

    # Test column flatteting
    table.column_flatten_format = "{0}: {1}"
    rendered = table.to_html()
    assert "df1: Alpha" in rendered
    assert "df1: Beta" in rendered
    assert "df2: Gamma" in rendered
