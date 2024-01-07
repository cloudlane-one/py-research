"""Test tables module."""

import pandas as pd

from py_research.tables import ResultTable, TableStyle, to_html


def test_result_table():
    """Test ResultTable."""
    # Create a DataFrame
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Create a ResultTable
    table = ResultTable(df)

    # Test the render method
    rendered = to_html(table.to_styled_df())
    assert isinstance(rendered, str)
    assert "A" in rendered
    assert "B" in rendered
    assert "C" in rendered

    # Test the styles attribute
    style = TableStyle(cols=["A"], css={"font-weight": "bold"})
    table.styles.append(style)
    rendered = to_html(table.to_styled_df())
    assert "font-weight: bold;" in rendered

    # Test the labels attribute
    table.labels = {"A": "Alpha", "B": "Beta", "C": "Gamma"}
    rendered = to_html(table.to_styled_df())
    assert "Alpha" in rendered
    assert "Beta" in rendered
    assert "Gamma" in rendered

    # Test widths attribute
    table.widths = {"A": "10rem", "B": "100px", "C": "auto"}
    rendered = to_html(table.to_styled_df())
    assert "width: 10rem;" in rendered
    assert "width: 100px;" in rendered
    assert "width: auto;" in rendered

    # Test relative widths attribute
    table.widths = {"A": 1, "B": 2, "C": 2}
    rendered = to_html(table.to_styled_df())
    assert "width: 20.0%;" in rendered
    assert "width: 40.0%;" in rendered
    assert "width: 40.0%;" in rendered

    # Test the title attribute
    table.title = "Test Table"
    rendered = to_html(table.to_styled_df())
    assert "Test Table" in rendered
