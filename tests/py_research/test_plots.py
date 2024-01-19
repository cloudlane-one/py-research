"""Test plots module."""

from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from py_research.plots import export_html, with_dropdown


def test_export_html():
    """Test export_html."""
    # Create a simple figure
    fig = go.Figure(data=[{"type": "scatter", "x": [1, 2, 3], "y": [4, 5, 6]}])

    # Test with a Path object
    path = Path(gettempdir()) / "test_file.html"
    try:
        export_html(fig, path)
        assert path.is_file()
    finally:
        path.unlink()  # Clean up


def plot_func(df: pd.DataFrame) -> go.Figure:
    """Plot a scatter plot of GDP per capita vs population with a slider for year."""
    return px.scatter(
        df,
        x="pop",
        y="gdpPercap",
        color="country",
        size="lifeExp",
        animation_frame="year",
    )


def test_with_dropdown():
    """Test ``with_dropdown`` decorator.""" ""
    df = px.data.gapminder()
    plot_func_wd = with_dropdown("continent")(plot_func)

    fig = plot_func_wd(df)
    assert isinstance(fig, go.Figure)
