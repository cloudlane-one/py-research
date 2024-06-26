"""Test plots module."""

from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from py_research.plots import export_html, plotly_to_html, with_dropdown


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


def test_export_html():
    """Test export_html."""
    # Create a simple figure
    fig = plot_func(df=px.data.gapminder())

    # Test with a Path object
    path = Path(gettempdir()) / "test_plot_1.html"
    try:
        export_html(fig, path)
        assert path.is_file()
    finally:
        path.unlink()  # Clean up


def test_plotly_to_html():
    """Test plotly_to_html."""
    path = Path(gettempdir()) / "test_plot_2.html"
    # Create a simple figure
    fig = plot_func(df=px.data.gapminder())

    html = plotly_to_html(fig, write_to=path)

    assert isinstance(html, str)
    assert path.is_file()


def test_with_dropdown():
    """Test ``with_dropdown`` decorator."""
    df = px.data.gapminder()
    plot_func_wd = with_dropdown("continent")(plot_func)

    fig = plot_func_wd(df)
    assert isinstance(fig, go.Figure)
