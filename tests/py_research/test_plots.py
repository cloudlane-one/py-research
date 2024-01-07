"""Test plots module."""

from pathlib import Path
from tempfile import gettempdir

from plotly.graph_objects import Figure
from py_research.plots import export_html


def test_export_html():
    """Test export_html."""
    # Create a simple figure
    fig = Figure(data=[{"type": "scatter", "x": [1, 2, 3], "y": [4, 5, 6]}])

    # Test with a Path object
    path = Path(gettempdir()) / "test_file.html"
    try:
        export_html(fig, path)
        assert path.is_file()
    finally:
        path.unlink()  # Clean up
