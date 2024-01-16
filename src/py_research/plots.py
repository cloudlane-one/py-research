"""Utilities for working with common plots."""

from pathlib import Path
from typing import Literal, TextIO, TypeAlias

from plotly.graph_objects import Figure as PlotlyFigure

ImageFormat: TypeAlias = Literal["svg", "pdf", "png", "jpg", "webp"]


def export_html(
    fig: PlotlyFigure,
    out: Path | str | TextIO,
    width: int = 800,
    height: int = 450,
    scale: float = 3,
    name: str = "plot",
    download_format: ImageFormat = "svg",
):
    """Save plotly figure to interactive html.

    Args:
        fig: Plotly figure to save.
        out: Path to save to or file-like object to write to.
        width: Width of the figure in pixels.
        height: Height of the figure in pixels.
        scale: Scale factor for the figure.
        name: Name of the figure.
        download_format: Format to use when downloading the figure.
    """
    fig.write_html(
        out,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "responsive": True,
            "toImageButtonOptions": {
                "format": download_format,
                "filename": name,
                "height": height,
                "width": width,
                "scale": scale,
            },
        },
    )
