"""Utilities for working with common plots."""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal, TextIO, TypeAlias, cast

from PIL.Image import Image
from PIL.Image import open as open_image
from plotly.graph_objects import Figure as PlotlyFigure
from plotly.graph_objects import Frame

ImageFormat: TypeAlias = Literal["svg", "pdf", "png", "jpg", "webp"]


def export_html(
    fig: PlotlyFigure,
    out: Path | str | TextIO,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    name: str = "plot",
    download_format: ImageFormat = "svg",
):
    """Save plotly figure to interactive html."""
    fig.write_html(
        out,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "responsive": True,
            "toImageButtonOptions": {
                "format": download_format,
                "filename": f"{name}.{download_format}",
                "height": height,
                "width": width,
                "scale": scale,
            },
        },
    )


def export_image(
    fig: PlotlyFigure,
    out: Path | str | BinaryIO,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    format: ImageFormat = "svg",
):
    """Save plotly figure to static image."""
    fig.write_image(out, format=format, width=width, height=height, scale=scale)


def export_gif(
    fig: PlotlyFigure,
    out: Path | str | BinaryIO,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    ms_per_frame: int | None = None,
):
    """Save plotly figure with slider to animated GIF."""
    fig = PlotlyFigure(fig)
    fig["layout"].pop("updatemenus")  # type: ignore

    frames: list[Image] = []

    for slider_pos, frame in enumerate(fig.frames):
        assert isinstance(frame, Frame)
        frame = cast(Frame, frame)

        fig.update(data=frame.data)
        fig.layout.sliders[0].update(active=slider_pos)
        frames.append(
            open_image(
                BytesIO(
                    fig.to_image(format="png", width=width, height=height, scale=scale)
                )
            )
        )

    # Create the gif file.
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=(
            ms_per_frame
            or fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]
        ),
        loop=0,
    )
