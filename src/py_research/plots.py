"""Utilities for working with common plots."""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias, cast, overload

from PIL.Image import Image
from PIL.Image import open as open_image
from plotly.graph_objects import Figure as PlotlyFigure
from plotly.graph_objects import Frame

ImageFormat: TypeAlias = Literal["svg", "pdf", "png", "jpg", "webp"]


@overload
def plot_to_html(
    fig: PlotlyFigure,
    path: Path | str,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    name: str = "plot",
    download_format: ImageFormat = "svg",
) -> None:
    ...


@overload
def plot_to_html(
    fig: PlotlyFigure,
    path: None,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    name: str = "plot",
    download_format: ImageFormat = "svg",
) -> str:
    ...


def plot_to_html(
    fig: PlotlyFigure,
    path: Path | str | None = None,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    name: str = "plot",
    download_format: ImageFormat = "svg",
) -> str | None:
    """Save plotly figure to interactive html."""
    return fig.to_html(
        path,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "responsive": True,
            "toImageButtonOptions": {
                "format": download_format,
                "filename": f"{name}.svg",
                "height": height,
                "width": width,
                "scale": scale,
            },
        },
    )


@overload
def plot_to_image(
    fig: PlotlyFigure,
    out: Path | str | BinaryIO,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    format: ImageFormat = "svg",
) -> None:
    ...


@overload
def plot_to_image(
    fig: PlotlyFigure,
    out: None = None,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    format: ImageFormat = "svg",
) -> bytes:
    ...


def plot_to_image(
    fig: PlotlyFigure,
    out: Path | str | BinaryIO | None = None,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    format: ImageFormat = "svg",
) -> bytes | None:
    """Save plotly figure to static image."""
    res = fig.to_image(out, format=format, width=width, height=height, scale=scale)
    return res if out is None else None


@overload
def plot_to_gif(
    fig: PlotlyFigure,
    out: Path | str | BinaryIO,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    ms_per_frame: int | None = None,
) -> None:
    ...


@overload
def plot_to_gif(
    fig: PlotlyFigure,
    out: None,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    ms_per_frame: int | None = None,
) -> bytes:
    ...


def plot_to_gif(
    fig: PlotlyFigure,
    out: Path | str | BinaryIO | None = None,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
    ms_per_frame: int | None = None,
) -> bytes | None:
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

    res = out or BytesIO()

    # Create the gif file.
    frames[0].save(
        res,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=(
            ms_per_frame
            or fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]
        ),
        loop=0,
    )

    return res.getvalue() if isinstance(res, BytesIO) else None
