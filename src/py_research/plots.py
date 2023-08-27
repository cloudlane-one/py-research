"""Utilities for working with common plots."""

import io
from pathlib import Path
from typing import cast

from PIL.Image import Image
from PIL.Image import open as open_image
from plotly.graph_objects import Figure as PlotlyFigure
from plotly.graph_objects import Frame


def plot_to_gif(
    fig: PlotlyFigure,
    path: Path | str,
    ms_per_frame: int | None = None,
    width: int = 800,
    height: int = 600,
    scale: float = 3,
):
    """Save plotly figure with slider to GIF."""
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
                io.BytesIO(
                    fig.to_image(format="png", width=width, height=height, scale=scale)
                )
            )
        )

    # Create the gif file.
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=(
            ms_per_frame
            or fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]
        ),
        loop=0,
    )
