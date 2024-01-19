"""Utilities for working with common plots."""

from collections.abc import Callable, Mapping
from functools import reduce, wraps
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, TextIO, TypeAlias, cast

import pandas as pd
import plotly.graph_objects as go

ImageFormat: TypeAlias = Literal["svg", "pdf", "png", "jpg", "webp"]


def export_html(
    fig: go.Figure,
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


P = ParamSpec("P")
PlottingFunction = Callable[Concatenate[pd.DataFrame, P], go.Figure]


def _merge_data(figs: Mapping[str, go.Figure | go.Frame]) -> list:
    return [d for fig in figs.values() for d in (fig.data or [])]


def _merge_layout(figs: Mapping[str, go.Figure | go.Frame]) -> dict[str, Any]:
    return reduce(
        lambda a, b: {**a, **b},
        [
            cast(dict, fig.layout.to_plotly_json())  # type: ignore
            for fig in figs.values()
        ],
    )


def _get_frames(figs: Mapping[str, go.Figure], label: str) -> dict[str, go.Frame]:
    return {
        group: frame
        for group, fig in figs.items()
        for f in fig.frames
        if (frame := cast(go.Frame, f)).name == label
    }


go.Frame


def with_dropdown(
    group_by: str,
    dropdown_kwars: dict[str, Any] | None = None,
) -> Callable[[PlottingFunction[P]], PlottingFunction[P]]:
    """Add a dropdown to a plotting function.

    Args:
        group_by: Column to group by for the dropdown.
        dropdown_kwars: Keyword arguments to pass to the dropdown layout.

    Returns:
        Decorator that adds a dropdown to the returned plot.
    """

    def decorator(func: PlottingFunction[P]) -> PlottingFunction[P]:
        """Add a dropdown to a plotting function."""

        @wraps(func)
        def wrapper(data: pd.DataFrame, *args: P.args, **kwargs: P.kwargs) -> go.Figure:
            figs = {
                str(group): func(sub_data, *args, **kwargs)
                for group, sub_data in data.groupby(group_by)
            }

            fig_data = _merge_data(figs)
            layout = _merge_layout(figs)

            frames = None
            if "sliders" in layout:
                labels = [step["label"] for step in layout["sliders"][0]["steps"]]
                frames = []
                for label in labels:
                    sub_frames = _get_frames(figs, label)
                    frames.append(
                        go.Frame(
                            data=_merge_data(sub_frames),
                            layout=_merge_layout(sub_frames),
                            name=label,
                        )
                    )

            split_fig = go.Figure(
                data=fig_data,
                layout=layout,
                frames=frames,
            )

            trace_sources = pd.Series(
                [group for group, fig in figs.items() for _ in fig.data]
            )

            split_fig.update_layout(
                updatemenus=[
                    *(
                        [split_fig.layout["updatemenus"][0]]  # type: ignore
                        if "updatemenus" in layout
                        else []
                    ),
                    {
                        "active": 0,
                        "buttons": [
                            {
                                "label": "All",
                                "method": "update",
                                "args": [
                                    {
                                        "visible": [True] * len(trace_sources),
                                    },
                                ],
                            },
                            *(
                                {
                                    "label": group,
                                    "method": "update",
                                    "args": [
                                        {
                                            "visible": (
                                                trace_sources == str(group)
                                            ).to_list(),
                                        },
                                    ],
                                }
                                for group in figs.keys()
                            ),
                        ],
                        "direction": "up",
                        "pad": {"r": 10, "t": 70},
                        "showactive": True,
                        "x": 1.1,
                        "xanchor": "left",
                        "y": 0,
                        "yanchor": "top",
                        **(dropdown_kwars or {}),
                    },
                ]
            )

            return split_fig

        return wrapper

    return decorator
