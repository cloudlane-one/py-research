"""Utilities for working with common plots."""

from collections.abc import Callable, Mapping
from functools import reduce, wraps
from pathlib import Path
from textwrap import indent
from typing import Any, Concatenate, Literal, ParamSpec, TextIO, TypeAlias, cast
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
from typing_extensions import deprecated

ImageFormat: TypeAlias = Literal["svg", "pdf", "png", "jpg", "webp"]


@deprecated("Use `plotly_to_html` instead.")
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
    fig = fig.update_layout(
        title=name,
    )

    with open(out, "w") if isinstance(out, Path | str) else out as f:
        f.write(
            plotly_to_html(
                fig=fig,
                default_width=width,
                default_height=height,
                download_scale=scale,
                download_format=download_format,
            )
        )


ResponsiveMode: TypeAlias = Literal[
    None, "scale-image", "scale-layout", "stretch-width"
]

default_responsiveness: dict[ResponsiveMode, int | None] = {
    None: 240,
    "scale-image": 640,
    "stretch-width": 1080,
    "scale-layout": 2160,
}

with open(Path(__file__).parent / "plotly_responsive.js") as f:
    plotly_responsive_js: str = f.read()


def _ind(text: str, count: int) -> str:
    """Indent all lines in text except for first by count spaces."""
    return indent(text, " " * count).lstrip(" " * count)


def plotly_to_html(
    fig: go.Figure,
    resizing: dict[ResponsiveMode, int | None] | None = default_responsiveness,
    default_width: int = 720,
    default_height: int = 540,
    download_format: ImageFormat = "svg",
    download_scale: float = 3,
    full_html: bool = True,
    plotly_js_url: str = "https://cdn.jsdelivr.net/npm/plotly.js@2/+esm",
) -> str:
    """Convert plotly figure to interactive (and responsive) html.

    Args:
        fig: Plotly figure to save.
        resizing:
            Responsive sizing modes to use.
            Accepts a dict mapping responsive modes to max widths in pixels.
            A mode is used up until that width, and then it switches to next
            or defaults to no responsiveness if no more modes are available.
            Responsiveness can be disabled by setting this to ``None``.
            Available modes are:
                - ``None``: No responsiveness.
                - ``"scale-image"``:
                    Scale everything to fit the container,
                    meaning text and markers will be scaled as well.
                    Aspect ratio is maintained.
                - ``"stretch-width"``:
                    Increase the figure width to fit the container.
                    Text and markers will not be scaled.
                    Aspect ratio is not maintained.
                - ``"scale-layout"``:
                    Scale the layout to fit the container.
                    Text and markers will not be scaled.
                    Aspect ratio is maintained.
        default_width:
            Default width of the figure in pixels.
            Used if responsiveness is disabled or when downloading an image.
        default_height:
            Default height of the figure in pixels.
            Used if responsiveness is disabled or when downloading an image.
        download_format: Format to use when downloading the figure.
        download_scale: Scale factor for the figure resolution when downloading.
        full_html: Whether to wrap the figure in a full html document.
        plotly_js_url:
            URL to load plotly.js library from.
            Must resolve to an ES module.

    Returns:
        HTML string.
    """
    # Get the name of the figure,
    # either from the figure itself or from the supplied argument.
    # Use fallback name "plot" if neither is available.
    name = (
        cast(dict, cast(go.Layout, fig.layout).to_plotly_json().get("title") or {}).get(
            "text"
        )
        or "plot"
    )

    fig_id = str(uuid4())[:10]

    fig = fig.update_layout(
        width=default_width,
        height=default_height,
    )

    # Render the figure to html.
    fig_html: str = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=fig_id,
        config={
            "toImageButtonOptions": {
                "format": download_format,
                "filename": name,
                "height": default_height,
                "width": default_width,
                "scale": download_scale,
            },
        },
    )

    res_html = fig_html
    if resizing is not None:
        sorted_sizing = dict(
            sorted(
                resizing.items(),
                key=lambda item: item[1] if item[1] is not None else float("inf"),
            )
        )

        script_attrs = {
            "plotly-js-url": plotly_js_url,
            "fig-id": fig_id,
            "fig-width": default_width,
            "fig-height": default_height,
            **{
                f"fig-bp-{prop}-{i}": value if value is not None else "inf"
                for i, (mode, width) in enumerate(sorted_sizing.items())
                for prop, value in [("mode", mode), ("width", width)]
            },
        }

        script_attr_str = " ".join(
            f'{attr}="{value}"' for attr, value in script_attrs.items()
        )

        res_html = f"""
        <div class="plotly-responsive-container" style="width: 100%;">
            <script
                type="module"
                {_ind(script_attr_str, 8)}
            >
                {_ind(plotly_responsive_js, 8)}
                {_ind(fig_html, 8)}
            </script>
        </div>
        """

    return (
        f"""
            <!doctype html>
            <html>
                <head>
                    <title>{name}</title>
                </head>
                <body>
                    {_ind(res_html, 8)}
                </body>
            </html>
            """
        if full_html
        else res_html
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
