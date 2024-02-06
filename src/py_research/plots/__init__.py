"""Utilities for working with common plots."""

from collections.abc import Callable, Mapping
from functools import reduce, wraps
from pathlib import Path
from textwrap import dedent, indent
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
                download_scale=scale,
                download_format=download_format,
            )
        )


ResponsiveMode: TypeAlias = Literal[None, "keep-aspect", "stretch-width"]

default_responsiveness: dict[ResponsiveMode, int | None] = {
    None: 400,
    "keep-aspect": 720,
    "stretch-width": 1080,
}

with open(Path(__file__).parent / "responsive_plotly.js") as f:
    plotly_responsive_js: str = f.read()


def _ind(text: str, count: int) -> str:
    """Indent all lines in text except for first by count spaces."""
    return indent(text, " " * count).lstrip(" " * count)


def plotly_to_html(
    fig: go.Figure,
    write_to: Path | str | TextIO | None = None,
    responsive: bool = True,
    min_width: int = 500,
    max_width: int = 1500,
    max_height: int = 600,
    download_format: ImageFormat = "svg",
    download_scale: float = 3,
    full_html: bool = True,
    plotly_js_url: str
    | None = "https://cdn.jsdelivr.net/npm/plotly.js@2/dist/plotly.min.js",
) -> str:
    """Convert plotly figure to interactive (and responsive) html.

    Args:
        fig: Plotly figure to save.
        write_to: Path to save to or file-like object to write to.
        responsive: Whether to make the figure responsive.
        min_width: Minimum width of the figure in pixels.
        max_width: Maximum width of the figure in pixels.
        max_height: Maximum height of the figure in pixels.
        download_format: Format to use when downloading the figure.
        download_scale: Scale factor for the figure resolution when downloading.
        full_html: Whether to wrap the figure in a full html document.
        plotly_js_url:
            URL to load plotly.js library from.
            Defaults to the latest version on jsdelivr.
            Set to None to leave out the plotly.js script tag.

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
    script_id = str(uuid4())[:10]

    layout = cast(go.Layout, fig.layout).to_plotly_json()
    width = layout.get("width") or 720
    height = layout.get("height") or 480

    # Render the figure to html.
    fig_html: str = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=fig_id,
        config={
            "toImageButtonOptions": {
                "format": download_format,
                "filename": name,
                "height": height,
                "width": width,
                "scale": download_scale,
            },
        },
    )

    res_html = fig_html
    if responsive:
        # Define attributes to pass to the script tag.
        script_attrs = {
            "plotly-js-url": plotly_js_url,
            "fig-id": fig_id,
            "fig-width": width,
            "fig-height": height,
            "min-width": min_width,
            "max-width": max_width,
            "max-height": max_height,
        }

        script_attr_str = " ".join(
            f'{attr}="{value}"' for attr, value in script_attrs.items()
        )

        plotly_js_script = (
            dedent(
                f"""
            <script src="{plotly_js_url}"></script>
            """
            )
            if plotly_js_url is not None
            else ""
        )

        # Combine the figure html with a script tag that makes it responsive,
        # and optionally with the plotly.js script tag.
        res_html = dedent(
            f"""
            <div class="plotly-responsive-container" style="width: 100%;">
                {_ind(plotly_js_script, 16)}
                {_ind(fig_html, 16)}
                <script
                    id="{script_id}"
                    {_ind(script_attr_str, 20)}
                >
                    const scriptID = "{script_id}";
                    {_ind(plotly_responsive_js, 20)}
                </script>
            </div>
            """
        )

    # Wrap the figure in a full html document if requested.
    html = dedent(
        f"""
        <!doctype html>
        <html>
            <head>
                <title>{name}</title>
            </head>
            <body>
                {_ind(res_html, 16)}
            </body>
            </html>
         """
        if full_html
        else res_html
    )

    # Write the html to a file if requested.
    if write_to is not None:
        with open(write_to, "w") if isinstance(write_to, Path | str) else write_to as f:
            f.write(html)

    return html


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

            # Show only the first group.
            first_group = list(figs.keys())[0]
            first_group_trace_count = trace_sources[trace_sources == first_group].size
            for i, trace in enumerate(split_fig.select_traces()):
                if i < first_group_trace_count:
                    trace.visible = True
                else:
                    trace.visible = False

            return split_fig

        return wrapper

    return decorator
