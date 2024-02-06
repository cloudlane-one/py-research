"""Utilities for creating pretty result tables."""

from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Any, Literal, TextIO, cast

import imgkit
import pandas as pd
import pdfkit
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype
from pandas.io.formats.style import Styler
from typing_extensions import deprecated


@dataclass
class TableStyle:
    """Define a pretty table column format."""

    dfs: str | list[str] | None = None
    """Dataframe names to apply this style to. If None, apply to all dataframes."""

    cols: str | list[str] | None = None
    """Column name(s) to apply this style to. If None, apply to all columns.
    Index levels can be styled via their names as if they were columns.
    """

    rows: pd.Series | None = None
    """Rows to apply this style to. If None, apply to all rows."""

    name: str | None = None
    """Name of this style"""

    format: str | Callable[[Any], str] | None = None
    """Format string or callback function to use for formatting data values"""

    css: dict[str, str] | Callable[[Any], str] | None = None
    """Custom CSS styles to apply to matched cells.
    Supplied as a mapping from CSS property to value.
    """

    header_css: dict[str, str] | None = None
    """Custom CSS styles applied to the headers of matched cells.
    Supplied as a mapping from CSS property to value.
    """

    hide: Literal["headers", "rows", "cols", "cells"] | None = None
    """Whether to hide headers, rows, columns or cells matched by this style"""

    include: bool | None = None
    """Whether to show rows if matched by this selection."""

    require: bool | None = None
    """Whether to only show rows if matched by this selection."""


@dataclass
class TableColors:
    """Define colors for a pretty table."""

    row_even: str = "#f2f2f2"
    """Background color for even rows."""

    row_odd: str = "#fff"
    """Background color for odd rows."""

    header_even: str = "#677a96"
    """Background color for even headers."""

    header_odd: str = "#677a96"
    """Background color for odd headers."""


@dataclass
class ResultTable:
    """Define and render a pretty result table with custom formatting and highlights."""

    df: InitVar[pd.DataFrame]
    """Dataframe to render as pretty table."""

    styles: list[TableStyle] = field(default_factory=list)
    """Styles to apply to the table."""

    labels: dict[str, str] | Callable[[str], str] = field(default_factory=dict)
    """Labels to use for the table headers.

    If dict, the keys are column names and the values are the labels.
    If callable, the function is called with the column name and should return the label
    """

    widths: dict[str, float | str] = field(default_factory=dict)
    """Widths to use for the table columns.

    If a float, the width is relative to the sum of all widths.
    If a string, the width is set to the CSS string value.
    """

    title: str | None = None
    """Title of the table."""

    max_row_cutoff: int = 100
    """Maximum number of rows to render."""

    font_size: float = 1.0
    """Default font size to use for the table."""

    default_style: TableStyle = field(default_factory=TableStyle)
    """Default style to apply to the table."""

    column_flatten_format: str | None = None
    """Format string to use for flatteting multi-index column labels.
    Leave as None to keep a multi-level column header.
    Format string must take two positional arguments, e.g. "{0}_{1}".
    """

    show_index: bool = False
    """Whether to show the index as the first columns of the styled table."""

    table_colors: TableColors = field(default_factory=TableColors)
    """Colors to use for this table."""

    table_css: dict[str, dict[str, str]] = field(default_factory=dict)
    """Additional styles to apply to the table.
    Dictionary keys must be CSS selectors and
    values must be dictionaries of CSS properties.
    """

    def __post_init__(self, df: pd.DataFrame):  # noqa: D105
        if not self.show_index:
            self.data = df
            return

        index_names = (
            df.index.names
            if all(df.index.names)
            else [f"index_{i}" for i in range(df.index.nlevels)]
        )

        df = df.rename_axis(index=index_names)

        index_col_names = [name for name in index_names]
        if df.columns.nlevels > 1:
            index_col_names = [("", name) for name in index_col_names]

        df = df.copy()
        for col_name in index_col_names:
            index_name = col_name if isinstance(col_name, str) else col_name[1]
            df[col_name] = df.index.get_level_values(index_name)

        self.data = df[
            [*index_col_names, *[c for c in df.columns if c not in index_col_names]]
        ]

    @property
    def _hidden_headers(self) -> set[tuple[str | None, str]]:
        hide_styles = [
            set(
                product(
                    style.dfs if isinstance(style.dfs, list) else [style.dfs],
                    style.cols if isinstance(style.cols, list) else [style.cols],
                )
            )
            for style in self.styles
            if style.cols is not None and style.hide == "headers"
        ]

        return (
            reduce(
                lambda x, y: x | y,
                hide_styles,
            )
            if len(hide_styles) > 0
            else set()
        )

    def html_description(self, full_doc: bool = True) -> str:
        """Return HTML description for table filters and highlights.

        Args:
            full_doc: Whether to wrap the description in a full HTML document.

        Returns:
            HTML code for the description.
        """
        highlights = [
            (
                h,
                (
                    "; ".join(f"{prop}: {val}" for prop, val in h.css.items())
                    if isinstance(h.css, dict)
                    else (
                        str(getattr(h.css, "__name__"))
                        if hasattr(h.css, "__name__")
                        else None
                    )
                ),
            )
            for h in self.styles
        ]

        exclusive_h = [
            (h, css)
            for h, css in highlights
            if h.name is not None and css is not None and h.require
        ]
        inclusive_h = [
            (h, css)
            for h, css in highlights
            if h.name is not None and css is not None and h.include
        ]
        highlight_h = [
            (h, css)
            for h, css in highlights
            if h.name is not None
            and css is not None
            and not any([h.require, h.include, h.hide in ["rows", "cells"]])
        ]
        hide_h = [h for h, _ in highlights if h.name is not None and h.hide is not None]

        desc = "\n".join(
            [
                (
                    f"""<h2>Show rows where all of:</h2>
            <ul>
                {
                    ''.join(
                        [
                            '<li style="margin-bottom: 0.5rem;">'
                            + f'<span style="display: inline-block; {css};">'
                            + css
                            + '</span>:'
                            + f' {h.name}'
                            + '</li>'
                            for h, css in exclusive_h
                        ]
                    )
                }
            </ul>
            """
                    if len(exclusive_h) > 0
                    else ""
                ),
                (
                    f"""<h2>Show rows where any of:</h2>
            <ul>
                {
                    ''.join(
                        [
                            '<li style="margin-bottom: 0.5rem;">'
                            + f'<span style="display: inline-block; {css};">'
                            + css
                            + '</span>:'
                            + f' {h.name}'
                            + '</li>'
                            for h, css in inclusive_h
                        ]
                    )
                }
            </ul>
            """
                    if len(inclusive_h) > 0
                    else ""
                ),
                (
                    f"""<h2>Highlight rows where:</h2>
            <ul>
                {
                    ''.join(
                        [
                            '<li style="margin-bottom: 0.5rem;">'
                            + f'<span style="display: inline-block; {css};">'
                            + css
                            + '</span>:'
                            + f' {h.name}'
                            + '</li>'
                            for h, css in highlight_h
                        ]
                    )
                }
            </ul>
            """
                    if len(highlight_h) > 0
                    else ""
                ),
                (
                    f"""<h2>Hide:</h2>
            <ul>
                {
                    ''.join(
                        [
                            '<li style="margin-bottom: 0.5rem;">'
                            + f'{h.hide} where {h.name}'
                            + '</li>'
                            for h in hide_h
                        ]
                    )
                }
            </ul>
            """
                    if len(hide_h) > 0
                    else ""
                ),
            ]
        )

        return (
            f"""
        <!doctype html>
        <html>
            <head>
                <title>{self.title} - highlight-description</title>
            </head>
            <body style="font-family: sans-serif">
                <h1>Description for table '{self.title}'</h1>
                {desc}
            </body>
        </html>
        """
            if full_doc
            else desc
        )

    def _get_cols_by_second_level(self, cols: list[str]) -> list[tuple[str, str]]:
        return [
            cast(tuple[str, str], cc)
            for c in cols
            for cc in self.data.columns
            if (isinstance(c, str) and cc[1] == c) or cc == c
        ]

    def _to_styler_col(self, col: str | tuple[str, str]) -> str | tuple[str, str]:
        return (
            col
            if isinstance(col, str)
            or self.data.columns.nlevels == 1
            or self.column_flatten_format is None
            else self.column_flatten_format.format(*col)
        )

    def _default_str_format(self, col: str) -> str:
        all_col_data = (
            self.data.loc[
                :,
                self._get_cols_by_second_level([col]),
            ]
            if self.data.columns.nlevels > 1
            else self.data[[col]]
        )
        if all(is_integer_dtype(col_data) for _, col_data in all_col_data.items()):
            return "{:d}"
        elif all(is_float_dtype(col_data) for _, col_data in all_col_data.items()):
            if all(
                col_data.map(float.is_integer).all()
                for _, col_data in all_col_data.items()
            ):
                return "{:.0f}"
            else:
                return "{:.3f}"
        else:
            return "{}"

    def _default_alignment(self, col: str) -> str:
        all_col_data = (
            self.data.loc[
                :,
                self._get_cols_by_second_level([col]),
            ]
            if self.data.columns.nlevels > 1
            else self.data[[col]]
        )
        return (
            "right"
            if all(is_numeric_dtype(col_data) for _, col_data in all_col_data.items())
            else "left"
        )

    def _prettify_df(self, table: pd.DataFrame, font_size: float = 1.0) -> Styler:
        """Apply styles to a DataFrame or Styler and make it pretty."""
        return table.style.set_table_styles(
            [
                {
                    "selector": "",
                    "props": (
                        "font-family: Arial, Helvetica, sans-serif;"
                        "border-collapse: collapse; width: 100%;"
                        "color: black; text-align: left;"
                    ),
                },
                {
                    "selector": "caption",
                    "props": (
                        f"margin-bottom: 1rem;" f"font-size: {(font_size*1.5):.3g}rem;"
                    ),
                },
                {
                    "selector": "td, th",
                    "props": (
                        f"font-size: {font_size:.3g}rem;"
                        "border: 1px solid #ddd;"
                        "padding: 8px;"
                    ),
                },
                {
                    "selector": "tr:nth-child(even)",
                    "props": f"background-color: {self.table_colors.row_even};",
                },
                {
                    "selector": "tr:nth-child(odd)",
                    "props": f"background-color: {self.table_colors.row_odd};",
                },
                {
                    "selector": "th",
                    "props": (
                        "padding: 12px 8px 12px;" "text-align: left;" "color: black;"
                    ),
                },
                {
                    "selector": "th:nth-child(even)",
                    "props": f"background-color: {self.table_colors.header_even};",
                },
                {
                    "selector": "th:nth-child(odd)",
                    "props": f"background-color: {self.table_colors.header_odd};",
                },
                *(
                    {
                        "selector": selector,
                        "props": " ".join(
                            [
                                prop_name
                                + ": "
                                + prop_value.rstrip(" ").rstrip(";")
                                + " !important;"
                                for prop_name, prop_value in props.items()
                            ]
                        ),
                    }
                    for selector, props in self.table_css.items()
                ),
            ]
        ).format(na_rep="")

    def _apply_default_style(self, styled: Styler) -> Styler:
        if isinstance(self.default_style.css, dict):
            styled = styled.set_properties(
                subset=None,
                **self.default_style.css,
            )
        elif isinstance(self.default_style.css, Callable):
            styled = styled.applymap(
                func=self.default_style.css,
            )

        return styled

    def _apply_col_defaults(self, styled: Styler) -> Styler:
        res = styled
        for col in self.data.columns:
            res = res.set_properties(
                subset=[self._to_styler_col(col)],
                **{"text-align": self._default_alignment(col)},
            )

            res = res.format(
                subset=[self._to_styler_col(col)],
                formatter=self.default_style.format or self._default_str_format(col),
            )

            if isinstance(self.default_style.css, dict):
                res = res.set_properties(
                    subset=[self._to_styler_col(col)], **self.default_style.css
                )
            elif isinstance(self.default_style.css, Callable):
                res = res.applymap(
                    self.default_style.css, subset=[self._to_styler_col(col)]
                )

        return res

    def _apply_styles(self, styled: Styler) -> Styler:
        for style in self.styles:
            rows, cols = (
                style.rows if style.rows is not None else slice(None),
                (
                    style.cols
                    if isinstance(style.cols, list)
                    else [style.cols] if style.cols is not None else slice(None)
                ),
            )

            if isinstance(cols, list) and self.data.columns.nlevels > 1:
                cols = [
                    self._to_styler_col(c) for c in self._get_cols_by_second_level(cols)
                ]

            subset = (rows, cols)

            if style.hide is not None:
                if style.hide in ["rows", "cells"]:
                    styled = styled.hide(
                        subset[0],  # type: ignore
                        axis="index",
                    )
                if style.hide in ["cols", "cells"]:
                    styled = styled.hide(
                        subset[1],  # type: ignore
                        axis="columns",
                    )
                continue

            if isinstance(style.css, dict):
                styled = styled.set_properties(
                    subset=subset,  # type: ignore
                    **style.css,
                )
            elif isinstance(style.css, Callable):
                styled = styled.applymap(
                    subset=subset,  # type: ignore
                    func=style.css,
                )

            if style.format is not None:
                styled = styled.format(
                    subset=subset,  # type: ignore
                    formatter=(
                        f"{{:{style.format}}}"
                        if isinstance(style.format, str)
                        else style.format
                    ),
                )

        return styled

    def _apply_widths(self, styled: Styler) -> Styler:
        widths = {
            c: (self.widths.get(c if isinstance(c, str) else c[1]) or 1)
            for c in self.data.columns
        }
        width_sum = sum(w for w in widths.values() if isinstance(w, int | float))
        for col, width in widths.items():
            styled = styled.set_properties(
                subset=(slice(None), [self._to_styler_col(col)]),  # type: ignore
                width=(
                    f"{(width / width_sum * 100):.1f}%"
                    if isinstance(width, int | float)
                    else width
                ),
            )

        return styled

    def _apply_labels(self, styled: Styler) -> Styler:
        label_func = self.labels.get if isinstance(self.labels, dict) else self.labels
        labels = (
            {
                c: "" if (None, c) in self._hidden_headers else label_func(c) or c
                for c in self.data.columns
            }
            if self.data.columns.nlevels == 1
            else {
                c: (
                    ("", "")
                    if c in self._hidden_headers or (None, c[1]) in self._hidden_headers
                    else (
                        (c[0], label_func(c[1]) or c[1])
                        if c[1]
                        else ("", label_func(c[0]) or c[0])
                    )
                )
                for c in self.data.columns
            }
        )

        if self.column_flatten_format is not None and self.data.columns.nlevels > 1:
            labels = {
                c: (
                    self.column_flatten_format.format(*label)
                    if c[1] not in self.data.index.names
                    else label[1]
                )
                for c, label in labels.items()
            }

        return styled.relabel_index(
            list(labels.values()),  # type: ignore
            axis="columns",
        )

    def to_styled_df(self) -> Styler:
        """Render table to styled pandas dataframe."""
        data = self.data.copy()

        if self.column_flatten_format is not None and self.data.columns.nlevels > 1:
            data.columns = [self.column_flatten_format.format(*c) for c in data.columns]

        incl_filters = [
            style.rows
            for style in self.styles
            if style.include is True and style.rows is not None
        ]
        if len(incl_filters) > 0:
            data = data.loc[reduce(lambda x, y: x | y, incl_filters)]

        excl_filters = [
            style.rows
            for style in self.styles
            if style.require is True and style.rows is not None
        ]
        if len(excl_filters) > 0:
            data = data.loc[reduce(lambda x, y: x & y, excl_filters)]

        styled = self._prettify_df(data.iloc[: self.max_row_cutoff], self.font_size)
        styled = self._apply_default_style(styled)
        styled = self._apply_col_defaults(styled)
        styled = self._apply_widths(styled)
        styled = self._apply_labels(styled)
        styled = self._apply_styles(styled)

        if self.title is not None:
            styled = styled.set_caption(self.title)

        styled = styled.hide(axis="index")

        return styled

    def to_html(
        self, write_to: Path | str | TextIO | None = None, full_html: bool = True
    ) -> str:
        """Return HTML representation and optionally write it to a file.

        Args:
            styled: Styled dataframe to render.
            write_to: File to write the HTML code to.
            full_html: Whether to wrap the table in a full HTML document.

        Returns:
            HTML code for the table.
        """
        styled = self.to_styled_df()
        html = (
            f"""
            <!doctype html>
            <html>
                <head>
                    <title>{getattr(styled, "caption") or ""}</title>
                </head>
                <body>
                    {styled.to_html(escape=False)}
                </body>
            </html>
            """
            if full_html
            else styled.to_html(escape=False)
        )

        if write_to is not None:
            with (
                open(write_to, "w") if isinstance(write_to, Path | str) else write_to
            ) as f:
                f.write(html)

        return html

    def _repr_html_(self) -> str:
        return self.to_html(full_html=False)


@deprecated("Use `ResultTable.to_html` instead.")
def to_html(styled: Styler, full_doc: bool = True) -> str:
    """Return HTML representation of a pretty table.

    Args:
        styled: Styled dataframe to render.
        full_doc: Whether to wrap the table in a full HTML document.

    Returns:
        HTML code for the table.
    """
    return (
        f"""
        <!doctype html>
        <html>
            <head>
                <title>{getattr(styled, "caption") or ""}</title>
            </head>
            <body>
                {styled.to_html(escape=False)}
            </body>
        </html>
        """
        if full_doc
        else styled.to_html(escape=False)
    )


@deprecated("Use the `render` module instead.")
def html_to_pdf(doc: str, file: Path):
    """Render and save HTML ``doc`` as PDF document.

    Args:
        doc: HTML document to render.
        file: File to save the rendered PDF document to.
    """
    pdfkit.from_string(doc, file)


@deprecated("Use the `render` module instead.")
def html_to_image(doc: str, file: Path):
    """Render and save HTML ``doc`` as PNG image.

    Args:
        doc: HTML document to render.
        file: File to save the rendered image to.
    """
    imgkit.from_string(doc, file, options={"zoom": "3.125", "width": "3125"})
