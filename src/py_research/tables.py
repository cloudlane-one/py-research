"""Utilities for creating pretty result tables."""
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Literal

import imgkit
import pandas as pd
import pdfkit
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype
from pandas.io.formats.style import Styler


def _prettify_df(table: pd.DataFrame | Styler, font_size: float = 1.0) -> Styler:
    """Apply styles to a DataFrame or Styler and make it pretty."""
    table = table.style if isinstance(table, pd.DataFrame) else table
    return table.set_table_styles(
        [
            {
                "selector": "",
                "props": "font-family: Arial, Helvetica, sans-serif;"
                "border-collapse: collapse; width: 100%;"
                "color: black; text-align: left;",
            },
            {
                "selector": "caption",
                "props": f"margin-bottom: 1rem; font-size: {(font_size*1.5):.3g}rem;",
            },
            {
                "selector": "td, th",
                "props": f"font-size: {font_size:.3g}rem;"
                "border: 1px solid #ddd; padding: 8px;",
            },
            {
                "selector": "tr:nth-child(even)",
                "props": "background-color: #f2f2f2;",
            },
            {
                "selector": "tr:nth-child(odd)",
                "props": "background-color: #fff;",
            },
            {
                "selector": "th",
                "props": "padding: 12px 8px 12px;"
                "background-color: #677a96; color: black;"
                "text-align: left",
            },
        ]
    ).format(na_rep="")


@dataclass
class TableStyle:
    """Define a pretty table column format."""

    cols: str | list[str] | None = None

    rows: pd.Series | None = None

    name: str | None = None
    """Name of this style"""

    str_format: str | Callable[[Any], str] | None = None
    """Format string or callback function to use for formatting numerical values"""

    alignment: Literal["left", "center", "right"] | None = None
    """How to align the column's text"""

    css: dict[str, str] | Callable[[Any], str] | None = None
    """Custom CSS styles to apply to the column"""

    hide_headers: bool | None = None
    """Whether to hide headers of given ``cols``."""

    hide_rows: bool | None = None
    """Whether to hide cols/rows if matched by this selection."""

    filter_inclusive: bool | None = None
    """Whether to show cols/rows if matched by this selection."""

    filter_exclusive: bool | None = None
    """Whether to only show cols/rows if matched by this selection."""


@dataclass
class ResultTable:
    """Define and render a pretty result table with custom formatting and highlights."""

    df: pd.DataFrame
    styles: list[TableStyle] = field(default_factory=list)
    labels: dict[str, str] | Callable[[str], str] = field(default_factory=dict)
    widths: dict[str, float | str] = field(default_factory=dict)
    title: str | None = None
    hide_index: bool = True
    max_row_cutoff: int = 100
    font_size: float = 1.0
    default_style: TableStyle = field(default_factory=TableStyle)

    @property
    def __hidden_headers(self) -> set[str]:
        hide_styles = [
            set([style.cols] if isinstance(style.cols, str) else style.cols)
            for style in self.styles
            if style.cols is not None and style.hide_headers is True
        ]
        return (
            reduce(
                lambda x, y: x | y,
                hide_styles,
            )
            if len(hide_styles) > 0
            else set()
        )

    @property
    def html_description(self, full_doc: bool = True) -> str:
        """HTML description for table filters and highlights."""
        highlights = [(h.name, h.css) for h in self.styles]
        filters = [
            (h.name, h.hide_rows)
            for h in self.styles
            if h.filter_inclusive is not None or h.hide_rows is not None
        ]

        desc = f"""
        <h1>Description for table '{self.title}'</h1>
        <h2>Filters</h2>
        <ul>
            {
                ''.join(
                    [
                        '<li style="margin-bottom: 0.5rem;">'
                        + ('hide ' if f[1] else 'show only ')
                        + 'rows where: ' + str(f[0])
                        + '</li>'
                        for f in filters if f[0]
                    ]
                )
            }
        </ul>
        <h2>Highlights</h2>
        <ul>
            {
                ''.join(
                    [
                        '<li style="margin-bottom: 0.5rem;">'
                        f'<span style="display: inline-block; {h};">{h}</span>{f}</li>'
                        for f, h in highlights if f
                    ]
                )
            }
        </ul>
        """

        return (
            f"""
        <!doctype html>
        <html>
            <head>
                <title>{self.title} - highlight-description</title>
            </head>
            <body>
                {desc}
            </body>
        </html>
        """
            if full_doc
            else desc
        )

    def _default_str_format(self, col: str) -> str:
        col_data = self.df[col]
        if is_integer_dtype(col_data):
            return "{:d}"
        elif is_float_dtype(col_data):
            if col_data.map(float.is_integer).all():
                return "{:.0f}"
            else:
                return "{:.3f}"
        else:
            return "{}"

    def _default_alignment(self, col: str | int) -> str:
        return "right" if is_numeric_dtype(self.df[col]) else "left"

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
        for col in self.df.columns:
            styled = styled.set_properties(
                subset=[col],
                **{
                    "text-align": self.default_style.alignment
                    or self._default_alignment(col)
                },
            )
            styled = styled.format(
                subset=[col],
                formatter=self.default_style.str_format
                or self._default_str_format(col),
            )

        return styled

    def _apply_styles(self, styled: Styler) -> Styler:
        for style in self.styles:
            subset = (
                style.rows if style.rows is not None else slice(None),
                style.cols
                if isinstance(style.cols, list)
                else [style.cols]
                if style.cols is not None
                else slice(None),
            )

            if style.hide_rows is not None:
                styled = styled.hide(
                    subset[0],  # type: ignore
                    axis="index",
                )
                continue

            if style.alignment is not None:
                styled = styled.set_properties(
                    subset=subset,  # type: ignore
                    **{"text-align": style.alignment},
                )

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

            if style.str_format is not None:
                styled = styled.format(
                    subset=subset,  # type: ignore
                    formatter=f"{{:{style.str_format}}}"
                    if isinstance(style.str_format, str)
                    else style.str_format,
                )

        return styled

    def __apply_widths(self, styled: Styler) -> Styler:
        widths = {
            c: (self.widths[c] if c in self.widths else 1) for c in self.df.columns
        }
        width_sum = sum(w for w in widths.values() if isinstance(w, int | float))

        for col, width in self.widths.items():
            styled = styled.set_properties(
                subset=[col],
                width=(
                    f"{width / width_sum * 100}%"
                    if isinstance(width, int | float)
                    else width
                ),
            )

        return styled

    def __apply_labels(self, styled: Styler) -> Styler:
        labels = (
            {
                c: (self.labels[c] if c in self.labels else str(c))
                for c in self.df.columns
            }
            if isinstance(self.labels, dict)
            else {c: self.labels(c) for c in self.df.columns}
        )

        hidden_headers = self.__hidden_headers

        return styled.relabel_index(
            [
                ""
                if c in hidden_headers
                else label
                if isinstance(label, str)
                else label(c)
                for c, label in labels.items()
            ],
            axis="columns",
        )

    def to_styled_df(self) -> Styler:
        """Styled pandas dataframe."""
        data = self.df.copy()

        incl_filters = [
            style.rows
            for style in self.styles
            if style.filter_inclusive is True and style.rows is not None
        ]
        if len(incl_filters) > 0:
            data = data.loc[reduce(lambda x, y: x | y, incl_filters)]

        excl_filters = [
            style.rows
            for style in self.styles
            if style.filter_exclusive is True and style.rows is not None
        ]
        if len(excl_filters) > 0:
            data = data.loc[reduce(lambda x, y: x & y, excl_filters)]

        styled = _prettify_df(data.iloc[: self.max_row_cutoff].style, self.font_size)
        styled = self._apply_default_style(styled)
        styled = self._apply_col_defaults(styled)
        styled = self.__apply_widths(styled)
        styled = self.__apply_labels(styled)
        styled = self._apply_styles(styled)

        if self.title is not None:
            styled = styled.set_caption(self.title)

        if self.hide_index:
            styled = styled.hide(axis="index")

        return styled

    def to_html_table(self, full_doc: bool = True) -> str:
        """HTML representation of the pretty table."""
        return (
            f"""
        <!doctype html>
        <html>
            <head>
                <title>{self.title}</title>
            </head>
            <body>
                {self.to_styled_df().to_html(escape=False)}
            </body>
        </html>
        """
            if full_doc
            else self.to_styled_df().to_html(escape=False)
        )


def html_to_pdf(doc: str, file: Path):
    """Render and save HTML ``doc`` as PDF document."""
    pdfkit.from_string(doc, file)


def html_to_png(doc: str, file: Path):
    """Render and save HTML ``doc`` as PNG image."""
    imgkit.from_string(doc, file, options={"zoom": "3.125", "width": "3125"})
