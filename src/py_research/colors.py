"""Simple & semi-automatic color-system for data visualization purposes."""

import re
from colorsys import hls_to_rgb, rgb_to_hls
from contextvars import ContextVar
from dataclasses import dataclass, field
from os import environ
from pathlib import Path
from typing import Any

from webcolors import IntegerRGB, hex_to_rgb, name_to_rgb, rgb_to_hex
from yaml import CLoader, load


def default_highlights():
    """Return default highlight colors."""
    return {
        0: "#663399",
        1: "#1E90FF",
        2: "#3CB371",
    }


def default_scales():
    """Return default color scale."""
    return {
        0: [
            (0, "#ebf6fa"),
            (0.01, "#d6edf5"),
            (0.03, "#99d3e6"),
            (0.10, "#5cb8d6"),
            (0.30, "#14d5cc"),
            (1, "#35d27e"),
        ],
        1: [
            (0, "#005b7f"),
            (0.15, "#39a9cd"),
            (0.30, "#14d5cc"),
            (1, "#35d27e"),
        ],
    }


@dataclass
class ColorTheme:
    """Define custom theme colors."""

    highlights: dict[Any, str] = field(default_factory=default_highlights)
    scales: dict[Any, list[tuple[float, str]]] = field(default_factory=default_scales)
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):  # noqa: D105
        self.__token = None

    def activate(self):
        """Set this the as current theme."""
        self.__token = active_theme.set(self)

    def __enter__(self):  # noqa: D105
        self.activate()
        return self

    def __exit__(self, *_):  # noqa: D105
        if self.__token is not None:
            active_theme.reset(self.__token)


color_file_path = Path(environ.get("COLOR_FILE_PATH") or (Path.cwd() / "colors.yaml"))


def _get_colors_from_file() -> ColorTheme | None:
    if color_file_path.is_file():
        with open(color_file_path, encoding="utf-8") as f:
            return ColorTheme(**load(f, Loader=CLoader))


active_theme: ContextVar[ColorTheme] = ContextVar(
    "active_color_theme", default=(_get_colors_from_file() or ColorTheme())
)


def _parse_css_color(color: str) -> IntegerRGB:
    if (rgb_match := re.match(r"^rgb\(([0-9]+,[0-9]+,[0-9]+)\)$", color)) is not None:
        return IntegerRGB(*(int(val) for val in rgb_match.groups()[0].split(",")))
    elif re.match(r"^#[0-9abcdefABCDEF]{3,6}$", color) is not None:
        return hex_to_rgb(color)
    else:
        try:
            return name_to_rgb(color)
        except ValueError as exc:
            raise ValueError(f"CSS color definition not recognized: {color}") from exc


def _integer_rgb_to_float(color: IntegerRGB) -> tuple[float, float, float]:
    return (color.red / 255, color.green / 255, color.blue / 255)


def _float_to_integer_rgb(color: tuple[float, float, float]) -> IntegerRGB:
    return IntegerRGB(*(int(c * 255) for c in color))


def _adjust_lightness(color: IntegerRGB, lightness: float) -> IntegerRGB:
    hls = rgb_to_hls(*_integer_rgb_to_float(color))
    changed = (hls[0], lightness, hls[2])
    return _float_to_integer_rgb(hls_to_rgb(*changed))


def get_theme() -> ColorTheme:
    """Return active color theme (may be derived from current execution context)."""
    return active_theme.get()


def to_bg_color(color: str, lightness: float = 0.8) -> str:
    """Turn a highlight color into."""
    return rgb_to_hex(_adjust_lightness(_parse_css_color(color), lightness))
