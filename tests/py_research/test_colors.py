"""Test the colors module."""

import pytest
from py_research.colors import ColorTheme, get_theme, to_bg_color


def test_get_theme():
    """Test the get_theme function."""
    theme = get_theme()
    assert isinstance(theme, ColorTheme)


def test_to_bg_color():
    """Test the to_bg_color function."""
    color = "#FF0000"  # red
    lightness = 0.8
    bg_color = to_bg_color(color, lightness)
    assert isinstance(bg_color, str)
    assert bg_color.startswith("#")

    # Test with invalid color
    with pytest.raises(ValueError):
        to_bg_color("invalid_color", lightness)
