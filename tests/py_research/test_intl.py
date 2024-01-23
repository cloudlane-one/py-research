"""Test intl module."""

from datetime import datetime
from typing import Any

import pytest
from py_research.intl import DtUnit, Format, Overrides, TextMatch, iter_locales


@pytest.fixture
def locales() -> list[str]:
    """Return sample locales."""
    return ["de_DE", "en_US"]


@pytest.fixture
def base_overrides() -> Overrides:
    """Return sample translation overrides."""
    return Overrides(
        {
            None: {
                "house": "mouse",
            },
            "headers": {("bike", "car"): "Vehicle: {0}"},
        },
    )


@pytest.fixture
def translations() -> dict[str, Overrides]:
    """Return sample translation overrides."""
    return {
        "en_US": Overrides(
            {
                None: {
                    "car": "automobile",
                },
                "starting_letters": {
                    TextMatch(r"^b.*$"): "{0} (label starts with b)",
                    TextMatch(
                        r"K.*", match_current=True
                    ): "{0} (rendered text starts with K)",
                },
            },
        ),
        "de_DE": Overrides({"car": "Karren", "house": "Haus"}),
    }


@pytest.mark.parametrize(
    "locale, ctx, label, expected",
    [
        ("en_US", "headers", "car", "Vehicle: automobile"),
        ("en_US", "starting_letters", "bike", "bike (label starts with b)"),
        ("en_US", None, "bike", "bike"),
        ("en_US", None, "house", "mouse"),
        ("en_US", "starting_letters", "book", "book (label starts with b)"),
        ("en_US", "starting_letters", "Karl", "Karl (rendered text starts with K)"),
        ("de_DE", "headers", "car", "Fahrzeug: Karren"),
        ("de_DE", None, "car", "Karren"),
        ("de_DE", "headers", "bike", "Fahrzeug: Fahrrad"),
        ("de_DE", None, "bike", "Fahrrad"),
        ("de_DE", None, "house", "Haus"),
        ("de_DE", None, "box", "Kasten"),
    ],
)
def test_localize_text(
    locales: list[str],
    base_overrides: Overrides,
    translations: dict[str, Overrides],
    locale: str,
    ctx: str | None,
    label: str,
    expected: str,
):
    """Test localization of text."""
    for loc in iter_locales(locales, translations, base_overrides):
        if str(loc.locale) == locale:
            assert loc.text(label, context=ctx) == expected


@pytest.mark.parametrize(
    "locale, format, value, expected",
    [
        ("en_US", Format(), 1, "1"),
        ("en_US", Format(), 1.5, "1.5"),
        ("en_US", Format(decimal_min_digits=2), 1.5, "1.50"),
        ("en_US", Format(decimal_max_digits=3), 1.56789, "1.568"),
        ("en_US", Format(datetime_format=DtUnit.year), datetime(2024, 1, 23), "2024"),
        ("en_US", Format(datetime_format=DtUnit.month), datetime(2024, 1, 23), "01"),
        ("en_US", Format(datetime_format=DtUnit.day), datetime(2024, 1, 23), "23"),
        ("en_US", Format(datetime_format="yyyy-MM"), datetime(2024, 1, 23), "2024-01"),
        ("en_US", Format(), "Hello", "Hello"),
        ("de_DE", Format(), 1, "1"),
        ("de_DE", Format(decimal_min_digits=2), 1.5, "1,50"),
        ("de_DE", Format(decimal_max_digits=3), 1.56789, "1,568"),
        ("de_DE", Format(), "Hello", "Hallo"),
    ],
)
def test_localize_value(
    locales: list[str],
    locale: str,
    format: Format,
    value: Any,
    expected: str,
):
    """Test localization of values."""
    for loc in iter_locales(locales):
        if str(loc.locale) == locale:
            assert loc.value(value, format) == expected


def test_auto_digits():
    """Test auto digits."""
    f = Format().auto_digits([1.5, 1.56789, 1.56789, 1.56789, 1.56789])
    assert f.decimal_min_digits == 2
    assert f.decimal_max_digits == 2

    f2 = Format().auto_digits([1.5, 1.56789, 1.56789, 1.56789, 1.56789], fixed=False)
    assert f2.decimal_min_digits == 2
    assert f2.decimal_max_digits is None
