"""Test intl module."""

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from py_research.intl import Args, DtUnit, Format, Overrides, iter_locales


@pytest.fixture
def locales() -> list[str]:
    """Return sample locales."""
    return ["de_DE", "en_US"]


@pytest.fixture
def overrides() -> dict[str, Overrides]:
    """Return sample translation overrides."""
    return {
        "en_US": Overrides(
            vocabulary={
                "car": "automobile",
                "house": "mouse",
            },
            templates={
                "vehicles": {("bike", "car"): "Vehicle: {0}"},
                "buildings": {Args({"house", "bridge"}): "Building: {0}"},
            },
        ),
        "de_DE": Overrides(vocabulary={"car": "Karren", "house": "Haus"}),
    }


@pytest.mark.parametrize(
    "locale, ctx, label, expected",
    [
        ("en_US", "vehicles", "car", "Vehicle: automobile"),
        ("en_US", None, "bike", "bike"),
        ("en_US", None, "house", "mouse"),
        ("de_DE", "vehicles", "car", "Fahrzeug: Karren"),
        ("de_DE", None, "car", "Karren"),
        ("de_DE", "vehicles", "bike", "Fahrzeug: Fahrrad"),
        ("de_DE", None, "bike", "Fahrrad"),
        ("de_DE", None, "house", "Haus"),
        ("de_DE", None, "box", "Kasten"),
        ("de_DE", "buildings", "house", "Gebäude: Haus"),
    ],
)
def test_localize_label(
    locales: list[str],
    overrides: dict[str, Overrides],
    locale: str,
    ctx: str | None,
    label: str,
    expected: str,
):
    """Test localization of text."""
    for loc in iter_locales(locales, overrides):
        if str(loc.locale) == locale:
            assert loc.label(label, context=ctx) == expected


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
        ("de_DE", Format(), None, ""),
        ("de_DE", Format(na_representation="N/A"), None, "N/A"),
        ("de_DE", Format(), pd.NA, ""),
        ("de_DE", Format(na_representation="N/A"), pd.NA, "N/A"),
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


def test_iter_locz_default(
    locales: list[str],
    overrides: dict[str, Overrides],
):
    """Test default locale iteration."""
    locz = [str(loc.locale) for loc in iter_locales(overrides=overrides)]
    assert set(locz) == set(locales)
