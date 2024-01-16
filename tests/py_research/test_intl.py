"""Test intl module."""

import pytest

from py_research.intl import Locale, Localization, Overrides


@pytest.fixture
def local_overrides():
    """Return sample local overrides."""
    return Overrides(
        labels=[
            {
                "car": "Auto",
                "house": "Maus",
            }
        ]
    )


@pytest.fixture
def localization(local_overrides: Overrides):
    """Return sample localization."""
    loc = Localization(local_overrides=local_overrides, local_locale=Locale("de_DE"))

    assert loc.label("car") == "Auto"
    assert loc.label("house") == "Maus"
    assert loc.label("dog") == "Hund"
