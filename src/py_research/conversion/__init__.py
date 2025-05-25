"""Arbitrary object conversion."""

from collections.abc import Set
from typing import Any

from py_research.reflect.types import SingleTypeDef

from .common import get_converter
from .converters import Convertible, Realm


def convert[T](obj: Any, targets: Set[T | SingleTypeDef[T]], realm: Realm) -> T:
    """Convert an object to a target type or format."""
    if isinstance(obj, Convertible):
        return obj.__convert_to__(targets, realm)

    converter = get_converter(obj, targets)
    if converter is None:
        raise TypeError(f"No converter found for {type(obj)} to {targets}")

    return converter.convert(obj, targets, realm)
