"""Arbitrary object conversion."""

from collections.abc import Set
from typing import Any, cast

from py_research.reflect.types import SingleTypeDef, is_subtype, typedef_to_typeset

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


def parse[T](source: Any, realm: Realm, annotation: SingleTypeDef[T]) -> T:
    """Parse an object from a source type or format."""
    if is_subtype(annotation, Convertible):
        conv = [
            t
            for t in typedef_to_typeset(annotation)
            if issubclass(t, Convertible) and t.__conv_types__().match_targets({source})
        ]

        if len(conv) > 0:
            return cast(
                T,
                conv[0].__parse_from__(source, realm, cast(SingleTypeDef, annotation)),
            )

    converter = get_converter(source, {annotation})
    if converter is None:
        raise TypeError(
            f"No parser found for {type(source)} with annotation {annotation}"
        )

    return converter.parse(source, realm, annotation)
