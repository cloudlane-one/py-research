"""Arbitrary object conversion."""

from typing import Any, cast

from py_research.reflect.types import SingleTypeDef, is_subtype, typedef_to_typeset

from .common import get_storage_driver
from .storables import Realm, Storable


def store(obj: Any, target: Any, realm: Realm) -> None:
    """Store an object to a target."""
    if isinstance(obj, Storable):
        return obj.__store__(target, realm)

    driver = get_storage_driver(obj, {type(target)})
    if driver is None:
        raise TypeError(f"No driver found for storing {type(obj)} to {target}")

    return driver.store(obj, target, realm)


def load[T](source: Any, realm: Realm, annotation: SingleTypeDef[T]) -> T:
    """Load an object from a source."""
    if is_subtype(annotation, Storable):
        conv = [
            t
            for t in typedef_to_typeset(annotation)
            if issubclass(t, Storable) and t.__conv_types__().match_targets({source})
        ]

        if len(conv) > 0:
            return cast(
                T,
                conv[0].__load__(source, realm, cast(SingleTypeDef, annotation)),
            )

    converter = get_storage_driver(source, {annotation})
    if converter is None:
        raise TypeError(
            f"No parser found for {type(source)} with annotation {annotation}"
        )

    return converter.load(source, realm, annotation)
