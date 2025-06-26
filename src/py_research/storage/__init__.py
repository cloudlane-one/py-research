"""Arbitrary object conversion."""

from collections.abc import Set
from types import UnionType
from typing import Any, cast

from py_research.reflect.types import (
    SingleTypeDef,
    TypeRef,
    is_subtype,
    typedef_to_typeset,
)

from .common import common_drivers
from .storables import Realm, Storable, StorageDriver, StorageTypes

custom_drivers: dict[SingleTypeDef | UnionType, type[StorageDriver]] = {}


def get_storage_driver[U, T](
    instance: type[U] | U, targets: Set[SingleTypeDef[T]]
) -> type[StorageDriver[U, T]] | None:
    """Get the converter for a given type or instance."""
    obj_type = instance if isinstance(instance, type) else type(instance)

    matching_sto = [
        c
        for t, c in (common_drivers | custom_drivers).items()
        if is_subtype(obj_type, t) and len(c.storage_types().match_targets(targets)) > 0
    ]

    if len(matching_sto) == 0:
        return None

    return matching_sto[0]


def get_storage_types(src_type: SingleTypeDef | UnionType) -> StorageTypes:
    """Get the storage types for a given type."""
    if is_subtype(src_type, Storable):
        return TypeRef(
            src_type
        ).common_type.__storage_types__()  # pyright: ignore[reportAbstractUsage]

    driver = get_storage_driver(src_type, {object})
    if driver is not None:
        return driver.storage_types()

    raise TypeError(f"No storage types found for {src_type}")


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
            if issubclass(t, Storable) and t.__storage_types__().match_targets({source})
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


def register_storage_driver(
    type_: SingleTypeDef | UnionType, driver: type[StorageDriver]
) -> None:
    """Register a custom storage driver."""
    custom_drivers[type_] = driver
