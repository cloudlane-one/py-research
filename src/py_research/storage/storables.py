"""Conversion of arbitrary types and data formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from dataclasses import InitVar, dataclass
from types import UnionType
from typing import Any, Generic, Protocol, get_args, runtime_checkable

import polars as pl
from typing_extensions import TypeVar

from py_research.caching import cached_method
from py_research.reflect.types import SingleTypeDef, get_typevar_map, is_subtype

T = TypeVar("T", contravariant=True)
T2 = TypeVar("T2")


class Realm:
    """A class representing the realm into / from which storage / loading occurs."""


@dataclass
class StorageTypes(Generic[T]):
    """A set of interfaces that can be stored to / loaded from."""

    types: InitVar[Sequence[SingleTypeDef[T]]]

    def __post_init__(self, types: Sequence[SingleTypeDef[T]]) -> None:
        """Initialize the TypeSet with a set of types."""
        self.__types = tuple(types)

    def match_targets(
        self: StorageTypes[T2], targets: Set[SingleTypeDef[T2]]
    ) -> list[SingleTypeDef[T2]]:
        """Filter the types in this set to only those that are in the provided set."""
        return [t2 for t in self.__types for t2 in targets if is_subtype(t, t2)]


U = TypeVar("U", bound="Storable")

B = TypeVar("B", contravariant=True, default=None)
B2 = TypeVar("B2")


@runtime_checkable
class Storable(Protocol[T]):  # pyright: ignore[reportInvalidTypeVarUse]
    """Protocol for classes that can be stored / loaded."""

    @classmethod
    def __storage_types__(cls) -> StorageTypes[T]:
        """Return the set of types that this class can store to/from."""
        ...

    @classmethod
    def __load__(
        cls: type[U],
        source: T,
        realm: Realm,
        annotation: SingleTypeDef[U] | None = None,
    ) -> U:
        """Parse from the source type or format."""
        ...

    def __store__(
        self: Storable[T2],
        target: T2,
        realm: Realm,
        annotation: SingleTypeDef[Storable[T2]] | None = None,
    ) -> None:
        """Convert to the target type or format."""
        ...


# @runtime_checkable
# class BatchStorable(
#     Storable[T], Protocol[T, B]
# ):  # pyright: ignore[reportInvalidTypeVarUse]
#     """Protocol for classes that can be stored / loaded in batches."""

#     @classmethod
#     def __batch_storage_types__(cls) -> StorageTypes[T]:
#         """Return the set of types that this class can batch store to/from."""
#         ...

#     @classmethod
#     def __batch_load__(
#         cls,
#         source: B,
#         realm: Realm,
#         annotation: SingleTypeDef[Self] | None = None,
#     ) -> pl.Series:
#         """Parse from the source type or format."""
#         ...

#     @classmethod
#     def __batch_store__(
#         cls: type[BatchStorable[Any, B2]],
#         batch: pl.Series,
#         target: B2,
#         realm: Realm,
#         annotation: SingleTypeDef[BatchStorable[Any, B2]] | None = None,
#     ) -> None:
#         """Convert to the target type or format."""
#         ...


V = TypeVar("V")
V2 = TypeVar("V2")


@dataclass
class StorageDriver(ABC, Generic[V, T, B]):
    """A class representing a storage driver for a specific type."""

    @cached_method
    @classmethod
    def typeargs(cls) -> dict[TypeVar, SingleTypeDef | UnionType]:
        """Return the type arguments for this converter."""
        return get_typevar_map(cls)

    @classmethod
    @cached_method
    def obj_type(cls) -> SingleTypeDef[V]:
        """Return the type of object this converter handles."""
        t = cls.typeargs()[V]
        assert not isinstance(t, UnionType), "U must not be a Union type"
        return t

    @classmethod
    @cached_method
    def storage_types(cls) -> StorageTypes[T]:
        """Return the set of types that this class can convert to/from."""
        t = cls.typeargs()[T]

        targets = get_args(t) if isinstance(t, UnionType) else (t,)

        return StorageTypes(targets)

    @abstractmethod
    @classmethod
    def load(
        cls: type[StorageDriver[V2, Any]],
        source: T,
        realm: Realm,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse from the source type or format."""
        ...

    @abstractmethod
    @classmethod
    def store(
        cls,
        instance: V,
        target: T,
        realm: Realm,
        annotation: SingleTypeDef[V] | None = None,
    ) -> None:
        """Convert to the target type or format."""
        ...

    @classmethod
    def batch_load(
        cls,
        source: B,
        realm: Realm,
        annotation: SingleTypeDef[V] | None = None,
    ) -> pl.Series:
        """Parse a batch of data from the source type or format."""
        raise NotImplementedError()

    @classmethod
    def batch_store(
        cls: type[StorageDriver[Any, Any, B2]],
        batch: pl.Series,
        target: B2,
        realm: Realm,
        annotation: SingleTypeDef[V] | None = None,
    ) -> None:
        """Convert a batch of data to the target type or format."""
        raise NotImplementedError()
