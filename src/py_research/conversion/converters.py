"""Conversion of arbitrary types and data formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from dataclasses import InitVar, dataclass
from types import UnionType
from typing import (
    Any,
    Generic,
    NewType,
    Protocol,
    Self,
    TypeAliasType,
    get_args,
    runtime_checkable,
)

from typing_extensions import TypeVar

from py_research.caching import cached_method
from py_research.reflect.types import (
    GenericAlias,
    SingleTypeDef,
    get_typevar_map,
    has_type,
    is_subtype,
)

T = TypeVar("T", contravariant=True)
T2 = TypeVar("T2")


class Realm:
    """A class representing the realm into / from which conversion / parsing occurs."""


@dataclass
class ConversionTypes(Generic[T]):
    """A set of types that can be converted to/from."""

    types: InitVar[Sequence[SingleTypeDef[T]]]

    def __post_init__(self, types: Sequence[SingleTypeDef[T]]) -> None:
        """Initialize the TypeSet with a set of types."""
        self.__types = tuple(types)

    def match_targets(
        self: ConversionTypes[T2], targets: Set[T2 | SingleTypeDef[T2]]
    ) -> list[T2 | SingleTypeDef[T2]]:
        """Filter the types in this set to only those that are in the provided set."""
        return [
            t2
            for t in self.__types
            for t2 in targets
            if isinstance(t2, GenericAlias | TypeAliasType | type | NewType)
            and is_subtype(t, t2)
            or has_type(t2, t)
        ]


@runtime_checkable
class Convertible(Protocol[T]):  # pyright: ignore[reportInvalidTypeVarUse]
    """Protocol for classes that can be converted to arbitrary types/formats."""

    @classmethod
    def __conv_types__(cls) -> ConversionTypes[T]:
        """Return the set of types that this class can convert to/from."""
        ...

    @classmethod
    def __parse_from__(
        cls,
        source: T,
        realm: Realm,
        annotation: SingleTypeDef[Convertible[T]] | None = None,
    ) -> Self:
        """Parse from the source type or format."""
        ...

    def __convert_to__(
        self: Convertible[T2],
        targets: Set[T2 | SingleTypeDef[T2]],
        realm: Realm,
        annotation: SingleTypeDef[Convertible[T2]] | None = None,
    ) -> T2:
        """Convert to the target type or format."""
        ...


U = TypeVar("U")


# class ParserFunc[U, T](Protocol):
#     """Parse from the source type or format."""

#     def __call__(  # noqa: D102
#         self,
#         source: T,
#         realm: Realm,
#         annotation: SingleTypeDef[U] | None = None,
#     ) -> U: ...


# class ConverterFunc[U, T2](Protocol):
#     """Convert to the target type or format."""

#     def __call__(  # noqa: D102
#         self,
#         instance: U,
#         targets: Set[T2 | SingleTypeDef[T2]],
#         realm: Realm,
#         annotation: SingleTypeDef[U] | None = None,
#     ) -> T2: ...


@dataclass
class Converter(ABC, Generic[U, T]):
    """A class representing a converter for a specific type."""

    @cached_method
    @classmethod
    def typeargs(cls) -> dict[TypeVar, SingleTypeDef | UnionType]:
        """Return the type arguments for this converter."""
        return get_typevar_map(cls)

    @classmethod
    @cached_method
    def obj_type(cls) -> SingleTypeDef[U]:
        """Return the type of object this converter handles."""
        t = cls.typeargs()[U]
        assert not isinstance(t, UnionType), "U must not be a Union type"
        return t

    @classmethod
    @cached_method
    def conv_types(cls) -> ConversionTypes[T]:
        """Return the set of types that this class can convert to/from."""
        t = cls.typeargs()[T]

        targets = get_args(t) if isinstance(t, UnionType) else (t,)

        return ConversionTypes(targets)

    @abstractmethod
    @classmethod
    def parse(
        cls,
        source: T,
        realm: Realm,
        annotation: SingleTypeDef[U] | None = None,
    ) -> U:
        """Parse from the source type or format."""
        ...

    @abstractmethod
    @classmethod
    def convert(
        cls: type[Converter[Any, T2]],
        instance: U,
        targets: Set[T2 | SingleTypeDef[T2]],
        realm: Realm,
        annotation: SingleTypeDef[U] | None = None,
    ) -> T2:
        """Convert to the target type or format."""
        ...
