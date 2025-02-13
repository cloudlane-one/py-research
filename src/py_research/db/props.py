"""Static schemas for universal relational databases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from copy import copy
from dataclasses import MISSING, Field, dataclass
from functools import cache, reduce
from inspect import get_annotations, getmodule
from types import ModuleType, NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Hashable,
    Literal,
    Mapping,
    ParamSpec,
    Self,
    Sequence,
    TypeGuard,
    Unpack,
    cast,
    final,
    get_origin,
    overload,
)

import pandas as pd
import polars as pl
import sqlalchemy as sqla
from typing_extensions import TypeVar, TypeVarTuple

from py_research.caching import cached_prop
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericProtocol,
    SingleTypeDef,
    get_lowest_common_base,
    get_typevar_map,
    has_type,
    hint_to_typedef,
    is_subtype,
    typedef_to_typeset,
)
from py_research.types import Not

ValT = TypeVar("ValT", covariant=True, default=Any)
ValT2 = TypeVar("ValT2")

TupT = TypeVar("TupT", bound=tuple, covariant=True)

KeyT = TypeVar("KeyT", bound=Hashable)
KeyTt = TypeVarTuple("KeyTt")
KeyTt2 = TypeVarTuple("KeyTt2")


@final
class Idx(Generic[*KeyTt]):
    """Define the custom index type of a dataset."""


@final
class HashIdx:
    """Index by itself."""


IdxT = TypeVar(
    "IdxT",
    covariant=True,
    bound=Idx | HashIdx,
    default=Idx[*tuple[Any, ...]],
)
IdxT2 = TypeVar(
    "IdxT2",
    bound=Idx | HashIdx,
)


class C:
    """Singleton to allow creation of new records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


class R:
    """Singleton to allow reading of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


class U:
    """Singleton to allow updating of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


class D:
    """Singleton to allow deletion of records."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


class RU(R, U):
    """Singleton to allow reading and updating of records."""


class CR(C, R):
    """Singleton to allow creation and reading of records."""


class CRU(C, RU):
    """Singleton to allow creation, reading, and updating of records."""


class RUD(RU, D):
    """Singleton to allow reading, updating and deletion of records."""


@final
class CRUD(CRU, RUD):
    """Singleton to allow creation, reading, updating, and deleting of records."""


CrudT = TypeVar("CrudT", bound=C | R | U | D, default=Any, contravariant=True)
CrudT2 = TypeVar("CrudT2", bound=C | R | U | D)

SchemaT = TypeVar("SchemaT", contravariant=True, default=Any)


class Base(Generic[SchemaT, CrudT]):
    """Base for retrieving/storing data."""


BaseT = TypeVar("BaseT", bound=Base | None, default=None, covariant=True)

CtxT = TypeVar("CtxT", default=Any, contravariant=True)
CtxT2 = TypeVar("CtxT2")


@final
class Ctx(Generic[CtxT, IdxT, CrudT]):
    """Node in context path."""


CtxTt = TypeVarTuple("CtxTt", default=Unpack[tuple[()]])
CtxTt2 = TypeVarTuple("CtxTt2")
CtxTt3 = TypeVarTuple("CtxTt3")


type AnyCtx[CtxT] = Ctx[CtxT, Any, Any]


@final
class Ungot:
    """Singleton to denote that superclass couldn't solve a descriptor get."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


@final
class Unset:
    """Singleton to denote that superclass couldn't process a descriptor set."""

    __hash__: ClassVar[None]  # pyright: ignore[reportIncompatibleMethodOverride]


type SqlExpr = (
    sqla.FromClause | sqla.ColumnElement | sqla.schema.ColumnCollectionMixin
)

SqlT = TypeVar(
    "SqlT",
    bound=SqlExpr | None,
    covariant=True,
    default=None,
)


DfT = TypeVar("DfT", bound=pd.DataFrame | pl.DataFrame)

type Input[Val, Sql: SqlExpr | None] = Val | Iterable[Val] | Mapping[
    Hashable, Val
] | pd.DataFrame | pl.DataFrame | Sql | Prop[Val]

Params = ParamSpec("Params")


def _get_prop_type(hint: SingleTypeDef | str) -> type[Prop] | type[None]:
    """Resolve the prop typehint."""
    if has_type(hint, SingleTypeDef):
        base = get_origin(hint)
        if base is None or not issubclass(base, Prop):
            return NoneType

        return base
    elif isinstance(hint, str):
        return _map_prop_type_name(hint)
    else:
        return NoneType


@cache
def _prop_type_name_map() -> dict[str, type[Prop]]:
    return {cls.__name__: cls for cls in get_subclasses(Prop) if cls is not Prop}


def _map_prop_type_name(name: str) -> type[Prop | None]:
    """Map property type name to class."""
    name_map = _prop_type_name_map()
    matches = [name_map[n] for n in name_map if name.startswith(n + "[")]
    return matches[0] if len(matches) == 1 else NoneType


@dataclass(kw_only=True)
class Prop(ABC, Generic[ValT, IdxT, CrudT, SqlT, BaseT, CtxT, *CtxTt]):
    """Property definition for a model."""

    # Attributes:

    init_after: ClassVar[set[type[Prop]]] = set()

    _name: str | None = None
    _type: SingleTypeDef[Prop[ValT]] | None = None
    _owner: type[CtxT] | None = None

    default: ValT | Input[ValT, SqlT] | Literal[Not.defined] = Not.defined
    default_factory: Callable[[], ValT | Input[ValT, SqlT]] | None = None

    init: bool = True
    repr: bool = True
    hash: bool = True
    compare: bool = True

    # Extension methods:

    def _value_get(
        self, instance: CtxT
    ) -> ValT | Literal[Not.handled, Not.resolved, Not.defined]:
        """Get the value of this property."""
        return Not.handled

    def _value_set(
        self: Prop[Any, Any, C | U], instance: CtxT, value: Any
    ) -> None | Literal[Not.handled]:
        """Get the value of this property."""
        return Not.handled

    def _df(
        self, contents: set[Literal["keys", "values"]] = {"keys", "values"}
    ) -> pl.DataFrame | None:
        """Dataframe representation of the property."""
        return None

    def _value_from_row(self, row: Mapping[str, Any]) -> ValT | Literal[Not.handled]:
        """Extract the value from a row."""
        return Not.handled

    @cached_prop
    def _sql_expr(self) -> SqlT:
        """SQL expression of the property."""
        assert get_typevar_map(self.typeform)[SqlT] is NoneType
        return cast(SqlT, None)

    def _sql_mutation(
        self: Prop[Any, Any, C | U | D, *tuple[Any, ...]]
    ) -> Sequence[sqla.Executable] | Literal[Not.handled]:
        """SQL mutation expression of the property."""
        return Not.handled

    # Context:

    # Type:

    # SQL:

    # Dataframes:

    # Values:

    # Items:

    # Selection and filtering:

    # Comparison:

    # Transformation:

    # Mutation:

    # Summary:

    # Other representations:

    @overload
    def __get__(
        self, instance: None, owner: type[CtxT2]
    ) -> Prop[ValT, CtxT2, CrudT]: ...

    @overload
    def __get__(
        self: Prop[Any, Any, Any, Idx[()]], instance: CtxT2, owner: type[CtxT2]
    ) -> ValT: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    @abstractmethod
    def __get__(self, instance: Any, owner: type | None) -> Any:  # noqa: D105
        if owner is None or not issubclass(owner, Model) or instance is None:
            return self

        return Ungot()

    @abstractmethod
    def __set__(  # noqa: D105
        self: Prop[Any, Any, U], instance: Any, value: Any
    ) -> Any:
        return Unset()

    def __set_name__(self, owner: type[CtxT], name: str) -> None:  # noqa: D105
        if not TYPE_CHECKING and not issubclass(owner, Model):
            return

        if self._name is None:
            self._name = name

        if self._owner is None:
            self._owner = owner

        if self._type is None:
            self._type = get_annotations(owner)[name]

    def __copy__(self) -> Self:  # noqa: D105
        obj = type(self)()  # type: ignore
        obj.__dict__.update(self.__dict__)
        return obj

    def copy(self, **overrides: Any) -> Self:
        """Return a copy of this object with overridden attributes."""
        obj = copy(self)
        obj.__dict__.update(overrides)
        return obj

    @property
    def name(self) -> str:
        """Name of the property."""
        assert self._name is not None
        return self._name

    @cached_prop
    def owner_module(self) -> ModuleType | None:
        """Module of the owner model type."""
        if self._owner is None:
            return None

        return (
            self._owner._src_mod
            if issubclass(self._owner, Model)
            else getmodule(self._owner)
        )

    @cached_prop
    def owner_typeargs(self) -> dict[TypeVar, SingleTypeDef | UnionType]:
        """Type arguments of the owner model type."""
        return {} if self._owner is None else get_typevar_map(self._owner)

    @cached_prop
    def typeform(self) -> SingleTypeDef[Self]:
        """Self type definition."""
        hint = hint_to_typedef(
            self._type or type(self),
            typevar_map=self.owner_typeargs,
            ctx_module=self.owner_module,
        )

        if not is_subtype(hint, Prop):
            return type(self)[hint]  # pyright: ignore[reportInvalidTypeArguments]

        return cast(SingleTypeDef[Self], hint)

    @cached_prop
    def value_typeform(self) -> SingleTypeDef[ValT] | UnionType:
        """Target typeform of this prop."""
        return get_typevar_map(self.typeform)[ValT]

    @cached_prop
    def value_type_set(self) -> set[type[ValT]]:
        """Target types of this prop (>1 in case of union typeform)."""
        return typedef_to_typeset(
            self.value_typeform, self.owner_typeargs, self.owner_module
        )

    @cached_prop
    def common_value_type(self) -> type:
        """Common base type of the target types."""
        return get_lowest_common_base(
            typedef_to_typeset(
                self.value_typeform,
                self.owner_typeargs,
                self.owner_module,
                remove_null=True,
            )
        )

    @cached_prop
    def typeargs(self) -> dict[TypeVar, SingleTypeDef | UnionType]:
        """Type arguments of this prop."""
        return get_typevar_map(self.typeform)

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this dataset based on relational path."""
        if self._owner is None:
            return self.name

        fqn = f"{self._owner._fqn}.{self.name}"

        return fqn

    @staticmethod
    def has_type[T: Prop](instance: Prop, typedef: type[T]) -> TypeGuard[T]:
        """Check if the dataset has the specified type."""
        orig = get_origin(typedef)
        if orig is None or not issubclass(_get_prop_type(instance._type or Prop), orig):
            return False

        own_typevars = get_typevar_map(instance._type or type(instance))
        target_typevars = get_typevar_map(typedef)

        for tv, tv_type in target_typevars.items():
            if tv not in own_typevars:
                return False
            if not is_subtype(own_typevars[tv], tv_type):
                return False

        return True

    def __truediv__(
        self: Prop[CtxT2, Any, CrudT2, Idx[*KeyTt]],
        other: Prop[ValT2, CtxT2, CrudT2, Idx[*KeyTt2]],
    ) -> Chain[ValT2, CtxT, CrudT2, Idx[*KeyTt, *KeyTt2]]:
        """Chain two matching properties together."""
        ...

    def __matmul__(
        self: Prop[Any, CtxT2, CrudT2, Any],
        other: Prop[ValT2, CtxT2, CrudT2, IdxT2],
    ) -> Alignment[tuple[ValT, ValT2], CtxT2, CrudT2, IdxT | IdxT2]:
        """Align two properties."""
        ...


@dataclass
class Chain(Prop[ValT, CtxT, CrudT, IdxT]):
    """Alignment of multiple props."""

    props: tuple[
        Prop[Model, CtxT, CrudT, Any],
        *tuple[Model, Any, CrudT, Any],
        Prop[ValT, Any, CrudT, Any],
    ]


@dataclass
class Alignment(Prop[TupT, CtxT, CrudT, IdxT]):
    """Alignment of multiple props."""

    props: tuple[Prop[Any, CtxT, CrudT, IdxT], ...]


class ModelMeta(type):
    """Metaclass for record types."""

    _root_class: bool = False
    _template: bool = False
    _derivate: bool = False

    _src_mod: ModuleType | None = None
    _model_superclasses: list[type[Model]]
    __class_props: dict[str, Prop] | None

    def __init__(cls, name, bases, dct):
        """Initialize a new record type."""
        super().__init__(name, bases, dct)

        if "_src_mod" not in dct:
            cls._src_mod = getmodule(cls if not cls._derivate else bases[0])

        # Copy all supplied data instances to get independent objects.
        for prop_name, prop in cls.__dict__.items():
            if isinstance(prop, Prop):
                setattr(cls, prop_name, copy(prop))

        cls.__class_props = None
        cls._get_class_props()

    @property
    def _super_types(cls) -> Iterable[type | GenericProtocol]:
        return (
            cls.__dict__["__orig_bases__"]
            if "__orig_bases__" in cls.__dict__
            else cls.__bases__
        )

    def _get_class_props(cls) -> dict[str, Prop]:
        if cls.__class_props is not None:
            return cls.__class_props

        # Construct list of Record superclasses and apply template superclasses.
        cls._model_superclasses = []

        props = {}
        typevar_map = {}

        for name, anno in get_annotations(cls).items():
            if name not in cls.__dict__:
                prop_type = _get_prop_type(anno)
                if prop_type is not NoneType and prop_type is not Prop:
                    prop = prop_type().copy(
                        _name=name, type=anno, owner=cast(type[Model], cls)
                    )
                else:
                    prop = Attr(_name=name, _type=anno, _owner=cast(type[Model], cls))

                props[name] = prop
                setattr(cls, name, prop)

        for c in cls._super_types:
            # Get proper origin class of generic supertype.
            orig = get_origin(c) if not isinstance(c, type) else c

            if orig is None:
                continue

            # Handle typevar substitutions.
            typevar_map = get_typevar_map(c, subs=typevar_map)

            # Skip root Record class and non-Record classes.
            if not isinstance(orig, ModelMeta) or orig._root_class:
                continue

            # Apply template classes.
            if orig._template or cls._derivate:
                # Pre-collect all template props
                # to prepend them in order.
                orig_props = {}
                for prop_name, super_prop in orig._get_class_props().items():
                    if prop_name not in props:
                        prop = super_prop.copy()
                        setattr(cls, prop_name, prop)
                        orig_props[prop_name] = prop

                props = {**orig_props, **props}  # Prepend template props.
            else:
                assert orig is c  # Must be concrete class, not a generic
                cls._model_superclasses.append(orig)

        cls.__class_props = props
        return props

    @property
    def _props(cls) -> dict[str, Prop]:
        """The statically defined properties of this record type."""
        return reduce(
            lambda x, y: {**x, **y},
            (c._props for c in cls._model_superclasses),
            cls._get_class_props(),
        )

    @property
    def _fqn(cls) -> str:
        mod_name: str = (
            getattr(cls._src_mod, "__name__")
            if cls._src_mod is not None
            else cls.__module__
        )

        return mod_name + "." + cls.__name__

    @property
    def __dataclass_fields__(cls) -> dict[str, Field]:  # noqa: D105
        return {
            p.name: Field(
                p._default if p._default is not Not.defined else MISSING,
                (
                    p._default_factory if p._default_factory is not None else MISSING
                ),  # pyright: ignore[reportArgumentType]
                p.init,
                metadata={},
                repr=p.repr,
                hash=p.hash,
                compare=p.compare,
                kw_only=True,
            )
            for p in cls._props.values()
        }
