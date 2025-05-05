"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import copy
from dataclasses import MISSING, Field, dataclass
from functools import cache, cmp_to_key, reduce
from inspect import get_annotations, getmodule
from types import ModuleType, NoneType, UnionType
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    cast,
    dataclass_transform,
    get_origin,
    overload,
)

import polars as pl
from typing_extensions import TypeVar

from py_research.data import Params, copy_and_override
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericAlias,
    SingleTypeDef,
    TypeRef,
    get_base_type,
    get_typevar_map,
    is_subtype,
)
from py_research.types import DataclassInstance, Not

from .data import Ctx, Data, Dx, EngineT, FullIdxT, Idx, RwxT, ShapeT, U, ValT, ValT2

OwnT = TypeVar("OwnT", bound="Model", contravariant=True, default=Any)
OwnT2 = TypeVar("OwnT2", bound="Model")

IdxT = TypeVar("IdxT", bound=Idx, default=Any)


@dataclass
class Prop(Generic[ValT, IdxT, RwxT, EngineT, ShapeT, OwnT]):
    """Property definition for a model."""

    init_after: ClassVar[set[type[Prop]]] = set()
    value_hook: ClassVar[SingleTypeDef | UnionType | None] = None
    owner_hook: ClassVar[SingleTypeDef | UnionType | None] = None

    @cache
    @staticmethod
    def _get_default_prop(
        val_type: SingleTypeDef | UnionType, owner: type
    ) -> Prop | None:
        """Get the default subclass for this property."""
        subclasses = reversed(get_subclasses(Prop))

        matching = [
            s
            for s in subclasses
            if s.value_hook is not None
            and is_subtype(val_type, s.value_hook)
            and s.owner_hook is not None
            and is_subtype(owner, s.owner_hook)
        ]

        if len(matching) == 0:
            return None

        return matching[0]()

    data: Data[ValT, IdxT, RwxT, Dx[EngineT, ShapeT], Ctx[OwnT]] | None = None
    getter: Callable[[OwnT], ValT | Literal[Not.resolved]] | None = None
    setter: Callable[[OwnT, ValT], None] | None = None

    alias: str | None = None
    typeref: TypeRef | None = None

    default: ValT | Literal[Not.defined] = Not.defined
    default_factory: Callable[[], ValT] | None = None
    init: bool = True
    repr: bool = True
    hash: bool = True
    compare: bool = True

    # Name and Ownership:

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        if self.alias is None:
            self.alias = name

        if self.typeref is None:
            typeargs = get_typevar_map(owner) | {OwnT: owner}

            self.typeref = TypeRef(
                get_annotations(owner)[name],
                var_map=typeargs,
                ctx_module=owner._src_mod or getmodule(owner),
            )

            default_prop = self._get_default_prop(typeargs[ValT], owner)

            if default_prop is not None:
                if self.data is not None:
                    self.data = copy(default_prop.data)  # type: ignore

                if self.getter is not None:
                    self.getter = default_prop.getter

                if self.setter is not None:
                    self.setter = default_prop.setter

    def _name(self) -> str:
        """Name of the property."""
        assert self.alias is not None
        return self.alias

    @property
    def name(self) -> str:
        """Name of the property."""
        return self._name()

    # Descriptor read/write:

    @overload
    def __get__(
        self, instance: None, owner: type[OwnT2]
    ) -> Prop[ValT, IdxT, RwxT, EngineT, ShapeT, OwnT2]: ...

    @overload
    def __get__(self, instance: OwnT2, owner: type[OwnT2]) -> ValT: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(self, instance: Any, owner: type | None) -> Any:  # noqa: D105
        if owner is None or not issubclass(owner, Model):
            return self

        if instance is None:
            assert self.data is not None
            return self.data

        assert self.getter is not None
        res = self.getter(instance)
        if res is not Not.resolved:
            return res

        if self.default is not Not.defined:
            return self.default

        assert self.default_factory is not None
        return self.default_factory()

    @overload
    def __set__(  # noqa: D105
        self: Prop[ValT2, Any, U], instance: OwnT, value: ValT2
    ) -> None: ...

    @overload
    def __set__(  # noqa: D105
        self: Prop[Any, Any, U], instance: Any, value: Any
    ) -> None: ...

    def __set__(  # noqa: D105
        self: Prop[Any, Any, U], instance: Any, value: Any
    ) -> None:
        if isinstance(instance, Model):
            assert self.setter is not None
            self.setter(instance, value)


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
    def _super_types(cls) -> Iterable[type | GenericAlias]:
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
        typevar_map = get_typevar_map(cls)

        for name, anno in get_annotations(cls).items():
            if name not in cls.__dict__:
                prop_type = get_base_type(anno, bound=Prop)
                prop = prop_type()

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
                        prop = copy(super_prop)
                        setattr(cls, prop_name, prop)
                        orig_props[prop_name] = prop

                props = {**orig_props, **props}  # Prepend template props.
            else:
                assert orig is c  # Must be concrete class, not a generic
                cls._model_superclasses.append(cast(type[Model], orig))

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
    def __dataclass_fields__(cls) -> dict[str, Field]:  # noqa: D105
        return {
            p._name(): Field(
                p.default if p.default is not Not.defined else MISSING,
                (
                    p.default_factory if p.default_factory is not None else MISSING
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


ModT = TypeVar("ModT", bound="Model", covariant=True)


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Prop,),
    eq_default=False,
)
class Model(DataclassInstance, Generic[FullIdxT], metaclass=ModelMeta):
    """Schema for a record in a database."""

    _template: ClassVar[bool]
    _derivate: ClassVar[bool] = False
    _root_class: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)
        cls._root_class = False
        if "_template" not in cls.__dict__:
            cls._template = False

    @classmethod  # pyright: ignore[reportArgumentType]
    def _from_existing(
        cls: Callable[Params, ModT],
        rec: ModT,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ModT:
        return copy_and_override(
            cls,
            rec,
            *(arg if arg is not Not.defined else MISSING for arg in args),
            **{k: v if v is not Not.defined else MISSING for k, v in kwargs.items()},
        )  # pyright: ignore[reportCallIssue]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record instance."""
        super().__init__()

        cls = type(self)
        props = cls._props
        prop_types = {
            get_base_type(
                p.typeref.hint if p.typeref is not None else type(p), bound=Prop
            )
            for p in props.values()
        }

        init_sequence = list(p for p in prop_types if p is not NoneType)
        init_sequence = sorted(
            init_sequence,
            key=cmp_to_key(
                lambda x, y: (
                    -1
                    if any(issubclass(x, p) for p in y.init_after)
                    else 1 if any(issubclass(y, p) for p in x.init_after) else 0
                )
            ),
        )

        for prop_type in init_sequence:
            props = {k: v for k, v in props.items() if isinstance(v, prop_type)}
            for prop_name in props:
                if prop_name in kwargs:
                    setattr(self, prop_name, kwargs[prop_name])

        self.__post_init__()

    def __post_init__(self) -> None:  # noqa: D105
        pass

    @overload
    def _to_dict(
        self,
        keys: Literal["names"] = ...,
        include: set[type[Prop]] | None = ...,
    ) -> dict[str, Any]: ...

    @overload
    def _to_dict(
        self,
        keys: Literal["instances"],
        include: set[type[Prop]] | None = ...,
    ) -> dict[Prop, Any]: ...

    def _to_dict(
        self,
        keys: Literal["instances", "names"] = "names",
        include: set[type[Prop]] | None = None,
    ) -> dict[Prop, Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Prop], ...] = (
            tuple(include) if include is not None else (Prop,)
        )

        vals = {
            (p if keys == "instances" else p.name): getattr(self, p.name)
            for p in type(self)._props.values()
            if isinstance(p, include_types) and p.name is not None
        }

        return cast(dict, vals)

    @classmethod
    def _is_complete_dict(
        cls,
        data: Mapping[Prop, Any] | Mapping[str, Any],
    ) -> bool:
        """Check if dict data contains all required info for record type."""
        in_data = {(p if isinstance(p, str) else p.name): v for p, v in data.items()}
        return all(
            (
                a.name in in_data
                or a.default is not Not.defined
                or a.default_factory is not None
            )
            for a in cls._props.values()
            if a.init is not False and a.name is not None
        )

    def __repr__(self) -> str:
        """Return a string representation of the record."""
        return f"{type(self).__name__}({repr(self._to_dict())})"


AttrT = TypeVar("AttrT")


@dataclass(kw_only=True, eq=False)
class Attr(Prop[AttrT, CtxT, CrudT], Generic[AttrT, CrudT, CtxT]):
    """Single-value attribute or column."""

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> AttrT: ...

    @overload
    def __get__(
        self, instance: None, owner: type[ModT]
    ) -> Attr[AttrT, CrudT, ModT]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride] (pyright bug?)
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this attribute."""
        super_get = Prop.__get__(self, instance, owner)
        if super_get is not Ungot():
            return super_get

        assert instance is not None

        if self.name not in instance.__dict__:
            if self.default is not Undef:
                instance.__dict__[self.name] = self.default
            else:
                assert self.default_factory is not None
                instance.__dict__[self.name] = self.default_factory()

        return instance.__dict__[self.name]

    def __set__(self: Attr[Any, U, Any], instance: Model, value: AttrT) -> None:
        """Set the value of this attribute."""
        super_set = Prop.__set__(self, instance, value)
        if super_set is not Unset():
            return

        instance.__dict__[self.name] = value

    @property
    def _default(self) -> AttrT | type[Undef]:
        """Default value of the property."""
        return self._default

    @property
    def _default_factory(self) -> Callable[[], AttrT] | None:
        """Default value factory of the property."""
        return self.default_factory


def get_pl_schema(
    col_map: Mapping[str, Column],
) -> dict[str, pl.DataType | type | None]:
    """Return the schema of the dataset."""
    exact_matches = {
        name: (pl_type_map.get(col.value_typeform), col)
        for name, col in col_map.items()
    }
    matches = {
        name: (
            (match, col.value_typeform)
            if match is not None
            else (pl_type_map.get(col.common_value_type), col.value_typeform)
        )
        for name, (match, col) in exact_matches.items()
    }

    return {
        name: (
            match if isinstance(match, pl.DataType | type | None) else match(match_type)
        )
        for name, (match, match_type) in matches.items()
    }


def records_to_df(
    records: Iterable[dict[str, Any] | Record],
) -> pl.DataFrame:
    """Convert values to a DataFrame."""
    model_types: set[type[Record]] = {
        type(val) for val in records if isinstance(val, Record)
    }
    df_data = [
        (
            val._to_dict(keys="names" if len(model_types) == 1 else "fqns")
            if isinstance(val, Record)
            else val
        )
        for val in records
    ]

    return pl.DataFrame(
        df_data,
        schema=get_pl_schema(
            {
                col.name if len(model_types) == 1 else col.fqn: col
                for mod in model_types
                for col in mod._cols()
            }
        ),
    )
