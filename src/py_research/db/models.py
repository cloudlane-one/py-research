from collections.abc import Callable, Iterable, Mapping
from copy import copy
from dataclasses import MISSING, Field, dataclass
from functools import cmp_to_key, reduce
from inspect import get_annotations, getmodule
from types import ModuleType, NoneType
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    dataclass_transform,
    get_origin,
    overload,
)

from py_research.data import copy_and_override
from py_research.reflect.types import GenericAlias, get_typevar_map
from py_research.types import DataclassInstance

from .props import Prop


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
        typevar_map = {}

        for name, anno in get_annotations(cls).items():
            if name not in cls.__dict__:
                prop_type = _get_prop_type(anno)
                if prop_type is not NoneType and prop_type is not Prop:
                    prop = prop_type().copy(
                        _name=name, type=anno, owner=cast(type[Model], cls)
                    )
                else:
                    prop = Attr(alias=name, _type=anno, _owner=cast(type[Model], cls))

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


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Prop,),
    eq_default=False,
)
class Model(DataclassInstance, Generic[IdxT], metaclass=ModelMeta):
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
            *(arg if arg is not Keep else MISSING for arg in args),
            **{k: v if v is not Keep else MISSING for k, v in kwargs.items()},
        )  # pyright: ignore[reportCallIssue]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record instance."""
        super().__init__()

        cls = type(self)
        props = cls._props
        prop_types = {_get_prop_type(p._type or Prop) for p in props.values()}

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
        keys: Literal["names", "fqns"] = ...,
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
        keys: Literal["instances", "names", "fqns"] = "names",
        include: set[type[Prop]] | None = None,
    ) -> dict[Prop, Any] | dict[str, Any]:
        """Convert the record to a dictionary."""
        include_types: tuple[type[Prop], ...] = (
            tuple(include) if include is not None else (Prop,)
        )

        vals = {
            (
                p if keys == "instances" else p.name if keys == "names" else p.fqn
            ): getattr(self, p.name)
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
                or a._default is not Undef
                or a._default_factory is not None
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

    default: AttrT | type[Undef] = Undef
    default_factory: Callable[[], AttrT] | None = None

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
