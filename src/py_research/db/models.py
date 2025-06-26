"""Basis for modeling complex data."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import copy
from dataclasses import MISSING, Field, dataclass, field
from functools import cache, reduce
from inspect import get_annotations, getmodule
from types import ModuleType, UnionType
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    cast,
    dataclass_transform,
    get_origin,
    overload,
)

import polars as pl
from pydantic import (
    BaseModel,
    GetJsonSchemaHandler,
    PydanticSchemaGenerationError,
    TypeAdapter,
    create_model,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from typing_extensions import TypeVar

from py_research.caching import cached_method, cached_prop
from py_research.data import Params, copy_and_override
from py_research.files import Dir
from py_research.hashing import gen_int_hash
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import (
    GenericAlias,
    SingleTypeDef,
    TypeRef,
    get_typevar_map,
    is_subtype,
)
from py_research.storage import get_storage_types
from py_research.storage.common import JSONDict, JSONFile
from py_research.storage.storables import Realm, Storable, StorageTypes
from py_research.types import Not

from .data import (
    PL,
    SQL,
    AutoIndexable,
    C,
    Ctx,
    CtxTt2,
    Data,
    DxT,
    DxT2,
    Expand,
    ExT,
    ExT2,
    Frame,
    FullIdx,
    Idx,
    Interface,
    KeyTt2,
    R,
    Root,
    RwxT,
    RwxT2,
    SxT2,
    Tab,
    U,
    ValT,
    ValT2,
)
from .utils import get_pl_schema

OwnT = TypeVar("OwnT", bound="Model", contravariant=True, default=Any)
OwnT2 = TypeVar("OwnT2", bound="Model")

IdxT = TypeVar("IdxT", bound=FullIdx, default=Any, covariant=True)
IdxT2 = TypeVar("IdxT2", bound=FullIdx)

AutoT = TypeVar("AutoT", bound=AutoIndexable)
CruT = TypeVar("CruT", bound=C | R | U, default=Any)
CruT2 = TypeVar("CruT2", bound=C | R | U)


@dataclass(frozen=True)
class Init(Generic[ValT]):
    """Wrapper object around init values."""

    value: ValT


@dataclass(kw_only=True)
class Prop(
    Data[ValT, Expand[IdxT], DxT, ExT, RwxT, Interface[OwnT], *tuple[()]],
    Generic[ValT, IdxT, CruT, OwnT, DxT, ExT, RwxT],
):
    """Property definition for a model."""

    @classmethod
    def _type_matcher(
        cls,
        val_type: SingleTypeDef | UnionType,
        index_type: SingleTypeDef | UnionType,
        owner_type: type[Model],
    ) -> bool:
        return False

    @cache
    @staticmethod
    def _get_default_class(typedef: SingleTypeDef, owner: type) -> type[Prop] | None:
        """Get the default subclass for this property."""
        typevars = get_typevar_map(typedef)
        subclasses = reversed(get_subclasses(Prop))

        matching = [
            s
            for s in subclasses
            if s._type_matcher(
                val_type=typevars[ValT].typeform,
                index_type=typevars[IdxT].typeform,
                owner_type=owner,
            )
        ]

        if len(matching) == 0:
            return None

        return matching[0]

    context: Interface[OwnT] | Data[Any, Any, Any, Any, Any, Interface[OwnT]] = field(
        default_factory=Interface
    )

    alias: str | None = None
    default: ValT | Literal[Not.defined] = Not.defined
    default_factory: Callable[[], ValT] | None = None
    init: bool = True
    repr: bool = True
    hash: bool = True
    compare: bool = True

    _owner_map: dict[type[OwnT], Prop[ValT, IdxT, CruT, OwnT, DxT, ExT, RwxT]] = field(
        default_factory=dict
    )

    def _getter(self, instance: OwnT) -> ValT | Literal[Not.resolved]:
        """Get the value of this property."""
        return Not.resolved

    def _setter(self: Prop[ValT2], instance: OwnT, value: ValT2) -> None:
        """Set the value of this property."""
        raise NotImplementedError()

    @property
    def init_level(self) -> int:
        """Get the initialization level of this property."""
        return 0

    def _index(self) -> Expand[IdxT]:
        """Get the index of this property."""
        raise NotImplementedError()

    def _frame(self: Data[Any, Any, SxT2]) -> Frame[PL, SxT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()

    @cached_prop
    def common_prop_type(self) -> type[ValT]:
        """Get the common type of this property."""
        return cast(type[ValT], self.typeref.common_type)

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        if self.alias is None:
            self.alias = name

        if self.owner is None:
            Prop._set_owner(self, owner)

            self.typeref = copy_and_override(
                TypeRef,
                self.typeref or TypeRef(get_annotations(owner).get(name, object)),
            )

    @property
    def name(self) -> str:
        """Name of the property."""
        assert self.alias is not None
        return self.alias

    def _id(self) -> str:
        """Name of the property."""
        return self.name

    @property
    def owner(self: Prop[Any, Any, Any, OwnT2]) -> type[OwnT2] | None:
        """Owner of the property."""
        owner = cast(type[OwnT2], self.typeref.args[OwnT].common_type)
        return owner if owner is not Any else None

    @staticmethod
    def _set_owner(
        prop: Data,
        owner: type[Model],
    ) -> None:
        """Set the owner of the property."""
        prop.context = Interface(owner)
        prop.typeref = copy_and_override(
            TypeRef,
            prop.typeref or TypeRef(),
            var_map=dict(prop.typeref.var_map if prop.typeref is not None else {})
            | get_typevar_map(owner)
            | {OwnT: TypeRef(owner)},
        )

    @overload
    @staticmethod
    def _with_new_root_owner(
        data: Prop[ValT2, IdxT2, CruT2, Any, DxT2, ExT2, RwxT2],
        owner: type[OwnT2],
    ) -> Prop[ValT2, IdxT2, CruT2, OwnT2, DxT2, ExT2, RwxT2]: ...

    @overload
    @staticmethod
    def _with_new_root_owner(
        data: Data[ValT2, Expand[IdxT2], DxT2, ExT2, RwxT2, Interface[Any], *CtxTt2],
        owner: type[OwnT2],
    ) -> Data[ValT2, Expand[IdxT2], DxT2, ExT2, RwxT2, Interface[OwnT2], *CtxTt2]: ...

    @staticmethod
    def _with_new_root_owner(
        data: Data,
        owner: type[OwnT2],
    ) -> Data:
        """Create a new data object with a new root owner."""
        if isinstance(data.context, Interface):
            root = copy_and_override(type(data), data, context=data.context)
            Prop._set_owner(root, owner)
            return root
        else:
            assert isinstance(data.context, Data)
            return copy_and_override(
                type(data), data, context=Prop._with_new_root_owner(data.context, owner)
            )

    def _get_with_owner(
        self, owner: type[OwnT2]
    ) -> Data[ValT, Expand[IdxT], DxT, ExT, RwxT, Interface[OwnT2]]:
        """Get the property from the owner map."""
        matching = [p for o, p in self._owner_map.items() if issubclass(owner, o)]

        if len(matching) == 0:
            prop = Prop._with_new_root_owner(self, owner)
            self._owner_map[cast(type[OwnT], owner)] = prop
            matching = [prop]

        return cast(
            Data[ValT, Expand[IdxT], DxT, ExT, RwxT, Interface[Any]],
            matching[0],
        )

    # Descriptor read/write:

    @overload
    def __get__(
        self, instance: None, owner: type[OwnT2]
    ) -> Prop[ValT, IdxT, CruT, OwnT2, DxT, ExT, RwxT]: ...

    @overload
    def __get__(
        self: Prop[Any, FullIdx[()]], instance: OwnT2, owner: type[OwnT2]
    ) -> ValT: ...

    @overload
    def __get__(
        self, instance: OwnT2, owner: type[OwnT2]
    ) -> Data[ValT, IdxT, DxT, ExT, RwxT, Root, Ctx[OwnT2, Idx[()]], Tab]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Self: ...

    def __get__(self: Prop, instance: Any, owner: type | None) -> Any:  # noqa: D105
        if owner is None or not issubclass(owner, Model):
            return self

        if instance is None and owner is not self.owner:
            return self._get_with_owner(owner)

        res = self._getter(cast(OwnT, instance))
        if res is not Not.resolved:
            return res

        if self.default is not Not.defined:
            return self.default

        assert self.default_factory is not None
        return self.default_factory()

    @overload
    def __set__(
        self: Prop[ValT2, Any, Any, Any, Any, Any, C],
        instance: OwnT,
        value: Init[ValT2],
    ) -> None: ...

    @overload
    def __set__(
        self: Prop[ValT2, Any, Any, Any, Any, Any, U],
        instance: OwnT,
        value: ValT2,
    ) -> None: ...

    @overload
    def __set__(self: Prop, instance: Any, value: Any) -> None: ...

    def __set__(self: Prop, instance: Any, value: Any) -> None:  # noqa: D105
        if isinstance(instance, Model):
            self._setter(instance, value)

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.typeref, self.alias, self.owner))

    @cached_prop
    def _val_pyd_type_adapter(self) -> TypeAdapter | None:
        """Get the pydantic type adapter for this property."""
        try:
            ta = TypeAdapter(self.typeref.typeform if self.typeref is not None else Any)
            ta.rebuild(raise_errors=True)
            return ta
        except PydanticSchemaGenerationError:
            return None

    @cached_prop
    def _val_storage_types(self) -> StorageTypes:
        """Get the storage types for this property."""
        try:
            return get_storage_types(
                self.typeref.typeform if self.typeref is not None else object
            )
        except TypeError:
            return StorageTypes([])

    def _get_val_json_schema(self) -> JsonSchemaValue | None:
        """Get the JSON schema for this property."""
        if self._val_pyd_type_adapter is not None:
            return self._val_pyd_type_adapter.json_schema()

        if self._val_storage_types.match_targets({JSONDict, JSONFile}):
            return {
                "type": "object",
            }

        return None


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

        for name, hint in get_annotations(cls).items():
            if name not in cls.__dict__:
                prop_type: type[Prop] | None = TypeRef(hint).base_type(bound=Prop)

                if prop_type is None:
                    continue

                default_type = Prop._get_default_class(prop_type, cls)
                if default_type is not None and is_subtype(default_type, prop_type):
                    prop_type = cast(type, default_type)

                prop = prop_type()

                props[name] = prop
                setattr(cls, name, prop)
                prop.__set_name__(cls, name)

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
                        prop.__set_name__(cls, prop_name)
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
            p.name: Field(
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


@dataclass(kw_only=True)
class Memory(Root["Model"]):
    """In-memory base for models."""


@dataclass(kw_only=True)
class Singleton(Data[ModT, Idx[()], Tab, PL, R, Memory]):
    """Local, in-memory singleton model instance."""

    context: Memory | Data[Any, Any, Any, Any, Any, Memory] = field(
        default_factory=Memory
    )
    model: ModT

    def _id(self) -> str:
        """Identity of the data object."""
        # TODO: Implement this method for the Singleton class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> Idx[()]:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Prop,),
    eq_default=False,
)
class Model(
    metaclass=ModelMeta,
):
    """Schema for a record in a database."""

    _template: ClassVar[bool]
    _derivate: ClassVar[bool] = False
    _root_class: ClassVar[bool] = True
    _fqn: ClassVar[str] = "dbase.Model"

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
    __pydantic_model: ClassVar[type[BaseModel]] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)
        cls._root_class = False
        if "_template" not in cls.__dict__:
            cls._template = False

        cls._fqn = f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def _pydantic_model(cls) -> type[BaseModel]:
        """Return the pydantic model for this record."""
        if cls.__pydantic_model is None:
            cls.__pydantic_model = create_model(
                cls.__name__,
                **cast(
                    dict[str, Any],
                    {
                        p.name: (
                            p.typeref.hint if p.typeref is not None else Any,
                            p.default,
                        )
                        for p in cls._props.values()
                    },
                ),
            )

        return cls.__pydantic_model

    @classmethod
    def _pl_schema(cls) -> dict[str, pl.DataType | type | None]:
        """Return the schema of the dataset."""
        return get_pl_schema(
            {
                name: prop.value_typeref.common_type
                for name, prop in cls._props.items()
                if prop._getter is not None
            }
        )

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

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: D105
        cls, source: type[Any], handler: Callable[[Any], CoreSchema]
    ) -> core_schema.CoreSchema:
        # TODO: Generate core schema based on pydantic model and storage types of props.
        # Wrap below `__store__` method for serialization: https://docs.pydantic.dev/latest/api/pydantic_core_schema/#pydantic_core.core_schema.simple_ser_schema
        # Wrap below `__load__` method for deserialization: https://docs.pydantic.dev/latest/api/pydantic_core_schema/#pydantic_core.core_schema.with_info_after_validator_function
        raise NotImplementedError()

    @classmethod
    def __get_pydantic_json_schema__(  # noqa: D105
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # TODO: Generate JSON schema based on json schema and storage types of props.
        # If all props have a value json schema, just return the composite schema.
        # If some props have no value json schema, replace them with references to their
        # storage files.
        raise NotImplementedError()

    @classmethod
    def __storage_types__(cls) -> StorageTypes[JSONDict | JSONFile | Dir]:
        """Return the set of types that this class can convert to/from."""
        # TODO: Return storage type based on storage types of props.
        # All JSON -> JSON | JSONFile | Dir
        # Some Non-JSON -> Dir
        raise NotImplementedError()

    @classmethod
    def __load__[T](
        cls: type[T],
        source: JSONDict | JSONFile | Dir,
        realm: Realm,
        annotation: SingleTypeDef[T] | None = None,
    ) -> T:
        """Parse from the source type or format."""
        # Upon deferring to pydantic, supply current root (e.g. Realm) as context info.
        ...

    def __store__[T](
        self: Storable[T],
        target: T,
        realm: Realm,
        annotation: SingleTypeDef[Storable[T]] | None = None,
    ) -> None:
        """Convert to the target type or format."""
        # Upon deferring to pydantic, supply current root (e.g. Realm) as context info.
        ...

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new record instance."""
        super().__init__()

        kwargs = self._pydantic_model().model_validate(kwargs).model_dump()

        cls = type(self)
        props = cls._props

        init_sequence = sorted(
            props.values(),
            key=lambda p: p.init_level,
        )

        for prop in init_sequence:
            setattr(self, prop.name, kwargs[prop.name])

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

    def __repr__(self) -> str:
        """Return a string representation of the record."""
        return f"{type(self).__name__}({repr(self._to_dict())})"

    @cached_method
    def _data(self) -> Data[Self, Idx[()], Tab, Any, Any, Root]:
        """Return the singleton class for this record."""
        return Singleton(model=self)

    def __getitem__(
        self,
        prop: Prop[
            ValT2,
            Idx[*KeyTt2],
            Any,
            Self,
            DxT2,
            ExT2,
            RwxT2,
        ],
    ) -> Data[
        ValT2,
        Idx[*KeyTt2],
        DxT2,
        ExT2,
        RwxT2,
        Root,
        Ctx[Self],
    ]:
        """Get the data representation of a prop."""
        return self._data()[prop]


ModTi = TypeVar("ModTi", bound=Model, default=Any)


class Crawl(Data[ModTi, Idx, Tab, SQL, R, Interface[ModTi, Any, Tab]]):
    """Perform recursive union through prop loops."""

    loops: Iterable[Data[ModTi, Any, Tab, SQL, Any, Interface[ModTi, Any, Tab]]]

    def _id(self) -> str:
        """Name of the property."""
        # TODO: Implement this method for the Crawl class.
        raise NotImplementedError()

    def _index(
        self,
    ) -> Idx:
        """Get the index of this data."""
        raise NotImplementedError()

    def _frame(
        self: Data[Any, Any, SxT2],
    ) -> Frame[PL, SxT2]:
        """Get SQL-side reference to this property."""
        raise NotImplementedError()
