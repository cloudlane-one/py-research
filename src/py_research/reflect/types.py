"""Reflection utilities for types."""

from __future__ import annotations

import inspect
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from functools import cached_property, reduce
from inspect import getmodule, getmro
from itertools import chain, groupby
from types import ModuleType, NoneType, UnionType, new_class
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Generic,
    Literal,
    Self,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    Union,
    cast,
    get_args,
    get_origin,
)

from beartype.door import is_bearable, is_subhint
from beartype.roar import BeartypeDoorNonpepException
from pydantic import TypeAdapter
from typing_extensions import TypeVar as ExtTypeVar
from typing_extensions import TypeVarTuple as ExtTypeVarTuple

from py_research.descriptors import prop
from py_research.reflect.runtime import get_subclasses
from py_research.types import (
    AnnotationScanType,
    GenericAlias,
    RuntimeValidated,
    SingleTypeDef,
)


def _resolve_typealias(
    type_: SingleTypeDef | UnionType | Annotated,
) -> SingleTypeDef | UnionType:
    return (
        type_.__value__
        if isinstance(type_, TypeAliasType)
        else get_args(type_)[0] if get_origin(type_) is Annotated else type_
    )


_type_alias_classes: dict[TypeAliasType, type] = {}


def typedef_to_typeset(
    typedef: SingleTypeDef | UnionType | None,
    typevar_map: dict[TypeVar | ExtTypeVar, TypeRef] | None = None,
    ctx_module: ModuleType | None = None,
    remove_null: bool = False,
) -> set[type]:
    """Convert type definition to set of types (>1 in case of union).

    Args:
        typedef: The type definition to convert.
        typevar_map: Mapping of type variables to their actual types.
        ctx_module: Context module for resolving forward-referenced types.
        remove_null: Whether to remove ``NoneType`` from the resulting typeset.

    Returns:
        Set of types.
    """
    typedef_set: set[SingleTypeDef | UnionType | None] = {typedef}
    root_orig = get_origin(typedef)

    if isinstance(typedef, UnionType) or root_orig is Union:
        typedef_set = {
            get_origin(union_arg) or union_arg for union_arg in get_args(typedef)
        }

        if remove_null:
            typedef_set &= {
                t for t in typedef_set if t is not None and not is_subtype(t, NoneType)
            }

    typeset: set[type] = set()

    for t in typedef_set:
        if t is None:
            typeset.add(NoneType)
            continue

        if isinstance(t, type):
            typeset.add(t)
            continue

        t_parsed = (
            TypeRef(t.__value__, subs=typevar_map or {}, ctx_module=ctx_module).typeform
            if isinstance(t, TypeAliasType)
            else t
        )

        if isinstance(t, TypeAliasType):
            cls = _type_alias_classes.get(t) or new_class(
                t.__name__,
                (t_parsed,),
                None,
                lambda ns: ns.update({"_src_mod": (ctx_module or getmodule(t))}),
            )
            _type_alias_classes[t] = cls
            typeset.add(cls)
            continue

        orig = get_origin(t_parsed)
        if orig is None or orig is Literal:
            typeset.add(object)
            continue

        if orig is UnionType or orig is Union:
            typeset |= typedef_to_typeset(
                t_parsed,
                typevar_map=typevar_map,
                ctx_module=ctx_module,
                remove_null=remove_null,
            )
            continue

        assert isinstance(orig, type)
        typeset.add(orig)

    return typeset


def is_subtype[T](
    type_: SingleTypeDef | UnionType | Annotated, supertype: T
) -> TypeGuard[T]:
    """Check if object is of given type hint.

    Args:
        type_: The type to check.
        supertype: The supertype to check against.

    Returns:
        True if type_ is a subtype of supertype.
    """
    type_args = get_args(type_)
    supertype_args = get_args(supertype)

    if len(type_args) == 0 and len(supertype_args) == 0:
        # Special case since beartype can't handle empty typevartuples as args.
        typeset = typedef_to_typeset(type_)
        supertype_tuple = tuple(
            typedef_to_typeset(cast(SingleTypeDef | UnionType, supertype))
        )
        return all(issubclass(t, supertype_tuple) for t in typeset)

    res_type = _resolve_typealias(type_)
    res_supertype = _resolve_typealias(supertype)

    try:
        return is_subhint(
            res_type,  # pyright: ignore[reportArgumentType]
            res_supertype,  # pyright: ignore[reportArgumentType]
        )
    except BeartypeDoorNonpepException:
        # Fallback to only checking origin types:
        type_origin = get_origin(res_type)
        supertype_origin = get_origin(res_supertype)
        return (
            isinstance(type_origin, type)
            and isinstance(supertype_origin, type)
            and issubclass(type_origin, supertype_origin)
        )


def has_type[T](obj: Any, type_: SingleTypeDef[T] | UnionType) -> TypeGuard[T]:
    """Check if object is of given type hint.

    Args:
        obj: The object to check.
        type_: The type to check against.

    Returns:
        True if object is of given type.
    """
    return is_bearable(obj, type_)  # pyright: ignore[reportArgumentType]


def get_lowest_common_base(types: Iterable[type]) -> type:
    """Return the lowest common base of given types.

    Args:
        types: The types to get the common base for.

    Returns:
        The lowest common base type.
    """
    if len(list(types)) == 0:
        return object

    bases_of_all = reduce(set.intersection, (set(getmro(t)) for t in types))
    return max(bases_of_all, key=lambda b: sum(issubclass(b, t) for t in bases_of_all))


def extract_nullable_type[T](
    type_: SingleTypeDef[T | None] | UnionType,
) -> type[T] | None:
    """Extract the non-none base type of a union.

    Args:
        type_: The type to extract from.

    Returns:
        The non-none base type or ``None`` if not found.
    """
    args = get_args(type_)

    if len(args) == 0 and has_type(type_, SingleTypeDef):
        return type_

    notna_args = {arg for arg in args if (get_origin(arg) or arg) is not NoneType}
    return get_lowest_common_base(notna_args) if notna_args else None


def get_inheritance_distance(cls: type, base: type) -> int | None:
    """Return the inheritance distance between two classes.

    Note: Positive direction is from subclass to base class. If arguments are
    are reversed, the sign will be negative.

    Warning: Untested function.

    Args:
        cls: The subclass.
        base: The base class.

    Returns:
        The signed inheritance distance between the two classes. If the base class is
        not a base of the subclass, None is returned.
    """
    if not isinstance(cls, type) or not isinstance(base, type):
        return None

    if cls is base:
        return 0

    if issubclass(cls, base):
        cls, base, sign = (cls, base, 1)
    elif issubclass(base, cls):
        cls, base, sign = (base, cls, -1)
    else:
        return None

    distance = 1
    bases = set(cls.__bases__)
    while base not in bases and distance < 100:
        bases = reduce(set.union, (set(b.__bases__) for b in bases))
        distance += 1

    return distance * sign


def set_typeargs[T](
    typedef: SingleTypeDef[T],
    args: (
        Sequence[SingleTypeDef | UnionType | TypeVar]
        | dict[TypeVar, SingleTypeDef | UnionType | TypeVar]
    ),
) -> GenericAlias[T]:
    """Set a typevar in a generic type hint.

    Args:
        typedef: The type definition to set type arguments for.
        args: The type arguments to set.

    Returns:
        The generic type with set type arguments.
    """
    orig = get_origin(typedef)
    assert orig is not None

    args = get_args(typedef)
    assert len(args) > 0

    if not isinstance(args, dict):
        return orig[*args]

    assert hasattr(orig, "__parameters__")
    typearg_map = dict(zip(getattr(orig, "__parameters__"), args))
    typevar_map = TypeRef(typedef).typevar_map

    for typevar, arg in args.items():
        # Go through substitutions if typevar is not directly in arg_map.
        while typevar not in typearg_map:
            subs_typevar = typevar_map[typevar]
            assert isinstance(subs_typevar, TypeVar)
            typevar = subs_typevar

        typearg_map[typevar] = arg

    return orig[*typearg_map.values()]


def get_typeargs(instance: Any) -> tuple[type, ...] | None:
    """Get the type arguments of a generic instance.

    Args:
        instance: The instance to get type arguments for.

    Returns:
        The type arguments or ``None`` if not found.
    """
    if hasattr(instance, "__orig_class__"):
        orig_class = getattr(instance, "__orig_class__")
        return get_args(orig_class)

    return None


T = TypeVar("T", covariant=True)


@dataclass
class TypeRef(Generic[T]):
    """Reference to a typeform."""

    hint: SingleTypeDef[T] | UnionType | ForwardRef | str | TypeVar | ExtTypeVar = cast(
        type[T], object
    )
    """Type hint."""

    ctx_module: ModuleType | None = None
    """Context module for resolving forward-referenced types."""

    ctx: dict[str, Any] = field(default_factory=dict)
    """Context mapping for resolving forward-referenced types."""

    subs: Mapping[TypeVar | ExtTypeVar, TypeRef] = field(default_factory=dict)
    """Substitutions for type variables."""

    overrides: Mapping[TypeVar | ExtTypeVar, TypeRef] = field(default_factory=dict)
    """Overrides for type variables."""

    @cached_property
    def typedef(self) -> SingleTypeDef[T] | UnionType | Annotated:
        """Resolved type definition."""
        if isinstance(self.hint, str):
            hint = eval(
                self.hint,
                {
                    **globals(),
                    **(vars(self.ctx_module) if self.ctx_module else {}),
                    **self.ctx,
                    **{k.__name__: v.hint for k, v in self.subs.items()},
                    **{k.__name__: v.hint for k, v in self.overrides.items()},
                },
            )
            return TypeRef(
                hint,
                ctx_module=self.ctx_module,
                ctx=self.ctx,
                subs=self.subs,
                overrides=self.overrides,
            ).typedef

        if isinstance(self.hint, ForwardRef):
            evaluated = self.hint._evaluate(
                {
                    **globals(),
                    **(vars(self.ctx_module) if self.ctx_module else {}),
                    **self.ctx,
                    **{k.__name__: v.hint for k, v in self.subs.items()},
                    **{k.__name__: v.hint for k, v in self.overrides.items()},
                },
                None,
                recursive_guard=frozenset(),
            )
            assert evaluated is not None
            return evaluated

        if isinstance(self.hint, TypeVar | ExtTypeVar):
            typeref = self.subs.get(
                self.hint,
                TypeRef(
                    (
                        self.hint.__default__
                        if isinstance(self.hint, ExtTypeVar)
                        and hasattr(self.hint, "__default__")
                        and self.hint.has_default()
                        and self.hint.__default__ is not Any
                        else (
                            self.hint.__bound__
                            if self.hint.__bound__ is not None
                            else object
                        )
                    ),
                    ctx_module=getmodule(self.hint),
                    subs=self.subs,
                    overrides=self.overrides,
                ),
            )

            return typeref.typedef

        return self.hint

    @cached_property
    def base_type(self) -> type[T] | None:
        """Base type in case of a generic typedef."""
        if isinstance(self.hint, ForwardRef | str):
            # Avoid evaluating the full typeform if supplied as str or ForwardRef.
            hint_str = (
                self.hint.__forward_value__
                if isinstance(self.hint, ForwardRef)
                else self.hint
            )

            if hint_str is None:
                return None

            type_str = (
                hint_str
                if "|" in hint_str or hint_str.startswith("Union[")
                else hint_str.split("[", 1)[0]
            )
            try:
                return eval(
                    type_str,
                    {
                        **globals(),
                        **(vars(self.ctx_module) if self.ctx_module else {}),
                        **self.ctx,
                    },
                )
            except Exception:
                return None

        return get_origin(self.typedef)

    @cached_property
    def typeset(self) -> set[type[T]]:
        """Set of concrete types (>1 in case of union typeform)."""
        return typedef_to_typeset(self.typedef)

    @cached_property
    def common_type(self) -> type[T]:
        """Common base type of the typeset."""
        return get_lowest_common_base(
            typedef_to_typeset(
                self.typedef,
                remove_null=True,
            )
        )

    @cached_property
    def origin(self) -> type | TypeAliasType:
        """Origin of the type reference."""
        return (
            self.typedef.__origin__
            if isinstance(self.typedef, type)
            and issubclass(self.typedef, TypeAwareClass)
            else (
                get_origin(self.typedef)
                or getattr(self.typedef, "__origin__", None)
                or self.common_type
            )
        )

    @cached_property
    def params(self) -> tuple[TypeVar | ExtTypeVar, ...]:
        """Type parameters of the type reference."""
        return (
            (self.origin,)
            if isinstance(self.origin, TypeVar | ExtTypeVar)
            else getattr(self.origin, "__parameters__", None)
            or getattr(self.origin, "__type_params__")
        )

    @cached_property
    def args(self) -> tuple[Any, ...]:
        """Type arguments of the type reference."""
        return (
            self.typedef.__args__
            if isinstance(self.typedef, type)
            and issubclass(self.typedef, TypeAwareClass)
            else (get_args(self.typedef) or getattr(self.typedef, "__args__", ()))
        )

    @cached_property
    def annotation(self) -> Any | None:
        """The dynamic annotation supplied via ``typing.Annotated``."""
        if self.origin is Annotated:
            return self.args[1] if len(self.args) > 1 else None
        return None

    @cached_property
    def literal_values(self) -> tuple[str | int | float | bool | Enum, ...]:
        """Values of a ``Literal`` type hint."""
        return self.args if self.origin is Literal else ()

    @property
    def single_typedef(self) -> SingleTypeDef[T]:
        """Resolved type definition without union."""
        return (
            self.typedef
            if not isinstance(self.typedef, UnionType)
            else self.common_type
        )

    @cached_property
    def bases(self) -> tuple[type, *tuple[type, ...]]:
        """Bases of the type reference."""
        bases = reduce(
            set.intersection,
            (set(t.__dict__.get("__orig_bases__", t.__bases__)) for t in self.typeset),
        )

        return tuple(bases) or (object,)

    @cached_property
    def local_typevar_map(self) -> dict[TypeVar | ExtTypeVar, TypeRef]:
        """Mapping of argument type variables to their values."""
        ctx_module = self.ctx_module or getmodule(self.typedef)

        # First get the typevar params of the typehint.
        raw_arg_map: dict[Any, Any] = {}

        if len(self.params) > 0 and len(self.args) > 0:
            # Map typevar params to typeargs, which may be typevars themselves.
            raw_arg_map = {}

            # Initialize a list of remaining type arguments,
            # which are yet to be matched to params.
            # Order is reversed for efficient popping.
            remaining_args = list(reversed(self.args))

            for i, p in enumerate(self.params):
                # If no more args are left, fill the rest of the map
                # with the type params themselves (identity map).
                if len(remaining_args) == 0:
                    raw_arg_map[p] = p

                if isinstance(p, TypeVar | ExtTypeVar):
                    # In case of a normal typevar, map the next arg.
                    raw_arg_map[p] = remaining_args.pop()
                elif isinstance(p, TypeVarTuple | ExtTypeVarTuple):
                    # In case of a typevar-tuple, map all remaining args
                    # minus those associated to downstream typevars.
                    remaining_params = len(self.params) - i - 1
                    mapped_args = range(len(remaining_args) - remaining_params)

                    if len(mapped_args) >= 0:
                        raw_arg_map[p] = tuple(
                            remaining_args.pop()
                            for _ in range(len(remaining_args) - remaining_params)
                        )
                    else:
                        raise TypeError("Incompatible type arguments for typevar-tuple")
                else:
                    raise TypeError("Unsupported typevar type")

        # Merge with defaults, substitutions, and overrides.
        local_typevar_map = (
            {
                p: TypeRef(
                    p,
                    ctx_module=ctx_module,
                )
                for p in self.params
            }
            | dict(self.subs)
            | {
                k: TypeRef(
                    v,
                    ctx_module=ctx_module,
                )
                for k, v in raw_arg_map.items()
            }
            | dict(self.overrides)
        )

        # Make sure every arg-typeref can access all other arg-typerefs.
        for s in local_typevar_map.values():
            s.subs = local_typevar_map

        return local_typevar_map

    @cached_property
    def typevar_map(
        self,
    ) -> dict[TypeVar | ExtTypeVar, TypeRef]:
        """Return a mapping of type variables to their values, including those of bases."""
        ctx_module = self.ctx_module or getmodule(self.typedef)
        typevar_map: dict[TypeVar | ExtTypeVar, TypeRef] = {}

        if isinstance(self.typedef, UnionType) or self.origin is Union:
            # Resolve typevar map of each union arg individually and
            # union the resulting types per typevar.
            base_typevar_items = chain(
                *(
                    TypeRef(
                        arg, subs=self.subs, overrides=self.overrides
                    ).typevar_map.items()
                    for arg in self.args
                )
            )
            groups = groupby(
                sorted(base_typevar_items, key=lambda x: x[0].__name__),
                key=lambda x: x[0].__name__,
            )
            group_values = [list(g) for _, g in groups]
            typevar_map = {
                g[0][0]: TypeRef(
                    reduce(operator.or_, (v.typedef for _, v in g)),
                    subs=self.subs,
                    overrides=self.overrides,
                    ctx_module=ctx_module,
                )
                for g in group_values
            }

        elif self.typedef is not Generic and self.origin is not Generic:
            # Anything else means we have a pure type or a generic typehint.

            # Ascend to generic base classes or the value type
            # in case of a named type alias.
            base_arg_map = {}
            if isinstance(self.origin, type):
                base_arg_map = (
                    reduce(
                        lambda x, y: x | y,
                        (
                            TypeRef(
                                base,
                                subs=self.local_typevar_map,
                            ).typevar_map
                            for base in self.bases
                        ),
                    )
                    if len(self.bases) > 1
                    else TypeRef(
                        self.bases[0],
                        subs=self.local_typevar_map,
                    ).typevar_map
                )
            elif isinstance(self.origin, TypeAliasType):
                base_arg_map = TypeRef(
                    self.origin.__value__,
                    subs=self.local_typevar_map,
                ).typevar_map

            # Merge with base typevar map.
            typevar_map = base_arg_map | self.local_typevar_map

        return typevar_map

    @cached_property
    def typeform(self) -> SingleTypeDef[T] | UnionType | Annotated:
        """Resolved type definition with arguments recursively resolved."""
        orig = self.origin

        if orig is UnionType:
            orig = Union

        if orig is not None and len(self.args) > 0:
            if orig is Literal:
                return Literal[*self.args]  # pyright: ignore[reportReturnType]

            if orig is Annotated:
                return Annotated[*(self.typevar_map[p] for p in self.params)]

            return orig[  # pyright: ignore[reportIndexIssue]
                *(self.typevar_map[p] for p in self.params)
            ]

        return self.typedef

    def validate(self, obj: Any) -> TypeGuard[T]:
        """Check if object is of this type."""
        checks_out = has_type(obj, self.typeform)

        if checks_out and (
            isinstance(self.typeform, RuntimeValidated)
            or isinstance(self.annotation, RuntimeValidated)
        ):
            checks_out = TypeAdapter(self.typeform).validate_python(obj)

        return checks_out


class TypeAwareClass:
    """Class, which is aware of its type arguments."""

    __origin__: type[TypeAwareClass]
    __args__: tuple[AnnotationScanType, ...]

    @prop(cached=True, mode="class")
    @classmethod
    def typeargs(cls) -> dict[TypeVar | ExtTypeVar, TypeRef]:
        """Type arguments of this class."""
        return TypeRef(cls).typevar_map

    @prop(mode="class")
    @classmethod
    def type_params(cls) -> tuple[TypeVar | ExtTypeVar, ...]:
        """Type arguments of this class."""
        return cls.__dict__.get("__parameters__", ()) or cls.__dict__.get(
            "__type_params__", ()
        )

    @prop(mode="class", cached=True)
    @classmethod
    def _generic_subclasses(cls) -> dict[Any, type[Self]]:
        return {}

    def __init_subclass__(cls):
        if "__origin__" not in cls.__dict__:
            cls.__origin__ = cls
        if "__args__" not in cls.__dict__:
            cls.__args__ = ()
        if "__parameters__" not in cls.__dict__ and Generic not in cls.__bases__:
            cls.__parameters__ = ()
        if "__type_params__" not in cls.__dict__ and Generic not in cls.__bases__:
            cls.__type_params__ = ()

        super().__init_subclass__()

    def __class_getitem__(cls, item: Any) -> type[Self]:
        if item in cls._generic_subclasses:
            return cls._generic_subclasses[item]

        subclass = new_class(
            f"{cls.__name__}[{item.__name__ if hasattr(item, '__name__') else item}]",
            bases=(cls,),
            exec_body=lambda ns: ns.update(
                {
                    "__origin__": cls,
                    "__args__": item if isinstance(item, tuple) else (item,),
                    "__module__": cls.__module__,
                }
            ),
        )
        cls._generic_subclasses[item] = subclass
        return subclass


def is_frozen_dataclass(obj: Any) -> bool:
    """Check if an object is a frozen dataclass.

    Args:
        obj: The object to check.

    Returns:
        ``True`` if the object is a frozen dataclass.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return is_dataclass(obj) and getattr(cls, "__dataclass_params__").frozen


def is_immutable(obj: Any) -> bool:
    """Check if an object is immutable.

    Args:
        obj: The object to check.

    Returns:
        ``True`` if the object is immutable.
    """
    # Built-in immutables
    if isinstance(obj, int | float | bool | str | bytes | frozenset):
        return True

    # Tuples: check recursively
    if isinstance(obj, tuple):
        return all(is_immutable(el) for el in obj)

    # Frozen dataclass
    if is_frozen_dataclass(obj):
        return True

    return False


def get_nondefault_methods(cls: type) -> dict[str, Any]:
    """Get the all attributes on a class without those inherited from ``object``.

    Args:
        cls: The class to get methods for.

    Returns:
        Mapping of method names to method objects.
    """
    return {
        name: member
        for name, member in inspect.getmembers(
            cls,
            predicate=lambda m: inspect.ismethoddescriptor(m)
            or inspect.isfunction(m)
            or inspect.isbuiltin(m),
        )
        if not (
            (
                inspect.getmodule(member) is None
                and str(member.__qualname__).startswith("object" + ".")
            )
            or name == "__subclasshook__"
        )
    }


def get_own_attrs(
    cls: type,
    predicate: Callable[[object], bool] | None = None,
    include_bases: tuple[type, ...] = (),
) -> set[str]:
    """Get all non-inherited attributes on a class.

    Args:
        cls: The class to get attributes for.
        predicate: Optional predicate to filter attributes.
        include_bases: Additional base classes to include attributes from.

    Returns:
        Set of attribute names.
    """
    attrs = set(cls.__dict__.keys())

    if hasattr(cls, "__annotations__"):
        attrs.update(cls.__annotations__.keys())

    included_cls_mods = [
        supcls
        for base in include_bases
        for supcls in (base, *get_subclasses(base))
        if issubclass(cls, supcls) and supcls is not cls
    ]
    sup_attrs = reduce(
        set.__or__,
        (
            get_own_attrs(
                supcls,
                predicate=predicate,
                include_bases=(),
            )
            for supcls in included_cls_mods
        ),
        set(),
    )

    return sup_attrs | attrs
