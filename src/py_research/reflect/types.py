"""Reflection utilities for types."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, reduce
from inspect import getmodule, getmro
from itertools import chain, groupby
from types import ModuleType, NoneType, UnionType, new_class
from typing import (
    Any,
    ForwardRef,
    Generic,
    Literal,
    NewType,
    Protocol,
    TypeAliasType,
    TypeGuard,
    Union,
    cast,
    final,
    get_args,
    get_origin,
    overload,
)

from beartype.door import is_bearable, is_subhint
from typing_extensions import TypeVar, runtime_checkable

from py_research.caching import cached_prop
from py_research.hashing import gen_str_hash
from py_research.reflect.runtime import get_subclasses
from py_research.types import Not

T = TypeVar("T", covariant=True)
T_cov = TypeVar("T_cov", covariant=True)
U_cov = TypeVar("U_cov", covariant=True)

type _AnnotationScanType = type[Any] | TypeAliasType | GenericAlias[
    Any
] | NewType | ForwardRef | str


@runtime_checkable
class GenericAlias(Protocol[T]):  # type: ignore
    """protocol for generic types.

    this since Python.typing _GenericAlias is private

    """

    __args__: tuple[_AnnotationScanType, ...]
    __origin__: type[T]


type SingleTypeDef[T] = GenericAlias[T] | TypeAliasType | type[T] | NewType


@runtime_checkable
class SupportsItems(Protocol[T_cov, U_cov]):
    """Protocol for objects that support item access."""

    def keys(self) -> Iterable[T_cov]: ...  # noqa: D102

    def values(self) -> Iterable[U_cov]: ...  # noqa: D102

    def items(self) -> Iterable[tuple[T_cov, U_cov]]: ...  # noqa: D102


T_contra = TypeVar("T_contra", contravariant=True)


@final
@dataclass
class ContraType(Generic[T_contra]):
    """Represent a contravariant type."""

    type_: type[T_contra] | None = None


def is_subtype(type_: SingleTypeDef | UnionType, supertype: T) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return (
        is_subhint(type_, supertype)
        if not isinstance(type_, TypeAliasType)
        else is_subhint(type_.__value__, supertype)
    )


def has_type(obj: Any, type_: SingleTypeDef[T] | UnionType) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    return is_bearable(obj, type_)


def get_lowest_common_base(types: Iterable[type]) -> type:
    """Return the lowest common base of given types."""
    if len(list(types)) == 0:
        return object

    bases_of_all = reduce(set.intersection, (set(getmro(t)) for t in types))
    return max(bases_of_all, key=lambda b: sum(issubclass(b, t) for t in bases_of_all))


def extract_nullable_type(type_: SingleTypeDef[T | None] | UnionType) -> type[T] | None:
    """Extract the non-none base type of a union."""
    args = get_args(type_)

    if len(args) == 0 and has_type(type_, SingleTypeDef):
        return type_

    notna_args = {arg for arg in args if get_origin(arg) is not NoneType}
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


_type_alias_classes: dict[TypeAliasType, type] = {}


@overload
def hint_to_typedef(
    hint: SingleTypeDef,
    *,
    typevar_map: Mapping[TypeVar, SingleTypeDef | UnionType | TypeVar] | None = ...,
    ctx_module: ModuleType | None = ...,
    unresolved_typevars: Literal["raise", "keep"] = ...,
) -> SingleTypeDef: ...


@overload
def hint_to_typedef(
    hint: SingleTypeDef | UnionType | TypeVar | str | ForwardRef,
    *,
    typevar_map: Mapping[TypeVar, SingleTypeDef | UnionType | TypeVar] | None = ...,
    ctx_module: ModuleType | None = ...,
    unresolved_typevars: Literal["raise"] = ...,
) -> SingleTypeDef | UnionType: ...


@overload
def hint_to_typedef(
    hint: SingleTypeDef | UnionType | TypeVar | str | ForwardRef,
    *,
    typevar_map: Mapping[TypeVar, SingleTypeDef | UnionType | TypeVar] | None = ...,
    ctx_module: ModuleType | None = ...,
    unresolved_typevars: Literal["keep"],
) -> SingleTypeDef | UnionType | TypeVar: ...


def hint_to_typedef(
    hint: SingleTypeDef | UnionType | TypeVar | str | ForwardRef,
    typevar_map: Mapping[TypeVar, SingleTypeDef | UnionType | TypeVar] | None = None,
    ctx_module: ModuleType | None = None,
    unresolved_typevars: Literal["raise", "keep"] = "raise",
) -> SingleTypeDef | UnionType | TypeVar:
    """Convert type hint to type definition."""
    typevar_map = typevar_map or {}

    typedef = hint

    if isinstance(typedef, str):
        typedef = eval(
            typedef,
            {**globals(), **(vars(ctx_module) if ctx_module else {})},
        )

    if isinstance(typedef, TypeVar):
        type_res = typevar_map.get(typedef, Not.defined)

        if (
            type_res is Not.defined
            and hasattr(typedef, "__default__")
            and typedef.has_default()
        ):
            type_res = typedef.__default__
        if type_res is Not.defined:
            if unresolved_typevars == "keep":
                return typedef
            else:
                raise TypeError(
                    f"Type variable `{typedef}` not bound for typehint `{hint}`."
                )

        typedef = type_res

    if isinstance(typedef, ForwardRef):
        typedef = typedef._evaluate(
            {**globals(), **(vars(ctx_module) if ctx_module else {})},
            {},
            recursive_guard=frozenset(),
        )

    orig = get_origin(typedef)
    args = get_args(typedef)
    if (
        orig is not None
        and len(args) > 0
        and not (isinstance(typedef, UnionType) or orig is Union)
    ):
        typedef = orig[
            *(
                hint_to_typedef(
                    arg,
                    typevar_map=typevar_map,
                    ctx_module=ctx_module,
                    unresolved_typevars=unresolved_typevars,
                )
                for arg in args
            )
        ]

    return cast(SingleTypeDef, typedef)


def typedef_to_typeset(
    typedef: SingleTypeDef | UnionType | None,
    typevar_map: dict[TypeVar, SingleTypeDef | UnionType] | None = None,
    ctx_module: ModuleType | None = None,
    remove_null: bool = False,
) -> set[type]:
    """Convert type definition to set of types (>1 in case of union)."""
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
            hint_to_typedef(t.__value__, typevar_map=typevar_map, ctx_module=ctx_module)
            if isinstance(t, TypeAliasType)
            else t
        )

        orig = get_origin(t_parsed)
        if orig is None or orig is Literal:
            typeset.add(object)
            continue

        if isinstance(t, TypeAliasType):
            cls = _type_alias_classes.get(t) or new_class(
                t.__name__,
                (t_parsed,),
                None,
                lambda ns: ns.update({"_src_mod": (ctx_module or getmodule(t))}),
            )
            _type_alias_classes[t] = cls
            typeset.add(cls)
        else:
            assert isinstance(orig, type)
            typeset.add(
                new_class(
                    orig.__name__ + "_" + gen_str_hash(get_args(t), 5),
                    (t,),
                    None,
                    lambda ns: ns.update({"_src_mod": (ctx_module or getmodule(t))}),
                )
            )

    return typeset


@overload
def get_typevar_map(
    c: SingleTypeDef | UnionType | tuple[type, ...],
    *,
    subs: Mapping[TypeVar, SingleTypeDef | UnionType | TypeVar] | None = ...,
    subs_defaults: Literal[True] = ...,
    arg_count: int | None = ...,
) -> dict[TypeVar, SingleTypeDef | UnionType]: ...


@overload
def get_typevar_map(
    c: SingleTypeDef | UnionType | tuple[type, ...],
    *,
    subs: Mapping[TypeVar, SingleTypeDef | UnionType | TypeVar] | None = ...,
    subs_defaults: Literal[False],
    arg_count: int | None = ...,
) -> dict[TypeVar, SingleTypeDef | UnionType | TypeVar]: ...


def get_typevar_map(
    c: SingleTypeDef | UnionType | tuple[type, ...],
    subs: Mapping[TypeVar, SingleTypeDef | UnionType | TypeVar] | None = None,
    subs_defaults: bool = True,
    arg_count: int | None = None,
) -> (
    dict[TypeVar, SingleTypeDef | UnionType]
    | dict[TypeVar, SingleTypeDef | UnionType | TypeVar]
):
    """Return a mapping of type variables to their actual types."""
    subs = subs or {}

    typevar_map = {}

    orig = get_origin(c) or c
    args = get_args(c)

    if isinstance(c, UnionType) or orig is Union:
        # Resolve typevar map of each union arg individually and
        # union the resulting types per typevar.
        base_typevar_items = chain(
            *(
                get_typevar_map(arg, subs=subs, subs_defaults=False).items()
                for arg in args
            )
        )
        groups = groupby(
            sorted(base_typevar_items, key=lambda x: x[0].__name__),
            key=lambda x: x[0].__name__,
        )
        group_values = [list(g) for _, g in groups]
        typevar_map = {
            list(g)[0][0]: reduce(operator.or_, (v for _, v in g)) for g in group_values
        }

    elif isinstance(c, tuple):
        # Tuple represents a type intersection via multiple base classes.
        # Just concat all typevar maps of the base classes.
        typevar_map = reduce(
            lambda x, y: x | y,
            (get_typevar_map(base, subs=subs, subs_defaults=False) for base in c),
        )

    elif c is not Generic and orig is not Generic:
        # Anything else means we have a pure type or a generic typehint.

        # First get the typevar params of the typehint.
        local_typevar_map = {}
        if hasattr(orig, "__parameters__"):
            params = getattr(orig, "__parameters__")

            # Map typevar params to typeargs, which may be typevars themselves.
            if len(args) > 0:
                local_typevar_map = dict(zip(params, args))
            else:
                local_typevar_map = {p: p for p in getattr(orig, "__parameters__")}

        # Apply substitutions.
        subs_typevar_map = {
            k: hint_to_typedef(v, typevar_map=subs, unresolved_typevars="keep")
            for k, v in local_typevar_map.items()
        }

        # Ascend to generic base classes or the value type
        # in case of a named type alias.
        base_typevar_map = {}
        if isinstance(orig, type):
            bases = tuple[type, ...](
                getattr(orig, "__orig_bases__")
                if hasattr(orig, "__orig_bases__")
                else orig.__bases__
            )
            if len(bases) > 1:
                base_typevar_map = get_typevar_map(
                    bases if len(bases) > 1 else bases[0],
                    subs=subs_typevar_map,
                    subs_defaults=False,
                )
        elif isinstance(orig, TypeAliasType):
            base_typevar_map = get_typevar_map(
                orig.__value__, subs=subs_typevar_map, subs_defaults=False
            )

        # Merge with base typevar map.
        typevar_map = subs_typevar_map | base_typevar_map

    if subs_defaults:
        # Substitute defaults.
        return {
            p: (
                (
                    cast(type, p.__default__)
                    if hasattr(p, "__default__") and p.has_default()
                    else object
                )
                if isinstance(p, TypeVar)
                else p
            )
            for p in typevar_map.keys()
        }

    return typevar_map


def set_typeargs[T](
    typedef: SingleTypeDef[T],
    args: (
        Sequence[SingleTypeDef | UnionType | TypeVar]
        | dict[TypeVar, SingleTypeDef | UnionType | TypeVar]
    ),
) -> GenericAlias[T]:
    """Set a typevar in a generic type hint."""
    orig = get_origin(typedef)
    assert orig is not None

    args = get_args(typedef)
    assert len(args) > 0

    if not isinstance(args, dict):
        return orig[*args]

    assert hasattr(orig, "__parameters__")
    typearg_map = dict(zip(getattr(orig, "__parameters__"), args))
    typevar_map = get_typevar_map(typedef, subs_defaults=False)

    for typevar, arg in args.items():
        # Go through substitutions if typevar is not directly in arg_map.
        while typevar not in typearg_map:
            subs_typevar = typevar_map[typevar]
            assert isinstance(subs_typevar, TypeVar)
            typevar = subs_typevar

        typearg_map[typevar] = arg

    return orig[*typearg_map.values()]


def get_typeargs(instance: Any) -> tuple[type, ...] | None:
    """Get the type arguments of a generic instance."""
    if hasattr(instance, "__orig_class__"):
        orig_class = getattr(instance, "__orig_class__")
        return get_args(orig_class)

    return None


@cache
def _map_str_annotation_base[U](anno: str, base: type[U]) -> type[U]:
    """Map property type name to class."""
    name_map = {cls.__name__: cls for cls in get_subclasses(base) if cls is not base}
    matches = [name_map[n] for n in name_map if anno.startswith(n + "[")]
    assert len(matches) == 1
    return matches[0]


def get_base_type(hint: SingleTypeDef[T] | str, bound: type[T]) -> type[T]:
    """Resolve base type of the typehint."""
    if has_type(hint, SingleTypeDef):
        base = get_origin(hint)
        if base is None or not issubclass(base, bound):
            raise TypeError(f"Type hint `{hint}` is not a subclass of `{bound}`.")

        return base
    else:
        assert isinstance(hint, str)
        return _map_str_annotation_base(hint, bound)


@dataclass
class TypeRef[T]:
    """Reference to a type."""

    hint: SingleTypeDef[T] | UnionType | str = "Any"
    var_map: Mapping[TypeVar, SingleTypeDef | UnionType] = field(default_factory=dict)
    ctx_module: ModuleType | None = None

    @cached_prop
    def typeform(self) -> SingleTypeDef[T] | UnionType:
        """Return the type format."""
        form = hint_to_typedef(
            self.hint,
            typevar_map=self.var_map,
            ctx_module=self.ctx_module,
        )
        return form

    @cached_prop
    def type_set(self) -> set[type[T]]:
        """Target types of this prop (>1 in case of union typeform)."""
        return typedef_to_typeset(self.typeform)

    @cached_prop
    def common_type(self) -> type:
        """Common base type of the target types."""
        return get_common_type(self.typeform)

    @property
    def base_type(self) -> type[T]:
        """Resolve the prop typehint."""
        typeargs = get_typeargs(self)
        bound: type[Any] = typeargs[0] if typeargs is not None else object

        return get_base_type(self.single_typedef, bound)

    @property
    def single_typedef(self) -> SingleTypeDef[T]:
        """Return the type format."""
        return (
            self.typeform
            if not isinstance(self.typeform, UnionType)
            else self.common_type
        )


def get_common_type[T](
    typedef: SingleTypeDef[T] | UnionType,
) -> type[T]:
    """Get the common type of a type definition."""
    return get_lowest_common_base(
        typedef_to_typeset(
            typedef,
            remove_null=True,
        )
    )


@dataclass(kw_only=True)
class TypeAware(Generic[T]):
    """Property definition for a model."""

    # Core attributes:

    typeref: TypeRef[TypeAware[T]] | None = None

    @cached_prop
    def resolved_type(self) -> SingleTypeDef[TypeAware[T]]:
        """Resolved type of this prop."""
        if self.typeref is None:
            return cast(SingleTypeDef[TypeAware[T]], type(self))

        return self.typeref.single_typedef

    @cached_prop
    def typeargs(self) -> dict[TypeVar, SingleTypeDef | UnionType]:
        """Type arguments of this prop."""
        return get_typevar_map(self.resolved_type)

    @staticmethod
    def has_type[U: TypeAware](instance: TypeAware, typedef: type[U]) -> TypeGuard[U]:
        """Check if the dataset has the specified type."""
        orig = get_origin(typedef)
        if orig is None or not issubclass(
            get_base_type(instance.resolved_type, bound=TypeAware), orig
        ):
            return False

        own_typevars = get_typevar_map(instance.resolved_type)
        target_typevars = get_typevar_map(typedef)

        for tv, tv_type in target_typevars.items():
            if tv not in own_typevars:
                return False
            if not is_subtype(own_typevars[tv], tv_type):
                return False

        return True
