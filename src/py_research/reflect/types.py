"""Reflection utilities for types."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property, reduce
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

from py_research.hashing import gen_str_hash
from py_research.reflect.runtime import get_subclasses
from py_research.types import GenericAlias, Not, SingleTypeDef, T


def is_subtype(type_: SingleTypeDef | UnionType, supertype: T) -> TypeGuard[T]:
    """Check if object is of given type hint."""
    if not isinstance(type_, TypeAliasType):
        type_args = get_args(type_)
        supertype_args = get_args(supertype)

        if len(type_args) == 0 and len(supertype_args) == 0:
            # Special case since beartype can't handle empty typevartuples as args.
            typeset = typedef_to_typeset(type_)
            supertype_tuple = tuple(
                typedef_to_typeset(cast(SingleTypeDef | UnionType, supertype))
            )
            return all(issubclass(t, supertype_tuple) for t in typeset)

        return is_subhint(type_, supertype)

    return is_subhint(type_.__value__, supertype)


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


def typedef_to_typeset(
    typedef: SingleTypeDef | UnionType | None,
    typevar_map: dict[TypeVar, TypeRef] | None = None,
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
            TypeRef(
                t.__value__, var_map=typevar_map or {}, ctx_module=ctx_module
            ).typeform
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


def get_typevar_map(
    c: SingleTypeDef | UnionType | tuple[type, ...],
    subs: Mapping[TypeVar, TypeRef] | None = None,
    ctx_module: ModuleType | None = None,
) -> dict[TypeVar, TypeRef]:
    """Return a mapping of type variables to their actual types."""
    ctx_module = ctx_module or getmodule(c)
    subs = subs or {}

    typevar_map: dict[TypeVar, TypeRef] = {}

    orig = get_origin(c) or c
    args = get_args(c)

    if isinstance(c, UnionType) or orig is Union:
        # Resolve typevar map of each union arg individually and
        # union the resulting types per typevar.
        base_typevar_items = chain(
            *(get_typevar_map(arg, subs=subs).items() for arg in args)
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
            (get_typevar_map(base, subs=subs) for base in c),
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
            k: TypeRef(
                v,
                var_map=subs,
                ctx_module=ctx_module,
            )
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
            if len(bases) > 0:
                base_typevar_map = get_typevar_map(
                    bases if len(bases) > 1 else bases[0],
                    subs=subs_typevar_map,
                )
        elif isinstance(orig, TypeAliasType):
            base_typevar_map = get_typevar_map(orig.__value__, subs=subs_typevar_map)

        # Merge with base typevar map.
        typevar_map = subs_typevar_map | base_typevar_map

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
    typevar_map = get_typevar_map(typedef)

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
def _map_str_annotation_base[U](anno: str, base: type[U]) -> type[U] | None:
    """Map property type name to class."""
    name_map = {cls.__name__: cls for cls in get_subclasses(base) if cls is not base}
    matches = [name_map[n] for n in name_map if anno.startswith(n + "[")]
    return matches[0] if len(matches) == 1 else None


@dataclass
class TypeRef[T]:
    """Reference to a type."""

    hint: SingleTypeDef[T] | UnionType | ForwardRef | str | TypeVar = cast(
        type[T], object
    )
    var_map: Mapping[TypeVar, TypeRef] = field(default_factory=dict)
    ctx_module: ModuleType | None = None
    ctx: dict[str, Any] = field(default_factory=dict)
    unresolved_typevars: Literal["default", "raise"] = "default"

    @cached_property
    def typeform(self) -> SingleTypeDef[T] | UnionType:
        """Return the type format."""

        if isinstance(self.hint, str):
            hint = eval(
                self.hint,
                {
                    **globals(),
                    **(vars(self.ctx_module) if self.ctx_module else {}),
                    **self.ctx,
                },
            )
            return TypeRef(
                hint,
                var_map=self.var_map,
                ctx_module=self.ctx_module,
                ctx=self.ctx,
                unresolved_typevars=self.unresolved_typevars,
            ).typeform

        if isinstance(self.hint, TypeVar):
            type_res = self.var_map.get(self.hint, Not.defined)

            if isinstance(type_res, Not) or type_res.hint is self.hint:
                if self.unresolved_typevars == "default":
                    v = type_res if isinstance(type_res, TypeVar) else self.hint

                    return cast(
                        SingleTypeDef[T] | UnionType,
                        (
                            TypeRef(v.__default__, ctx_module=getmodule(v)).typeform
                            if hasattr(v, "__default__")
                            and v.has_default()
                            and v.__default__ is not Any
                            else (v.__bound__ if v.__bound__ is not None else object)
                        ),
                    )
                else:
                    raise TypeError(
                        f"Type variable `{self.hint}` not bound for typehint `{self.hint}`."
                    )

            return type_res.typeform

        if isinstance(self.hint, ForwardRef):
            evaluated = self.hint._evaluate(
                {
                    **globals(),
                    **(vars(self.ctx_module) if self.ctx_module else {}),
                    **self.ctx,
                },
                None,
                recursive_guard=frozenset(),
            )
            assert evaluated is not None
            return evaluated

        orig = get_origin(self.hint)
        args = get_args(self.hint)
        if orig is not None and len(args) > 0:
            return orig[
                *(
                    TypeRef(
                        arg,
                        var_map=self.var_map,
                        ctx_module=self.ctx_module,
                        ctx=self.ctx,
                        unresolved_typevars=self.unresolved_typevars,
                    ).typeform
                    for arg in args
                )
            ]

        return self.hint

    @cached_property
    def typeset(self) -> set[type[T]]:
        """Target types of this prop (>1 in case of union typeform)."""
        return typedef_to_typeset(self.typeform)

    @cached_property
    def common_type(self) -> type[T]:
        """Common base type of the target types."""
        return get_lowest_common_base(
            typedef_to_typeset(
                self.typeform,
                remove_null=True,
            )
        )

    def base_type(self, bound: type[T] | None = None) -> type[T] | None:
        """Resolve the prop typehint."""
        bound = bound if bound is not None else cast(type[T], object)

        if has_type(self.hint, SingleTypeDef):
            base = get_origin(self.hint)
            if base is None or not issubclass(base, bound):
                return None

            return base
        else:
            assert isinstance(self.hint, str)
            return _map_str_annotation_base(self.hint, bound)

    @property
    def single_typedef(self) -> SingleTypeDef[T]:
        """Return the type format."""
        return (
            self.typeform
            if not isinstance(self.typeform, UnionType)
            else self.common_type
        )

    @cached_property
    def args(self) -> dict[TypeVar, TypeRef]:
        """Type arguments of this prop."""
        return get_typevar_map(self.single_typedef)


@dataclass(kw_only=True)
class TypeAware(Generic[T]):
    """Property definition for a model."""

    # Core attributes:

    typeref: TypeRef[TypeAware[T]] = field(default_factory=TypeRef)

    def __post_init__(self) -> None:
        if self.typeref.hint is object:
            self.typeref.hint = type(self)

    # @staticmethod
    # def has_type[U: TypeAware](instance: TypeAware, typedef: type[U]) -> TypeGuard[U]:
    #     """Check if the dataset has the specified type."""
    #     orig = get_origin(typedef)
    #     base = get_base_type(instance.resolved_type, bound=TypeAware)
    #     if orig is None or base is None or not issubclass(base, orig):
    #         return False

    #     own_typevars = get_typevar_map(instance.resolved_type)
    #     target_typevars = get_typevar_map(typedef)

    #     for tv, tv_type in target_typevars.items():
    #         if tv not in own_typevars:
    #             return False
    #         if not is_subtype(own_typevars[tv], tv_type):
    #             return False

    #     return True
