"""Tests for py_research.reflect.types utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Generic, Literal, TypeAliasType, TypeVar

import py_research.reflect.types as ftyping


def test_get_subclasses_respects_levels() -> None:
    """Ensure subclass discovery includes nested classes and honors max_level."""

    class Base: ...

    class Child(Base): ...

    class GrandChild(Child): ...

    all_subs = ftyping.get_subclasses(Base)
    limited_subs = ftyping.get_subclasses(Base, max_level=1)

    assert Child in all_subs and GrandChild in all_subs
    assert Child in limited_subs and GrandChild not in limited_subs


def test_is_subtype_with_alias_and_primitives() -> None:
    """Is_subtype handles aliases, annotated types, and simple classes."""
    alias = TypeAliasType(  # pyright: ignore[reportGeneralTypeIssues]
        "AliasInt",  # pyright: ignore[reportGeneralTypeIssues]
        int,
    )

    assert ftyping.is_subtype(int, object)
    assert ftyping.is_subtype(alias, int)
    assert ftyping.is_subtype(Annotated[int, "meta"], int)


def test_has_type_accepts_union_members() -> None:
    """has_type accepts values matching any member of a union hint."""
    union_hint = int | str

    assert ftyping.has_type(5, union_hint)
    assert not ftyping.has_type(3.14, union_hint)


def test_get_lowest_common_base_and_empty_iterable() -> None:
    """Lowest common base falls back to object and respects inheritance depth."""

    class Parent: ...

    class Child(Parent): ...

    assert ftyping.get_lowest_common_base([Child, Parent]) is Parent
    assert ftyping.get_lowest_common_base([]) is object


def test_extract_nullable_type_handles_optional_and_plain_union() -> None:
    """Extract_nullable_type returns non-None member or None when absent."""
    optional_int = int | None
    no_none_union = str | bytes

    assert ftyping.extract_nullable_type(optional_int) is int
    assert ftyping.extract_nullable_type(
        no_none_union
    ) is ftyping.get_lowest_common_base({str, bytes})


def test_get_inheritance_distance_direction_and_unrelated() -> None:
    """Inheritance distance is signed, and unrelated classes yield None."""

    class Parent: ...

    class Child(Parent): ...

    class Unrelated: ...

    assert ftyping.get_inheritance_distance(Child, Parent) == 1
    assert ftyping.get_inheritance_distance(Parent, Child) == -1
    assert ftyping.get_inheritance_distance(Parent, Unrelated) is None


def test_typedef_to_typeset_for_union_and_alias() -> None:
    """Typedef_to_typeset resolves unions, strips None, and handles type aliases."""
    alias = TypeAliasType(  # pyright: ignore[reportGeneralTypeIssues]
        "AliasList", list[int]  # pyright: ignore[reportGeneralTypeIssues]
    )
    types_no_null = ftyping.typedef_to_typeset(int | None, remove_null=True)
    alias_types = ftyping.typedef_to_typeset(alias)

    assert types_no_null == {int}
    alias_cls = next(iter(alias_types))
    assert isinstance(alias_cls, type)
    assert issubclass(alias_cls, list)


def test_get_typevar_map_on_generic_and_union() -> None:
    """get_typevar_map maps type variables across generics and unions."""
    T_co = TypeVar("T_co")

    class Box(Generic[T_co]): ...

    mapping = ftyping.get_typevar_map(Box[int])
    union_mapping = ftyping.get_typevar_map(int | str)

    assert mapping[T_co].typedef is int
    assert union_mapping == {}


def test_set_typeargs_uses_existing_args() -> None:
    """set_typeargs currently returns the original generic with its existing args."""
    result = ftyping.set_typeargs(list[int], (str,))

    assert result == list[int]


def test_get_typeargs_reads_orig_class() -> None:
    """get_typeargs reads __orig_class__ when present on an instance."""
    S = TypeVar("S")

    class Holder(Generic[S]): ...

    inst = Holder[int]()
    setattr(inst, "__orig_class__", Holder[int])

    assert ftyping.get_typeargs(inst) == (int,)


def test_typeref_resolves_types_and_validation() -> None:
    """TypeRef resolves bounds, annotations, literals, and performs validation."""
    bound_var = TypeVar(  # pyright: ignore[reportGeneralTypeIssues]
        "Bounded", bound=int
    )  # pyright: ignore[reportGeneralTypeIssues]
    annotated = Annotated[int, "meta"]

    ref_from_var = ftyping.TypeRef(bound_var)
    ref_list = ftyping.TypeRef(list[int])
    ref_literal = ftyping.TypeRef(Literal["a", "b"])
    ref_annotated = ftyping.TypeRef(annotated)  # pyright: ignore[reportArgumentType]

    assert ref_from_var.typedef is int
    assert ref_list.typeset == {list}
    assert ref_list.common_type is list
    assert ref_literal.literal_values == ("a", "b")
    assert ref_annotated.annotation == "meta"
    assert isinstance(ref_list.args, dict)
    assert ref_list.validate([1, 2])
    assert not ref_list.validate(["x"])


def test_typeawareclass_preserves_origin_and_typeargs() -> None:
    """TypeAwareClass tracks origin, type parameters, and caches subclasses."""
    U = TypeVar("U")

    class Wrapper(ftyping.TypeAwareClass, Generic[U]): ...

    specialized_once = Wrapper[int]
    specialized_twice = Wrapper[int]

    assert specialized_once is specialized_twice
    assert specialized_once.__origin__ is Wrapper
    assert specialized_once.__args__ == (int,)
    assert isinstance(specialized_once.type_params, tuple)
    assert specialized_once.typeargs[U].typedef is int


def test_is_frozen_dataclass_and_is_immutable() -> None:
    """Frozen dataclasses and nested immutables are detected correctly."""

    @dataclass(frozen=True)
    class Frozen:
        value: int

    @dataclass
    class Mutable:
        value: int

    assert ftyping.is_frozen_dataclass(Frozen(1))
    assert not ftyping.is_frozen_dataclass(Mutable(1))
    assert ftyping.is_immutable((1, (2, 3)))
    assert not ftyping.is_immutable(([],))


def test_get_nondefault_methods_filters_object_methods() -> None:
    """get_nondefault_methods returns declared methods but skips object defaults."""

    class Demo:
        def foo(self) -> str:
            return "bar"

    methods = ftyping.get_nondefault_methods(Demo)

    assert "foo" in methods
    assert "__repr__" not in methods


def test_get_own_attrs_includes_optional_bases() -> None:
    """get_own_attrs can include attributes from supplied base classes."""

    class Extra:
        extra_attr = 1

    class Parent:
        parent_attr = 2

    class Child(Parent, Extra):
        child_attr = 3

    attrs_without_includes = ftyping.get_own_attrs(Child)
    attrs_with_includes = ftyping.get_own_attrs(Child, include_bases=(Extra,))

    assert "child_attr" in attrs_without_includes
    assert "parent_attr" not in attrs_without_includes
    assert "extra_attr" in attrs_with_includes
