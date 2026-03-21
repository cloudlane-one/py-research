"""Useful descriptors for instrumenting Python objects."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from typing import Any, Concatenate, Generic, Literal, Self, cast, overload

from typing_extensions import TypeVar


class dualmethod[C, **P, T]:  # noqa: N801
    """Method that can be called on class or instance.

    Note:
        The first argument of the decorated function should be called ``self``
        and be annotated as ``Self | type[Self]``.
    """

    func: Callable[Concatenate[type[C] | C, P], T]
    """The underlying function definition."""

    def __init__(self, func: Callable[Concatenate[type[C] | C, P], T]):
        functools.update_wrapper(self, func)  # pyright: ignore[reportArgumentType]
        self.func = func

    @overload
    def __get__(self, instance: None, owner: None) -> Self: ...

    @overload
    def __get__(self, instance: C | None, owner: type[C]) -> Callable[P, T]: ...

    def __get__(
        self, instance: C | None, owner: type[C] | None = None
    ) -> Self | Callable[P, T]:
        if owner is None:
            return self

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            if instance is not None:  # called via instance
                return self.func(instance, *args, **kwargs)
            elif owner is not None:  # called via class
                return self.func(owner, *args, **kwargs)

        return wrapper


class Read:
    """Readable property."""


class Write(Read):
    """Readable and writable property."""


class Delete(Read):
    """Readable and deletable property."""


class RWD(Write, Delete):
    """Readable, writable, and deletable property."""


type Mode = Literal["instance", "class", "dual"]

C = TypeVar("C", contravariant=True)
G = TypeVar("G", covariant=True)
S = TypeVar("S", contravariant=True, default=G)
M = TypeVar("M", bound=Mode, covariant=True)
W = TypeVar("W", bound=Read, covariant=True)

type AnyGetter[C, T] = Callable[[type[C] | C], T] | Callable[[type[C]], T] | Callable[
    [C], T
]

type AnySetter[C, T] = Callable[[type[C] | C, T], None] | Callable[[C, T], None]

type AnyDeleter[C] = Callable[[type[C] | C], None] | Callable[[C], None]


class prop(Generic[C, G, M, W]):  # noqa: N801
    """Modern, flexible and well-typed property descriptor."""

    mode: M
    """The mode of the property (instance, class, dual)."""

    cached: bool
    """Whether the property value is cached after first access."""

    writable: bool
    """Whether the property is writable."""

    deletable: bool
    """Whether the property is deletable."""

    fget: AnyGetter[C, G] | None
    """Getter function."""

    fset: AnySetter[C, G] | None
    """Setter function."""

    fdel: AnyDeleter[C] | None
    """Deleter function."""

    _name: str | None

    _cls_cache_dict: dict[type, G]
    _instance_cache_dict: dict[int, G]

    @property
    def name(self) -> str:
        """Name of the property."""
        assert self._name is not None, "Property name is not set"
        return self._name

    @overload
    def __init__[C2, T2](
        self: prop[C2, T2, Literal["class"], Read],
        fget: Callable[[type[C2]], T2],
    ) -> None: ...

    @overload
    def __init__[C2, T2](
        self: prop[C2, T2, Literal["dual"], Read],
        fget: Callable[[type[C2] | C2], T2],
    ) -> None: ...

    @overload
    def __init__[C2, T2](
        self: prop[C2, T2, Literal["instance"], Read],
        fget: Callable[[C2], T2],
    ) -> None: ...

    @overload
    def __init__[C2, T2, M2: Mode](
        self: prop[C2, T2, M2, Write],
        *,
        writable: Literal[True],
        deletable: Literal[False] = ...,
        cached: bool = ...,
        mode: M2 = ...,
    ) -> None: ...

    @overload
    def __init__[C2, T2, M2: Mode](
        self: prop[C2, T2, M2, Delete],
        *,
        writable: Literal[False] = ...,
        deletable: Literal[True],
        cached: bool = ...,
        mode: M2 = ...,
    ) -> None: ...

    @overload
    def __init__[C2, T2, M2: Mode](
        self: prop[C2, T2, M2, RWD],
        *,
        writable: Literal[True],
        deletable: Literal[True],
        cached: bool = ...,
        mode: M2 = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        cached: bool = ...,
        mode: M = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    ) -> None: ...

    def __init__(
        self,
        fget: AnyGetter[C, G] | None = None,
        cached: bool = False,
        mode: M | None = None,
        writable: bool = False,
        deletable: bool = False,
    ) -> None:
        self.__isabstractmethod__ = False

        self.fget = None
        self._name = None
        self.fset = None
        self.fdel = None
        self.mode = mode or "instance"  # pyright: ignore[reportAttributeAccessIssue]
        self.cached = cached
        self.writable = writable
        self.deletable = deletable

        self._cls_cache_dict = {}
        self._instance_cache_dict = {}

        if fget is not None:
            self.getter(fget)

    @overload
    def __call__[C2, T2](
        self: prop[Any, Any, Literal["instance"], Any], fget: Callable[[C2], T2]
    ) -> prop[C2, T2, M, W]: ...

    @overload
    def __call__[C2, T2](
        self: prop[Any, Any, Literal["dual"], Any], fget: Callable[[type[C2] | C2], T2]
    ) -> prop[C2, T2, M, W]: ...

    @overload
    def __call__[C2, T2](
        self: prop[Any, Any, Literal["class"], Any], fget: Callable[[type[C2]], T2]
    ) -> prop[C2, T2, M, W]: ...

    @overload
    def __call__[C2, T2](
        self: prop[Any, Any, Any, Any], fget: AnyGetter[C2, T2]
    ) -> prop[C2, T2, Literal["instance"], W]: ...

    def __call__[C2, T2](
        self: prop[Any, Any, Any, Any], fget: AnyGetter[C2, T2]
    ) -> prop[C2, T2, M | Literal["instance"], W]:
        """Define the getter for this property."""
        return self.getter(fget)

    @overload
    def getter[C2, T2](
        self: prop[Any, Any, Literal["instance"], Any], fget: Callable[[C2], T2]
    ) -> prop[C2, T2, M, W]: ...

    @overload
    def getter[C2, T2](
        self: prop[Any, Any, Literal["dual"], Any], fget: Callable[[type[C2] | C2], T2]
    ) -> prop[C2, T2, M, W]: ...

    @overload
    def getter[C2, T2](
        self: prop[Any, Any, Literal["class"], Any], fget: Callable[[type[C2]], T2]
    ) -> prop[C2, T2, M, W]: ...

    @overload
    def getter[C2, T2](
        self: prop[Any, Any, Any, Any], fget: AnyGetter[C2, T2]
    ) -> prop[C2, T2, M, W]: ...

    def getter[C2, T2](self, fget: AnyGetter[C2, T2]) -> prop[C2, T2, M, W]:
        """Define the getter for this property."""
        assert self.fget is None, "Getter already defined"

        self.fget = fget  # type: ignore
        self._name = fget.__name__

        self.__module__ = fget.__module__
        self.__qualname__ = fget.__qualname__

        if sys.version_info >= (3, 13):
            self.__name__ = fget.__name__

        return cast(prop[C2, T2, M, W], self)

    @overload
    def setter(
        self: prop[Any, Any, Literal["instance"], W], fset: Callable[[C, G], None]
    ) -> prop[C, G, M, W]: ...

    @overload
    def setter(
        self: prop[Any, Any, Literal["dual"], W], fset: Callable[[type[C] | C, G], None]
    ) -> prop[C, G, M, W]: ...

    def setter(self: prop[Any, Any, Any, W], fset: AnySetter[C, G]) -> prop[C, G, M, W]:
        """Define the setter for this property."""
        assert self.fset is None, "Setter already defined"

        self.fset = fset
        return cast(prop[C, G, M, W], self)

    @overload
    def deleter(
        self: prop[Any, Any, Literal["instance"], W], fdel: Callable[[C], None]
    ) -> prop[C, G, M, W]: ...

    @overload
    def deleter(
        self: prop[Any, Any, Literal["dual"], W], fdel: Callable[[type[C] | C], None]
    ) -> prop[C, G, M, W]: ...

    def deleter(self: prop[Any, Any, Any, W], fdel: AnyDeleter[C]) -> prop[C, G, M, W]:
        """Define the deleter for this property."""
        assert self.fdel is None, "Deleter already defined"

        self.fdel = fdel
        return cast(prop[C, G, M, W], self)

    def __set_name__[C2](self, owner: type[C2], name: str) -> None:
        self._name = name

        if self.writable:
            assert self.fset is not None, "Setter must be defined if writable is True"

        if self.deletable:
            assert self.fdel is not None, "Deleter must be defined if deletable is True"

    @overload
    def __get__(self, instance: None, owner: None) -> Self: ...

    @overload
    def __get__[C2](
        self: prop[C2, Any, Literal["dual", "class"], Read],
        instance: C | None,
        owner: type[C2],
    ) -> G: ...

    @overload
    def __get__[C2](
        self: prop[C2, Any, Literal["instance"], Read], instance: None, owner: type[C2]
    ) -> prop[C2, G, M, W]: ...

    @overload
    def __get__[C2](
        self: prop[C2, Any, Literal["instance"], Read], instance: C2, owner: type[C2]
    ) -> G: ...

    def __get__[C2](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: prop[C2, Any, Any, Any],
        instance: C2 | None,
        owner: type[C2] | None = None,
    ) -> prop[C2, G, M, W] | G:
        if self.mode in ("instance", "dual") and instance is not None:
            assert self.fget is not None, "no getter"
            fget = cast(Callable[[C2], G], self.fget)
            if self.cached:
                if id(instance) not in self._instance_cache_dict:
                    self._instance_cache_dict[id(instance)] = fget(instance)
                return self._instance_cache_dict[id(instance)]

            return fget(instance)
        elif self.mode in ("dual", "class") and owner is not None:
            assert self.fget is not None, "no getter"
            fget = cast(
                Callable[[type[C2]], G],
                (
                    self.fget.__func__
                    if isinstance(self.fget, classmethod)
                    else self.fget
                ),
            )
            if self.cached:
                if owner not in self._cls_cache_dict:
                    self._cls_cache_dict[owner] = fget(owner)
                return self._cls_cache_dict[owner]

            return fget(owner)

        return self

    def __set__[C2, T2](
        self: prop[C2, T2, Literal["instance", "dual"], Write], instance: C2, value: T2
    ) -> None:
        assert self.mode in ("instance", "dual")

        if self.cached:
            if id(instance) in self._instance_cache_dict:
                del self._instance_cache_dict[id(instance)]
        else:
            assert self.fset is not None, "no setter"

        if self.fset is not None:
            fset = cast(Callable[[C2, T2], None], self.fset)
            fset(instance, value)

    def __delete__[C2](
        self: prop[C2, Any, Literal["instance", "dual"], Any], instance: C2
    ) -> None:
        assert self.mode in ("instance", "dual")

        if self.cached:
            if id(instance) in self._instance_cache_dict:
                del self._instance_cache_dict[id(instance)]

        if self.fdel is not None:
            fdel = cast(Callable[[C2], None], self.fdel)
            fdel(instance)
