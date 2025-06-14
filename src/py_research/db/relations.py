"""Higher-order relational building blocks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from inspect import getmodule
from types import ModuleType, new_class
from typing import TYPE_CHECKING, Any, ClassVar, Generic, cast, get_args, override
from uuid import uuid4

from typing_extensions import TypeVar

from py_research.caching import cached_prop
from py_research.db.records import Attr, Key, Link, LnT, Record, RecT
from py_research.hashing import gen_str_hash
from py_research.reflect.types import get_common_type
from py_research.types import UUID4

from .data import (
    SQL,
    Col,
    Ctx,
    CtxTt,
    Data,
    Expand,
    ExT,
    Interface,
    KeyT,
    R,
    RwxT,
    SxT,
    Tab,
    ValT,
)
from .models import IdxT, ModelMeta, OwnT, Prop, PropT


class DynRecordMeta(ModelMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(self, name: str) -> Attr:
        """Get dynamic attribute by dynamic name."""
        return Attr(alias=name, context=Interface(self))

    def __getattr__(self, name: str) -> Attr:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)
        return Attr(alias=name, context=Interface(self))


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""

    _template = True


x = DynRecord


def dynamic_record_type[T: Record](
    base: type[T] | tuple[type[T], ...],
    name: str,
    props: Iterable[Prop] = [],
    src_module: ModuleType | None = None,
    extra_attrs: dict[str, Any] = {},
) -> type[T]:
    """Create a dynamically defined record type."""
    base = base if isinstance(base, tuple) else (base,)
    return cast(
        type[T],
        new_class(
            name,
            base,
            None,
            lambda ns: ns.update(
                {
                    **{p.name: p for p in props},
                    "__annotations__": {
                        p.name: p.typeref.single_typedef
                        for p in props
                        if p.typeref is not None
                    },
                    "_src_mod": src_module or base[0]._src_mod or getmodule(base[0]),
                    **extra_attrs,
                }
            ),
        ),
    )


@dataclass
class Dyn(
    Prop[PropT, IdxT, RwxT, ExT, SxT, ValT, OwnT, R, *CtxTt],
):
    """Dynamic property reference."""

    ref: Data[ValT, Expand[IdxT], SxT, ExT, RwxT, Interface[OwnT], *CtxTt]
    converter: Callable[[Iterable[ValT]], PropT] | None = None

    def __post_init__(self):  # noqa: D105
        self._data = self.ref

    def _getter(self, instance: OwnT) -> PropT:
        """Get the value of this property."""
        data = instance._data()[self.ref]
        values = data.values()

        if self.converter is not None:
            return self.converter(values)

        if (
            issubclass(data.common_value_type, self.common_prop_type)
            and data.index() is None
        ):
            return cast(PropT, values[0])

        constructor = cast(Callable[[Any], PropT], self.common_prop_type)
        return constructor(values[0] if data.index() is None else values)


class HashRec(Record[str]):
    """Record type with a default hashed primary key."""

    _template = True

    def __post_init__(self) -> None:  # noqa: D105
        setattr(
            self,
            self._primary_key().components[0].name,
            gen_str_hash(
                {a.name: getattr(self, a.name) for a in type(self)._attrs().values()}
            ),
        )


class Entity(Record[UUID4]):
    """Record type with a default UUID4 primary key."""

    _template = True
    _pk: Key[UUID4] = Key(default_factory=lambda: (UUID4(uuid4()),))


TgtT = TypeVar("TgtT", bound="Record", covariant=True, default=Any)


class Edge(Record[str, str], Generic[RecT, TgtT]):
    """Automatically defined relation record type."""

    _template = True

    _from: Link[RecT] = Link()
    _to: Link[RecT] = Link()

    _pk: Key[str, str] = Key(
        components=[
            *(
                a
                for fk_map in _from.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            *(
                a
                for fk_map in _to.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
        ]
    )


class LabelEdge(Record[str, str, KeyT], Generic[RecT, TgtT, KeyT]):
    """Automatically defined relation record type with index substitution."""

    _template = True

    _from: Link[RecT] = Link()
    _to: Link[RecT] = Link()
    _rel_idx: Attr[KeyT] = Attr()

    _pk: Key[str, str, KeyT] = Key(
        components=[
            *(
                a
                for fk_map in _from.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            *(
                a
                for fk_map in _to.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            _rel_idx,
        ]
    )


EdgeT = TypeVar("EdgeT", bound=Edge | LabelEdge, default=Any)


@dataclass(kw_only=True, eq=False)
class Rel(
    Var[LnT, IdxT, SQL, Tab, TgtT, RecT, Ctx[EdgeT, IdxT, Tab]],
    Generic[LnT, EdgeT, IdxT, TgtT, RecT],
):
    """Backlink record set."""

    @property
    @override
    def init_level(self) -> int:
        return 1

    def edge_type(
        self,
    ) -> type[EdgeT]:
        """Return the type of the edge record."""
        return get_common_type(self.typeargs[EdgeT])


class Item(Record[KeyT], Generic[ValT, KeyT, RecT]):
    """Dynamically defined scalar record type."""

    _template = True
    _array: ClassVar[Array]

    _from: Link[RecT] = Link()
    _idx: Attr[KeyT] = Attr()
    _val: Attr[ValT]

    _pk: Key[*tuple[Any, ...], KeyT] = Key(
        components=[
            *(
                a
                for fk_map in _from.join_maps.values()
                for a in fk_map.keys()
                if isinstance(a, Attr)
            ),
            _idx,
        ]
    )


_item_classes: dict[str, type[Item]] = {}


@dataclass(eq=False)
class Array(Var[ValT, IdxT, SQL, Col, ValT, RecT], Generic[ValT, IdxT, RecT]):
    """Set / array of scalar values."""

    @property
    @override
    def init_level(self) -> int:
        return 1

    @cached_prop
    def _key_type(self) -> type:
        idx_type = self.typeargs[KeyT]
        key_tuple = get_args(idx_type)
        return key_tuple[0] if len(key_tuple) == 1 else tuple[*key_tuple]

    @cached_prop
    def item_type(self) -> type[Item[ValT, KeyT, RecT]]:
        """Return the dynamic item record type."""
        assert self.owner is not None
        base_array_fqn = self.fqn

        rec = _item_classes.get(
            base_array_fqn,
            dynamic_record_type(
                Item[self.common_value_type, self._key_type, self.owner],
                f"{self.owner.__name__}.{self.name}",
                src_module=self.owner._src_mod or getmodule(self.owner),
                extra_attrs={"_array": self},
            ),
        )
        _item_classes[base_array_fqn] = rec

        return rec
