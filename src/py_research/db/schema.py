"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property, reduce
from inspect import getmodule
from secrets import token_hex
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)
from uuid import UUID, uuid4

import sqlalchemy as sqla
import sqlalchemy.orm as orm
import sqlalchemy.sql.type_api as sqla_types
from bidict import bidict
from sqlalchemy_utils import UUIDType
from typing_extensions import TypeVar, TypeVarTuple

from py_research.hashing import gen_int_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import has_type


class BaseIdx:
    """Singleton to mark dataset index as default index."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


class SingleIdx:
    """Singleton to mark dataset index as a single value."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


Key = TypeVar("Key", bound=Hashable)
Key_def = TypeVar("Key_def", contravariant=True, bound=Hashable, default=Any)

Val = TypeVar("Val")
Val2 = TypeVar("Val2")
Val3 = TypeVar("Val3")
Val_cov = TypeVar("Val_cov", covariant=True)
Val_def = TypeVar("Val_def", covariant=True, default=Any)

PVal = TypeVar("PVal", bound="Prop")

Rec = TypeVar("Rec", bound="Record")
Rec2 = TypeVar("Rec2", bound="Record")
Rec3 = TypeVar("Rec3", bound="Record")
Rec_cov = TypeVar("Rec_cov", covariant=True, bound="Record")
Rec2_cov = TypeVar("Rec2_cov", covariant=True, bound="Record")
Rec_def = TypeVar("Rec_def", covariant=True, bound="Record", default="Record")

type RecordValue[Rec: Record] = Rec | Iterable[Rec] | Mapping[Any, Rec]
Recs_cov = TypeVar("Recs_cov", bound=RecordValue, covariant=True)

Sch = TypeVar("Sch", bound="Schema")
DBS = TypeVar("DBS", bound="Record | Schema")

RelTup = TypeVarTuple("RelTup")


class RO:
    """Read-only flag."""


class RW(RO):
    """Read-write flag."""


RWT = TypeVar("RWT", RW, RO, default=RW, covariant=True)


class Unfetched:
    """Singleton to mark unfetched values."""


def _extract_record_type(hint: Any) -> type[Record]:
    origin: type = get_origin(hint) or hint

    if issubclass(origin, Record):
        return origin
    if issubclass(origin, Mapping):
        return get_args(hint)[1]
    if issubclass(origin, Iterable):
        return get_args(hint)[0]

    raise ValueError("Invalid record value.")


@dataclass(frozen=True)
class TypeDef(Generic[PVal]):
    """Reference to a type."""

    hint: str | type[PVal] | None = None
    ctx: ModuleType | None = None

    def prop_type(self) -> type[Prop]:
        """Resolve the property type reference."""
        hint = self.hint or Prop

        if isinstance(hint, type):
            base = get_origin(hint)
            assert base is not None and issubclass(base, Prop)
            return base
        else:
            return Attr if "Attr" in hint else Rel if "Rel" in hint else Prop

    def value_type(self) -> type:
        """Resolve the value type reference."""
        hint = self.hint or Prop

        generic = (
            cast(type[PVal], eval(hint, vars(self.ctx) if self.ctx else None))
            if isinstance(hint, str)
            else hint
        )

        assert issubclass(get_origin(generic) or generic, Prop)

        args = get_args(generic)
        return args[0] if len(args) > 0 else object


@dataclass(eq=False)
class Prop(Generic[Val_cov, RWT]):
    """Reference a property of a record."""

    alias: str | None = None
    default: Val_cov | None = None
    default_factory: Callable[[], Val_cov] | None = None

    getter: Callable[[Record], Val_cov] | None = None
    setter: Callable[[Record, Val_cov], None] | None = None

    _name: str | None = None

    @property
    def name(self) -> str:
        """Property name."""
        if self.alias is not None:
            return self.alias
        elif self._name is not None:
            return self._name
        else:
            raise ValueError("Property name not set.")

    def __set_name__(self, _, name: str) -> None:  # noqa: D105
        self._name = name

    def __hash__(self) -> int:
        """Hash the Prop."""
        return gen_int_hash(self)

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Val_cov: ...

    @overload
    def __get__(self, instance: None, owner: type[Rec]) -> PropRef[Rec, Val_cov]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[Rec] | None
    ) -> Val_cov | PropRef[Rec, Val_cov] | Self:
        if isinstance(instance, Record):
            if self.getter is not None:
                value = self.getter(instance)
            else:
                value = instance._values[self.name]

            if isinstance(value, Unfetched) and instance._fetcher is not None:
                try:
                    value = instance._fetcher(self.name)
                except KeyError:
                    pass

            if isinstance(value, Unfetched):
                if self.default_factory is not None:
                    value = self.default_factory()
                elif self.default is not None:
                    value = self.default

            if isinstance(value, Unfetched):
                raise ValueError("Property value could not fetched.")

            instance._values[self.name] = value
            return value
        elif owner is not None and issubclass(owner, Record):
            t = (
                AttrRef
                if isinstance(self, Attr)
                else RelRef if isinstance(self, Rel) else PropRef
            )
            return t(
                **{
                    **{f.name: getattr(self, f.name) for f in fields(self)},
                    **dict(
                        record_type=cast(type[Rec], owner),
                        prop_type=owner._prop_defs[self.name],
                    ),
                }
            )
        return self

    def __set__(self: Prop[Val, RW], instance: Record, value: Val | Unfetched) -> None:
        """Set the value of the property."""
        if self.setter is not None and not isinstance(value, Unfetched):
            self.setter(instance, value)
        instance._values[self.name] = value


@dataclass(eq=False)
class Attr(Prop[Val_cov, RWT]):
    """Define an attribute of a record."""

    index: bool = False
    primary_key: bool = False

    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = None

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Val_cov: ...

    @overload
    def __get__(self, instance: None, owner: type[Rec]) -> AttrRef[Rec, Val_cov]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[Rec] | None
    ) -> Val_cov | PropRef[Rec, Val_cov] | Self:
        return super().__get__(instance, owner)


@dataclass(eq=False)
class Rel(Prop[Recs_cov, RWT], Generic[Recs_cov, RWT, Rec_def]):
    """Define a relation to another record."""

    index: bool = False
    primary_key: bool = False

    on: (
        AttrRef
        | Iterable[AttrRef]
        | dict[AttrRef, AttrRef[Rec_def, Any]]
        | RelRef[Rec_def, Any, Any]
        | tuple[RelRef, RelRef[Rec_def, Any, Any]]
        | type[Rec_def]
        | type[Record]
        | None
    ) = None
    order_by: Mapping[AttrRef, int] | None = None
    map_by: AttrRef | None = None
    collection: Callable[[Any], Recs_cov] | None = None

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Recs_cov: ...

    @overload
    def __get__(
        self: Rel[Rec | Iterable[Rec] | Mapping[Any, Rec]],
        instance: None,
        owner: type[Rec2],
    ) -> RelRef[Rec2, Recs_cov, Rec]: ...

    @overload
    def __get__(
        self, instance: None, owner: type[Rec2]
    ) -> RelRef[Rec2, Recs_cov, Any]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105 # type: ignore
        self, instance: object | None, owner: type | type[Rec2] | None
    ) -> Recs_cov | RelRef[Rec2, Recs_cov, Any] | Self:
        recs = super().__get__(instance, owner)

        if self.collection is not None:
            recs = self.collection(recs)

        return recs


@dataclass(kw_only=True, eq=False)
class PropRef(Prop[Val_cov], Generic[Rec_cov, Val_cov]):
    """Reference a property of a record."""

    record_type: type[Rec_cov]
    prop_type: TypeDef[Prop[Val_cov]]

    @staticmethod
    def get_tag(rec_type: type[Record]) -> RelRef | None:
        """Retrieve relation-tag of a record type, if any."""
        try:
            rel = getattr(rec_type, "_rel")
            return rel if isinstance(rel, RelRef) else None
        except AttributeError:
            return None

    @cached_property
    def parent(self) -> RelRef | None:
        """Parent relation of this Rel."""
        return self.get_tag(self.record_type)

    @cached_property
    def path(self) -> list[RelRef]:
        """Path from base record type to this Rel."""
        return [
            *(self.parent.path if self.parent is not None else []),
            *([self] if isinstance(self, RelRef) else []),
        ]

    def __rrshift__(self, left: type[Record] | RelRef) -> Self:
        """Append a prop to the relation path."""
        current_root: RelRef[Record, Record, Any] = self.path[0]
        new_root = (
            left if isinstance(left, RelRef) else left.rel(current_root.record_type)
        )

        prefixed_rel = reduce(
            lambda r1, r2: RelRef(**asdict(r2), record_type=r1.a),  # type: ignore
            self.path,
            new_root,
        )

        return cast(
            Self,
            (
                prefixed_rel
                if isinstance(self, RelRef)
                else getattr(prefixed_rel.a, self.name)
            ),
        )


@dataclass(kw_only=True, eq=False)
class AttrRef(  # type: ignore
    Attr[Val],
    PropRef[Rec_cov, Val],
    sqla.ColumnClause[Val],
    Generic[Rec_cov, Val],
):
    """Reference an attribute of a record."""

    def __post_init__(self) -> None:  # noqa: D105
        # Initialize fields required by SQLAlchemy superclass.
        self.table = None
        self.is_literal = False

    # Replace fields of SQLAlchemy superclass with properties:

    @property
    def name(self) -> str:
        """Column key."""
        assert Prop.name.fget is not None
        return Prop.name.fget(self)

    @property
    def key(self) -> str:
        """Column key."""
        return self.name

    @cached_property
    def sql_type(self) -> sqla_types.TypeEngine:
        """Column key."""
        return sqla_types.to_instance(self._value_type)  # type: ignore

    def all(self) -> sqla.CollectionAggregate[bool]:
        """Return a SQL ALL expression for this attribute."""
        return sqla.all_(self)

    def any(self) -> sqla.CollectionAggregate[bool]:
        """Return a SQL ANY expression for this attribute."""
        return sqla.any_(self)


@dataclass(kw_only=True, eq=False)
class RelRef(
    Rel[Recs_cov, RW, Rec2_cov],
    PropRef[Rec_cov, Recs_cov],
    Generic[Rec_cov, Recs_cov, Rec2_cov],
):
    """Reference a relation to another record."""

    @property
    def target_type(self) -> type[Rec2_cov]:
        """Dynamic target type of the relation."""
        if self.prop_type is not None:
            value_type = self.prop_type.value_type()
            if issubclass(value_type, Record):
                return cast(type[Rec2_cov], _extract_record_type(value_type))

        match self.on:
            case dict():
                self.on
                return next(iter(self.on.values())).record_type
            case type():
                return cast(type[Rec2_cov], self.on)
            case RelRef():
                if issubclass(self.record_type, self.on.target_type):
                    return self.on.record_type
                else:
                    return self.on.target_type
            case tuple():
                via_1 = self.on[1]
                assert isinstance(via_1, RelRef)
                return via_1.target_type
            case _:
                raise ValueError("Invalid relation definition.")

    @cached_property
    def fk_record_type(self) -> type[Record]:
        """Record type of the foreign key."""
        match self.on:
            case type():
                return self.on
            case RelRef():
                return self.on.record_type
            case tuple():
                return self.on[0].record_type
            case dict() | AttrRef() | Iterable() | None:
                return self.record_type

    @cached_property
    def fk_map(self) -> bidict[AttrRef, AttrRef]:
        """Map source foreign keys to target attrs."""
        target = self.target_type

        match self.on:
            case type() | RelRef() | tuple():
                return bidict()
            case dict():
                return bidict(
                    {
                        AttrRef(
                            _name=fk.name,
                            record_type=self.record_type,
                            prop_type=fk.prop_type,
                        ): pk
                        for fk, pk in self.on.items()
                    }
                )
            case AttrRef() | Iterable():
                attrs = self.on if isinstance(self.on, Iterable) else [self.on]
                source_attrs = [
                    AttrRef(
                        _name=attr.name,
                        record_type=self.record_type,
                        prop_type=attr.prop_type,
                    )
                    for attr in attrs
                ]
                target_attrs = target._primary_keys.values()
                fk_map = dict(zip(source_attrs, target_attrs))

                assert all(
                    issubclass(
                        self.record_type._static_props[
                            fk_attr.name
                        ].prop_type.value_type(),
                        pk_attr.prop_type.value_type(),
                    )
                    for fk_attr, pk_attr in fk_map.items()
                    if pk_attr.prop_type is not None
                ), "Foreign key value types must match primary key value types."

                return bidict(fk_map)
            case None:
                return bidict(
                    {
                        AttrRef(
                            _name=f"{self._name}_{target_attr.name}",
                            record_type=self.record_type,
                            prop_type=target_attr.prop_type,
                        ): target_attr
                        for target_attr in target._primary_keys.values()
                    }
                )

    @cached_property
    def inter_joins(
        self,
    ) -> dict[type[Record], list[Mapping[AttrRef, AttrRef]]]:
        """Intermediate joins required by this rel."""
        match self.on:
            case RelRef():
                # Relation is defined via other, backlinking relation
                other_rel = self.on
                assert isinstance(
                    other_rel, RelRef
                ), "Back-reference must be an explicit relation"

                if issubclass(other_rel.target_type, self.record_type):
                    # Supplied record type object is a backlinking relation
                    return {}
                else:
                    # Supplied record type object is a forward relation
                    # on a relation table
                    back_rels = [
                        rel
                        for rel in other_rel.record_type._rels.values()
                        if issubclass(rel.target_type, self.record_type)
                        and len(rel.fk_map) > 0
                    ]

                    return {
                        other_rel.record_type: [
                            back_rel.fk_map.inverse for back_rel in back_rels
                        ]
                    }
            case type():
                if issubclass(self.on, self.target_type):
                    # Relation is defined via all direct backlinks of given record type.
                    return {}

                # Relation is defined via relation table
                back_rels = [
                    rel
                    for rel in self.on._rels.values()
                    if issubclass(rel.target_type, self.record_type)
                    and len(rel.fk_map) > 0
                ]

                return {self.on: [back_rel.fk_map.inverse for back_rel in back_rels]}
            case tuple() if has_type(self.on, tuple[RelRef, RelRef]):
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                back, _ = self.on
                assert len(back.fk_map) > 0, "Back relation must be direct."
                return {back.record_type: [back.fk_map.inverse]}
            case _:
                # Relation is defined via foreign key attributes
                return {}

    @cached_property
    def joins(self) -> list[Mapping[AttrRef, AttrRef]]:
        """Mappings of column keys to join the target on."""
        match self.on:
            case RelRef():
                # Relation is defined via other relation or relation table
                other_rel = self.on
                assert (
                    len(other_rel.fk_map) > 0
                ), "Backref or forward-ref on relation table must be a direct relation"
                return [
                    (
                        other_rel.fk_map.inverse
                        if issubclass(other_rel.target_type, self.record_type)
                        else other_rel.fk_map
                    )
                ]

            case type():
                if issubclass(self.on, self.target_type):
                    # Relation is defined via all direct backlinks of given record type.
                    back_rels = [
                        rel
                        for rel in self.on._rels.values()
                        if issubclass(rel.target_type, self.record_type)
                        and len(rel.fk_map) > 0
                    ]
                    assert len(back_rels) > 0, "No direct backlinks found."
                    return [back_rel.fk_map.inverse for back_rel in back_rels]

                # Relation is defined via relation table
                fwd_rels = [
                    rel
                    for rel in self.on._rels.values()
                    if issubclass(rel.target_type, self.record_type)
                    and len(rel.fk_map) > 0
                ]
                assert (
                    len(fwd_rels) > 0
                ), "No direct forward rels found on relation table."
                return [fwd_rel.fk_map for fwd_rel in fwd_rels]

            case tuple():
                assert has_type(self.on, tuple[RelRef, RelRef])
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                _, fwd = self.on
                assert len(fwd.fk_map) > 0, "Forward relation must be direct."
                return [fwd.fk_map]

            case _:
                # Relation is defined via foreign key attributes
                return [self.fk_map]

    @cached_property
    def a(self) -> type[Rec2_cov]:
        """Reference props of the target record type."""
        return cast(
            type[Rec2_cov], type(token_hex(5), (self.target_type,), {"_rel": self})
        )

    @property
    def path_str(self) -> str:
        """String representation of the relation path."""
        prefix = (
            self.record_type.__name__ if len(self.path) == 0 else self.path[-1].path_str
        )
        return f"{prefix}.{self.name}"

    def get_subdag(
        self,
        backlink_records: set[type[Record]] | None = None,
        _traversed: set[RelRef] | None = None,
    ) -> set[RelRef]:
        """Find all paths to the target record type."""
        backlink_records = backlink_records or set()
        _traversed = _traversed or set()

        # Get relations of the target type as next relations
        next_rels = set(self.record_type._rels.values())

        for backlink_record in backlink_records:
            next_rels |= backlink_record._backrels_to_rels(self.record_type)

        # Filter out already traversed relations
        next_rels = {rel for rel in next_rels if rel not in _traversed}

        # Add next relations to traversed set
        _traversed |= next_rels

        # Prefix next relations with current relation
        prefixed_rels = {self >> rel for rel in next_rels}

        # Return next relations + recurse
        return prefixed_rels | {
            rel
            for next_rel in next_rels
            for rel in next_rel.get_subdag(backlink_records, _traversed)
        }

    def __add__(self, other: RelRef[Any, Any, Rec3]) -> RelTree[Rec2_cov, Rec3]:
        """Add another prop to the merge."""
        return RelTree({self}) + other

    def __hash__(self) -> int:
        """Hash the rel."""
        return gen_int_hash(self)


type RelDict = dict[RelRef, RelDict]


@dataclass
class RelTree(Generic[*RelTup]):
    """Tree of relations starting from the same root."""

    rels: Iterable[RelRef] = field(default_factory=set)

    def __post_init__(self) -> None:  # noqa: D105
        assert all(
            rel.path[0].record_type == self.root for rel in self.rels
        ), "Relations in set must all start from same root."
        self.targets = [rel.target_type for rel in self.rels]

    @cached_property
    def root(self) -> type[Record]:
        """Root record type of the set."""
        return list(self.rels)[-1].path[0].record_type

    @cached_property
    def dict(self) -> RelDict:
        """Tree representation of the relation set."""
        tree: RelDict = {}

        for rel in self.rels:
            subtree = tree
            for ref in rel.path:
                if ref not in subtree:
                    subtree[ref] = {}
                subtree = subtree[ref]

        return tree

    def __rrshift__(self, prefix: type[Record] | RelRef) -> Self:
        """Prefix all relations in the set with given relation."""
        rels = {prefix >> rel for rel in self.rels}
        return cast(Self, RelTree(rels))

    def __add__(self, other: RelRef[Any, Any, Rec] | RelTree) -> RelTree[*RelTup, Rec]:
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    def __or__(
        self: RelTree[Rec], other: RelRef[Any, Any, Rec2] | RelTree[Rec2]
    ) -> RelTree[Rec | Rec2]:
        """Add more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])


type DirectLink[Rec: Record] = (
    AttrRef | Iterable[AttrRef] | dict[AttrRef, AttrRef[Rec, Any]]
)

type BackLink[Rec: Record] = (RelRef[Rec, Any, Any] | type[Rec])

type BiLink[Rec: Record, Rec2: Record] = (
    RelRef[Rec2, Any, Rec]
    | tuple[RelRef[Rec2, Any, Any], RelRef[Rec2, Any, Rec]]
    | type[Rec2]
)


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: bool = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Attr[Val]: ...


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: bool | Literal["fk"] = ...,
    primary_key: bool | Literal["fk"] = ...,
    collection: None = ...,
    link_on: DirectLink[Rec],
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Rel[Rec]: ...


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal["fk"],
    primary_key: bool | Literal["fk"] = ...,
    collection: None = ...,
    link_on: DirectLink[Rec] | None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Rel[Rec]: ...


@overload
def prop(
    *,
    default: Val | Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: bool | Literal["fk"] = ...,
    primary_key: Literal["fk"],
    collection: None = ...,
    link_on: DirectLink[Rec] | None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Rel[Rec]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: None = ...,
    sql_getter: None = ...,
) -> Prop[Val, RO]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: Callable[[Record, Val], None],
    sql_getter: None = ...,
) -> Prop[Val, RW]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: None = ...,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement],
) -> Attr[Val, RO]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: bool = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: Callable[[Record, Val], None],
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement],
) -> Attr[Val, RW]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[True] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: None = ...,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = ...,
) -> Attr[Val, RO]: ...


@overload
def prop(
    *,
    default: Val | None = ...,  # type: ignore
    default_factory: Callable[[], Val | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[True] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: None = ...,
    link_via: None = ...,
    order_by: None = ...,
    map_by: None = ...,
    getter: Callable[[Record], Val],
    setter: Callable[[Record, Val], None],
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = ...,
) -> Attr[Val, RW]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,  # type: ignore
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,
    order_by: Mapping[AttrRef[Rec | Rec2, Any], int] | None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Rel[list[Rec]]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: Callable[[Iterable[Rec]], Recs_cov],
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,
    order_by: Mapping[AttrRef[Rec | Rec2, Any], int] | None = ...,
    map_by: None = ...,
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Rel[Recs_cov]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: None = ...,
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,
    order_by: None = ...,
    map_by: AttrRef[Rec | Rec2, Val3],
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Rel[dict[Val3, Rec]]: ...


@overload
def prop(
    *,
    default: Rec | None = ...,
    default_factory: Callable[[], Rec | Val2] | None = ...,
    alias: str | None = ...,
    index: Literal[False] = ...,
    primary_key: Literal[False] = ...,
    collection: Callable[[Mapping[Val3, Rec]], Recs_cov],
    link_on: None = ...,
    link_from: BackLink[Rec] | None = ...,
    link_via: BiLink[Rec, Rec2] | None = ...,
    order_by: None = ...,
    map_by: AttrRef[Rec | Rec2, Val3],
    getter: None = ...,
    setter: None = ...,
    sql_getter: None = ...,
) -> Rel[Recs_cov]: ...


def prop(
    *,
    default: Val | Rec | None = None,
    default_factory: Callable[[], Val | Rec | Val2] | None = None,
    alias: str | None = None,
    index: bool | Literal["fk"] = False,
    primary_key: bool | Literal["fk"] = False,
    collection: Callable[[Val2], Any] | None = None,
    link_on: DirectLink[Rec] | None = None,
    link_from: BackLink[Rec] | None = None,
    link_via: BiLink[Rec, Rec2] | None = None,
    order_by: Mapping[AttrRef[Rec | Rec2, Any], int] | None = None,
    map_by: AttrRef[Rec | Rec2, Val] | None = None,
    getter: Callable[[Record], Any] | None = None,
    setter: Callable[[Record, Any], None] | None = None,
    sql_getter: Callable[[sqla.FromClause], sqla.ColumnElement] | None = None,
) -> Prop[Any, Any]:
    """Define a backlinking relation to another record."""
    if any(a is not None for a in (link_on, link_from, link_via)) or any(
        a == "fk" for a in (index, primary_key)
    ):
        return Rel(
            default=default,
            default_factory=default_factory,
            alias=alias,
            index=index == "fk",
            primary_key=primary_key == "fk",
            on=(
                link_on
                if link_on is not None
                else link_from if link_from is not None else link_via
            ),
            order_by=order_by,
            map_by=map_by,
            collection=collection,
        )

    if getter is not None and not (primary_key or index or sql_getter is not None):
        return Prop(
            default=default,
            default_factory=default_factory,
            alias=alias,
            getter=getter,
            setter=setter,
        )

    return Attr(
        default=default,
        default_factory=default_factory,
        alias=alias,
        index=index is not False,
        primary_key=primary_key is not False,
        getter=getter,
        setter=setter,
        sql_getter=sql_getter,
    )


# def computed(func: Callable[[Record], Val]) -> CompProp[Val, RO]:
#     """Define a computed property."""
#     return CompProp(getter=func)


@dataclass_transform(kw_only_default=True, field_specifiers=(prop,))
class RecordMeta(type):
    """Metaclass for record types."""

    def __new__(cls, name, bases, namespace, **_):
        """Create a new record type."""
        return super().__new__(cls, name, bases, namespace)

    @property
    def _prop_defs(cls) -> dict[str, TypeDef]:
        return {
            name: TypeDef(hint, ctx=getmodule(cls))
            for name, hint in cls.__annotations__.items()
            if issubclass(get_origin(hint) or type, Prop)
            or isinstance(hint, str)
            and ("Attr" in hint or "Rel" in hint)
        }

    def __init__(cls, name, bases, dct):
        """Initialize a new record type."""
        super().__init__(name, bases, dct)

        for name, type_ in cls._prop_defs.items():
            if name not in cls.__dict__:
                setattr(cls, name, type_.prop_type()(_name=name))

    @property
    def _record_bases(cls) -> list[type[Record]]:
        """Get all direct record superclasses of this class."""
        return [
            c
            for c in cls.__bases__
            if issubclass(c, Record) and c is not Record and c is not cls
        ]

    @property
    def _base_pks(cls: type[Record]) -> dict[type[Record], dict[AttrRef, AttrRef]]:
        return {
            base: {
                AttrRef(
                    alias=pk.name,
                    record_type=cls,
                    prop_type=pk.prop_type,
                    primary_key=True,
                ): pk
                for pk in base._primary_keys.values()
            }
            for base in cls._record_bases
        }

    @property
    def _static_props(cls) -> dict[str, PropRef]:
        return {name: getattr(cls, name) for name in cls._prop_defs.keys()}

    @property
    def _dynamic_props(cls) -> dict[str, PropRef]:
        return {
            name: getattr(cls, name)
            for name, prop in cls.__dict__.items()
            if isinstance(prop, Prop)
        }

    @property
    def _defined_props(cls) -> dict[str, PropRef]:
        return {
            **{
                name: prop
                for name, prop in [
                    *cls._static_props.items(),
                    *cls._dynamic_props.items(),
                ]
            },
        }

    @property
    def _defined_attrs(cls) -> dict[str, AttrRef]:
        return {
            name: getattr(cls, name)
            for name, attr in cls._defined_props.items()
            if isinstance(attr, AttrRef)
        }

    @property
    def _defined_rels(cls) -> dict[str, RelRef]:
        return {
            name: getattr(cls, name)
            for name, rel in cls._defined_props.items()
            if isinstance(rel, RelRef)
        }

    @property
    def _defined_rel_attrs(cls) -> dict[str, AttrRef]:
        return {
            a.name: a for rel in cls._defined_rels.values() for a in rel.fk_map.keys()
        }

    @property
    def _primary_keys(cls) -> dict[str, AttrRef]:
        """Return the defined primary key attributes of this record."""
        base_pks = {v.name: v for vs in cls._base_pks.values() for v in vs.keys()}

        if len(base_pks) > 0:
            assert all(not a.primary_key for a in cls._defined_attrs.values())
            return base_pks

        return {name: p for name, p in cls._defined_attrs.items() if p.primary_key}

    @property
    def _rels(cls) -> dict[str, RelRef]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._rels for c in cls._record_bases),
            cls._defined_rels,
        )

    @property
    def _attrs(cls) -> dict[str, AttrRef]:
        return reduce(
            lambda x, y: {**x, **y},
            (c._attrs for c in cls._record_bases),
            cls._defined_attrs,
        )

    @property
    def _rel_types(cls) -> set[type[Record]]:
        """Return all record types that are related to this record."""
        return {rel.target_type for rel in cls._rels.values()}


class Record(Generic[Key_def], metaclass=RecordMeta):
    """Schema for a record in a database."""

    _table_name: ClassVar[str]
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID: UUIDType(binary=False),  # Binary type causes issues with DuckDB
    }

    def __post_init__(self):
        """Initialize a new record."""
        self._values: dict[str, Any] = {}
        self._fetcher: Callable[[str], Any] | None = None

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        return (
            cls._table_name
            if hasattr(cls, "_table_name")
            else PyObjectRef.reference(cls).fqn.replace(".", "_")
        )

    @classmethod
    def _sql_table_name(
        cls,
        subs: Mapping[type[Record], sqla.TableClause],
    ) -> str:
        """Return a SQLAlchemy table object for this schema."""
        sub = subs.get(cls)
        return sub.name if sub is not None else cls._default_table_name()

    @classmethod
    def _columns(cls, registry: orm.registry) -> list[sqla.Column]:
        """Columns of this record type's table."""
        table_attrs = (
            set(cls._defined_attrs.values())
            | set(cls._defined_rel_attrs.values())
            | set(cls._primary_keys.values())
        )

        return [
            sqla.Column(
                attr.name,
                (
                    registry._resolve_type(attr.prop_type.value_type())
                    if attr.prop_type
                    else None
                ),
                primary_key=attr.primary_key,
                index=attr.index,
            )
            for attr in table_attrs
        ]

    @classmethod
    def _foreign_keys(
        cls, metadata: sqla.MetaData, subs: Mapping[type[Record], sqla.TableClause]
    ) -> list[sqla.ForeignKeyConstraint]:
        """Foreign key constraints for this record type's table."""
        return [
            *(
                sqla.ForeignKeyConstraint(
                    [attr.name for attr in rel.fk_map.keys()],
                    [attr.name for attr in rel.fk_map.values()],
                    table=rel.target_type._table(metadata, subs),
                    name=f"{cls._sql_table_name(subs)}_{rel.name}_fk",
                )
                for rel in cls._defined_rels.values()
                if rel.fk_record_type is cls
            ),
            *(
                sqla.ForeignKeyConstraint(
                    [attr.name for attr in pks.keys()],
                    [attr.name for attr in pks.values()],
                    table=base._table(metadata, subs),
                    name=f"{cls._sql_table_name(subs)}_{base._sql_table_name(subs)}_inheritance_fk",
                )
                for base, pks in cls._base_pks.items()
            ),
        ]

    @classmethod
    def _table(
        cls,
        metadata: sqla.MetaData,
        subs: Mapping[type[Record], sqla.TableClause],
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        table_name = cls._sql_table_name(subs)

        if table_name in metadata.tables:
            # Return the table object from metadata if it already exists.
            # This is necessary to avoid circular dependencies.
            return metadata.tables[table_name]

        registry = orm.registry(metadata=metadata, type_annotation_map=cls._type_map)
        sub = subs.get(cls)

        # Create a partial SQLAlchemy table object from the class definition
        # without foreign keys to avoid circular dependencies.
        # This adds the table to the metadata.
        sqla.Table(
            table_name,
            registry.metadata,
            *cls._columns(registry),
            schema=(sub.schema if sub is not None else None),
        )

        # Re-create the table object with foreign keys and return it.
        return sqla.Table(
            table_name,
            registry.metadata,
            *cls._columns(registry),
            *cls._foreign_keys(metadata, subs),
            schema=(sub.schema if sub is not None else None),
            extend_existing=True,
        )

    @classmethod
    def _joined_table(
        cls,
        metadata: sqla.MetaData,
        subs: Mapping[type[Record], sqla.TableClause],
    ) -> sqla.Table | sqla.Join:
        """Recursively join all bases of this record to get the full data."""
        table = cls._table(metadata, subs)

        base_joins = [
            (base._joined_table(metadata, subs), pk_map)
            for base, pk_map in cls._base_pks.items()
        ]
        for target_table, pk_map in base_joins:
            table = table.join(
                target_table,
                reduce(
                    sqla.and_,
                    (
                        table.c[pk.name] == target_table.c[target_pk.name]
                        for pk, target_pk in pk_map.items()
                    ),
                ),
            )

        return table

    @classmethod
    def _backrels_to_rels(
        cls, target: type[Rec]
    ) -> set[RelRef[Rec, Iterable[Self], Self]]:
        """Get all direct relations to a target record type."""
        return {
            RelRef(
                on=cls,
                record_type=target,
                prop_type=TypeDef(Rel[Iterable[cls]]),
            )
            for rel in cls._rels.values()
            if issubclass(target, rel.target_type)
        }

    @overload
    @classmethod
    def rel(
        cls, other: type[Rec], index: None = ...
    ) -> RelRef[Self, Iterable[Rec], Rec]: ...

    @overload
    @classmethod
    def rel(
        cls, other: type[Rec], index: type[Key]
    ) -> RelRef[Self, Mapping[Key, Rec], Rec]: ...

    @classmethod
    def rel(cls, other: type[Rec], index: type[Key] | None = None) -> RelRef:
        """Dynamically define a relation to another record type."""
        value_type = Mapping[index, other] if index is not None else Iterable[other]
        return RelRef(
            on=other,
            record_type=cls,
            prop_type=TypeDef(Rel[value_type]),
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:  # noqa: D105
        assert cls._default_table_name() is not None
        return sqla.table(cls._default_table_name())

    def __hash__(self) -> int:
        """Hash the record."""
        return gen_int_hash(self)

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)


class Schema:
    """Group multiple record types into a schema."""

    _record_types: set[type[Record]]
    _rel_record_types: set[type[Record]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        subclasses = get_subclasses(cls, max_level=1)
        cls._record_types = {s for s in subclasses if isinstance(s, Record)}
        cls._rel_record_types = {rr for r in cls._record_types for rr in r._rel_types}
        super().__init_subclass__()


@dataclass
class Require:
    """Mark schema or record type as required."""

    present: bool = True


class RecUUID(Record[UUID]):
    """Dynamically defined record type."""

    _id: Attr[UUID] = prop(primary_key=True, default_factory=uuid4)


class Scalar(Record[Key_def], Generic[Val, Key_def]):
    """Dynamically defined record type."""

    _id: Attr[Key_def] = prop(primary_key=True, default_factory=uuid4)
    _value: Attr[Val]


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls: type[Record], name: str) -> AttrRef:
        """Get dynamic attribute by dynamic name."""
        return AttrRef(_name=name, record_type=cls, prop_type=TypeDef())

    def __getattr__(cls: type[Record], name: str) -> AttrRef:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)

        return AttrRef(_name=name, record_type=cls, prop_type=TypeDef())


class DynRecord(Record, metaclass=DynRecordMeta):
    """Dynamically defined record type."""


a = DynRecord


def dynamic_record_type(name: str, props: Iterable[Prop] = []) -> type[DynRecord]:
    """Create a dynamically defined record type."""
    return type(name, (DynRecord,), {p.name: p for p in props})


type AggMap[Rec: Record] = dict[AttrRef[Rec, Any], AttrRef | sqla.Function]


@dataclass(frozen=True)
class Agg(Generic[Rec]):
    """Define an aggregation map."""

    target: type[Rec]
    map: AggMap[Rec]
