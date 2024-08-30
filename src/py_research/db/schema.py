"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import InitVar, asdict, dataclass, field
from functools import cached_property, reduce
from inspect import getmodule
from secrets import token_hex
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Self,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)
from uuid import UUID

import sqlalchemy as sqla
import sqlalchemy.orm as orm
import sqlalchemy.sql.roles as sqla_roles
import sqlalchemy.sql.type_api as sqla_types
from bidict import bidict
from pydantic._internal._model_construction import ModelMetaclass
from sqlalchemy_utils import UUIDType
from typing_extensions import TypeVar, TypeVarTuple

from py_research.hashing import gen_int_hash
from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import has_type


class BaseIdx:
    """Singleton to mark dataset index as default index."""

    __hash__: ClassVar[None]  # type: ignore[assignment]


class SingleIdx(BaseIdx):
    """Singleton to mark dataset index as a single value."""


Idx = TypeVar("Idx", bound="Hashable | BaseIdx")
Key = TypeVar("Key", bound=Hashable)

Val = TypeVar("Val")
Val2 = TypeVar("Val2")

PVal = TypeVar("PVal", bound="Prop")

Rec = TypeVar("Rec", bound="Record")
Rec2 = TypeVar("Rec2", bound="Record")
Rec3 = TypeVar("Rec3", bound="Record")
Rec_cov = TypeVar("Rec_cov", covariant=True, bound="Record")
Rec_def = TypeVar("Rec_def", bound="Record", default="Record")

type RecordValue[Rec: Record] = Rec | Iterable[Rec] | Mapping[Any, Rec]
Recs = TypeVar("Recs", bound=RecordValue)
Recs_def = TypeVar("Recs_def", bound=RecordValue, default=Rec_def)

Sch = TypeVar("Sch", bound="Schema")
DBS = TypeVar("DBS", bound="Record | Schema")

RelTup = TypeVarTuple("RelTup")


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
class TypeRef(Generic[PVal]):
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


@dataclass(kw_only=True, eq=False)
class Prop(Generic[Val]):
    """Reference a property of a record."""

    alias: str | None = None
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


@dataclass(kw_only=True, eq=False)
class PropRef(Prop[Val], sqla_roles.ExpressionElementRole[Val], Generic[Rec_cov, Val]):
    """Reference a property of a record."""

    record_type: type[Rec_cov]
    prop_type: TypeRef[Prop[Val]]

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

    def label(self, name: str | None) -> sqla.Label[Val]:
        """Label the attribute for SQL queries."""
        return sqla.Label(
            name, self, type_=sqla_types.to_instance(self.prop_type.value_type())  # type: ignore
        )

    def __clause_element__(self) -> Self:  # noqa: D105
        return self


@dataclass(kw_only=True, eq=False)
class Attr(Prop[Val]):
    """Define an attribute of a record."""

    primary_key: bool = False
    default: InitVar[Val | None] = None

    _value: Val | None = None

    def __post_init__(self, default: Val | None) -> None:  # noqa: D105
        self.__default = default

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Val: ...

    @overload
    def __get__(self, instance: None, owner: type[Rec]) -> AttrRef[Rec, Val]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: object | None, owner: type | type[Rec] | None
    ) -> Val | AttrRef[Rec, Val] | Self:
        if isinstance(instance, Record):
            return instance._values[self.name]
        elif owner is not None and issubclass(owner, Record):
            return AttrRef(
                primary_key=self.primary_key,
                _name=self._name,
                alias=self.alias,
                default=self.__default,
                record_type=cast(type[Rec], owner),
                prop_type=owner._prop_defs[self.name],
            )
        return self

    def __set__(self, instance: Record, value: Val) -> None:
        """Set the value of the property."""
        instance._values[self.name] = value


@dataclass(kw_only=True, eq=False)
class AttrRef(
    Attr[Val],
    PropRef[Rec_cov, Val],
    Generic[Rec_cov, Val],
):
    """Reference an attribute of a record."""


@dataclass(kw_only=True, eq=False)
class Rel(Prop[Recs_def], Generic[Rec_def, Recs_def]):
    """Define a relation to another record."""

    on: (
        AttrRef
        | Iterable[AttrRef]
        | dict[AttrRef, AttrRef]
        | RelRef
        | tuple[RelRef, RelRef]
        | type
        | None
    ) = None
    order_by: Mapping[AttrRef, int] | None = None
    map_by: AttrRef | None = None

    @overload
    def __get__(self, instance: Record, owner: type[Record]) -> Recs_def: ...

    @overload
    def __get__(
        self, instance: None, owner: type[Rec]
    ) -> RelRef[Rec, Recs_def, Rec_def]: ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105 # type: ignore
        self, instance: object | None, owner: type | type[Rec] | None
    ) -> Recs_def | RelRef[Rec, Recs_def, Rec_def] | Self:
        if isinstance(instance, Record):
            return instance._values[self.name]
        elif owner is not None and issubclass(owner, Record):
            owner_rec = cast(type[Rec], owner)
            typeref = owner._prop_defs.get(self.name)
            return RelRef(
                _name=self._name,
                record_type=owner_rec,
                prop_type=typeref or TypeRef(),
                on=self.on,
                order_by=self.order_by,
                map_by=self.map_by,
            )
        return self

    def __set__(self, instance: Record, value: Recs_def) -> None:
        """Set the value of the property."""
        instance._values[self.name] = value


@overload
def backrel(
    *,
    to: RelRef[Rec, Any, Any] | type[Rec],
    via: None = None,
    order_by: None = None,
    map_by: None = None,
) -> Rel[Rec, set[Rec]]: ...


@overload
def backrel(
    *,
    to: RelRef[Rec, Any, Any] | type[Rec],
    via: None = None,
    order_by: Mapping[AttrRef[Rec, Any], int],
    map_by: None = None,
) -> Rel[Rec, list[Rec]]: ...


@overload
def backrel(
    *,
    to: RelRef[Rec, Any, Any] | type[Rec],
    via: None = None,
    order_by: None = None,
    map_by: AttrRef[Rec, Val],
) -> Rel[Rec, dict[Val, Rec]]: ...


@overload
def backrel(
    *,
    to: None = None,
    via: (
        RelRef[Any, Any, Rec]
        | tuple[RelRef[Rec2, Any, Any], RelRef[Rec2, Any, Rec]]
        | type[Rec2]
    ),
    order_by: None = None,
    map_by: None = None,
) -> Rel[Rec, set[Rec]]: ...


@overload
def backrel(
    *,
    to: None = None,
    via: (
        RelRef[Any, Any, Rec]
        | tuple[RelRef[Rec2, Any, Any], RelRef[Rec2, Any, Rec]]
        | type[Rec2]
    ),
    order_by: Mapping[AttrRef[Rec | Rec2, Any], int],
    map_by: None = None,
) -> Rel[Rec, list[Rec]]: ...


@overload
def backrel(
    *,
    to: None = None,
    via: (
        RelRef[Any, Any, Rec]
        | tuple[RelRef[Rec2, Any, Any], RelRef[Rec2, Any, Rec]]
        | type[Rec2]
    ),
    order_by: None = None,
    map_by: AttrRef[Rec | Rec2, Val],
) -> Rel[Rec, dict[Val, Rec]]: ...


def backrel(
    *,
    to: RelRef[Rec, Any, Any] | type[Rec] | None = None,
    via: (
        RelRef[Any, Any, Rec]
        | tuple[RelRef[Rec2, Any, Any], RelRef[Rec2, Any, Rec]]
        | type[Rec2]
        | None
    ) = None,
    order_by: Mapping[AttrRef[Rec | Rec2, Any], int] | None = None,
    map_by: AttrRef[Rec | Rec2, Val] | None = None,
) -> Rel[Rec, set[Rec]] | Rel[Rec, list[Rec]] | Rel[Rec, dict[Val, Rec]]:
    """Define a backlinking relation to another record."""
    return Rel(on=via or to, order_by=order_by, map_by=map_by)


@dataclass(kw_only=True, eq=False)
class RelRef(Rel[Rec2, Recs], PropRef[Rec_cov, Recs], Generic[Rec_cov, Recs, Rec2]):
    """Reference a relation to another record."""

    @property
    def target_type(self) -> type[Rec2]:
        """Dynamic target type of the relation."""
        if self.prop_type is not None:
            value_type = self.prop_type.value_type()
            if issubclass(value_type, Record):
                return cast(type[Rec2], _extract_record_type(value_type))

        match self.on:
            case dict():
                return next(iter(self.on.values())).record_type
            case type():
                return self.on
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
                        for rel in other_rel.record_type._rels()
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
    def a(self) -> type[Rec2]:
        """Reference props of the target record type."""
        return cast(type[Rec2], type(token_hex(5), (self.target_type,), {"_rel": self}))

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

    def __add__(self, other: RelRef[Any, Any, Rec3]) -> RelTree[Rec2, Rec3]:
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


@dataclass_transform(kw_only_default=True, field_specifiers=(Attr,))
class RecordMeta(ModelMetaclass, type):
    """Metaclass for record types."""

    def __new__(cls, name, bases, namespace, **_):
        """Create a new record type."""
        return super().__new__(cls, name, bases, namespace)

    @property
    def _id(cls: type[Record]) -> AttrRef:
        """Default primary key attribute."""
        return AttrRef(
            _name="_id",
            record_type=cls,
            prop_type=TypeRef(Attr[UUID]),
            primary_key=True,
        )

    @property
    def _prop_defs(cls) -> dict[str, TypeRef]:
        return {
            name: TypeRef(hint, ctx=getmodule(cls))
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
            "_id": cls._id,
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

    @property
    def _primary_keys(cls) -> dict[str, AttrRef]:
        """Return the defined primary key attributes of this record."""
        return {name: p for name, p in cls._attrs.items() if p.primary_key}


class Record(Generic[Idx, Val], metaclass=RecordMeta):
    """Schema for a record in a database."""

    _table_name: ClassVar[str]
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID: UUIDType(binary=False),  # Binary type causes issues with DuckDB
    }

    def __init__(self):
        """Initialize a new record."""
        self._values = {}

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
        base_pk_cols = {pk for pks in cls._base_pks.values() for pk in pks.keys()}
        table_attrs = (
            set(cls._defined_attrs.values())
            | base_pk_cols
            | {a for rel in cls._defined_rels.values() for a in rel.fk_map.keys()}
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
                prop_type=TypeRef(Rel[cls, Iterable[cls]]),
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
            prop_type=TypeRef(Rel[other, value_type]),
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:  # noqa: D105
        assert cls._default_table_name() is not None
        return sqla.table(cls._default_table_name())


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


class DynRecordMeta(RecordMeta):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls: type[Record], name: str) -> AttrRef:
        """Get dynamic attribute by dynamic name."""
        return AttrRef(_name=name, record_type=cls, prop_type=TypeRef())

    def __getattr__(cls: type[Record], name: str) -> AttrRef:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)

        return AttrRef(_name=name, record_type=cls, prop_type=TypeRef())


class DynRecord(Record, metaclass=DynRecordMeta):  # noqa: N801
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
