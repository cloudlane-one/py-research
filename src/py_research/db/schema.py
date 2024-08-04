"""Static schemas for universal relational databases."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
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
    TypeVar,
    TypeVarTuple,
    cast,
    get_args,
    get_origin,
    overload,
)

import sqlalchemy as sqla
import sqlalchemy.orm as orm
import sqlalchemy.sql.type_api as sqla_types
from bidict import bidict

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

Rec = TypeVar("Rec", bound="Record")
Rec2 = TypeVar("Rec2", bound="Record")
Rec3 = TypeVar("Rec3", bound="Record")
Rec_cov = TypeVar("Rec_cov", covariant=True, bound="Record")

type RecordValue[Rec: Record] = Rec | Iterable[Rec] | Mapping[Any, Rec]
Recs = TypeVar("Recs", bound=RecordValue)

Sch = TypeVar("Sch", bound="Schema")
DBS = TypeVar("DBS", bound="Record | Schema")

RelTup = TypeVarTuple("RelTup")


def _extract_record_type(hint: Any) -> type["Record"]:
    origin: type = get_origin(hint)

    if issubclass(origin, Record):
        return origin
    if issubclass(origin, Mapping):
        return get_args(hint)[1]
    if issubclass(origin, Iterable):
        return get_args(hint)[0]

    raise ValueError("Invalid record value.")


@dataclass(frozen=True)
class TypeRef(Generic[Val]):
    """Reference to a type."""

    hint: str | type[Val]
    ctx: ModuleType | None = None

    def resolve(self) -> type[Val]:
        """Resolve the type reference."""
        generic = (
            self.hint
            if isinstance(self.hint, type)
            else cast(type[Val], eval(self.hint, vars(self.ctx)))
        )
        assert issubclass(get_origin(generic) or type, Prop)
        return get_args(generic)[0]


@dataclass(kw_only=True, eq=False)
class Prop(Generic[Val]):
    """Reference a property of a record."""

    primary_key: bool = False

    _name: str | None = None
    _value: Val | None = None

    @property
    def name(self) -> str:
        """Property name."""
        assert self._name is not None, "Property name not set."
        return self._name

    @property
    def value(self) -> Val:
        """Property value."""
        assert self._value is not None, "Property value not set."
        return self._value

    def __set_name__(self, _, name: str) -> None:  # noqa: D105
        self._name = name

    @overload
    def __get__(self, instance: "Record", owner: type["Record"]) -> Val: ...

    @overload
    def __get__(self, instance: None, owner: type[Rec]) -> "PropRef[Rec, Val]": ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: "object | None", owner: type | type[Rec] | None
    ) -> "Val | PropRef[Rec, Val] | Self":
        if instance is not None:
            return self.value
        elif owner is not None and issubclass(owner, Record):
            return PropRef(
                primary_key=self.primary_key,
                _name=self._name,
                record_type=cast(type[Rec], owner),
                value_type=owner._static_types[self.name],
            )
        return self

    def __set__(self, instance: "Record", value: Val) -> None:
        """Set the value of the property."""
        self._value = value

    def __hash__(self) -> int:
        """Hash the Prop."""
        return gen_int_hash(self)


@dataclass(kw_only=True, eq=False)
class PropRef(Prop[Val], Generic[Rec_cov, Val]):
    """Reference a property of a record."""

    record_type: type[Rec_cov]
    value_type: TypeRef[Val]

    @staticmethod
    def get_tag(rec_type: type["Record"]) -> "RelRef | None":
        """Retrieve relation-tag of a record type, if any."""
        try:
            rel = getattr(rec_type, "_rel")
            return rel if isinstance(rel, RelRef) else None
        except AttributeError:
            return None

    @cached_property
    def parent(self) -> "RelRef | None":
        """Parent relation of this Rel."""
        return self.get_tag(self.record_type)

    @cached_property
    def path(self) -> list["RelRef"]:
        """Path from base record type to this Rel."""
        return [
            *(self.parent.path if self.parent is not None else []),
            *([self] if isinstance(self, RelRef) else []),
        ]

    def __rrshift__(self, left: "type[Record] | RelRef") -> Self:
        """Append a prop to the relation path."""
        current_root = self.path[0]
        new_root = (
            left if isinstance(left, RelRef) else left.rel(current_root.record_type)
        )

        prefixed_rel = reduce(
            lambda r1, r2: RelRef(**asdict(r2), record_type=r1.a),
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
class Attr(Prop[Val]):
    """Define an attribute of a record."""

    attr_name: str | None = None

    @overload
    def __get__(self, instance: "Record", owner: type["Record"]) -> Val: ...

    @overload
    def __get__(self, instance: None, owner: type[Rec]) -> "AttrRef[Rec, Val]": ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: "object | None", owner: type | type[Rec] | None
    ) -> "Val | AttrRef[Rec, Val] | Self":
        if instance is not None:
            return self.value
        elif owner is not None and issubclass(owner, Record):
            return AttrRef(
                primary_key=self.primary_key,
                _name=self._name,
                record_type=cast(type[Rec], owner),
                value_type=owner._static_types[self.name],
                attr_name=self.attr_name,
            )
        return self


@dataclass(kw_only=True, eq=False)
class AttrRef(
    Attr[Val], PropRef[Rec_cov, Val], sqla.ColumnClause[Val], Generic[Rec_cov, Val]
):
    """Reference an attribute of a record."""

    def __post_init__(self) -> None:  # noqa: D105
        # Initialize fields required by SQLAlchemy superclass.
        self.table = None
        self.is_literal = False

    # Replace fields of SQLAlchemy superclass with properties:

    @property
    def name(self) -> str:
        """Column name."""
        assert self._name is not None, "Prop name not set."
        return self.attr_name or self._name

    @property
    def key(self) -> str:
        """Column key."""
        return self.name

    @cached_property
    def type(self) -> sqla_types.TypeEngine:
        """Column key."""
        return sqla_types.to_instance(self._value_type)  # type: ignore


@dataclass(kw_only=True, eq=False)
class Rel(Prop[Recs]):
    """Define a relation to another record."""

    via: "AttrRef | Iterable[AttrRef] | dict[AttrRef, AttrRef] | type[Record] | RelRef | tuple[RelRef, RelRef] | None" = (  # noqa: E501
        None
    )
    index_by: "AttrRef | None" = None

    @overload
    def __get__(self, instance: "Record", owner: type["Record"]) -> Recs: ...

    @overload
    def __get__(
        self: "Rel[Rec2]", instance: None, owner: type[Rec]
    ) -> "RelRef[Rec, Rec2, Rec2]": ...

    @overload
    def __get__(
        self: "Rel[Iterable[Rec2]]", instance: None, owner: type[Rec]
    ) -> "RelRef[Rec, Iterable[Rec2], Rec2]": ...

    @overload
    def __get__(
        self: "Rel[Sequence[Rec2]]", instance: None, owner: type[Rec]
    ) -> "RelRef[Rec, Sequence[Rec2], Rec2]": ...

    @overload
    def __get__(
        self: "Rel[Mapping[Idx, Rec2]]",
        instance: None,
        owner: type[Rec],
    ) -> "RelRef[Rec, Mapping[Idx, Rec2], Rec2]": ...

    @overload
    def __get__(self, instance: None, owner: type[Rec]) -> "RelRef[Rec, Recs, Any]": ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105 # type: ignore
        self, instance: "object | None", owner: type | type[Rec] | None
    ) -> "Recs | RelRef | Self":
        if instance is not None:
            return self.value
        elif owner is not None and issubclass(owner, Record):
            typeref = owner._static_types.get(self.name)
            target = (
                _extract_record_type(typeref.resolve()) if typeref is not None else None
            )
            return RelRef(
                primary_key=self.primary_key,
                _name=self._name,
                record_type=owner,
                value_type=typeref or TypeRef(Iterable[target or Any]),
                via=self.via,
                index_by=self.index_by,
                _target_type=target,
            )
        return self


@dataclass(kw_only=True, eq=False)
class RelRef(Rel[Recs], PropRef[Rec_cov, Recs], Generic[Rec_cov, Recs, Rec2]):
    """Reference a relation to another record."""

    _target_type: type[Rec2] | None = None

    @property
    def target_type(self) -> type["Record"]:
        """Dynamic target type of the relation."""
        if self._target_type is not None:
            return self._target_type

        match self.via:
            case dict():
                return next(iter(self.via.values())).record_type
            case type():
                return self.via
            case RelRef():
                if issubclass(self.record_type, self.via.target_type):
                    return self.via.record_type
                else:
                    return self.via.target_type
            case tuple():
                return self.via[1].target_type
            case _:
                raise ValueError("Invalid relation definition.")

    @cached_property
    def fk_record_type(self) -> type["Record"]:
        """Record type of the foreign key."""
        match self.via:
            case type():
                return self.via
            case RelRef():
                return self.via.record_type
            case tuple():
                return self.via[0].record_type
            case dict() | AttrRef() | Iterable() | None:
                return self.record_type

    @cached_property
    def fk_map(self) -> bidict["AttrRef", "AttrRef"]:
        """Map source foreign keys to target attrs."""
        target = self.target_type

        match self.via:
            case type() | RelRef() | tuple():
                return bidict()
            case dict():
                return bidict(
                    {
                        AttrRef(
                            _name=fk.name,
                            record_type=self.record_type,
                            value_type=fk.value_type,
                        ): pk
                        for fk, pk in self.via.items()
                    }
                )
            case AttrRef() | Iterable():
                attrs = self.via if isinstance(self.via, Iterable) else [self.via]
                source_attrs = [
                    AttrRef(
                        _name=attr.name,
                        record_type=self.record_type,
                        value_type=attr.value_type,
                    )
                    for attr in attrs
                ]
                target_attrs = target._indexes
                fk_map = dict(zip(source_attrs, target_attrs))

                assert all(
                    issubclass(
                        self.record_type._static_props[
                            fk_attr.name
                        ].value_type.resolve(),
                        pk_attr.value_type.resolve(),
                    )
                    for fk_attr, pk_attr in fk_map.items()
                    if pk_attr.value_type is not None
                ), "Foreign key value types must match primary key value types."

                return bidict(fk_map)
            case None:
                return bidict(
                    {
                        AttrRef(
                            _name=f"{self._name}.{target_attr.name}",
                            record_type=self.record_type,
                            value_type=target_attr.value_type,
                        ): target_attr
                        for target_attr in target._indexes
                    }
                )

    @cached_property
    def inter_joins(
        self,
    ) -> dict[type["Record"], list[Mapping["AttrRef", "AttrRef"]]]:
        """Intermediate joins required by this rel."""
        match self.via:
            case RelRef():
                # Relation is defined via other, backlinking relation
                other_rel = self.via
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
                if issubclass(self.via, self.target_type):
                    # Relation is defined via all direct backlinks of given record type.
                    return {}

                # Relation is defined via relation table
                back_rels = [
                    rel
                    for rel in self.via._rels
                    if issubclass(rel.target_type, self.record_type)
                    and len(rel.fk_map) > 0
                ]

                return {self.via: [back_rel.fk_map.inverse for back_rel in back_rels]}
            case tuple() if has_type(self.via, tuple[RelRef, RelRef]):
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                back, _ = self.via
                assert len(back.fk_map) > 0, "Back relation must be direct."
                return {back.record_type: [back.fk_map.inverse]}
            case _:
                # Relation is defined via foreign key attributes
                return {}

    @cached_property
    def joins(self) -> list[Mapping["AttrRef", "AttrRef"]]:
        """Mappings of column keys to join the target on."""
        match self.via:
            case RelRef():
                # Relation is defined via other relation or relation table
                other_rel = self.via
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
                if issubclass(self.via, self.target_type):
                    # Relation is defined via all direct backlinks of given record type.
                    back_rels = [
                        rel
                        for rel in self.via._rels
                        if issubclass(rel.target_type, self.record_type)
                        and len(rel.fk_map) > 0
                    ]
                    assert len(back_rels) > 0, "No direct backlinks found."
                    return [back_rel.fk_map.inverse for back_rel in back_rels]

                # Relation is defined via relation table
                fwd_rels = [
                    rel
                    for rel in self.via._rels
                    if issubclass(rel.target_type, self.record_type)
                    and len(rel.fk_map) > 0
                ]
                assert (
                    len(fwd_rels) > 0
                ), "No direct forward rels found on relation table."
                return [fwd_rel.fk_map for fwd_rel in fwd_rels]

            case tuple():
                assert has_type(self.via, tuple[RelRef, RelRef])
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                _, fwd = self.via
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
        backlink_records: set[type["Record"]] | None = None,
        _traversed: set["RelRef"] | None = None,
    ) -> set["RelRef"]:
        """Find all paths to the target record type."""
        backlink_records = backlink_records or set()
        _traversed = _traversed or set()

        # Get relations of the target type as next relations
        next_rels = self.record_type._rels

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

    def __add__(self, other: "RelRef[Any, Any, Rec3]") -> "RelTree[Rec2, Rec3]":
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
    def root(self) -> type["Record"]:
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

    def __rrshift__(self, prefix: type["Record"] | RelRef) -> Self:
        """Prefix all relations in the set with given relation."""
        rels = {prefix >> rel for rel in self.rels}
        return cast(Self, RelTree(rels))

    def __add__(
        self, other: "RelRef[Any, Any, Rec] | RelTree"
    ) -> "RelTree[*RelTup, Rec]":
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    def __or__(
        self: "RelTree[Rec]", other: "RelRef[Any, Any, Rec2] | RelTree[Rec2]"
    ) -> "RelTree[Rec | Rec2]":
        """Add more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])


class RecordMeta(type):
    """Metaclass for record types."""

    @property
    def _id(cls) -> AttrRef:
        """Default primary key attribute."""
        return AttrRef(
            _name="id",
            record_type=cls,
            value_type=TypeRef(int),
            primary_key=True,
        )

    @property
    def _static_types(cls) -> dict[str, TypeRef]:
        return {
            name: TypeRef(hint, ctx=getmodule(cls))
            for name, hint in cls.__annotations__.items()
            if issubclass(get_origin(hint) or type, Prop)
            or isinstance(hint, str)
            and "Attr" in hint
            or "Rel" in hint
        }

    @property
    def _static_props(cls) -> dict[str, PropRef]:
        return {name: getattr(cls, name) for name in cls._static_types.keys()}

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
            name: prop
            for name, prop in [
                *cls._static_props.items(),
                *cls._dynamic_props.items(),
            ]
        }

    @property
    def _defined_attrs(cls) -> set[AttrRef]:
        return {
            getattr(cls, name)
            for name, attr in cls._defined_props.items()
            if isinstance(attr, AttrRef)
        }

    @property
    def _defined_rels(cls) -> set[RelRef]:
        return {
            getattr(cls, name)
            for name, rel in cls._defined_props.items()
            if isinstance(rel, RelRef)
        }

    @property
    def _record_bases(cls) -> list[type["Record"]]:
        """Get all direct record superclasses of this class."""
        return [
            c
            for c in cls.__bases__
            if issubclass(c, Record) and c is not Record and c is not cls
        ]

    @property
    def _base_pks(cls) -> dict[type["Record"], dict[AttrRef, AttrRef]]:
        return {
            base: {
                AttrRef(
                    attr_name=pk.name,
                    record_type=cls,
                    value_type=pk.value_type,
                    primary_key=True,
                ): pk
                for pk in base._indexes
            }
            for base in cls._record_bases
        }

    @property
    def _rels(cls) -> set[RelRef]:
        return reduce(
            set.union, (c._rels for c in cls._record_bases), cls._defined_rels
        )

    @property
    def _attrs(cls) -> set[AttrRef]:
        return reduce(
            set.union, (c._attrs for c in cls._record_bases), cls._defined_attrs
        )

    @property
    def _rel_types(cls) -> set[type["Record"]]:
        """Return all record types that are related to this record."""
        return {rel.target_type for rel in cls._rels}

    @property
    def _indexes(cls) -> set[AttrRef]:
        """Return the primary key attributes of this schema."""
        defined_pks = {p for p in cls._attrs if p.primary_key}
        return defined_pks or {cls._id}

    if not TYPE_CHECKING:

        def __getattr__(cls, name: str) -> AttrRef:
            """Get and instantiate attribute by name."""
            if name in cls.__dict__:
                return cls.__dict__[name]

            if name.startswith("__"):
                return super().__getattribute__(name)

            static_types = cls._static_types
            if name in static_types:
                return Prop(_name=name)

            raise AttributeError(f"{cls.__name__} has no attribute '{name}'.")


class Record(Generic[Idx, Val], metaclass=RecordMeta):
    """Schema for a record in a database."""

    _table_name: str
    _type_map: dict[type, sqla.types.TypeEngine] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
    }

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        return (
            cls._table_name
            if hasattr(cls, "_table_name")
            else PyObjectRef.reference(cls).fqn.replace(".", "_")
        )

    @classmethod
    def _columns(cls, registry: orm.registry) -> list[sqla.Column]:
        """Columns of this record type's table."""
        base_pk_cols = {pk for pks in cls._base_pks.values() for pk in pks.keys()}
        table_attrs = (
            cls._defined_attrs
            | base_pk_cols
            | {a for rel in cls._defined_rels for a in rel.fk_map.keys()}
        )

        return [
            sqla.Column(
                attr.name,
                (
                    registry._resolve_type(attr.value_type.resolve())
                    if attr.value_type
                    else None
                ),
                primary_key=attr.primary_key,
            )
            for attr in table_attrs
        ]

    @classmethod
    def _foreign_keys(
        cls, metadata: sqla.MetaData, subs: Mapping[type["Record"], sqla.TableClause]
    ) -> list[sqla.ForeignKeyConstraint]:
        """Foreign key constraints for this record type's table."""
        return [
            *(
                sqla.ForeignKeyConstraint(
                    [attr.name for attr in rel.fk_map.keys()],
                    [attr.name for attr in rel.target_type._indexes],
                    table=(rel.target_type._table(metadata, subs)),
                    name=f"{cls._default_table_name()}_{rel._name}_fk",
                )
                for rel in cls._defined_rels
            ),
            *(
                sqla.ForeignKeyConstraint(
                    [attr.name for attr in pks.keys()],
                    [attr.name for attr in pks.values()],
                    table=(base._table(metadata, subs)),
                    name=f"{cls._default_table_name()}_{base._default_table_name()}_inheritance_fk",
                )
                for base, pks in cls._base_pks.items()
            ),
        ]

    @classmethod
    def _table(
        cls,
        metadata: sqla.MetaData,
        subs: Mapping[type["Record"], sqla.TableClause],
    ) -> sqla.Table:
        """Return a SQLAlchemy table object for this schema."""
        registry = orm.registry(metadata=metadata, type_annotation_map=cls._type_map)

        sub = subs.get(cls)

        # Create a SQLAlchemy table object from the class definition
        return sqla.Table(
            sub.name if sub is not None else cls._default_table_name(),
            registry.metadata,
            *cls._columns(registry),
            *cls._foreign_keys(metadata, subs),
            schema=(sub.schema if sub is not None else None),
        )

    @classmethod
    def _joined_table(
        cls,
        metadata: sqla.MetaData,
        subs: Mapping[type["Record"], sqla.TableClause],
    ) -> sqla.Table | sqla.Join:
        """Recursively join all bases of this record to get the full data."""
        table = cls._table(metadata, subs)

        base_joins = [
            (base._joined_table(metadata, subs), pk_map)
            for base, pk_map in cls._base_pks.items()
        ]
        for target, pk_map in base_joins:
            table = table.join(
                target,
                reduce(
                    sqla.and_,
                    (
                        table.c[pk.name] == target.c[target_pk.name]
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
                via=cls,
                record_type=target,
                value_type=TypeRef(Iterable[cls]),
                _target_type=cls,
            )
            for rel in cls._rels
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
            via=other,
            record_type=cls,
            value_type=TypeRef(value_type),
            _target_type=other,
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:  # noqa: D105
        assert cls._default_table_name() is not None
        return sqla.table(cls._default_table_name())


class Schema:
    """Group multiple record types into a schema."""

    _record_types: set[type["Record"]]
    _rel_record_types: set[type["Record"]]

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

    def __getitem__(cls, name: str) -> AttrRef:
        """Get dynamic attribute by dynamic name."""
        return AttrRef(_name=name, record_type=cls, value_type=TypeRef(Any))

    def __getattr__(cls, name: str) -> AttrRef:
        """Get dynamic attribute by name."""
        if not TYPE_CHECKING and name.startswith("__"):
            return super().__getattribute__(name)

        return AttrRef(_name=name, record_type=cls, value_type=TypeRef(Any))


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
