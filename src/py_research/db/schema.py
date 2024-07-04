"""Static schemas for SQL databases."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property, reduce
from secrets import token_hex
from typing import (
    Any,
    ClassVar,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    cast,
    get_args,
    get_origin,
    get_type_hints,
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

RecordValue: TypeAlias = Rec | Iterable[Rec] | Mapping[Any, Rec]
Recs = TypeVar("Recs", bound=RecordValue)

Sch = TypeVar("Sch", bound="Schema")
DBS = TypeVar("DBS", bound="Record | Schema")

RelTup = TypeVarTuple("RelTup")


def _extract_record_type(hint: Any) -> type["Record"]:
    origin: type = get_origin(hint)

    if issubclass(origin, Record):
        return origin
    if isinstance(origin, Mapping):
        return get_args(hint)[1]
    if isinstance(origin, Iterable):
        return get_args(hint)[0]

    raise ValueError("Invalid record value.")


@dataclass(kw_only=True, eq=False)
class Prop(Generic[Rec, Val]):
    """Define property of a record."""

    primary_key: bool = False

    _name: str | None = None
    _record_type: type["Record"] | None = None
    _value_type: type[Val] | None = None
    _value: Val | None = None

    @property
    def name(self) -> str:
        """Property name."""
        assert self._name is not None, "Property name not set."
        return self._name

    @property
    def record_type(self) -> type[Rec]:
        """Record type."""
        assert self._record_type is not None, "Record type not set."
        return cast(type[Rec], self._record_type)

    @property
    def value_type(self) -> type[Val]:
        """Property value type."""
        return cast(
            type[Val],
            self._value_type or self.record_type._value_types.get(self.name) or Any,
        )

    @property
    def value(self) -> Val:
        """Property value."""
        assert self._value is not None, "Property value not set."
        return self._value

    def __setname__(self, _, name: str) -> None:  # noqa: D105
        self._name = name

    @overload
    def __get__(self, instance: "Record", owner: type["Record"]) -> Val: ...

    @overload
    def __get__(
        self: "Attr[Any, Val]", instance: None, owner: type[Rec]
    ) -> "Attr[Rec, Val]": ...

    @overload
    def __get__(
        self: "Rel[Any, Rec2, Any]", instance: None, owner: type[Rec]
    ) -> "Rel[Rec, Rec2, Rec2]": ...

    @overload
    def __get__(
        self: "Rel[Any, Iterable[Rec2], Any]", instance: None, owner: type[Rec]
    ) -> "Rel[Rec, Iterable[Rec2], Rec2]": ...

    @overload
    def __get__(
        self: "Rel[Any, Sequence[Rec2], Any]", instance: None, owner: type[Rec]
    ) -> "Rel[Rec, Sequence[Rec2], Rec2]": ...

    @overload
    def __get__(
        self: "Rel[Any, Mapping[Idx, Rec2], Any]", instance: None, owner: type[Rec]
    ) -> "Rel[Rec, Mapping[Idx, Rec2], Rec2]": ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: "object | None", owner: type["Record"] | None
    ) -> "Val | Attr | type[Record] | Rel | Self":
        if owner is not None:
            self._record_type = owner

        if instance is not None:
            return self.value

        return self

    def __set__(self, instance: "Record", value: Val) -> None:
        """Set the value of the property."""
        self._value = value

    @staticmethod
    def get_tag(rec_type: type["Record"]) -> "Rel | None":
        """Retrieve relation-tag of a record type, if any."""
        try:
            rel = getattr(rec_type, "_rel")
            return rel if isinstance(rel, Rel) else None
        except AttributeError:
            return None

    @cached_property
    def parent(self) -> "Rel | None":
        """Parent relation of this Rel."""
        return self.get_tag(self.record_type)

    @cached_property
    def path(self) -> list["Rel"]:
        """Path from base record type to this Rel."""
        return [
            *(self.parent.path if self.parent is not None else []),
            *([self] if isinstance(self, Rel) else []),
        ]

    def __rrshift__(self, left: "type[Record] | Rel") -> Self:
        """Append a prop to the relation path."""
        current_root = self.path[0]
        new_root = left if isinstance(left, Rel) else left.rel(current_root.record_type)

        prefixed_rel = reduce(
            lambda r1, r2: Rel(**asdict(r2), _record_type=r1.a), self.path, new_root
        )

        return cast(
            Self,
            (
                prefixed_rel
                if isinstance(self, Rel)
                else getattr(prefixed_rel.a, self.name)
            ),
        )

    def __hash__(self) -> int:
        """Hash the Prop."""
        return gen_int_hash(self)


@dataclass(kw_only=True, eq=False)
class Attr(Prop[Rec, Val], sqla.ColumnClause[Val]):
    """Define attribute of a record."""

    attr_name: str | None = None

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


@dataclass
class Rel(Prop[Rec, Recs], Generic[Rec, Recs, Rec2]):
    """Define relation to another record."""

    via: (
        Attr
        | Iterable[Attr]
        | dict[Attr, "Attr"]
        | type["Record"]
        | "Rel"
        | tuple["Rel", "Rel"]
        | None
    ) = None
    index_by: "Attr | None" = None

    _target_type: type["Record"] | None = None

    @cached_property
    def target_type(self) -> type["Record"]:
        """Target record type."""
        assert self._target_type is not None, "Target record type not set."
        return self._target_type

    @cached_property
    def fk_record_type(self) -> type["Record"]:
        """Record type of the foreign key."""
        match self.via:
            case type():
                return self.via
            case Rel():
                return self.via.record_type
            case tuple():
                return self.via[0].record_type
            case dict() | Attr() | Iterable() | None:
                return self.record_type

    @cached_property
    def fk_map(self) -> bidict["Attr", "Attr"]:
        """Map source foreign keys to target attrs."""
        target = self.target_type

        match self.via:
            case type() | Rel() | tuple():
                return bidict()
            case dict():
                return bidict(
                    {
                        Attr(
                            _record_type=self.record_type,
                            _name=fk.name,
                            _value_type=fk.value_type,
                        ): pk
                        for fk, pk in self.via.items()
                    }
                )
            case Attr() | Iterable():
                attrs = self.via if isinstance(self.via, Iterable) else [self.via]
                source_attrs = [
                    Attr(
                        _record_type=self.record_type,
                        _name=attr.name,
                        _value_type=attr.value_type,
                    )
                    for attr in attrs
                ]
                target_attrs = target._indexes()
                fk_map = dict(zip(source_attrs, target_attrs))

                assert all(
                    issubclass(
                        self.record_type._value_types[fk_attr.name], pk_attr.value_type
                    )
                    for fk_attr, pk_attr in fk_map.items()
                    if pk_attr.value_type is not None
                ), "Foreign key value types must match primary key value types."

                return bidict(fk_map)
            case None:
                return bidict(
                    {
                        Attr(
                            _record_type=self.record_type,
                            _name=f"{self._name}.{target_attr.name}",
                            _value_type=target_attr.value_type,
                        ): target_attr
                        for target_attr in target._indexes()
                    }
                )

    @cached_property
    def inter_joins(self) -> dict[type["Record"], list[Mapping["Attr", "Attr"]]]:
        """Intermediate joins required by this rel."""
        match self.via:
            case Rel():
                # Relation is defined via other, backlinking relation
                other_rel = self.via
                assert isinstance(
                    other_rel, Rel
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
                    for rel in self.via._rels()
                    if issubclass(rel.target_type, self.record_type)
                    and len(rel.fk_map) > 0
                ]

                return {self.via: [back_rel.fk_map.inverse for back_rel in back_rels]}
            case tuple() if has_type(self.via, tuple[Rel, Rel]):
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                back, _ = self.via
                assert len(back.fk_map) > 0, "Back relation must be direct."
                return {back.record_type: [back.fk_map.inverse]}
            case _:
                # Relation is defined via foreign key attributes
                return {}

    @cached_property
    def joins(self) -> list[Mapping["Attr", "Attr"]]:
        """Mappings of column keys to join the target on."""
        match self.via:
            case Rel():
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
                        for rel in self.via._rels()
                        if issubclass(rel.target_type, self.record_type)
                        and len(rel.fk_map) > 0
                    ]
                    assert len(back_rels) > 0, "No direct backlinks found."
                    return [back_rel.fk_map.inverse for back_rel in back_rels]

                # Relation is defined via relation table
                fwd_rels = [
                    rel
                    for rel in self.via._rels()
                    if issubclass(rel.target_type, self.record_type)
                    and len(rel.fk_map) > 0
                ]
                assert (
                    len(fwd_rels) > 0
                ), "No direct forward rels found on relation table."
                return [fwd_rel.fk_map for fwd_rel in fwd_rels]

            case tuple():
                assert has_type(self.via, tuple[Rel, Rel])
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
        _traversed: set["Rel"] | None = None,
    ) -> set["Rel"]:
        """Find all paths to the target record type."""
        backlink_records = backlink_records or set()
        _traversed = _traversed or set()

        # Get relations of the target type as next relations
        next_rels = self.record_type._rels()

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

    def __add__(self, other: "Rel[Any, Any, Rec3]") -> "RelTree[Rec2, Rec3]":
        """Add another prop to the merge."""
        return RelTree({self}) + other

    def __hash__(self) -> int:
        """Hash the rel."""
        return gen_int_hash(self)


RelDict: TypeAlias = dict[Rel, "RelDict"]


@dataclass
class RelTree(Generic[*RelTup]):
    """Tree of relations starting from the same root."""

    rels: Iterable[Rel] = field(default_factory=set)

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

    def __rrshift__(self, prefix: type["Record"] | Rel) -> Self:
        """Prefix all relations in the set with given relation."""
        rels = {prefix >> rel for rel in self.rels}
        return cast(Self, RelTree(rels))

    def __add__(self, other: "Rel[Any, Any, Rec] | RelTree") -> "RelTree[*RelTup, Rec]":
        """Append more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])

    def __or__(
        self: "RelTree[Rec]", other: "Rel[Any, Any, Rec2] | RelTree[Rec2]"
    ) -> "RelTree[Rec | Rec2]":
        """Add more rels to the set."""
        other = other if isinstance(other, RelTree) else RelTree([other])
        return RelTree([*self.rels, *other.rels])


class Record(Generic[Idx, Val]):
    """Schema for a record in a database."""

    _table_name: str
    _type_map: dict[type, sqla.types.TypeEngine] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
    }

    _value_types: dict[str, type]
    _defined_attrs: set[Attr]
    _defined_rels: set[Rel]

    def __init_subclass__(cls) -> None:  # noqa: D105
        # Retrieve property type definitions from class annotations
        cls._value_types = {
            name: get_args(hint)[0]
            for name, hint in get_type_hints(cls)
            if issubclass(get_origin(hint) or type, Attr)
        }

        cls._defined_attrs = {
            getattr(cls, name)
            for name, attr in cls.__dict__.items()
            if isinstance(attr, Attr)
        }

        cls._defined_rels = {
            getattr(cls, name)
            for name, rel in cls.__dict__.items()
            if isinstance(rel, Rel)
        }

        return super().__init_subclass__()

    @classmethod
    def _record_bases(cls) -> list[type["Record"]]:
        """Get all direct record superclasses of this class."""
        return [
            c
            for c in cls.__bases__
            if issubclass(c, Record) and c is not Record and c is not cls
        ]

    @classmethod
    def _base_pks(cls) -> dict[type["Record"], dict[Attr, Attr]]:
        return {
            base: {
                Attr(
                    _record_type=cls,
                    attr_name=pk.name,
                    _value_type=pk.value_type,
                    primary_key=True,
                ): pk
                for pk in base._indexes()
            }
            for base in cls._record_bases()
        }

    @classmethod
    def _rels(cls) -> set[Rel]:
        return reduce(
            set.union, (c._rels() for c in cls._record_bases()), cls._defined_rels
        )

    @classmethod
    def _attrs(cls) -> set[Attr]:
        return reduce(
            set.union, (c._attrs() for c in cls._record_bases()), cls._defined_attrs
        )

    @classmethod
    def _rel_types(cls) -> set[type["Record"]]:
        """Return all record types that are related to this record."""
        return {rel.target_type for rel in cls._rels()}

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        return (
            cls._table_name
            if hasattr(cls, "_table_name")
            else PyObjectRef.reference(cls).fqn.replace(".", "_")
        )

    @classmethod
    def _indexes(cls) -> set[Attr]:
        """Return the primary key attributes of this schema."""
        return {p for p in cls._attrs() if p.primary_key}

    @classmethod
    def _columns(cls, registry: orm.registry) -> list[sqla.Column]:
        """Columns of this record type's table."""
        base_pk_cols = {pk for pks in cls._base_pks().values() for pk in pks.keys()}
        table_attrs = (
            cls._defined_attrs
            | base_pk_cols
            | {a for rel in cls._defined_rels for a in rel.fk_map.keys()}
        )

        return [
            sqla.Column(
                attr.name,
                (registry._resolve_type(attr.value_type) if attr.value_type else None),
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
                    [attr.name for attr in rel.target_type._indexes()],
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
                for base, pks in cls._base_pks().items()
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
            for base, pk_map in cls._base_pks().items()
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
    def _backrels_to_rels(cls, target: type[Rec]) -> set[Rel[Any, Any, Rec]]:
        """Get all direct relations to a target record type."""
        return {
            Rel(
                _record_type=target,
                _value_type=Iterable[cls],
                _target_type=cls,
                via=cls,
            )
            for rel in cls._rels()
            if issubclass(target, rel.target_type)
        }

    @overload
    @classmethod
    def rel(
        cls, other: type[Rec], index: None = ...
    ) -> Rel[Self, Iterable[Rec], Rec]: ...

    @overload
    @classmethod
    def rel(
        cls, other: type[Rec], index: type[Key]
    ) -> Rel[Self, Mapping[Key, Rec], Rec]: ...

    @classmethod
    def rel(cls, other: type[Rec], index: type[Key] | None = None) -> Rel:
        """Dynamically define a relation to another record type."""
        value_type = Mapping[index, other] if index is not None else Iterable[other]
        return Rel(
            _record_type=cls, _value_type=value_type, _target_type=other, via=other
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
        cls._rel_record_types = {rr for r in cls._record_types for rr in r._rel_types()}
        super().__init_subclass__()


@dataclass
class Require:
    """Mark schema or record type as required."""

    present: bool = True


class DynRecordMeta(type):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls, name: str) -> Attr:
        """Get dynamic attribute by dynamic name."""
        return Attr(_record_type=cls, _name=name, _value_type=Any)

    def __getattribute__(cls, name: str) -> Attr:
        """Get dynamic attribute by name."""
        return Attr(_record_type=cls, _name=name, _value_type=Any)


class DynRecord(Record, metaclass=DynRecordMeta):  # noqa: N801
    """Dynamically defined record type."""


a = DynRecord


def dynamic_record_type(name: str, props: Iterable[Prop] = []) -> type[DynRecord]:
    """Create a dynamically defined record type."""
    return type(name, (DynRecord,), {p.name: p for p in props})


AggMap: TypeAlias = dict[Attr[Rec, Any], Attr | sqla.Function]


@dataclass(frozen=True)
class Agg(Generic[Rec]):
    """Define an aggregation map."""

    target: type[Rec]
    map: AggMap[Rec]
