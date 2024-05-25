"""Static schemas for SQL databases."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property, reduce
from itertools import groupby
from secrets import token_hex
from typing import (
    Any,
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
from bidict import bidict

from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_subclasses
from py_research.reflect.types import has_type, is_subtype

Idx = TypeVar("Idx", bound="Hashable")

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

MergeTup = TypeVarTuple("MergeTup")


def _extract_record_type(hint: Any) -> type["Record"]:
    origin: type = get_origin(hint)

    if issubclass(origin, Record):
        return origin
    if isinstance(origin, Mapping):
        return get_args(hint)[1]
    if isinstance(origin, Iterable):
        return get_args(hint)[0]

    raise ValueError("Invalid record value.")


class Keyless:
    """Singleton to mark a plural relational prop as keyless."""


@dataclass
class Prop(Generic[Val]):
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
    def record_type(self) -> type["Record"]:
        """Record type."""
        assert self._record_type is not None, "Record type not set."
        return self._record_type

    @property
    def value_type(self) -> type[Val]:
        """Property value type."""
        assert self._value_type is not None, "Property value type not set."
        return self._value_type

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
        self: "Attr[Val]", instance: None, owner: type[Rec]
    ) -> "AttrRef[Rec, Val]": ...

    @overload
    def __get__(
        self: "Rel[Rec2]", instance: None, owner: type[Rec]
    ) -> "RelRef[Rec, Rec2, None]": ...

    @overload
    def __get__(
        self: "Rel[Iterable[Rec2]]", instance: None, owner: type[Rec]
    ) -> "RelRef[Rec, Rec2, Keyless]": ...

    @overload
    def __get__(
        self: "Rel[Sequence[Rec2]]", instance: None, owner: type[Rec]
    ) -> "RelRef[Rec, Rec2, int]": ...

    @overload
    def __get__(
        self: "Rel[Mapping[Idx, Rec2]]", instance: None, owner: type[Rec]
    ) -> "RelRef[Rec, Rec2, Idx]": ...

    @overload
    def __get__(self, instance: object | None, owner: type | None) -> Self: ...

    def __get__(  # noqa: D105
        self, instance: "object | None", owner: type[Rec] | None
    ) -> "Val | AttrRef[Rec, Val] | type[Record] | RelRef | Self":
        if owner is None or not issubclass(owner, Record):
            return self

        if instance is not None:
            return self.value

        if isinstance(self, Attr):
            return cast(AttrRef[Rec, Val], owner._all_attrs()[self.name])

        if has_type(self, Rel[Record]):
            return RelRef(
                rec_type=owner,
                val_type=_extract_record_type(self.value_type),
                rel=self,
                idx=None,
            )

        if has_type(self, Rel[Sequence[Record]]):
            return RelRef(
                rec_type=owner,
                val_type=_extract_record_type(self.value_type),
                rel=self,
                idx=int,
            )

        if has_type(self, Rel[Iterable[Record]]):
            return RelRef(
                rec_type=owner,
                val_type=_extract_record_type(self.value_type),
                rel=self,
                idx=Keyless,
            )

        if isinstance(self, Rel) and is_subtype(
            self.value_type, Mapping[Hashable, Record]
        ):
            return RelRef(
                rec_type=owner,
                val_type=_extract_record_type(self.value_type),
                rel=self,
                idx=None,
            )

        raise NotImplementedError()

    def __set__(self, instance: "Record", value: Val) -> None:
        """Set the value of the property."""
        self._value = value


@dataclass
class Attr(Prop[Val]):
    """Define attribute of a record."""

    attr_name: str | None = None

    @property
    def name(self) -> str:
        """Column name."""
        return self.attr_name or super().name


@dataclass
class Rel(Prop[Recs]):
    """Define relation to another record."""

    via: (
        Attr
        | Iterable[Attr]
        | dict[Attr, "AttrRef"]
        | type["Record"]
        | "RelRef"
        | tuple["RelRef", "RelRef"]
        | None
    ) = None
    index_by: "AttrRef | None" = None

    _target: type["Record"] | None = None

    @cached_property
    def target_type(self) -> type["Record"]:
        """Target record type."""
        assert self._target is not None, "Target record type not set."
        return self._target

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
                        AttrRef(self.record_type, fk.name, fk.value_type): pk
                        for fk, pk in self.via.items()
                    }
                )
            case Attr() | Iterable():
                attrs = self.via if isinstance(self.via, Iterable) else [self.via]
                source_attrs = [
                    self.record_type._all_attrs()[attr.name] for attr in attrs
                ]
                target_attrs = target._primary_keys()
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
                        AttrRef(
                            rec_type=self.record_type,
                            name=f"{self._name}.{target_attr.name}",
                            val_type=target_attr.value_type,
                        ): target_attr
                        for target_attr in target._primary_keys()
                    }
                )

    @cached_property
    def inter_join(self) -> tuple[type["Record"], Mapping["AttrRef", "AttrRef"]] | None:
        """Intermediate joins required by this rel."""
        match self.via:
            case RelRef():
                # Relation is defined via other relation or relation table
                other_rel = self.via.rel
                assert isinstance(
                    other_rel, Rel
                ), "Back-reference must be an explicit relation"

                if issubclass(other_rel.target_type, self.record_type):
                    # Supplied record type object is a backlinking relation
                    return None
                else:
                    # Supplied record type object is a forward relation
                    # on a relation table
                    back_rel = [
                        r
                        for rs in other_rel.record_type._rels.values()
                        for r in rs
                        if issubclass(r.target_type, self.record_type)
                    ][0]

                    return (
                        back_rel.record_type,
                        back_rel.fk_map.inverse,
                    )
            case type():
                # Relation is defined via relation table
                back_rel = [
                    r
                    for rs in self.via._rels.values()
                    for r in rs
                    if issubclass(r.target_type, self.record_type)
                ][0]

                return (back_rel.record_type, back_rel.fk_map.inverse)
            case tuple() if has_type(self.via, tuple[RelRef, RelRef]):
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                back, _ = self.via

                assert isinstance(back.rel, Rel)

                return (back.rel.record_type, back.rel.fk_map.inverse)
            case _:
                # Relation is defined via foreign key attributes
                return None

    @cached_property
    def join_on(self) -> Mapping["AttrRef", "AttrRef"]:
        """Mapping of column keys to join the target on."""
        match self.via:
            case RelRef():
                # Relation is defined via other relation or relation table
                other_rel = self.via.rel
                assert isinstance(
                    other_rel, Rel
                ), "Back-reference must be an explicit relation"

                if issubclass(other_rel.target_type, self.record_type):
                    return self.fk_map.inverse
                else:
                    return other_rel.fk_map
            case type():
                # Relation is defined via relation table
                fwd_rel = [
                    r
                    for rs in self.via._rels.values()
                    for r in rs
                    if issubclass(r.target_type, self.target_type)
                ][0]

                return fwd_rel.fk_map
            case tuple() if has_type(self.via, tuple[RelRef, RelRef]):
                # Relation is defined via back-rel + forward-rel
                # on a relation table.
                _, fwd = self.via

                assert isinstance(fwd.rel, Rel)

                return fwd.rel.fk_map
            case _:
                # Relation is defined via foreign key attributes
                return self.fk_map


@dataclass
class PropRef(Generic[Rec, Val]):
    """Reference to related record type, optionally indexed."""

    rec_type: type[Rec]
    val_type: type[Val]

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
        """Parent relation of this relref."""
        return self.get_tag(self.rec_type)

    @cached_property
    def path(self) -> list["RelRef"]:
        """Path from base record type to this relref."""
        return [
            *(self.parent.path if self.parent is not None else []),
            *([self] if isinstance(self, RelRef) else []),
        ]

    def __rrshift__(self, left: type[Rec]) -> Self:
        """Append a prop to the relation path."""
        return cast(Self, PropRef(rec_type=left, val_type=self.val_type))


class AttrRef(sqla.ColumnClause[Val], PropRef[Rec, Val], Generic[Rec, Val]):
    """Reference a property by its containing record type, name and value type."""

    def __init__(  # noqa: D107
        self, rec_type: type[Rec], name: str, val_type: type[Val]
    ) -> None:
        self.rec_type = rec_type
        self.name = name
        self.val_type = val_type

        super().__init__(
            self.name,
            type_=self.record_type._table.c[self.name].type,
        )

    def __getitem__(self, value_type: type[Val2]) -> "AttrRef[Rec, Val2]":
        """Specialize the value type of this attribute."""
        return AttrRef(rec_type=self.rec_type, name=self.name, val_type=value_type)

    def __rrshift__(self, left: type[Rec]) -> Self:
        """Append a prop to the relation path."""
        return cast(
            Self, AttrRef(rec_type=left, name=self.name, val_type=self.val_type)
        )


@dataclass
class RelRef(PropRef[Rec, Rec2], Generic[Rec, Rec2, Idx]):
    """Reference to related record type, optionally indexed."""

    rel: Rel[RecordValue[Rec2]] | None = None
    """Relation prop on source record."""

    idx: type[Idx] | None = None
    """Index type."""

    @cached_property
    def idx_attr(self) -> AttrRef[Any, Idx] | None:
        """Index attr."""
        if self.idx is None:
            return None

        assert (
            self.rel is not None
            and self.rel.index_by is not None
            and issubclass(self.rel.index_by.val_type, self.idx)
        )
        return (
            self.parent >> self.rel.index_by
            if self.parent is not None
            else self.rel.index_by
        )

    @cached_property
    def a(self) -> type[Rec2]:
        """Reference props of the target record type."""
        return cast(type[Rec2], type(token_hex(5), (self.val_type,), {"_rel": self}))

    @cached_property
    def rels(self) -> list[Rel[RecordValue]]:
        """All relations referenced."""
        return (
            [self.rel]
            if self.rel is not None
            else self.rec_type._rels.get(self.val_type, [])
        )

    @cached_property
    def inter_joins(self) -> dict[type["Record"], list[dict["AttrRef", "AttrRef"]]]:
        """Intermediate joins required for all of the rels."""
        rel_grouped = groupby(
            self.rels,
            lambda rel: rel.inter_join[0] if rel.inter_join is not None else None,
        )

        return {
            rec: [dict(rel.inter_join[1]) for rel in rels if rel.inter_join is not None]
            for rec, rels in rel_grouped
            if rec is not None
        }

    @cached_property
    def join_ons(self) -> list[dict["AttrRef", "AttrRef"]]:
        """Mapping of all column keys to join the target on."""
        return [dict(rel.join_on) for rel in self.rels]

    @property
    def path_str(self) -> str:
        """Path to join target table."""
        prefix = (
            self.rec_type.__name__ if len(self.path) == 0 else self.path[-1].path_str
        )

        if self.rel is not None:
            return f"{prefix}.{self.rel.name}"
        else:
            return f"{self.rec_type.__name__}.rel({self.val_type.__name__})"

    def __add__(self, other: "RelRef[Any, Rec3, Any]") -> "RelMerge[Rec2, Rec3]":
        """Add another prop to the merge."""
        return RelMerge([self]) + other

    @overload
    def __rshift__(self, other: AttrRef[Rec2, Val]) -> AttrRef[Rec2, Val]: ...

    @overload
    def __rshift__(
        self, other: "RelRef[Rec2, Rec3, Any]"
    ) -> "RelRef[Rec2, Rec3, Any]": ...

    @overload
    def __rshift__(self, other: "PropRef[Rec2, Val]") -> "PropRef[Rec2, Val]": ...

    def __rshift__(self, other: PropRef[Rec2, Val]) -> PropRef[Rec2, Val]:
        """Append a prop to the relation path."""
        return PropRef(rec_type=self.a, val_type=other.val_type)

    def __rrshift__(self, left: type[Rec]) -> Self:
        """Append a prop to the relation path."""
        return cast(
            Self,
            RelRef(rec_type=left, val_type=self.val_type, rel=self.rel, idx=self.idx),
        )


RelTree: TypeAlias = dict[RelRef, "RelTree"]


@dataclass(frozen=True)
class RelMerge(Generic[*MergeTup]):
    """Tree of joined properties."""

    rels: list[RelRef] = field(default_factory=list)

    @cached_property
    def rel_tree(self) -> RelTree:
        """Tree representation of the merge."""
        tree = {}

        for rel in self.rels:
            subtree = tree
            for ref in rel.path:
                if ref not in subtree:
                    subtree[ref] = {}
                subtree = subtree[ref]

        return tree

    def __add__(
        self, other: "RelRef[Any, Rec, Any] | RelMerge"
    ) -> "RelMerge[*MergeTup, Rec]":
        """Add another relref to the merge."""
        other = other if isinstance(other, RelMerge) else RelMerge([other])
        assert all(
            self.rels[-1].path[0].rec_type == other_rel.path[0].rec_type
            for other_rel in other.rels
        ), "Invalid relation merge."
        return RelMerge([*self.rels, *other.rels])

    def __rrshift__(self, prefix: RelRef) -> "RelMerge[*MergeTup]":
        """Prepend a relref to the merge."""
        rels = [
            reduce(lambda r1, r2: RelRef(r1.a, r2.val_type, r2.rel), rel.path, prefix)
            for rel in self.rels
        ]
        return RelMerge(rels)


class Record(Generic[Idx, Val]):
    """Schema for a record in a database."""

    _table_name: str
    _type_map: dict[type, sqla.types.TypeEngine] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
    }

    _value_types: dict[str, type]
    _defined_attrs: list[Attr]
    _rels: dict[type["Record"], list[Rel[RecordValue]]]
    _rel_types: set[type["Record"]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        # Retrieve property type definitions from class annotations
        cls._value_types = {
            name: get_args(hint)[0]
            for name, hint in get_type_hints(cls)
            if issubclass(get_origin(hint) or type, Attr)
        }

        cls._defined_attrs = [
            Attr(
                **{
                    **(asdict(attr) if isinstance(attr, Attr) else {}),
                    "_name": name,
                    "_record": cls,
                    "_value_type": cls._value_types[name],
                }
            )
            for name, attr in cls.__dict__.items()
            if isinstance(attr, Attr)
        ]

        cls._rels = {}
        for name, rel in cls.__dict__.items():
            if isinstance(rel, Rel):
                target = _extract_record_type(cls._value_types[name])
                cls._rels[target] = [
                    *cls._rels.get(target, {}),
                    Rel(via=rel.via, _target=target),
                ]

        # Crawl through relational tree, visiting each not at most once,
        # to get all related record types.
        cls._rel_types = set(cls._rels.keys())
        crawled = set()
        while remaining := cls._rel_types - crawled:
            rel_type = remaining.pop()
            cls._rel_types |= set(
                rel.target_type for rels in rel_type._rels.values() for rel in rels
            )
            crawled |= {rel_type}

        return super().__init_subclass__()

    @classmethod
    def rel(
        cls, other: type[Rec], index: type[Idx] | None = None
    ) -> RelRef[Self, Rec, Idx]:
        """Dynamically define a relation to another record type."""
        return RelRef(cls, other, idx=index)

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        return (
            cls._table_name
            if hasattr(cls, "_table_name")
            else PyObjectRef.reference(cls).fqn.replace(".", "_")
        )

    @classmethod
    def _all_attrs(cls) -> dict[str, AttrRef]:
        """Return all data attributes of this schema."""
        return {
            **{a.name: AttrRef(cls, a.name, a.value_type) for a in cls._defined_attrs},
            **{
                c.name: AttrRef(cls, c.name, c.value_type)
                for rels in cls._rels.values()
                for rel in rels
                for c in rel.fk_map.keys()
            },
        }

    @classmethod
    def _primary_keys(cls) -> list[AttrRef]:
        """Return the primary key attributes of this schema."""
        return [p for p in cls._all_attrs().values() if p.primary_key]

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
            *(
                sqla.Column(
                    attr.name,
                    (
                        registry._resolve_type(attr.value_type)
                        if attr.value_type
                        else None
                    ),
                    primary_key=attr.primary_key,
                )
                for attr in cls._all_attrs().values()
            ),
            *(
                sqla.ForeignKeyConstraint(
                    [attr.name for attr in rel.fk_map.keys()],
                    [attr.name for attr in rel.target_type._primary_keys()],
                    table=(rel.target_type._table(metadata, subs)),
                    name=f"{cls._default_table_name()}_{rel._name}_fk",
                )
                for rels in cls._rels.values()
                for rel in rels
            ),
            schema=(sub.schema if sub is not None else None),
        )

    def __clause_element__(self) -> sqla.TableClause:  # noqa: D105
        assert self._default_table_name() is not None
        return sqla.table(self._default_table_name())


class Schema:
    """Group multiple record types into a schema."""

    _record_types: set[type["Record"]]
    _rel_record_types: set[type["Record"]]

    def __init_subclass__(cls) -> None:  # noqa: D105
        subclasses = get_subclasses(cls, max_level=1)
        cls._record_types = {s for s in subclasses if isinstance(s, Record)}
        cls._rel_record_types = {rr for r in cls._record_types for rr in r._rel_types}
        super().__init_subclass__()


@dataclass(frozen=True)
class Required(Generic[DBS]):
    """Define required record type in a schema."""

    type: type[DBS]
    """Record type."""


class DynRecordMeta(type):
    """Metaclass for dynamically defined record types."""

    def __getitem__(cls, name: str) -> AttrRef:
        """Get dynamic attribute by dynamic name."""
        return AttrRef(cls, name, Any)


class a(Record, metaclass=DynRecordMeta):  # noqa: N801
    """Dynamically defined record type."""

    def __getattribute__(self, name: str) -> AttrRef:
        """Get dynamic attribute by name."""
        return AttrRef(type(self), name, Any)


@dataclass(frozen=True)
class AttrMap(Generic[Rec, Val]):
    """Map an attribute to a function."""

    attr: AttrRef[Rec, Val]
    func: sqla.Function[Val]


@dataclass(frozen=True)
class Agg(Generic[Rec, Rec2]):
    """Define an aggregation."""

    base_type: type[Rec]
    target_type: type[Rec2]
    attrs: set[AttrMap[Rec2, Any]]
