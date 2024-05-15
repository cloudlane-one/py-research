"""Static schemas for SQL databases."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from secrets import token_hex
from typing import (
    Any,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
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
Rec_cov = TypeVar("Rec_cov", covariant=True, bound="Record")

RecordValue: TypeAlias = Rec | Iterable[Rec] | Mapping[Any, Rec]
Recs = TypeVar("Recs", bound=RecordValue)

Sch = TypeVar("Sch", bound="Schema")
DBS = TypeVar("DBS", bound="Record | Schema")


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

    _target: type["Record"] | None = None

    @property
    def target_type(self) -> type["Record"]:
        """Target record type."""
        assert self._target is not None, "Target record type not set."
        return self._target

    @property
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

    @property
    def join_path(self) -> list[tuple[type["Record"], Mapping["AttrRef", "AttrRef"]]]:
        """Path to join target table."""
        match self.via:
            case RelRef():
                # Relation is defined via other relation or relation table
                other_rel = self.via.rel
                assert isinstance(
                    other_rel, Rel
                ), "Back-reference must be an explicit relation"

                if issubclass(other_rel.target_type, self.record_type):
                    # Supplied record type object is a backlinking relation
                    return [(self.target_type, self.fk_map.inverse)]
                else:
                    # Supplied record type object is a forward relation
                    # on a relation table
                    back_rel = [
                        r
                        for rs in other_rel.record_type._rels.values()
                        for r in rs
                        if issubclass(r.target_type, self.record_type)
                    ][0]

                    return [
                        (back_rel.record_type, back_rel.fk_map.inverse),
                        (other_rel.target_type, other_rel.fk_map),
                    ]
            case type():
                # Relation is defined via relation table
                fwd_rel = [
                    r
                    for rs in self.via._rels.values()
                    for r in rs
                    if issubclass(r.target_type, self.target_type)
                ][0]
                back_rel = [
                    r
                    for rs in self.via._rels.values()
                    for r in rs
                    if issubclass(r.target_type, self.record_type)
                ][0]

                return [
                    (back_rel.record_type, back_rel.fk_map.inverse),
                    (fwd_rel.target_type, fwd_rel.fk_map),
                ]
            case tuple():
                # Relation is defined via back-rel + forward-rel on a relation table.
                back_rel_target, fwd_rel_target = self.via

                assert hasattr(back_rel_target, "_rel") and hasattr(
                    fwd_rel_target, "_rel"
                )
                back_rel: Rel = getattr(back_rel_target, "_rel")
                fwd_rel: Rel = getattr(fwd_rel_target, "_rel")

                return [
                    (back_rel.record_type, back_rel.fk_map.inverse),
                    (fwd_rel.target_type, fwd_rel.fk_map),
                ]
            case _:
                # Relation is defined via foreign key attributes
                return [(self.target_type, self.fk_map)]


@dataclass
class PropRef(Generic[Rec_cov, Val]):
    """Reference to related record type, optionally indexed."""

    rec_type: type[Rec_cov]
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
    def path(self) -> list["RelRef"]:
        """Path from base record type to this relref."""
        parent = self.get_tag(self.rec_type)
        return [
            *(parent.path if parent is not None else []),
            *([self] if isinstance(self, RelRef) else []),
        ]


class AttrRef(sqla.ColumnClause[Val], PropRef[Rec_cov, Val], Generic[Rec_cov, Val]):
    """Reference a property by its containing record type, name and value type."""

    def __init__(  # noqa: D107
        self, rec_type: type[Rec_cov], name: str, val_type: type[Val]
    ) -> None:
        self.rec_type = rec_type
        self.name = name
        self.val_type = val_type

        super().__init__(
            self.name,
            type_=self.record_type._table.c[self.name].type,
        )

    def __getitem__(self, value_type: type[Val2]) -> "AttrRef[Rec_cov, Val2]":
        """Specialize the value type of this attribute."""
        return AttrRef(rec_type=self.rec_type, name=self.name, val_type=value_type)


@dataclass
class RelRef(PropRef[Rec, Rec2], Generic[Rec, Rec2, Idx]):
    """Reference to related record type, optionally indexed."""

    rel: Rel[RecordValue[Rec2]] | None = None
    """Relation prop on source record."""

    idx: type[Idx] | None = None
    """Index type."""

    @cached_property
    def r(self) -> type[Rec2]:
        """Reference props of the target record type."""
        return cast(type[Rec2], type(token_hex(5), (self.val_type,), {"_rel": self}))


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


class a(Record):  # noqa: N801
    """Dynamically defined record type."""

    def __getattribute__(self, name: str) -> Any:
        """Get dynamic attribute by name."""
        return AttrRef(type(self), name, Any)
