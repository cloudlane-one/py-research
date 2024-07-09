"""Utilities for importing different data representations into relational format."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import reduce
from itertools import chain
from typing import Any, Generic, Literal, Self, TypeAlias, TypeVar, overload

from lxml.etree import _ElementTree as ElementTree

from py_research.hashing import gen_str_hash
from py_research.reflect.types import has_type

from .base import Backend, DataBase, DataSet, Name
from .conflicts import DataConflictError, DataConflictPolicy, DataConflicts
from .schema import Attr, Prop, Rec, Rec2, Record, Rel, Schema

TreeData: TypeAlias = Mapping[str | int, Any] | ElementTree | Sequence

Dat = TypeVar("Dat")


class All:
    """Select all nodes on a level."""


def _select_on_level(data: TreeData, selector: str | int | slice | type[All]) -> list:
    """Select attribute from hierarchical data."""
    if isinstance(selector, type):
        assert selector is All
        if isinstance(data, Iterable):
            return list(data)
        else:
            return [data]

    match data:
        case Mapping():
            assert not isinstance(selector, slice)
            return [data.get(selector)]
        case Sequence():
            match selector:
                case int() | slice():
                    res = data[selector]
                    return list(res) if isinstance(res, Sequence) else [res]
                case str():
                    return list(
                        chain(*(_select_on_level(item, selector) for item in data))
                    )
        case ElementTree():
            assert not isinstance(selector, int | slice)
            return data.findall(selector, namespaces={})


PathLevel: TypeAlias = (
    str | int | slice | set[str] | set[int] | type[All] | Callable[[Any], bool]
)


@dataclass(frozen=True)
class TreePath:
    """Path to a specific node in a hierarchical data structure."""

    path: Iterable[PathLevel]

    def select(self, data: TreeData) -> list:
        """Select the node in the data structure."""
        node = data if isinstance(data, list) else [data]
        for part in self.path:
            match part:
                case str() | int() | slice():
                    node = _select_on_level(node, part)
                case set():
                    node = list(chain(*(_select_on_level(node, p) for p in part)))
                case type():
                    assert part is All
                    node = node
                case Callable():
                    node = (
                        list(filter(part, node))
                        if isinstance(node, list)
                        else node if part(node) else []
                    )
                case _:
                    raise ValueError(f"Unsupported path part {part}")

        return node

    def __truediv__(self, other: "PathLevel | TreePath") -> "TreePath":
        """Join two paths."""
        if isinstance(other, TreePath):
            return TreePath(list(self.path) + list(other.path))
        return TreePath(list(self.path) + [other])

    def __rtruediv__(self, other: PathLevel) -> "TreePath":
        """Join two paths."""
        return TreePath([other] + list(self.path))


NodeSelector: TypeAlias = str | int | TreePath | type[All]
"""Select a node in a hierarchical data structure."""


_PushMapping: TypeAlias = Mapping[NodeSelector, "bool | XMap | PushMap[Rec]"]

PushMap: TypeAlias = (
    _PushMapping[Rec]
    | set["Attr | RelMap"]
    | Attr[Rec, Any]
    | Callable[[TreeData | str], "PushMap[Rec]"]
)
"""Mapping of hierarchical attributes to record props or other records."""


@dataclass(frozen=True)
class XSelect:
    """Select node for further processing."""

    @staticmethod
    def parse(obj: "NodeSelector | XSelect") -> "XSelect":
        """Parse the object into an x selector."""
        if isinstance(obj, XSelect):
            return obj
        return XSelect(sel=obj)

    sel: NodeSelector
    """Node selector to use."""

    def prefix(self, prefix: NodeSelector) -> Self:
        """Prefix the selector."""
        sel = self.sel if isinstance(self.sel, TreePath) else TreePath([self.sel])
        return type(self)(**asdict(self), sel=prefix / sel)

    def select(self, data: TreeData) -> list:
        """Select the node in the data structure."""
        match self.sel:
            case str() | int() | slice() | type():
                return _select_on_level(data, self.sel)
            case TreePath():
                return self.sel.select(data)


PullMap: TypeAlias = Mapping[Prop[Rec, Any], "NodeSelector | XSelect"]
_PullMapping: TypeAlias = Mapping[Prop[Rec, Any], XSelect]


@dataclass(frozen=True)
class XMap(Generic[Rec, Dat]):
    """Configuration for how to map (nested) dictionary items to relational tables."""

    push: PushMap[Rec]
    """Mapping of hierarchical attributes to record props or other records."""

    pull: PullMap[Rec] | None = None
    """Mapping of record props to hierarchical attributes."""

    load: Callable[[Dat], TreeData] | None = None
    """Loader function to load data for this record from a source."""

    match: bool | list[Attr[Rec, Any]] = False
    """Try to match this mapped data to target record table (by given attr)
    before creating a new row.
    """

    conflicts: DataConflictPolicy = "raise"
    """Which policy to use if import conflicts occur for this record table."""

    def full_map(self, rec: type[Rec], data: TreeData) -> _PullMapping[Rec]:
        """Get the full mapping."""
        return {
            **_push_to_pull_map(rec, self.push, data),
            **{k: XSelect.parse(v) for k, v in (self.pull or {}).items()},
        }


@dataclass(frozen=True)
class Transform(XSelect):
    """Select and transform attributes."""

    func: Callable

    def select(self, data: TreeData) -> list:
        """Select the node in the data structure."""
        return [self.func(v) for v in XSelect.select(self, data)]


@dataclass(frozen=True)
class Hash(XSelect):
    """Hash the selected attribute."""

    with_path: bool = False
    """If True, the id will be generated based on the data and the full tree path.
    """

    def select(self, data: TreeData) -> list[str]:
        """Select the node in the data structure."""
        value_hashes = [gen_str_hash(v) for v in XSelect.select(self, data)]
        return (
            [gen_str_hash((v, self.sel)) for v in value_hashes]
            if self.with_path
            else value_hashes
        )


@dataclass(frozen=True)
class SubMap(XMap, XSelect):
    """Select and map nested data to another record."""

    link: "RootMap | None" = None
    """Mapping to optional attributes of the link record."""


@dataclass(frozen=True, kw_only=True)
class RelMap(XMap[Rec2, Dat], Generic[Rec, Rec2, Dat]):
    """Map nested data via a relation to another record."""

    rel: Rel[Rec, Any, Rec2]
    """Relation to use for mapping."""

    link: XMap | None = None
    """Mapping to optional attributes of the link record."""


@dataclass(frozen=True, kw_only=True)
class RootMap(XMap[Rec, Dat]):
    """Root mapping for hierarchical data."""

    rec: type[Rec]


def _parse_pushmap(push_map: PushMap, data: TreeData) -> _PushMapping:
    """Parse push map into a more usable format."""
    match push_map:
        case Mapping():
            return push_map
        case str():
            return {All: push_map}
        case set():
            return {
                k.name if isinstance(k, Attr) else k.rel.name: True for k in push_map
            }
        case Callable():
            return _parse_pushmap(push_map(data), data)
        case _:
            raise TypeError(
                f"Unsupported mapping type {type(push_map)}"
                f" for data of type {type(data)}"
            )


def _get_selector_name(selector: NodeSelector) -> str:
    """Get the name of the selector."""
    match selector:
        case str():
            return selector
        case int() | TreePath() | type():
            raise ValueError(f"Cannot get name of selector with type {type(selector)}.")


def _push_to_pull_map(
    rec: type[Record], push_map: PushMap, node: TreeData
) -> _PullMapping:
    """Extract hierarchical data into set of scalar attributes + ref'd data objects."""
    mapping = _parse_pushmap(push_map, node)

    # First list and handle all scalars, hence data attributes on the current level,
    # which are to be mapped to record attributes.
    pull_map: _PullMapping = {
        **{
            (
                target
                if isinstance(target, Attr)
                else getattr(rec, _get_selector_name(sel))
            ): XSelect(sel)
            for sel, target in mapping.items()
            if isinstance(target, Attr | bool)
        },
        **{
            (
                target.rel
                if isinstance(target, RelMap) and target.rel is not None
                else getattr(rec, _get_selector_name(sel))
            ): (
                SubMap(**asdict(target), sel=sel)
                if isinstance(target, RelMap)
                else XSelect.parse(sel)
            )
            for sel, target in mapping.items()
            if isinstance(target, XMap)
        },
    }

    # Handle nested data attributes (which come as dict types).
    for sel, target in mapping.items():
        if has_type(target, Mapping):
            sub_node = XSelect.parse(sel).select(node)
            if has_type(sub_node, TreeData):
                sub_pull_map = _push_to_pull_map(
                    rec,
                    target,
                    sub_node,
                )
                pull_map = {
                    **pull_map,
                    **{
                        prop: sub_sel.prefix(sel)
                        for prop, sub_sel in sub_pull_map.items()
                    },
                }

    return pull_map


def _map_record(  # noqa: C901
    db: DataBase,
    rec: type[Rec],
    xmap: XMap[Rec, Dat],
    in_data: Dat,
    collect_conflicts: bool = False,
) -> tuple[dict[Attr[Rec, Any], Any], dict[Attr, Any] | None, DataConflicts]:
    """Map a data record to its relational representation."""
    conflicts = {}

    data: TreeData
    if xmap.load is not None:
        data = xmap.load(in_data)
    else:
        if not has_type(in_data, TreeData):
            raise ValueError(f"Supplied data has unsupported type {type(in_data)}")
        data = in_data

    mapping = xmap.full_map(rec, data)

    attrs = {
        a: sel.select(data)[0] for a, sel in mapping.items() if isinstance(a, Attr)
    }

    rels = {
        r: sel
        for r, sel in mapping.items()
        if isinstance(r, Rel) and isinstance(sel, SubMap)
    }

    # Handle nested data, which is to be extracted into separate tables and referenced.
    for rel, sub_map in rels.items():
        sub_data_items = sub_map.select(data)
        rel_rec = rel.target_type

        for sub_data in sub_data_items:
            sub_attrs, sub_link_attrs, sub_conflicts = _map_record(
                db, rel_rec, sub_map, sub_data, collect_conflicts
            )
            conflicts = {**conflicts, **sub_conflicts}

            # Get the foreign keys via `rel.fk_map` and
            # use `rel.fk_record_type` to determine where to insert them.
            if issubclass(rec, rel.fk_record_type):
                # - Case 1: Insert into attrs of current record.
                for fk_attr, fk in rel.fk_map.items():
                    attrs[fk_attr] = sub_attrs[fk]
            elif issubclass(rel_rec, rec):
                # - Case 2: Fks are already included in linked record (backlink).
                pass
            else:
                # - Case 3: Create record in link table and insert there.
                sub_link_attrs = sub_link_attrs or {}

                for _, fk_maps in rel.inter_joins.items():
                    for fk_map in fk_maps:
                        for fk_attr, fk in fk_map.items():
                            sub_link_attrs[fk_attr] = attrs[fk]

                for fk_map in rel.joins:
                    for fk_attr, fk in rel.fk_map.items():
                        sub_link_attrs[fk_attr] = sub_attrs[fk]

                db[rel.fk_record_type] <<= sub_link_attrs

    link_map = xmap.link if isinstance(xmap, SubMap) else None
    link_attrs = None
    if link_map is not None:
        link_attrs, _, link_conflicts = _map_record(
            db, rec, link_map, data, collect_conflicts
        )
        conflicts = {**conflicts, **link_conflicts}

    if len(conflicts) > 0:
        raise DataConflictError(conflicts)

    if xmap.match:
        # Match against existing records in the database and update them.
        match_attrs = attrs if xmap.match is True else {a: attrs[a] for a in xmap.match}
        match_expr = reduce(
            lambda x, y: x & y, [a == v for a, v in match_attrs.items()]
        )
        existing: DataSet = db[rec][match_expr]

        if len(existing) > 1:
            for a, v in attrs.items():
                existing[a] = v
    else:
        # Do an index-based upsert.
        db[rec] <<= attrs

    return attrs, link_attrs, conflicts


@overload
def tree_to_db(
    data: TreeData,
    mapping: RootMap,
    backend: Backend[Name] = ...,  # type: ignore
    schema: type[Schema] | None = ...,
    collect_conflicts: Literal[True] = ...,
) -> tuple[DataBase[Name], DataConflicts]: ...


@overload
def tree_to_db(
    data: TreeData,
    mapping: RootMap,
    backend: Backend[Name] = ...,  # type: ignore
    schema: type[Schema] | None = ...,
    collect_conflicts: Literal[False] = ...,
) -> DataBase[Name]: ...


def tree_to_db(
    data: TreeData,
    mapping: RootMap,
    backend: Backend[Name] = Backend("main"),
    schema: type[Schema] | None = None,
    collect_conflicts: bool = False,
) -> DataBase[Name] | tuple[DataBase[Name], DataConflicts]:
    """Transform recursive data into relational format.

    Args:
        data: The data to be transformed
        mapping:
            Configuration for how to performm the mapping.
        backend:
            The backend to use for the database.
        schema:
            The schema to use for the database.
        collect_conflicts:
            Collect all conflicts and return them, rather than stopping right away.

    Returns:
        The relational database representation of the data.
        If ``collect_conflicts`` is ``True``, a tuple of the database and the conflicts
        is returned.
    """
    db = DataBase(backend, schema)

    _, _, conflicts = _map_record(db, mapping.rec, mapping, data, collect_conflicts)

    return (db, conflicts) if collect_conflicts else db
