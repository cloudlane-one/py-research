"""Utilities for importing different data representations into relational format."""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, Self, cast

from lxml.etree import _ElementTree as ElementTree

from py_research.hashing import gen_str_hash
from py_research.reflect.types import SupportsItems, has_type

from .base import DB, LoadStatus, Record, RelSet, Set, ValueSet
from .conflicts import DataConflictPolicy

type TreeData = Mapping[str | int, Any] | ElementTree | Sequence


class All:
    """Select all nodes on a level."""


type DirectPath = tuple[str | int | type[All], ...]


def _select_on_level(
    data: TreeData, selector: str | int | type[All]
) -> SupportsItems[DirectPath, Any]:
    """Select attribute from hierarchical data."""
    res = LoadStatus.unloaded

    if isinstance(selector, type):
        assert selector is All
        res = data

    if res is LoadStatus.unloaded:
        match data:
            case Mapping():
                assert not isinstance(selector, type)
                res = data[selector]
            case Sequence():
                assert not isinstance(selector, type | str)
                res = data[selector]
            case ElementTree():
                assert not isinstance(selector, type | int)
                res = data.findall(selector, namespaces={})
                assert len(res) == 1
                res = res[0]

    if has_type(res, Mapping[str, Any]):
        return {(selector,): res}
    elif isinstance(res, Mapping):
        return {(selector, k): v for k, v in res.items()}
    elif isinstance(res, Sequence) and not isinstance(res, str):
        return {(selector, i): v for i, v in enumerate(res)}

    return {(selector,): res}


type PathLevel = str | int | slice | set[str] | set[int] | type[All] | Callable[
    [Any], bool
]


@dataclass(frozen=True)
class TreePath:
    """Path to a specific node in a hierarchical data structure."""

    path: Sequence[PathLevel]

    def select(self, data: TreeData) -> SupportsItems[DirectPath, Any]:
        """Select the node in the data structure."""
        res: SupportsItems[DirectPath, Any] = {}
        if len(self.path) == 0:
            res = _select_on_level(data, All)
        else:
            top = self.path[0]
            subpath = TreePath(self.path[1:])
            match top:
                case str() | int() | type():
                    res = _select_on_level(data, top)
                case slice():
                    assert isinstance(data, Sequence)
                    index_range = range(
                        top.start or 0, top.stop or len(data), top.step or 1
                    )
                    res = {
                        (i, *sub_i, *sub_sub_i): v
                        for i in index_range
                        for sub_i, v in _select_on_level(data, i).items()
                        for sub_sub_i, v in subpath.select(v).items()
                    }
                case set():
                    res = {
                        (i, *sub_i, *sub_sub_i): v
                        for i in top
                        for sub_i, v in _select_on_level(data, i).items()
                        for sub_sub_i, v in subpath.select(v).items()
                    }
                case Callable():
                    res = {
                        (top.__name__, *sub_i): v
                        for sub_i, v in subpath.select(
                            list(filter(top, data))
                            if isinstance(data, Iterable) and not isinstance(data, str)
                            else data if top(data) else []
                        ).items()
                    }

        return res

    def __truediv__(self, other: "PathLevel | DirectPath | TreePath") -> "TreePath":
        """Join two paths."""
        if isinstance(other, TreePath):
            return TreePath(list(self.path) + list(other.path))
        if isinstance(other, tuple):
            return TreePath(list(self.path) + list(other))
        return TreePath(list(self.path) + [other])

    def __rtruediv__(self, other: PathLevel | DirectPath) -> "TreePath":
        """Join two paths."""
        if isinstance(other, tuple):
            return TreePath(list(other) + list(self.path))
        return TreePath([other] + list(self.path))


type NodeSelector = str | int | TreePath | type[All]
"""Select a node in a hierarchical data structure."""


type _PushMapping[Rec: Record] = SupportsItems[NodeSelector, bool | PushMap[Rec]]

type PushMap[Rec: Record] = _PushMapping[Rec] | ValueSet[
    Any, Any, None, Rec
] | RelMap | Iterable[ValueSet[Any, Any, None, Rec] | RelMap]
"""Mapping of hierarchical attributes to record props or other records."""


@dataclass
class DataSelect:
    """Select node for further processing."""

    @staticmethod
    def parse(obj: "NodeSelector | DataSelect") -> "DataSelect":
        """Parse the object into an x selector."""
        if isinstance(obj, DataSelect):
            return obj
        return DataSelect(sel=obj)

    sel: NodeSelector
    """Node selector to use."""

    def prefix(self, prefix: NodeSelector) -> Self:
        """Prefix the selector."""
        sel = self.sel if isinstance(self.sel, TreePath) else TreePath([self.sel])
        return type(self)(
            **{
                **{f.name: getattr(self, f.name) for f in fields(self)},
                "sel": prefix / sel,
            }
        )

    def select(
        self, data: TreeData, parent_path: DirectPath = tuple()
    ) -> SupportsItems[DirectPath, Any]:
        """Select the node in the data structure."""
        sel = self.sel
        if not isinstance(sel, TreePath):
            sel = TreePath([sel])

        return sel.select(data)


type PullMap[Rec: Record] = SupportsItems[
    Set[Any, Any, None, Rec], "NodeSelector | DataSelect"
]
type _PullMapping[Rec: Record] = Mapping[Set[Any, Any, None, Rec], DataSelect]


@dataclass(kw_only=True)
class RecMap[Rec: Record, Dat]:
    """Configuration for how to map (nested) dictionary items to relational tables."""

    push: PushMap[Rec]
    """Mapping of hierarchical attributes to record props or other records."""

    pull: PullMap[Rec] | None = None
    """Mapping of record props to hierarchical attributes."""

    loader: Callable[[Dat], TreeData] | None = None
    """Loader function to load data for this record from a source."""

    match: (
        bool | ValueSet[Any, Any, None, Rec] | list[ValueSet[Any, Any, None, Rec]]
    ) = False
    """Try to match this mapped data to target record table (by given attr)
    before creating a new row.
    """

    conflicts: DataConflictPolicy = "raise"
    """Which policy to use if import conflicts occur for this record table."""

    def full_map(self, rec: type[Rec]) -> _PullMapping[Rec]:
        """Get the full mapping."""
        return {
            **_push_to_pull_map(rec, self.push),
            **{k: DataSelect.parse(v) for k, v in (self.pull or {}).items()},
        }


@dataclass
class Transform(DataSelect):
    """Select and transform attributes."""

    func: Callable[[DirectPath, Any], Any]

    def select(
        self, data: TreeData, parent_path: DirectPath = tuple()
    ) -> SupportsItems[DirectPath, Any]:
        """Select the node in the data structure."""
        return {
            p: self.func((*parent_path, *p), v)
            for p, v in DataSelect.select(self, data).items()
        }


@dataclass
class Hash(DataSelect):
    """Hash the selected attribute."""

    with_path: bool = False
    """If True, the id will be generated based on the data and the full tree path.
    """

    def select(
        self, data: TreeData, parent_path: DirectPath = tuple()
    ) -> SupportsItems[DirectPath, str]:
        """Select the node in the data structure."""
        return {
            p: (
                gen_str_hash(((*parent_path, *p), v))
                if self.with_path
                else gen_str_hash(v)
            )
            for p, v in DataSelect.select(self, data).items()
        }


@dataclass
class SelIndex(DataSelect):
    """Select and transform attributes."""

    sel: NodeSelector = All
    levels_up: int = 1

    def select(
        self, data: TreeData, parent_path: DirectPath = tuple()
    ) -> SupportsItems[DirectPath, Any]:
        """Select the node in the data structure."""
        idx_sel = slice(-self.levels_up) if self.levels_up > 1 else -1
        res = {
            p: ((*parent_path, *(pi for pi in p if pi is not All))[idx_sel])
            for p, _ in DataSelect.select(self, data).items()
        }
        return res


@dataclass(kw_only=True)
class SubMap(RecMap, DataSelect):
    """Select and map nested data to another record."""

    sel: NodeSelector = All

    link: "RecMap | None" = None
    """Mapping to optional attributes of the link record."""


@dataclass(kw_only=True)
class RelMap[Rec: Record, Rec2: Record, Dat](RecMap[Rec2, Dat]):
    """Map nested data via a relation to another record."""

    rel: RelSet[Rec2, Any, None, Any, Rec]
    """Relation to use for mapping."""

    link: "RecMap | None" = None
    """Mapping to optional attributes of the link record."""


def _parse_pushmap(push_map: PushMap) -> _PushMapping:
    """Parse push map into a more usable format."""
    match push_map:
        case Mapping() if has_type(push_map, _PushMapping):  # type: ignore
            return push_map
        case ValueSet() | RelMap() | str():
            return {All: push_map}
        case Iterable() if has_type(push_map, Iterable[ValueSet | RelMap]):
            return {
                k.prop.name if isinstance(k, ValueSet) else k.rel.prop.name: True
                for k in push_map
            }
        case _:
            raise TypeError(f"Unsupported mapping type {type(push_map)}")


def _get_selector_name(selector: NodeSelector) -> str:
    """Get the name of the selector."""
    match selector:
        case str():
            return selector
        case int() | TreePath() | type():
            raise ValueError(f"Cannot get name of selector with type {type(selector)}.")


def _push_to_pull_map(rec: type[Record], push_map: PushMap) -> _PullMapping:
    """Extract hierarchical data into set of scalar attributes + ref'd data objects."""
    mapping = _parse_pushmap(push_map)

    # First list and handle all scalars, hence data attributes on the current level,
    # which are to be mapped to record attributes.
    pull_map: _PullMapping = {
        **{
            (
                target
                if isinstance(target, ValueSet)
                else getattr(rec, _get_selector_name(sel))
            ): DataSelect(sel)
            for sel, target in mapping.items()
            if isinstance(target, ValueSet | bool)
        },
        **{
            (
                target.rel
                if isinstance(target, RelMap) and target.rel is not None
                else getattr(rec, _get_selector_name(sel))
            ): (
                SubMap(
                    **{
                        **{
                            f.name: getattr(target, f.name)
                            for f in fields(target)
                            if f in fields(RecMap)
                        },
                        "sel": sel,
                    }
                )
                if isinstance(target, RelMap)
                else DataSelect.parse(sel)
            )
            for sel, targets in mapping.items()
            if has_type(targets, RecMap) or has_type(targets, Iterable[RecMap])
            for target in ([targets] if isinstance(targets, RecMap) else targets)
        },
    }

    # Handle nested data attributes (which come as dict types).
    for sel, target in mapping.items():
        if has_type(target, Mapping):
            sub_pull_map = _push_to_pull_map(
                rec,
                target,
            )
            pull_map = {
                **pull_map,
                **{prop: sub_sel.prefix(sel) for prop, sub_sel in sub_pull_map.items()},
            }

    return pull_map


def _map_record[  # noqa: C901
    Rec: Record, Dat
](
    rec_type: type[Rec],
    xmap: RecMap[Rec, Dat],
    in_data: Dat,
    py_cache: dict[type[Record], dict[Hashable, Any]],
    db_cache: DB | None = None,
    path: DirectPath = tuple(),
) -> Rec:
    """Map a data source to a record."""
    if rec_type not in py_cache:
        py_cache[rec_type] = {}

    data: TreeData
    if xmap.loader is not None:
        if db_cache is not None and isinstance(in_data, Hashable):
            # Try to load from cache directly.
            recs = db_cache[rec_type][[in_data]].load()
            if len(recs) == 1:
                rec = next(iter(recs.values()))
                py_cache[rec_type][in_data] = rec
                return rec
        # Load data from source and continue mapping.
        data = xmap.loader(in_data)
    else:
        if not has_type(in_data, TreeData):  # type: ignore
            raise ValueError(f"Supplied data has unsupported type {type(in_data)}")
        data = in_data

    mapping = xmap.full_map(rec_type)

    attrs = {
        a.prop.name: (a, *list(sel.select(data, path).items())[0])
        for a, sel in mapping.items()
        if isinstance(a, ValueSet)
    }

    rec_dict: dict[Set, Any] = {a[0]: a[2] for a in attrs.values()}
    rec = rec_type._from_partial_dict(rec_dict)

    if rec._index in py_cache[rec_type]:
        return py_cache[rec_type][rec._index]

    if db_cache is not None:
        recs = db_cache[rec_type][[rec._index]].load()
        if len(recs) == 1:
            rec = next(iter(recs.values()))
            py_cache[rec_type][rec._index] = rec
            return rec

    rels = {
        cast(RelSet[Any, Any, Any, Any, Rec], rel): sel
        for rel, sel in mapping.items()
        if isinstance(rel, RelSet) and isinstance(sel, SubMap)
    }

    # Handle nested data, which is to be extracted into separate records and referenced.
    for rel, target_map in rels.items():
        sub_data = target_map.select(data, path)
        target_type = rel.record_type

        for sub_path, sub_data in sub_data.items():
            target_rec = _map_record(
                target_type,
                target_map,
                sub_data,
                py_cache,
                db_cache,
                path + sub_path,
            )

            if issubclass(rel.link_type, Record) and target_map.link is not None:
                link_rec = _map_record(
                    rel.link_type,
                    target_map.link,
                    sub_data,
                    py_cache,
                    db_cache,
                    path + sub_path,
                )
            else:
                link_rec = None

            if rel.direct_rel is True:
                setattr(rec, rel.prop.name, target_rec)
            elif rel.prop.map_by is not None:
                idx = (
                    getattr(target_rec, rel.prop.map_by.name)
                    if issubclass(target_type, rel.prop.map_by.parent_type)
                    else getattr(link_rec, rel.prop.map_by.name)
                )
                setattr(
                    rec,
                    rel.prop.name,
                    {
                        **rec._to_dict(name_keys=True).get(rel.prop.name, {}),
                        idx: target_rec,
                    },
                )
            else:
                setattr(
                    rec,
                    rel.prop.name,
                    [*rec._to_dict(name_keys=True).get(rel.prop.name, []), target_rec],
                )

    py_cache[rec_type][rec._index] = rec

    return rec


@dataclass(kw_only=True)
class DataSource[Rec: Record, Dat](RecMap[Rec, Dat]):
    """Root mapping for hierarchical data."""

    rec: type[Rec]
    """Root record type to load."""

    cache: DB | bool = True
    """Use a database for caching loaded data."""

    @cached_property
    def db(self) -> DB:
        """Automatically created DB instance for caching."""
        return self.cache if isinstance(self.cache, DB) else DB()

    def load(self, input_data: Dat) -> Rec:
        """Parse recursive data from a data source.

        Args:
            input_data:
                Input for the data source

        Returns:
            A Record instance
        """
        py_cache = {}
        db_cache = self.db if self.cache is not False else None

        rec = _map_record(self.rec, self, input_data, py_cache, db_cache)

        if db_cache is not None:
            db_cache[type(rec)] |= rec

        return rec
