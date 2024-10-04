"""Utilities for importing different data representations into relational format."""

import asyncio
from collections.abc import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, Self, cast

from aioitertools.builtins import zip as azip
from lxml.etree import _ElementTree as ElementTree

from py_research.hashing import gen_str_hash
from py_research.reflect.types import SupportsItems, has_type

from .base import DB, Col, Record, RelSet, State
from .conflicts import DataConflictPolicy

type TreeData = Mapping[str | int, Any] | ElementTree | Sequence


class All:
    """Select all nodes on a level."""


type DirectPath = tuple[str | int | type[All], ...]


def _select_on_level(
    data: TreeData, selector: str | int | type[All]
) -> SupportsItems[DirectPath, Any]:
    """Select attribute from hierarchical data."""
    res = State.undef

    if isinstance(selector, type):
        assert selector is All
        res = data

    if res is State.undef:
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

type PushMap[Rec: Record] = _PushMapping[Rec] | Col[
    Any, Any, Any, Record, Rec
] | RelMap | Iterable[Col[Any, Any, Any, Record, Rec] | RelMap]
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
    Col[Any, Any, Any, Record, Rec] | RelSet[Any, Any, Any, Any, Any, Rec],
    "NodeSelector | DataSelect",
]
type _PullMapping[Rec: Record] = Mapping[
    Col[Any, Any, Any, Record, Rec] | RelSet[Any, Any, Any, Any, Any, Rec], DataSelect
]


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
        bool | Col[Any, Any, Any, Record, Rec] | list[Col[Any, Any, Any, Record, Rec]]
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

    rel: RelSet[Rec2, Any, Any, Any, Record, Rec, Rec2]
    """Relation to use for mapping."""

    link: "RecMap | None" = None
    """Mapping to optional attributes of the link record."""


def _parse_pushmap(push_map: PushMap) -> _PushMapping:
    """Parse push map into a more usable format."""
    match push_map:
        case Mapping() if has_type(push_map, _PushMapping):  # type: ignore
            return push_map
        case Col() | RelMap() | str():
            return {All: push_map}
        case Iterable() if has_type(push_map, Iterable[Col | RelMap]):
            return {
                k.attr.name if isinstance(k, Col) else k.rel.name: True
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
                if isinstance(target, Col)
                else getattr(rec, _get_selector_name(sel))
            ): DataSelect(sel)
            for sel, target in mapping.items()
            if isinstance(target, Col | bool)
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


async def sliding_batch_map[
    H: Hashable, T
](
    data: Iterable[H],
    func: Callable[[H], Coroutine[Any, Any, T]],
    concurrency_limit: int = 1000,
    wait_time: float = 0.001,
) -> AsyncGenerator[T]:
    """Sliding batch loop for async functions.

    Args:
        data:
            Data to load
        func:
            Function to load data
        concurrency_limit:
            Maximum number of concurrent loads
        wait_time:
            Time in seconds to wait between checks for free slots
    Returns:
        List with all loaded data
    """
    done: set[asyncio.Task[T]] = set()
    running: set[asyncio.Task[T]] = set()

    # Dispatch tasks and collect results simultaneously
    for idx in data:
        # Wait for a free slot
        while len(running) >= concurrency_limit:
            await asyncio.sleep(wait_time)

            # Filter done tasks
            done |= {t for t in running if t.done()}
            running -= done

            # Yield results
            for t in done:
                yield t.result()

        # Start new task once a slot is free
        running.add(asyncio.create_task(func(idx)))

    # Wait for all tasks to finish
    while len(running) > 0:
        new_done, running = await asyncio.wait(
            running, return_when=asyncio.FIRST_COMPLETED
        )
        for t in new_done:
            yield t.result()

    return


async def _load_record[  # noqa: C901
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
            recs = db_cache[rec_type][[in_data]]
            if len(recs) == 1:
                rec = next(iter(recs))
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
        a.attr.name: (a, *list(sel.select(data, path).items())[0])
        for a, sel in mapping.items()
        if isinstance(a, Col)
    }

    rec_dict: dict[Col | RelSet, Any] = {a[0]: a[2] for a in attrs.values()}
    rec = rec_type._from_partial_dict(rec_dict)

    if rec._index in py_cache[rec_type]:
        return py_cache[rec_type][rec._index]

    if db_cache is not None:
        recs = db_cache[rec_type][[rec._index]]
        if len(recs) == 1:
            rec = next(iter(recs))
            py_cache[rec_type][rec._index] = rec
            return rec

    rels = {
        cast(RelSet[Any, Any, Any, Any, Any, Rec], rel): sel
        for rel, sel in mapping.items()
        if isinstance(rel, RelSet) and isinstance(sel, SubMap)
    }

    # Handle nested data, which is to be extracted into separate records and referenced.
    for rel, target_map in rels.items():
        sub_data = target_map.select(data, path)
        target_type = rel.record_type

        rec_collection = None
        if rel.direct_rel is not True:
            rec_collection = {}

        async def load_record(item: tuple[DirectPath, Any]) -> Record:
            sub_path, sub_data = item
            return await _load_record(
                target_type,
                target_map,
                sub_data,
                py_cache,
                db_cache,
                path + sub_path,
            )

        async for (sub_path, sub_data), target_rec in azip(
            sub_data.items(), sliding_batch_map(sub_data.items(), load_record)
        ):
            if issubclass(rel.link_type, Record) and target_map.link is not None:
                link_rec = await _load_record(
                    rel.link_type,
                    target_map.link,
                    sub_data,
                    py_cache,
                    db_cache,
                    path + sub_path,
                )
            else:
                link_rec = None

            if rec_collection is None:
                setattr(rec, rel.name, target_rec)
            else:
                idx = (
                    (
                        getattr(target_rec, rel.map_by.name)
                        if issubclass(target_type, rel.map_by.parent_type)
                        else getattr(link_rec, rel.map_by.name)
                    )
                    if rel.map_by is not None
                    else rec._index
                )
                rec_collection[idx] = target_rec

        if rec_collection is not None:
            setattr(rec, rel.name, rec_collection)

    py_cache[rec_type][rec._index] = rec

    return rec


@dataclass(kw_only=True)
class DataSource[Rec: Record, Dat](RecMap[Rec, Dat]):
    """Root mapping for hierarchical data."""

    rec: type[Rec]
    """Root record type to load."""

    @cached_property
    def _obj_cache(self) -> dict[type[Record], dict[Hashable, Any]]:
        return {}

    async def load(
        self, data: Dat, db: DB | None = None, cache_with_db: bool = True
    ) -> Rec:
        """Parse recursive data from a data source.

        Args:
            data:
                Input for the data source
            db:
                Database to insert the data into (and use for caching)
            cache_with_db:
                If True, use the database as cache

        Returns:
            A Record instance
        """
        db_cache = (
            db if cache_with_db and db is not None else DB() if cache_with_db else None
        )

        rec = await _load_record(self.rec, self, data, self._obj_cache, db_cache)

        if db_cache is not None:
            db_cache[type(rec)] |= rec
        elif db is not None:
            db[type(rec)] |= rec

        return rec
