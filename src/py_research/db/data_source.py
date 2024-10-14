"""Utilities for importing different data representations into relational format."""

from __future__ import annotations

import asyncio
import operator
from collections.abc import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, fields
from functools import cached_property, reduce
from typing import Any, Literal, Self, cast

import sqlalchemy as sqla
from aioitertools.builtins import zip as azip
from lxml.etree import _ElementTree as ElementTree

from py_research.data import copy_and_override
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.types import SupportsItems, has_type
from py_research.telemetry import tqdm

from .base import DB, Col, LocalStat, Record, RelSet, State
from .conflicts import DataConflictPolicy

type TreeNode = Mapping[str | int, Any] | ElementTree | Hashable
type TreeData = TreeNode | Sequence[TreeNode]


class All:
    """Select all nodes on a level."""


type DirectPath = tuple[str | int | type[All], ...]


def _select_on_level(
    data: TreeData, selector: str | int | type[All]
) -> SupportsItems[DirectPath, Any]:
    """Select attribute from hierarchical data."""
    if isinstance(selector, type):
        assert selector is All
        res = data
    else:
        match data:
            case Mapping():
                res = data[selector]
            case Sequence():
                assert not isinstance(selector, str)
                res = data[selector]
            case ElementTree():
                assert not isinstance(selector, int)
                res = data.findall(selector, namespaces={})
                assert len(res) == 1
                res = res[0]
            case Hashable():
                res = data if data == selector else []

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

    def __truediv__(self, other: PathLevel | DirectPath | TreePath) -> TreePath:
        """Join two paths."""
        if isinstance(other, TreePath):
            return TreePath(list(self.path) + list(other.path))
        if isinstance(other, tuple):
            return TreePath(list(self.path) + list(other))
        return TreePath(list(self.path) + [other])

    def __rtruediv__(self, other: PathLevel | DirectPath) -> TreePath:
        """Join two paths."""
        if isinstance(other, tuple):
            return TreePath(list(other) + list(self.path))
        return TreePath([other] + list(self.path))


type NodeSelector = str | int | TreePath | type[All]
"""Select a node in a hierarchical data structure."""


type _PushMapping[Rec: Record] = SupportsItems[NodeSelector, bool | PushMap[Rec]]

type PushMap[Rec: Record] = _PushMapping[Rec] | Col[
    Any, Any, Any, LocalStat, Rec
] | RelMap | Iterable[Col[Any, Any, Any, LocalStat, Rec] | RelMap]
"""Mapping of hierarchical attributes to record props or other records."""


@dataclass
class DataSelect:
    """Select node for further processing."""

    @staticmethod
    def parse(obj: NodeSelector | DataSelect) -> DataSelect:
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
    Col[Any, Any, Any, LocalStat, Rec] | RelSet[Any, Any, Any, Any, Any, Rec],
    "NodeSelector | DataSelect",
]
type _PullMapping[Rec: Record] = Mapping[
    Col[Any, Any, Any, LocalStat, Rec] | RelSet[Any, Any, Any, Any, Any, Rec],
    DataSelect,
]

type RecMatchBy[Rec: Record] = (
    Literal["index", "all"]
    | Col[Any, Any, Any, LocalStat, Rec]
    | list[Col[Any, Any, Any, LocalStat, Rec]]
)


@dataclass(kw_only=True, eq=False)
class RecMap[Rec: Record, Dat]:
    """Configuration for how to map (nested) dictionary items to relational tables."""

    push: PushMap[Rec] | None = None
    """Mapping of hierarchical attributes to record props or other records."""

    pull: PullMap[Rec] | None = None
    """Mapping of record props to hierarchical attributes."""

    loader: Callable[[Dat], TreeNode | Coroutine[Any, Any, TreeNode]] | None = None
    """Loader function to load data for this record from a source."""

    async_loader: bool = True
    """If True, the loader is an async function."""

    load_by_index: bool = True
    """If True, expect input to loader to match the resulting record's index."""

    match_by: RecMatchBy = "index"
    """Match this data to existing records via given attributes instead of index."""

    conflicts: DataConflictPolicy = "collect"
    """Which policy to use if import conflicts occur for this record table."""

    rec_type: type[Rec] = field(init=False, repr=False, default=Record)

    @cached_property
    def full_map(self) -> _PullMapping[Rec]:
        """Get the full mapping."""
        return {
            **(
                _push_to_pull_map(self.rec_type, self.push)
                if self.push is not None
                else {}
            ),
            **{k: DataSelect.parse(v) for k, v in (self.pull or {}).items()},
        }

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(self)


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


@dataclass(kw_only=True, eq=False)
class SubMap(RecMap, DataSelect):
    """Select and map nested data to another record."""

    sel: NodeSelector = All

    link: RecMap | None = None
    """Mapping to attributes of the link record."""


@dataclass(kw_only=True, eq=False)
class RelMap[Rec: Record, Dat, Rec2: Record](RecMap[Rec, Dat]):
    """Map nested data via a relation to another record."""

    rel: RelSet[Rec, Any, Any, Any, LocalStat, Rec2, Rec]
    """Relation to use for mapping."""

    index: DataSelect | tuple[DataSelect, ...] | None = None
    """Selector to use for indexing the relation set."""

    link: RecMap | None = None
    """Mapping to optional attributes of the link record."""

    def __post_init__(self) -> None:  # noqa: D105
        self.rec_type = self.rel.record_type
        if self.link is not None:
            self.link.rec_type = self.rel.link_type


def _parse_pushmap(push_map: PushMap) -> _PushMapping:
    """Parse push map into a more usable format."""
    match push_map:
        case Mapping() if has_type(push_map, _PushMapping):  # type: ignore
            return push_map
        case Col() | RelMap() | str():
            return {All: push_map}
        case Iterable() if has_type(push_map, Iterable[Col | RelMap]):
            return {
                k.name if isinstance(k, Col) else k.rel.name: True for k in push_map
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
                cast(SubMap, copy_and_override(target, SubMap, sel=sel))
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


type MapTreeData[Dat: TreeNode] = dict[tuple[Hashable | None, DirectPath], Dat]


async def _load_tree_data[
    Rec: Record
](
    rec_type: type[Rec],
    loader: Callable[[Any], TreeNode | Coroutine[Any, Any, TreeNode]],
    async_loader: bool,
    data_input: MapTreeData[Hashable],
    db: DB,
) -> MapTreeData[TreeNode]:
    id_set = set(data_input.values())

    recs = db[rec_type][list(id_set)]
    rec_keys = recs.keys()
    id_set -= set(rec_keys)

    data: dict[Hashable, TreeNode] = {}
    if async_loader:
        async for rel_idx, dat in tqdm(
            azip(
                id_set,
                sliding_batch_map(
                    id_set,
                    cast(Callable[[Any], Coroutine[Any, Any, TreeNode]], loader),
                ),
            ),
            desc=f"Async-loading `{rec_type.__name__}`",
            total=len(id_set),
        ):
            data[rel_idx] = dat
    else:
        data = {
            idx: cast(TreeNode, loader(idx))
            for idx in tqdm(id_set, desc=f"Loading `{rec_type.__name__}`")
        }

    tree_output: MapTreeData[TreeNode] = {
        cast(tuple[Hashable, DirectPath], (rel_idx, path_idx)): data[idx]
        for (rel_idx, path_idx), idx in data_input.items()
    }

    return tree_output


type RecMatchExpr = list[Hashable] | sqla.ColumnElement[bool]


def _gen_match_expr(
    rec_type: type[Record],
    rec_idx: Hashable | None,
    rec_dict: dict[str, Any],
    match_by: RecMatchBy,
) -> RecMatchExpr:
    if match_by == "index":
        assert rec_idx is not None
        return [rec_idx]
    else:
        match_cols = (
            list(rec_type._cols.values())
            if match_by == "all"
            else [match_by] if isinstance(match_by, Col) else match_by
        )
        return reduce(operator.and_, (col == rec_dict[col.name] for col in match_cols))


def _get_record[
    Rec: Record
](
    db: DB,
    rec_map: RecMap[Rec, TreeNode],
    path_idx: DirectPath,
    node: TreeNode,
) -> (
    Hashable | Rec | Literal[State.undef]
):
    attrs = {
        a.name: (a, *list(sel.select(node, path_idx).items())[0])
        for a, sel in rec_map.full_map.items()
        if isinstance(a, Col)
    }
    rec_dict = {a[0].name: a[2] for a in attrs.values()}

    rec = None
    if rec_map.rec_type._is_complete_dict(rec_dict):
        rec = rec_map.rec_type(**rec_dict)

    match_expr = _gen_match_expr(
        rec_map.rec_type,
        rec._index if rec is not None else None,
        rec_dict,
        rec_map.match_by,
    )

    recs_keys = db[rec_map.rec_type][match_expr].keys()
    if len(recs_keys) == 1:
        return recs_keys[0]

    return rec if rec is not None else State.undef


type RestTreeData = dict[RecMap, MapTreeData[TreeData]]


async def _load_records(
    db: DB,
    rec_map: RecMap,
    input_data: MapTreeData[TreeNode],
) -> RestTreeData:
    rest_tree_data: RestTreeData = {}
    if rec_map not in rest_tree_data:
        rest_tree_data[rec_map] = {}

    rec_type = cast(
        type[Record],
        rec_map.rec_type,
    )

    tree_data: MapTreeData[TreeNode]

    if rec_map.loader is not None and has_type(input_data, MapTreeData[Hashable]):
        tree_data = await _load_tree_data(
            rec_type, rec_map.loader, rec_map.async_loader, input_data, db
        )
    else:
        tree_data = input_data

    rest_tree_data |= await _load_rels(db, rec_map, tree_data, "out")

    records: dict[Hashable, Record | Hashable] = {}

    for (parent_idx, path_idx), node in tree_data.items():
        rec_res = _get_record(db, rec_map, path_idx, node)

        if rec_res is State.undef:
            rest_tree_data[rec_map][(parent_idx, path_idx)] = node
            continue

        rec_idx = rec_res._index if isinstance(rec_res, Record) else rec_res

        rel_idx = None
        if isinstance(rec_map, RelMap):
            link_res = None
            if rec_map.link is not None and issubclass(rec_map.rel.link_type, Record):
                link_res = _get_record(db, rec_map.link, path_idx, node)

            if isinstance(rec_map.index, DataSelect):
                rel_idx = list(
                    dict(rec_map.index.select(node, path_idx).items()).values()
                )[0]
            elif isinstance(rec_map.index, tuple):
                rel_idx = tuple(
                    list(dict(sel.select(node, path_idx).items()).values())[0]
                    for sel in rec_map.index
                )

            if rel_idx is None and rec_map.rel.map_by is not None:
                if issubclass(rec_type, rec_map.rel.map_by.parent_type):
                    rec = (
                        rec_res
                        if isinstance(rec_res, Record)
                        else db[rec_type][rec_res]
                    )
                    rel_idx = getattr(rec, rec_map.rel.map_by.name)
                elif (
                    link_res is not None
                    and rec_map.link is not None
                    and issubclass(
                        rec_map.link.rec_type, rec_map.rel.map_by.parent_type
                    )
                ):
                    link_rec = (
                        link_res
                        if isinstance(link_res, Record)
                        else db[rec_map.link.rec_type][link_res].get()
                    )
                    if link_rec is not None:
                        rel_idx = getattr(link_rec, rec_map.rel.map_by.name)

            if rel_idx is None:
                rel_idx = rec_idx
        else:
            rel_idx = rec_idx

        full_idx = (
            *(parent_idx if isinstance(parent_idx, tuple) else [parent_idx]),
            *(rel_idx if isinstance(rel_idx, tuple) else [rel_idx]),
        )
        records[full_idx] = rec_res

    target_set = db[rec_map.rel] if isinstance(rec_map, RelMap) else db[rec_type]
    target_set |= records

    rest_tree_data |= await _load_rels(db, rec_map, tree_data, "in")

    return rest_tree_data


async def _load_rels(
    db: DB,
    rec_map: RecMap[Record, TreeNode],
    tree_data: MapTreeData[TreeNode],
    direction: Literal["out", "in"],
) -> RestTreeData:
    rels = {
        cast(RelSet[Any, Any, Any, Any, Any, Record], rel): sel
        for rel, sel in rec_map.full_map.items()
        if isinstance(rel, RelSet) and isinstance(sel, SubMap)
    }

    rest_tree_data: RestTreeData = {}

    for rel, target_map in rels.items():
        if (direction == "out" and rel._direct_rel is True) or (
            direction == "in" and rel._direct_rel is not True
        ):
            for (parent_idx, path_idx), dat in tree_data.items():
                sub_data = target_map.select(dat, path_idx)
                rel_data = {
                    (parent_idx, (path_idx + item_path)): item_data
                    for item_path, item_data in sub_data.items()
                }
                rest_tree_data |= await _load_records(
                    db,
                    copy_and_override(target_map, RelMap, rel=rel),
                    rel_data,
                )

    return rest_tree_data


@dataclass(kw_only=True, eq=False)
class DataSource[Rec: Record, Dat: TreeNode](RecMap[Rec, Dat]):
    """Root mapping for hierarchical data."""

    target: type[Rec]
    """Root record type to load."""

    def __post_init__(self) -> None:  # noqa: D105
        self.rec_type = self.target

    @cached_property
    def _obj_cache(self) -> dict[type[Record], dict[Hashable, Any]]:
        return {}

    async def load(self, data: Iterable[Dat], db: DB | None = None) -> None:
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
        db = db if db is not None else DB()
        in_data: MapTreeData[TreeNode] = {(None, ()): dat for dat in data}
        rest_data = await _load_records(db, self, in_data)

        for rec_map, tree_data in rest_data.items():
            rest_rest = await _load_records(db, rec_map, tree_data)
            assert all(len(v) == 0 for v in rest_rest.values())
