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

from .base import (
    DB,
    RW,
    BackLink,
    DataSet,
    DynRecord,
    Link,
    One,
    Open,
    Record,
    RefSet,
    Symbolic,
    Value,
)
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


type AttrTarget[Rec: Record] = DataSet[
    Rec, Any, Any, Any, Symbolic, Rec, None, One, None, Value
]

type AttrSetTarget[Rec: Record] = DataSet[
    DynRecord, Any, Any, Any, Symbolic, DynRecord, None, Open, Rec, Value
]

type RelTarget[Rec: Record] = DataSet[
    Any, Any, Any, Any, Symbolic, Any, None, Any, Rec, Any
]

type PropTarget[Rec: Record] = AttrTarget[Rec] | AttrSetTarget[Rec] | RelTarget[Rec]


type _PushMapping[Rec: Record] = SupportsItems[NodeSelector, bool | PushMap[Rec]]

type PushMap[Rec: Record] = _PushMapping[Rec] | PropTarget[
    Rec
] | RefMap | Index | Iterable[PropTarget[Rec] | RefMap | Index]
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


@dataclass
class Index:
    """Map to the index of the record."""

    pos: int | slice = slice(None)
    pks: Iterable[AttrTarget[Record]] | None = None


type PullMap[Rec: Record] = SupportsItems[
    PropTarget[Rec] | Index,
    "NodeSelector | DataSelect",
]
type _PullMapping[Rec: Record] = Mapping[
    PropTarget[Rec] | Index,
    DataSelect,
]

type RecMatchBy[Rec: Record] = (
    Literal["index", "all"] | AttrTarget[Rec] | list[AttrTarget[Rec]]
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

    @cached_property
    def rels(
        self,
    ) -> dict[DataSet[Rec, Any, Any, RW, Symbolic, Rec, Any, Any, Record, Any], SubMap]:
        """Get all relation mappings."""
        return {
            cast(
                DataSet[Rec, Any, Any, RW, Symbolic, Rec, Any, Any, Record, Any], rel
            ): sel
            for rel, sel in self.full_map.items()
            if isinstance(rel, DataSet) and isinstance(sel, SubMap)
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
class RefMap[Rec: Record, Dat, Rec2: Record](RecMap[Rec, Dat]):
    """Map nested data via a relation to another record."""

    ref: DataSet[Rec, Any, Any, RW, Symbolic, Rec, Any, Any, Rec2, Any]
    """Relation to use for mapping."""

    rel: RecMap | None = None
    """Mapping to optional attributes of the link record."""

    def __post_init__(self) -> None:  # noqa: D105
        self.rec_type = self.ref.record_type
        if self.rel is not None:
            self.rel.rec_type = self.ref.rel_type

    @cached_property
    def rel_map(self) -> RefMap | None:
        """Get the link map."""
        if issubclass(self.ref.rel_type, Record) and self.rel is not None:
            return copy_and_override(RefMap, self.rel, ref=self.ref.rels)

        return None


def _parse_pushmap(push_map: PushMap) -> _PushMapping:
    """Parse push map into a more usable format."""
    match push_map:
        case Mapping() if has_type(push_map, _PushMapping):  # type: ignore
            return push_map
        case DataSet() | RefMap() | str() | Index():
            return {All: push_map}
        case Iterable() if has_type(push_map, Iterable[DataSet | RefMap]):
            return {
                k.name if isinstance(k, DataSet) else k.ref.name: True for k in push_map
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
                if isinstance(target, DataSet | Index)
                else getattr(rec, _get_selector_name(sel))
            ): DataSelect(sel)
            for sel, target in mapping.items()
            if isinstance(target, DataSet | Index | bool)
        },
        **{
            (
                target.ref
                if isinstance(target, RefMap) and target.ref is not None
                else getattr(rec, _get_selector_name(sel))
            ): (
                copy_and_override(SubMap, target, sel=sel)
                if isinstance(target, RefMap)
                else DataSelect.parse(sel)
            )
            for sel, targets in mapping.items()
            if has_type(targets, RecMap)
            or (
                has_type(targets, Iterable[RecMap]) and not isinstance(targets, DataSet)
            )
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


async def _sliding_batch_map[
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


type RecMatchExpr = list[Hashable] | sqla.ColumnElement[bool]


def _gen_match_expr(
    rec_set: DataSet[Record, Any, Any, Any, Any, Record, Any, Any, Any, Any],
    rec_idx: Hashable | None,
    rec_dict: dict[str, Any],
    match_by: RecMatchBy,
) -> RecMatchExpr:
    if isinstance(match_by, str) and match_by == "index":
        assert rec_idx is not None
        return [rec_idx]
    else:
        match_cols = (
            list(
                getattr(rec_set.record_type, a.name)
                for a in rec_set.record_type._attrs.values()
            )
            if isinstance(match_by, str) and match_by == "all"
            else [match_by] if isinstance(match_by, DataSet) else match_by
        )
        return reduce(operator.and_, (col == rec_dict[col.name] for col in match_cols))


type InData = dict[tuple[Hashable, DirectPath], TreeNode]
type RefData = list[
    tuple[
        RefMap[Record, TreeNode, Record] | AttrSetTarget[Record],
        InData,
    ]
]
type RestData = list[tuple[RecMap[Record, TreeNode], InData]]


async def _load_record[
    Rec: Record
](
    rec_set: DataSet[Record, Any, Any, Any, Any, Record, Any, Any, Any, Any],
    ref_data: RefData,
    rec_map: RecMap[Any, Any],
    parent_idx: Hashable | None,
    path_idx: DirectPath,
    data: TreeNode,
    inject: dict[str, Any] | None,
) -> tuple[Hashable | Rec | None, Hashable | Rec | None]:
    if rec_map.loader is not None:
        if rec_map.async_loader:
            data = await cast(
                Callable[[Any], Coroutine[Any, Any, TreeNode]],
                rec_map.loader,
            )(data)
        else:
            data = rec_map.loader(data)

    ref_fks: dict[str, Hashable] = {}

    for rel, target_map in rec_map.rels.items():
        if isinstance(rel._prop, Link):
            sub_items = list(target_map.select(data, path_idx).items())
            assert len(sub_items) == 1

            sub_path = sub_items[0][0]
            sub_data = sub_items[0][1]

            sub_rec = await _load_record(
                rec_set[rel], ref_data, target_map, None, sub_path, sub_data, None
            )

            if sub_rec is None:
                return None, None

            ref_fks |= rel._gen_fk_value_map(
                sub_rec._index if isinstance(sub_rec, Record) else sub_rec
            )

    rec_idx: tuple | Hashable | None = None
    indexes = {
        idx: list(sel.select(data, path_idx).items())[0][1]
        for idx, sel in rec_map.full_map.items()
        if isinstance(idx, Index)
    }
    if len(indexes) > 0:
        rec_pks = list(rec_set._pk_cols.values())
        idx_maps = [
            dict(
                zip(
                    (
                        [rec_pks[idx.pos]]
                        if isinstance(idx.pos, int)
                        else (
                            rec_pks[idx.pos]
                            if isinstance(idx.pos, slice)
                            else (
                                [pk for pk in rec_pks if pk in idx.pks]
                                if idx.pks is not None
                                else rec_pks
                            )
                        )
                    ),
                    idx_val if isinstance(idx_val, tuple) else [idx_val],
                )
            )
            for idx, idx_val in indexes.items()
        ]
        idx_map = reduce(lambda x, y: x | y, idx_maps)
        rec_idx = (
            tuple(idx_map[pk] for pk in rec_pks)
            if len(idx_maps) > 1
            else list(idx_map.values())[0]
        )

    parent_rel_fks = {}
    if isinstance(rec_map, RefMap) and isinstance(rec_map.ref._prop, BackLink):
        assert parent_idx is not None
        parent_rel_fks = rec_map.ref._gen_fk_value_map(parent_idx)

    attr_values = {
        a.name: (a, list(sel.select(data, path_idx).items()))
        for a, sel in rec_map.full_map.items()
        if isinstance(a, DataSet)
    }

    attrs = {
        name: (attr, *values[0])
        for name, (attr, values) in attr_values.items()
        if attr._ref is None
    }

    ref_data.extend(
        (
            cast(AttrSetTarget[Record], attr_set),
            {(rec_idx, item_path): item_data for item_path, item_data in values},
        )
        for attr_set, values in attr_values.values()
        if attr_set._ref is not None and attr_set._attr is not None
    )

    rec_dict = {
        **{a[0].name: a[2] for a in attrs.values()},
        **ref_fks,
        **parent_rel_fks,
        **(inject if inject is not None else {}),
    }

    rec: Record | None = None
    is_new: bool = True
    if rec_map.rec_type._is_complete_dict(rec_dict):
        rec = rec_map.rec_type(_db=rec_set._db, **rec_dict)

    match_expr = _gen_match_expr(
        rec_set,
        rec._index if rec is not None else rec_idx if rec_idx is not None else None,
        rec_dict,
        rec_map.match_by,
    )

    recs_keys = rec_set[match_expr].keys()
    if len(recs_keys) == 1:
        rec_idx = recs_keys[0]
        is_new = False

    if rec_idx is None:
        return None, None

    ref_data.extend(
        (
            copy_and_override(RefMap, target_map, ref=ref),
            {
                (rec_idx, item_path): item_data
                for item_path, item_data in target_map.select(data, path_idx).items()
            },
        )
        for ref, target_map in rec_map.rels.items()
    )

    rel: Record | Hashable | None = None
    if isinstance(rec_map, RefMap) and rec_map.rel_map is not None:
        rel, _ = await _load_record(
            rec_set,
            ref_data,
            rec_map.rel_map,
            parent_idx,
            path_idx,
            data,
            inject=rec_map.ref._gen_fk_value_map(rec_idx),
        )

    return rec if rec is not None and is_new else rec_idx, rel


async def _load_records(
    db: DB,
    rec_map: RecMap[Any, Any],
    in_data: InData,
    rest_data: RestData,
) -> dict[Hashable, Record | Hashable]:
    rec_set = db[rec_map.rec_type]
    ref_data: RefData = []

    async def _load_rec_from_item(
        item: tuple[tuple[Hashable, DirectPath], TreeNode]
    ) -> tuple[Hashable | Record | None, Hashable | Record | None]:
        (parent_idx, path_idx), data = item
        return await _load_record(rec_set, [], rec_map, parent_idx, path_idx, data, {})

    loaded = tqdm(
        azip(
            in_data.items(),
            _sliding_batch_map(in_data.items(), _load_rec_from_item),
        ),
        desc=f"Async-loading `{rec_map.rec_type.__name__}`",
        total=len(in_data),
    )

    records: dict[Hashable, Record | Hashable] = {}
    rest_records: InData = {}

    async for ((parent_idx, path_idx), data), (rec, rel) in loaded:
        if rec is None:
            rest_records[(parent_idx, path_idx)] = data
            continue

        rec_idx = rec._index if isinstance(rec, Record) else rec
        rel_idx = (
            *(parent_idx if isinstance(parent_idx, tuple) else [parent_idx]),
            *(rec_idx if isinstance(rec_idx, tuple) else [rec_idx]),
        )

        if isinstance(rec_map, RefMap):
            if rec_map.rel is not None:
                if rel is None:
                    rest_records[(parent_idx, path_idx)] = data
                    continue
                else:
                    rel_idx = rel
            elif (
                isinstance(rec_map.ref._ref, RefSet)
                and rec_map.ref._ref.map_by is not None
            ):
                if issubclass(rec_map.rec_type, rec_map.ref._ref.map_by.parent_type):
                    rec = rec if isinstance(rec, Record) else db[rec_map.rec_type][rec]
                    mapped_idx = getattr(rec, rec_map.ref._ref.map_by.name)
                else:
                    assert rel is not None
                    rel = (
                        rel
                        if isinstance(rel, Record)
                        else db[rec_map.ref.rel_type][rel]
                    )
                    mapped_idx = getattr(rel, rec_map.ref._ref.map_by.name)

                rel_idx = (
                    *(parent_idx if isinstance(parent_idx, tuple) else [parent_idx]),
                    *(mapped_idx if isinstance(mapped_idx, tuple) else [mapped_idx]),
                )

        records[rel_idx] = rec

    # Descend into incoming relations after loading the main records
    for rel_tgt, sub_data in ref_data:
        if isinstance(rel_tgt, RefMap):
            # Full record relation set
            rec_set[rel_tgt.ref] |= await _load_records(
                db,
                rel_tgt,
                sub_data,
                rest_data,
            )
        else:
            # Attribute relation set
            rec_type = cast(type[DynRecord], rel_tgt.record_type)
            ref_map = RefMap[DynRecord, TreeNode, Record](
                pull={
                    rec_type[rel_tgt.attr_set.attr.name]: DataSelect(All),
                    rec_type[rel_tgt.attr_set.idx_attr.name]: SelIndex(All),
                },
                ref=rel_tgt,
            )
            rec_set[rel_tgt] |= await _load_records(
                db,
                ref_map,
                sub_data,
                rest_data,
            )

    if len(rest_records) > 0:
        rest_data.append((rec_map, rest_records))

    return records


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

    async def load(self, data: Iterable[Dat], db: DB | None = None) -> set[Hashable]:
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
        in_data: InData = {((), ()): dat for dat in data}
        rest_data: RestData = []
        loaded = await _load_records(
            db, cast(RecMap[Record, TreeNode], self), in_data, rest_data
        )

        # Perform second pass to load remaining data
        for rec_map, tree_data in rest_data:
            rest_rest: RestData = []
            loaded |= await _load_records(db, rec_map, tree_data, rest_rest)
            assert all(len(v[1]) == 0 for v in rest_rest)

        db[self.rec_type] |= loaded
        return set(loaded.keys())
