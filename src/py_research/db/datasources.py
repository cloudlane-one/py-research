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

from .conflicts import DataConflictPolicy
from .databases import (
    Array,
    BackLink,
    Ctx,
    Data,
    DataBase,
    Idx,
    Item,
    Link,
    Record,
    Symbolic,
    Table,
    Value,
)

type TreeNode = Mapping[str | int, Any] | ElementTree | Hashable
type TreeData = TreeNode | Sequence[TreeNode]


class All:
    """Select all nodes on a level."""


type DirectPath = tuple[str | int | type[All], ...]


def _select_on_level(
    data: TreeData, selector: str | int | type[All], parent_path: DirectPath = ()
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
        return {
            (
                *parent_path,
                selector,
            ): res
        }
    elif isinstance(res, Mapping):
        return {(*parent_path, selector, k): v for k, v in res.items()}
    elif isinstance(res, Sequence) and not isinstance(res, str):
        return {(*parent_path, selector, i): v for i, v in enumerate(res)}

    return {(*parent_path, selector): res}


type PathLevel = str | int | slice | set[str] | set[int] | type[All] | Callable[
    [Any], bool
]


@dataclass(frozen=True)
class TreePath:
    """Path to a specific node in a hierarchical data structure."""

    path: Sequence[PathLevel]

    def select(
        self, data: TreeData, parent_path: DirectPath = ()
    ) -> SupportsItems[DirectPath, Any]:
        """Select the node in the data structure."""
        res: SupportsItems[DirectPath, Any] = {}
        if len(self.path) == 0:
            res = _select_on_level(data, All, parent_path)
        else:
            top = self.path[0]
            subpath = TreePath(self.path[1:])
            match top:
                case str() | int() | type():
                    res = _select_on_level(data, top, parent_path)
                case slice():
                    assert isinstance(data, Sequence)
                    index_range = range(
                        top.start or 0, top.stop or len(data), top.step or 1
                    )
                    res = {
                        (i, *sub_i, *sub_ij): v_ij
                        for i in index_range
                        for sub_i, v_i in _select_on_level(data, i, parent_path).items()
                        for sub_ij, v_ij in subpath.select(v_i).items()
                    }
                case set():
                    res = {
                        (i, *sub_i, *sub_ij): v_ij
                        for i in top
                        for sub_i, v_i in _select_on_level(data, i, parent_path).items()
                        for sub_ij, v_ij in subpath.select(v_i).items()
                    }
                case Callable():
                    res = {
                        (*parent_path, top.__name__, *sub_i): v
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


type ValueTarget[Val, Rec: Record] = Data[Val, Idx[()], Any, Any, Ctx[Rec], Symbolic]
type TableTarget[Rec: Record] = Data[Rec, Any, Any, Any, Any, Symbolic]
type ArrayTarget[Val, Rec: Record] = Data[Val, Idx, Any, Item, Ctx[Rec], Symbolic]

type PropTarget[Rec: Record] = ValueTarget[Any, Rec] | TableTarget[Rec] | ArrayTarget[
    Any, Rec
]


type _PushMapping[Rec: Record] = SupportsItems[NodeSelector, bool | PushMap[Rec]]

type PushMap[Rec: Record] = _PushMapping[Rec] | PropTarget[
    Rec
] | SubTableMap | Index | Iterable[ValueTarget[Any, Rec] | SubTableMap | Index]
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
        self, data: TreeData, parent_path: DirectPath = ()
    ) -> SupportsItems[DirectPath, Any]:
        """Select the node in the data structure."""
        sel = self.sel
        if not isinstance(sel, TreePath):
            sel = TreePath([sel])

        return sel.select(data, parent_path)


@dataclass
class Index:
    """Map to the index of the record."""

    pos: int | slice = slice(None)
    pks: Iterable[ValueTarget[Any, Record]] | None = None


type PullMap[Rec: Record] = SupportsItems[
    PropTarget[Rec] | Index,
    "NodeSelector | DataSelect",
]
type _PullMapping[Rec: Record] = Mapping[
    PropTarget[Rec] | Index,
    DataSelect,
]

type RecMatchBy[Rec: Record] = (Literal["index", "all"] | list[ValueTarget[Any, Rec]])


@dataclass(kw_only=True, eq=False)
class TableMap[Rec: Record, Dat]:
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

    target_type: type[Rec] | None = None
    """Record type to use for this mapping."""

    @property
    def record_type(self) -> type[Rec]:
        """Get the record type."""
        assert self.target_type is not None
        return self.target_type

    @cached_property
    def full_map(self) -> _PullMapping[Rec]:
        """Get the full mapping."""
        return {
            **(
                _push_to_pull_map(self.record_type, self.push)
                if self.push is not None
                else {}
            ),
            **{
                tgt: (
                    copy_and_override(SubMap, sel, target_type=tgt.record_type)
                    if isinstance(sel, SubMap) and isinstance(tgt, Data)
                    else DataSelect.parse(sel)
                )
                for tgt, sel in (self.pull or {}).items()
            },
        }

    @cached_property
    def sub_tables(
        self,
    ) -> dict[TableTarget | ArrayTarget, SubMap]:
        """Get all relation mappings."""
        return {
            cast(Data[Any, Any, Any, Any, Ctx[Rec], Symbolic], rel): sel
            for rel, sel in self.full_map.items()
            if isinstance(rel, Table) and isinstance(sel, SubMap)
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
        self, data: TreeData, parent_path: DirectPath = ()
    ) -> SupportsItems[DirectPath, Any]:
        """Select the node in the data structure."""
        idx_sel = slice(-self.levels_up, None) if self.levels_up > 1 else -1
        res = {
            p: ((*parent_path, *(pi for pi in p if pi is not All))[idx_sel])
            for p, _ in DataSelect.select(self, data, parent_path).items()
        }
        return res


@dataclass(kw_only=True, eq=False)
class IdxMap(DataSelect):
    """Select and map nested data to an array."""

    sel: NodeSelector = All

    idx: DataSelect = field(default_factory=SelIndex)


@dataclass(kw_only=True, eq=False)
class ArrayMap[Val, Rec: Record]:
    """Select and map nested data to an array."""

    target: ArrayTarget[Val, Rec]

    idx: DataSelect = field(default_factory=SelIndex)


@dataclass(kw_only=True, eq=False)
class SubMap(TableMap, DataSelect):
    """Select and map nested data to another record."""

    sel: NodeSelector = All

    rel_map: TableMap | None = None
    """Mapping to attributes of the relation record."""


@dataclass(kw_only=True, eq=False)
class SubTableMap[Rec: Record, Dat, Rec2: Record](TableMap[Rec, Dat]):
    """Map nested data via a relation to another record."""

    target: Data[Rec, Any, Any, Any, Ctx[Rec2], Symbolic]
    """Relation to use for mapping."""

    rel_map: TableMap | None = None
    """Mapping to optional attributes of the relation record."""

    def __post_init__(self) -> None:  # noqa: D105
        self.target_type = self.target.record_type
        if self.rel_map is not None:
            self.rel_map.target_type = self.target.relation_type


def _parse_pushmap(push_map: PushMap) -> _PushMapping:
    """Parse push map into a more usable format."""
    match push_map:
        case Mapping() if has_type(push_map, _PushMapping):  # type: ignore
            return push_map
        case Data() | SubTableMap() | str() | Index():
            return {All: push_map}
        case Iterable() if has_type(push_map, Iterable[Data | SubTableMap]):
            return {
                k.name if isinstance(k, Data) else k.target.name: True for k in push_map
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
                if isinstance(target, Data | Index)
                else getattr(rec, _get_selector_name(sel))
            ): DataSelect(sel)
            for sel, target in mapping.items()
            if isinstance(target, Data | Index | bool)
        },
        **{
            (
                target.target
                if isinstance(target, SubTableMap) and target.target is not None
                else getattr(rec, _get_selector_name(sel))
            ): (
                copy_and_override(
                    SubMap, target, sel=sel, target_type=target.record_type
                )
                if isinstance(target, SubTableMap)
                else DataSelect.parse(sel)
            )
            for sel, targets in mapping.items()
            if has_type(targets, TableMap)
            or (has_type(targets, Iterable[TableMap]) and not isinstance(targets, Data))
            for target in ([targets] if isinstance(targets, TableMap) else targets)
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
    table: Data[Record | None, Any, Any, Any, Any, Any],
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
                getattr(table.record_type, a.name)
                for a in table.record_type._values.values()
            )
            if isinstance(match_by, str) and match_by == "all"
            else [match_by] if isinstance(match_by, Data) else match_by
        )
        return reduce(operator.and_, (col == rec_dict[col.name] for col in match_cols))


type InData = dict[tuple[Hashable, DirectPath], TreeNode]
type LazyData = list[
    tuple[
        SubTableMap[Record, TreeNode, Record],
        InData,
    ]
    | tuple[
        ArrayTarget[Any, Record],
        dict[Hashable, Any],
    ]
]
type RestData = list[tuple[TableMap[Record, TreeNode], InData]]


async def _load_record(
    table: Data[Record | None, Any, Any, Any, Any, Any],
    table_map: TableMap[Record, Any],
    path_idx: DirectPath,
    data: TreeNode,
    injections: dict[str, Hashable] | None,
) -> Hashable | Record | None:
    if table_map.loader is not None:
        if table_map.async_loader:
            data = await cast(
                Callable[[Any], Coroutine[Any, Any, TreeNode]],
                table_map.loader,
            )(data)
        else:
            data = table_map.loader(data)

    rec_idx: tuple | Hashable | None = None
    indexes = {
        idx: list(sel.select(data, path_idx).items())[0][1]
        for idx, sel in table_map.full_map.items()
        if isinstance(idx, Index)
    }
    if len(indexes) > 0:
        rec_pks = list(table.record_type._pk_values.values())
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
        idx_sel = reduce(lambda x, y: x | y, idx_maps)
        rec_idx = (
            tuple(idx_sel[pk] for pk in rec_pks)
            if len(idx_maps) > 1
            else list(idx_sel.values())[0]
        )

    attrs = {
        a.name: (a, *list(sel.select(data, path_idx).items())[0])
        for a, sel in table_map.full_map.items()
        if isinstance(a, Value)
    }

    rec_dict = {
        **{a[0].name: a[2] for a in attrs.values()},
        **(injections or {}),
    }

    rec: Record | None = None
    is_new: bool = True
    if table_map.record_type._is_complete_dict(rec_dict):
        rec = table_map.record_type(**rec_dict)
        rec_idx = rec._index

    if rec_idx is None and table_map.match_by == "index":
        return None

    match_expr = _gen_match_expr(
        table,
        rec_idx,
        rec_dict,
        table_map.match_by,
    )

    recs_keys = table[match_expr].keys()
    if len(recs_keys) > 0:
        rec_idx = recs_keys[0]
        is_new = False

    if rec_idx is None:
        return None

    if is_new and table_map.match_by != "index":
        table |= {rec_idx: rec}

    return rec if rec is not None and is_new else rec_idx


async def _load_records(
    table: Data[Record | None, Any, Any, Any, Any, Any],
    table_map: TableMap[Any, Any],
    in_data: InData,
    rest_data: RestData,
    injects: dict[DirectPath, dict[str, Hashable]] | None = None,
) -> dict[tuple[Hashable, Hashable, DirectPath], Record | Hashable]:
    # Prepare data injections for all records.
    injects = injects or {}
    injects |= {path_idx: injects.get(path_idx, {}) for (_, path_idx) in in_data.keys()}

    # Preload all forward-linked records.
    for rel, target_map in table_map.sub_tables.items():
        if isinstance(rel, Link):
            link_data: InData = {
                ((), path_idx): list(target_map.select(data_item, path_idx).items())[0][
                    1
                ]
                for (_, path_idx), data_item in in_data.items()
            }

            link_recs = await _load_records(
                table.db[rel.record_type],
                target_map,
                link_data,
                rest_data,
            )

            for link_idx, _, path_idx in link_recs.keys():
                injects[path_idx] |= rel._gen_fk_value_map(link_idx)

    # Define function for async-loading records.
    async def _load_rec_from_item(
        item: tuple[tuple[Hashable, DirectPath], TreeNode]
    ) -> Hashable | Record | None:
        (parent_idx, path_idx), data = item

        rec_inj = injects[path_idx]

        if isinstance(table_map, SubTableMap) and isinstance(
            table_map.target, BackLink
        ):
            rec_inj |= table_map.target.link._gen_fk_value_map(parent_idx)

        rec = await _load_record(table, table_map, path_idx, data, rec_inj)

        return rec

    # Queue up all record loading tasks.
    async_recs = tqdm(
        azip(
            in_data.items(),
            _sliding_batch_map(in_data.items(), _load_rec_from_item),
        ),
        desc=f"Async-loading `{table_map.record_type.__name__}`",
        total=len(in_data),
    )

    records: dict[tuple[Hashable, Hashable, DirectPath], Record | Hashable] = {}
    rest_records: InData = {}

    # Collect all loaded records.
    async for ((parent_idx, path_idx), data), rec in async_recs:
        if rec is None:
            rest_records[(parent_idx, path_idx)] = data
            continue

        rec_idx = rec._index if isinstance(rec, Record) else rec
        records[(rec_idx, parent_idx, path_idx)] = rec

    # Upload main records into the database.
    table |= {idx: rec for (idx, _, _), rec in records.items()}

    # Load relation records.
    if isinstance(table_map, SubTableMap) and issubclass(
        table_map.target.relation_type, Record
    ):
        assert table_map.target._backlink is not None
        assert table_map.target._link is not None

        rel_map = copy_and_override(
            SubTableMap,
            table_map.rel_map if table_map.rel_map is not None else TableMap(),
            target=table_map.target._backlink,
        )

        rel_injects = {
            path_idx: table_map.target._link._gen_fk_value_map(rec_idx)
            for rec_idx, _, path_idx in records.keys()
        }

        await _load_records(
            table.db[rel_map.target.record_type],
            rel_map,
            in_data,
            rest_data,
            rel_injects,
        )

    # Handle relations with incoming links.
    for tgt, sel in table_map.full_map.items():
        if isinstance(tgt, Table):
            # Load relation table.
            rel_map = (
                copy_and_override(SubTableMap, sel, target=tgt)
                if isinstance(sel, SubMap)
                else SubTableMap[Record, TreeNode, Record](
                    pull={
                        v: v.name
                        for v in cast(
                            type[Record], tgt.record_type
                        )._col_values.values()
                    },
                    target=tgt,
                )
            )

            rel_data = {
                (rec_idx, item_path): item_data
                for (rec_idx, parent_idx, path_idx) in records.keys()
                for item_path, item_data in sel.select(
                    in_data[(parent_idx, path_idx)], path_idx
                ).items()
            }

            await _load_records(
                table.db[rel_map.target.record_type],
                rel_map,
                rel_data,
                rest_data,
            )

        elif isinstance(tgt, Array):
            # Load array data.
            idx_sel = sel.idx if isinstance(sel, IdxMap) else SelIndex(sel=sel.sel)

            array_data = {}
            for rec_idx, parent_idx, path_idx in records.keys():
                data_idx = (parent_idx, path_idx)
                data = in_data[data_idx]
                array_data |= {
                    (
                        *(rec_idx if isinstance(rec_idx, tuple) else (rec_idx,)),
                        *(idx if isinstance(idx, tuple) else (idx,)),
                    ): val
                    for (_, idx), (_, val) in zip(
                        idx_sel.select(data, path_idx).items(),
                        sel.select(data, path_idx).items(),
                    )
                }

            table[tgt] |= array_data

    if len(rest_records) > 0:
        rest_data.append((table_map, rest_records))

    return records


@dataclass(kw_only=True, eq=False)
class DataSource[Rec: Record, Dat: TreeNode](TableMap[Rec, Dat]):
    """Root mapping for hierarchical data."""

    target: type[Rec]
    """Root record type to load."""

    def __post_init__(self) -> None:  # noqa: D105
        self.target_type = self.target

    @cached_property
    def _obj_cache(self) -> dict[type[Record], dict[Hashable, Any]]:
        return {}

    async def load(
        self, data: Iterable[Dat], db: DataBase | None = None
    ) -> set[Hashable]:
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
        db = db if db is not None else DataBase()
        in_data: InData = {((), ()): dat for dat in data}
        rest_data: RestData = []
        loaded = await _load_records(db[self.record_type], self, in_data, rest_data)

        # Perform second pass to load remaining data
        for table_map, tree_data in rest_data:
            rest_rest: RestData = []
            loaded |= await _load_records(
                db[table_map.record_type], table_map, tree_data, rest_rest
            )
            assert all(len(v[1]) == 0 for v in rest_rest)

        db[self.record_type] |= loaded
        return set(loaded.keys())
