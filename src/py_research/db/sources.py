"""Utilities for importing different data representations into relational format."""

from __future__ import annotations

import operator
from collections.abc import Callable, Coroutine, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property, reduce
from typing import Any, Literal, Self, cast

from lxml.etree import _ElementTree as ElementTree

from py_research.async_tools import sliding_batch_map
from py_research.data import copy_and_override
from py_research.db.data import SQL, Data, Filter, FullIdx
from py_research.db.models import Prop
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.types import SupportsItems, has_type
from py_research.telemetry import tqdm

from .records import Attr, DataBase, Key, Link, Record, Table

type TreeNode = Mapping[str | int, Any] | ElementTree | Hashable
type TreeData = TreeNode | Sequence[TreeNode]


class All:
    """Select all nodes on a level."""


type DirectPath = tuple[str | int | type[All], ...]

type DataConflictPolicy = Literal["raise", "ignore", "override", "collect"]
"""Policy for handling data conflicts."""

type DataConflicts = dict[tuple[str, str, str], tuple[Any, Any]]
"""Conflicting values indexed by their location in a table."""


class DataConflictError(ValueError):
    """Irreconsilable conflicts during import / merging of data."""

    def __init__(self, conflicts: DataConflicts) -> None:  # noqa: D107
        self.conflicts = conflicts
        super().__init__(
            f"Conflicting values: {conflicts}"
            if len(conflicts) < 5
            else f"{len(conflicts)} in table-columns "
            + str(set((k[0], k[2]) for k in conflicts.keys()))
        )


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


type _PushMapping[Rec: Record] = SupportsItems[NodeSelector, bool | PushMap[Rec]]

type PushMap[Rec: Record] = _PushMapping[Rec] | Prop[
    Any, Any, Any, Any, Any, Any, Rec
] | RelMap | Iterable[Prop[Any, Any, Any, Any, Any, Any, Rec] | RelMap[Any, Any, Rec]]
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


type PullMap[Rec: Record] = SupportsItems[
    Prop[Rec],
    "NodeSelector | DataSelect",
]
type _PullMapping[Rec: Record] = Mapping[
    Prop,
    DataSelect,
]


type RecMatchBy[Rec: Record] = (Literal["index", "all"] | list[Attr[Any, Rec]])


@dataclass(kw_only=True, eq=False)
class RecordMap[Rec: Record, Dat]:
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
        assert isinstance(
            self.target_type, type
        ), "Mapping only possible for concrete record types (no unions)."
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
                    copy_and_override(SubMap, sel, target_type=tgt.common_value_type)
                    if isinstance(sel, SubMap)
                    else DataSelect.parse(sel)
                )
                for tgt, sel in (self.pull or {}).items()
            },
        }

    @cached_property
    def sub_maps(
        self,
    ) -> dict[Prop, SubMap[Record]]:
        """Get all relation mappings."""
        return {
            cast(Prop[Any, Any, Any, Any, Any, Any, Rec], rel): sel
            for rel, sel in self.full_map.items()
            if isinstance(sel, SubMap)
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

    target: Prop[Val, Any, Any, Any, Any, Any, Rec]

    idx: DataSelect = field(default_factory=SelIndex)


@dataclass(kw_only=True, eq=False)
class SubMap[Rec: Record](RecordMap[Rec, Any], DataSelect):
    """Select and map nested data to another record."""

    sel: NodeSelector = All

    edge_map: RecordMap | None = None
    """Mapping to attributes of the relation record."""


@dataclass(kw_only=True, eq=False)
class RelMap[Rec: Record, Dat, Rec2: Record](RecordMap[Rec, Dat]):
    """Map nested data via a relation to another record."""

    target: Prop[Rec, Any, Any, Any, Any, Any, Rec2]
    """Prop to use for mapping."""

    edge_map: RecordMap | None = None
    """Mapping to optional attributes of the relation record."""

    def __post_init__(self) -> None:  # noqa: D105
        self.target_type = self.target.common_value_type

        if self.edge_map is not None:
            assert isinstance(self.target.context, Prop)
            self.edge_map.target_type = self.target.context.common_value_type


def _parse_pushmap(push_map: PushMap) -> _PushMapping:
    """Parse push map into a more usable format."""
    match push_map:
        case Mapping() if has_type(push_map, _PushMapping):  # type: ignore
            return push_map
        case Prop() | RelMap() | str():
            return {All: push_map}
        case Iterable() if has_type(push_map, Iterable[Prop | RelMap]):
            return {
                k.name if isinstance(k, Prop) else k.target.name: True for k in push_map
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
                if isinstance(target, Prop)
                else getattr(rec, _get_selector_name(sel))
            ): DataSelect(sel)
            for sel, target in mapping.items()
            if isinstance(target, Prop | bool)
        },
        **{
            (
                target.target
                if isinstance(target, RelMap) and target.target is not None
                else getattr(rec, _get_selector_name(sel))
            ): (
                copy_and_override(
                    SubMap, target, sel=sel, target_type=target.record_type
                )
                if isinstance(target, RelMap)
                else DataSelect.parse(sel)
            )
            for sel, targets in mapping.items()
            if has_type(targets, RecordMap)
            or (
                has_type(targets, Iterable[RecordMap]) and not isinstance(targets, Prop)
            )
            for target in ([targets] if isinstance(targets, RecordMap) else targets)
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


type RecMatchExpr = list[Hashable] | Filter[SQL]


def _gen_match_expr(
    rec_type: type[Record],
    rec_idx: Hashable | None,
    rec_dict: dict[str, Any],
    match_by: RecMatchBy,
) -> RecMatchExpr:
    if isinstance(match_by, str) and match_by == "index":
        assert rec_idx is not None
        return [rec_idx]
    else:
        match_cols = (
            list(rec_type._attrs().values())
            if match_by == "all"
            else list(rec_type._pk.components) if match_by == "index" else match_by
        )
        return reduce(operator.and_, (col == rec_dict[col.name] for col in match_cols))


type InData = dict[tuple[Hashable, DirectPath], TreeNode]
type LazyData = list[
    tuple[
        RelMap[Record, TreeNode, Record],
        InData,
    ]
    | tuple[
        Prop,
        dict[Hashable, Any],
    ]
]
type RestData = list[tuple[RecordMap[Record, TreeNode], InData]]


async def _load_record(
    table: Data[Record, FullIdx[Hashable]],
    table_map: RecordMap[Record, Any],
    path_idx: DirectPath,
    data: TreeNode,
    injections: dict[str, Hashable] | None,
) -> Hashable | Record | None:
    assert len(table.value_type_set) == 1
    rec_type = table.common_value_type

    if table_map.loader is not None:
        if table_map.async_loader:
            data = await cast(
                Callable[[Any], Coroutine[Any, Any, TreeNode]],
                table_map.loader,
            )(data)
        else:
            data = table_map.loader(data)

    attrs = {
        a.name: (a, *list(sel.select(data, path_idx).items())[0])
        for a, sel in table_map.full_map.items()
        if isinstance(a, Attr) and a.init is True
    }

    rec_pk = None
    rec_dict = {
        **{a[0].name: a[2] for a in attrs.values()},
        **(injections or {}),
    }

    keys = {
        key: list(sel.select(data, path_idx).items())[0][1]
        for key, sel in table_map.full_map.items()
        if isinstance(key, Key)
    }
    if len(keys) > 0:
        key_vals = [key.gen_component_map(key_val) for key, key_val in keys.items()]
        rec_dict |= reduce(lambda x, y: x | y, key_vals)
        rec_pk = tuple(rec_dict[a.name] for a in rec_type._pk.components)

    if rec_pk is None and table_map.match_by == "index":
        return None

    is_new: bool = True
    match_expr = _gen_match_expr(
        rec_type,
        rec_pk,
        rec_dict,
        table_map.match_by,
    )
    recs_keys = table[match_expr].keys()

    if len(recs_keys) > 0:
        rec_pk = recs_keys[0]
        is_new = False

    if rec_pk is None:
        return None

    rec: Record | None = None
    if table_map.record_type._is_complete_dict(rec_dict):
        rec = table_map.record_type(**rec_dict)

    if rec is not None and is_new and table_map.match_by != "index":
        table |= {rec_pk: rec}

    return rec if rec is not None and is_new else rec_pk


async def _load_records(
    table: Table,
    table_map: RecordMap[Any, Any],
    in_data: InData,
    rest_data: RestData,
    injects: dict[DirectPath, dict[str, Hashable]] | None = None,
    post_round: bool = False,
) -> dict[tuple[Hashable, Hashable, DirectPath], Record | Hashable]:
    # Prepare data injections for all records.
    injects = injects or {}
    injects |= {path_idx: injects.get(path_idx, {}) for (_, path_idx) in in_data.keys()}

    # Preload all forward-linked records.
    for prop, target_map in table_map.sub_maps.items():
        if issubclass(prop.common_value_type, Record) and prop.init_level < 0:
            ref_data: InData = {
                ((), path_idx): list(target_map.select(data_item, path_idx).items())[0][
                    1
                ]
                for (_, path_idx), data_item in in_data.items()
            }

            sub_table = table.root()[target_map.record_type]
            assert isinstance(sub_table, Table)

            ref_recs = await _load_records(
                sub_table,
                target_map,
                ref_data,
                rest_data,
                None,
                post_round,
            )

            if prop.init:
                for (_, _, path_idx), ref_rec in ref_recs.items():
                    injects[path_idx] |= {prop.name: ref_rec}

    # Define function for async-loading records.
    async def _load_rec_from_item(
        item: tuple[tuple[Hashable, DirectPath], TreeNode],
    ) -> tuple[Hashable, DirectPath, TreeNode, Hashable | Record | None]:
        (parent_idx, path_idx), data = item
        rec_inj = injects[path_idx]
        rec = await _load_record(table, table_map, path_idx, data, rec_inj)
        return parent_idx, path_idx, data, rec

    # Queue up all record loading tasks.
    async_recs = tqdm(
        sliding_batch_map(in_data.items(), _load_rec_from_item),
        desc=("Loading: " if not post_round else "Post-loading: ")
        + f"`{table_map.record_type.__name__}`",
        total=len(in_data),
        leave=True,
    )

    records: dict[tuple[Hashable, Hashable, DirectPath], Record | Hashable] = {}
    rest_records: InData = {}

    # Collect all loaded records.
    async for parent_idx, path_idx, data, rec in async_recs:
        if rec is None:
            rest_records[(parent_idx, path_idx)] = data
            continue

        rec_idx = rec._pk if isinstance(rec, Record) else rec
        records[(rec_idx, parent_idx, path_idx)] = rec

    # Upload main records into the database.
    table |= {idx: rec for (idx, _, _), rec in records.items()}

    if len(rest_records) > 0:
        # Collect remaining data for later loading.
        rest_data.append((table_map, rest_records))

    # Load relation records.
    # TODO: Handle post-loading of context (relation) records.
    # TODO: Handle post-loading of sub-relations.

    # Handle relations with incoming links.
    for tgt, sel in table_map.full_map.items():
        if isinstance(tgt, Table) and not isinstance(tgt, Link):
            # Load relation table.
            if isinstance(sel, SubMap):
                rel_map = copy_and_override(RelMap, sel, target=tgt)
            else:
                assert len(tgt.origin_type_set) == 1
                rel_map = RelMap[Record, TreeNode, Record](
                    pull={
                        v: v.name
                        for v in cast(
                            type[Record], next(iter(tgt.origin_type_set))
                        )._col_values.values()
                    },
                    target=tgt,
                )

            rel_data = {
                (rec_idx, item_path): item_data
                for (rec_idx, parent_idx, path_idx) in records.keys()
                for item_path, item_data in sel.select(
                    in_data[(parent_idx, path_idx)], path_idx
                ).items()
            }

            await _load_records(
                Table(_base=table.base, _type=rel_map.record_type),
                rel_map,
                rel_data,
                rest_data,
                None,
                post_round,
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

    return records


@dataclass(kw_only=True, eq=False)
class DataSource[Rec: Record, Dat: TreeNode](RecordMap[Rec, Dat]):
    """Root mapping for hierarchical data."""

    target: type[Rec]
    """Root record type to load."""

    def __post_init__(self) -> None:  # noqa: D105
        self.target_type = self.target

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
                db[table_map.record_type], table_map, tree_data, rest_rest, None, True
            )
            assert all(len(v[1]) == 0 for v in rest_rest)

        return set(loaded.keys())
