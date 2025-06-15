"""Utilities for importing different data representations into relational format."""

from __future__ import annotations

import operator
from collections.abc import (
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    Set,
)
from dataclasses import dataclass, field, fields
from functools import cached_property, reduce
from typing import Any, Literal, Self, cast, override

from lxml.etree import _ElementTree as ElementTree

from py_research.async_tools import sliding_batch_map
from py_research.data import copy_and_override
from py_research.db.data import SQL, Data, Filter, FullIdx, Interface
from py_research.db.models import Prop
from py_research.hashing import gen_int_hash, gen_str_hash
from py_research.reflect.types import SupportsItems, has_type
from py_research.telemetry import tqdm

from .records import Attr, DataBase, Link, Record, Table

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


type NodeSelect = str | int | TreePath | type[All]
"""Select a node in a hierarchical data structure."""


def _get_selector_name(selector: NodeSelect) -> str:
    """Get the name of the selector."""
    match selector:
        case str():
            return selector
        case int() | TreePath() | type():
            raise ValueError(f"Cannot get name of selector with type {type(selector)}.")


@dataclass
class DataSelect:
    """Select node for further processing."""

    @staticmethod
    def parse(obj: NodeSelect | DataSelect) -> DataSelect:
        """Parse the object into an x selector."""
        if isinstance(obj, DataSelect):
            return obj
        return DataSelect(sel=obj)

    sel: NodeSelect
    """Node selector to use."""

    def prefix(self, prefix: NodeSelect) -> Self:
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
class SelIdx(DataSelect):
    """Select and transform attributes."""

    sel: NodeSelect = All
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


type InData = dict[tuple[tuple, DirectPath], TreeNode]
type LazyData = list[
    tuple[
        SubMapPush[Record, Record],
        InData,
    ]
    | tuple[
        Prop,
        dict[Hashable, Any],
    ]
]
type RestData = list[tuple[RecMap[Record], InData]]


@dataclass(kw_only=True, eq=False)
class DataMap[Ldd, Tgt]:
    """Configuration for how to map (nested) dictionary items to relational tables."""

    loader: Callable[[TreeNode], Ldd | Coroutine[Any, Any, Ldd]] | None = None
    """Loader function to load data for this record from a source."""

    async_loader: bool = True
    """If True, the loader is an async function."""

    conflicts: DataConflictPolicy = "collect"
    """Which policy to use if import conflicts occur for this record table."""

    idx: DataSelect = field(default_factory=SelIdx)

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(self)

    @property
    def target_type(self) -> type[Tgt]:
        """Get the target type."""
        raise NotImplementedError()

    async def _load_item(
        self,
        item: tuple[tuple[tuple, DirectPath], TreeNode],
    ) -> tuple[tuple, DirectPath, Ldd]:
        """Perform the loading and mapping of a single item."""
        (parent_idx, path_idx), data = item

        if self.loader is not None:
            if self.async_loader:
                data = await cast(
                    Callable[[Any], Coroutine[Any, Any, TreeNode]],
                    self.loader,
                )(data)
            else:
                data = self.loader(data)
                assert not isinstance(data, Coroutine)

        return parent_idx, path_idx, cast(Ldd, data)

    async def load(
        self,
        in_data: InData,
        post_round: bool = False,
    ) -> dict[tuple[tuple, tuple, DirectPath], Ldd]:
        """Perform the loading and mapping of a single item."""
        # Queue up all record loading tasks.
        async_vals = tqdm(
            sliding_batch_map(in_data.items(), self._load_item),
            desc=("Loading: " if not post_round else "Post-loading: ")
            + f"`{self.target_type.__name__}`",
            total=len(in_data),
            leave=True,
        )

        vals: dict[tuple[tuple, tuple, DirectPath], Ldd] = {}
        rest_vals: InData = {}

        # Collect all loaded data.
        async for parent_idx, path_idx, val in async_vals:
            if val is None:
                rest_vals[(parent_idx, path_idx)] = val
                continue

            val_idx = tuple(self.idx.select(val, path_idx).values())[0]
            vals[(parent_idx, val_idx, path_idx)] = cast(Ldd, val)

        return vals

    async def map(
        self,
        db: DataBase,
        tgt: type[Tgt] | Data[Tgt, FullIdx[*tuple[Any, ...]], Any, Any, Any, Interface],
        in_data: InData,
        rest_data: RestData,
        injects: Mapping[tuple[tuple, DirectPath], dict[str, Any]] | None = None,
        post_round: bool = False,
    ) -> dict[tuple, Tgt | None]:
        """Map the loaded data to the target."""
        vals = await self.load(in_data, post_round)
        assert has_type(vals, dict[tuple[tuple, tuple, DirectPath], self.target_type])

        return {(*parent_idx, *idx): val for (parent_idx, idx, _), val in vals.items()}


@dataclass
class Index:
    """Map to the index."""

    pos: int | slice = slice(None)
    pks: Iterable[Attr] | None = None


type _PushMapping[Rec: Record] = SupportsItems[NodeSelect, bool | PushMap[Rec]]

type PushMap[Rec: Record] = _PushMapping[Rec] | Prop[Any, Any, Any, Rec] | SubMapPush[
    Any, Rec
] | Index | Iterable[Prop[Any, Any, Any, Rec] | SubMapPush[Any, Rec] | Index]
"""Mapping of hierarchical attributes to record props or other records."""


type PullMap[Rec: Record] = SupportsItems[
    Prop[Rec],
    "NodeSelect | DataSelect",
]
type _PullMapping[Rec: Record] = Mapping[
    Prop,
    DataSelect,
]


type RecMatchBy[Rec: Record] = (Literal["index", "all"] | list[Attr[Any, Any, Rec]])
type RecMatchExpr = list[Hashable] | Filter[SQL]


@dataclass(kw_only=True, eq=False)
class RecMap[Rec: Record](DataMap[TreeNode, Rec]):
    """Configuration for how to map dictionary items to records."""

    push: PushMap[Rec] | None = None
    """Mapping of hierarchical attributes to record props or other records."""

    pull: PullMap[Rec] | None = None
    """Mapping of record props to hierarchical attributes."""

    match_by: RecMatchBy = "index"
    """Match this data to existing records via given attributes instead of index."""

    @staticmethod
    def _parse_push(push_map: PushMap) -> _PushMapping:
        match push_map:
            case Mapping() if has_type(push_map, _PushMapping):  # type: ignore
                return push_map
            case Prop() | SubMapPush() | str():
                return {All: push_map}
            case Iterable() if has_type(push_map, Iterable[Prop | SubMapPush]):
                return {
                    k.name if isinstance(k, Prop) else k.target.name: True
                    for k in push_map
                }
            case _:
                raise TypeError(f"Unsupported mapping type {type(push_map)}")

    @staticmethod
    def _push_to_pull(rec: type[Record], push_map: PushMap) -> _PullMapping:
        mapping = RecMap._parse_push(push_map)

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
                    if isinstance(target, SubMapPush) and target.target is not None
                    else getattr(rec, _get_selector_name(sel))
                ): (
                    copy_and_override(
                        SubMapPull, target, sel=sel, _target_type=target.target_type
                    )
                    if isinstance(target, SubMapPush)
                    else DataSelect.parse(sel)
                )
                for sel, targets in mapping.items()
                if has_type(targets, RecMap)
                or (
                    has_type(targets, Iterable[RecMap])
                    and not isinstance(targets, Prop)
                )
                for target in ([targets] if isinstance(targets, RecMap) else targets)
            },
        }

        # Handle nested data attributes (which come as dict types).
        for sel, target in mapping.items():
            if has_type(target, Mapping):
                sub_pull_map = RecMap._push_to_pull(
                    rec,
                    target,
                )
                pull_map = {
                    **pull_map,
                    **{
                        prop: sub_sel.prefix(sel)
                        for prop, sub_sel in sub_pull_map.items()
                    },
                }

        return pull_map

    def _gen_match_expr(
        self,
        rec_idx: Hashable | None,
        rec_dict: dict[str, Any],
    ) -> RecMatchExpr:
        if isinstance(self.match_by, str) and self.match_by == "index":
            assert rec_idx is not None
            return [rec_idx]
        else:
            match_cols = (
                list(self.target_type._attrs().values())
                if self.match_by == "all"
                else (
                    list(self.target_type._pk.components)
                    if self.match_by == "index"
                    else self.match_by
                )
            )
            return reduce(
                operator.and_, (col == rec_dict[col.name] for col in match_cols)
            )

    @cached_property
    def full_map(self) -> _PullMapping[Rec]:
        """Get the full mapping."""
        return {
            **(
                RecMap._push_to_pull(self.target_type, self.push)
                if self.push is not None
                else {}
            ),
            **{
                tgt: (
                    copy_and_override(
                        SubMapPull, sel, sel=sel.sel, _target_type=tgt.common_value_type
                    )
                    if isinstance(sel, SubMapPull)
                    else DataSelect.parse(sel)
                )
                for tgt, sel in (self.pull or {}).items()
            },
        }

    @cached_property
    def props(
        self,
    ) -> dict[Prop, DataSelect]:
        """Get all relation mappings."""
        return {
            cast(Prop[Any, Any, Any, Rec], rel): sel
            for rel, sel in self.full_map.items()
            if not isinstance(sel, SubMapPull)
        }

    @cached_property
    def sub_maps(
        self,
    ) -> dict[Prop, SubMapPull]:
        """Get all relation mappings."""
        return {
            cast(Prop[Any, Any, Any, Rec], rel): sel
            for rel, sel in self.full_map.items()
            if isinstance(sel, SubMapPull)
        }

    async def _load_record(
        self,
        table: Data[Any, FullIdx[*tuple[Any, ...]]],
        path_idx: DirectPath,
        data: TreeNode,
        injections: dict[str, Hashable] | None,
    ) -> Rec | Hashable | None:
        """Perform the loading and mapping of a single item."""
        rec_type = self.target_type

        attrs = {
            a.name: (a, *list(sel.select(data, path_idx).items())[0])
            for a, sel in self.props.items()
            if a.init
        }

        rec_dict = {
            **{a[0].name: a[2] for a in attrs.values()},
            **(injections or {}),
        }

        if (
            not all(a.name in rec_dict for a in rec_type._pk.components)
            and self.match_by == "index"
        ):
            return None

        rec_pk = tuple(rec_dict[a.name] for a in rec_type._pk.components)
        is_new: bool = True

        match_expr = self._gen_match_expr(
            rec_pk,
            rec_dict,
        )
        recs_keys = table[match_expr].keys()

        if len(recs_keys) > 0:
            rec_pk = recs_keys[0]
            is_new = False

        if rec_pk is None:
            return None

        rec: Record | None = None
        if self.target_type._is_complete_dict(rec_dict):
            rec = self.target_type(**rec_dict)

        if rec is not None and is_new and self.match_by != "index":
            table |= {rec_pk: rec}

        return rec if rec is not None and is_new else rec_pk

    @override
    async def map(
        self,
        db: DataBase,
        tgt: type[Rec] | Data[Rec, FullIdx[*tuple[Any, ...]], Any, Any, Any, Interface],
        in_data: InData,
        rest_data: RestData,
        injects: Mapping[tuple[tuple, DirectPath], dict[str, Any]] | None = None,
        post_round: bool = False,
    ) -> dict[tuple, Rec | None]:
        # Prepare injects.
        injects = dict(injects or {})

        # Load input values.
        vals = await self.load(
            in_data,
            post_round,
        )

        # Pre-load props.
        for prop, sub_map in self.sub_maps.items():
            if prop.init_level < 0:
                sub_data: InData = {
                    ((*parent_idx, *idx), item_path_idx): item
                    for (parent_idx, idx, path_idx), val in vals.items()
                    for item_path_idx, item in sub_map.select(val, path_idx).items()
                }

                sub_vals = await sub_map.map(
                    db,
                    tgt[prop] if isinstance(tgt, Data) else getattr(tgt, prop.name),
                    sub_data,
                    rest_data,
                    None,
                    post_round,
                )

                if prop.init:
                    for (parent_idx, _, path_idx), sub_val in sub_vals.items():
                        injects[(parent_idx, path_idx)] |= {prop.name: sub_val}

        table = db[self.target_type]
        assert isinstance(table, Table)

        records: dict[tuple, Rec | None] = {}
        rest_recs: InData = {}

        # Load records.
        for (parent_idx, idx, path_idx), data in vals.items():
            injections = injects.get((parent_idx, path_idx), {})
            rec = await self._load_record(
                table,
                path_idx,
                data,
                injections,
            )

            if isinstance(rec, Record | None):
                records[(*parent_idx, *idx)] = cast(Rec, rec)
            else:
                records[(*parent_idx, idx)] = cast(Rec, table[rec])

            if rec is None:
                rest_recs[(parent_idx, path_idx)] = data

        if len(rest_recs) > 0:
            rest_data.append((cast(RecMap, self), rest_recs))

        # Post-load props.
        for prop, sub_map in self.sub_maps.items():
            if prop.init_level >= 0:
                sub_data: InData = {
                    ((*parent_idx, *idx), item_path_idx): item
                    for (parent_idx, idx, path_idx), val in vals.items()
                    for item_path_idx, item in sub_map.select(val, path_idx).items()
                }

                sub_injects = None
                if isinstance(prop, Link):
                    assert has_type(prop.on, Set[Link] | Link)
                    links = prop.on if isinstance(prop.on, Set) else {prop.on}
                    sub_injects = {
                        (parent_idx, path_idx): {
                            link: records[parent_idx]
                            for link in links
                            if issubclass(self.target_type, link.target_type)
                        }
                        for (parent_idx, path_idx) in sub_data.keys()
                    }

                sub_vals = await sub_map.map(
                    db,
                    tgt[prop] if isinstance(tgt, Data) else getattr(tgt, prop.name),
                    sub_data,
                    rest_data,
                    sub_injects,
                    post_round,
                )

                db[cast(Data[object], tgt)] |= {
                    (*parent_idx, *idx): val
                    for (parent_idx, idx, _), val in sub_vals.items()
                }

        return records


@dataclass(kw_only=True, eq=False)
class SubMapPush[Val, Rec: Record](DataMap[Val, Rec]):
    """Select and map nested data."""

    target: Prop[Val, Any, Any, Rec]

    @property
    def target_type(self) -> type[Rec]:
        """Get the target type."""
        return cast(type[Rec], self.target.common_value_type)


@dataclass(kw_only=True, eq=False)
class SubMapPull[Val, Rec: Record](DataMap[Val, Rec], DataSelect):
    """Select and map nested data."""

    _target_type: type[Rec] | None = None

    @property
    def target_type(self) -> type[Rec]:
        """Get the target type."""
        assert self._target_type is not None, "Target type is not set."
        return self._target_type


@dataclass(kw_only=True, eq=False)
class RelMapPush[Rec: Record, Rec2: Record](  # type: ignore
    SubMapPush[Rec, Rec2], RecMap[Rec]
):
    """Select and map nested data to records."""


class RelMapPull[Rec: Record, Rec2: Record](  # type: ignore
    SubMapPull[Rec, Rec2], RecMap[Rec]
):
    """Select and map nested data to records."""


@dataclass(kw_only=True, eq=False)
class DataSource[Rec: Record, Dat: TreeNode](RecMap[Rec]):
    """Root mapping for hierarchical data."""

    target: type[Rec]
    """Root record type to load."""

    @property
    def target_type(self) -> type[Rec]:
        """Get the target type."""
        return self.target

    async def to_db(
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
        loaded = await self.map(db, self.target_type, in_data, rest_data)

        # Perform second pass to load remaining data
        for table_map, tree_data in rest_data:
            rest_rest: RestData = []
            loaded |= await table_map.map(
                db, table_map.target_type, tree_data, rest_rest, None, True
            )
            assert all(len(v[1]) == 0 for v in rest_rest)

        return set(loaded.keys())
