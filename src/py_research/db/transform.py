"""Functions and classes for transforming between recursive and relational format."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, groupby
from typing import Any, Literal, cast, overload

import pandas as pd
from inflect import engine as inflect_engine

from py_research.db import DB
from py_research.db.conflicts import (
    DataConflictError,
    DataConflictPolicy,
    DataConflicts,
)
from py_research.hashing import gen_str_hash

Scalar = str | int | float | datetime

AttrMap = Mapping[str, str | bool | dict]
"""Mapping of hierarchical attributes to table rows"""

RelationalMap = Mapping[str, "str | bool | dict | TableMap | list[TableMap]"]


@dataclass
class TableMap:
    """Defines how to map a data item to the database."""

    table: str
    map: RelationalMap | set[str] | str | Callable[
        [dict | str], RelationalMap | set[str] | str
    ]
    name: str | None = None
    link_map: AttrMap | None = None

    hash_id_subset: list[str] | None = None
    """Supply a list of column names as a subset of all columns to use
    for auto-generating row ids via sha256 hashing.
    """
    match_by_attr: bool | str = False
    """Match this mapped data to target table (by given attr)."""

    ext_maps: "list[TableMap] | None" = None
    """Map attributes on the same level to different tables"""

    ext_attr: str | dict[str, Any] | None = None
    """Override attr to use when linking this table with a parent table."""

    id_attr: str | None = None
    """Use given attr as id directly, no hashing."""

    conflict_policy: DataConflictPolicy = "raise"
    """Which policy to use if import conflicts occur for this table."""


SubMap = tuple[dict | list, TableMap | list[TableMap]]


def _resolve_relmap(
    node: dict, mapping: RelationalMap
) -> tuple[pd.Series, dict[str, SubMap]]:
    """Extract hierarchical data into set of scalar attributes + linked data objects."""
    # Split the current mapping level into groups based on type.
    target_groups: dict[type, dict] = {
        t: dict(g)  # type: ignore
        for t, g in groupby(
            sorted(
                mapping.items(),
                key=lambda item: str(type(item[1])),
            ),
            key=lambda item: type(item[1]),
        )
    }  # type: ignore

    # First list and handle all scalars, hence data attributes on the current level,
    # which are to be mapped to table columns.
    scalars = dict(
        chain(
            (target_groups.get(str) or {}).items(),
            (target_groups.get(bool) or {}).items(),
        )
    )

    cols = {
        (col if isinstance(col, str) else attr): data
        for attr, col in scalars.items()
        if isinstance(data := node.get(attr), Scalar)
    }

    links = {
        attr: (data, cast(TableMap | list[TableMap], sub_map))
        for attr, sub_map in {
            **(target_groups.get(TableMap) or {}),
            **(target_groups.get(list) or {}),
        }.items()
        if isinstance(data := node.get(attr), dict | list)
    }

    # Handle nested data attributes (which come as dict types).
    for attr, sub_map in (target_groups.get(dict) or {}).items():
        sub_node = node.get(attr)
        if isinstance(sub_node, dict):
            sub_row, sub_links = _resolve_relmap(sub_node, cast(dict, sub_map))
            cols = {**cols, **sub_row}
            links = {**links, **sub_links}

    return pd.Series(cols, dtype=object), links


def _gen_row_id(row: pd.Series, hash_subset: list[str] | None = None) -> str:
    """Generate and assign row id."""
    hash_row = (
        row[list(set(hash_subset) & set(row.index))] if hash_subset is not None else row
    )
    row_id = gen_str_hash(hash_row.to_dict())

    return row_id


DictDB = dict[str, dict[Hashable, dict[str, Any]]]

inf = inflect_engine()


def _resolve_links(
    mapping: TableMap,
    database: DictDB,
    row: pd.Series,
    links: list[tuple[str | dict[str, Any] | None, SubMap]],
    collect_conflicts: bool = False,
    _all_conflicts: DataConflicts | None = None,
) -> DataConflicts:
    _all_conflicts = _all_conflicts or {}
    # Handle nested data, which is to be extracted into separate tables and linked.
    for attr, (sub_data, sub_maps) in links:
        # Get info about the link table to use from mapping
        # (or generate a new for the link table).

        if not isinstance(sub_maps, list):
            sub_maps = [sub_maps]

        for sub_map in sub_maps:
            link_table = f"{mapping.table}_{sub_map.table}"
            link_table_alt = f"{sub_map.table}_{mapping.table}"

            if not isinstance(sub_data, list):
                sub_data = [sub_data]

            if link_table not in database:
                if link_table_alt in database:
                    link_table = link_table_alt
                else:
                    database[link_table] = {}

            for sub_data_item in sub_data:
                if isinstance(sub_data_item, dict):
                    rel_row, _all_conflicts = _recursive_to_db(
                        sub_data_item,
                        sub_map,
                        database,
                        collect_conflicts,
                        _all_conflicts,
                    )

                    link_row, _ = (
                        _resolve_relmap(sub_data_item, sub_map.link_map)
                        if sub_map.link_map is not None
                        else (pd.Series(dtype=object), None)
                    )

                    link_row[
                        f"{mapping.table}.id"
                        + ("" if mapping.table != sub_map.table else ".0")
                    ] = row.name
                    link_row[
                        f"{sub_map.table}.id"
                        + ("" if mapping.table != sub_map.table else ".1")
                    ] = rel_row.name

                    if isinstance(attr, str | None):
                        link_row["attribute"] = attr
                    else:
                        link_row = pd.Series({**link_row.to_dict(), **attr})

                    link_row.name = _gen_row_id(link_row)

                    database[link_table][link_row.name] = link_row.to_dict()

    return _all_conflicts


def _recursive_to_db(  # noqa: C901
    data: dict | str,
    mapping: TableMap,
    database: DictDB,
    collect_conflicts: bool = False,
    _all_conflicts: DataConflicts | None = None,
) -> tuple[pd.Series, DataConflicts]:
    database = database or {}
    _all_conflicts = _all_conflicts or {}
    # Initialize new table as defined by mapping, if it doesn't exist yet.
    if mapping.table not in database:
        database[mapping.table] = {}

    resolved_map = (
        mapping.map(data) if isinstance(mapping.map, Callable) else mapping.map
    )

    # Extract row of data attributes and links to other objects.
    row = None
    links: list[tuple[str | dict[str, Any] | None, SubMap]] = []
    # If mapping is only a string, extract the target attr directly.
    if isinstance(data, str):
        assert isinstance(resolved_map, str)
        row = pd.Series({resolved_map: data}, dtype=object)
    else:
        if isinstance(resolved_map, set):
            row = pd.Series(
                {k: v for k, v in data.items() if k in resolved_map}, dtype=object
            )
        elif isinstance(resolved_map, dict):
            row, link_dict = _resolve_relmap(data, resolved_map)
            links = [*links, *link_dict.items()]
        else:
            raise TypeError(
                f"Unsupported mapping type {type(resolved_map)}"
                f" for data of type {type(data)}"
            )

    row.name = (
        _gen_row_id(row, mapping.hash_id_subset)
        if not mapping.id_attr
        else row[mapping.id_attr]
    )

    if not isinstance(row.name, str | int):
        raise ValueError(
            f"Value of `'{mapping.id_attr}'` (`TableMap.id_attr`) "
            f"must be a string or int for all objects, but received {row.name}"
        )

    if mapping.ext_maps is not None:
        assert isinstance(data, dict)
        links = [*links, *((m.ext_attr, (data, m)) for m in mapping.ext_maps)]

    _all_conflicts = _resolve_links(
        mapping, database, row, links, collect_conflicts, _all_conflicts
    )

    existing_row: dict[str, Any] | None = None
    if mapping.match_by_attr:
        # Make sure any existing data in database is consistent with new data.
        match_by = cast(str, mapping.match_by_attr)
        if mapping.match_by_attr is True:
            assert isinstance(resolved_map, str)
            match_by = resolved_map

        match_to = row[match_by]

        # Match to existing row or create new one.
        existing_rows: list[tuple[str, dict[str, Any]]] = list(
            filter(
                lambda i: i[1][match_by] == match_to, database[mapping.table].items()
            )
        )

        if len(existing_rows) > 0:
            existing_row_id, existing_row = existing_rows[0]
            # Take over the id of the existing row.
            row.name = existing_row_id
    else:
        existing_row = database[mapping.table].get(row.name)

    if existing_row is not None:
        existing_attrs = set(
            str(k) for k, v in existing_row.items() if k and pd.notna(v)
        )
        new_attrs = set(str(k) for k, v in row.items() if k and pd.notna(v))

        # Assert that overlapping attributes are equal.
        intersect = existing_attrs & new_attrs

        if mapping.conflict_policy == "raise":
            conflicts = {
                (mapping.table, row.name, c): (existing_row[c], row[c])
                for c in intersect
                if existing_row[c] != row[c]
            }

            if len(conflicts) > 0:
                if not collect_conflicts:
                    raise DataConflictError(conflicts)

                _all_conflicts = {**_all_conflicts, **conflicts}
        if mapping.conflict_policy == "ignore":
            row = pd.Series({**row.loc[list(new_attrs)], **existing_row}, name=row.name)
        else:
            row = pd.Series({**existing_row, **row.loc[list(new_attrs)]}, name=row.name)

    # Add row to database table or update it.
    database[mapping.table][row.name] = row.to_dict()

    # Return row (used for recursion).
    return row, _all_conflicts


@overload
def recursive_to_db(
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: Literal[True] = ...,
) -> tuple[DB, DataConflicts]:
    ...


@overload
def recursive_to_db(
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: bool = ...,
) -> DB:
    ...


def recursive_to_db(  # noqa: C901
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: bool = False,
) -> DB | tuple[DB, DataConflicts]:
    """Transform recursive dictionary data into relational format.

    Args:
        data: The data to be transformed
        mapping:
            Configuration for how to map (nested) dictionary items
            to relational tables
        collect_conflicts:
            Collect all conflicts and return them, rather than stopping right away.
    """
    df_dict = {}
    _, conflicts = _recursive_to_db(data, mapping, df_dict, collect_conflicts)
    db = DB()._import_df_dict(df_dict)
    return (db, conflicts) if collect_conflicts else db
