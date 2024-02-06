"""Utilities for importing different data representations into relational format."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, groupby
from typing import Any, Literal, TypeAlias, cast, overload
from uuid import uuid4

import pandas as pd

from py_research.hashing import gen_str_hash

from .base import DB
from .conflicts import DataConflictError, DataConflictPolicy, DataConflicts

Scalar = str | int | float | datetime

_AttrMap = Mapping[str, "str | bool | _AttrMap"]
"""Mapping of hierarchical attributes to table columns"""

_RelationalMap = Mapping[str, "str | bool | _RelationalMap | TableMap | list[TableMap]"]
"""Mapping of hierarchical attributes to table columns or other tables."""


@dataclass
class TableMap:
    """Configuration for how to map (nested) dictionary items to relational tables."""

    table: str
    """Name of the table to map to attributes in ``map`` to."""

    map: (
        _RelationalMap
        | set[str]
        | str
        | Callable[[dict | str], _RelationalMap | set[str] | str]
    )
    """Mapping of hierarchical attributes to table columns or other tables."""

    ext_maps: "list[TableMap] | None" = None
    """Map attributes on the same level to different tables"""

    link_attr: str | None = None
    """Override attribute to use when referencing this table from a parent table."""

    link_type: Literal["1-n", "n-1", "n-m"] = "n-m"
    """Type of reference between this table and the parent table.
    Only ``n-m`` will create a join table.
    """

    join_table_name: str | None = None
    """Name of the join table to use for referencing this table to its parent, if any"""

    join_table_map: _AttrMap | None = None
    """Mapping of hierarchical attributes to join table rows."""

    id_type: Literal["hash", "attr", "uuid"] = "hash"
    """Type of id to use for this table.
    - ``hash``: sha256 hash of a subset of all mapped attributes
    - ``attr``: use given attr as id directly, no hashing
    - ``uuid``: generate random uuid for each row
    """

    id_attr: str | list[str] | None = None
    """Name of unique, mapped attribute to use as id directly or
    list of mapped attributes to use for auto-generating row ids via sha256 hashing.
    If None, all mapped attributes will be used for hashing.
    """

    hash_id_with_path: bool = False
    """If True, the id will be generated based on the data and the full tree path.
    """

    match_by_attr: bool | str = False
    """Try to match this mapped data to target table (by given attr)
    before creating a new row.
    """

    conflict_policy: DataConflictPolicy = "raise"
    """Which policy to use if import conflicts occur for this table."""


_SubMap = tuple[dict | list, TableMap | list[TableMap]]
"""Combination of data and mapping for a subtree."""


def _map_sublayer(
    node: dict, mapping: _RelationalMap
) -> tuple[pd.Series, dict[str, _SubMap]]:
    """Extract hierarchical data into set of scalar attributes + ref'd data objects."""
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

    refs = {
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
            sub_row, sub_refs = _map_sublayer(sub_node, cast(dict, sub_map))
            cols = {**cols, **sub_row}
            refs = {**refs, **sub_refs}

    return pd.Series(cols, dtype=object), refs


def _gen_row_hash(
    row: pd.Series,
    context_path: list[str | int] | None = None,
    hash_subset: list[str] | None = None,
) -> str:
    """Generate hash for given row."""
    hash_row = (
        row[list(set(hash_subset) & set(row.index))] if hash_subset is not None else row
    )
    row_id = gen_str_hash(
        hash_row.to_dict()
        if context_path is not None
        else (hash_row.to_dict(), context_path)
    )

    return row_id


_DictDB: TypeAlias = dict[str, dict[Hashable, dict[str, Any]]]
"""All-dictionary representation of a relational database for performant extending."""

_Rels: TypeAlias = dict[tuple[str, str], tuple[str, str]]
"""Relations between tables, indexed by the referencing table and column."""

_JoinTables: TypeAlias = set[str]
"""Names of all join tables in the database."""

RelDB = tuple[_DictDB, _Rels, _JoinTables]
"""Full relational database representation."""


def _handle_refs(  # noqa: C901
    mapping: TableMap,
    db: RelDB,
    row: pd.Series,
    refs: list[tuple[str | None, _SubMap]],
    collect_conflicts: bool = False,
    _all_conflicts: DataConflicts | None = None,
    _path: list[str | int] | None = None,
) -> DataConflicts:
    """Handle references to other tables."""
    _all_conflicts = _all_conflicts or {}
    _path = _path or []
    database, relations, join_tables = db

    # Handle nested data, which is to be extracted into separate tables and referenced.
    for attr, (sub_data, sub_maps) in refs:
        # Get info about the ref table to use from mapping
        # (or generate a new for the ref table).

        if not isinstance(sub_maps, list):
            sub_maps = [sub_maps]

        for sub_map in sub_maps:
            join_table_name = sub_map.join_table_name
            join_table_exists = False
            alt_join_table_names = [
                f"{mapping.table}_{sub_map.table}",
                f"{sub_map.table}_{mapping.table}",
            ]

            if join_table_name is None:
                join_table_name = alt_join_table_names[0]
                for name in alt_join_table_names:
                    if name in database:
                        join_table_name = name
                        join_table_exists = True
                        break
            else:
                join_table_exists = join_table_name in database

            if not isinstance(sub_data, list):
                sub_data = [sub_data]

            for i, sub_data_item in enumerate(sub_data):
                _sub_path = [_path, attr, i]

                rel_row, _all_conflicts = _tree_to_db(
                    sub_data_item,
                    sub_map,
                    db,
                    collect_conflicts,
                    _all_conflicts,
                    _sub_path,
                )

                if sub_map.link_type == "n-m":
                    # Map via join table.
                    if not join_table_exists:
                        database[join_table_name] = {}
                        join_table_exists = True

                    ref_row, _ = (
                        _map_sublayer(sub_data_item, sub_map.join_table_map)
                        if isinstance(sub_data_item, dict)
                        and sub_map.join_table_map is not None
                        else (pd.Series(dtype=object), None)
                    )

                    left_col = f"{attr}_of" if attr is not None else mapping.table
                    relations[(join_table_name, left_col)] = (
                        mapping.table,
                        "_id",
                    )
                    ref_row[left_col] = row.name

                    right_col = attr if attr is not None else sub_map.table
                    relations[(join_table_name, right_col)] = (
                        sub_map.table,
                        "_id",
                    )
                    ref_row[right_col] = rel_row.name

                    ref_row.name = _gen_row_hash(ref_row, _sub_path)

                    database[join_table_name][ref_row.name] = ref_row.to_dict()
                    join_tables |= {join_table_name}
                elif sub_map.link_type == "1-n":
                    # Map via direct reference from children to parent.
                    col = f"{attr}_of" if attr is not None else mapping.table

                    if col in rel_row.index:
                        assert rel_row[col] is None or rel_row[col] == row.name

                    relations[(sub_map.table, col)] = (
                        mapping.table,
                        "_id",
                    )

                    database[sub_map.table][rel_row.name] = {
                        **rel_row.to_dict(),
                        col: row.name,
                    }
                elif sub_map.link_type == "n-1":
                    # Map via direct reference from parents to child.
                    col = attr if attr is not None else sub_map.table

                    if col in row.index:
                        assert row[col] is None or row[col] == row.name

                    relations[(mapping.table, col)] = (
                        sub_map.table,
                        "_id",
                    )

                    database[mapping.table][row.name] = {
                        **row.to_dict(),
                        col: rel_row.name,
                    }

    return _all_conflicts


def _tree_to_db(  # noqa: C901
    data: dict | str,
    mapping: TableMap,
    db: RelDB,
    collect_conflicts: bool = False,
    _all_conflicts: DataConflicts | None = None,
    _path: list[str | int] | None = None,
) -> tuple[pd.Series, DataConflicts]:
    """Transform recursive dictionary data into relational format."""
    _all_conflicts = _all_conflicts or {}
    _path = _path or []
    database, _, _ = db

    # Initialize new table as defined by mapping, if it doesn't exist yet.
    if mapping.table not in database:
        database[mapping.table] = {}

    resolved_map = (
        mapping.map(data) if isinstance(mapping.map, Callable) else mapping.map
    )

    # Extract row of data attributes and refs to other objects.
    row = None
    refs: list[tuple[str | None, _SubMap]] = []
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
            row, ref_dict = _map_sublayer(data, resolved_map)
            refs = [*refs, *ref_dict.items()]
        else:
            raise TypeError(
                f"Unsupported mapping type {type(resolved_map)}"
                f" for data of type {type(data)}"
            )

    row.name = (
        row[mapping.id_attr]
        if mapping.id_type == "attr" and isinstance(mapping.id_attr, str)
        else (
            _gen_row_hash(
                row,
                _path,
                (
                    [mapping.id_attr]
                    if isinstance(mapping.id_attr, str)
                    else mapping.id_attr
                ),
            )
            if mapping.id_type == "hash"
            else str(uuid4())[-10:]
        )
    )

    if not isinstance(row.name, str | int):
        raise ValueError(
            f"Value of `'{mapping.id_attr}'` (`TableMap.id_attr`) "
            f"must be a string or int for all objects, but received {row.name}"
        )

    if mapping.ext_maps is not None:
        assert isinstance(data, dict)
        refs = [*refs, *((m.link_attr, (data, m)) for m in mapping.ext_maps)]

    _all_conflicts = _handle_refs(
        mapping, db, row, refs, collect_conflicts, _all_conflicts
    )

    if not row.empty:
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
                    lambda i: i[1][match_by] == match_to,
                    database[mapping.table].items(),
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
                row = pd.Series(
                    {**row.loc[list(new_attrs)], **existing_row}, name=row.name
                )
            else:
                row = pd.Series(
                    {**existing_row, **row.loc[list(new_attrs)]}, name=row.name
                )

        # Add row to database table or update it.
        database[mapping.table][row.name] = row.to_dict()

    # Return row (used for recursion).
    return row, _all_conflicts


@overload
def tree_to_db(
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: Literal[True] = ...,
) -> tuple[DB, DataConflicts]: ...


@overload
def tree_to_db(
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: Literal[False] = ...,
) -> DB: ...


def tree_to_db(  # noqa: C901
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: bool = False,
) -> DB | tuple[DB, DataConflicts]:
    """Transform recursive dictionary data into relational format.

    Args:
        data: The data to be transformed
        mapping:
            Configuration for how to performm the mapping.
        collect_conflicts:
            Collect all conflicts and return them, rather than stopping right away.

    Returns:
        The relational database representation of the data.
        If ``collect_conflicts`` is ``True``, a tuple of the database and the conflicts
        is returned.
    """
    df_dict = {}
    rels = {}
    join_tables = set()
    _, conflicts = _tree_to_db(
        data, mapping, (df_dict, rels, join_tables), collect_conflicts
    )
    db = DB(
        table_dfs={
            name: pd.DataFrame.from_dict(df, orient="index").rename_axis(
                "_id", axis="index"
            )
            for name, df in df_dict.items()
        },
        relations=rels,
        join_tables=join_tables,
    )
    return (db, conflicts) if collect_conflicts else db
