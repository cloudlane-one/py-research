"""Functions and classes for transforming between recursive and relational format."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, groupby
from typing import Any, Literal, TypeAlias, cast, overload

import pandas as pd

from py_research.hashing import gen_str_hash

from .base import DB, SingleTable
from .conflicts import DataConflictError, DataConflictPolicy, DataConflicts

Scalar = str | int | float | datetime

AttrMap = Mapping[str, str | bool | dict]
"""Mapping of hierarchical attributes to table rows"""

RelationalMap = Mapping[str, "str | bool | dict | TableMap | list[TableMap]"]


@dataclass
class TableMap:
    """Defines how to map a data item to the database."""

    table: str
    """Name of the table to map to."""

    map: RelationalMap | set[str] | str | Callable[
        [dict | str], RelationalMap | set[str] | str
    ]
    """Mapping of hierarchical attributes to table columns."""

    link_type: Literal["1-n", "n-1", "n-m"] = "n-m"
    """Type of link between this table and the parent table."""

    link_table_name: str | None = None
    """Name of the link table to use for this table."""

    link_map: AttrMap | None = None
    """Mapping of hierarchical attributes to link table rows."""

    hash_id_subset: list[str] | None = None
    """Supply a list of column names as a subset of all columns to use
    for auto-generating row ids via sha256 hashing.
    """

    match_by_attr: bool | str = False
    """Match this mapped data to target table (by given attr)."""

    ext_maps: "list[TableMap] | None" = None
    """Map attributes on the same level to different tables"""

    ext_attr: str | None = None
    """Override attr to use when linking this table with a parent table."""

    id_attr: str | None = None
    """Use given attr as id directly, no hashing."""

    conflict_policy: DataConflictPolicy = "raise"
    """Which policy to use if import conflicts occur for this table."""


SubMap = tuple[dict | list, TableMap | list[TableMap]]


def _map_sublayer(
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
            sub_row, sub_links = _map_sublayer(sub_node, cast(dict, sub_map))
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


def _handle_links(  # noqa: C901
    mapping: TableMap,
    database: DictDB,
    row: pd.Series,
    links: list[tuple[str | None, SubMap]],
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

        relations = {}

        for sub_map in sub_maps:
            link_table_name = sub_map.link_table_name
            link_table_exists = True
            alt_link_table_names = [
                sub_map.link_table_name,
                f"{mapping.table}_{sub_map.table}",
                f"{sub_map.table}_{mapping.table}",
            ]

            if link_table_name is None:
                link_table_name = alt_link_table_names[0]
                for name in alt_link_table_names:
                    if name in database:
                        link_table_name = name
                        link_table_exists = True
                        break
            else:
                link_table_exists = link_table_name in database

            if not link_table_exists:
                database[link_table_name] = {}

            if not isinstance(sub_data, list):
                sub_data = [sub_data]

            for sub_data_item in sub_data:
                rel_row, _all_conflicts = _tree_to_db(
                    sub_data_item,
                    sub_map,
                    database,
                    collect_conflicts,
                    _all_conflicts,
                )

                if sub_map.link_type == "n-m":
                    # Map via join table.
                    link_row, _ = (
                        _map_sublayer(sub_data_item, sub_map.link_map)
                        if isinstance(sub_data_item, dict)
                        and sub_map.link_map is not None
                        else (pd.Series(dtype=object), None)
                    )

                    left_col = f"{attr}_of" if attr is not None else mapping.table
                    relations[(link_table_name, left_col)] = (
                        mapping.table,
                        "_id",
                    )
                    link_row[left_col] = row.name

                    right_col = attr if attr is not None else sub_map.table
                    relations[(link_table_name, right_col)] = (
                        sub_map.table,
                        "_id",
                    )
                    link_row[right_col] = rel_row.name

                    link_row.name = _gen_row_id(link_row)

                    database[link_table_name][link_row.name] = link_row.to_dict()
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
    links: list[tuple[str | None, SubMap]] = []
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
            row, link_dict = _map_sublayer(data, resolved_map)
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

    _all_conflicts = _handle_links(
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
def tree_to_db(
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: Literal[True] = ...,
) -> tuple[DB, DataConflicts]:
    ...


@overload
def tree_to_db(
    data: dict | str,
    mapping: TableMap,
    collect_conflicts: bool = ...,
) -> DB:
    ...


def tree_to_db(  # noqa: C901
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
    _, conflicts = _tree_to_db(data, mapping, df_dict, collect_conflicts)
    db = DB(
        table_dfs={
            name: pd.DataFrame.from_dict(df, orient="index").rename_axis(
                "id", axis="index"
            )
            for name, df in df_dict.items()
        }
    )
    return (db, conflicts) if collect_conflicts else db


NodesAndEdges: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]


def db_to_graph(db: DB, nodes: Sequence[SingleTable]) -> NodesAndEdges:
    """Export links between select database objects in a graph format.

    E.g. for usage with `Gephi`_

    .. _Gephi: https://gephi.org/
    """
    # Concat all node tables into one.
    node_dfs = [db[n.name].df.reset_index().assign(table=n.name) for n in nodes]
    node_df = (
        pd.concat(node_dfs, ignore_index=True)
        .reset_index()
        .rename(columns={"index": "id"})
    )

    # Find all link tables between the given node tables.
    node_names = [n.name for n in nodes]
    relations = pd.concat(
        [db._get_rels(sources=[j], targets=node_names) for j in db.join_tables]
    )

    # Concat all edges into one table.
    edge_df = pd.concat(
        [
            db[str(source_table)]
            .df.merge(
                node_df.loc[node_df["table"] == rel.iloc[0]["target_table"]],
                left_on=rel.iloc[0]["source_col"],
                right_on=rel.iloc[0]["target_col"],
            )
            .rename(columns={"id": "source"})
            .merge(
                node_df.loc[node_df["table"] == rel.iloc[1]["target_table"]],
                left_on=rel.iloc[1]["source_col"],
                right_on=rel.iloc[1]["target_col"],
            )
            .rename(columns={"id": "target"})[["source", "target"]]
            for source_table, rel in relations.groupby("source_table")
            if len(rel) == 2  # We do not export hyper-graphs.
        ],
        ignore_index=True,
    )

    return node_df, edge_df
