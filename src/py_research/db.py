"""Functions and classes for importing new data."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial, reduce
from itertools import chain, groupby, product
from pathlib import Path
from typing import Any, TypeAlias, cast

import pandas as pd
from inflect import engine as inflect_engine

from py_research.hashing import gen_str_hash

Scalar = str | int | float | datetime

AttrMap = Mapping[str, str | bool | dict]
"""Mapping of hierarchical attributes to table rows"""

RelationalMap = Mapping[str, "str | bool | dict | TableMap | list[TableMap]"]


@dataclass
class TableMap:
    """Defines how to map a data item to the database."""

    table: str
    map: RelationalMap | set[str] | str
    name: str | None = None
    link_map: AttrMap | None = None

    hash_id_subset: list[str] | None = None
    """Supply a list of column names as a subset of all columns to use
    for auto-generating row ids via sha256 hashing.
    """
    match_by_attr: bool | str = False
    """Match this mapped data to target table"""

    ext_maps: "list[TableMap] | None" = None
    """Map attributes on the same level to different tables"""


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

TableSelect: TypeAlias = str | tuple[str, pd.DataFrame]
MergePlan: TypeAlias = TableSelect | list[TableSelect | list[TableSelect]]
NodesAndEdges: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]


def _resolve_links(
    mapping: TableMap,
    database: DictDB,
    row: pd.Series,
    links: list[tuple[str | None, SubMap]],
):
    # Handle nested data, which is to be extracted into separate tables and linked.
    for attr, (sub_data, sub_maps) in links:
        # Get info about the link table to use from mapping
        # (or generate a new for the link table).

        if not isinstance(sub_maps, list):
            sub_maps = [sub_maps]

        for sub_map in sub_maps:
            link_table = f"{mapping.table}_{sub_map.table}"
            # Set up link via link table.
            if not isinstance(sub_data, list):
                sub_data = [sub_data]

            if link_table not in database:
                database[link_table] = {}

            for sub_data_item in sub_data:
                if isinstance(sub_data_item, dict):
                    rel_row = _nested_to_relational(sub_data_item, sub_map, database)

                    link_row, _ = (
                        _resolve_relmap(sub_data_item, sub_map.link_map)
                        if sub_map.link_map is not None
                        else (pd.Series(dtype=object), None)
                    )

                    link_row[f"{mapping.table}.id"] = row.name
                    link_row[f"{sub_map.table}.id"] = rel_row.name
                    link_row["attribute"] = attr

                    link_row.name = _gen_row_id(link_row)

                    database[link_table][link_row.name] = link_row.to_dict()


def _nested_to_relational(
    data: dict | str,
    mapping: TableMap,
    database: DictDB,
) -> pd.Series:
    # Initialize new table as defined by mapping, if it doesn't exist yet.
    if mapping.table not in database:
        database[mapping.table] = {}

    # Extract row of data attributes and links to other objects.
    row = None
    links: list[tuple[str | None, SubMap]] = []
    # If mapping is only a string, extract the target attr directly.
    if isinstance(data, str):
        assert isinstance(mapping.map, str)
        row = pd.Series({mapping.map: data}, dtype=object)
    else:
        if isinstance(mapping.map, set):
            row = pd.Series(
                {k: v for k, v in data.items() if k in mapping.map}, dtype=object
            )
        elif isinstance(mapping.map, dict):
            row, link_dict = _resolve_relmap(data, mapping.map)
            links = [*links, *link_dict.items()]
            # After all data attributes were extracted, generate the row id.
        else:
            raise TypeError(
                f"Unsupported mapping type {type(mapping.map)}"
                f" for data of type {type(data)}"
            )

    row.name = _gen_row_id(row, mapping.hash_id_subset)

    if mapping.ext_maps is not None:
        assert isinstance(data, dict)
        links = [*links, *((None, (data, m)) for m in mapping.ext_maps)]

    _resolve_links(mapping, database, row, links)

    existing_row: pd.Series | None = None
    if mapping.match_by_attr is True:
        # Make sure any existing data in database is consistent with new data.
        match_by = mapping.match_by_attr
        if mapping.match_by_attr is True:
            assert isinstance(mapping.map, str)
            match_by = mapping.map
        match_by = cast(str, match_by)
        # Match to existing row or create new one.
        existing_rows: list[pd.Series] = list(
            filter(
                lambda r: r[match_by] == r[match_by], database[mapping.table].values()
            )
        )
        if len(existing_rows) > 0:
            existing_row = existing_rows[0]
            # Take over the id of the existing row.
            row.name = existing_row.name
    else:
        existing_row = pd.Series(database[mapping.table].get(row.name), dtype=object)

    if existing_row is not None:
        existing_attrs = set(k for k in existing_row.keys() if pd.notna(k))
        new_attrs = set(k for k in row.keys() if pd.notna(k))
        intersect = existing_attrs & new_attrs

        for k in intersect:
            assert existing_row[k] == row[k]

        merged_row = pd.concat([existing_row, row])
        merged_row.name = row.name
        row = merged_row

    # Add row to database table or update it.
    database[mapping.table][row.name] = row.to_dict()

    # Return row (used for recursion).
    return row


class DB(dict[str, pd.DataFrame]):
    """Relational database represented as dictionary of dataframes."""

    @staticmethod
    def from_dicts(db: DictDB) -> "DB":
        """Transform dictionary representation of the database into dataframes."""
        return DB(
            **{
                name: pd.DataFrame.from_dict(data, orient="index")
                for name, data in db.items()
            }
        )

    @staticmethod
    def from_excel(file_path: Path | None = None) -> "DB":
        """Export database into single excel file."""
        file_path = Path.cwd() / "database.xlsx" if file_path is None else file_path

        return DB(
            **{
                str(k): df
                for k, df in pd.read_excel(
                    file_path, sheet_name=None, index_col=0
                ).items()
            }
        )

    @staticmethod
    def from_nested(data: dict, mapping: TableMap) -> "DB":
        """Map hierarchical data to columns and tables in in a new database.

        Args:
            data: Hierarchical data to be mapped to a relational database
            mapping:
                Definition of how to map hierarchical attributes to
                database tables and columns
        """
        db = {}
        _nested_to_relational(data, mapping, db)
        return DB.from_dicts(db)

    def import_nested(self, data: dict, mapping: TableMap) -> "DB":
        """Map hierarchical data to columns and tables in database & insert new data.

        Args:
            data: Hierarchical data to be mapped to a relational database
            mapping:
                Definition of how to map hierarchical attributes to
                database tables and columns
            database: Relational database to map data to
        """
        dict_db = self.to_dicts()
        _nested_to_relational(data, mapping, dict_db)
        return DB.from_dicts(dict_db)

    def to_dicts(self) -> DictDB:
        """Transform database into dictionary representation."""
        return {name: df.to_dict(orient="index") for name, df in self.items()}

    def copy(self, deep: bool = True) -> "DB":
        """Create a copy of this database, optionally deep."""
        return DB(**{name: (df.copy() if deep else df) for name, df in self.items()})

    def to_excel(self, file_path: Path | None = None) -> None:
        """Export database into single excel file."""
        file_path = Path.cwd() / "database.xlsx" if file_path is None else file_path

        writer = pd.ExcelWriter(  # pylint: disable=E0110:abstract-class-instantiated
            file_path,
            engine="openpyxl",
        )

        for name, df in self.items():
            df.to_excel(writer, sheet_name=name, index=True)

        writer.close()

    def merge(
        self,
        base: TableSelect,
        plan: MergePlan,
        auto_prefix: bool = True,
    ) -> pd.DataFrame:
        """Merge selected database tables according to ``plan``.

        Auto-resolves links via join tables or direct foreign keys
        and allows for subsitituting filtered/extended versions of tables
        via supplying a dict as merge plan or explicitly via ``subs``.
        """

        def double_merge(
            left: tuple[str, pd.DataFrame], right: tuple[str, pd.DataFrame]
        ) -> tuple[str, pd.DataFrame]:
            """reduce-compatible double-merge of two tables via a third join table."""
            left_name, left_df = left
            right_name, right_df = right

            left_fk = f"{left_name}.{left_df.index.name or 'id'}"

            middle_name = f"{left_name}_{right_name}"
            middle_df = self.get(middle_name)

            if right_df is not None:
                right_fk = f"{right_name}.{right_df.index.name or 'id'}"

                if left_fk in right_df.columns:
                    return (
                        left_name,
                        (
                            left_df.rename(columns=lambda c: f"{left_name}.{c}")
                            if auto_prefix
                            else left_df
                        ).merge(
                            right_df,
                            left_index=True,
                            right_on=left_fk,
                            how="left",
                        ),
                    )
                elif right_fk in left_df.columns:
                    return (
                        left_name,
                        (
                            left_df.rename(columns=lambda c: f"{left_name}.{c}")
                            if auto_prefix
                            else left_df
                        ).merge(
                            right_df,
                            left_on=right_fk,
                            right_index=True,
                            how="left",
                        ),
                    )

            assert middle_df is not None

            left_merge = (
                left_df.rename(columns=lambda c: f"{left_name}.{c}")
                if auto_prefix
                else left_df
            ).merge(
                middle_df,
                left_index=True,
                right_on=left_fk,
                how="left",
            )

            return (
                left_name,
                (
                    left_merge.merge(
                        right_df.rename(columns=lambda c: f"{right_name}.{c}")
                        if auto_prefix
                        else right_df,
                        left_on=f"{right_name}.{right_df.index.name or 'id'}",
                        right_index=True,
                        how="left",
                    )
                    if right_df is not None
                    else left_merge
                ),
            )

        plan = plan if isinstance(plan, list) else [plan]

        base_name, base_df = (base, self[base]) if isinstance(base, str) else base

        merged: list[pd.DataFrame] = []
        for path in plan:
            # Figure out pair-wise joins and save them to a list.

            path = path if isinstance(path, list) else [path]
            path = [(p, self.get(p)) if isinstance(p, str) else p for p in path]
            path = [(base_name, base_df), *path]

            # Perform reduction to aggregate all tables into one.
            merged.append(reduce(double_merge, path)[1])

        overlap_cols = [f"{base}.{col}" for col in base_df.columns]
        base_index = f"{base_name}.{base_df.index.name or 'id'}"

        return (
            reduce(
                partial(pd.merge, on=base_index, how="outer"),
                [df.drop(columns=overlap_cols) for df in merged[1:]],
                merged[0],
            )
            if len(merged) > 1
            else merged[0]
        )

    def to_graph(self, nodes: list[TableSelect]) -> NodesAndEdges:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        # Concat all node tables into one.
        node_dfs = [
            (self[n] if isinstance(n, str) else n[1])
            .reset_index()
            .rename(columns={"index": "db_index"})
            for n in nodes
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "id"})
        )

        # Find all link tables between the given node tables.
        node_names = [n if isinstance(n, str) else n[0] for n in nodes]
        edge_tables = {
            e: (t1, t2)
            for t1, t2 in product(node_names, repeat=2)
            if (e := f"{t1}_{t2}") in self
        }

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                self[link_tab]
                .merge(
                    node_df.loc[node_df["table"] == tabs[0]],
                    left_on=f"{tabs[0]}.{self[tabs[0]].index.name or 'id'}",
                    right_on="db_index",
                )
                .rename(columns={"id": "source"})
                .merge(
                    node_df.loc[node_df["table"] == tabs[1]],
                    left_on=f"{tabs[1]}.{self[tabs[1]].index.name or 'id'}",
                    right_on="db_index",
                )
                .rename(columns={"id": "target"})[["source", "target"]]
                for link_tab, tabs in edge_tables.items()
            ],
            ignore_index=True,
        )

        return node_df, edge_df

    def describe(self) -> dict[str, str]:
        """Describe this database."""
        return {
            name: f"{len(df)} rows x {len(df.columns)} cols"
            for name, df in self.items()
        }
