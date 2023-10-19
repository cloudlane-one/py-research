"""Functions and classes for importing new data."""

import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial, reduce
from itertools import chain, groupby, product
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypeAlias, cast

import pandas as pd
from inflect import engine as inflect_engine

from py_research.hashing import gen_str_hash

ImportConflictPolicy: TypeAlias = Literal["raise", "ignore", "override"]


class ImportConflictError(ValueError):
    """Irreconsilable conflicts during import of new data into an existing database."""

    def __init__(  # noqa: D107
        self, conflicts: dict[tuple[str, Hashable, str], tuple[Any, Any]]
    ) -> None:
        self.conflicts = conflicts
        super().__init__(
            f"Conflicting values: {conflicts}"
            if len(conflicts) < 5
            else f"{len(conflicts)} in table-columns "
            + str(set((k[0], k[2]) for k in conflicts.keys()))
        )


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

    conflict_policy: ImportConflictPolicy = "raise"
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

TableSelect: TypeAlias = str | tuple[str, pd.DataFrame]
MergePlan: TypeAlias = TableSelect | Iterable[TableSelect | Iterable[TableSelect]]
NodesAndEdges: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]


def _resolve_links(
    mapping: TableMap,
    database: DictDB,
    row: pd.Series,
    links: list[tuple[str | dict[str, Any] | None, SubMap]],
):
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
                    rel_row = _nested_to_relational(sub_data_item, sub_map, database)

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


def _nested_to_relational(  # noqa: C901
    data: dict | str,
    mapping: TableMap,
    database: DictDB,
) -> pd.Series:
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

    _resolve_links(mapping, database, row, links)

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

        if (
            mapping.conflict_policy == "raise"
            and len(intersect) != 0
            and any(existing_row[k] != row[k] for k in intersect)
        ):
            raise ImportConflictError(
                {
                    (mapping.table, row.name, c): (existing_row[c], row[c])
                    for c in intersect
                    if existing_row[c] != row[c]
                }
            )
        if mapping.conflict_policy == "ignore":
            row = pd.Series({**row.loc[list(new_attrs)], **existing_row}, name=row.name)
        else:
            row = pd.Series({**existing_row, **row.loc[list(new_attrs)]}, name=row.name)

    # Add row to database table or update it.
    database[mapping.table][row.name] = row.to_dict()

    # Return row (used for recursion).
    return row


class DFDB(dict[str, pd.DataFrame]):
    """Relational database represented as dictionary of dataframes."""

    @staticmethod
    def from_dicts(db: DictDB) -> "DFDB":
        """Transform dictionary representation of the database into dataframes."""
        return DFDB(
            **{
                name: pd.DataFrame.from_dict(data, orient="index")
                for name, data in db.items()
            }
        )

    @staticmethod
    def from_excel(file_path: Path | str | None = None) -> "DFDB":
        """Export database into single excel file."""
        file_path = Path.cwd() / "database.xlsx" if file_path is None else file_path

        return DFDB(
            **{
                str(k): df
                for k, df in pd.read_excel(
                    file_path, sheet_name=None, index_col=0
                ).items()
            }
        )

    @staticmethod
    def from_nested(data: dict, mapping: TableMap) -> "DFDB":
        """Map hierarchical data to columns and tables in in a new database.

        Args:
            data: Hierarchical data to be mapped to a relational database
            mapping:
                Definition of how to map hierarchical attributes to
                database tables and columns
        """
        db = {}
        _nested_to_relational(data, mapping, db)
        return DFDB.from_dicts(db)

    def import_nested(self, data: dict, mapping: TableMap) -> "DFDB":
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
        return DFDB.from_dicts(dict_db)

    def combine(
        self,
        other: "DFDB",
        conflict_policy: ImportConflictPolicy
        | dict[str, ImportConflictPolicy | dict[str, ImportConflictPolicy]] = "raise",
    ) -> "DFDB":
        """Import other database into this one, returning a new database object.

        Args:
            other: Other database to import
            conflict_policy:
                Policy to use for resolving conflicts. Can be a global setting,
                per-table via supplying a dict with table names as keys, or per-column
                via supplying a dict of dicts.

        Returns:
            New database representing a merge of information in ``self`` and ``other``.
        """
        # Get the union of all table (names) in both databases.
        tables = set(self.keys()) | set(other.keys())

        # Set up variable to contain the merged database.
        merged: dict[str, pd.DataFrame] = {}

        for t in tables:
            if t not in self:
                merged[t] = other[t]
            elif t not in other:
                merged[t] = self[t]
            # Perform more complicated matching if table exists in both databases.
            else:
                # Align index and columns of both tables.
                left, right = self[t].align(other[t])

                # First merge data, ignoring conflicts per default.
                result = left.combine_first(right)

                # Find conflicts, i.e. same-index & same-column values
                # that are different in both tables and neither of them is NaN.
                conflicts = (left == right) | left.isna() | right.isna()

                if any(conflicts):
                    # Deal with conflicts according to `conflict_policy`.

                    # Determine default policy.
                    default_policy: ImportConflictPolicy = (
                        ("raise" if isinstance(cp := conflict_policy[t], dict) else cp)
                        if isinstance(conflict_policy, dict)
                        else conflict_policy
                    )
                    # Expand default policy to all columns.
                    policies: dict[str, ImportConflictPolicy] = {
                        c: default_policy for c in conflicts.columns
                    }
                    # Assign column-level custom policies.
                    if isinstance(conflict_policy, dict) and isinstance(
                        cp := conflict_policy[t], dict
                    ):
                        policies = {**policies, **cp}

                    errors = {}

                    # Iterate over columns and associated policies:
                    for c, p in policies.items():
                        # Only do conflict resolution if there are any for this col.
                        if any(conflicts[c]):
                            match (p):
                                case "raise":
                                    # Record all conflicts.
                                    errors = {
                                        **errors,
                                        **{
                                            (t, i, c): (lv, rv)
                                            for (i, lv), (i, rv) in zip(
                                                left.loc[conflicts[c]][c].items(),
                                                right.loc[conflicts[c]][c].items(),
                                            )
                                        },
                                    }

                                case "override" if len(errors) == 0:
                                    # Override conflicts with data from left.
                                    result[c][conflicts[c]] = right[conflicts[c]]
                                    result[c] = result[c]

                                case "ignore" if len(errors) == 0:
                                    # Nothing to do here.
                                    pass

                merged[t] = result

        return DFDB(merged)

    def __or__(self, other: "DFDB") -> "DFDB":  # noqa: D105
        return self.combine(other)

    def relations(self) -> pd.DataFrame:
        """Return dataframe containing all edges of the DB's relation DAG."""
        return pd.DataFrame.from_records(
            [
                (t, m[1], m[2])
                for t, df in self.items()
                for c in df.columns
                if (m := re.match(r"(\w+)\.(\w+)\.?\w*", c))
            ],
            columns=["src_table", "target_table", "target_col"],
        )

    def valid_refs(self) -> pd.DataFrame:
        """Return all valid references contained in this database."""
        rel_vals = []
        rel_keys = []
        for t, df in self.items():
            df = df.rename_axis("id", axis="index") if not df.index.name else df

            rels = pd.DataFrame.from_records(
                [
                    (m[1], m[2], m[0])
                    for c in df.columns
                    if (m := re.match(r"(\w+)\.(\w+)\.?\w*", c))
                ],
                columns=["target_table", "target_col", "fk_col"],
            )

            for tt, r in rels.groupby("target_table"):
                f_df = pd.DataFrame(
                    {
                        c: (
                            df[[c]]
                            .reset_index()
                            .merge(
                                self[str(tt)].pipe(
                                    lambda df: df.rename_axis("id", axis="index")
                                    if not df.index.name
                                    else df
                                ),
                                left_on=c,
                                right_on=tc,
                            )
                            .set_index(df.index.name)[str(c)]
                            .groupby(df.index.name)
                            .agg("first")
                        )
                        for c, tc in r[["fk_col", "target_col"]].itertuples(index=False)
                    }
                )
                rel_vals.append(f_df)
                rel_keys.append((t, tt))

        return pd.concat(rel_vals, keys=rel_keys, names=["src_table", "target_table"])

    def trim(self) -> "DFDB":
        """Return new database without orphan data (data w/ no refs to or from)."""
        # Get the status of each single reference.
        valid_refs = self.valid_refs()

        result = {}
        for t, df in self.items():
            f = pd.Series(False, index=df.index)

            try:
                # Include all rows with any valid outgoing reference.
                outgoing = valid_refs.loc[(t, slice(None), slice(None)), :]
                assert isinstance(outgoing, pd.DataFrame)
                for _, refs in outgoing.groupby("target_table"):
                    f |= (
                        refs.droplevel(["src_table", "target_table"])
                        .notna()
                        .any(axis="columns")
                    )
            except KeyError:
                pass

            try:
                # Include all rows with any valid incoming reference.
                incoming = valid_refs.loc[(slice(None), t, slice(None)), :]
                assert isinstance(incoming, pd.DataFrame)
                for _, refs in incoming.groupby("src_table"):
                    for _, col in refs.items():
                        f |= pd.Series(True, index=col.unique())
            except KeyError:
                pass

            result[t] = df.loc[f]

        return DFDB(result)

    def centered_trim(
        self, centers: list[str], circuit_breakers: list[str] | None = None
    ) -> "DFDB":
        """Return new database minus data without (indirect) refs to any given table."""
        circuit_breakers = circuit_breakers or []

        # Get the status of each single reference.
        valid_refs = self.valid_refs()

        current_stage = {c: set(self[c].index) for c in centers}
        visit_counts = {t: pd.Series(0, index=df.index) for t, df in self.items()}
        visited: set[str] = set()

        while any(len(s) > 0 for s in current_stage.values()):
            next_stage = {t: set() for t in self.keys()}
            for t, idx in current_stage.items():
                if t in visited and t in circuit_breakers:
                    continue

                current, additions = visit_counts[t].align(
                    pd.Series(1, index=list(idx)), fill_value=0
                )
                visit_counts[t] = current + additions

                idx_sel = list(
                    idx & set(visit_counts[t].loc[visit_counts[t] == 1].index)
                )

                if len(idx_sel) == 0:
                    continue

                visited |= set([t])

                try:
                    # Include all rows with any valid outgoing reference.
                    outgoing = valid_refs.loc[(t, slice(None), idx_sel), :]
                    assert isinstance(outgoing, pd.DataFrame)
                    for tt, refs in outgoing.groupby("target_table"):
                        for _, col in refs.items():
                            next_stage[str(tt)] |= set(col.dropna().unique())
                except KeyError:
                    pass

                try:
                    # Include all rows with any valid incoming reference.
                    incoming = valid_refs.loc[(slice(None), t, slice(None)), :]
                    assert isinstance(incoming, pd.DataFrame)
                    for st, refs in incoming.groupby("src_table"):
                        next_stage[str(st)] |= set(
                            refs.droplevel(["src_table", "target_table"])
                            .isin(idx_sel)
                            .any(axis="columns")
                            .replace(False, pd.NA)
                            .dropna()
                            .index
                        )
                except KeyError:
                    pass

            current_stage = next_stage

        return DFDB({t: self[t].loc[rc > 0] for t, rc in visit_counts.items()})

    def filter(
        self, filters: dict[str, pd.Series], extra_cb: list[str] | None = None
    ) -> "DFDB":
        """Return new db only containing data related to rows matched by ``filters``.

        Args:
            filters: Mapping of table names to boolean filter series
            extra_cb:
                Additional circuit breakers (on top of the filtered tables)
                to use when trimming the database according to the filters
        """
        # Filter out unmatched rows of filter tables.
        new_db = DFDB(
            {t: (df[filters[t]] if t in filters else df) for t, df in self.items()}
        )

        # Always use the filter tables as circuit_breakes.
        # Otherwise filtered-out rows may be re-included.
        cb = list(set(filters.keys()) | set(extra_cb or []))

        # Trim all other tables such that only rows with (indirect) references to
        # remaining rows in filter tables are left.
        return new_db.centered_trim(list(filters.keys()), circuit_breakers=cb)

    def to_dicts(self) -> DictDB:
        """Transform database into dictionary representation."""
        return {name: df.to_dict(orient="index") for name, df in self.items()}

    def copy(self, deep: bool = True) -> "DFDB":
        """Create a copy of this database, optionally deep."""
        return DFDB(**{name: (df.copy() if deep else df) for name, df in self.items()})

    def to_excel(self, file_path: Path | str | None = None) -> None:
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
        subs: dict[str, pd.DataFrame] | None = None,
        auto_prefix: bool = True,
    ) -> pd.DataFrame:
        """Merge selected database tables according to ``plan``.

        Auto-resolves links via join tables or direct foreign keys
        and allows for subsitituting filtered/extended versions of tables.
        """

        def double_merge(
            left: tuple[str, pd.DataFrame], right: tuple[str, pd.DataFrame]
        ) -> tuple[str, pd.DataFrame]:
            """reduce-compatible double-merge of two tables via a third join table."""
            left_name, left_df = left
            right_name, right_df = right

            left_fk = f"{left_name}.{left_df.index.name or 'id'}" + (
                "" if left_name != right_name else ".0"
            )

            middle_name = f"{left_name}_{right_name}"
            middle_name_alt = f"{right_name}_{left_name}"

            middle_df = None
            if middle_name in self:
                middle_df = self[middle_name]
            elif middle_name_alt in self:
                middle_df = self[middle_name_alt]
                middle_name = middle_name_alt

            if right_df is not None:
                right_fk = f"{right_name}.{right_df.index.name or 'id'}" + (
                    "" if left_name != right_name else ".1"
                )

                if left_fk in right_df.columns:
                    # Case 1:
                    # - Right table exists in db.
                    # - Right table references left directly via foreign key.
                    return (
                        right_name,
                        left_df.merge(
                            right_df.rename(columns=lambda c: f"{right_name}.{c}"),
                            left_index=True,
                            right_on=left_fk,
                            how="left",
                            suffixes=(".0", ".1"),
                        ),
                    )
                elif right_fk in left_df.columns:
                    # Case 2:
                    # - Right table exists in db.
                    # - Left table references right directly via foreign key.
                    return (
                        right_name,
                        left_df.merge(
                            right_df.rename(columns=lambda c: f"{right_name}.{c}"),
                            left_on=right_fk,
                            right_index=True,
                            how="left",
                            suffixes=(".0", ".1"),
                        ),
                    )

            # If case 1 and 2 do not apply, links have to be resolved indirectly
            # via middle (join) table. Hence this table has to exist in the db.
            assert middle_df is not None

            left_merge = left_df.merge(
                middle_df,
                left_index=True,
                right_on=left_fk,
                how="left",
            )

            if right_df is not None:
                # Case 3:
                # - Right table exists in db.
                # - Right is linked with left table indirectly via middle (join) table.
                right_fk = f"{right_name}.{right_df.index.name or 'id'}"
                return (
                    right_name,
                    left_merge.rename(
                        columns={
                            c: f"{middle_name}.{c}"
                            for c in middle_df.columns
                            if c not in (left_fk, right_fk)
                        }
                    ).merge(
                        right_df.rename(columns=lambda c: f"{right_name}.{c}")
                        if auto_prefix
                        else right_df,
                        left_on=right_fk,
                        right_index=True,
                        how="left",
                        suffixes=(".0", ".1"),
                    ),
                )
            else:
                # Case 4:
                # - Right table does not exist in db.
                # - Virtual right is linked with left table indirectly via middle.
                return (
                    right_name,
                    left_merge.rename(
                        columns={
                            c: f"{middle_name}.{c}"
                            for c in middle_df.columns
                            if c != left_fk and not c.startswith(f"{right_name}.")
                        }
                    ),
                )

        plan = [plan] if isinstance(plan, str) else plan
        subs = subs or {}

        base_name, base_df = (
            (base, subs.get(base) or self[base]) if isinstance(base, str) else base
        )
        base_df = base_df.rename(columns=lambda c: f"{base_name}.{c}")

        merged: list[pd.DataFrame] = []
        for path in plan:
            # Figure out pair-wise joins and save them to a list.

            path = path if isinstance(path, list) else [path]
            path = [
                (p, subs.get(p) or self.get(p)) if isinstance(p, str) else p
                for p in path
            ]
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
            .assign(table=n)
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

    def describe(self) -> dict[str, dict[str, str]]:
        """Describe this database."""
        return {
            "entity tables": {
                name: f"{len(df)} objects x {len(df.columns)} attributes"
                for name, df in self.items()
                if "_" not in name
            },
            "relation tables": {
                name: f"{len(df)} links"
                + (
                    f" x {n_attrs} attributes"
                    if (n_attrs := len(df.columns) - 2) > 0
                    else ""
                )
                for name, df in self.items()
                if "_" in name
            },
        }
