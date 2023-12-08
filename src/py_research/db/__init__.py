"""Omni-purpose, easy to use relational database."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial, reduce
from itertools import product
from pathlib import Path
from typing import Self, TypeAlias

import pandas as pd
from sqlalchemy import URL

from py_research.db.conflicts import DataConflictPolicy

TableSelect: TypeAlias = str | tuple[str, pd.DataFrame]
MergePlan: TypeAlias = TableSelect | Iterable[TableSelect | Iterable[TableSelect]]
NodesAndEdges: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]


@dataclass
class Table:
    """Relational database table."""

    df: pd.DataFrame
    db: "DB"


class DBSchema:
    """Base class for database schemas defined in Python."""

    version: str | None = None


@dataclass
class DB:
    """Relational database consisting of multiple named tables."""

    schema: type[DBSchema] | None = None
    url: URL | Path | None = None
    last_update: date | datetime | None = None

    @staticmethod
    def _df_dict_from_excel(file_path: Path) -> dict[str, pd.DataFrame]:
        """Import database from an excel file."""
        return {
            str(k): df
            for k, df in pd.read_excel(file_path, sheet_name=None, index_col=0).items()
        }

    def _import_df_dict(self, df_dict: dict[str, pd.DataFrame]) -> Self:
        """Import database from an excel file."""
        self.__df_dict = {**self.__df_dict, **df_dict}
        return self

    def __post_init__(self):  # noqa: D105
        if isinstance(self.url, URL):
            raise NotImplementedError("SQL backends via URL are not implemented yet.")

        self.__df_dict = {}
        self.__file_path = None

        # TODO: initialize _meta table from schema spec

        if self.url is not None:
            self.__file_path = (
                Path.cwd() / "database.xlsx" if self.url.is_dir() is None else self.url
            )
            self.__df_dict = DB._df_dict_from_excel(self.__file_path)
            if (
                "_meta" not in self.__df_dict.keys()
                or "_rels" not in self.__df_dict.keys()
            ):
                raise ValueError("Malformatted excel file.")

            # TODO: Validate schema including version
            # TODO: Validate last_update

    def __getitem__(self, name: str) -> Table:  # noqa: D105
        return Table(df=self.__df_dict[name], db=self)

    def combine(
        self,
        other: "DB",
        conflict_policy: DataConflictPolicy
        | dict[str, DataConflictPolicy | dict[str, DataConflictPolicy]] = "raise",
    ) -> "DB":
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
        tables = set(self.__df_dict.keys()) | set(other.__df_dict.keys())

        # Set up variable to contain the merged database.
        merged: dict[str, pd.DataFrame] = {}

        for t in tables:
            if t not in self.__df_dict:
                merged[t] = other[t].df
            elif t not in other.__df_dict:
                merged[t] = self[t].df
            # Perform more complicated matching if table exists in both databases.
            else:
                # Align index and columns of both tables.
                left, right = self[t].df.align(other[t].df)

                # First merge data, ignoring conflicts per default.
                result = left.combine_first(right)

                # Find conflicts, i.e. same-index & same-column values
                # that are different in both tables and neither of them is NaN.
                conflicts = (left == right) | left.isna() | right.isna()

                if any(conflicts):
                    # Deal with conflicts according to `conflict_policy`.

                    # Determine default policy.
                    default_policy: DataConflictPolicy = (
                        ("raise" if isinstance(cp := conflict_policy[t], dict) else cp)
                        if isinstance(conflict_policy, dict)
                        else conflict_policy
                    )
                    # Expand default policy to all columns.
                    policies: dict[str, DataConflictPolicy] = {
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
                            match p:
                                case "override" if len(errors) == 0:
                                    # Override conflicts with data from left.
                                    result[c][conflicts[c]] = right[conflicts[c]]
                                    result[c] = result[c]

                                case "ignore" if len(errors) == 0:
                                    # Nothing to do here.
                                    pass
                                case _:
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

                merged[t] = result

        return DB()._import_df_dict(merged)

    def __or__(self, other: "DB") -> "DB":  # noqa: D105
        return self.combine(other)

    def relations(self) -> pd.DataFrame:
        """Return dataframe containing all edges of the DB's relation DAG."""
        return pd.DataFrame.from_records(
            [
                (t, m[1], m[2])
                for t, df in self.__df_dict.items()
                for c in df.columns
                if (m := re.match(r"(\w+)\.(\w+)\.?\w*", c))
            ],
            columns=["src_table", "target_table", "target_col"],
        )

    def valid_refs(self) -> pd.DataFrame:
        """Return all valid references contained in this database."""
        rel_vals = []
        rel_keys = []
        for t, df in self.__df_dict.items():
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
                                self[str(tt)].df.pipe(
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

    def trim(self) -> "DB":
        """Return new database without orphan data (data w/ no refs to or from)."""
        # Get the status of each single reference.
        valid_refs = self.valid_refs()

        result = {}
        for t, df in self.__df_dict.items():
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

        return DB()._import_df_dict(result)

    def centered_trim(
        self, centers: list[str], circuit_breakers: list[str] | None = None
    ) -> "DB":
        """Return new database minus data without (indirect) refs to any given table."""
        circuit_breakers = circuit_breakers or []

        # Get the status of each single reference.
        valid_refs = self.valid_refs()

        current_stage = {c: set(self[c].df.index) for c in centers}
        visit_counts = {
            t: pd.Series(0, index=df.index) for t, df in self.__df_dict.items()
        }
        visited: set[str] = set()

        while any(len(s) > 0 for s in current_stage.values()):
            next_stage = {t: set() for t in self.__df_dict.keys()}
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

        return DB()._import_df_dict(
            {t: self[t].df.loc[rc > 0] for t, rc in visit_counts.items()}
        )

    def filter(
        self, filters: dict[str, pd.Series], extra_cb: list[str] | None = None
    ) -> "DB":
        """Return new db only containing data related to rows matched by ``filters``.

        Args:
            filters: Mapping of table names to boolean filter series
            extra_cb:
                Additional circuit breakers (on top of the filtered tables)
                to use when trimming the database according to the filters
        """
        # Filter out unmatched rows of filter tables.
        new_db = DB()._import_df_dict(
            {
                t: (df[filters[t]] if t in filters else df)
                for t, df in self.__df_dict.items()
            }
        )

        # Always use the filter tables as circuit_breakes.
        # Otherwise filtered-out rows may be re-included.
        cb = list(set(filters.keys()) | set(extra_cb or []))

        # Trim all other tables such that only rows with (indirect) references to
        # remaining rows in filter tables are left.
        return new_db.centered_trim(list(filters.keys()), circuit_breakers=cb)

    def copy(self, deep: bool = True) -> "DB":
        """Create a copy of this database, optionally deep."""
        return DB()._import_df_dict(
            {name: (df.copy() if deep else df) for name, df in self.__df_dict.items()}
        )

    def save(self, file_path: Path | str | None = None) -> None:
        """Save this database to its default or a custom path."""
        file_path = file_path or self.__file_path or Path.cwd() / "database.xlsx"

        writer = pd.ExcelWriter(  # pylint: disable=E0110:abstract-class-instantiated
            file_path,
            engine="openpyxl",
        )

        for name, df in self.__df_dict.items():
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
            if middle_name in self.__df_dict:
                middle_df = self[middle_name].df
            elif middle_name_alt in self.__df_dict:
                middle_df = self[middle_name_alt].df
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
            (base, subs.get(base) or self[base].df) if isinstance(base, str) else base
        )
        base_df = base_df.rename(columns=lambda c: f"{base_name}.{c}")

        merged: list[pd.DataFrame] = []
        for path in plan:
            # Figure out pair-wise joins and save them to a list.

            path = path if isinstance(path, list) else [path]
            path = [
                (p, subs.get(p) or self.__df_dict.get(p)) if isinstance(p, str) else p
                for p in path
            ]
            path = [(base_name, base_df), *path]

            # Perform reduction to aggregate all tables into one.
            merged.append(reduce(double_merge, path)[1])

        overlap_cols = base_df.columns
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

    def to_graph(self, nodes: Sequence[TableSelect]) -> NodesAndEdges:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        # Concat all node tables into one.
        node_dfs = [
            (self[n].df if isinstance(n, str) else n[1])
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
            if (e := f"{t1}_{t2}") in self.__df_dict
        }

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                self[link_tab]
                .df.merge(
                    node_df.loc[node_df["table"] == tabs[0]],
                    left_on=(
                        tabs[0]
                        + (
                            self[tabs[0]].df.index.name
                            or (".id" if tabs[0] != tabs[1] else ".id.0")
                        )
                    ),
                    right_on="db_index",
                )
                .rename(columns={"id": "source"})
                .merge(
                    node_df.loc[node_df["table"] == tabs[1]],
                    left_on=(
                        tabs[1]
                        + (
                            self[tabs[1]].df.index.name
                            or (".id" if tabs[0] != tabs[1] else ".id.1")
                        )
                    ),
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
                for name, df in self.__df_dict.items()
                if "_" not in name
            },
            "relation tables": {
                name: f"{len(df)} links"
                + (
                    f" x {n_attrs} attributes"
                    if (n_attrs := len(df.columns) - 2) > 0
                    else ""
                )
                for name, df in self.__df_dict.items()
                if "_" in name
            },
        }
