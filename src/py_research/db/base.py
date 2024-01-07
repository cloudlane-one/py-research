"""Base classes and types for relational database package."""

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from py_research.db.conflicts import DataConflictError, DataConflictPolicy


@dataclass
class Table:
    """Table in a relational database."""

    # Attributes:

    db: "DB"
    """Database this table belongs to."""

    df: pd.DataFrame
    """Dataframe containing the data of this table."""

    source_map: str | dict[str, str]
    """Mapping to source tables of this table.

    For single source tables, this is a string containing the name of the source table.

    For multiple source tables, the dataframe hast multi-level columns. ``source_map``
    is then a mapping from the first level of these columns to the source tables.
    """

    # Public methods:

    def filter(self, filter_series: pd.Series) -> "Table":
        """Filter this table.

        Args:
            filter_series: Boolean series to filter the table with.

        Returns:
            New table containing the filtered data.
        """
        return Table(
            self.db,
            self.df.loc[filter_series],
            self.source_map,
        )

    # Dictionary interface:

    def __getitem__(self, name: str) -> pd.Series:  # noqa: D105
        return self.df[name]

    def __contains__(self, name: str) -> bool:  # noqa: D105
        return name in self.df.columns

    def __iter__(self) -> Iterable[Hashable]:  # noqa: D105
        return iter(self.df)

    def __len__(self) -> int:  # noqa: D105
        return len(self.df)

    def keys(self) -> Iterable[str]:  # noqa: D102
        return self.df.keys()

    def get(self, name: str, default: Any = None) -> Any:  # noqa: D102
        return self.df.get(name, default)

    # DataFrame interface:

    @property
    def columns(self) -> Sequence[str | tuple[str, str]]:
        """Return the columns of this table."""
        return self.df.columns.tolist()

    # Internals:

    def _get_incoming_rels(self) -> dict[str, dict[str, str]]:
        all_rels = self.db._get_rels(
            targets=list(self.source_map.values())
            if isinstance(self.source_map, dict)
            else [self.source_map]
        )
        return {
            str(st): dict(df[["source_col", "target_col"]].itertuples(index=False))
            for st, df in all_rels.groupby("source_table")
        }

    def _get_outgoing_rels(self) -> dict[str, dict[str, str]]:
        all_rels = self.db._get_rels(
            sources=list(self.source_map.values())
            if isinstance(self.source_map, dict)
            else [self.source_map]
        )
        return {
            str(tt): dict(df[["source_col", "target_col"]].itertuples(index=False))
            for tt, df in all_rels.groupby("target_table")
        }


class SingleTable(Table):
    """Relational database table with a single source table."""

    # Custom constructor:

    def __init__(
        self,
        name: str,
        db: "DB",
        df: pd.DataFrame,
    ) -> None:
        """Initialize this table.

        Args:
            name: Name of the source table.
            db: Database this table belongs to.
            df: Dataframe containing the data of this table.
        """
        self.name = name
        self.db = db
        self.df = df

    # Computed properties:

    @property
    def source_map(self) -> str:
        """Name of the source table of this table."""
        return self.name

    # Method overrides:

    def filter(self, filter_series: pd.Series) -> "SingleTable":
        """Return a filtered version of this table."""
        return super().filter(filter_series)  # type: ignore[no-any-return]

    # Public methods:

    def merge(self, other: "SingleTable") -> "Table":
        """Merge this table with another, returning a new table.

        Args:
            other: Other table to merge with.

        Returns:
            New table containing the merged data.
            The returned table will have a multi-level column index,
            where the first level references the source table of each column
            via the ``source_map`` attribute.
        """
        if self.db is not other.db:
            raise ValueError("Both tables must be from the same database.")

        merged = self.df.rename(columns=lambda c: f"{self.source_map}->{c}")
        merged = merged.rename_axis(
            f"{self.source_map}->{merged.index.name or 'id'}", axis="index"
        )
        merged_source_map = {self.source_map: self.source_map}

        outgoing = self._get_outgoing_rels()
        for tt, cols in outgoing.items():
            for sc, tc in cols.items():
                if other.source_map == tt:
                    merged = merged.merge(
                        other.df.rename(columns=lambda c: f"{sc}->{c}"),
                        left_on=f"{self.source_map}->{sc}",
                        right_on=f"{sc}->{tc}",
                        how="left",
                    )
                    merged_source_map[sc] = tt

        incoming = self._get_incoming_rels()
        for st, cols in incoming.items():
            for sc, tc in cols.items():
                if other.source_map == st:
                    merged = merged.merge(
                        other.df.rename(columns=lambda c: f"{sc}_of->{c}"),
                        left_on=f"{self.source_map}->{tc}",
                        right_on=f"{sc}_of->{sc}",
                        how="left",
                    )
                    merged_source_map[f"{sc}_of"] = st
                elif st in self.db.join_tables:
                    join_table = self.db[st]
                    joint = join_table._get_outgoing_rels()

                    if other.source_map in joint:
                        # Perform a double merge across a join table.
                        half_merged = merged.merge(
                            join_table.df.rename(
                                columns=lambda c: f"{join_table.source_map}->{c}"
                            ),
                            left_on=f"{self.source_map}->{tc}",
                            right_on=f"{join_table.source_map}->{sc}",
                            how="left",
                        )
                        merged_source_map[join_table.source_map] = join_table.source_map

                        for tt, (sc, tc) in joint.items():
                            if other.source_map == tt:
                                merged = half_merged.merge(
                                    other.df.rename(columns=lambda c: f"{sc}->{c}"),
                                    left_on=f"{join_table.source_map}->{sc}",
                                    right_on=f"{sc}->{tc}",
                                    how="left",
                                )
                                merged_source_map[sc] = tt

        merged = merged.reindex(
            columns=pd.MultiIndex.from_tuples(
                merged.columns.map(lambda c: str(c).split("->")).to_list()
            )
        )

        return Table(self.db, merged, merged_source_map)


class BaseTable(SingleTable):
    """Original table in a relational database."""

    # Custom constuctor:

    def __init__(self, name: str, db: "DB") -> None:
        """Initialize this table.

        Args:
            name: Name of the source table.
            db: Database this table belongs to.
        """
        self.name = name
        self.db = db

    # Computed properties:

    @property
    def df(self) -> pd.DataFrame:
        """Return the dataframe of this table."""
        return self.db.table_dfs[self.name]

    # Public methods:

    def extend(
        self,
        other: "pd.DataFrame",
        conflict_policy: DataConflictPolicy | dict[str, DataConflictPolicy] = "raise",
    ) -> "SingleTable":
        """Extend this table with data from another, returning a new table.

        Args:
            other: Other table to extend with.
            conflict_policy:
                Policy to use for resolving conflicts. Can be a global setting,
                per-column via supplying a dict with column names as keys.

        Returns:
            New table containing the extended data.
        """
        # Align index and columns of both tables.
        left, right = self.df.align(other)

        # First merge data, ignoring conflicts per default.
        extended_df = left.combine_first(right)

        # Find conflicts, i.e. same-index & same-column values
        # that are different in both tables and neither of them is NaN.
        conflicts = ~((left == right) | left.isna() | right.isna())

        if any(conflicts):
            # Deal with conflicts according to `conflict_policy`.

            # Determine default policy.
            default_policy: DataConflictPolicy = (
                conflict_policy if isinstance(conflict_policy, str) else "raise"
            )
            # Expand default policy to all columns.
            policies: dict[str, DataConflictPolicy] = {
                c: default_policy for c in conflicts.columns
            }
            # Assign column-level custom policies.
            if isinstance(conflict_policy, dict):
                policies = {**policies, **conflict_policy}

            # Iterate over columns and associated policies:
            for c, p in policies.items():
                # Only do conflict resolution if there are any for this col.
                if any(conflicts[c]):
                    match p:
                        case "override":
                            # Override conflicts with data from left.
                            extended_df[c][conflicts[c]] = right[conflicts[c]]
                            extended_df[c] = extended_df[c]

                        case "ignore":
                            # Nothing to do here.
                            pass
                        case "raise":
                            # Raise an error.
                            raise DataConflictError(
                                {
                                    (c, c, str(i)): (lv, rv)
                                    for (i, lv), (i, rv) in zip(
                                        left.loc[conflicts[c]][c].items(),
                                        right.loc[conflicts[c]][c].items(),
                                    )
                                }
                            )
                        case _:
                            pass

        return SingleTable(
            self.name,
            self.db,
            extended_df,
        )


def _df_dict_from_excel(file_path: Path) -> dict[str, pd.DataFrame]:
    """Import multiple dataframes from an excel file."""
    return {
        str(k): df
        for k, df in pd.read_excel(file_path, sheet_name=None, index_col=0).items()
    }


def _check_version(lower_bound: str, version: str) -> bool:
    """Check if a version is compatible with a lower bound."""
    lb = tuple(map(int, lower_bound.split(".")))
    v = tuple(map(int, version.split(".")))
    return v >= lb and v[0] == lb[0]


@dataclass
class DB:
    """Relational database consisting of multiple named tables."""

    # Attributes:

    table_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Dataframes containing the data of each table in this database."""

    relations: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    """Relations between tables in this database."""

    join_tables: set[str] = field(default_factory=set)
    """Names of tables that are used as n-to-m join tables in this database."""

    version: str | None = None
    """Version of this database."""

    # Creation:

    @staticmethod
    def load(file_path: Path, version: str | None = None) -> "DB":
        """Load a database from an excel file.

        Args:
            file_path: Path to the excel file.
            version: Minimum version of the database to load.

        Returns:
            Database object.
        """
        df_dict = _df_dict_from_excel(file_path)

        given_version = None
        if "_attributes" in df_dict and "version" in df_dict["_attributes"]:
            given_version = df_dict["_attributes"]["version"]["value"]
            if version and not _check_version(
                given_version,
                version,
            ):
                raise ValueError(
                    f"Requested database version {version} is not compatible with "
                    + f"given database version {given_version}."
                )

        relations = {}
        if "_relations" in df_dict:
            relations = df_dict["_relations"]
            assert isinstance(relations, pd.DataFrame)
            relations = relations.set_index(["source_table", "source_col"]).apply(
                lambda r: (r["target_table"], r["target_col"]),
                axis="columns",
            )
            relations = relations.to_dict()

        join_tables = set()
        if "_join_tables" in df_dict:
            join_tables = set(df_dict["_join_tables"]["name"].tolist())

        return DB(df_dict, relations, join_tables, version)

    # Public methods:

    def describe(self) -> dict[str, dict[str, str]]:
        """Return a description of this database.

        Returns:
            Mapping of table names to table descriptions.
        """
        join_tables = {
            name: f"{len(self[name])} links"
            + (
                f" x {n_attrs} attributes"
                if (n_attrs := len(self[name].columns) - 2) > 0
                else ""
            )
            for name in self.join_tables
        }

        return {
            "tables": {
                name: f"{len(df)} objects x {len(df.columns)} attributes"
                for name, df in self.items()
                if name not in join_tables
            },
            "join tables": join_tables,
        }

    def copy(self, deep: bool = True) -> "DB":
        """Create a copy of this database, optionally deep.

        Args:
            deep: Whether to perform a deep copy (copy all dataframes).

        Returns:
            Copy of this database.
        """
        return DB(
            table_dfs={
                name: (table.df.copy() if deep else table.df)
                for name, table in self.items()
            }
        )

    def extend(
        self,
        other: "DB | dict[str, pd.DataFrame]",
        conflict_policy: DataConflictPolicy
        | dict[str, DataConflictPolicy | dict[str, DataConflictPolicy]] = "raise",
    ) -> "DB":
        """Extend this database with data from another, returning a new database.

        Args:
            other: Other database to extend with.
            conflict_policy:
                Policy to use for resolving conflicts. Can be a global setting,
                per-table via supplying a dict with table names as keys, or per-column
                via supplying a dict of dicts.

        Returns:
            New database containing the extended data.
        """
        # Get the union of all table (names) in both databases.
        tables = set(self.keys()) | set(other.keys())

        # Set up variable to contain the merged database.
        merged: dict[str, pd.DataFrame] = {}

        for t in tables:
            if t not in self:
                merged[t] = other[t].df if isinstance(other, DB) else other[t]
            elif t not in other:
                merged[t] = self[t].df
            # Perform more complicated matching if table exists in both databases.
            else:
                table_conflict_policy = (
                    conflict_policy
                    if isinstance(conflict_policy, str)
                    else conflict_policy.get(t) or "raise"
                )
                merged[t] = (
                    self[t]
                    .extend(
                        other[t].df if isinstance(other, DB) else other[t],
                        conflict_policy=table_conflict_policy,
                    )
                    .df
                )

        return DB(table_dfs=merged)

    def trim(
        self,
        centers: list[str] | None = None,
        circuit_breakers: list[str] | None = None,
    ) -> "DB":
        """Return new database minus orphan data (relative to given ``centers``).

        Args:
            centers: Tables to use as centers for the trim.
            circuit_breakers: Tables to use as circuit breakers for the trim.

        Returns:
            New database containing the trimmed data.
        """
        return (
            self._isotropic_trim()
            if centers is None
            else self._centered_trim(centers, circuit_breakers)
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

        Returns:
            New database containing the filtered data.
            The returned database will only contain the filtered tables and
            all tables that have (indirect) references to them.

        Note:
            This is equivalent to trimming the database with the filtered tables
            as centers and the filtered tables and ``extra_cb`` as circuit breakers.
        """
        # Filter out unmatched rows of filter tables.
        new_db = DB(
            table_dfs={
                t: (table.df[filters[t]] if t in filters else table.df)
                for t, table in self.items()
            },
            relations=self.relations,
            join_tables=self.join_tables,
        )

        # Always use the filter tables as circuit_breakes.
        # Otherwise filtered-out rows may be re-included.
        cb = list(set(filters.keys()) | set(extra_cb or []))

        # Trim all other tables such that only rows with (indirect) references to
        # remaining rows in filter tables are left.
        return new_db.trim(list(filters.keys()), circuit_breakers=cb)

    def save(self, file_path: Path) -> None:
        """Save this database to an excel file.

        Args:
            file_path: Path to excel file. Will be overridden if it exists.
        """
        writer = pd.ExcelWriter(  # pylint: disable=E0110:abstract-class-instantiated
            file_path,
            engine="openpyxl",
        )

        for name, table in self.items():
            table.df.to_excel(writer, sheet_name=name, index=True)

        writer.close()

    # Dictionary interface:

    def __getitem__(self, name: str) -> BaseTable:  # noqa: D105
        if name.startswith("_"):
            raise KeyError(f"Table '{name}' does not exist in this database.")
        return BaseTable(name=name, db=self)

    def __contains__(self, name: str) -> bool:  # noqa: D105
        return name in self.table_dfs and not name.startswith("_")

    def __iter__(self) -> Iterable[str]:  # noqa: D105
        return (k for k in self.table_dfs if not k.startswith("_"))

    def __len__(self) -> int:  # noqa: D105
        return len(list(k for k in self.table_dfs if not k.startswith("_")))

    def keys(self) -> Iterable[str]:  # noqa: D102
        return (k for k in self.table_dfs if not k.startswith("_"))

    def values(self) -> Iterable[BaseTable]:  # noqa: D102
        return [
            BaseTable(name, self) for name in self.table_dfs if not name.startswith("_")
        ]

    def items(self) -> Iterable[tuple[str, BaseTable]]:  # noqa: D102
        return [
            (name, BaseTable(name, self))
            for name in self.table_dfs
            if not name.startswith("_")
        ]

    def get(self, name: str) -> BaseTable | None:  # noqa: D102
        return BaseTable(name, self) if not name.startswith("_") else None

    # Custom operators:

    def __or__(self, other: "DB") -> "DB":  # noqa: D105
        return self.extend(other)

    # Internals:

    def _get_rels(
        self, sources: list[str] | None = None, targets: list[str] | None = None
    ) -> pd.DataFrame:
        """Return all relations contained in this database."""
        return pd.DataFrame.from_records(
            [
                (st, sc, tt, tc)
                for (st, sc), (tt, tc) in self.relations.items()
                if (sources is None or st in sources)
                and (targets is None or tt in targets)
            ],
            columns=["source_table", "source_col", "target_table", "target_col"],
        )

    def _get_valid_refs(
        self, sources: list[str] | None = None, targets: list[str] | None = None
    ) -> pd.DataFrame:
        """Return all valid references contained in this database."""
        rel_vals = []
        rel_keys = []

        all_rels = self._get_rels(sources, targets)

        for t, rels in all_rels.groupby("source_table"):
            table = self[str(t)]
            df = (
                table.df.rename_axis("id", axis="index")
                if not table.df.index.name
                else table.df
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
                        for c, tc in r[["source_col", "target_col"]].itertuples(
                            index=False
                        )
                    }
                )
                rel_vals.append(f_df)
                rel_keys.append((t, tt))

        return pd.concat(rel_vals, keys=rel_keys, names=["src_table", "target_table"])

    def _isotropic_trim(self) -> "DB":
        """Return new database without orphan data (data w/ no refs to or from)."""
        # Get the status of each single reference.
        valid_refs = self._get_valid_refs()

        result = {}
        for t, table in self.items():
            f = pd.Series(False, index=table.df.index)

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

            result[t] = table.df.loc[f]

        return DB(table_dfs=result)

    def _centered_trim(
        self, centers: list[str], circuit_breakers: list[str] | None = None
    ) -> "DB":
        """Return new database minus data without (indirect) refs to any given table."""
        circuit_breakers = circuit_breakers or []

        # Get the status of each single reference.
        valid_refs = self._get_valid_refs()

        current_stage = {c: set(self[c].df.index) for c in centers}
        visit_counts = {
            t: pd.Series(0, index=table.df.index) for t, table in self.items()
        }
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

        return DB(
            table_dfs={t: self[t].df.loc[rc > 0] for t, rc in visit_counts.items()}
        )

    def to_graph(
        self, nodes: Sequence[SingleTable]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        # Concat all node tables into one.
        node_dfs = [self[n.name].df.reset_index().assign(table=n.name) for n in nodes]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "id"})
        )

        # Find all link tables between the given node tables.
        node_names = [n.name for n in nodes]
        relations = pd.concat(
            [self._get_rels(sources=[j], targets=node_names) for j in self.join_tables]
        )

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                self[str(source_table)]
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
