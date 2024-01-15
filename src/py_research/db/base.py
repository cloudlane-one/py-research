"""Base classes and types for relational database package."""

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast, overload

import pandas as pd

from py_research.data import parse_dtype
from py_research.reflect import PyObjectRef

from .conflicts import DataConflictError, DataConflictPolicy


def _unmerge(df: pd.DataFrame, with_relations: bool = True) -> dict[str, pd.DataFrame]:
    """Extract this table into its own database.

    Returns:
        Database containing only this table.
    """
    if df.columns.nlevels != 2:
        raise ValueError("Cannot only unmerge dataframe with two column levels.")

    return {
        s: cast(pd.DataFrame, df[s]).drop_duplicates()
        for s in df.columns.get_level_values(0).unique()
    }


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

    indexes: dict[str, str] = field(default_factory=dict)
    """Mapping from source table names to index column names."""

    # Creation:

    @staticmethod
    def from_excel(path: Path) -> "Table":
        """Load relational table from Excel file."""
        source_map = pd.read_excel(path, sheet_name="_source_tables", index_col=0)[
            "table"
        ].to_dict()

        indexes = pd.read_excel(path, sheet_name="_indexes", index_col=0)[
            "column"
        ].to_dict()

        data = pd.read_excel(
            path,
            sheet_name="data",
            index_col=0,
            header=(0 if len(source_map) == 1 else (0, 1)),
        )

        df_dict = (
            {source_map[s]: df for s, df in _unmerge(data).items()}
            if len(source_map) > 1
            else {list(source_map.keys())[0]: data}
        )

        return Table(
            DB(df_dict),
            data,
            source_map if len(source_map) > 1 else list(source_map.keys())[0],
            indexes,
        )

    # Public methods:

    def to_excel(self, path: Path) -> None:
        """Save this table to an Excel file."""
        sheets = {
            "_source": pd.Series({"database": self.db.backend})
            .rename("value")
            .to_frame(),
            "_source_tables": pd.DataFrame.from_dict(
                {k: [v] for k, v in self.source_map.items()}
                if isinstance(self.source_map, dict)
                else {self.source_map: [None]},
                orient="index",
                columns=["table"],
            ).rename_axis(index="col_prefix"),
            "_indexes": pd.DataFrame.from_dict(
                {k: [v] for k, v in self.indexes.items()},
                orient="index",
                columns=["column"],
            ).rename_axis(index="table"),
            "data": self.df,
        }

        with pd.ExcelWriter(
            path,
            engine="openpyxl",
        ) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=True)

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

    @overload
    def merge(
        self,
        link_to_right: str | tuple[str, str] = ...,
        link_to_left: str | None = None,
        right: "SingleTable | None" = None,
        link_table: "SingleTable | None" = None,
    ) -> "Table":
        ...

    @overload
    def merge(
        self,
        link_to_right: str | tuple[str, str] | None = None,
        link_to_left: str | None = None,
        right: "SingleTable" = ...,
        link_table: "SingleTable | None" = None,
    ) -> "Table":
        ...

    @overload
    def merge(
        self,
        link_to_right: str | tuple[str, str] | None = None,
        link_to_left: str | None = None,
        right: "SingleTable" = ...,
        link_table: "SingleTable" = ...,
    ) -> "Table":
        ...

    def merge(  # noqa: C901
        self,
        link_to_right: str | tuple[str, str] | None = None,
        link_to_left: str | None = None,
        right: "SingleTable | None" = None,
        link_table: "SingleTable | None" = None,
    ) -> "Table":
        """Merge this table with another, returning a new table.

        Args:
            link_to_right: Name of column to use for linking from left to right table.
            link_to_left: Name of column to use for linking from right to left table.
            right: Other (left) table to merge with.
            link_table: Link table (join table) to use for double merging.

        Note:
            At least one of ``link_to_right``, ``right`` or ``link_table`` must be
            supplied.

        Returns:
            New table containing the merged data.
            The returned table will have a multi-level column index,
            where the first level references the source table of each column
            via the ``source_map`` attribute.
        """
        # Standardize format of dataframe, making sure its columns are multi-level.
        merged = self.df
        if isinstance(self.source_map, str):
            merged = merged.rename_axis(merged.index.name or "id", axis="index")
            merged = merged.reset_index()
            merged.columns = pd.MultiIndex.from_product(
                [[self.source_map], merged.columns]
            )

        # Standardize format of source map, making sure it is a dict.
        merge_source_map = (
            self.source_map
            if isinstance(self.source_map, dict)
            else {self.source_map: self.source_map}
        )
        inv_src_map = {
            v: [k for k, v2 in merge_source_map.items() if v2 == v]
            for v in set(merge_source_map.values())
        }
        sources = set(merge_source_map.values())

        merge_indexes = self.indexes.copy()

        # Set up a list of all merges to be applied with their parameters.
        merges: list[tuple[tuple[str, str], str, tuple[SingleTable, str]]] = []

        # Get a list of all applicable forward merges with their parameters.
        if link_to_right is not None:
            rels = (
                [
                    ((sp, link_to_right), self.db.relations.get((s, link_to_right)))
                    for s in sources
                    for sp in inv_src_map[s]
                ]
                if isinstance(link_to_right, str)
                else [(link_to_right, self.db.relations.get(link_to_right))]
            )
            rels = [(s, r) for s, r in rels if r is not None]
            if len(rels) == 0:
                raise ValueError(
                    "Cannot find relation with source table "
                    f"in {sources} and source col = '{link_to_right}'."
                )

            if right is not None:
                rels = [(s, r) for s, r in rels if r[0] == right.name]
                if len(rels) == 0:
                    raise ValueError(
                        f"Cannot find relation with source table in {sources}, "
                        f"source col = '{link_to_right}' and "
                        f"target table = {right.name}."
                    )

            merges += [
                (
                    (sp, sc),
                    f"{sp}->{sc}=>{tc}<-{tt}"
                    if tc != (self.db[tt].df.index.name or "id")
                    else f"{sp}->{sc}",
                    (right or self.db[tt], tc),
                )
                for (sp, sc), (tt, tc) in rels
            ]
        elif right is None:
            outgoing = self.db._get_rels(
                sources=list(self.source_map.values())
                if isinstance(self.source_map, dict)
                else [self.source_map]
            )
            merges += [
                (
                    (sp, sc),
                    f"{sp}->{sc}=>{tc}<-{tt}"
                    if tc != (self.db[tt].df.index.name or "id")
                    else f"{sp}->{sc}",
                    (tt, tc),
                )
                for st, sc, tt, tc in outgoing.itertuples(index=False)
                for sp in inv_src_map[st]
                if right is None or right.name == tt
            ]

        # Get all incoming relations.
        incoming = self.db._get_rels(
            targets=list(self.source_map.values())
            if isinstance(self.source_map, dict)
            else [self.source_map]
        )

        # Get a list of all applicable backward merges with their parameters.
        if len(merges) == 0 and right is not None:
            # Get all matchin incoming links.
            from_right = [
                (st, sc, tt, tp, tc)
                for st, sc, tt, tc in incoming.itertuples(index=False)
                for tp in inv_src_map[tt]
                if right.name == st
            ]

            # If there is only one, use it as explicit link_to_left.
            if len(from_right) == 1:
                link_to_left = from_right[0][1]

            if link_to_left is not None:
                rel = self.db.relations.get((right.name, link_to_left))
                if rel is None:
                    raise ValueError(
                        f"Cannot find relation with target table in {self.source_map} "
                        f", src table = {right.name} and src col = '{link_to_left}'."
                    )
                tt, tc = rel
                merges += [
                    (
                        (tp, tc),
                        f"{tp}->{tc}<={right.name}"
                        if tc != (self.db[tt].df.index.name or "id")
                        else f"{tp}<={right.name}",
                        (right, link_to_left),
                    )
                    for tp in inv_src_map[tt]
                ]
            else:
                merges += [
                    (
                        (tp, tc),
                        f"{tp}->{tc}<={sc}<-{st}"
                        if tc != (self.db[tt].df.index.name or "id")
                        else f"{tp}<={sc}<-{st}",
                        (self.db[st], sc),
                    )
                    for st, sc, tt, tp, tc in from_right
                ]

        # Get a list of all related join / link tables.
        link_tables = []
        if link_table is not None:
            link_tables = [link_table]
        else:
            link_tables = [
                self.db[t]
                for t in incoming["source_table"].unique()
                if t in self.db.join_tables
            ]

        # Get a list of all applicable double merges with their parameters.
        if len(merges) == 0:
            for jt in link_tables:
                jt_links = self.db._get_rels(
                    sources=list(jt.source_map.values())
                    if isinstance(jt.source_map, dict)
                    else [jt.source_map]
                )
                backlinks = jt_links.loc[jt_links["target_table"].isin(sources)]
                other_links = jt_links.loc[~jt_links["target_table"].isin(sources)]

                if right is not None:
                    other_links = other_links.loc[
                        other_links["target_table"] == right.name
                    ]
                    if len(other_links) == 0:
                        continue

                for _, bsc, btt, btc in backlinks.itertuples(index=False):
                    backlink_count = sum(backlinks["target_table"] == btt)
                    for btp in inv_src_map[btt]:
                        jt_prefix = (
                            (
                                f"{btp}->{btc}"
                                if btc != (self.db[btt].df.index.name or "id")
                                else btp
                            )
                            + "<="
                            + (f"{bsc}<-{jt.name}" if backlink_count > 1 else jt.name)
                        )
                        # First perform a merge with the joint table.
                        merges.append(((btp, btc), jt_prefix, (jt, bsc)))
                        # Then perform a merge with all other tables linked from there.
                        merges += [
                            (
                                (jt_prefix, osc),
                                (jt_prefix if len(link_tables) > 1 else btp)
                                + "->"
                                + (
                                    f"{osc}<={otc}<-{ott}"
                                    if otc != (self.db[ott].df.index.name or "id")
                                    else osc
                                ),
                                (right or self.db[ott], otc),
                            )
                            for _, osc, ott, otc in other_links.itertuples(index=False)
                        ]

        if len(merges) == 0:
            raise ValueError(
                "Could not find any relations between the given tables to merge on."
            )

        for (sp, sc), tp, (tt, tc) in merges:
            # Flatten multi-level columns of left table.
            left_df = merged.copy()
            left_df.columns = merged.columns.map(lambda c: "->".join(c))
            left_on = f"{sp}->{sc}"

            # Properly name columns of right table.
            right_df = tt.df.reset_index().rename(columns=lambda c: f"{tp}->{c}")
            right_on = f"{tp}->{tc}"

            new_merged = left_df.merge(
                right_df,
                left_on=left_on,
                right_on=right_on,
                how="left",
            )

            # Restore two-level column index.
            new_merged.columns = pd.MultiIndex.from_tuples(
                [
                    ("->".join(c[:-1]), c[-1])
                    for c in new_merged.columns.map(
                        lambda c: str(c).split("->")
                    ).to_list()
                ]
            )
            merged = new_merged.copy()

            # Add to source_map
            merge_source_map[tp] = tt.name

            # Add to indexes
            merge_indexes[tt.name] = tt.df.index.name or "id"

        return Table(self.db, merged, merge_source_map, merge_indexes)

    def flatten(
        self,
        sep: str = "->",
        prefix_strategy: Literal["always", "on_conflict"] = "always",
    ) -> pd.DataFrame:
        """Collapse multi-dim. column labels of multi-source table, returning new df.

        Args:
            sep: Separator to use between column levels.
            prefix_strategy: Strategy to use for prefixing column names.

        Returns:
            Dataframe representation of this table with flattened multi-dim columns.
        """
        level_counts = (
            self.df.columns.to_frame()
            .groupby(level=(1 if self.df.columns.nlevels > 1 else 0))
            .size()
        )

        res_df = self.df.copy()
        res_df.columns = [
            c
            if isinstance(c, str)
            or (
                isinstance(c, tuple)
                and prefix_strategy == "on_conflict"
                and level_counts[c[1]] == 1
            )
            else sep.join(c)
            for c in self.df.columns.to_frame().itertuples(index=False)
        ]

        return res_df

    def extract(self, with_relations: bool = True) -> "DB":
        """Extract this table into its own database.

        Returns:
            Database containing only this table.
        """
        source_map = (
            self.source_map
            if isinstance(self.source_map, dict)
            else {self.source_map: self.source_map}
        )
        inv_src_map = {
            v: [k for k, v2 in source_map.items() if v2 == v]
            for v in set(source_map.values())
        }

        source_tables = {
            t: pd.concat(
                [cast(pd.DataFrame, self.df[s]).drop_duplicates() for s in all_s]
            )
            .dropna(subset=[self.indexes[t]])
            .set_index(self.indexes[t])
            for t, all_s in inv_src_map.items()
        }

        return (
            self.db.filter(
                {
                    s: self.db[s].df.index.to_series().isin(df.index)
                    for s, df in source_tables.items()
                }
            )
            if with_relations
            else DB(table_dfs=source_tables)
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

    @property
    def indexes(self) -> dict[str, str]:
        """Name of the source table of this table."""
        return {self.name: self.df.index.name or "id"}

    # Method overrides:

    def filter(self, filter_series: pd.Series) -> "SingleTable":
        """Return a filtered version of this table."""
        table = super().filter(filter_series)
        assert isinstance(table.source_map, str)
        return SingleTable(name=table.source_map, db=table.db, df=table.df)


class SourceTable(SingleTable):
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


class DBSchema:
    """Base class for static database schemas."""


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

    schema: type[DBSchema] | None = None
    """Schema of this database."""

    updates: dict[datetime, dict[str, Any]] = field(
        default_factory=lambda: {datetime.now(): {}}
    )
    """List of update times with comments, tags, authors, etc."""

    backend: Path | None = None
    """File backend of this database, hence where it was loaded from and is
    saved to by default.
    """

    # Public methods:

    @staticmethod
    def load(path: Path, auto_parse_dtype: bool = False) -> "DB":
        """Load a database from an excel file.

        Args:
            path: Path to the excel file.
            auto_parse_dtype: Whether to automatically parse the dtypes of the data.

        Returns:
            Database object.
        """
        rel_df = pd.read_excel(path, sheet_name="_relations", index_col=[0, 1])
        relations = rel_df.apply(
            lambda r: (r["target_table"], r["target_col"]),
            axis="columns",
        ).to_dict()

        jt_df = pd.read_excel(path, sheet_name="_join_tables", index_col=0)
        join_tables = set(jt_df["name"].tolist())

        schema = None
        if "_schema" in pd.ExcelFile(path).sheet_names:
            schema_df = pd.read_excel(path, sheet_name="_schema", index_col=0)
            schema_ref = PyObjectRef(**schema_df["value"].to_dict())
            schema_ref.type = type
            if pd.isna(schema_ref.version):
                schema_ref.version = None

            schema = schema_ref.resolve()
            if not issubclass(schema, DBSchema):
                raise ValueError("Database schema must be a subclass of `DBSchema`.")

        update_df = pd.read_excel(path, sheet_name="_updates", index_col=0)
        update_df.index = pd.to_datetime(update_df.index)
        updates = cast(
            dict[datetime, dict[str, Any]],
            update_df.to_dict(orient="index"),
        )

        df_dict = {
            str(k): df.apply(parse_dtype, axis="index") if auto_parse_dtype else df
            for k, df in pd.read_excel(path, sheet_name=None, index_col=0).items()
            if not k.startswith("_")
        }

        return DB(df_dict, relations, join_tables, schema, updates, path)

    def save(self, path: Path | None = None) -> None:
        """Save this database to an excel file.

        Args:
            path:
                Path to excel file. Will be overridden if it exists.
                Uses this database's backend as default path, if none given.
        """
        if path is None:
            if self.backend is None:
                raise ValueError(
                    "Need to supply explicit path for databases without a backend."
                )
            path = self.backend

        sheets = {
            "_relations": pd.DataFrame.from_records(
                list(self.relations.values()),
                columns=["target_table", "target_col"],
                index=pd.MultiIndex.from_tuples(list(self.relations.keys())),
            ).rename_axis(index=["source_table", "source_col"]),
            "_join_tables": pd.DataFrame({"name": list(self.join_tables)}),
            **(
                {
                    "_schema": pd.Series(asdict(PyObjectRef.reference(self.schema)))
                    .rename("value")
                    .to_frame()
                    .rename_axis(index="key")
                }
                if self.schema is not None
                else {}
            ),
            "_updates": pd.DataFrame.from_dict(self.updates, orient="index"),
            **self.table_dfs,
        }
        with pd.ExcelWriter(
            path,
            engine="openpyxl",
        ) as writer:
            for name, sheet in sheets.items():
                sheet.to_excel(writer, sheet_name=name, index=True)

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
        other: "DB | dict[str, pd.DataFrame] | Table",
        conflict_policy: DataConflictPolicy
        | dict[str, DataConflictPolicy | dict[str, DataConflictPolicy]] = "raise",
    ) -> "DB":
        """Extend this database with data from another, returning a new database.

        Args:
            other: Other database, dataframe dict or table to extend with.
            conflict_policy:
                Policy to use for resolving conflicts. Can be a global setting,
                per-table via supplying a dict with table names as keys, or per-column
                via supplying a dict of dicts.

        Returns:
            New database containing the extended data.
        """
        # Standardize type of other.
        other = (
            other
            if isinstance(other, DB)
            else DB(other)
            if isinstance(other, dict)
            else other.extract()
        )

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

    # Dictionary interface:

    def __getitem__(self, name: str) -> SourceTable:  # noqa: D105
        if name.startswith("_"):
            raise KeyError(f"Table '{name}' does not exist in this database.")
        return SourceTable(name=name, db=self)

    def __contains__(self, name: str) -> bool:  # noqa: D105
        return name in self.table_dfs and not name.startswith("_")

    def __iter__(self) -> Iterable[str]:  # noqa: D105
        return (k for k in self.table_dfs if not k.startswith("_"))

    def __len__(self) -> int:  # noqa: D105
        return len(list(k for k in self.table_dfs if not k.startswith("_")))

    def keys(self) -> Iterable[str]:  # noqa: D102
        return (k for k in self.table_dfs if not k.startswith("_"))

    def values(self) -> Iterable[SourceTable]:  # noqa: D102
        return [
            SourceTable(name, self)
            for name in self.table_dfs
            if not name.startswith("_")
        ]

    def items(self) -> Iterable[tuple[str, SourceTable]]:  # noqa: D102
        return [
            (name, SourceTable(name, self))
            for name in self.table_dfs
            if not name.startswith("_")
        ]

    def get(self, name: str) -> SourceTable | None:  # noqa: D102
        return SourceTable(name, self) if not name.startswith("_") else None

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
