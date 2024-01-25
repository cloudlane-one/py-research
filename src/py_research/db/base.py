"""Base classes and types for relational database package."""

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Literal, cast, overload

import pandas as pd

from py_research.data import parse_dtype
from py_research.reflect import PyObjectRef

from .conflicts import DataConflictError, DataConflictPolicy


def _unmerge(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extract this table into its own database.

    Returns:
        Database containing only this table.
    """
    if df.columns.nlevels != 2:
        raise ValueError("Cannot only unmerge dataframe with two column levels.")

    return {
        str(s): cast(pd.DataFrame, df[s]).drop_duplicates()
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
        right: "SingleTable | None" = None,
        link_to_right: str | tuple[str, str] = ...,
        link_to_left: str | None = None,
        link_table: "SingleTable | None" = None,
        naming: Literal["source", "path"] = ...,
    ) -> "Table":
        ...

    @overload
    def merge(
        self,
        right: "SingleTable" = ...,
        link_to_right: str | tuple[str, str] | None = None,
        link_to_left: str | None = None,
        link_table: "SingleTable | None" = None,
        naming: Literal["source", "path"] = ...,
    ) -> "Table":
        ...

    @overload
    def merge(
        self,
        right: "SingleTable | None" = ...,
        link_to_right: str | tuple[str, str] | None = None,
        link_to_left: str | None = None,
        link_table: "SingleTable" = ...,
        naming: Literal["source", "path"] = ...,
    ) -> "Table":
        ...

    def merge(  # noqa: C901
        self,
        right: "SingleTable | None" = None,
        link_to_right: str | tuple[str, str] | None = None,
        link_to_left: str | None = None,
        link_table: "SingleTable | None" = None,
        naming: Literal["source", "path"] = "source",
    ) -> "Table":
        """Merge this table with another, returning a new table.

        Args:
            link_to_right: Name of column to use for linking from left to right table.
            link_to_left: Name of column to use for linking from right to left table.
            right: Other (left) table to merge with.
            link_table: Link table (join table) to use for double merging.
            naming:
                Naming strategy to use for naming the first level of merged columns.
                Use "path" if you merge multiple times from the same source table.

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
                    tt
                    if naming == "source"
                    else f"{sp}->{sc}=>{tc}<-{tt}"
                    if tc != (self.db[tt].df.index.name or "id")
                    else f"{sp}->{sc}",
                    (right or self.db[tt], tc),
                )
                for (sp, sc), (tt, tc) in rels
            ]
        elif right is None and link_table is None:
            outgoing = self.db._get_rels(
                sources=list(self.source_map.values())
                if isinstance(self.source_map, dict)
                else [self.source_map]
            )
            merges += [
                (
                    (sp, sc),
                    tt
                    if naming == "source"
                    else f"{sp}->{sc}=>{tc}<-{tt}"
                    if tc != (self.db[tt].df.index.name or "id")
                    else f"{sp}->{sc}",
                    (self.db[tt], tc),
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
                        tt
                        if naming == "source"
                        else f"{tp}->{tc}<={right.name}"
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
                        tt
                        if naming == "source"
                        else f"{tp}->{tc}<={sc}<-{st}"
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
                            jt.name
                            if naming == "source"
                            else (
                                (
                                    f"{btp}->{btc}"
                                    if btc != (self.db[btt].df.index.name or "id")
                                    else btp
                                )
                                + "<="
                                + (
                                    f"{bsc}<-{jt.name}"
                                    if backlink_count > 1
                                    else jt.name
                                )
                            )
                        )
                        # First perform a merge with the joint table.
                        merges.append(
                            (
                                (btp, btc),
                                jt_prefix,
                                (jt, bsc),
                            )
                        )
                        # Then perform a merge with all other tables linked from there.
                        merges += [
                            (
                                (jt_prefix, osc),
                                ott
                                if naming == "source"
                                else (
                                    (jt_prefix if len(link_tables) > 1 else btp)
                                    + "->"
                                    + (
                                        f"{osc}<={otc}<-{ott}"
                                        if otc != (self.db[ott].df.index.name or "id")
                                        else osc
                                    )
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
        sep: str = ".",
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
            c[0]
            if len(c) == 1
            else c[1]
            if (
                len(c) > 1
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
    def source_map(self) -> str:  # type: ignore[override]
        """Name of the source table of this table."""
        return self.name

    @property
    def indexes(self) -> dict[str, str]:  # type: ignore[override]
        """Name of the source table of this table."""
        return {self.name: self.df.index.name or "id"}

    # Method overrides:

    def filter(self, filter_series: pd.Series) -> "SingleTable":
        """Return a filtered version of this table."""
        table = super().filter(filter_series)
        assert isinstance(table.source_map, str)
        return SingleTable(name=table.source_map, db=table.db, df=table.df)

    def trim(self, cols: list[str]) -> "SingleTable":
        """Return a trimmed version of this table.

        Args:
            cols: Columns to keep.
        """
        return SingleTable(name=self.name, db=self.db, df=self.df[cols])

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

    updates: dict[pd.Timestamp, dict[str, Any]] = field(default_factory=dict)
    """List of update times with comments, tags, authors, etc."""

    backend: Path | None = None
    """File backend of this database, hence where it was loaded from and is
    saved to by default.
    """

    _copied: bool = False
    """Whether this database has been copied from somewhere else."""

    # Construction:

    def __post_init__(self) -> None:  # noqa: D105
        if not self._copied:
            self.updates[pd.Timestamp.now().round("s")] = {
                "action": "construct",
                "table_dfs": str(set(self.table_dfs.keys())),
                "relations": str(set(self.relations.items())),
                "join_tables": str(self.join_tables),
            }

    # Public methods:

    @staticmethod
    def load(path: Path | str, auto_parse_dtype: bool = False) -> "DB":
        """Load a database from an excel file.

        Args:
            path: Path to the excel file.
            auto_parse_dtype: Whether to automatically parse the dtypes of the data.

        Returns:
            Database object.
        """
        path = Path(path) if isinstance(path, str) else path

        relations = {}
        if "_relations" in pd.ExcelFile(path).sheet_names:
            rel_df = pd.read_excel(path, sheet_name="_relations", index_col=[0, 1])
            relations = rel_df.apply(
                lambda r: (r["target_table"], r["target_col"]),
                axis="columns",
            ).to_dict()

        join_tables = set()
        if "_join_tables" in pd.ExcelFile(path).sheet_names:
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

        updates = {}
        if "_updates" in pd.ExcelFile(path).sheet_names:
            update_df = pd.read_excel(path, sheet_name="_updates", index_col=0).drop(
                columns=["comment"]
            )
            update_df.index = pd.to_datetime(
                update_df.index, infer_datetime_format=True
            )
            updates = {
                cast(pd.Timestamp, t): c.to_dict() for t, c in update_df.iterrows()
            }

        df_dict = {
            str(k): df.apply(parse_dtype, axis="index") if auto_parse_dtype else df
            for k, df in pd.read_excel(path, sheet_name=None, index_col=0).items()
            if not k.startswith("_")
        }

        return DB(df_dict, relations, join_tables, schema, updates, path, True)

    def save(self, path: Path | str | None = None) -> None:
        """Save this database to an excel file.

        Args:
            path:
                Path to excel file. Will be overridden if it exists.
                Uses this database's backend as default path, if none given.
        """
        path = Path(path) if isinstance(path, str) else path
        if path is None:
            if self.backend is None:
                raise ValueError(
                    "Need to supply explicit path for databases without a backend."
                )
            path = self.backend

        sheets = {
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
            "_relations": pd.DataFrame.from_records(
                list(self.relations.values()),
                columns=["target_table", "target_col"],
                index=pd.MultiIndex.from_tuples(list(self.relations.keys())),
            ).rename_axis(index=["source_table", "source_col"]),
            "_join_tables": pd.DataFrame({"name": list(self.join_tables)}),
            "_updates": pd.DataFrame.from_dict(
                {t.strftime("%Y-%m-%d %X"): d for t, d in self.updates.items()},
                orient="index",
            )
            .rename_axis(index="time")
            .assign(comment=""),
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
            },
            relations=self.relations,
            join_tables=self.join_tables,
            schema=self.schema,
            updates=self.updates,
            _copied=True,
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

        return DB(
            table_dfs=merged,
            relations={**self.relations, **other.relations},
            join_tables={*self.join_tables, *other.join_tables},
            schema=self.schema,
            updates={
                **self.updates,
                pd.Timestamp.now().round("s"): {
                    "action": "extend",
                    "table_dfs": str(set(other.table_dfs.keys())),
                    "relations": str(set(other.relations.items())),
                    "join_tables": str(other.join_tables),
                },
            },
        )

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
        res = (
            self._isotropic_trim()
            if centers is None
            else self._centered_trim(centers, circuit_breakers)
        )
        res.updates[pd.Timestamp.now().round("s")] = {
            "action": "trim",
            "centers": str(set(centers or [])),
            "circuit_breakers": str(set(circuit_breakers or [])),
            "remaining_tables": str(set(self.keys())),
        }
        return res

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
        res = new_db.trim(list(filters.keys()), circuit_breakers=cb)
        res.updates[pd.Timestamp.now().round("s")] = {
            "action": "filter",
            "filter_tables": str(set(filters.keys())),
            "extra_cb": str(set(extra_cb or [])),
            "remaining_tables": str(set(self.keys())),
        }
        return res

    def to_graph(
        self, nodes: Sequence[Table | str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export links between select database objects in a graph format.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        nodes = [self[n] if isinstance(n, str) else n for n in nodes]
        # Concat all node tables into one.
        node_dfs = [
            n.df.reset_index().assign(table=n.source_map)
            if isinstance(n.source_map, str)
            else pd.concat(
                [
                    cast(pd.DataFrame, n.df[p])
                    .drop_duplicates()
                    .reset_index()
                    .assign(table=s)
                    for p, s in n.source_map.items()
                    if s not in self.join_tables
                ],
                ignore_index=True,
            )
            for n in nodes
        ]
        node_df = (
            pd.concat(node_dfs, ignore_index=True)
            .reset_index()
            .rename(columns={"index": "node_id"})
        )

        if "level_0" in node_df.columns:
            node_df = node_df.drop(columns=["level_0"])

        # Find all link tables between the given node tables.
        node_names = list(node_df["table"].unique())

        directed_edges = self._get_rels(sources=node_names, targets=node_names)
        undirected_edges = pd.concat(
            [self._get_rels(sources=[j], targets=node_names) for j in self.join_tables]
        )

        # Concat all edges into one table.
        edge_df = pd.concat(
            [
                *[
                    node_df.loc[node_df["table"] == str(rel["source_table"])]
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[node_df["table"] == str(rel["target_table"])],
                        left_on=rel["source_col"],
                        right_on=rel["target_col"],
                    )
                    .rename(columns={"node_id": "target"})[["source", "target"]]
                    .assign(ltr=rel["source_col"], rtl=None)
                    for _, rel in directed_edges.iterrows()
                ],
                *[
                    self[str(source_table)]
                    .df.merge(
                        node_df.loc[
                            node_df["table"] == str(rel.iloc[0]["target_table"])
                        ].dropna(axis="columns", how="all"),
                        left_on=rel.iloc[0]["source_col"],
                        right_on=rel.iloc[0]["target_col"],
                        how="inner",
                    )
                    .rename(columns={"node_id": "source"})
                    .merge(
                        node_df.loc[
                            node_df["table"] == str(rel.iloc[1]["target_table"])
                        ].dropna(axis="columns", how="all"),
                        left_on=rel.iloc[1]["source_col"],
                        right_on=rel.iloc[1]["target_col"],
                        how="inner",
                    )
                    .rename(columns={"node_id": "target"})[
                        list(
                            {"source", "target", *self[str(source_table)].columns}
                            - {rel.iloc[0]["source_col"], rel.iloc[1]["source_col"]}
                        )
                    ]
                    .assign(
                        ltr=rel.iloc[1]["source_col"], rtl=rel.iloc[0]["source_col"]
                    )
                    for source_table, rel in undirected_edges.groupby("source_table")
                    if len(rel) == 2  # We do not export hyper-graphs.
                ],
            ],
            ignore_index=True,
        )

        return node_df, edge_df

    # Dictionary interface:

    def __getitem__(self, name: str) -> SingleTable:  # noqa: D105
        if name.startswith("_"):
            raise KeyError(f"Table '{name}' does not exist in this database.")
        return (
            SourceTable(name=name, db=self)
            if name in self.table_dfs
            else self._manifest_virtual_table(name)
        )

    def __setitem__(  # noqa: D105
        self, name: str, value: SingleTable | pd.DataFrame
    ) -> None:
        if name.startswith("_"):
            raise KeyError("Table names starting with '_' are not allowed.")

        value = value.df if isinstance(value, SingleTable) else value
        rels = self._get_rels()
        target_tables = rels["target_table"].unique()
        if name in target_tables:
            target_cols = rels.loc[rels["target_table"] == name, "target_col"].unique()
            if not set(target_cols).issubset(value.reset_index().columns):
                raise ValueError(
                    "Relations to given table name already exist, "
                    f"but not all referenced columns were supplied: {target_cols}."
                )

        self.table_dfs[name] = value
        self.updates[pd.Timestamp.now().round("s")] = {
            "action": "set_table",
            "table": name,
        }

    def __contains__(self, name: str) -> bool:  # noqa: D105
        return name in self.keys() and not name.startswith("_")

    def __iter__(self) -> Iterable[str]:  # noqa: D105
        return (k for k in self.keys() if not k.startswith("_"))

    def __len__(self) -> int:  # noqa: D105
        return len(list(k for k in self.keys() if not k.startswith("_")))

    def keys(self) -> set[str]:  # noqa: D102
        return set(k for k in self.table_dfs if not k.startswith("_")) | set(
            self._get_rels()["target_table"].unique()
        )

    def values(self) -> Iterable[SingleTable]:  # noqa: D102
        return [self[name] for name in self.keys() if not name.startswith("_")]

    def items(self) -> Iterable[tuple[str, SingleTable]]:  # noqa: D102
        return [(name, self[name]) for name in self.keys() if not name.startswith("_")]

    def get(self, name: str) -> SingleTable | None:  # noqa: D102
        return (
            self[name] if not name.startswith("_") and name in self.table_dfs else None
        )

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

    def _manifest_virtual_table(
        self, name: str, rel: tuple[str, str] | None = None
    ) -> "SingleTable":
        """Manifest a virtual table with the required cols."""
        rels = (
            self._get_rels(targets=[name])
            if rel is None
            else pd.DataFrame.from_records(
                [(*rel, *self.relations[rel])],
                columns=["source_table", "source_col", "target_table", "target_col"],
            )
        )
        if len(rels) == 0:
            raise ValueError(f"Cannot find any relations with target table '{name}'.")

        frames = []
        for tc, rel_group in rels.groupby("target_col"):
            col_values = reduce(
                set.union,
                (
                    set(self[st][sc].unique())
                    for st, sc in rel_group[["source_table", "source_col"]]
                    .drop_duplicates()
                    .itertuples(index=False)
                ),
            )
            frames.append(pd.DataFrame({tc: list(col_values)}))

        return SingleTable(name=name, db=self, df=pd.concat(frames, ignore_index=True))

    def _isotropic_trim(self) -> "DB":
        """Return new database without orphan data (data w/ no refs to or from)."""
        # Get the status of each single reference.
        valid_refs = self._get_valid_refs()

        result = {}
        for t, table in self.items():
            f = pd.Series(False, index=table.df.index)

            # Include all rows with any valid outgoing reference.
            try:
                outgoing = valid_refs.loc[(t, slice(None), slice(None)), :]

                for _, refs in outgoing.groupby("target_table"):
                    f |= (
                        refs.droplevel(["src_table", "target_table"])
                        .notna()
                        .any(axis="columns")
                    )
            except KeyError:
                pass

            # Include all rows with any valid incoming reference.
            try:
                incoming = valid_refs.loc[(slice(None), t, slice(None)), :]

                for _, refs in incoming.groupby("src_table"):
                    for _, col in refs.items():
                        f |= pd.Series(True, index=col.unique())
            except KeyError:
                pass

            result[t] = table.df.loc[f]

        return DB(
            table_dfs=result,
            relations=self.relations,
            join_tables=self.join_tables,
            schema=self.schema,
        )

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

                # Include all rows with any valid outgoing reference.
                try:
                    outgoing = valid_refs.loc[(t, slice(None), idx_sel), :]

                    for tt, refs in outgoing.groupby("target_table"):
                        tt = str(tt)
                        for col_name, col in refs.items():
                            ref_vals = col.dropna().unique()
                            target_df = self[tt].df
                            target_col = self.relations[(t, str(col_name))][1]
                            target_idx = (
                                ref_vals
                                if target_col in target_df.index.names
                                else target_df.loc[
                                    target_df[target_col].isin(ref_vals)
                                ].index.unique()
                            )
                            next_stage[tt] |= set(target_idx)
                except KeyError:
                    pass

                # Include all rows with any valid incoming reference.
                try:
                    incoming = valid_refs.loc[(slice(None), t, slice(None)), :]

                    for st, refs in incoming.groupby("src_table"):
                        refs = refs.dropna(axis="columns", how="all")
                        target_df = self[t].df
                        target_cols = {
                            c: self.relations[(str(st), str(c))][1]
                            for c in refs.columns
                        }
                        target_values = {
                            c: target_df.loc[idx_sel][tc].dropna().unique()
                            if tc in target_df.columns
                            else idx_sel
                            for c, tc in target_cols.items()
                        }
                        next_stage[str(st)] |= set(
                            pd.concat(
                                [
                                    col.isin(target_values[str(c)])
                                    for c, col in refs.droplevel(
                                        ["src_table", "target_table"]
                                    ).items()
                                ],
                                axis="columns",
                            )
                            .any(axis="columns")
                            .replace(False, pd.NA)
                            .dropna()
                            .index
                        )
                except KeyError:
                    pass

            current_stage = next_stage

        return DB(
            table_dfs={t: self[t].df.loc[rc > 0] for t, rc in visit_counts.items()},
            relations=self.relations,
            join_tables=self.join_tables,
            schema=self.schema,
        )
