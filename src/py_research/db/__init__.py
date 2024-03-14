"""Easy to use relational database."""

import secrets
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial, reduce, wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
    cast,
    overload,
)

import pandas as pd
import sqlalchemy as sqla
import sqlalchemy.orm as orm
from pandas.api.types import (
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.util import hash_pandas_object

from py_research.reflect.ref import PyObjectRef
from py_research.reflect.runtime import get_all_subclasses

from .conflicts import DataConflictError, DataConflictPolicy


def _hash_df(df: pd.DataFrame | pd.Series) -> str:
    return hex(abs(sum(hash_pandas_object(df))))[2:12]


Params = ParamSpec("Params")


DBSchema = orm.DeclarativeBase


default_type_map = {
    str: sqla.types.String().with_variant(
        sqla.types.String(50), "oracle"
    ),  # Avoid oracle error when VARCHAR has no size parameter
}


V = TypeVar("V")
S = TypeVar("S", bound=DBSchema)
S_cov = TypeVar("S_cov", bound=DBSchema, covariant=True)
S_contra = TypeVar("S_contra", bound=DBSchema, contravariant=True)
V_contra = TypeVar("V_contra", contravariant=True)


class ColRef(orm.InstrumentedAttribute[V], Generic[V, S_contra]):
    """Reference a column by scheme type, name and value type."""  # type: ignore[override]


class Col(orm.MappedColumn[V]):
    """Define table column within a table schema."""

    if TYPE_CHECKING:

        @overload
        def __get__(self, instance: None, owner: type[S]) -> ColRef[V, S]: ...

        @overload
        def __get__(self, instance: object, owner: type) -> V: ...

        def __get__(  # noqa: D105
            self, instance: object | None, owner: type[S]
        ) -> ColRef[V, S] | V: ...

    @classmethod
    def cast_sqla_attr(cls, attr: orm.MappedColumn) -> Self:
        """Cast a sqlalchemy attr object to this type."""
        return cast(Self, attr)


def _wrap_mapped_col(
    func: Callable[Params, orm.MappedColumn[V]]
) -> Callable[Params, Col[V]]:
    @wraps(func)
    def inner(*args: Params.args, **kwargs: Params.kwargs) -> Col[V]:
        return Col.cast_sqla_attr(func(*args, **kwargs))

    return inner


# Wrap sqlalchemy function, preserving all param types but changing return type.
col = _wrap_mapped_col(orm.mapped_column)


Relation = sqla.ForeignKey


def _map_df_dtype(c: pd.Series) -> sqla.types.TypeEngine:
    if is_datetime64_dtype(c):
        return sqla.types.DATETIME()
    elif is_integer_dtype(c):
        return sqla.types.INTEGER()
    elif is_numeric_dtype(c):
        return sqla.types.FLOAT()
    elif is_string_dtype(c):
        max_len = c.str.len().max()
        if max_len < 16:
            return sqla.types.CHAR(max_len)
        elif max_len < 256:
            return sqla.types.VARCHAR(max_len)

    return sqla.types.BLOB()


def _cols_from_df(df: pd.DataFrame) -> dict[str, sqla.Column]:
    if len(df.index.names) > 1:
        raise NotImplementedError("Multi-index not supported yet.")

    return {
        **{
            level: sqla.Column(
                level,
                _map_df_dtype(df.index.get_level_values(level).to_series()),
                primary_key=True,
            )
            for level in df.index.names
        },
        **{
            str(name): sqla.Column(str(col.name), _map_df_dtype(col))
            for name, col in df.items()
        },
    }


def _transfer_metadata(cls: type[DBSchema], schema_name: str) -> sqla.MetaData:
    new_meta = sqla.MetaData()

    for table in cls.metadata.tables.values():
        table.to_metadata(
            new_meta,
            schema=schema_name,
        )

    return new_meta


@dataclass(frozen=True)
class Table:
    """Table in a relational database."""

    # Attributes:

    db: "DB"
    """Database this table belongs to."""

    query: sqla.Select
    """SQL select statement for this table."""

    primary_keys: set[str] = field(default_factory=set)
    """Column names that are used as primary keys in this table."""

    foreign_keys: dict[str, sqla.Table | sqla.ColumnElement] = field(
        default_factory=dict
    )
    """Relations to other tables in this database."""

    def filter(self, condition: sqla.ColumnElement[bool] | pd.Series[bool]) -> "Table":
        """Return a table filtered by the given condition."""
        raise NotImplementedError("Filtering tables is not yet supported.")

    def focus(
        self,
        target: list["str | sqla.ColumnElement | sqla.Table | Table"] | None = None,
    ) -> "Table":
        """Return new table only containing only selected columns.

        Args:
            target: Selector for columns to keep in the new table.

        Returns:
            New table containing the focused data.
        """
        raise NotImplementedError("Trimming tables is not yet supported.")

    def merge(
        self,
        to: "Table | sqla.Table | sqla.ColumnElement | str",
        via: "Table | sqla.Table | None" = None,
        prefix: Literal["source", "path"] = "source",
        sep: str = ".",
    ) -> "Table":
        """Merge this table with another, returning a new table.

        Args:
            to:
              Relation to merge on.
              May be:
                * a reference to another table, which this table is related to
                * a column in this table, which references a column in another table
                * a column in another table, which references a column in this table
            via:
              Association table to use for resolving relations.
            prefix:
              Prefixing strategy to use for naming the first level of merged columns.
              Use "path" if you merge multiple times from the same source table.
            sep:
              Separator to use when prefixing column names.

        Returns:
            New table containing the merged data.
            The returned table will have a column names prefixed by their
            merge path or source table, depending on the `naming` parameter.
        """
        # Standardize format of dataframe, making sure its columns are multi-level.
        merged = self.df
        if isinstance(self.sources, str):
            merged = merged.rename_axis(merged.index.name or "id", axis="index")
            merged = merged.reset_index()
            merged.columns = pd.MultiIndex.from_product(
                [[self.sources], merged.columns]
            )

        # Standardize format of source map, making sure it is a dict.
        merge_source_map = (
            self.sources
            if isinstance(self.sources, dict)
            else {self.sources: self.sources}
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
                    (
                        tt
                        if naming == "source"
                        else (
                            f"{sp}->{sc}=>{tc}<-{tt}"
                            if tc != (self.db[tt].df.index.name or "id")
                            else f"{sp}->{sc}"
                        )
                    ),
                    (right or self.db[tt], tc),
                )
                for (sp, sc), (tt, tc) in rels
            ]
        elif right is None and link_table is None:
            outgoing = self.db._get_rels(
                sources=(
                    list(self.sources.values())
                    if isinstance(self.sources, dict)
                    else [self.sources]
                )
            )
            merges += [
                (
                    (sp, sc),
                    (
                        tt
                        if naming == "source"
                        else (
                            f"{sp}->{sc}=>{tc}<-{tt}"
                            if tc != (self.db[tt].df.index.name or "id")
                            else f"{sp}->{sc}"
                        )
                    ),
                    (self.db[tt], tc),
                )
                for st, sc, tt, tc in outgoing.itertuples(index=False)
                for sp in inv_src_map[st]
                if right is None or right.name == tt
            ]

        # Get all incoming relations.
        incoming = self.db._get_rels(
            targets=(
                list(self.sources.values())
                if isinstance(self.sources, dict)
                else [self.sources]
            )
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
                        f"Cannot find relation with target table in {self.sources} "
                        f", src table = {right.name} and src col = '{link_to_left}'."
                    )
                tt, tc = rel
                merges += [
                    (
                        (tp, tc),
                        (
                            tt
                            if naming == "source"
                            else (
                                f"{tp}->{tc}<={right.name}"
                                if tc != (self.db[tt].df.index.name or "id")
                                else f"{tp}<={right.name}"
                            )
                        ),
                        (right, link_to_left),
                    )
                    for tp in inv_src_map[tt]
                ]
            else:
                merges += [
                    (
                        (tp, tc),
                        (
                            tt
                            if naming == "source"
                            else (
                                f"{tp}->{tc}<={sc}<-{st}"
                                if tc != (self.db[tt].df.index.name or "id")
                                else f"{tp}<={sc}<-{st}"
                            )
                        ),
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
                    sources=(
                        list(jt.source_map.values())
                        if isinstance(jt.source_map, dict)
                        else [jt.source_map]
                    )
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
                                (
                                    ott
                                    if naming == "source"
                                    else (
                                        (jt_prefix if len(link_tables) > 1 else btp)
                                        + "->"
                                        + (
                                            f"{osc}<={otc}<-{ott}"
                                            if otc
                                            != (self.db[ott].df.index.name or "id")
                                            else osc
                                        )
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

    def extend(
        self,
        other: "pd.DataFrame",
        conflict_policy: DataConflictPolicy | dict[str, DataConflictPolicy] = "raise",
    ) -> "Table":
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

    def transform(
        self,
        func: Callable[["Table"], sqla.Select | pd.DataFrame],
        batch_size: int | None = None,
        new_primary_keys: set[str] | None = None,
        new_foreign_keys: dict[str, sqla.Table | sqla.Column] | None = None,
    ) -> "Table":
        """Transform this table using the given function.

        Args:
            func: Function to transform this table with.
            batch_size:
                Size of batches to use when transforming the table via pandas.
                Batch results will be concatenated to form the final result.
                Defaults to None, i.e. transforming the whole table at once.
            new_primary_keys:
                New primary keys to use for the transformed table.
                Defaults to None, i.e. keeping the old primary keys.
            new_foreign_keys:
                New foreign keys to use for the transformed table.
                Defaults to None, i.e. keeping the old foreign keys.

        Returns:
            New table containing the transformed data.
        """
        raise NotImplementedError("Transforming tables is not yet supported.")

    def extract(self, with_relations: bool = True) -> "DB":
        """Extract this table into its own database (incl. related data).

        Returns:
            New database instance.
        """
        source_map = (
            self.sources
            if isinstance(self.sources, dict)
            else {self.sources: self.sources}
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

    @overload
    def df(self, chunksize: None) -> pd.DataFrame: ...

    @overload
    def df(self, chunksize: int) -> Iterable[pd.DataFrame]: ...

    def df(self, chunksize: int | None = None) -> pd.DataFrame | Iterable[pd.DataFrame]:
        """Return this table's data as a pandas dataframe."""
        with self.db.engine.connect() as con:
            if chunksize is None:
                return pd.read_sql(self.query, con)
            else:
                return pd.read_sql(self.query, con, chunksize=chunksize)

    def __clause_element__(self) -> sqla.Subquery:
        """Return the SQL clause element of this table."""
        return self.query.subquery()

    def __getitem__(self, name: str) -> sqla.ColumnElement:
        """Return the column of this table with the given name."""
        return self.query.selected_columns[name]

    def __setitem__(self, name: str, value: sqla.ColumnElement | pd.Series) -> None:
        """Set the column of this table with the given name."""
        raise NotImplementedError("Setting columns is not yet supported.")

    def __contains__(self, name: str) -> bool:
        """Check whether this table has a column with the given name."""
        return name in self.query.selected_columns

    def get(self, name: str, default: Any = None) -> sqla.ColumnElement | None:
        """Return the column of this table with the given name, if it exists."""
        return self.query.selected_columns.get(name)

    def keys(self) -> Iterable[str]:
        """Return the column names of this table."""
        return self.query.selected_columns.keys()

    @property
    def columns(self) -> Sequence[sqla.ColumnElement]:
        """Return the columns of this table."""
        return self.query.selected_columns.values()


@dataclass
class DB:
    """Relational database consisting of multiple named tables."""

    backend: str | Path | sqla.URL

    schema: type[DBSchema]
    """Schema of this database."""

    schema_name: str | None = None

    session_key: str = field(default_factory=partial(secrets.token_hex, nbytes=4))

    overlay_key: str | None = None

    validate: bool = True
    """Whether to validate the DB schema on connection."""

    _copied: bool = False
    """Whether this database has been copied from somewhere else."""

    def _temp_table_name(self, name: str) -> str:
        raise NotImplementedError("Temporary tables are not yet supported.")

    def _overlay_table_name(self, name: str) -> str:
        raise NotImplementedError("Overlays are not yet supported.")

    def __post_init__(self) -> None:  # noqa: D105
        self.path: Path | None = None
        self.url: sqla.URL | None = None
        self.engine: sqla.engine.Engine | None = None
        self.excel_backend = False

        # Parse backend.
        if isinstance(self.backend, Path):
            self.path = self.backend
        elif isinstance(self.backend, str):
            # Try to parse supplied backend as a file path.
            try:
                self.path = Path(self.backend)
            except ValueError:
                # Try to parse supplied backend as a URL.
                self.url = sqla.make_url(self.backend)
        elif isinstance(self.backend, sqla.URL):
            self.url = self.backend

        # Set up URL and connect args (from path).
        if self.path is not None:
            if self.path.suffix == ".sqlite":
                self.url = sqla.URL.create(
                    drivername="sqlite",
                    database=str(self.path),
                )
            elif self.path.suffix == ".db":
                self.url = sqla.URL.create(
                    drivername="duckdb",
                    database=str(self.path),
                )
            elif self.path.suffix == ".xls" or self.path.suffix == ".xlsx":
                self.url = sqla.URL.create(
                    drivername="duckdb",
                    database=":memory:",
                )
                self.excel_backend = True
            else:
                raise ValueError("Unsupported database file format.")
        else:
            assert self.url is not None

        # Create engine from URL and connect args.
        self.engine = sqla.create_engine(self.url)

        # Load data from excel file.
        if self.excel_backend:
            raise NotImplementedError("Excel backends are not yet supported.")

        # Validate schema.
        if self.validate:
            inspector = sqla.inspect(self.engine)

            # Iterate over all tables in metadat and perform checks on each.
            for table in self.schema.metadata.tables.values():
                # Check if explicit schema name exists.
                if table.schema is not None:
                    assert inspector.has_schema(
                        table.schema
                    ), f"Schema '{table.schema}' does not exist in database."

                # Check if table exists.
                assert inspector.has_table(
                    table.name, table.schema
                ), f"Table '{table.name}' does not exist in database."

                # Check if all columns exist and have the correct type and nullability.
                db_columns = {
                    c["name"]: c
                    for c in inspector.get_columns(table.name, table.schema)
                }
                for column in table.columns:
                    assert (
                        column.name in db_columns
                    ), f"Column '{column.name}' does not exist in table '{table.name}'."

                    db_col = db_columns[column.name]
                    assert isinstance(
                        db_col["type"], type(column.type)
                    ), f"Column '{column.name}' in table '{table.name}' has wrong type."
                    assert (
                        db_col["nullable"] == column.nullable or column.nullable is None
                    ), f"Column '{column.name}' in table '{table.name}' has wrong nullability."

                # Check if all primary keys are compatible.
                db_pk = inspector.get_pk_constraint(table.name, table.schema)
                if len(db_pk["constrained_columns"]) > 0:
                    # Allow source tables without pk
                    assert set(db_pk["constrained_columns"]) == set(
                        table.primary_key.columns.keys()
                    ), f"Primary key of table '{table.name}' does not match database."
                else:
                    warnings.warn(
                        f"Table '{table.name}' has no primary key in database."
                    )

                # Check if all foreign keys are compatible.
                db_fks = inspector.get_foreign_keys(table.name, table.schema)
                for fk in table.foreign_key_constraints:
                    for db_fk in db_fks:
                        if set(db_fk["constrained_columns"]) == set(fk.column_keys):
                            assert (
                                db_fk["referred_table"].lower()
                                == fk.referred_table.name.lower()
                            ) and set(db_fk["referred_columns"]) == set(
                                f.column.name for f in fk.elements
                            ), f"Foreign key '{fk}' in table '{table.name}' does not match database."

        # Discover association tables.
        self.assoc_tables = NotImplemented

        # Load, parse, match and narrow database schema.
        raise NotImplementedError()

    def describe(self) -> dict[str, str | dict[str, str]]:
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

        schema_desc = {}
        if self.schema is not None:
            schema_ref = PyObjectRef.reference(self.schema)

            schema_desc = {
                "schema": {
                    "repo": schema_ref.repo,
                    "package": schema_ref.package,
                    "class": f"{schema_ref.module}.{schema_ref.object}",
                }
            }

            if schema_ref.object_version is not None:
                schema_desc["schema"]["version"] = schema_ref.object_version
            elif schema_ref.package_version is not None:
                schema_desc["schema"]["version"] = schema_ref.package_version

            if schema_ref.repo_revision is not None:
                schema_desc["schema"]["revision"] = schema_ref.repo_revision

            if schema_ref.docs_url is not None:
                schema_desc["schema"]["docs"] = schema_ref.docs_url

        return {
            **schema_desc,
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
            deep: Whether to perform a deep copy (copy all table data).

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
        conflict_policy: (
            DataConflictPolicy
            | dict[str, DataConflictPolicy | dict[str, DataConflictPolicy]]
        ) = "raise",
        inplace: bool = False,
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
            else DB(other) if isinstance(other, dict) else other.extract()
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
        bases: (
            set[str | Table]
            | dict[
                str | Table, sqla.Select | sqla.ColumnElement[bool] | pd.Series[bool]
            ]
        ),
        aggregators: dict[sqla.ForeignKey, sqla.Select] | None = None,
        inplace: bool = False,
    ) -> "DB":
        """Return new database with only the data that is related to the given bases.

        Args:
            bases:
                Tables to trim the database to. May be a set of table names or
                table instances, or a dict mapping tables to a query or
                a filter expression.
            aggregators:
                Mapping of foreign keys to aggregating queries. Given queries must
                group the foreign key's table by the foreign key.
            inplace: Whether to perform the trim in place.

        Returns:
            New database containing the trimmed data.
        """
        res = (
            self._isotropic_trim()
            if centers is None
            else self._centered_trim(centers, circuit_breakers)
        )
        return res

    def transfer(self, new_backend: str | Path | sqla.URL) -> "DB":
        """Transfer this database to a new backend.

        Args:
            new_backend: New backend to transfer to.

        Returns:
            New database instance.
        """
        raise NotImplementedError("Transferring databases is not yet supported.")

    def to_graph(
        self,
        node_attrs: set[str | sqla.Column] | None | Literal["all"] = "all",
        edge_attrs: set[str | sqla.Column] | None | Literal["all"] = "all",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export relational structure of database in a graph format.

        Args:
            node_attrs: Attributes to include in the node table. Defaults to all.
            edge_attrs: Attributes to include in the edge table. Defaults to all.

        Returns:
            Tuple of node and edge tables.

        E.g. for usage with `Gephi`_

        .. _Gephi: https://gephi.org/
        """
        node_dfs = [
            (
                n.df.reset_index().assign(table=n.sources)
                if isinstance(n.sources, str)
                else pd.concat(
                    [
                        cast(pd.DataFrame, n.df[p])
                        .drop_duplicates()
                        .reset_index()
                        .assign(table=s)
                        for p, s in n.sources.items()
                        if s not in self.join_tables
                    ],
                    ignore_index=True,
                )
            )
            for n in self.values()
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

    def __or__(self, other: "DB") -> "DB":  # noqa: D105
        return self.extend(other)

    def load(
        self,
        src: pd.DataFrame | sqla.Select,
        schema: type[S_cov] | None = None,
        name: str | None = None,
    ) -> Table:
        """Transfer dataframe or sql query results to manifested SQL table.

        Args:
            src: Source / definition of data.
            schema: Schema of the table to create.
            name: Name of the table to create.
            with_external_fk: Whether to include foreign keys to external tables.

        Returns:
            Reference to manifested SQL table.
        """
        match src:
            case pd.DataFrame():
                return self._load_from_df(src, schema, name)
            case sqla.Select():
                return self._load_from_query(src)

    # Internals:

    def _load_from_df(
        self,
        df: pd.DataFrame,
        schema: type[S_cov] | None = None,
        name: str | None = None,
    ) -> Table:
        table_name = f"{self.tag}_df_{name + '_' if name else ''}{_hash_df(df)}"

        if table_name not in self.schema.metadata().tables and not sqla.inspect(
            self.engine
        ).has_table(table_name, schema=None):
            cols = (
                {
                    c.key: sqla.Column(c.key, c.type, primary_key=c.primary_key)
                    for c in schema.__table__.columns
                }
                if schema is not None
                else _cols_from_df(df)
            )
            table = Table(
                sqla.Table(table_name, self.schema.metadata(), *cols.values())
            )
            table.sqla_table.create(self.engine)
            df.reset_index()[list(cols.keys())].to_sql(
                table_name, self.engine, if_exists="append", index=False
            )

        return (
            Table(self.schema.metadata().tables[table_name])
            if table_name in self.schema.metadata().tables
            else Table(
                sqla.Table(
                    table_name, self.schema.metadata(), autoload_with=self.engine
                )
            )
        )

    def _load_from_query(self, q: sqla.Select) -> Table:
        table_name = f"{self.tag}_query_{q.name}"

        if table_name in self.schema.metadata().tables:
            return Table(self.schema.metadata().tables[table_name])
        elif sqla.inspect(self.engine).has_table(table_name, schema=None):
            return Table(
                sqla.Table(
                    table_name,
                    self.schema.metadata(),
                    autoload_with=self.engine,
                )
            )
        else:
            sel_res = cast(sqla.Subquery | sqla.Table, q.__clause_element__())

            sqla_table = (
                q.schema.__table__.to_metadata(
                    self.schema.metadata(),
                    schema=None,  # type: ignore
                    referred_schema_fn=partial(
                        _map_foreignkey_schema, schema_dict=self.schema.schema_dict()
                    ),
                    name=table_name,
                )
                if q.schema is not None and isinstance(q.schema.__table__, sqla.Table)
                else sqla.Table(
                    table_name,
                    self.schema.metadata(),
                    *(
                        sqla.Column(name, col.type, primary_key=col.primary_key)
                        for name, col in sel_res.columns.items()
                    ),
                )
            )
            if not with_external_fk:
                _remove_external_fk(sqla_table)
            sqla_table.create(self.engine)

            table = Table(sqla_table)

            with self.engine.begin() as con:
                con.execute(
                    sqla.insert(table).from_select(
                        sel_res.exported_columns.keys(), sel_res
                    )
                )

            return table

    def _to_excel(self, path: Path | str | None = None) -> None:
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
                                    lambda df: (
                                        df.rename_axis("id", axis="index")
                                        if not df.index.name
                                        else df
                                    )
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
                            try:
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
                            c: (
                                target_df.loc[idx_sel][tc].dropna().unique()
                                if tc in target_df.columns
                                else idx_sel
                            )
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
