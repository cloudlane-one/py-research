"""Static schemas for universal relational databases."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from functools import cmp_to_key, reduce
from typing import TYPE_CHECKING, Any, ClassVar, Generic, cast, overload

import polars as pl
import sqlalchemy as sqla
import sqlalchemy.types as sqla_type
from typing_extensions import TypeVar

from py_research.caching import cached_prop
from py_research.hashing import gen_int_hash
from py_research.types import UUID4, Not

from .data import RwxT, U


@dataclass
class Table(Generic[RecT, BackT, RwxT]):
    """Table matching a table model, may be filtered."""

    base: Base[Any, BackT]
    rec_types: set[type[RecT]]
    content: (
        str | sqla.FromClause | pd.DataFrame | pl.DataFrame | Iterable[Record] | None
    ) = None

    @cached_prop
    def fqn(self) -> str:
        """Fully qualified name of this table based on its record types."""
        return "|".join(rec._fqn for rec in self.rec_types)

    @cached_prop
    def sql(self) -> sqla.FromClause:
        """This table's from clause."""
        match self.content:
            case str():
                return sqla.table(self.content)
            case sqla.FromClause():
                return self.content
            case pd.DataFrame() | pl.DataFrame():
                return self._upload_df(self.content)
            case Iterable():
                return self._upload_df(records_to_df(self.content))
            case None:
                return self._get_base_union()

    def df(self: Table[Any, DynBackendID, Any]) -> pl.DataFrame:
        """Return the table as a DataFrame."""
        return pl.read_database(self.sql, self.base.connection)

    def mutate(
        self: Table[Any, Any, CRUD],
        input_table: Table[RecT2, BackT, Any],
        mode: MutationMode = "update",
    ) -> None:
        """Mutate given table."""
        tables = {rec: table for rec, (table, _) in self._base_table_map.items()}
        input_sql = input_table.sql

        statements: list[sqla.Executable] = []

        if mode in ("replace", "delete"):
            # Delete current records first.
            statements += [
                safe_delete(table, input_sql, self.base.engine)
                for table in tables.values()
            ]

        if mode != "delete":
            for base_type, table in tables.items():
                # Rename input columns to match base table columns.
                input_sql = input_sql.select().with_only_columns(
                    *(
                        input_sql.c[col.name].label(name)
                        for name, col in self._base_col_map[base_type].items()
                    )
                )

                if mode == "update":
                    statements.append(safe_update(table, input_sql, self.base.engine))
                else:
                    statements.append(
                        safe_insert(
                            table,
                            input_sql,
                            self.base.engine,
                            upsert=(mode == "upsert"),
                        )
                    )

        # Execute delete / insert / update statements.
        con = self.base.connection
        for statement in statements:
            con.execute(statement)

        if self.base.backend_type == "excel-file":
            self.base._save_to_excel(
                # {self.record_type, *self.record_type._record_superclasses}
            )

    def validate(
        self, inspector: sqla.Inspector | None = None, required: bool = False
    ) -> None:
        """Perform pre-defined schema validations."""
        inspector = inspector or sqla.inspect(self.base.engine)

        for table, _ in self._base_table_map.values():
            validate_sql_table(inspector, table, required)

    def __clause_element__(self) -> sqla.FromClause:
        """Return the table clause element."""
        return self.sql

    @cached_prop
    def _base_table_map(
        self,
    ) -> dict[type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]]:
        """Return the base SQLAlchemy table objects for all record types."""
        base_table_map: dict[
            type[Record], tuple[sqla.Table, sqla.ColumnElement[bool] | None]
        ] = {}
        for rec in self.rec_types:
            super_recs = (r for r in rec._model_superclasses if issubclass(r, Record))
            for base_rec in (rec, *super_recs):
                if base_rec not in base_table_map:
                    table = self.base._get_base_table(rec)
                    join_on = (
                        None
                        if base_rec is rec
                        else reduce(
                            sqla.and_,
                            (
                                pk_sel == pk_super
                                for pk_sel, pk_super in zip(
                                    self.base._map_pk(rec).columns,
                                    self.base._map_pk(base_rec).columns,
                                )
                            ),
                        )
                    )
                    base_table_map[base_rec] = (table, join_on)

        return dict(
            sorted(
                base_table_map.items(),
                key=cmp_to_key(
                    lambda x, y: (
                        -1
                        if issubclass(y[0], x[0])
                        else 1 if issubclass(x[0], y[0]) else 0
                    )
                ),
            )
        )

    @cached_prop
    def _base_col_map(self) -> dict[type[Record], dict[str, sqla.Column]]:
        """Return the column set for all record types."""
        name_attr = attrgetter("name" if len(self.rec_types) == 1 else "fqn")
        return {
            rec: {
                name_attr(col): sql_col
                for col, sql_col in self.base._map_cols(rec).items()
            }
            for rec in self._base_table_map.keys()
        }

    def _get_base_union(self) -> sqla.FromClause:
        """Recursively join all bases of this record to get the full data."""
        sel_tables = {
            rec: table
            for rec, (table, join_on) in self._base_table_map.items()
            if join_on is None
        }

        def _union(
            union_stage: tuple[type[Record] | None, sqla.Select],
            next_union: tuple[type[Record], sqla.Join | sqla.Table],
        ) -> tuple[type[Record] | None, sqla.Select]:
            left_rec, left_table = union_stage
            right_rec, right_table = next_union

            right_table = right_table.select().with_only_columns(
                *(
                    col.label(name)
                    for name, col in self._base_col_map[right_rec].items()
                )
            )

            if left_rec is not None and len(
                set(left_rec._pk.columns) & set(right_rec._pk.columns)  # type: ignore
            ):
                return left_rec, coalescent_join_sql(
                    left_table.alias(),
                    right_table.alias(),
                    reduce(
                        sqla.and_,
                        (
                            left_pk == right_pk
                            for left_pk, right_pk in zip(
                                self.base._map_pk(left_rec).columns,
                                self.base._map_pk(right_rec).columns,
                            )
                        ),
                    ),
                )
            else:
                return None, left_table.union(right_table).select()

        if len(sel_tables) > 1:
            rec_query_items = list(sel_tables.items())
            first_rec, first_select = rec_query_items[0]
            first: tuple[type[Record] | None, sqla.Select] = (
                first_rec,
                first_select.select().with_only_columns(
                    *(
                        col.label(name)
                        for name, col in self._base_col_map[first_rec].items()
                    )
                ),
            )

            _, base_union = reduce(_union, rec_query_items[1:], first)
        else:
            base_union = list(sel_tables.values())[0]

        base_join = base_union
        for base_table, join_on in self._base_table_map.values():
            if join_on is not None:
                base_join = base_join.join(base_table, join_on)

        res = base_join.subquery() if isinstance(base_join, sqla.Select) else base_join
        setattr(res, "_tab", self)
        return res

    def _get_upload_table(self) -> sqla.Table:
        cols = {
            name: col.copy()
            for cols in self._base_col_map.values()
            for name, col in cols.items()
        }
        for name, col in cols.items():
            col.name = name

        return sqla.Table(
            f"upload/{self.fqn.replace('.', '_')}/{token_hex(5)}",
            self.base._metadata,
            *cols.values(),
        )

    def _upload_df(self, df: pd.DataFrame | pl.DataFrame) -> sqla.Table:
        table = self._get_upload_table()

        if isinstance(df, pl.DataFrame):
            df.write_database(
                str(table),
                self.base.connection,
                if_table_exists="replace",
            )
        else:
            df.reset_index().to_sql(
                table.name,
                self.base.connection,
                if_exists="replace",
                index=False,
            )

        return table


ColT = TypeVar("ColT")

KeyT = TypeVar("KeyT", bound=Hashable, default=Any)


@dataclass(kw_only=True)
class Column(
    Attr[ColT, RwxT, OwnT],
    sqla.SQLColumnExpression[ColT],
    Generic[ColT, RwxT, OwnT, BackT],
):
    """Column property for a table model."""

    _table: Table[OwnT] | None = None

    sql_default: sqla.ColumnElement[ColT] | None = None

    @cached_prop
    def sql_type(self) -> sqla_type.TypeEngine:
        """Return the SQL type of this column."""
        return sqla_type.to_instance(self.common_value_type)

    @property
    def sql_col(self) -> sqla.ColumnClause[ColT]:
        """Return the SQL column of this property."""
        table = (
            self._table
            if self._table is not None
            else (
                self._owner._table
                if self._owner is not None and issubclass(self._owner, Record)
                else None
            )
        )
        col = sqla.column(
            self.name,
            self.sql_type,
            _selectable=table.sql if table else None,
        )
        setattr(col, "_prop", self)
        return col

    def __clause_element__(self) -> sqla.ColumnClause[ColT]:
        """Return the column clause element."""
        return self.sql_col

    @cached_prop
    def sql_comparator(self) -> sqla_type.TypeEngine.Comparator:
        """Return the comparator for this column."""
        return self.sql_type.comparator_factory(self.sql_col)

    if not TYPE_CHECKING:

        def __getattr__(self, key: str) -> Any:
            """Support rich comparison methods."""
            return getattr(self.sql_comparator, key)

    def label(self, name: str | None) -> sqla.Label[ColT]:
        """Label this column."""
        return self.sql_col.label(name)

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> ColT: ...

    @overload
    def __get__(
        self, instance: None, owner: type[RecT]
    ) -> Column[ColT, RwxT, RecT, Symbolic]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride] (TODO: report pyright bug)
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this column."""
        super_get = Attr.__get__(self, instance, owner)
        if super_get is not Ungot():
            return super_get

        assert instance is not None

        if isinstance(instance, Record):
            instance._load_dict()

        if self.name not in instance.__dict__:
            if self.default is not Not.defined:
                instance.__dict__[self.name] = self.default
            else:
                assert self.default_factory is not None
                instance.__dict__[self.name] = self.default_factory()

        return instance.__dict__[self.name]

    def __set__(self: Column[Any, U, Any], instance: Model, value: ColT) -> None:
        """Set the value of this attribute."""
        super_set = Attr.__set__(self, instance, value)
        if super_set is Unset():
            return

        instance.__dict__[self.name] = value

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self._owner, self._table))


TupT = TypeVar("TupT")


@dataclass
class ColTuple(Prop[TupT, OwnT, R]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...]

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> TupT: ...

    @overload
    def __get__(self, instance: None, owner: type[RecT]) -> ColTuple[TupT, RecT]: ...

    @overload
    def __get__(self, instance: Any, owner: type | None) -> Any: ...

    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, instance: Any, owner: type | None
    ) -> Any:
        """Get the value of this attribute."""
        val = tuple(col.__get__(instance, owner) for col in self.columns)
        return val[0] if len(val) == 1 else val

    def __set__(self, instance: Any, value: Any) -> None:
        """Set the value of this attribute."""
        raise AttributeError("Cannot set an index.")

    def gen_value_map(self, val: Any) -> dict[str, Hashable]:
        """Generate a key-value map for this key."""
        idx_names = [c.name for c in self.columns]

        if len(idx_names) == 1:
            return {idx_names[0]: val}

        assert isinstance(val, tuple) and len(val) == len(idx_names)
        return {idx_name: idx_val for idx_name, idx_val in zip(idx_names, val)}

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash((self.name, self.context, self.columns))


@dataclass(eq=False)
class Key(ColTuple[KeyT, OwnT]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...] = ()

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if len(self.columns) == 0:
            self.columns = (
                Column(
                    _name=self.name,
                    _type=Column[self.common_value_type],
                    _owner=self.context,
                ),
            )
            setattr(owner, f"{name}_col", self.columns[0])


TgtT = TypeVar("TgtT", bound="Record", covariant=True)


@dataclass(eq=False)
class ForeignKey(ColTuple[KeyT, OwnT], Generic[TgtT, KeyT, OwnT]):
    """Index property for a record type."""

    columns: tuple[Column[Hashable, Any, OwnT], ...] = field(init=False)

    column_map: dict[Column[Hashable, Any, OwnT], Column[Hashable, Any, TgtT]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:  # noqa: D105
        self.columns = (
            tuple(self.column_map.keys()) if self.column_map is not None else ()
        )

    def __set_name__(self, owner: type[OwnT], name: str) -> None:  # noqa: D105
        super().__set_name__(owner, name)

        if self.column_map is None:
            pk_cols = self.target._pk.columns

            col_map = {
                Column(
                    _name=f"{self.name}_{pk_col.name}_fk",
                    _type=Column[pk_col.common_value_type],
                    _owner=self.context,
                ): pk_col
                for pk_col in pk_cols
            }

            self.columns = tuple(col_map.keys())
            for col in col_map.keys():
                setattr(owner, col.name, col)

    @property
    def target(self) -> type[TgtT]:
        """Return the target type for this foreign key."""
        target_type = self.typeargs[TgtT]
        assert isinstance(target_type, type)
        return cast(type[TgtT], target_type)


class RecordTable:
    """Table, which a record belongs to."""

    @overload
    def __get__(
        self, instance: None, owner: type[RecT2]
    ) -> Table[RecT2, Symbolic, R]: ...

    @overload
    def __get__(
        self, instance: RecT2, owner: type[RecT2]
    ) -> Table[RecT2, DynBackendID, Any]: ...

    def __get__(  # noqa: D105
        self, instance: Record | None, owner: type[Record]
    ) -> Table[Record, Any, Any]:
        return (
            symbol_base.table(owner)
            if instance is None
            else instance._base.table(owner)
        )


class Record(Model, Generic[KeyT]):
    """Schema for a table in a database."""

    _root_class = True

    _table_name: ClassVar[str] | None = None
    _type_map: ClassVar[dict[type, sqla.types.TypeEngine]] = {
        str: sqla.types.String().with_variant(
            sqla.types.String(50), "oracle"
        ),  # Avoid oracle error when VARCHAR has no size parameter
        UUID4: sqla.types.CHAR(36),
    }

    _table = RecordTable()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a new record subclass."""
        super().__init_subclass__(**kwargs)

        for superclass in cls._model_superclasses:
            if superclass in cls.__bases__ and issubclass(superclass, Record):
                super_pk = superclass._pk  # type: ignore
                col_map = {
                    Column(
                        _name=pk_col.name,
                        _type=Column[pk_col.common_target_type],
                        _owner=cls,
                    ): pk_col
                    for pk_col in super_pk.columns
                }
                for col in col_map.keys():
                    setattr(cls, col.name, col)

                pk = Key[super_pk.typeargs[KeyT], cls](columns=tuple(col_map.keys()))
                setattr(cls, pk.name, pk)

                fk = ForeignKey[superclass, super_pk.typeargs[KeyT], cls](
                    column_map=col_map
                )
                setattr(cls, fk.name, fk)

    @classmethod
    def _default_table_name(cls) -> str:
        """Return the name of the table for this schema."""
        name = cls._table_name or cls._fqn
        fqn_parts = name.split(".")

        name = fqn_parts[-1]
        for part in reversed(fqn_parts[:-1]):
            name = part + "_" + name
            if len(name) > 40:
                break

        return name

    @classmethod
    def _cols(cls) -> set[Column]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Column)
        }

    @classmethod
    def _keys(cls) -> set[Key]:
        """Columns of this record type's table."""
        return {
            prop for prop in cls._get_class_props().values() if isinstance(prop, Key)
        }

    @classmethod
    def _fks(cls) -> set[ForeignKey]:
        """Columns of this record type's table."""
        return {
            prop
            for prop in cls._get_class_props().values()
            if isinstance(prop, ForeignKey)
        }

    @classmethod
    def _pl_schema(cls) -> dict[str, pl.DataType | type | None]:
        """Return the schema of the dataset."""
        return get_pl_schema(
            {name: col for name, col in cls._props.items() if isinstance(col, Column)}
        )

    @classmethod
    def __clause_element__(cls) -> sqla.TableClause:
        """Return the table clause element."""
        return cls._table._base_table_map[cls][0]

    _published: bool = False
    _base: Attr[Base[Any, DynBackendID]] = Attr(default_factory=Base)

    _pk: Key[KeyT, Self]

    @cached_prop
    def _whereclause(self) -> sqla.ColumnElement[bool]:
        cols = {
            col.name: sql_col
            for col, sql_col in self._base._map_cols(type(self)).items()
        }
        pk_map = type(self)._pk.gen_value_map(self._pk)

        return sqla.and_(
            *(cols[col] == val for col, val in pk_map.items()),
        )

    def _load_dict(self) -> None:
        if not self._published or self._pk in self._base._get_valid_cache_set(
            type(self)
        ):
            return

        query = self._table.sql.select().where(self._whereclause)

        with self._base.engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()

        if row is not None:
            self.__dict__.update(row._asdict())

    def _save_dict(self) -> None:
        df = records_to_df([self])
        self._table.mutate(self._base.table({type(self)}, df))
        self._base._get_valid_cache_set(type(self)).remove(self._pk)

    def __hash__(self) -> int:
        """Identify the record by database and id."""
        return gen_int_hash((self._base if self._published else None, self._pk))

    def __eq__(self, value: Hashable) -> bool:
        """Check if the record is equal to another record."""
        return hash(self) == hash(value)
