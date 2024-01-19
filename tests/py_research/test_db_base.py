"""Test the db module."""

from datetime import date
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import pytest
from py_research.data import parse_dtype
from py_research.db import DB, DBSchema, Table


@pytest.fixture
def table_df_projects():
    """Return a dataframe for the projects table."""
    return (
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": [
                    "baking cake",
                    "cleaning shoes",
                    "fixing cars",
                    "writing book",
                ],
                "start": [
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 1, 3),
                    date(2020, 1, 4),
                ],
                "end": [
                    date(2020, 1, 4),
                    date(2020, 1, 5),
                    date(2020, 1, 6),
                    date(2020, 1, 7),
                ],
                "status": ["done", "done", "started", "abandoned"],
            }
        )
        .set_index("id")
        .apply(parse_dtype, axis="index")
    )


@pytest.fixture
def table_df_persons():
    """Return a dataframe for the persons table."""
    return (
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "name": ["John", "Jane", "Joe"],
                "age": [20, 30, 40],
                "height": [1.8, 1.7, 1.9],
                "weight": [80, 70, 90],
                "gender": ["male", "male", "diverse"],
            }
        )
        .set_index("id")
        .apply(parse_dtype, axis="index")
    )


@pytest.fixture
def table_df_memberships():
    """Return a dataframe for the memberships table."""
    return (
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "project": [1, 2, 3, 1],
                "member": ["a", "b", "c", "b"],
                "role": ["leader", "member", "member", "member"],
            }
        )
        .set_index(["id"])
        .apply(parse_dtype, axis="index")
    )


@pytest.fixture
def table_df_tasks():
    """Return a dataframe for the tasks table."""
    return (
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7],
                "name": ["task1", "task2", "task3", "task4", "task5", "task6", "task7"],
                "project": [1, 1, 1, 2, 2, 3, 3],
                "assignee": ["a", "a", "b", "a", "b", "a", "c"],
                "status": ["todo", "todo", "done", "todo", "todo", "todo", "done"],
            }
        )
        .set_index("id")
        .apply(parse_dtype, axis="index")
    )


@pytest.fixture
def relations():
    """Return a dict of relations between table columns."""
    return {
        ("memberships", "project"): ("projects", "id"),
        ("memberships", "member"): ("persons", "id"),
        ("tasks", "project"): ("projects", "id"),
        ("tasks", "assignee"): ("persons", "id"),
        ("persons", "gender"): ("genders", "name"),
    }


@pytest.fixture
def join_tables():
    """Return a list of join tables."""
    return set(["memberships"])


@pytest.fixture
def db_from_tables(
    table_df_projects: pd.DataFrame,
    table_df_persons: pd.DataFrame,
    table_df_memberships: pd.DataFrame,
    table_df_tasks: pd.DataFrame,
    relations: dict[tuple[str, str], tuple[str, str]],
    join_tables: set[str],
):
    """Return DB instance with tables."""
    db = DB(
        {
            "projects": table_df_projects,
            "persons": table_df_persons,
            "memberships": table_df_memberships,
            "tasks": table_df_tasks,
        },
        relations,
        join_tables,
    )
    return db


def test_create_db_empty():
    """Test the creation of a DB instance."""
    db = DB()
    assert isinstance(db, DB)


def test_create_db_from_tables(db_from_tables: DB):
    """Test the creation of a DB instance from tables."""
    assert isinstance(db_from_tables, DB)
    assert set(db_from_tables.keys()) == {
        "projects",
        "persons",
        "memberships",
        "tasks",
        "genders",
    }
    assert isinstance(db_from_tables["projects"].df, pd.DataFrame)
    assert "persons" in db_from_tables
    assert "genders" in db_from_tables
    assert set(db_from_tables["genders"]["name"]) == {"male", "diverse"}
    assert len(db_from_tables) == 5


def test_trim_db_anisotropic(db_from_tables: DB):
    """Test the trimming of a DB instance."""
    db = db_from_tables.trim()
    assert isinstance(db, DB)
    assert set(db["projects"].df.index) == {1, 2, 3}


def test_extend_db(db_from_tables: DB):
    """Test the extension of a DB instance."""
    db = db_from_tables.extend(
        {
            "projects": pd.DataFrame(
                {
                    "id": [5, 6],
                    "name": ["baking cake", "cleaning shoes"],
                    "start": [date(2020, 1, 1), date(2020, 1, 2)],
                    "end": [date(2020, 1, 4), date(2020, 1, 5)],
                    "status": ["done", "done"],
                }
            ).set_index("id"),
            "persons": pd.DataFrame(
                {
                    "id": ["d", "e"],
                    "name": ["John", "Jane"],
                    "age": [20, 30],
                    "height": [1.8, 1.7],
                    "weight": [80, 70],
                }
            ).set_index("id"),
            "memberships": pd.DataFrame(
                {
                    "id": [4, 5],
                    "project": ["a", "b"],
                    "member": [1, 2],
                    "role": ["leader", "member"],
                }
            ).set_index(["id"]),
            "tasks": pd.DataFrame(
                {
                    "id": [8, 9],
                    "name": ["task8", "task9"],
                    "project": [1, 1],
                    "assignee": ["a", "b"],
                    "status": ["todo", "todo"],
                }
            ).set_index("id"),
        }
    )
    assert isinstance(db, DB)
    assert set(db["projects"].df.index) == {1, 2, 3, 4, 5, 6}
    assert set(db["persons"].df.index) == {"a", "b", "c", "d", "e"}
    assert set(db["memberships"].df.index) == {0, 1, 2, 3, 4, 5}
    assert set(db["tasks"].df.index) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_filter_db(db_from_tables: DB):
    """Test the filtering of a DB instance."""
    db = db_from_tables.filter(
        {"projects": db_from_tables["projects"].df["status"] == "done"}
    )
    assert isinstance(db, DB)
    assert set(db["projects"].df.index) == {1, 2}
    assert set(db["persons"].df.index) == {"a", "b"}
    assert len(db["memberships"].df) == 3
    assert len(db["tasks"].df) == 6


def test_save_load_db(db_from_tables: DB):
    """Test the saving of a DB instance."""
    tempdir = Path(gettempdir())
    db_file_path = tempdir / "test_db.xlsx"

    db_from_tables.schema = DBSchema  # Use base class as empty schema.
    db_from_tables.save(db_file_path)
    assert db_file_path.is_file()

    db_loaded = DB.load(db_file_path)
    assert isinstance(db_loaded, DB)
    for t in db_loaded.keys():
        pd.testing.assert_frame_equal(db_from_tables[t].df, db_loaded[t].df)
    assert db_from_tables.relations == db_loaded.relations
    assert db_from_tables.join_tables == db_loaded.join_tables
    assert db_from_tables.updates == db_loaded.updates
    assert db_from_tables.schema == db_loaded.schema


def test_table_to_from_excel(db_from_tables: DB):
    """Test saving / loading table to / from excel."""
    tempdir = Path(gettempdir())
    table_file_path = tempdir / "test_table.xlsx"

    db_from_tables["persons"].to_excel(table_file_path)
    loaded_persons = Table.from_excel(table_file_path)
    pd.testing.assert_frame_equal(db_from_tables["persons"].df, loaded_persons.df)
    assert db_from_tables["persons"].source_map == loaded_persons.source_map

    merged = db_from_tables["persons"].merge(right=db_from_tables["projects"])
    merged.to_excel(table_file_path)
    merged_loaded = Table.from_excel(table_file_path)
    pd.testing.assert_frame_equal(merged.df, merged_loaded.df)
    assert merged.source_map == merged_loaded.source_map


def test_filter_db_table(db_from_tables: DB):
    """Test the filtering of a DB table."""
    table = db_from_tables["projects"]
    filtered_table = table.filter(table.df["status"] == "done")
    assert isinstance(filtered_table, Table)
    assert set(filtered_table.df.index) == {1, 2}


def test_merge_db_table_forward(db_from_tables: DB):
    """Test the forward merging of a DB table."""
    table = db_from_tables["tasks"]

    table_merged = table.merge(link_to_right="project", naming="source")
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten(".", "always")
    assert set(db_from_tables["tasks"]["name"].unique()) == set(
        table_flat["tasks.name"].unique()
    )
    assert set(db_from_tables["tasks"]["project"].unique()) == set(
        table_flat["projects.id"].unique()
    )


def test_merge_db_table_forward_virtual(db_from_tables: DB):
    """Test the forward merging of a DB table."""
    table = db_from_tables["persons"]

    table_merged = table.merge(link_to_right="gender", naming="source")
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten(".", "always")
    assert set(db_from_tables["persons"]["name"].unique()) == set(
        table_flat["persons.name"].unique()
    )
    assert set(db_from_tables["persons"]["gender"].unique()) == set(
        table_flat["genders.name"].unique()
    )


def test_merge_db_table_forward_filtered(db_from_tables: DB):
    """Test the forward merging of a DB table with filters."""
    left_table = db_from_tables["tasks"].filter(
        db_from_tables["tasks"]["status"] == "todo"
    )
    right_table = db_from_tables["projects"].filter(
        db_from_tables["projects"]["status"] == "done"
    )

    table_merged = left_table.merge(
        right=right_table, link_to_right="project", naming="path"
    )
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten("->", "always")
    assert set(left_table["name"].unique()) == set(table_flat["tasks->name"].unique())
    assert set(right_table.df.index) == set(
        table_flat["tasks->project->id"].dropna().unique()
    )


def test_merge_db_table_backward(db_from_tables: DB):
    """Test the backward merging of a DB table."""
    table = db_from_tables["projects"]

    table_merged = table.merge(right=db_from_tables["tasks"], naming="path")
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten("->", "always")
    assert set(db_from_tables["projects"]["name"].unique()) == set(
        table_flat["projects->name"].unique()
    )
    assert set(db_from_tables["tasks"]["project"].unique()) == set(
        table_flat["projects<=tasks->project"].dropna().unique()
    )


def test_merge_db_table_backward_explicit(db_from_tables: DB):
    """Test the explicit backward merging of a DB table."""
    table = db_from_tables["projects"]

    table_merged = table.merge(
        right=db_from_tables["tasks"], link_to_left="project", naming="path"
    )
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten("->", "always")
    assert set(db_from_tables["projects"]["name"].unique()) == set(
        table_flat["projects->name"].unique()
    )
    assert set(db_from_tables["tasks"]["project"].unique()) == set(
        table_flat["projects<=tasks->project"].dropna().unique()
    )


def test_merge_db_table_double(db_from_tables: DB):
    """Test the double merging of a DB table."""
    table = db_from_tables["persons"]

    table_merged = table.merge(right=db_from_tables["projects"], naming="path")
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten("->", "always")
    assert set(db_from_tables["persons"]["name"].unique()) == set(
        table_flat["persons->name"].unique()
    )
    assert set(db_from_tables["memberships"]["project"].unique()) == set(
        table_flat["persons->project->id"].unique()
    )


def test_merge_db_table_double_filtered(db_from_tables: DB):
    """Test the double merging of a DB table with filtering."""
    left_table = db_from_tables["persons"].filter(db_from_tables["persons"]["age"] < 30)
    link_table = db_from_tables["memberships"].filter(
        db_from_tables["memberships"]["role"] == "leader"
    )

    table_merged = left_table.merge(
        right=db_from_tables["projects"], link_table=link_table, naming="path"
    )
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten("->", "always")
    assert set(left_table["name"].unique()) == set(table_flat["persons->name"].unique())
    assert set(table_flat["persons->project->id"].unique()) == {1}


def test_merge_db_table_double_indefinite(db_from_tables: DB):
    """Test the indefinite double merging of a DB table."""
    table = db_from_tables["persons"]

    table_merged = table.merge(link_table=db_from_tables["memberships"], naming="path")
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten("->", "always")
    assert set(db_from_tables["persons"]["name"].unique()) == set(
        table_flat["persons->name"].unique()
    )
    assert set(db_from_tables["memberships"]["project"].unique()) == set(
        table_flat["persons->project->id"].unique()
    )


def test_merge_db_table_double_explicit(db_from_tables: DB):
    """Test the indefinite double merging of a DB table."""
    table = db_from_tables["persons"]

    table_merged = table.merge(
        link_table=db_from_tables["memberships"],
        right=db_from_tables["projects"],
        naming="path",
    )
    assert isinstance(table_merged, Table)
    assert table_merged.df.columns.nlevels == 2

    table_flat = table_merged.flatten("->", "always")
    assert set(db_from_tables["persons"]["name"].unique()) == set(
        table_flat["persons->name"].unique()
    )
    assert set(db_from_tables["memberships"]["project"].unique()) == set(
        table_flat["persons->project->id"].unique()
    )


def test_extract_db_table(db_from_tables: DB):
    """Test the extraction of a db table."""
    table = db_from_tables["projects"]

    table_merged = table.merge(right=db_from_tables["tasks"], naming="path")

    extracted_db1 = table_merged.extract()
    assert isinstance(extracted_db1, DB)
    assert set(extracted_db1.keys()) == {
        "projects",
        "tasks",
        "memberships",
        "persons",
        "genders",
    }
    assert set(extracted_db1["projects"].df.index) == {1, 2, 3, 4}

    extracted_db2 = table_merged.extract(with_relations=False)
    assert isinstance(extracted_db2, DB)
    assert set(extracted_db2.keys()) == {"projects", "tasks"}
    assert set(extracted_db2["projects"].df.index) == {1, 2, 3, 4}


def test_extend_db_table(db_from_tables: DB):
    """Test the extending of a DB table."""
    table = db_from_tables["projects"]

    table_extended = table.extend(
        pd.DataFrame(
            {
                "id": [5, 6],
                "name": ["baking cake", "cleaning shoes"],
                "start": [date(2020, 1, 1), date(2020, 1, 2)],
                "end": [date(2020, 1, 4), date(2020, 1, 5)],
                "status": ["done", "done"],
            }
        ).set_index("id")
    )
    assert isinstance(table_extended, Table)
    assert set(table_extended.df.index) == {1, 2, 3, 4, 5, 6}


def test_db_to_graph(db_from_tables: DB):
    """Test transformation of database to graph."""
    nodes, edges = db_from_tables.to_graph(
        [
            trimmed_proj := db_from_tables["projects"].trim(["name", "status"]),
            trimmed_pers := db_from_tables["persons"].trim(["name", "age"]),
            "tasks",
        ]
    )
    assert isinstance(nodes, pd.DataFrame)
    assert len(nodes) == len(trimmed_proj.df) + len(trimmed_pers.df) + len(
        db_from_tables["tasks"].df
    )
    assert set(nodes.columns) == {
        "node_id",
        "table",
        "id",
        "name",
        "status",
        "project",
        "assignee",
        "age",
    }
    assert isinstance(edges, pd.DataFrame)
    assert set(edges.columns) == {"source", "target", "ltr", "rtl", "role"}


def test_db_to_graph_merged(db_from_tables: DB):
    """Test transformation of database to graph."""
    table = db_from_tables["persons"]
    table_merged = table.merge(right=db_from_tables["projects"], naming="path")

    nodes, edges = db_from_tables.to_graph([table_merged])
    assert isinstance(nodes, pd.DataFrame)
    assert set(nodes.columns) == {
        "node_id",
        "table",
        "id",
        "name",
        "status",
        "age",
        "start",
        "end",
        "height",
        "weight",
        "gender",
    }
    assert isinstance(edges, pd.DataFrame)
    assert set(edges.columns) == {"source", "target", "ltr", "rtl", "role"}
