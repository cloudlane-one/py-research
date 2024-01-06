"""Test the db module."""

from datetime import date
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import pytest

from py_research.db import DB, Table


@pytest.fixture
def table_df_projects():
    """Return a dataframe for the projects table."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["baking cake", "cleaning shoes", "fixing cars", "writing book"],
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
    ).set_index("id")


@pytest.fixture
def table_df_persons():
    """Return a dataframe for the persons table."""
    return pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["John", "Jane", "Joe"],
            "age": [20, 30, 40],
            "height": [1.8, 1.7, 1.9],
            "weight": [80, 70, 90],
        }
    ).set_index("id")


@pytest.fixture
def table_df_memberships():
    """Return a dataframe for the memberships table."""
    return pd.DataFrame(
        {
            "project": [1, 2, 3, 1],
            "member": ["a", "b", "c", "b"],
            "role": ["leader", "member", "member", "member"],
        }
    )


@pytest.fixture
def table_df_tasks():
    """Return a dataframe for the tasks table."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "name": ["task1", "task2", "task3", "task4", "task5", "task6", "task7"],
            "project": [1, 1, 1, 2, 2, 3, 3],
            "assignee": ["a", "a", "b", "a", "b", "a", "c"],
            "status": ["todo", "todo", "done", "todo", "todo", "todo", "done"],
        }
    ).set_index("id")


@pytest.fixture
def relations():
    """Return a dict of relations between table columns."""
    return {
        ("memberships", "project"): ("projects", "id"),
        ("memberships", "member"): ("persons", "id"),
        ("tasks", "project"): ("projects", "id"),
        ("tasks", "assignee"): ("persons", "id"),
    }


@pytest.fixture
def join_tables():
    """Return a list of join tables."""
    return set("memberships")


@pytest.fixture
def db_from_tables(
    table_df_projects: pd.DataFrame,
    table_df_persons: pd.DataFrame,
    table_df_memberships: pd.DataFrame,
    table_df_tasks: pd.DataFrame,
    relations: dict[tuple[str, str], tuple[str, str]],
    join_tables: set[str],
):
    """Test the creation of a DB instance from tables."""
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
    }
    assert isinstance(db_from_tables["projects"].df, pd.DataFrame)
    assert "persons" in db_from_tables
    assert len(db_from_tables) == 4


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
                    "project": ["a", "b"],
                    "member": [1, 2],
                    "role": ["leader", "member"],
                }
            ),
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
    assert set(db["memberships"].df.index) == {0, 1, 2, 3}
    assert set(db["tasks"].df.index) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def filter_db(db_from_tables: DB):
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

    db_from_tables.save(db_file_path)
    assert db_file_path.is_file()

    db_loaded = DB.load(db_file_path)
    assert isinstance(db_loaded, DB)
    assert set(db_loaded.keys()) == set(db_from_tables.keys())


def test_filter_db_table(db_from_tables: DB):
    """Test the filtering of a DB table."""
    table = db_from_tables["projects"]
    filtered_table = table.filter(table.df["status"] == "done")
    assert isinstance(filtered_table, Table)
    assert set(filtered_table.df.index) == {1, 2}


def test_merge_db_table(db_from_tables: DB):
    """Test the merging of a DB table."""
    table = db_from_tables["projects"]

    table_merged = table.merge(db_from_tables["tasks"])
    assert isinstance(table_merged, Table)

    table_merged_2 = table.merge(db_from_tables["persons"])
    assert isinstance(table_merged_2, Table)


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
