"""Test db importing module."""

import pytest

from py_research.db import DB
from py_research.db.importing import TableMap, tree_to_db


@pytest.fixture
def nested_db_dict() -> dict:
    """Return nested data dict for import testing."""
    return {
        "resultCount": 3,
        "search": "test",
        "results": [
            {
                "project_name": "baking cake",
                "project_start": "2020-01-01",
                "project_end": "2020-01-04",
                "project_status": "done",
                "organization_name": "Bakery",
                "organization_address": "Main Street 1",
                "organization_city": "Bakerville",
                "tasks": [
                    {
                        "task_name": "task1",
                        "task_assignee": "John",
                        "task_status": "todo",
                    },
                    {
                        "task_name": "task2",
                        "task_assignee": "John",
                        "task_status": "todo",
                    },
                    {
                        "task_name": "task3",
                        "task_assignee": "Jane",
                        "task_status": "done",
                    },
                ],
                "members": [
                    {"name": "John", "age": 20, "role": "baker"},
                    {"name": "John", "age": 20, "role": "manager"},
                ],
            },
            {
                "project_name": "cleaning shoes",
                "project_start": "2020-01-02",
                "project_end": "2020-01-05",
                "project_status": "done",
                "organization_name": "Shoe Shop",
                "organization_address": "Main Street 2",
                "organization_city": "Shoetown",
                "tasks": [
                    {
                        "task_name": "task4",
                        "task_assignee": "John",
                        "task_status": "todo",
                    },
                    {
                        "task_name": "task5",
                        "task_assignee": "Jane",
                        "task_status": "todo",
                    },
                ],
                "members": [
                    {"name": "John", "age": 20, "role": "cleaner"},
                    {"name": "Jane", "age": 30, "role": "manager"},
                ],
            },
            {
                "project_name": "fixing cars",
                "project_start": "2020-01-03",
                "project_end": "2020-01-06",
                "project_status": "started",
                "organization_name": "Car Shop",
                "organization_address": "Main Street 3",
                "organization_city": "Cartown",
                "tasks": [
                    {
                        "task_name": "task6",
                        "task_assignee": "John",
                        "task_status": "todo",
                    },
                ],
                "members": [
                    {"name": "John", "age": 20, "role": "mechanic"},
                    {"name": "Jane", "age": 30, "role": "manager"},
                    {"name": "Jack", "age": 40, "role": "manager"},
                ],
            },
        ],
    }


@pytest.fixture
def root_table_mapping() -> TableMap:
    """Return root table mapping for import testing."""
    return TableMap(
        table="searches",
        id_type="hash",
        id_attr="search_term",
        map={
            "resultCount": "result_count",
            "search": "search_term",
            "results": TableMap(
                table="projects",
                id_type="uuid",
                map={
                    "project_name": "name",
                    "project_start": "start",
                    "project_end": "end",
                    "project_status": "status",
                    "tasks": TableMap(
                        table="tasks",
                        map={
                            "task_name": "name",
                            "task_assignee": TableMap(
                                table="users",
                                map="name",
                                match_by_attr="name",
                            ),
                            "task_status": "status",
                        },
                        id_type="uuid",
                    ),
                    "members": TableMap(
                        table="users",
                        map={"name", "age"},
                        join_table_name="memberships",
                        join_table_map={
                            "role": "role",
                        },
                    ),
                },
                ext_maps=[
                    TableMap(
                        table="organizations",
                        map={
                            "organization_name": "name",
                            "organization_address": "address",
                            "organization_city": "city",
                        },
                        link_type="n-1",
                        link_attr="organization",
                        id_type="attr",
                        id_attr="name",
                    ),
                ],
            ),
        },
    )


def test_import_db_from_tree(nested_db_dict: dict, root_table_mapping: TableMap):
    """Test importing nested data dict to database."""
    db = tree_to_db(nested_db_dict, root_table_mapping, False)
    assert isinstance(db, DB)
    assert set(db.keys()) == {
        "searches",
        "projects",
        "tasks",
        "users",
        "organizations",
        "memberships",
        "searches_projects",
        "projects_tasks",
    }
    assert len(db["searches"].df) == 1
    assert len(db["projects"].df) == 3
    assert len(db["tasks"].df) == 6
    assert len(db["users"].df) == 3
    assert len(db["organizations"].df) == 3
    assert len(db["memberships"].df) > 0
    assert len(db["searches_projects"].df) > 0
    assert len(db["projects_tasks"].df) > 0
    assert db.relations == {
        ("searches_projects", "results"): ("projects", "_id"),
        ("searches_projects", "results_of"): ("searches", "_id"),
        ("projects_tasks", "tasks"): ("tasks", "_id"),
        ("projects_tasks", "tasks_of"): ("projects", "_id"),
        ("memberships", "members"): ("users", "_id"),
        ("memberships", "members_of"): ("projects", "_id"),
        ("projects", "organization"): ("organizations", "_id"),
    }
    assert db.join_tables == {"searches_projects", "projects_tasks", "memberships"}
