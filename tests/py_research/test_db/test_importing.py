"""Test db importing module."""

import json
from datetime import date
from typing import Any, Literal

import pytest

from py_research.db import Attr, Record, Rel, RootMap


class SearchResult(Record):
    """Link search to a result."""

    search: Rel[Any, "Search", "Search"]
    result: Rel[Any, "Project", "Project"]


class Search(Record[str, str]):
    """Defined search against the API."""

    search_term: Attr[Any, str] = Attr(primary_key=True)
    result_count: Attr[Any, int]
    results: Rel[Any, list["Project"], "Project"] = Rel(via=SearchResult)


class Task(Record):
    """Link search to a result."""

    name: Attr[Any, str]
    project: Rel[Any, "Project", "Project"]
    assignee: Rel[Any, "User", "User"]
    status: Attr[Any, Literal["todo", "done"]]


class User(Record):
    """A generic user."""

    name: Attr[Any, str]
    tasks = Rel(via=Task.assignee)


class Project(Record):
    """A generic project record."""

    name: Attr[Any, str]
    start: Attr[Any, date]
    end: Attr[Any, date]
    status: Attr[Any, Literal["planned", "started", "done"]]
    tasks = Rel(via=Task.project)


@pytest.fixture
def nested_db_dict() -> dict:
    """Return nested data dict for import testing."""
    with open("./nested_data.json") as f:
        return json.load(f)


@pytest.fixture
def root_table_mapping() -> RootMap:
    """Return root table mapping for import testing."""
    return RootMap(
        table="searches",
        id_type="hash",
        id_attr="search_term",
        push={
            "resultCount": "result_count",
            "search": "search_term",
            "results": XMap(
                table="projects",
                id_type="uuid",
                push={
                    "project_name": "name",
                    "project_start": "start",
                    "project_end": "end",
                    "project_status": "status",
                    "tasks": XMap(
                        table="tasks",
                        push={
                            "task_name": "name",
                            "task_assignee": XMap(
                                table="users",
                                push="name",
                                match_by_attrs="name",
                            ),
                            "task_status": "status",
                        },
                        id_type="uuid",
                    ),
                    "members": XMap(
                        table="users",
                        push={"name", "age"},
                        join_table_name="memberships",
                        join_table_map={
                            "role": "role",
                        },
                    ),
                },
                ext_maps=[
                    XMap(
                        table="organizations",
                        push={
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


def test_import_db_from_tree(nested_db_dict: dict, root_table_mapping: XMap):
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
    assert len(db["searches"].df()) == 1
    assert len(db["projects"].df()) == 3
    assert len(db["tasks"].df()) == 6
    assert len(db["users"].df()) == 3
    assert len(db["organizations"].df()) == 3
    assert len(db["memberships"].df()) > 0
    assert len(db["searches_projects"].df()) > 0
    assert len(db["projects_tasks"].df()) > 0
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
