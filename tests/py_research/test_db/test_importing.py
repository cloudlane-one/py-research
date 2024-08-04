"""Test db importing module."""

from __future__ import annotations

import json
from datetime import date
from typing import Literal

import pytest
from py_research.db import Attr, DataBase, Record, Rel, RootMap
from py_research.db.importing import RelMap, SubMap, tree_to_db


class SearchResult(Record):
    """Link search to a result."""

    search: Rel[Search]
    result: Rel[Project]


class Search(Record[str, str]):
    """Defined search against the API."""

    term: Attr[str] = Attr(primary_key=True)
    result_count: Attr[int]
    results: Rel[list[Project]] = Rel(via=SearchResult)


class Task(Record):
    """Link search to a result."""

    name: Attr[str]
    project: Rel[Project]
    assignee: Rel[User]
    status: Attr[Literal["todo", "done"]]


class User(Record):
    """A generic user."""

    name: Attr[str]
    age: Attr[int]
    tasks = Rel(via=Task.assignee)


class Membership(Record):
    """Link user to a project."""

    member: Rel[User]
    project: Rel[Project]
    role: Attr[str]


class Project(Record):
    """A generic project record."""

    name: Attr[str]
    start: Attr[date]
    end: Attr[date]
    status: Attr[Literal["planned", "started", "done"]]
    tasks = Rel(via=Task.project)
    members = Rel(via=Membership)
    org: Rel[Organization]


class Organization(Record):
    """A generic organization record."""

    name: Attr[str]
    address: Attr[str]
    city: Attr[str]
    projects = Rel(via=Project.org)


@pytest.fixture
def nested_db_dict() -> dict:
    """Return nested data dict for import testing."""
    with open("tests/py_research/test_db/nested_data.json") as f:
        return json.load(f)


@pytest.fixture
def root_table_mapping() -> RootMap:
    """Return root table mapping for import testing."""
    return RootMap(
        rec=Search,
        push={
            "resultCount": Search.result_count,
            "search": Search.term,
            "results": RelMap(
                rel=Search.results,
                push={
                    "project_name": Project.name,
                    "project_start": Project.start,
                    "project_end": Project.end,
                    "project_status": Project.status,
                    "tasks": RelMap(
                        rel=Project.tasks,
                        push={
                            "task_name": Task.name,
                            "task_assignee": RelMap(
                                rel=Task.assignee,
                                push=User.name,
                                match=User.name,
                            ),
                            "task_status": Task.status,
                        },
                    ),
                    "members": RelMap(
                        rel=Project.members,
                        push={User.name, User.age},
                        link=RootMap(
                            rec=Membership,
                            push={
                                Membership.role,
                            },
                        ),
                    ),
                },
                pull={
                    Project.org: SubMap(
                        push={
                            "organization_name": Organization.name,
                            "organization_address": Organization.address,
                            "organization_city": Organization.city,
                        },
                    ),
                },
            ),
        },
    )


def test_import_db_from_tree(nested_db_dict: dict, root_table_mapping: RootMap):
    """Test importing nested data dict to database."""
    db, conflicts = tree_to_db(nested_db_dict, root_table_mapping)

    assert isinstance(db, DataBase)
    assert len(conflicts) == 0

    assert len(db[Search]) == 1
    assert len(db[Project]) == 3
    assert len(db[Task]) == 6
    assert len(db[User]) == 3
    assert len(db[Organization]) == 3
    assert len(db[Membership]) > 0
    assert len(db[SearchResult]) > 0
