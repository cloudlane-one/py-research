"""Test db importing module."""

from __future__ import annotations

import json
from datetime import date
from typing import Literal, reveal_type

import pytest

from py_research.db import (
    Array,
    BackLink,
    DataBase,
    DataSource,
    Link,
    RecMap,
    Record,
    RecUUID,
    RefMap,
    Relation,
    SubMap,
    Table,
    Value,
)


class SearchResult(Record):
    """Link search to a result."""

    search: Link[Search] = Link(primary_key=True)
    result: Link[Project] = Link(primary_key=True)
    score: Value[float]


class Search(Record[str]):
    """Defined search against the API."""

    term: Value[str] = Value(primary_key=True)
    result_count: Value[int]
    results: Table[Project, SearchResult] = Table(default=True)


Assignment = Relation["User", "Task"]


class Task(RecUUID):
    """Link search to a result."""

    name: Value[str]
    project: Link[Project]
    assignees: Table[User, Assignment]
    status: Value[Literal["todo", "done"]]


class User(RecUUID):
    """A generic user."""

    name: Value[str]
    age: Value[int]
    tasks: Table[Task, Assignment]

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(Relation["User", "Project"]):
    """Link user to a project."""

    member: Link[User] = Link(primary_key=True)
    project: Link[Project] = Link(primary_key=True)
    role: Value[str] = Value(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Value[int] = Value(primary_key=True)
    name: Value[str]
    start: Value[date]
    end: Value[date]
    status: Value[Literal["planned", "started", "done"]]
    org: Link[Organization]
    tasks: BackLink[Task] = BackLink(link=Task.project)
    members: Table[User, Membership]


class Organization(RecUUID):
    """A generic organization record."""

    name: Value[str]
    address: Value[str]
    city: Value[str]
    projects: BackLink[Project] = BackLink(link=Project.org, default=True)
    countries: Array[str]


@pytest.fixture
def nested_db_dict() -> dict:
    """Return nested data dict for import testing."""
    with open("tests/py_research/test_db/nested_data.json") as f:
        return json.load(f)


@pytest.fixture
def data_source() -> DataSource:
    """Return root table mapping for import testing."""
    return DataSource(
        target=Search,
        push={
            "resultCount": Search.result_count,
            "search": Search.term,
            "results": RefMap(
                ref=Search.results,
                push={
                    "project_name": Project.name,
                    "project_start": Project.start,
                    "project_end": Project.end,
                    "project_status": Project.status,
                    "tasks": RefMap(
                        ref=Project.tasks,
                        push={
                            "task_name": Task.name,
                            "task_assignees": RefMap(
                                ref=Task.assignees,
                                push=User.name,
                                match_by=User.name,
                            ),
                            "task_status": Task.status,
                        },
                    ),
                    "members": RefMap(
                        ref=Project.members,
                        push={User.name, User.age},
                        rel=RecMap(
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


def test_import_db_from_tree(nested_db_dict: dict, data_source: DataSource):
    """Test importing nested data dict to database."""
    db = DataBase()
    rec = data_source.load(nested_db_dict, db)

    assert isinstance(rec, Search)
    assert isinstance(db, Table)

    assert len(db[Search]) == 1
    assert len(db[Project]) == 3
    assert len(db[Task]) == 6
    assert len(db[User]) == 3
    assert len(db[Organization]) == 3
    assert len(db[Membership]) > 0
    assert len(db[SearchResult]) > 0

    reveal_type(s := db[Search])
    reveal_type(s[Search.result_count])
    reveal_type(sr := s[Search.results])
    reveal_type(s[Search.results.x.org])
    reveal_type(sr[("first", 0)])
    reveal_type(sr[[("first", 0), ("second", 1)]])
    reveal_type(db[Project][0:3])
    reveal_type(db[Project][0])
    reveal_type(s[Search.term == "first"])
    reveal_type(sr[Project.name == "Project 1"])
    reveal_type(list(iter(db[Project])))

    s[Search.result_count] @= 10
    s[Search.term == "first"][Search.result_count] @= 5
    s[Search.results] |= {("first", 0): list(iter(db[Project]))[0]}
