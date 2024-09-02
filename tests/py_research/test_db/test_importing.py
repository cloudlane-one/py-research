"""Test db importing module."""

from __future__ import annotations

import json
from datetime import date
from typing import Literal, reveal_type

import pytest
from py_research.db import Attr, DataBase, Record, RecUUID, Rel, RootMap, prop
from py_research.db.importing import RelMap, SubMap, tree_to_db


class SearchResult(Record):
    """Link search to a result."""

    search: Rel[Search] = prop(primary_key="fk")
    result: Rel[Project] = prop(primary_key="fk")
    score: Attr[float]


class Search(Record[str]):
    """Defined search against the API."""

    term: Attr[str] = prop(primary_key=True)
    result_count: Attr[int]
    results: Rel[dict[int, Project]] = prop(
        link_via=SearchResult.result,
        order_by={SearchResult.score: -1},
        collection=lambda s: dict(enumerate(s)),
    )


class Task(RecUUID):
    """Link search to a result."""

    name: Attr[str]
    project: Rel[Project]
    assignee: Rel[User]
    status: Attr[Literal["todo", "done"]]


class User(RecUUID):
    """A generic user."""

    name: Attr[str]
    age: Attr[int]
    tasks: Rel[list[Task]] = prop(link_from=Task.assignee)

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(RecUUID):
    """Link user to a project."""

    member: Rel[User] = prop(primary_key="fk")
    project: Rel[Project] = prop(primary_key="fk")
    role: Attr[str] = Attr(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Attr[int] = prop(primary_key=True)
    name: Attr[str]
    start: Attr[date]
    end: Attr[date]
    status: Attr[Literal["planned", "started", "done"]]
    org: Rel[Organization]
    tasks: Rel[list[Task]] = prop(link_from=Task.project)
    members: Rel[list[User]] = prop(link_via=Membership)


class Organization(RecUUID):
    """A generic organization record."""

    name: Attr[str]
    address: Attr[str]
    city: Attr[str]
    projects: Rel[list[Project]] = prop(link_from=Project.org)


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

    reveal_type(s := db[Search])
    reveal_type(s[Search.result_count])
    reveal_type(sr := s[Search.results])
    reveal_type(s[Search.results.r.org])
    reveal_type(sr0 := sr[("first", 0)])
    reveal_type(sr[[("first", 0), ("second", 1)]])
    reveal_type(db[Project][0:3])
    reveal_type(db[Project][0])
    reveal_type(s[Search.term == "first"])
    reveal_type(sr[Project.name == "Project 1"])
    reveal_type(sr0.load())

    s[Search.result_count] = 10
    s[Search.term == "first"][Search.result_count] = 5
    s[Search.results] = {("first", 0): db[Project][0].load()}
