"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import (
    Array,
    BackLink,
    Link,
    Record,
    RecUUID,
    Relation,
    Schema,
    Table,
    Value,
)


class TestSchema(Schema):
    """Test schema."""


class SearchResult(Relation["Search", "Project"]):
    """Link search to a result."""

    score: Value[float]


class Search(Record[str], TestSchema):
    """Defined search against the API."""

    term: Value[str] = Value(primary_key=True)
    result_count: Value[int]
    results: Table[Project, SearchResult] = Table(default=True)


type Assignment = Relation["User", "Task"]


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
    countries: Array[str, int]
