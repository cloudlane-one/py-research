"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import (
    Array,
    BackLink,
    Entity,
    Link,
    Record,
    Relation,
    Schema,
    Table,
    Var,
)


class TestSchema(Schema):
    """Test schema."""


class SearchResult(Relation["Search", "Project"]):
    """Link search to a result."""

    score: Var[float]


class Search(Record[str], TestSchema):
    """Defined search against the API."""

    term: Var[str] = Var(primary_key=True)
    result_count: Var[int]
    results: Table[Project, SearchResult] = Table(default=True)


type Assignment = Relation["User", "Task"]


class Task(Entity):
    """Link search to a result."""

    name: Var[str]
    project: Link[Project]
    assignees: Table[User, Assignment]
    status: Var[Literal["todo", "done"]]


class User(Entity):
    """A generic user."""

    name: Var[str]
    age: Var[int]
    tasks: Table[Task, Assignment]

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(Relation["User", "Project"]):
    """Link user to a project."""

    role: Var[str] = Var(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Var[int] = Var(primary_key=True)
    name: Var[str]
    start: Var[date]
    end: Var[date]
    status: Var[Literal["planned", "started", "done"]]
    org: Link[Organization]
    tasks: BackLink[Task] = BackLink(to=Task.project)
    members: Table[User, Membership]


class Organization(Entity):
    """A generic organization record."""

    name: Var[str]
    address: Var[str]
    city: Var[str]
    projects: BackLink[Project] = BackLink(to=Project.org, default=True)
    countries: Array[str, int]
