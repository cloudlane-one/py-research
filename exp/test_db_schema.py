"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import (
    Array,
    BackLink,
    Record,
    RecUUID,
    Ref,
    Relation,
    RelTable,
    Value,
)


class SearchResult(Record):
    """Link search to a result."""

    search: Ref[Search] = Ref(primary_key=True)
    result: Ref[Project] = Ref(primary_key=True)
    score: Value[float]


class Search(Record[str]):
    """Defined search against the API."""

    term: Value[str] = Value(primary_key=True)
    result_count: Value[int]
    results: RelTable[Project, SearchResult] = RelTable(default=True)


Assignment = Relation["User", "Task"]


class Task(RecUUID):
    """Link search to a result."""

    name: Value[str]
    project: Ref[Project]
    assignees: RelTable[User, Assignment]
    status: Value[Literal["todo", "done"]]


class User(RecUUID):
    """A generic user."""

    name: Value[str]
    age: Value[int]
    tasks: RelTable[Task, Assignment]

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(RecUUID):
    """Link user to a project."""

    member: Ref[User] = Ref(primary_key=True)
    project: Ref[Project] = Ref(primary_key=True)
    role: Value[str] = Value(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Value[int] = Value(primary_key=True)
    name: Value[str]
    start: Value[date]
    end: Value[date]
    status: Value[Literal["planned", "started", "done"]]
    org: Ref[Organization]
    tasks: BackLink[Task] = BackLink(link=Task.project)
    members: RelTable[User, Membership]


class Organization(RecUUID):
    """A generic organization record."""

    name: Value[str]
    address: Value[str]
    city: Value[str]
    projects: BackLink[Project] = BackLink(link=Project.org, default=True)
    countries: Array[str]
