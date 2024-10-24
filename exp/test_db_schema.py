"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import Attr, BackRel, Link, Record, RecUUID, Rel, RelSet


class SearchResult(Record):
    """Link search to a result."""

    search: Rel[Search] = Rel(primary_key=True)
    result: Rel[Project] = Rel(primary_key=True)
    score: Attr[float]


class Search(Record[str]):
    """Defined search against the API."""

    term: Attr[str] = Attr(primary_key=True)
    result_count: Attr[int]
    results: RelSet[Project, SearchResult]


Assignment = Link["User", "Task"]


class Task(RecUUID):
    """Link search to a result."""

    name: Attr[str]
    project: Rel[Project]
    assignees: RelSet[User, Assignment]
    status: Attr[Literal["todo", "done"]]


class User(RecUUID):
    """A generic user."""

    name: Attr[str]
    age: Attr[int]
    tasks: RelSet[Task, Assignment]

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(RecUUID):
    """Link user to a project."""

    member: Rel[User] = Rel(primary_key=True)
    project: Rel[Project] = Rel(primary_key=True)
    role: Attr[str] = Attr(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Attr[int] = Attr(primary_key=True)
    name: Attr[str]
    start: Attr[date]
    end: Attr[date]
    status: Attr[Literal["planned", "started", "done"]]
    org: Rel[Organization]
    tasks: RelSet[Task] = BackRel(to=Task.project)
    members: RelSet[User, Membership]


class Organization(RecUUID):
    """A generic organization record."""

    name: Attr[str]
    address: Attr[str]
    city: Attr[str]
    projects: RelSet[Project] = BackRel(to=Project.org)
