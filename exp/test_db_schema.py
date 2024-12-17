"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import Attr, AttrSet, BackRef, Record, RecUUID, Ref, Rel, RelSet


class SearchResult(Record):
    """Link search to a result."""

    search: Ref[Search] = Ref(primary_key=True)
    result: Ref[Project] = Ref(primary_key=True)
    score: Attr[float]


class Search(Record[str]):
    """Defined search against the API."""

    term: Attr[str] = Attr(primary_key=True)
    result_count: Attr[int]
    results: RelSet[Project, SearchResult] = RelSet(default=True)


Assignment = Rel["User", "Task"]


class Task(RecUUID):
    """Link search to a result."""

    name: Attr[str]
    project: Ref[Project]
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

    member: Ref[User] = Ref(primary_key=True)
    project: Ref[Project] = Ref(primary_key=True)
    role: Attr[str] = Attr(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Attr[int] = Attr(primary_key=True)
    name: Attr[str]
    start: Attr[date]
    end: Attr[date]
    status: Attr[Literal["planned", "started", "done"]]
    org: Ref[Organization]
    tasks: BackRef[Task] = BackRef(to=Task.project)
    members: RelSet[User, Membership]


class Organization(RecUUID):
    """A generic organization record."""

    name: Attr[str]
    address: Attr[str]
    city: Attr[str]
    projects: BackRef[Project] = BackRef(to=Project.org, default=True)
    countries: AttrSet[str]
