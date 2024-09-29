"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import Attr, Link, Record, RecUUID, Rel, prop


class SearchResult(Record):
    """Link search to a result."""

    search: Rel[Search] = prop(primary_key="fk")
    result: Rel[Project] = prop(primary_key="fk")
    score: Attr[float]


class Search(Record[str]):
    """Defined search against the API."""

    term: Attr[str] = prop(primary_key=True)
    result_count: Attr[int]
    results: Rel[dict[int, Project], SearchResult] = prop(
        order_by={SearchResult.score: -1},
        collection=lambda s: dict(enumerate(s)),
    )


Assignment = Link["User", "Task"]


class Task(RecUUID):
    """Link search to a result."""

    name: Attr[str]
    project: Rel[Project]
    assignees: Rel[list[User], Assignment]
    status: Attr[Literal["todo", "done"]]


class User(RecUUID):
    """A generic user."""

    name: Attr[str]
    age: Attr[int]
    tasks: Rel[list[Task], Assignment]

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(RecUUID):
    """Link user to a project."""

    member: Rel[User] = prop(primary_key="fk")
    project: Rel[Project] = prop(primary_key="fk")
    role: Attr[str] = prop(default="member")


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
