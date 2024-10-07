"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import Col, Link, Record, RecUUID, Rel, RelSet, prop


class SearchResult(Record):
    """Link search to a result."""

    search: Rel[Search] = prop(primary_key="fk")
    result: Rel[Project] = prop(primary_key="fk")
    score: Col[float]


class Search(Record[str]):
    """Defined search against the API."""

    term: Col[str] = prop(primary_key=True)
    result_count: Col[int]
    results: RelSet[Project, SearchResult, int] = prop(
        order_by={SearchResult.score: -1},
    )


Assignment = Link["User", "Task"]


class Task(RecUUID):
    """Link search to a result."""

    name: Col[str]
    project: Rel[Project]
    assignees: RelSet[User, Assignment]
    status: Col[Literal["todo", "done"]]


class User(RecUUID):
    """A generic user."""

    name: Col[str]
    age: Col[int]
    tasks: RelSet[Task, Assignment]

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(RecUUID):
    """Link user to a project."""

    member: Rel[User] = prop(primary_key="fk")
    project: Rel[Project] = prop(primary_key="fk")
    role: Col[str] = prop(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Col[int] = prop(primary_key=True)
    name: Col[str]
    start: Col[date]
    end: Col[date]
    status: Col[Literal["planned", "started", "done"]]
    org: Rel[Organization]
    tasks: RelSet[Task] = prop(link_from=Task.project)
    members: RelSet[User] = prop(link_via=Membership)


class Organization(RecUUID):
    """A generic organization record."""

    name: Col[str]
    address: Col[str]
    city: Col[str]
    projects: RelSet[Project] = prop(link_from=Project.org)
