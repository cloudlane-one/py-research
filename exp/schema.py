"""Test DB schema."""

from __future__ import annotations

from datetime import date
from typing import Literal

from py_research.db import (
    Array,
    Attr,
    AutoIdx,
    Edge,
    Entity,
    Idx,
    Key,
    Link,
    Record,
    Rel,
    Schema,
)


class TestSchema(Schema):
    """Test schema."""


class SearchResult(Edge["Search", "Project"]):
    """Link search to a result."""

    score: Attr[float]


class Search(Record[str], TestSchema):
    """Defined search against the API."""

    term: Attr[str] = Attr()
    result_count: Attr[int]
    results: Rel[Project, SearchResult]

    _pk = Key(term)


type Assignment = Edge["User", "Task"]


class Task(Entity):
    """Link search to a result."""

    name: Attr[str]
    project: Link[Project]
    assignees: Rel[User, Assignment]
    status: Attr[Literal["todo", "done"]]


class User(Entity):
    """A generic user."""

    name: Attr[str]
    age: Attr[int]
    tasks: Rel[Task, Assignment]

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Membership(Edge["User", "Project"]):
    """Link user to a project."""

    role: Attr[str] = Attr(default="member")


class Project(Record[int]):
    """A generic project record."""

    number: Attr[int] = Attr()
    name: Attr[str]
    start: Attr[date]
    end: Attr[date]
    status: Attr[Literal["planned", "started", "done"]]
    org: Link[Organization]
    tasks: Link[Task, AutoIdx] = Link(on=Task.project)
    members: Rel[User, Membership]

    _pk = Key(number)

    @property
    def all_done(self) -> bool:
        """Return True if all tasks are done."""
        return all(task.status == "done" for task in self.tasks)


class Organization(Entity):
    """A generic organization record."""

    name: Attr[str]
    address: Attr[str]
    city: Attr[str]
    projects: Link[Project, AutoIdx] = Link(on=Project.org)
    countries: Array[str, Idx[int]]
