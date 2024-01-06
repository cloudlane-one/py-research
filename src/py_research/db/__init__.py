"""Easy to use relational database."""

from .base import DB as DB
from .base import Table as Table
from .conflicts import DataConflictError as DataConflictError
from .importing import tree_to_db as tree_to_db
