"""Easy to use relational database."""

from .conflicts import DataConflictError as DataConflictError
from .importing import tree_to_db as tree_to_db
from .old_db import DB as DB
from .old_db import DBSchema as DBSchema
from .old_db import Table as Table
