"""Types and functions for data conflict management."""

from typing import Any, Literal, TypeAlias

DataConflictPolicy: TypeAlias = Literal["raise", "ignore", "override"]

DataConflicts: TypeAlias = dict[tuple[str, str, str], tuple[Any, Any]]


class DataConflictError(ValueError):
    """Irreconsilable conflicts during import / merging of data."""

    def __init__(self, conflicts: DataConflicts) -> None:  # noqa: D107
        self.conflicts = conflicts
        super().__init__(
            f"Conflicting values: {conflicts}"
            if len(conflicts) < 5
            else f"{len(conflicts)} in table-columns "
            + str(set((k[0], k[2]) for k in conflicts.keys()))
        )
