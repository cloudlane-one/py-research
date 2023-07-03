"""Helper functions for handling files and directories."""

from pathlib import Path


def ensure_dir_exists(path: Path | str):
    """Make sure the given path exists and is a directory.

    Create dir if necessary. Then return the path unchanged.
    """
    path = Path(path).absolute()
    p = Path(path.parts[0])
    for d in path.parts[1:]:
        p = p / d
        if not p.is_dir():
            if not p.exists():
                p.mkdir()
            else:
                raise NotADirectoryError()

    return path


def ensure_parent_dir_exists(path: Path):
    """Make sure the given path's parent exists and is a directory.

    Create dir if necessary. Then return the path unchanged.
    """
    return ensure_dir_exists(path.parent)
