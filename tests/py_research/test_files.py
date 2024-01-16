"""Test files module."""

from pathlib import Path
from tempfile import gettempdir

import pytest
from py_research.files import ensure_dir_exists


def test_ensure_dir_exists():
    """Test ensure_dir_exists."""
    path = Path(gettempdir()) / "test_dir"
    try:
        assert ensure_dir_exists(path) == path.absolute()
        assert path.is_dir()
    finally:
        path.rmdir()  # Clean up

    # Test with a string
    path_str = f"{gettempdir()}/test_dir"
    try:
        assert ensure_dir_exists(path_str) == Path(path_str).absolute()
        assert Path(path_str).is_dir()
    finally:
        Path(path_str).rmdir()  # Clean up


def test_ensure_dir_exists_not_a_dir():
    """Test ensure_dir_exists with a file."""
    # Create a file
    path = Path(gettempdir()) / "test_file"
    path.touch()
    try:
        with pytest.raises(NotADirectoryError):
            ensure_dir_exists(path)
    finally:
        path.unlink()  # Clean up
