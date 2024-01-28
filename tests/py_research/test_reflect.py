"""Test reflect module."""

import py_research
from py_research.reflect import env_info, get_outdated_deps


def test_get_outdated_deps():
    """Test get_outdated_deps function."""
    outdated_deps = get_outdated_deps(py_research)
    assert isinstance(outdated_deps, dict)


def test_env_info():
    """Test env_info function."""
    info = env_info()
    assert isinstance(info, dict)
    assert "repo" in info
    assert "requirements" in info
    assert "python_version" in info
