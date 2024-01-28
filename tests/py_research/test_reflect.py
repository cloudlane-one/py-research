"""Test reflect module."""

import py_research
from py_research.reflect import PyObjectRef, env_info, get_outdated_deps


class StaticObject:
    """Static object for test."""


def test_py_obj_ref():
    """Test Python object referencing."""
    ref = PyObjectRef.reference(StaticObject)
    assert ref.package == "py-research"

    obj = ref.resolve()
    assert obj is StaticObject


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
