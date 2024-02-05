"""Test reflect module."""

import py_research
from py_research.reflect.deps import get_outdated_deps
from py_research.reflect.dist import get_module_distribution, get_project_urls
from py_research.reflect.env import env_info, is_in_jupyter
from py_research.reflect.ref import PyObjectRef


class StaticObject:
    """Static object for test."""


def test_py_obj_ref():
    """Test Python object referencing."""
    ref = PyObjectRef.reference(StaticObject)
    assert ref.package == "py-research"

    obj = ref.resolve()
    assert obj is StaticObject

    assert ref.docs_url is not None

    dist = get_module_distribution(py_research)
    assert dist is not None
    docs_urls = get_project_urls(dist, "Documentation")
    assert len(docs_urls) > 0
    assert ref.docs_url.startswith(docs_urls[0])


def test_get_outdated_deps():
    """Test get_outdated_deps function."""
    outdated_deps = get_outdated_deps(py_research)
    assert isinstance(outdated_deps, dict)


def test_env_info():
    """Test env_info function."""
    info = env_info()
    assert isinstance(info, dict)
    assert "repo" in info
    assert "python_version" in info


def test_is_in_jupyter():
    """Test is_in_jupyter function."""
    assert is_in_jupyter() is False
