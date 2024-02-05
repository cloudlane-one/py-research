"""Use pip to find outdated dependencies with major version upgrades."""

import py_research
from py_research.reflect.deps import get_outdated_deps
from py_research.reflect.dist import get_module_distribution

if __name__ == "__main__":
    dist = get_module_distribution(py_research)
    assert dist is not None, "Could not find distribution for py_research"

    outdated = get_outdated_deps(dist)
    assert len(outdated) == 0, f"Found outdated packages: {outdated}"
