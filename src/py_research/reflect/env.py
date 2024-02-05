"""Utils for reflecting the Python and system environment."""

import platform
from pathlib import Path
from typing import Any

from IPython.core.getipython import get_ipython

from .dist import get_file_repo, get_module_repo
from .runtime import get_calling_module


def env_info() -> dict[str, Any]:
    """Return information about the current Python environment."""
    mod = get_calling_module()

    repo_dict = {}
    req_dict = {}

    repo = (
        get_module_repo(mod)
        if mod is not None and mod.__name__ is not None
        else get_file_repo(Path.cwd()) if is_in_jupyter() else None
    )

    if repo is not None:
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = None

        repo_info = {
            "url": repo.remote().url,
            "branch": branch,
        }

        current_commit = repo.head.commit
        latest_tag = repo.tags[-1] if len(repo.tags) > 0 else None
        if latest_tag is not None and current_commit == latest_tag.commit:
            repo_info["tag"] = latest_tag.name
        else:
            repo_info["commit"] = current_commit.hexsha

        if repo.is_dirty():
            repo_info["dirty"] = True

        repo_dict = {"repo": {k: v for k, v in repo_info.items() if v is not None}}

    return {
        **repo_dict,
        **req_dict,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.version(),
    }


def is_in_jupyter() -> bool:
    """Return whether the runtime is a Jupyter environment.

    Returns:
        True if the runtime is a Jupyter environment, False otherwise.
    """
    shell = get_ipython()
    if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
        return True  # Jupyter notebook or qtconsole

    return False
