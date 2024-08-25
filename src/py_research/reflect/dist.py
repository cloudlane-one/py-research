"""Utils for reflecting the Python distributions."""

import platform
import posixpath
from functools import cache, reduce
from io import BytesIO
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

import importlib_metadata as meta
import requests
import sphinx.util.inventory as inv
from git import GitError, Repo


@cache
def get_distributions() -> dict[str, meta.Distribution]:
    """Get all installed Python package distributions."""
    return {d.metadata["Name"]: d for d in meta.distributions()}


def get_module_file(module: ModuleType) -> Path | None:
    """Get the file of given module, if any."""
    return (
        Path(str(module.__file__))
        if hasattr(module, "__file__") and module.__file__ is not None
        else None
    )


def _file_url_to_path(file_url: str):
    # Get the operating system name
    os_name = platform.system()

    # Parse the file URL
    parsed_url = urlparse(file_url)

    # Extract the path component from the parsed URL
    file_path = parsed_url.path

    # Convert the path to a Windows-compatible path
    # On Windows, you may need to remove the leading '/' from the path
    if os_name == "Windows" and file_path.startswith("/"):
        file_path = file_path[1:]

    # Create a Path object
    path_object = Path(file_path)

    return path_object


def get_module_distribution(module: ModuleType) -> meta.Distribution | None:
    """Get the distribution package of given module, if any."""
    mod_file = get_module_file(module)
    if mod_file is None:
        return None

    dists = {
        Path(str(dist.locate_file(f"{name}"))): dist
        for name, dist in get_distributions().items()
    }

    mod_dists = []
    for dist_path, dist in dists.items():
        if mod_file.is_relative_to(dist_path):
            mod_dists.append(dist)
        elif dist.origin is not None and "url" in dist.origin.__dict__:  # type: ignore
            url = str(dist.origin.url)
            if url.startswith("file://"):
                path = _file_url_to_path(url)
                if mod_file.is_relative_to(path):
                    mod_dists.append(dist)

    return mod_dists[0] if len(mod_dists) > 0 else None


def get_file_repo(path: Path) -> Repo | None:
    """Get the Git repository of given file, if any."""
    try:
        return Repo(path, search_parent_directories=True)
    except GitError:
        return None


def get_module_repo(module: ModuleType) -> Repo | None:
    """Get the Git repository of given module, if any."""
    mod_file = get_module_file(module)
    if mod_file is None:
        return None

    return get_file_repo(mod_file)


def get_project_urls(dist: meta.Distribution, key: str) -> list[str]:
    """Get the documentation URL of given distribution, if any."""
    if dist.metadata is None:
        return []

    proj_urls = dist.metadata.get_all("Project-URL")
    if proj_urls is None:
        return []

    proj_urls_split = [str(url).split(", ") for url in proj_urls]
    urls = [v for k, v in proj_urls_split if k == key]
    if len(urls) == 0:
        return []

    return urls


@cache
def get_py_inventory(docs_url: str) -> dict[str, tuple[str, str, str, str]] | None:
    """Return object inventory for given documentation URL."""
    inv_url = f"{docs_url.rstrip('/')}/objects.inv"

    res = requests.get(inv_url, allow_redirects=True)

    if res.status_code != 200:
        return None

    inv_dict = inv.InventoryFile.load(BytesIO(res.content), docs_url, posixpath.join)
    py_inv_dict = reduce(
        lambda a, b: {**a, **b}, [v for k, v in inv_dict.items() if k.startswith("py:")]
    )

    return py_inv_dict
