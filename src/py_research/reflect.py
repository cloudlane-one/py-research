"""Utils for Python code reflection."""

import inspect
import platform
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, reduce
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Generic, TypeVar
from urllib.parse import urlparse

import importlib_metadata as meta
from git import Repo

T = TypeVar("T")
T2 = TypeVar("T2")


def _get_calling_frame(offset=0):
    stack = inspect.stack()
    if len(stack) < offset + 3:
        raise RuntimeError("No caller!")
    return stack[offset + 2]


def get_calling_module_name() -> str | None:
    """Return the name of the module calling the current function."""
    mod = inspect.getmodule(_get_calling_frame(offset=1).frame)
    return mod.__name__ if mod is not None else None


def get_full_args_dict(
    func: Callable, args: Sequence, kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return dict of all args + kwargs.

    Args:
        func: Function to inspect.
        args: Positional arguments given to the function.
        kwargs: Keyword arguments given to the function.

    Returns:
        Dictionary of all args + kwargs.
    """
    argspec = inspect.getfullargspec(func)

    arg_defaults = argspec.defaults or []
    kwdefaults = dict(zip(argspec.args[-len(arg_defaults) :], arg_defaults))

    posargs = dict(zip(argspec.args[: len(args)], args))

    return {**kwdefaults, **posargs, **(kwargs or {})}


def get_return_type(func: Callable) -> type | None:
    """Get the return type annotation of given function, if any."""
    sig = inspect.signature(func)
    return (
        sig.return_annotation
        if sig.return_annotation != inspect.Signature.empty
        else None
    )


def get_all_subclasses(cls: type[T]) -> set[type[T]]:
    """Return all subclasses of given class."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


@cache
def _get_distributions() -> dict[str, meta.Distribution]:
    return {d.metadata["Name"]: d for d in meta.distributions()}


def _get_module_file(module: ModuleType) -> Path | None:
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
    mod_file = _get_module_file(module)
    if mod_file is None:
        return None

    dists = {
        Path(dist.locate_file(f"{name}")): dist
        for name, dist in _get_distributions().items()
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


def get_all_module_dependencies(
    module: ModuleType,
    _ext_deps: set[str] | None = None,
    _int_deps: set[ModuleType] | None = None,
) -> tuple[set[str], set[ModuleType]]:
    """Return all (indirect) dependency modules of given module.

    Args:
        module: Module to inspect.

    Returns:
        Tuple of external and internal dependencies.
    """
    if _ext_deps is None:
        _ext_deps = set()
    if _int_deps is None:
        _int_deps = set()

    deps = [
        dep
        for _, m in inspect.getmembers(module)
        if (dep := inspect.getmodule(m)) is not None
    ]
    ext_deps_map = {
        dep: dist.metadata["Name"]
        for dep in deps
        if (dist := get_module_distribution(dep)) is not None
    }
    new_ext_deps = set(ext_deps_map.values()) - _ext_deps

    these_int_deps = set(deps) - set(ext_deps_map.keys())
    new_int_deps = these_int_deps - _int_deps

    if len(new_ext_deps) > 0 or len(new_int_deps) > 0:
        _ext_deps |= new_ext_deps
        _int_deps |= new_int_deps

        sub_ext_deps, sub_int_deps = zip(
            *[
                get_all_module_dependencies(d, _ext_deps, _int_deps)
                for d in new_int_deps
            ]
        )

        return set.union(*sub_ext_deps), set.union(*sub_int_deps)
    else:
        return _ext_deps, _int_deps


def get_module_repo(module: ModuleType) -> Repo | None:
    """Get the Git repository of given module, if any."""
    mod_file = _get_module_file(module)
    if mod_file is None:
        return None

    return Repo(mod_file, search_parent_directories=True)


def _check_version(lower_bound: str, version: str) -> bool:
    """Check if a version is compatible with a lower bound."""
    lb = tuple(map(int, lower_bound.split(".")))
    v = tuple(map(int, version.split(".")))
    return v >= lb and v[0] == lb[0]


@dataclass
class PyObjectRef(Generic[T]):
    """Reference to a static Python object."""

    type: type[T]
    """Type of the object."""

    package: str
    """Name of the package the object belongs to."""

    module: str
    """Name of the module the object belongs to."""

    object: str
    """Name of the object."""

    version: str | None = None
    """Version of the package or object."""

    repo: str | None = None
    """URL of the package's repository."""

    revision: str | None = None
    """Revision of the repo."""

    @staticmethod
    def _follows_semver(version: str) -> bool:
        return re.match(r"^\d+\.\d+\.\d+$", version) is not None

    @staticmethod
    def reference(obj: T2, version: str | None = None) -> "PyObjectRef[T2]":
        """Create a reference to given object."""
        qualname = getattr(obj, "__qualname__")
        if qualname is None:
            raise ValueError("Object must have fully qualified name (`__qualname__`)")

        module = inspect.getmodule(obj)
        if module is None:
            raise ValueError("Object must be associated with a module.")

        dist = get_module_distribution(module)
        if dist is None:
            raise ValueError("Object must be associated with a package.")

        url: str | None = (
            dict(dist.origin.__dict__).get("url") if dist.origin is not None else None
        )

        repo = None
        if url is not None and str(url).startswith("file://"):
            repo = get_module_repo(module)
            if repo is not None:
                url = repo.remote().url

        return PyObjectRef(
            type=type(obj),
            package=dist.name,
            module=module.__name__,
            object=qualname,
            version=(
                version
                if version
                else getattr(obj, "__version__")
                if hasattr(obj, "__version__")
                else dist.version
                if PyObjectRef._follows_semver(dist.version)
                else None
            ),
            repo=(url if url else "https://pypi.org"),
            revision=repo.head.commit.hexsha if repo is not None else None,
        )

    def resolve(self) -> T:
        """Resolve object reference."""
        dist = _get_distributions().get(self.package)
        if dist is None or (
            self.version is not None and not _check_version(self.version, dist.version)
        ):
            raise ImportError(
                f"Please install package '{self.package}' with version '{self.version}'"
                f"from '{self.repo}' to resolve this object reference."
            )

        try:
            module = import_module(self.module)
        except ModuleNotFoundError:
            raise ImportError(
                f"Cannot import module '{self.module}' from package '{self.package}'."
            )

        url: str | None = (
            dict(dist.origin.__dict__).get("url") if dist.origin is not None else None
        )
        if url is not None:
            if str(url).startswith("file://"):
                repo = get_module_repo(module)
                if repo is not None:
                    url = repo.remote().url
            if url != self.repo:
                raise ImportError(
                    f"URL mismatch: Package '{self.package} should be from "
                    f"'{self.repo}' but is from '{url}'."
                )

        obj = reduce(getattr, self.object.split("."), module)
        if not isinstance(obj, self.type):
            raise TypeError(
                f"Object `{'.'.join([self.package, self.module, self.object])}` "
                f"must have type `{self.type}`"
            )

        if (
            self.version is not None
            and hasattr(obj, "__version__")
            and not _check_version(
                self.version,
                given_version := getattr(obj, "__version__"),
            )
        ):
            raise ValueError(
                f"Requested version {self.version} is not compatible with "
                + f"existing version {given_version}."
            )

        return obj


def stref(obj: Any) -> str:
    """Get string representation of given object reference."""
    obj_ref = PyObjectRef.reference(obj)
    return f"{obj_ref.module}.{obj_ref.object}"
