"""Utils for Python code reflection."""

import inspect
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, reduce
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Generic, Literal, TypeAlias, TypeVar
from urllib.parse import urlparse

import importlib_metadata as meta
import numpy as np
import requests
from git import Repo
from packaging.requirements import Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version

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


VersionStrategy: TypeAlias = Literal["exact", "minor", "major"]


def _version_to_range(version: Version, strategy: VersionStrategy = "major") -> str:
    return (
        f"=={version}"
        if strategy == "exact"
        else f"~{version}"
        if strategy == "minor"
        else f"^{version}"
    )


def _semver_range_to_spec(semver_range: str) -> SpecifierSet:
    op = semver_range[0] if semver_range[0] in "~^>=<" else None
    version = Version(semver_range.lstrip("^~>=<"))
    return SpecifierSet(
        (f">={version.public}" f",<{version.major + 1}")
        if op == "^"
        else (f">={version.public},<{version.major}" f".{version.minor + 1}")
        if op == "~"
        else f"{op}{version.public}"
        if op in list(">=<")
        else f"=={version.public}"
    )


@dataclass
class PyObjectRef(Generic[T]):
    """Reference to a static Python object."""

    object_type: type[T]
    """Type of the object."""

    package: str
    """Name of the package the object belongs to."""

    module: str
    """Name of the module the object belongs to."""

    object: str
    """Name of the object."""

    package_version: str | None = None
    """Semver version range of the package or object."""

    object_version: str | None = None
    """Semver version range of the object (independent of package-version)."""

    repo: str | None = None
    """URL of the package's repository."""

    repo_revision: str | None = None
    """Revision of the repo."""

    @staticmethod
    def reference(
        obj: T2,
        version: str | None = None,
        pkg_version_strategy: VersionStrategy = "major",
        obj_version_strategy: VersionStrategy = "major",
    ) -> "PyObjectRef[T2]":
        """Create a reference to given object."""
        object_version = None
        try:
            obj_version_exact = (
                (
                    Version(
                        version
                        if version is not None
                        else getattr(obj, "__version__")
                        if hasattr(obj, "__version__")
                        else "*"
                    )
                )
                if version is not None
                else None
            )
            if obj_version_exact is not None:
                object_version = _version_to_range(
                    obj_version_exact, obj_version_strategy
                )
        except InvalidVersion:
            pass

        qualname = getattr(obj, "__qualname__")
        if qualname is None:
            raise ValueError("Object must have fully qualified name (`__qualname__`)")

        module = inspect.getmodule(obj)
        if module is None:
            raise ValueError("Object must be associated with a module.")

        dist = get_module_distribution(module)
        if dist is None:
            raise ValueError("Object must be associated with a package.")

        package_version = None
        try:
            package_version_exact = Version(dist.version)
            package_version = _version_to_range(
                package_version_exact, pkg_version_strategy
            )
        except InvalidVersion:
            pass

        url: str | None = (
            dict(dist.origin.__dict__).get("url") if dist.origin is not None else None
        )

        repo = None
        if url is not None and str(url).startswith("file://"):
            repo = get_module_repo(module)
            if repo is not None:
                url = repo.remote().url

        return PyObjectRef(
            object_type=type(obj),
            package=dist.name,
            module=module.__name__,
            object=qualname,
            object_version=object_version,
            package_version=package_version,
            repo=(url if url else "https://pypi.org"),
            repo_revision=repo.head.commit.hexsha if repo is not None else None,
        )

    @staticmethod
    def from_url(url: str, obj_type: type[T]) -> "PyObjectRef[T]":
        """Create a reference from given URL."""
        url = url.lstrip("py+")  # Remove protocol
        repo, url = url.split("?")  # Split repo and package
        package, url = url.split("#")  # Split package and object
        module_spec, object_spec = url.split(":")  # Split module and object

        object_version = None
        if "@" in object_spec:
            object_spec, object_version = object_spec.split("@")

        package_version = None
        if "=" in package:
            package, package_version = package.split("=")

        return PyObjectRef(
            object_type=obj_type,
            package=package,
            module=module_spec,
            object=object_spec,
            package_version=package_version,
            object_version=object_version,
            repo=repo,
        )

    def to_url(self) -> str:
        """Convert object reference to URL."""
        return (
            f"py+{self.repo}?{self.package}"
            + (f"={self.package_version}" if self.package_version is not None else "")
            + f"#{self.module}:{self.object}"
            + (f"@{self.object_version}" if self.object_version is not None else "")
        )

    def resolve(self) -> T:
        """Resolve object reference."""
        dist = _get_distributions().get(self.package)
        if dist is None:
            raise ImportError(
                f"Package '{self.package}' "
                f"with version '{self.package_version or '*'}' is not installed."
            )
        elif self.package_version is not None and Version(
            dist.version
        ) not in _semver_range_to_spec(self.package_version):
            raise ImportError(
                f"Please install correct version of package '{self.package}': "
                f"'{self.package_version}'"
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
        if not isinstance(obj, self.object_type):
            raise TypeError(
                f"Object `{'.'.join([self.module, self.object])}` "
                f"must have type `{self.object_type}`"
            )

        if (
            self.object_version is not None
            and hasattr(obj, "__version__")
            and (given_version := Version(getattr(obj, "__version__")))
            not in _semver_range_to_spec(self.object_version)
        ):
            raise ValueError(
                f"Requested version {self.object_version} is not compatible with "
                + f"existing version {given_version}."
            )

        return obj


def stref(obj: Any) -> str:
    """Get string representation of given object reference."""
    obj_ref = PyObjectRef.reference(obj)
    return f"{obj_ref.module}.{obj_ref.object}"


def get_dist_requirements(dist: meta.Distribution) -> list[Requirement] | None:
    """Get a list of declared packages via pdm."""
    return (
        [Requirement(dep) for dep in dist.requires]
        if dist.requires is not None
        else None
    )


def get_versions_on_pypi(package: meta.Distribution | str) -> set[Version]:
    """Get all available versions of given distribution."""
    if isinstance(package, meta.Distribution):
        if package.origin is not None:
            return set()
        package = package.name

    url = f"https://pypi.org/pypi/{package}/json"

    response = requests.get(url)
    if response.status_code == 404:
        return set()

    data = response.json()

    versions = set()
    for v in data["releases"].keys():
        try:
            versions.add(Version(v))
        except InvalidVersion:
            pass

    return versions


def version_diff(v1: Version, v2: Version) -> Version | None:
    """Get the difference between two versions (v1 - v2).

    if v1 is smaller than v2, returns None.
    """
    if v1 < v2:
        return None

    v1_arr = np.array((v1.major, v1.minor, v1.micro))
    v2_arr = np.array((v2.major, v2.minor, v2.micro))

    diff = v1_arr - v2_arr

    if diff[0] > 0:
        diff[1:] = v1_arr[1:]
    elif diff[1] > 0:
        diff[2] = v1_arr[2]

    return Version(".".join(str(v) for v in diff))


def get_outdated_deps(
    dist: meta.Distribution | ModuleType,
    allowed_diff: Specifier = Specifier("<=1.1.1"),
) -> dict[str, tuple[Version, Version]]:
    """Get a list of outdated dependencies of a distribution.

    Args:
        dist:
            Distribution to inspect.
            Can also be supplied as a module within the distribution in question.
        allowed_diff: Allowed difference between current and latest version.

    Returns:
        Dictionary of outdated package names with current and latest version.
    """
    if isinstance(dist, ModuleType):
        mod_dist = get_module_distribution(dist)
        if mod_dist is None:
            raise ValueError("Supplied module is not part of a distribution.")
        dist = mod_dist

    deps = get_dist_requirements(dist)

    if deps is None:
        return {}

    outdated = {}
    for dep in deps:
        try:
            dep_dist = meta.distribution(dep.name)
        except meta.PackageNotFoundError:
            dep_dist = dep.name

        versions = get_versions_on_pypi(dep_dist)
        versions = {
            v
            for v in versions
            if not v.is_prerelease and not v.is_postrelease and not v.is_devrelease
        }

        matching_req = set(dep.specifier.filter(versions))
        newest_matching = max(matching_req)

        newer = versions - matching_req
        if len(newer) == 0:
            continue

        newest = max(newer)
        diff = version_diff(newest, newest_matching)
        if diff is not None and diff not in allowed_diff:
            outdated[dep.name] = (newest_matching, newest)

    return outdated


def env_info() -> dict[str, Any]:
    """Return information about the current Python environment."""
    mod_name = get_calling_module_name()

    repo_dict = {}
    req_dict = {}
    if mod_name is not None:
        mod = import_module(mod_name)
        repo = get_module_repo(mod)

        if repo is not None:
            try:
                branch = repo.active_branch.name
            except TypeError:
                branch = None

            repo_info = {
                "url": repo.remote().url,
                "revision": repo.head.commit.hexsha,
                "branch": branch,
                "tag": repo.tags[0].name if len(repo.tags) > 0 else None,
                **({"dirty": True} if repo.is_dirty() else {}),
            }
            repo_dict = {"repo": {k: v for k, v in repo_info.items() if v is not None}}

            repo_root = repo.working_tree_dir
            if repo_root is not None:
                root_path = Path(repo_root)
                matching_files = [
                    f
                    for f in ["pyproject.toml", "requirements.txt"]
                    if (root_path / f).is_file()
                ]

                if len(matching_files) > 0:
                    req_dict = {"requirements": matching_files[0]}

    if len(req_dict) == 0:
        mods = sys.modules.copy()
        dists = {name: get_module_distribution(mod) for name, mod in mods.items()}
        dists = {name: dist for name, dist in dists.items() if dist is not None}
        reqs = {name: f"{dist.name}=={dist.version}" for name, dist in dists.items()}
        req_dict = {"requirements": reqs}

    return {
        **repo_dict,
        **req_dict,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.version(),
    }
