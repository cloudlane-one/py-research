"""Utils for creating presistent references to Python objects."""

import inspect
import platform
from dataclasses import dataclass
from functools import cached_property, reduce
from importlib import import_module
from types import ModuleType
from typing import Any, Generic, Literal, TypeVar

import importlib_metadata as meta
import numpy as np
import requests
from packaging.requirements import Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from yarl import URL

from .dist import (
    get_module_distribution,
    get_module_repo,
    get_project_urls,
    get_py_inventory,
)


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
        and dist.metadata is not None
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


def get_dist_requirements(dist: meta.Distribution) -> list[Requirement] | None:
    """Get a list of declared packages via pyproject metadata.

    Args:
        dist: Distribution to inspect.

    Returns:
        List of declared requirements, or ``None`` if not specified.
    """
    return (
        [Requirement(dep) for dep in dist.requires]
        if dist.requires is not None
        else None
    )


def get_versions_on_pypi(package: meta.Distribution | str) -> set[Version]:
    """Get all available versions of given distribution.

    Args:
        package: Distribution or package name to inspect.

    Returns:
        Set of available versions.
    """
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

    Args:
        v1: Newer version.
        v2: Older version.

    Returns:
        Difference between ``v1`` and ``v2`` as Version, or ``None`` if ``v1 < v2``
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


type VersionStrategy = Literal["exact", "minor", "major"]
"""Strategy for version range specification."""


def version_to_range(version: Version, strategy: VersionStrategy = "major") -> str:
    """Convert exact version to version range.

    Args:
        version: Exact version to convert.
        strategy: Version strategy to use.

    Returns:
        Version range string.
    """
    return (
        f"=={version}"
        if strategy == "exact"
        else f"~{version}" if strategy == "minor" else f"^{version}"
    )


def semver_range_to_spec(semver_range: str) -> SpecifierSet:
    """Convert semver range to Python version specifier.

    Args:
        semver_range: Semver range string.

    Returns:
        Corresponding Python version specifier set.
    """
    op = semver_range[0] if semver_range[0] in "~^>=<" else None
    version = Version(semver_range.lstrip("^~>=<"))
    return SpecifierSet(
        (f">={version.public}" f",<{version.major + 1}")
        if op == "^"
        else (
            (f">={version.public},<{version.major}" f".{version.minor + 1}")
            if op == "~"
            else f"{op}{version.public}" if op in list(">=<") else f"=={version.public}"
        )
    )


T = TypeVar("T")
T2 = TypeVar("T2")


@dataclass
class PyObjectRef(Generic[T]):
    """Reference to a static Python object."""

    object_type: type[T]
    """Type of the object."""

    qualname: str
    """Name of the object."""

    module_name: str
    """Name of the module the object belongs to."""

    repo_url: str | None = None
    """URL of the Git repo or index where the package is hosted. If None, the object is a built-in key."""

    version: str | None = None
    """
    (Semver) version specifier (exact or range) of the package.
    This maps to branch or tag names in Git repos.
    None implies the newest available version.
    """

    py_version: str | None = None
    """Python version specifier (exact or range) for which the object is valid."""

    _object: T | None = None
    """Cached resolved object."""

    _module: ModuleType | None = None
    """Cached module object."""

    _dist: meta.Distribution | None = None
    """Cached distribution object."""

    @staticmethod
    def reference(  # noqa: C901
        obj: T2,
        version: str | None = None,
        version_strategy: VersionStrategy = "major",
    ) -> "PyObjectRef[T2]":
        """Create a reference to given object.

        Args:
            obj: Object to create reference for.
            version: Exact or range version specifier for the object's package/repo.
            version_strategy: Strategy to use for version range specification.

        Returns:
            Reference to the given object.
        """
        qualname = getattr(obj, "__qualname__")
        if qualname is None:
            raise ValueError("Object must have fully qualified name (`__qualname__`)")

        module = inspect.getmodule(obj)
        if module is None:
            raise ValueError("Object must be associated with a module.")

        dist = get_module_distribution(module)
        package = dist.name if dist is not None else None

        git_repo = None
        repo = None

        if dist is not None and dist.origin is not None:
            # Fetch the origin URL of the distribution, if any.
            repo = dict(dist.origin.__dict__).get("url")

        if dist is None or str(repo).startswith("file://"):
            # Not part of distribution or local file origin URL
            # -> assume first-party code commited to Git repo.
            git_repo = get_module_repo(module)
            if git_repo is not None:
                repo = git_repo.remote().url

        if repo is None:
            if dist is not None:
                # Code is part of distribution, but no explicit origin URL
                # -> assume PyPI indexed package
                repo = f"https://pypi.org/simple/{package}"
            elif module.__name__ != "builtins":
                # No distribution, no Git repo and not built-in module
                raise ValueError(
                    "Non-builtin object must be associated with a repository."
                )

        version_exact = None
        semver_exact = None

        if version is None:
            if dist is not None:
                version_exact = dist.version
                try:
                    semver_exact = Version(version_exact)
                except InvalidVersion:
                    pass
            elif git_repo is not None:
                current_tags = [
                    tag for tag in git_repo.tags if tag.commit == git_repo.head.commit
                ]
                if len(current_tags) > 0:
                    for tag in current_tags:
                        version_exact = tag.name
                        try:
                            semver_exact = Version(version_exact)
                            break
                        except InvalidVersion:
                            continue
                elif not git_repo.head.is_detached:
                    version_exact = git_repo.active_branch.name
                    try:
                        semver_exact = Version(version_exact)
                    except InvalidVersion:
                        pass

        if semver_exact is not None:
            version = version_to_range(semver_exact, version_strategy)
        elif version_strategy == "exact" and version_exact is not None:
            version = f"=={version_exact}"

        return PyObjectRef(
            object_type=type(obj),
            qualname=qualname,
            module_name=module.__name__,
            repo_url=repo,
            version=version,
            py_version=version_to_range(
                Version(f"{platform.python_version()}"), "minor"
            ),
            _module=module,
            _dist=dist,
            _object=obj,
        )

    @staticmethod
    def from_url(url_text: str, obj_type: type[T]) -> "PyObjectRef[T]":
        """Create a reference from given URL.

        Args:
            url_text: URL text to parse.
            obj_type: Type of the referenced object.

        Returns:
            Reference to the object.
        """
        url = URL(url_text)

        repo_url = None
        if url.scheme.startswith("py+"):
            repo_url = str(
                url.with_scheme(url.scheme[3:])
                .without_query_params()
                .with_fragment(None)
            )

        module, object = url.fragment.split(":")  # Split module and object

        return PyObjectRef(
            object_type=obj_type,
            repo_url=repo_url,
            version=url.query.get("version"),
            py_version=url.query.get("python"),
            module_name=module,
            qualname=object,
        )

    @property
    def module(self) -> ModuleType:
        """Import and return the module object."""
        if self._module is None:
            try:
                self._module = import_module(self.module_name)
            except ModuleNotFoundError:
                raise ImportError(
                    f"Cannot import module '{self.module_name}' from repo '{self.repo_url}'."
                )

        return self._module

    @property
    def dist(self) -> meta.Distribution:
        """Return the distribution object, if any."""
        if self._dist is None:
            module = self.module
            dist = get_module_distribution(module)

            if dist is None:
                raise ImportError(
                    f"Repo '{self.repo_url}' "
                    + (
                        f", version '{self.version}' "
                        if self.version is not None
                        else ""
                    )
                    + "is not available in this environment."
                )
            elif self.version is not None and Version(
                dist.version
            ) not in semver_range_to_spec(self.version):
                raise ImportError(
                    f"Please install correct version of repo '{self.repo_url}': "
                    f"'{self.version}'"
                )

            self._dist = dist

        return self._dist

    @property
    def docs_url(self) -> URL | None:
        """Deep-link to this object's section within the package's API reference."""
        if self.dist is None:
            return None

        if self.module_name is None:
            return None

        docs_url = None

        docs_urls = get_project_urls(self.dist, "Documentation")

        if len(docs_urls) > 0:
            docs_url = docs_urls[0]

            obj_inv = get_py_inventory(docs_url) if docs_url is not None else None
            if obj_inv is not None:
                obj_inv_key = f"{self.module_name}.{self.qualname}"
                if obj_inv_key in obj_inv:
                    docs_url = str(obj_inv[obj_inv_key][2])

        if docs_url is None:
            api_urls = get_project_urls(self.dist, "API Reference")
            base_api_url = api_urls[0] if len(api_urls) > 0 else None

            if base_api_url is not None:
                possible_docs_url = f"{base_api_url.rstrip('/')}/{self.module_name}.html#{self.qualname}"
                test_request = requests.get(possible_docs_url)
                if test_request.status_code != 404:
                    docs_url = possible_docs_url

        return URL(docs_url) if docs_url is not None else None

    @property
    def object(self) -> T:
        """Resolved object."""
        if self._object is None:
            url: str | None = (
                dict(self.dist.origin.__dict__).get("url")
                if self.dist.origin is not None
                else None
            )
            if url is not None:
                if str(url).startswith("file://"):
                    repo = get_module_repo(self.module)
                    if repo is not None:
                        url = repo.remote().url
                if url != self.repo_url:
                    raise ImportError(
                        f"Expected module '{self.module_name}' to be from '{self.repo_url}' but is from '{url}'."
                    )

            obj = reduce(getattr, self.qualname.split("."), self.module)
            if not isinstance(obj, self.object_type):
                raise TypeError(
                    f"Object `{'.'.join([self.module_name, self.qualname])}` "
                    f"must have type `{self.object_type}`"
                )

            self._object = obj

        return self._object

    @cached_property
    def url(self) -> URL:
        """URL representation of this reference."""
        repo_url = URL(self.repo_url) if self.repo_url is not None else None
        return (
            (
                URL().with_scheme("py")
                if repo_url is None
                else URL()
                .with_scheme(f"py+{repo_url.scheme}")
                .with_host(repo_url.host if repo_url.host is not None else "")
                .with_port(repo_url.port)
                .with_path(repo_url.path)
            )
            .with_query(
                {
                    **({"version": self.version} if self.version is not None else {}),
                    **(
                        {"python": self.py_version}
                        if self.py_version is not None
                        else {}
                    ),
                }
            )
            .with_fragment(f"{self.module_name}:{self.qualname}")
        )


def stref(obj: Any) -> str:
    """Get string representation of given object reference.

    Args:
        obj: Object to create reference for.

    Returns:
        String representation of the object reference.
    """
    obj_ref = PyObjectRef.reference(obj)
    return f"{obj_ref.module_name}.{obj_ref.qualname}"
