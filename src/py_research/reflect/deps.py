"""Utils for reflecting the Python dependencies."""

import inspect
from types import ModuleType
from typing import Literal, TypeAlias

import importlib_metadata as meta
import numpy as np
import requests
from packaging.requirements import Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .dist import get_module_distribution


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


VersionStrategy: TypeAlias = Literal["exact", "minor", "major"]


def version_to_range(version: Version, strategy: VersionStrategy = "major") -> str:
    """Convert exact version to version range."""
    return (
        f"=={version}"
        if strategy == "exact"
        else f"~{version}" if strategy == "minor" else f"^{version}"
    )


def semver_range_to_spec(semver_range: str) -> SpecifierSet:
    """Convert semver range to Python version specifier."""
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
