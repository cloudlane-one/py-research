"""Utils for creating presistent references to Python objects."""

import inspect
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from typing import Any, Generic, TypeVar

import requests
from packaging.version import InvalidVersion, Version

from .deps import VersionStrategy, semver_range_to_spec, version_to_range
from .dist import (
    get_distributions,
    get_module_distribution,
    get_module_repo,
    get_project_urls,
    get_py_inventory,
)

T = TypeVar("T")
T2 = TypeVar("T2")


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

    docs_url: str | None = None
    """Deep-link to this object's section within the package's API reference."""

    @staticmethod
    def reference(  # noqa: C901
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
                        else (
                            getattr(obj, "__version__")
                            if hasattr(obj, "__version__")
                            else "*"
                        )
                    )
                )
                if version is not None
                else None
            )
            if obj_version_exact is not None:
                object_version = version_to_range(
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
            package_version = version_to_range(
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

        docs_urls = get_project_urls(dist, "Documentation")
        docs_url = docs_urls[0] if len(docs_urls) > 0 else None

        obj_inv = get_py_inventory(docs_url) if docs_url is not None else None
        if obj_inv is not None:
            obj_inv_key = f"{module.__name__}.{qualname}"
            if obj_inv_key in obj_inv:
                docs_url = obj_inv[obj_inv_key][2]
        else:
            api_urls = get_project_urls(dist, "API Reference")
            base_api_url = api_urls[0] if len(api_urls) > 0 else None
            if base_api_url is not None:
                possible_docs_url = (
                    f"{base_api_url.rstrip('/')}/{module.__name__}.html#{qualname}"
                )
                test_request = requests.get(possible_docs_url)
                if test_request.status_code != 404:
                    docs_url = possible_docs_url

        return PyObjectRef(
            object_type=type(obj),
            package=dist.name,
            module=module.__name__,
            object=qualname,
            object_version=object_version,
            package_version=package_version,
            repo=(url if url else "https://pypi.org"),
            repo_revision=repo.head.commit.hexsha if repo is not None else None,
            docs_url=docs_url,
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
        dist = get_distributions().get(self.package)
        if dist is None:
            raise ImportError(
                f"Package '{self.package}' "
                f"with version '{self.package_version or '*'}' is not installed."
            )
        elif self.package_version is not None and Version(
            dist.version
        ) not in semver_range_to_spec(self.package_version):
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
            not in semver_range_to_spec(self.object_version)
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
