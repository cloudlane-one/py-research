"""Utils for creating presistent references to Python objects."""

import inspect
import sysconfig
from dataclasses import dataclass
from functools import cache, reduce
from importlib import import_module
from pathlib import Path
from types import ModuleType
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


@cache
def std_modules() -> list[str]:
    """Get list of standard library modules."""
    ret_list = []
    std_lib = sysconfig.get_path("stdlib")

    for root, _, files in Path(std_lib).walk():
        for file in files:
            p = root / file
            if p.name != "__init__.py" and p.suffix == ".py":
                ret_list.append(
                    p.relative_to(std_lib).with_suffix("").as_posix().replace("/", ".")
                )

    return ret_list


@dataclass
class PyObjectRef(Generic[T]):
    """Reference to a static Python object."""

    object_type: type[T]
    """Type of the object."""

    package: str
    """Name of the package the object belongs to."""

    module: str
    """Name of the module the object belongs to."""

    object: str | None = None
    """Name of the object."""

    package_version: str | None = None
    """Semver version range of the package or object."""

    object_version: str | None = None
    """Semver version range of the object (independent of package-version)."""

    repo: str | None = None
    """URL of the package's repository."""

    repo_revision: str | None = None
    """Revision of the repo."""

    module_path: Path | None = None
    """Path to the module file within the repo."""

    module_dirty: bool = False
    """Whether the referenced module is in a dirty state (untracked changes)."""

    docs_url: str | None = None
    """Deep-link to this object's section within the package's API reference."""

    @property
    def fqn(self) -> str:
        """Fully qualified name of the object."""
        return f"{self.module}.{self.object}"

    @cache
    @staticmethod
    def reference(  # noqa: C901
        obj: T2,
        version: str | None = None,
        pkg_version_strategy: VersionStrategy = "major",
        obj_version_strategy: VersionStrategy = "major",
        fetch_obj_inv: bool = True,
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

        qualname = getattr(obj, "__qualname__", None)
        if qualname is None and not isinstance(obj, ModuleType):
            raise ValueError("Object must have fully qualified name (`__qualname__`)")

        module = obj if isinstance(obj, ModuleType) else inspect.getmodule(obj)
        if module is None:
            raise ValueError("Object must be associated with a module.")

        dist = get_module_distribution(module)
        if dist is not None:
            dist_name = dist.name
            dist_version = dist.version
            dist_origin = dist.origin

            docs_urls = get_project_urls(dist, "Documentation")
            docs_url = docs_urls[0] if len(docs_urls) > 0 else None
            api_urls = get_project_urls(dist, "API Reference")
            base_api_url = api_urls[0] if len(api_urls) > 0 else None
        elif module.__name__ in std_modules() or module.__name__ == "builtins":
            dist_name = "python"
            dist_version = sysconfig.get_python_version()
            dist_origin = None

            docs_url = f"https://docs.python.org/{dist_version}/library"
            base_api_url = docs_url
        else:
            raise ValueError("Object must be associated with a package.")

        package_version = None
        try:
            package_version_exact = Version(dist_version)
            package_version = version_to_range(
                package_version_exact, pkg_version_strategy
            )
        except InvalidVersion:
            pass

        url: str | None = (
            dict(dist_origin.__dict__).get("url") if dist_origin is not None else None
        )

        repo = None
        module_path = None
        if url is not None and str(url).startswith("file://"):
            repo = get_module_repo(module)
            if repo is not None:
                url = repo.remote().url
                if module.__file__ is not None:
                    module_path = Path(module.__file__).relative_to(repo.working_dir)

        if not isinstance(obj, ModuleType) and fetch_obj_inv:
            obj_inv = get_py_inventory(docs_url) if docs_url is not None else None
            if obj_inv is not None:
                obj_inv_key = f"{module.__name__}.{qualname}"
                if obj_inv_key in obj_inv:
                    docs_url = obj_inv[obj_inv_key][2]
            elif base_api_url is not None:
                possible_docs_url = (
                    f"{base_api_url.rstrip('/')}/{module.__name__}.html#{qualname}"
                )
                test_request = requests.get(possible_docs_url)
                if test_request.status_code != 404:
                    docs_url = possible_docs_url

        return PyObjectRef(
            object_type=type(obj),
            package=dist_name,
            module=module.__name__,
            object=qualname,
            object_version=object_version,
            package_version=package_version,
            repo=(url if url else "https://pypi.org"),
            repo_revision=repo.head.commit.hexsha if repo is not None else None,
            module_path=module_path,
            module_dirty=(
                repo.is_dirty(path=module_path)
                if repo is not None and module_path is not None
                else False
            ),
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

        if self.object is None:
            assert self.object_type is ModuleType
            return module  # type: ignore

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
