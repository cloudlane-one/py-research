"""Utils for Python code reflection."""

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, reduce
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Generic, TypeVar

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
        if not hasattr(module, "__file__") or module.__file__ is None
        else None
    )


def get_module_distribution(module: ModuleType) -> meta.Distribution | None:
    """Get the distribution package of given module, if any."""
    mod_file = _get_module_file(module)
    if mod_file is None:
        return None

    dists = {
        Path(dist.locate_file(f"{name}")): dist
        for name, dist in _get_distributions().items()
    }

    mod_dists = [
        dist for dist_path, dist in dists.items() if mod_file.is_relative_to(dist_path)
    ]

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

    return Repo(mod_file)


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

    version: str = "1.0.0"
    """Custom version of the object."""

    url: str = "https://pypi.org"
    """URL of the package's index."""

    @staticmethod
    def _get_pkg_url(module: ModuleType, dist: meta.Distribution) -> str | None:
        url: str | None = (
            dict(dist.origin).get("url") if dist.origin is not None else None
        )

        if url is not None and url.startswith("file://"):
            repo = get_module_repo(module)
            if repo is not None:
                url = repo.remote().url

        return url

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
            raise ValueError("Object's module ")

        url = PyObjectRef._get_pkg_url(module, dist)

        return PyObjectRef(
            type(obj),
            dist.name,
            module.__name__,
            qualname,
            **(dict(version=version) if version else {}),
            **(dict(url=url) if url else {}),
        )

    def resolve(self) -> T:
        """Resolve object reference."""
        dist = _get_distributions().get(self.package)
        if dist is None:
            raise ImportError(
                f"Please install package '{self.package}' "
                f"from '{self.url}' to resolve this object reference."
            )

        module = import_module(".".join([self.package, self.module]))
        url = PyObjectRef._get_pkg_url(module, dist)
        if url is not None and url != self.url:
            raise ImportError(
                f"URL mismatch: Package '{self.package} should be from "
                f"'{self.url}' but is from '{url}'."
            )

        obj = reduce(getattr, self.object.split("."), module)
        if not isinstance(obj, self.type):
            raise TypeError(
                f"Object `{'.'.join([self.package, self.module, self.object])}` "
                f"must have type `{self.type}`"
            )

        return obj
