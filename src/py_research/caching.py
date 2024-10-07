"""On-the-fly, file-based caching of function return values."""

import datetime
import json
import pickle
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast, overload

import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup, Tag

from py_research.data import gen_id
from py_research.files import ensure_dir_exists
from py_research.reflect.runtime import (
    get_calling_module,
    get_full_args_dict,
    get_return_type,
)
from py_research.reflect.types import is_subtype
from py_research.telemetry import get_logger

log = get_logger()

default_root_path = Path.cwd() / ".cache"


P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class FileCache:
    """Local, directory-based cache for storing function results."""

    path: Path
    """Root directory for storing cached results."""

    max_cache_time: datetime.timedelta = datetime.timedelta(days=7)
    """After how long to invalidate cached objects and recompute."""

    def __post_init__(self):  # noqa: D105
        now = datetime.datetime.now()
        self.__earliest_date = now - self.max_cache_time
        self.__now_str = now.strftime("%Y-%m-%d")

    @staticmethod
    def __get_date_from_filename(f: Path):
        return datetime.datetime.strptime(f.name.split(".")[0], "%Y-%m-%d")

    def __filter_outdated(self, f: Path):
        if self.__get_date_from_filename(f) > self.__earliest_date:
            return True
        else:
            f.unlink()
            return False

    def _get_id(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        use_raw_arg: bool,
        id_arg_subset: list[int] | list[str] | None,
        id_callback: Callable[P, dict[str, Any] | None] | None,
    ) -> str:
        id_args = None

        if id_callback is not None:
            id_args = id_callback(*args, **kwargs)

        if id_args is None:
            id_args = get_full_args_dict(func, args, kwargs)
            if id_arg_subset is not None:
                id_args = {k: v for k, v in id_args.items() if k in id_arg_subset}

        return gen_id(
            id_args if len(id_args) > 1 else list(id_args.values())[0],
            raw_str=use_raw_arg,
        )

    def _get_cached_result(
        self, func: Callable, path: Path, id_value: str
    ) -> Any | None:
        filename_pattern = f"[0-9]*-[0-9]*-[0-9]*.{id_value}.*"

        all_cached = [
            f
            for f in path.iterdir()
            if f.match(filename_pattern) and self.__filter_outdated(f)
        ]

        if len(all_cached) == 0:
            return None

        log.debug(f"💾 Taking cached result for: '{id_value}'")

        last_cached = sorted(all_cached, key=self.__get_date_from_filename)[-1]
        extension = last_cached.name.split(".")[-1]
        return_type = get_return_type(func) or type

        cached_result = None
        if is_subtype(return_type, str) and extension == "txt":
            with open(
                last_cached,
                encoding="utf-8",
            ) as f:
                cached_result = f.read()
        elif is_subtype(return_type, dict | list) and extension in (
            "yaml",
            "yml",
        ):
            cached_result = yaml.load(
                open(last_cached, encoding="utf-8"), Loader=yaml.CLoader
            )
        elif is_subtype(return_type, dict | list) and extension == "json":
            cached_result = json.load(open(last_cached, encoding="utf-8"))
        elif is_subtype(return_type, pd.DataFrame | pd.Series) and extension == "xlsx":
            cached_result = pd.read_excel(last_cached, header=0, index_col=0)
        elif is_subtype(return_type, pd.DataFrame | pd.Series) and extension == "csv":
            cached_result = pd.read_csv(last_cached, header=0, index_col=0)
        elif is_subtype(return_type, np.ndarray) and extension == "npy":
            cached_result = np.load(last_cached)
        elif is_subtype(return_type, BeautifulSoup | Tag) and extension == "html":
            cached_result = BeautifulSoup(open(last_cached))
        elif extension == "pkl":
            cached_result = pickle.load(open(last_cached, mode="rb"))

        return cached_result

    def _cache_result(
        self, result: Any, path: Path, id_value: str, use_json: bool
    ) -> None:
        match (result):
            case str():
                with open(
                    path / f"{self.__now_str}.{id_value}.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(result)
            case dict() | list():
                if use_json:
                    json.dump(
                        result,
                        open(
                            path / f"{self.__now_str}.{id_value}.json",
                            "w",
                            encoding="utf-8",
                        ),
                        indent=2,
                    )
                else:
                    yaml.dump(
                        result,
                        open(
                            path / f"{self.__now_str}.{id_value}.yaml",
                            "w",
                            encoding="utf-8",
                        ),
                        allow_unicode=True,
                    )
            case pd.DataFrame():
                result.to_excel(path / f"{self.__now_str}.{id_value}.xlsx", index=True)
            case np.ndarray():
                np.save(path / f"{self.__now_str}.{id_value}.npy", result)
            case BeautifulSoup() | Tag():
                with open(path / f"{self.__now_str}.{id_value}.html", mode="w") as file:
                    file.write(str(result))
            case _:
                pickle.dump(
                    result,
                    open(path / f"{self.__now_str}.{id_value}.pkl", mode="wb"),
                )

    @overload
    def function(self, func: Callable[P, R]) -> Callable[P, R]: ...

    @overload
    def function(
        self,
        *,
        id_arg_subset: list[int] | list[str] | None = ...,
        use_raw_arg: bool = False,
        id_callback: Callable[P, dict[str, Any] | None] | None = ...,  # type: ignore
        use_json: bool = ...,
        name: str | None = ...,
        async_func: bool = ...,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def function(
        self,
        func: Callable[P, R] | None = None,
        *,
        id_arg_subset: list[int] | list[str] | None = None,
        use_raw_arg: bool = False,
        id_callback: Callable[P, dict[str, Any] | None] | None = None,
        use_json: bool = True,
        name: str | None = None,
        async_func: bool = False,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to cache wrapped function.

        Args:
            func: Function to cache.
            id_arg_subset:
                Number or name of the arguments to base hash id of result on.
            use_raw_arg:
                If ``True``, use the unhashed, string-formatted value of the id arg
                as filename. Only works for single id arg.
            id_callback:
                Callback function to use for retrieving a unique id from the arguments.
            use_json:
                Whether to use JSON as the format for caching dicts (instead of YAML).
            name:
                Custom name for the cache directory. Uses function name if not provided.
            async_func:
                Whether the function is async.
        """

        def inner(func: Callable[P, R]) -> Callable[P, R]:
            path = ensure_dir_exists(self.path / (name or func.__name__))

            @wraps(func)
            def inner_inner(*args: P.args, **kwargs: P.kwargs) -> R:
                id_value = self._get_id(
                    func, args, kwargs, use_raw_arg, id_arg_subset, id_callback
                )

                result = self._get_cached_result(func, path, id_value)
                if result is None:
                    log.debug(
                        f"⬇ Performing operation / fetching resource: '{id_value}'"
                    )

                    result = func(*args, **kwargs)
                    self._cache_result(result, path, id_value, use_json)

                return result

            return inner_inner

        def async_inner(async_func: Callable[P, R]) -> Callable[P, R]:
            func = cast(Callable[P, Awaitable[Any]], async_func)
            path = ensure_dir_exists(self.path / (name or func.__name__))

            @wraps(func)
            async def async_inner_inner(*args: P.args, **kwargs: P.kwargs) -> Any:
                id_value = self._get_id(
                    func, args, kwargs, use_raw_arg, id_arg_subset, id_callback
                )

                result = self._get_cached_result(func, path, id_value)
                if result is None:
                    log.debug(
                        f"⬇ Performing operation / fetching resource: '{id_value}'"
                    )

                    result = await func(*args, **kwargs)

                    self._cache_result(result, path, id_value, use_json)

                return result

            return cast(Callable[P, R], async_inner_inner)

        if async_func:
            return async_inner if func is None else async_inner(func)

        return inner if func is None else inner(func)


def get_cache(
    name: str | None = None,
    root_path: Path | None = None,
    max_cache_time: datetime.timedelta = datetime.timedelta(days=7),
):
    """Return a named cache instance private to the calling module.

    Args:
        name: Name of the cache (directory) to create.
        root_path: Root directory, where to store cache.
        max_cache_time: After how many days to invalidate cached objects and recompute.

    Returns:
        A cache instance.
    """
    root_path = root_path or default_root_path
    calling_module = get_calling_module()
    module_name = calling_module.__name__ if calling_module is not None else None

    return FileCache(
        ensure_dir_exists(root_path / (module_name or "root") / (name or "")),
        max_cache_time,
    )
