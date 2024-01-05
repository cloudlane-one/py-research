"""On-the-fly, file-based caching of function return values."""

import datetime
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup, Tag

from py_research.files import ensure_dir_exists
from py_research.hashing import gen_str_hash
from py_research.reflect import (
    get_calling_module_name,
    get_full_args_dict,
    get_return_type,
)
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

    max_cache_days: int = 7
    """After how many days to invalidate cached objects and recompute."""

    def __post_init__(self):  # noqa: D105
        now = datetime.datetime.now()
        self.__earliest_date = now - datetime.timedelta(days=self.max_cache_days)
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

    def function(  # noqa: C901
        self,
        id_arg_subset: list[int] | list[str] | None = None,
        use_raw_arg: bool = False,
        id_callback: Callable[P, dict[str, Any] | None] | None = None,
        use_json: bool = True,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to cache wrapped function.

        Args:
            id_arg_subset:
                Number or name of the arguments to base hash id of result on.
            use_raw_arg:
                If ``True``, use the unhashed, string-formatted value of the id arg
                as filename. Only works for single id arg.
            id_callback:
                Callback function to use for retrieving a unique id from the arguments.
            use_json:
                Whether to use JSON as the format for caching dicts (instead of YAML).
        """

        def inner(func: Callable[P, R]) -> Callable[P, R]:  # noqa: C901
            path = ensure_dir_exists(self.path / func.__name__)

            @wraps(func)
            def inner_inner(*args: P.args, **kwargs: P.kwargs) -> R:  # noqa: C901
                id_args = None

                if id_callback is not None:
                    id_args = id_callback(*args, **kwargs)

                if id_args is None:
                    id_args = get_full_args_dict(func, args, kwargs)
                    if id_arg_subset is not None:
                        id_args = {
                            k: v for k, v in id_args.items() if k in id_arg_subset
                        }

                id_value = gen_str_hash(
                    id_args if len(id_args) > 1 else list(id_args.values())[0],
                    raw_str=use_raw_arg,
                )

                filename_pattern = f"[0-9]*-[0-9]*-[0-9]*.{id_value}.*"

                result: R | None = None

                all_cached = [
                    f
                    for f in path.iterdir()
                    if f.match(filename_pattern) and self.__filter_outdated(f)
                ]

                if len(all_cached):
                    log.debug(f"💾 Taking cached result for: '{id_value}'")

                    last_cached = sorted(all_cached, key=self.__get_date_from_filename)[
                        -1
                    ]
                    extension = last_cached.name.split(".")[-1]
                    return_type = get_return_type(func) or type

                    cached_result = None
                    if issubclass(return_type, str) and extension == "txt":
                        with open(
                            last_cached,
                            encoding="utf-8",
                        ) as f:
                            cached_result = f.read()
                    elif issubclass(return_type, dict | list) and extension in (
                        "yaml",
                        "yml",
                    ):
                        cached_result = yaml.load(
                            open(last_cached, encoding="utf-8"), Loader=yaml.CLoader
                        )
                    elif issubclass(return_type, dict | list) and extension == "json":
                        cached_result = json.load(open(last_cached, encoding="utf-8"))
                    elif (
                        issubclass(return_type, pd.DataFrame | pd.Series)
                        and extension == "xlsx"
                    ):
                        cached_result = pd.read_excel(
                            last_cached, header=0, index_col=0
                        )
                    elif (
                        issubclass(return_type, pd.DataFrame | pd.Series)
                        and extension == "csv"
                    ):
                        cached_result = pd.read_csv(last_cached, header=0, index_col=0)
                    elif issubclass(return_type, np.ndarray) and extension == "npy":
                        cached_result = np.load(last_cached)
                    elif (
                        issubclass(return_type, BeautifulSoup | Tag)
                        and extension == "html"
                    ):
                        cached_result = BeautifulSoup(open(last_cached))
                    elif extension == "pkl":
                        cached_result = pickle.load(open(last_cached, mode="rb"))

                    result = cast(R | None, cached_result)

                if result is None:
                    log.debug(
                        f"⬇ Performing operation / fetching resource: '{id_value}'"
                    )

                    result = func(*args, **kwargs)

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
                            result.to_excel(
                                path / f"{self.__now_str}.{id_value}.xlsx", index=True
                            )
                        case np.ndarray():
                            np.save(path / f"{self.__now_str}.{id_value}.npy", result)
                        case BeautifulSoup() | Tag():
                            with open(
                                path / f"{self.__now_str}.{id_value}.html", mode="w"
                            ) as file:
                                file.write(str(result))
                        case _:
                            pickle.dump(
                                result,
                                open(
                                    path / f"{self.__now_str}.{id_value}.pkl", mode="wb"
                                ),
                            )

                return result

            return inner_inner

        return inner


def get_cache(
    name: str | None = None,
    root_path: Path = default_root_path,
    max_cache_time: datetime.timedelta = datetime.timedelta(days=365),
):
    """Return a named cache instance private to the calling module.

    Args:
        name: Name of the cache (directory) to create.
        root_path: Root directory, where to store cache.
        max_cache_time: After how many days to invalidate cached objects and recompute.

    Returns:
        A cache instance.
    """
    calling_module = get_calling_module_name() or "root"

    return FileCache(
        ensure_dir_exists(root_path / calling_module / (name or "")),
        max_cache_time.days,
    )
