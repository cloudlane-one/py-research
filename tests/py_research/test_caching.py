"""Test the caching module."""

import datetime
from pathlib import Path
from tempfile import gettempdir
from unittest.mock import patch

import py_research.files
import py_research.reflect
from py_research.caching import FileCache, get_cache
from py_research.reflect import stref


def test_get_cache_default_params():
    """Test the default parameters of get_cache."""
    cache = get_cache()
    assert isinstance(cache, FileCache)


@patch(stref(py_research.reflect.get_calling_module_name), return_value="test")
@patch(stref(py_research.files.ensure_dir_exists), new=lambda x: x)
def test_get_cache_custom_params():
    """Test the custom parameters of get_cache."""
    custom_root_path = Path("/custom/path")
    custom_max_cache_time = datetime.timedelta(days=1)

    # Create via positional args.
    cache1 = get_cache("custom_cache", custom_root_path, custom_max_cache_time)

    # Assert positional args result in correct instance.
    assert isinstance(cache1, FileCache)
    assert cache1.path == custom_root_path / "test" / "custom_cache"
    assert cache1.max_cache_time == custom_max_cache_time

    # Create via keyword args.
    cache2 = get_cache(
        name="custom_cache",
        root_path=custom_root_path,
        max_cache_time=custom_max_cache_time,
    )

    # Assert keyword args result in correct instance.
    assert isinstance(cache2, FileCache)
    assert cache2.path == custom_root_path / "test" / "custom_cache"
    assert cache2.max_cache_time == custom_max_cache_time


@patch(stref(py_research.reflect.get_calling_module_name), return_value="test")
def test_cache_function():
    """Test the cache_function decorator."""
    tempdir = Path(gettempdir())

    # Create cache.
    cache = get_cache("test_cache", tempdir)

    # Assert cache dir was created.
    assert cache.path == tempdir / "test" / "test_cache"
    assert (tempdir / "test" / "test_cache").exists()

    # Create function to cache.
    @cache.function(id_arg_subset=["a", "b"])
    def test_func(a: int, b: int, c: int) -> int:
        return a + b + c

    # Assert function returns correct value.
    assert test_func(1, 2, 3) == 6

    # Assert function returns cached value for same id args.
    assert test_func(1, 2, 4) == 6

    # Assert function returns new value for different id args.
    assert test_func(1, 3, 4) == 8

    # Create another function to cache.
    @cache.function(id_arg_subset=["a"], use_raw_arg=True)
    def test_func2(a: int, b: int) -> int:
        return a + b

    # Assert function returns correct value.
    assert test_func2(1, 2) == 3

    # Assert function returns cached value for same id args.
    assert test_func2(1, 3) == 3

    # Assert function returns new value for different id args.
    assert test_func2(2, 3) == 5
