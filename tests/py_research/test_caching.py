"""Test the caching module."""

import datetime
from pathlib import Path
from tempfile import gettempdir

from py_research.caching import FileCache, get_cache


def test_get_cache_custom_params():
    """Test the custom parameters of get_cache."""
    custom_root_path = Path(gettempdir())
    custom_max_cache_time = datetime.timedelta(days=1)

    # Create via positional args.
    cache1 = get_cache("custom_cache", custom_root_path, custom_max_cache_time)

    # Assert positional args result in correct instance.
    assert isinstance(cache1, FileCache)
    assert (
        cache1.path
        == custom_root_path / "tests.py_research.test_caching" / "custom_cache"
    )
    assert cache1.max_cache_time == custom_max_cache_time

    # Create via keyword args.
    cache2 = get_cache(
        name="custom_cache",
        root_path=custom_root_path,
        max_cache_time=custom_max_cache_time,
    )

    # Assert keyword args result in correct instance.
    assert isinstance(cache2, FileCache)
    assert (
        cache2.path
        == custom_root_path / "tests.py_research.test_caching" / "custom_cache"
    )
    assert cache2.max_cache_time == custom_max_cache_time


def test_cache_function():
    """Test the cache_function decorator."""
    tempdir = Path(gettempdir())

    # Create cache.
    cache = get_cache("test_cache", tempdir)

    # Assert cache dir was created.
    assert cache.path == tempdir / "tests.py_research.test_caching" / "test_cache"
    assert (tempdir / "tests.py_research.test_caching" / "test_cache").exists()

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
