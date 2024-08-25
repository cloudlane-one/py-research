"""Utilities for producing hashes of common objects."""

import hashlib
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from functools import reduce
from numbers import Number
from types import GenericAlias, ModuleType
from typing import Any, get_args, get_origin

import pandas as pd
from pandas.util import hash_pandas_object

from py_research.reflect.ref import PyObjectRef


def _stable_hash(
    data: Number | str | bytes | date | time | datetime | timedelta,
) -> int:
    m = hashlib.md5(usedforsecurity=False)
    m.update(bytes(str(data), "utf-8"))
    return int.from_bytes(m.digest())


def _hash_sequence(s: Sequence) -> int:
    return reduce(lambda x, y: gen_int_hash(str(x) + str(gen_int_hash(y))), s, 0)


def gen_int_hash(obj: Any) -> int:  # noqa: C901
    """Generate stable hash for obj (must be known, hashable or composed of such)."""
    match obj:
        case Number() | str() | bytes() | date() | time() | datetime() | timedelta():
            return _stable_hash(obj)
        case pd.DataFrame() | pd.Series() | pd.Index():
            return sum(hash_pandas_object(obj))
        case list() | tuple():
            return _hash_sequence(obj)
        case dict():
            return sum(_hash_sequence(item) for item in obj.items())
        case set():
            return sum(gen_int_hash(item) for item in obj)
        case type() | ModuleType():
            return gen_int_hash(PyObjectRef.reference(obj).to_url())
        case GenericAlias():
            base = get_origin(obj)
            args = get_args(obj)
            return gen_int_hash(
                (
                    PyObjectRef.reference(base).to_url(),
                    *(PyObjectRef.reference(arg).to_url() for arg in args),
                )
            )
        case None:
            return 0
        case _:
            if hasattr(obj, "__getstate__"):
                state = obj.__getstate__()
                state = state if isinstance(state, tuple) else (state,)
                state_dicts = [s for s in state if isinstance(s, dict)]

                if len(state_dicts) > 0:
                    return sum(
                        sum(
                            _hash_sequence((k, v)) for k, v in s.items() if v is not obj
                        )
                        for s in state_dicts
                    )
            if hasattr(obj, "__getnewargs_ex__"):
                args, kwargs = obj.__getnewargs_ex__()
                return sum(gen_int_hash(item) for item in args) + sum(
                    gen_int_hash(item) for item in kwargs.items()
                )
            if hasattr(obj, "__getnewargs__"):
                args = obj.__getnewargs__()
                return sum(gen_int_hash(item) for item in args)

    raise ValueError()


def gen_str_hash(x: Any, length: int = 10) -> str:
    """Generate stable hash for obj (must be known, hashable or composed of such).

    Args:
        x: Object to hash.
        length: Length of the hash.
        raw_str:
            Whether to use the raw string representation of the object,
            if it is a string.

    Returns:
        Hash of the object as string.
    """
    s = str(abs(gen_int_hash(x)))
    return s[:length].rjust(length, "0")
