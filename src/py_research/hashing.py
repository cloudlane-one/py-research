"""Utilities for producing hashes of common objects."""

import hashlib
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from functools import reduce
from numbers import Number
from typing import Any

import pandas as pd
from pandas.util import hash_pandas_object


def _stable_hash(
    data: Number | str | bytes | date | time | datetime | timedelta,
) -> int:
    m = hashlib.md5(usedforsecurity=False)
    m.update(bytes(str(data), "utf-8"))
    return int.from_bytes(m.digest())


def _hash_sequence(s: Sequence) -> int:
    return reduce(lambda x, y: gen_int_hash(str(x) + str(gen_int_hash(y))), s, 0)


def gen_int_hash(obj: Any) -> int:
    """Generate stable hash for obj (must be hashable or composed of hashable types)."""
    match obj:
        case (Number() | str() | bytes() | date() | time() | datetime() | timedelta()):
            return _stable_hash(obj)
        case pd.DataFrame() | pd.Series() | pd.Index():
            return sum(hash_pandas_object(obj))
        case list() | tuple():
            return _hash_sequence(obj)
        case dict():
            return sum(_hash_sequence(item) for item in obj.items())
        case None:
            return 0
        case _:
            if hasattr(obj, "__getstate__"):
                state = obj.__getstate__()
                state = state if isinstance(state, tuple) else (state,)
                state_dicts = [s for s in state if isinstance(s, dict)]

                if len(state_dicts) > 0:
                    return sum(
                        sum(_hash_sequence(item) for item in s.items())
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


def gen_str_hash(x: Any, length: int = 10, raw_str: bool = False) -> str:
    """Generate stable hash for obj (must be hashable or composed of hashable types)."""
    s = None
    match (x):
        case int() | float() | complex() | Decimal():
            s = str(x)
        case Fraction():
            s = str(x).replace("/", "_over_")
        case str() if raw_str:
            s = x
        case date():
            s = x.isoformat()
        case time():
            s = f"HHMMSS{'ffffff' if x.microsecond != 0 else ''}".format(x)
        case datetime():
            s = str(x.timestamp())
        case timedelta():
            s = str(x.total_seconds())
        case _:
            s = str(abs(gen_int_hash(x)))

    return s[:length]
