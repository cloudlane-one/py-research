"""Utilities for producing hashes of common objects."""

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from functools import reduce
from numbers import Number
from typing import Any

import pandas as pd
from pandas.util import hash_pandas_object


def _hash_sequence(s: Sequence) -> int:
    return reduce(lambda x, y: hash(str(x) + str(hash(y))), s, 0)


def gen_int_hash(x: Any) -> int:
    """Generate a hash-id from a flat dict."""
    match (x):
        case (Number() | str() | bytes() | date() | time() | datetime() | timedelta()):
            return hash(x)
        case pd.DataFrame() | pd.Series() | pd.Index():
            return sum(hash_pandas_object(x))
        case list() | tuple():
            return _hash_sequence(x)
        case dict():
            return sum(_hash_sequence(item) for item in x.items())
        case _:
            if hasattr(x, "__getstate__"):
                return sum(gen_int_hash(item) for item in x.__getstate__())
            if hasattr(x, "__getnewargs_ex__"):
                args, kwargs = x.__getnewargs_ex__()
                return sum(gen_int_hash(item) for item in args) + sum(
                    gen_int_hash(item) for item in kwargs.items()
                )
            if hasattr(x, "__getnewargs__"):
                args = x.__getnewargs__()
                return sum(gen_int_hash(item) for item in args)

    raise ValueError()


def gen_str_hash(x: Any, length: int = 10, raw_str: bool = False) -> str:
    """Generate a hash-id from a flat dict."""
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
