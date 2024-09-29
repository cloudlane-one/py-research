"""Utilities for producing hashes of common objects."""

import hashlib
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timedelta
from functools import reduce
from numbers import Number
from types import FunctionType, GenericAlias, ModuleType
from typing import Any, get_args, get_origin

import pandas as pd
from pandas.util import hash_pandas_object

from py_research.reflect.ref import PyObjectRef


def _stable_hash(
    data: Number | str | bytes | date | time | datetime | timedelta,
) -> int:
    data = f"{type(data)}:{data}"
    m = hashlib.md5(usedforsecurity=False)
    m.update(bytes(data, "utf-8"))
    return int.from_bytes(m.digest())


def _hash_sequence(s: Sequence, _ctx: set[int]) -> int:
    return reduce(
        lambda x, y: gen_int_hash(str(x) + str(gen_int_hash(y, _ctx)), _ctx), s, 0
    )


def gen_int_hash(obj: Any, _ctx: set[int] | None = None) -> int:  # noqa: C901
    """Generate stable hash for obj (must be known, hashable or composed of such)."""
    _ctx = _ctx or set()

    if id(obj) in _ctx:
        return 0

    match obj:
        case None:
            return 0
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
        case Number() | str() | bytes() | date() | time() | datetime() | timedelta():
            return _stable_hash(obj)
        case pd.DataFrame() | pd.Series() | pd.Index():
            return sum(hash_pandas_object(obj))
        case list() | tuple():
            _ctx.add(id(obj))
            return _hash_sequence(obj, _ctx)
        case dict():
            _ctx.add(id(obj))
            return sum(_hash_sequence(item, _ctx) for item in obj.items())
        case set():
            _ctx.add(id(obj))
            return sum(gen_int_hash(item, _ctx) for item in obj)
        case FunctionType():
            return gen_int_hash(
                (getattr(obj, "__name__", None), list(obj.__code__.co_lines()))
            )
        case _:
            _ctx.add(id(obj))

            if is_dataclass(obj):
                return gen_int_hash(
                    {f.name: getattr(obj, f.name) for f in fields(obj)}, _ctx
                )
            if hasattr(obj, "__getstate__"):
                state = obj.__getstate__()
                state = state if isinstance(state, tuple) else (state,)
                state_dicts = [s for s in state if isinstance(s, dict)]

                if len(state_dicts) > 0:
                    return sum(
                        sum(_hash_sequence((k, v), _ctx) for k, v in s.items())
                        for s in state_dicts
                    )
            if hasattr(obj, "__getnewargs_ex__"):
                args, kwargs = obj.__getnewargs_ex__()
                return sum(gen_int_hash(item, _ctx) for item in args) + sum(
                    gen_int_hash(item, _ctx) for item in kwargs.items()
                )
            if hasattr(obj, "__getnewargs__"):
                args = obj.__getnewargs__()
                return sum(gen_int_hash(item, _ctx) for item in args)

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
