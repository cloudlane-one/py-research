"""Utilities for producing hashes of common objects."""

import hashlib
from collections.abc import Callable, Hashable, Sequence
from contextvars import ContextVar
from dataclasses import fields
from datetime import date, datetime, time, timedelta
from functools import reduce
from numbers import Number
from types import FunctionType, GenericAlias, ModuleType, UnionType
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    TypeAliasType,
    TypeVar,
    get_args,
    get_origin,
)

import pandas as pd
import polars as pl
from pandas.util import hash_pandas_object

from py_research.reflect.ref import PyObjectRef
from py_research.types import (
    ArgsPicklable,
    ArgsPicklableEx,
    DataclassInstance,
    StatePicklable,
)


def _stable_hash(
    data: Number | str | bytes | date | time | datetime | timedelta,
) -> int:
    data = f"{type(data)}:{data}"
    m = hashlib.md5(usedforsecurity=False)
    m.update(bytes(data, "utf-8"))
    return int.from_bytes(m.digest())


def _hash_sequence(s: Sequence, _ctx: set[int]) -> int:
    return reduce(lambda x, y: gen_int_hash(str(x) + str(gen_int_hash(y))), s, 0)


current_ctx: ContextVar[set[int] | None] = ContextVar("current_ctx", default=None)

T = TypeVar("T")
P = ParamSpec("P")


def _hashing_context(func: Callable[Concatenate[set[int], P], T]) -> Callable[P, T]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        ctx = current_ctx.get()
        if ctx is None:
            new_ctx: set[int] = set()
            current_ctx.set(new_ctx)
            try:
                return func(new_ctx, *args, **kwargs)
            finally:
                current_ctx.set(None)
        else:
            return func(ctx, *args, **kwargs)

    return inner


@_hashing_context
def gen_int_hash(_ctx: set[int], obj: Any) -> int:
    """Generate stable hash for obj (must be known, hashable or composed of such)."""
    if id(obj) in _ctx:
        return 0

    match obj:
        case Hashable():
            _ctx.add(id(obj))
            return hash(obj)
        case None:
            return 0
        case type() | ModuleType() | TypeAliasType():
            return gen_int_hash(PyObjectRef.reference(obj).to_url())
        case FunctionType() if hasattr(obj, "__name__"):
            return gen_int_hash(PyObjectRef.reference(obj).to_url())
        case FunctionType():
            # This may be unstable as closures are not considered.
            return gen_int_hash(list(obj.__code__.co_lines()))
        case GenericAlias() | UnionType():
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
        case pl.DataFrame():
            return sum(obj.hash_rows())
        case pl.Series():
            return sum(obj.hash())
        case list() | tuple():
            _ctx.add(id(obj))
            return _hash_sequence(obj, _ctx)
        case dict():
            _ctx.add(id(obj))
            return sum(_hash_sequence(item, _ctx) for item in obj.items())
        case set():
            _ctx.add(id(obj))
            return sum(gen_int_hash(item) for item in obj)
        case DataclassInstance():
            _ctx.add(id(obj))
            return gen_int_hash({f.name: getattr(obj, f.name) for f in fields(obj)})
        case StatePicklable():
            state = obj.__getstate__()
            state = state if isinstance(state, tuple) else (state,)
            state_dicts = [s for s in state if isinstance(s, dict)]

            if len(state_dicts) == 0:
                raise ValueError()

            return sum(
                sum(_hash_sequence((k, v), _ctx) for k, v in s.items())
                for s in state_dicts
            )
        case ArgsPicklable():
            args = obj.__getnewargs__()
            return sum(gen_int_hash(item) for item in args)
        case ArgsPicklableEx():
            args, kwargs = obj.__getnewargs_ex__()
            return sum(gen_int_hash(item) for item in args) + sum(
                gen_int_hash(item) for item in kwargs.items()
            )
        case _:
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
