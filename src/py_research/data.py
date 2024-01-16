"""Utilities for data handling."""

import locale
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy.typing as npt
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)
from pandas.core.dtypes.base import ExtensionDtype
from pandas.errors import ParserError

from py_research.hashing import gen_int_hash

YES = ["y", "t", "1", "yes", "true"]
NO = ["n", "f", "0", "no", "false"]


def is_number_dtype(dtype: str | npt.DTypeLike | ExtensionDtype) -> bool:
    """Check if dtype is number-like.

    Args:
      dtype: dtype to check.

    Returns:
      True if dtype is number-like.
    """
    return (
        is_numeric_dtype(dtype)
        or is_datetime64_any_dtype(dtype)
        or is_timedelta64_dtype(dtype)
    )


def to_boolean(s: pd.Series) -> pd.Series:
    """Parse boolean series from string series.

    Args:
      s: string series.

    Returns:
      Boolean series.
    """
    s_lower = s.str.lower()
    if not s_lower.isin([*YES, *NO]).all():
        raise ValueError("Series contains invalid values.")

    return s_lower.isin(YES)


def to_integer(s: pd.Series) -> pd.Series:
    """Parse integer series from string series with locale-awareness.

    Args:
      s: string series.

    Returns:
      Integer series.
    """
    return s.astype(str).map(locale.atoi)


def to_float(s: pd.Series) -> pd.Series:
    """Parse float series from string series with locale-awareness.

    Args:
      s: string series.

    Returns:
      Float series.
    """
    return s.astype(str).map(locale.atof)


def parse_dtype(  # noqa: C901
    s: pd.Series,
    dtype: str | type | npt.DTypeLike | None = None,
    src_locale: str | None = None,
) -> pd.Series:
    """Parse series to dtype with locale-awareness.

    Args:
      s: series to convert.
      dtype: dtype to convert to.
      src_locale: locale to use for conversion.

    Returns:
      Converted series.
    """
    if s.dtype == "object":
        s = s.astype(str)

    if not is_string_dtype(s):
        return s

    result = None

    context_locale = None
    if src_locale is not None:
        try:
            context_locale, _ = locale.getlocale(locale.LC_ALL)
        except (TypeError, ValueError):
            context_locale = None
        locale.setlocale(locale.LC_ALL, src_locale)

    if dtype is not None:
        result = (
            to_boolean(s)
            if is_bool_dtype(dtype)
            else to_integer(s)
            if is_integer_dtype(dtype)
            else to_float(s)
            if is_float_dtype(dtype)
            else s
        ).astype(
            dtype  # type: ignore
        )
    else:
        try:
            result = to_boolean(s)
        except (ValueError, TypeError):
            try:
                result = to_integer(s)
            except (ValueError, TypeError):
                try:
                    result = to_float(s)
                except (ValueError, TypeError):
                    try:
                        result = pd.to_numeric(s)
                    except (ValueError, TypeError):
                        try:
                            result = pd.to_datetime(s, infer_datetime_format=True)
                        except (ParserError, ValueError, TypeError):
                            if s.nunique() < len(s) / 5:
                                result = s.astype("category")

                            if src_locale is None:
                                result = parse_dtype(s, dtype, "C")
                            else:
                                result = s

    if src_locale is not None and context_locale is not None:
        locale.setlocale(locale.LC_ALL, context_locale)

    return result


def gen_id(x: Any, length: int = 10, raw_str: bool = False) -> str:
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
