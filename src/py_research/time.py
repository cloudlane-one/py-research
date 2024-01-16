"""Utilities for working with date and time in data."""

from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import pandas as pd

from py_research.data import parse_dtype


def _auto_interval_format(
    time_interval: pd.offsets.BaseOffset, min_bin: datetime, max_bin: datetime
) -> tuple[str | Callable[[Any], str], str]:
    format_func = "%c"
    interval_name = "interval"

    date_prefix = (
        ""
        if max_bin < min_bin + pd.offsets.DateOffset(days=1)
        else "%d "
        if max_bin < min_bin + pd.offsets.DateOffset(months=1)
        else "%m-%d "
        if max_bin < min_bin + pd.offsets.DateOffset(years=1)
        else "%Y-%m-%d "
    )

    match (time_interval):
        case pd.offsets.YearBegin() | pd.offsets.YearEnd():
            format_func = "%Y"
            interval_name = "year"
        case pd.offsets.QuarterBegin() | pd.offsets.QuarterEnd():

            def quarter_format(d: datetime) -> str:
                q = (d.month - 1) // 3 + 1
                if max_bin < min_bin + pd.offsets.DateOffset(years=1):
                    return f"Q{q}"
                return f"{d.year} Q{q}"

            format_func = quarter_format
        case pd.offsets.MonthBegin() | pd.offsets.MonthEnd():
            interval_name = "month"
            if max_bin < min_bin + pd.offsets.DateOffset(years=1):
                format_func = "%Y"
            else:
                format_func = "%Y-%m"
        case pd.offsets.Day():
            interval_name = "day"
            if max_bin < min_bin + pd.offsets.DateOffset(years=1):
                format_func = "%Y"
            elif max_bin < min_bin + pd.offsets.DateOffset(months=1):
                format_func = "%Y-%m"
            else:
                format_func = "%Y-%m-%d"
        case pd.offsets.Week():
            interval_name = "week"
            if max_bin < min_bin + pd.offsets.DateOffset(years=1):
                format_func = "%W"
            else:
                format_func = "%Y week %W"
        case pd.offsets.Hour():
            interval_name = "time"
            format_func = f"{date_prefix}%Hh"
        case pd.offsets.Minute():
            interval_name = "time"
            if max_bin < min_bin + pd.offsets.DateOffset(hours=1):
                format_func = "%M"
            else:
                format_func = f"{date_prefix}%H:%M"
        case pd.offsets.Second():
            interval_name = "time"
            if max_bin < min_bin + pd.offsets.DateOffset(minutes=1):
                format_func = f"{date_prefix}%S"
            elif max_bin < min_bin + pd.offsets.DateOffset(hours=1):
                format_func = f"{date_prefix}%M:%S"
            else:
                format_func = f"{date_prefix}%H:%M:%S"
        case _:
            interval_name = "datetime"
            format_func = "%c"

    return format_func, interval_name


def datetime_to_interval_series(
    datetime_series: pd.Series,
    time_interval: pd.offsets.BaseOffset = pd.offsets.YearEnd(),
    format: str | None = None,
    interval_col: str | None = None,
) -> pd.Series:
    """Assign intervals matching ``datetime_col`` to new column.

    Args:
        datetime_series: Series of datetime values.
        time_interval: Interval to use for grouping.
        format: Format to use for the interval column.
        interval_col: Name of the interval column.

    Returns:
        Series of intervals matching ``datetime_col``.
    """
    datetime_df = (
        datetime_series.to_frame().assign(time_bin=datetime_series).reset_index()
    )
    resampled = (
        datetime_df.resample(time_interval, group_keys=True, on="time_bin")
        .apply(lambda df: pd.DataFrame(df, index=df.index))
        .reset_index()
    ).set_index(datetime_series.index.name or "index")["time_bin"]

    format_func = format
    if format_func is None:
        format_func, interval_name = _auto_interval_format(
            time_interval, resampled.min(), resampled.max()
        )
        interval_col = interval_col or interval_name

    def apply_format(s: pd.Series) -> pd.Series:
        assert format_func is not None
        return (
            s.dt.strftime(format_func)
            if isinstance(format_func, str)
            else s.map(format_func)
        )

    if time_interval.n == 1:
        resampled = apply_format(resampled)
        resampled = parse_dtype(resampled)
    else:
        start = resampled
        end = cast(pd.Series, resampled + time_interval)  # type: ignore
        resampled = apply_format(start) + " - " + apply_format(end)

        interval_col = (
            f"{interval_col + ' ' if interval_col is not None else ''}interval"
        )

    return resampled.rename(interval_col)
