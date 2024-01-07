"""Test time module."""

from datetime import datetime

import pandas as pd
from py_research.time import datetime_to_interval_series


def test_datetime_to_interval_series():
    """Test datetime_to_interval_series."""
    # Test with a Series of datetime objects
    dt_series = pd.Series(
        [
            datetime(2021, 1, 1),
            datetime(2021, 1, 2),
            datetime(2021, 1, 3),
            datetime(2022, 1, 4),
            datetime(2023, 1, 5),
        ]
    )
    interval_series = datetime_to_interval_series(dt_series)
    assert interval_series.equals(pd.Series([2021, 2021, 2021, 2022, 2023]))
