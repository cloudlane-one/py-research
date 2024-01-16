"""Test the geo module."""

import country_converter as coco
import pandas as pd
import pytest

from py_research.geo import (
    CountryScheme,
    GeoRegion,
    GeoScheme,
    countries_to_scheme,
    expand_geo_col_to_cc,
    gen_flag_url,
    match_to_geo_region,
    merge_geo_regions,
)


@pytest.mark.parametrize(
    "countries, target, src, expected",
    [
        (
            pd.Series(["United States", "Canada", "Mexico"]),
            GeoScheme.cc_iso3,
            GeoScheme.country_name,
            pd.Series(["USA", "CAN", "MEX"]),
        ),
        (
            pd.Series(["USA", "CAN", "MEX"]),
            GeoScheme.cc_iso3,
            None,
            pd.Series(["USA", "CAN", "MEX"]),
        ),
        (
            pd.Series([], dtype=object),
            GeoScheme.cc_iso3,
            None,
            pd.Series([], dtype=object),
        ),
    ],
)
def test_countries_to_scheme(
    countries: pd.Series, target: CountryScheme, src: CountryScheme, expected: pd.Series
):
    """Test the countries_to_scheme function."""
    result = countries_to_scheme(countries, target, src)
    pd.testing.assert_series_equal(
        result, expected, check_names=False, check_dtype=False
    )


@pytest.mark.parametrize(
    "df, geo_col, scheme, cc_scheme, expected",
    [
        (
            pd.DataFrame({"geo_region": ["America", "Europe"]}),
            "geo_region",
            GeoScheme.continent,
            GeoScheme.cc_iso3,
            set(
                [
                    *coco.convert(["America"], src="continent"),
                    *coco.convert(["Europe"], src="continent"),
                ]
            ),
        ),
        (
            pd.DataFrame({"geo_region": ["Asia"]}),
            "geo_region",
            GeoScheme.continent,
            GeoScheme.cc_iso3,
            set(
                [
                    *coco.convert(["Asia"], src="continent"),
                ]
            ),
        ),
    ],
)
def test_expand_geo_col_to_cc(
    df: pd.DataFrame,
    geo_col: str,
    scheme: GeoScheme,
    cc_scheme: CountryScheme,
    expected: pd.DataFrame,
):
    """Test the expand_geo_col_to_cc function."""
    result = expand_geo_col_to_cc(df, geo_col, scheme, cc_scheme)
    assert set(result["cc"]) == expected


@pytest.mark.parametrize(
    "df, geo_col, geo_regions, input_scheme, expected",
    [
        (
            pd.DataFrame(
                {"cc": ["USA", "CAN", "MEX", "FRA", "DEU", "GBR", "CHN", "JPN", "IND"]}
            ),
            "cc",
            [
                GeoRegion("DEU", GeoScheme.cc_iso3),
                GeoRegion("America", GeoScheme.continent),
                GeoRegion("EU", GeoScheme.alliance),
            ],
            GeoScheme.cc_iso3,
            pd.Series(
                [
                    "America",
                    "America",
                    "America",
                    "Rest of EU",
                    "Germany",
                    "Rest of world",
                    "Rest of world",
                    "Rest of world",
                    "Rest of world",
                ]
            ).sort_values(),
        ),
        (
            pd.DataFrame({"cc": []}),
            "cc",
            [
                GeoRegion("DEU", GeoScheme.cc_iso3),
                GeoRegion("America", GeoScheme.continent),
                GeoRegion("EU", GeoScheme.alliance),
            ],
            GeoScheme.cc_iso3,
            pd.Series([], dtype=object),
        ),
    ],
)
def test_merge_geo_regions(
    df: pd.DataFrame,
    geo_col: str,
    geo_regions: list[GeoRegion],
    input_scheme: GeoScheme,
    expected: pd.Series,
):
    """Test the merge_geo_regions function."""
    result = merge_geo_regions(df, geo_col, geo_regions, input_scheme)
    pd.testing.assert_series_equal(
        result["geo_region"].sort_values(),
        expected,
        check_index=False,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.parametrize(
    "countries, geo_region, country_scheme, expected",
    [
        (
            pd.Series(["USA", "CAN", "MEX"]),
            GeoRegion("America", GeoScheme.continent),
            GeoScheme.cc_iso3,
            pd.Series([True, True, True]),
        ),
        (
            pd.Series(["FRA", "DEU", "GBR"]),
            GeoRegion("Europe", GeoScheme.continent),
            GeoScheme.cc_iso3,
            pd.Series([True, True, True]),
        ),
        (
            pd.Series(["CHN", "JPN", "IND"]),
            GeoRegion("Asia", GeoScheme.continent),
            GeoScheme.cc_iso3,
            pd.Series([True, True, True]),
        ),
        (
            pd.Series(["USA", "CAN", "DEU"]),
            GeoRegion("EU", GeoScheme.alliance),
            GeoScheme.cc_iso3,
            pd.Series([False, False, True]),
        ),
    ],
)
def test_match_to_geo_region(
    countries: pd.Series,
    geo_region: GeoRegion,
    country_scheme: CountryScheme,
    expected: pd.Series,
):
    """Test the match_to_geo_region function."""
    result = match_to_geo_region(countries, geo_region, country_scheme)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "cc, width",
    [
        (pd.Series(["USA"]), 160),
        (pd.Series(["FRA"]), 320),
        (pd.Series(["CHN"]), 320),
        (pd.Series(["USA", "FRA", "CHN"]), 160),
    ],
)
def test_gen_flag_url(cc: pd.Series, width: int):
    """Test the gen_flag_url function."""
    result = gen_flag_url(cc, width)
    assert isinstance(result, pd.Series)
