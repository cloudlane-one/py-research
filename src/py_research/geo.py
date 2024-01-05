"""Utilities for working with geographical data, esp. data associated with countries."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import auto
from typing import Literal, TypeAlias, cast

import country_converter as coco
import pandas as pd
from typing_extensions import Self

from py_research.enums import StrEnum


class GeoAlliance(StrEnum):
    """List of international alliances used to define geo-regions of interest."""

    EU = auto()
    """https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:EU_enlargements"""

    EU12 = auto()
    EU15 = auto()
    EU25 = auto()
    EU27 = auto()
    EU27_2007 = auto()
    EU28 = auto()

    EEA = auto()
    """https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:European_Economic_Area_(EEA)"""

    G7 = auto()
    """https://en.wikipedia.org/wiki/G7"""

    G20 = auto()
    """https://en.wikipedia.org/wiki/G20"""

    APEC = auto()
    """https://en.wikipedia.org/wiki/Asia-Pacific_Economic_Cooperation"""

    BRIC = auto()
    """https://en.wikipedia.org/wiki/BRIC"""

    BASIC = auto()
    """https://en.wikipedia.org/wiki/BASIC_countries"""

    CIS = auto()
    """https://en.wikipedia.org/wiki/Commonwealth_of_Independent_States"""

    OECD = auto()
    """https://www.oecd.org/about/members-and-partners/"""


class GeoScheme(StrEnum):
    """List of schemes, which can be used to define geo-regions of interest."""

    country_name = auto()
    """Short name of a country."""

    continent = auto()
    """Name of a continent."""

    cc_iso3 = auto()
    """ISO3 code of a country."""

    cc_iso2 = auto()
    """ISO2 code of a country."""

    alliance = auto()
    """Name of an international alliance, to which a country belongs.
    """

    world = auto()
    """Dummy scheme to match all of the world."""

    @staticmethod
    def _scheme_map() -> dict["GeoScheme", str]:
        return {
            GeoScheme.country_name: "name_short",
            GeoScheme.continent: "continent",
            GeoScheme.cc_iso3: "ISO3",
            GeoScheme.cc_iso2: "ISO2",
        }

    def to_coco_scheme(self) -> str | None:
        """Return associated coco scheme, if applicable."""
        return self._scheme_map().get(self)


@dataclass(frozen=True)
class GeoRegion:
    """Define a geo-region according to some scheme (country, continent, etc.)."""

    label: str
    """The geo-location's label according to `scheme`."""

    scheme: GeoScheme = GeoScheme.country_name
    """The naming/classification scheme used."""

    display_label: str | None = None
    """Optional, custom display label for the geo-location."""

    exclude_already_covered: bool = True
    """When listing multiple geo-regions, exclude locations
    which have already been covered by previously listed regions."""

    def get_label(self) -> str:
        """Return string label of this region."""
        return self.display_label or self.label

    def to_country_list(
        self,
        scheme: "CountryScheme" = GeoScheme.cc_iso3,
    ) -> list[str]:
        """Return list of matching countries in given ``scheme``.

        Args:
            scheme: The scheme to convert to countries.

        Returns:
            List of countries in ``scheme``.
        """
        res = None

        match (self.scheme):
            case (
                GeoScheme.country_name
                | GeoScheme.continent
                | GeoScheme.cc_iso3
                | GeoScheme.cc_iso2
            ):
                res = coco.convert(
                    self.label,
                    src=self.scheme.to_coco_scheme(),
                    to=scheme.to_coco_scheme(),
                )
            case GeoScheme.alliance:
                res = coco.convert(
                    self.label, src=self.label, to=scheme.to_coco_scheme()
                )
            case GeoScheme.world:
                res = coco.CountryConverter().data[scheme.to_coco_scheme()].to_list()
            case _:
                raise ValueError(f"Unsupported geo-location scheme: {self.scheme}")

        return res if isinstance(res, list) else [res]


def _region_to_country_map(
    self,
    scheme: "CountryScheme" = GeoScheme.cc_iso3,
) -> pd.DataFrame:
    """Resolve geo-region definition to a dataframe.

    Result df maps the region's (display) label to one or more
    ISO3 country codes.
    """
    iso3_list = self.to_country_list(scheme=scheme)
    return pd.DataFrame({"cc": iso3_list}).assign(geo_region=self.get_label())


CountryScheme: TypeAlias = Literal[
    GeoScheme.country_name, GeoScheme.cc_iso2, GeoScheme.cc_iso3
]
"""A :py:class:`GeoScheme` which can be used to define a single country.`"""


@dataclass(frozen=True)
class Country:
    """A country represented by ISO2 code, ISO3 code or name."""

    label: str
    """The country's label according to `scheme`."""

    scheme: CountryScheme | None = None
    """The naming/classification scheme used."""

    def to(self, scheme: CountryScheme) -> Self:
        """Convert to other scheme."""
        return cast(
            Self,
            Country(
                cast(
                    str,
                    coco.convert(
                        self.label, to=scheme.to_coco_scheme(), src=self.scheme
                    ),
                ),
                scheme,
            )
            if scheme != self.scheme
            else self,
        )

    def __str__(self) -> str:  # noqa: D105
        return self.label

    def __format__(self, spec: str) -> str:  # noqa: D105
        return str(self.to(cast(CountryScheme, GeoScheme(spec))))


def countries_to_scheme(
    countries: pd.Series,
    target: CountryScheme = GeoScheme.cc_iso3,
    src: CountryScheme | None = None,
) -> pd.Series:
    """Translate given series of country labels to ``scheme``.

    Args:
        countries: Series of country labels.
        target: Target scheme to translate to.
        src: Source scheme to translate from.

    Returns:
        Series of translated country labels.
    """
    return pd.Series(
        coco.convert(
            countries,
            src=src.to_coco_scheme() if src is not None else None,
            to=target.to_coco_scheme(),
        ),
        index=countries.index,
    )


def expand_geo_col_to_cc(
    df: pd.DataFrame,
    geo_col: str,
    scheme: GeoScheme = GeoScheme.country_name,
    cc_scheme: CountryScheme = GeoScheme.cc_iso3,
) -> pd.DataFrame:
    """Expand geo-regions present in ``geo_col`` to country codes.

    Expand such that rows of ``df`` with multiple mapped CCs are multiplicated.

    Args:
        df: The dataframe to expand.
        geo_col: The column containing geo-regions.
        scheme: The scheme used to define the geo-regions.
        cc_scheme: The scheme to expand to.

    Returns:
        The expanded dataframe.
    """
    cc_map = pd.concat(
        [
            _region_to_country_map(GeoRegion(g, scheme=scheme), scheme=cc_scheme)
            for g in df[geo_col].dropna().unique()
        ]
    )

    return df.merge(cc_map, left_on=geo_col, right_on="geo_region", how="left").drop(
        columns="geo_region"
    )


def merge_geo_regions(
    df: pd.DataFrame,
    geo_col: str,
    geo_regions: Iterable[GeoRegion | str],
    input_scheme: GeoScheme = GeoScheme.country_name,
    rest_of_world: bool = True,
    pretty_labels: bool = True,
) -> pd.DataFrame:
    """Right-merge ``geo_regions`` onto ``df`` based on ``geo_col``.

    Merge such that rows with multiple mapped regions are multiplicated.

    Args:
        df: The dataframe to merge into.
        geo_col: The column containing geo-regions.
        geo_regions: The geo-regions to merge.
        input_scheme: The scheme used to define the geo-regions.
        rest_of_world: Whether to add a "Rest of World" region.
        pretty_labels: Whether to use pretty labels for regions.

    Returns:
        The merged dataframe.
    """
    src_df = expand_geo_col_to_cc(df, geo_col=geo_col, scheme=input_scheme)

    cc_coverage = set()
    res_df = pd.DataFrame()

    if rest_of_world:
        geo_regions = [
            *geo_regions,
            GeoRegion("world", GeoScheme.world, exclude_already_covered=True),
        ]

    for gr in geo_regions:
        gr = gr if isinstance(gr, GeoRegion) else GeoRegion(gr)
        geo_region_res = gr.get_label()

        cc_map = _region_to_country_map(gr)
        cc_set = set(cc_map["cc"])
        cc_already_covered = list(cc_coverage & cc_set)

        if pretty_labels and (
            gr.scheme == GeoScheme.cc_iso2 or gr.scheme == GeoScheme.cc_iso3
        ):
            geo_region_res = coco.convert(
                geo_region_res, src=gr.scheme.to_coco_scheme(), to="name"
            )

        if len(cc_already_covered) > 0 and gr.exclude_already_covered:
            geo_region_res = f"Rest of {gr.get_label()}"
            cc_map = cc_map.loc[~cc_map["cc"].isin(cc_already_covered)]

        res_df = pd.concat(
            [
                res_df,
                src_df.merge(cc_map, on="cc", how="inner").assign(
                    geo_region=geo_region_res, geo_region_label=gr.get_label()
                ),
            ]
        )
        cc_coverage |= cc_set

    return res_df


def match_to_geo_region(
    countries: pd.Series,
    geo_region: GeoRegion,
    country_scheme: CountryScheme | None = None,
) -> pd.Series:
    """Check whether countries are in given geo-region.

    Args:
        countries: Series of countries to check.
        geo_region: The geo-region to check against.
        country_scheme: The scheme of the countries.

    Returns:
        Series of booleans indicating whether countries are in geo-region.
    """
    return countries_to_scheme(countries, GeoScheme.cc_iso3, src=country_scheme).isin(
        geo_region.to_country_list(scheme=GeoScheme.cc_iso3)
    )


flag_sizes = pd.Series([20, 40, 80, 160, 320, 640, 1280, 2560])
"""List of available flag image sizes."""


def gen_flag_url(cc: pd.Series, width: int) -> pd.Series:
    """Get the URL of a small flag image for a given country code.

    Args:
        cc: Series of country codes.
        width: The desired width of the flag.

    Returns:
        Series of flag image URLs.
    """
    return (
        "https://flagcdn.com/w"
        + str(flag_sizes.loc[flag_sizes > width].min())
        + "/"
        + countries_to_scheme(cc, GeoScheme.cc_iso2).str.lower()
        + ".png"
    )


def gen_flag_img_tag(cc: pd.Series, width: int) -> pd.Series:
    """Generate a HTML image tag with a small flag for a given country code.

    Args:
        cc: Series of country codes.
        width: The desired width of the flag.

    Returns:
        Series of HTML image tags.
    """
    flags = (
        "<img "
        + (' src="' + gen_flag_url(cc, width) + '"')
        + (
            (' srcset="' + gen_flag_url(cc, width * 2) + ' 2x"')
            if width != 2560
            else ""
        )
        + f' width="{width}"'
        + (' alt="' + countries_to_scheme(cc, GeoScheme.country_name) + '"')
        + ' style="border: 1px solid #00000080"'
        + "/>"
    )

    return flags.where(~cc.isin(["not found", "", "?"]) & cc.notna(), "")
