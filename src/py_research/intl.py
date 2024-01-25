"""Simple & semi-automatic intl. + localization of data analysis functions."""

import re
from collections.abc import Callable, Generator, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial
from itertools import groupby
from locale import LC_ALL, getlocale, normalize, setlocale
from os import environ
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeAlias, TypeVar, cast

import pandas as pd
from babel import Locale, UnknownLocaleError
from babel.dates import (
    format_date,
    format_datetime,
    format_interval,
    format_time,
    format_timedelta,
)
from babel.numbers import format_decimal
from deep_translator import GoogleTranslator
from pydantic import parse_obj_as
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Self
from yaml import CLoader, load

from py_research.caching import get_cache
from py_research.enums import StrEnum
from py_research.geo import Country, CountryScheme, GeoScheme
from py_research.hashing import gen_int_hash

cache = get_cache()


P = ParamSpec("P")


class DtUnit(StrEnum):
    """Common datetime units with their LDML pattern strings."""

    year = "yyyy"
    quarter = "QQQ"
    month = "MM"
    week = "ww"
    week_of_month = "W"
    day = "dd"
    day_of_year = "DD"
    hour = "HH"
    minute = "mm"
    second = "ss"
    millisecond = "SSS"


@pydantic_dataclass
class Format:
    """Custom options for localized formatting."""

    decimal_notation: Literal["plain", "scientific", "percent"] | None = None
    """How to format decimals."""

    decimal_digits: int | None = None
    """How many digits to show after the decimal point."""

    decimal_min_digits: int | None = None
    """Minimum number of digits to show after the decimal point."""

    decimal_max_digits: int | None = None
    """Maximum number of digits to show after the decimal point."""

    decimal_group_separator: bool | None = None
    """Whether to use decimal group separators."""

    datetime_auto_format: Literal["short", "medium", "long"] | None = None
    """How to auto-format datetimes."""

    datetime_format: DtUnit | str | None = None
    """`ldml pattern`_ for formatting datetime objects.
    .. _ldml pattern: https://www.unicode.org/reports/tr35/tr35-dates.html#Date_Field_Symbol_Table
    """

    timedelta_auto_format: Literal[
        "narrow", "short", "medium", "long", "raw"
    ] | None = None
    """How to auto-format timedeltas."""

    timedelta_resolution: Literal[
        "year", "month", "week", "day", "hour", "minute", "second"
    ] | None = None
    """Unit to use for formatting timedeltas. Has no effect for auto-format 'raw'."""

    country_format: CountryScheme | None = None
    """How to format countries."""

    fallback_to_translation: bool = True
    """Whether to fallback to text translation if no other formatting method matches."""

    na_representation: str = ""
    """String to use for NaN or None values."""

    def merge(self, other: Self) -> Self:
        """Merge and override ``self`` with ``other``."""
        return cast(
            Self,
            parse_obj_as(
                Format,
                {
                    **asdict(self),
                    **{k: v for k, v in asdict(other).items() if v is not None},
                },
            ),
        )

    def spec(self, t: type) -> str:
        """Generate format-spec matching this format usign current locale."""
        return get_localization().format_spec(t, self)

    def auto_digits(
        self, series: Iterable[float], min_sig: int = 3, fixed: bool = True
    ) -> Self:
        """Copy and change options according to numbers in ``sample``.

        Change such that all given numbers have minimum ``min_sig`` significant digits.

        Args:
            series: Series of numbers to use for determining options.
            min_sig: Minimum number of significant digits.
            fixed: Whether to fix the number of digits after the decimal point.

        Returns:
            New instance with changed options.
        """
        minimum = min(abs(i) for i in series)
        min_int_digits = (
            (
                len(str(i))
                if (
                    i := int(
                        minimum if self.decimal_notation == "plain" else minimum * 100
                    )
                )
                != 0
                else 0
            )
            if self.decimal_notation in ("plain", "percent")
            else 1
        )
        min_after_comma = min_sig - min_int_digits
        return cast(
            Self,
            Format(
                **{
                    **asdict(self),
                    **dict(
                        decimal_min_digits=min_after_comma,
                        decimal_max_digits=(min_after_comma if fixed else None),
                    ),
                }
            ),
        )


@dataclass
class TextMatch:
    """Match the current rendered text instead of the original label."""

    regex: str | re.Pattern
    match_current: bool = False

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(self.__dict__)


TextOverrides: TypeAlias = dict[str | tuple[str, ...] | TextMatch, str]

K = TypeVar("K")
V = TypeVar("V")


def _merge_ordered_dicts(d: list[dict[K, V]]) -> dict[K, V]:
    d = d if len(d) > 0 else [{}]
    merged = d[0].copy()

    for di in d[1:]:
        for k, v in di.items():
            merged[k] = v

    return merged


@pydantic_dataclass
class Overrides:
    """Custom, language-dependent text overrides."""

    texts: TextOverrides | Mapping[str | None, TextOverrides] = field(
        default_factory=dict
    )
    """(Context-dependent) text overrides."""

    format: Format = field(default_factory=Format)
    """Overrides for formatting."""

    def get_texts(self) -> dict[str | None, TextOverrides]:
        """Return all text overrides."""
        return (
            cast(dict[str | None, TextOverrides], self.texts)
            if all(isinstance(v, dict) for v in self.texts.values())
            else {None: cast(TextOverrides, self.texts)}
        )

    def merge(self, other: Self) -> Self:
        """Merge and override ``self`` with ``other``."""
        self_texts = self.get_texts()
        other_texts = other.get_texts()
        return cast(
            Self,
            Overrides(
                texts={
                    ctx: _merge_ordered_dicts(
                        [self_texts.get(ctx, {}), other_texts.get(ctx, {})]
                    )
                    for ctx in set(self_texts.keys()).union(other_texts.keys())
                },
                format=self.format.merge(other.format),
            ),
        )


texts_file_path = Path(environ.get("TEXTS_FILE_PATH") or (Path.cwd() / "texts.yaml"))

file_overrides = None
if texts_file_path.is_file():
    with open(texts_file_path, encoding="utf-8") as f:
        file_overrides = parse_obj_as(dict[str, Overrides], load(f, Loader=CLoader))


def _get_default_locale() -> Locale:
    """Get default locale, falling back to English."""
    try:
        return Locale.parse(normalize(getlocale()[0] or "en_US"), sep="_")
    except UnknownLocaleError:
        return Locale("en", "US")


_ldml_to_posix = {
    "a": "%p",  # AM or PM
    "dd": "%d",  # day of month, padded
    "d": "%e",  # day of month, unpadded
    "DDD": "%j",  # day of year, padded
    "DD": "%-j",  # day of year, unpadded
    "D": "%-j",  # day of year, unpadded
    "EEEE": "%A",  # weekday
    "hh": "%I",  # 12-hour time, padded
    "h": "%-I",  # 12-hour time, unpadded
    "HH": "%H",  # 24-hour time, padded
    "H": "%-H",  # 24-hour time, unpadded
    "MMMM": "%B",  # month: name, full
    "MMM": "%b",  # month name, short
    "MM": "%m",  # month, padded
    "M": "%-m",  # month, unpadded
    "mm": "%M",  # minute, padded
    "m": "%-M",  # minute, unpadded
    "ss": "%S",  # second, padded
    "s": "%-S",  # second, unpadded
    "yyyy": "%Y",  # year, full
    "yy": "%y",  # year, two-digit
    "y": "%Y",  # year, full
    "Z": "%Z",  # timezone
    "zzzz": "%Z",  # timezone
    "z": "%Z",  # timezone
    "SSSSSS": "%f",  # microseconds
}


def _match_ldml_to_posix(ldml: str) -> tuple[int, str] | None:
    for k in _ldml_to_posix:
        pos = ldml.find(k)
        if pos != -1:
            return pos, k

    return None


def _ldml_to_posix_format(ldml: str) -> str:
    res = _match_ldml_to_posix(ldml)
    if res is None:
        return ldml

    pos, k = res

    return (
        _ldml_to_posix_format(ldml[:pos])
        + _ldml_to_posix[k]
        + _ldml_to_posix_format(ldml[pos + len(k) :])
    )


@cache.function(id_arg_subset=["lang", "text"])
def _cached_translate(lang: str, text: str, translator: GoogleTranslator | None) -> str:
    translator = (
        translator
        if translator is not None
        and translator.source == "en"
        and translator.target == lang
        else GoogleTranslator(source="en", target=lang)
    )
    return translator.translate(text)


@dataclass
class Localization:
    """Locale config, which can be used as a context manager to activate the locale."""

    loc: Locale | None = None
    """Locale to use for this localization. If None, use parent context's locale."""

    base: Overrides | None = None
    """Base overrides in English."""

    translations: Overrides | dict[Locale, Overrides] | None = None
    """Optional translation overrides for the current or other locales."""

    show_raw: bool = False
    """Whether to show raw function calls instead of localized text, for debugging."""

    def __machine_translate(self, text: str, locale: Locale | None = None) -> str:
        return _cached_translate(
            locale.language if locale is not None else self.locale.language,
            text,
            self.__translator,
        )

    @property
    def locale(self) -> Locale:
        """Return first locale found up the parent tree or default locale."""
        if self.loc is not None:
            return self.loc
        elif self.__parent is not None and self.__parent is not self:
            return self.__parent.locale

        return _get_default_locale()

    @staticmethod
    def _get_file_overrides(loc: Locale) -> tuple[Overrides, Overrides]:
        if file_overrides is None:
            return Overrides(), Overrides()

        base = file_overrides.get("base") or Overrides()
        translated = (
            file_overrides.get(str(loc))
            or file_overrides.get(str(loc.language))
            or Overrides()
        )
        return base, translated

    def get_overrides(self, locale: Locale) -> tuple[Overrides, Overrides]:
        """Return merger of all the parents' and self's overrides for given locale."""
        parent_base, parent_transl = (
            self.__parent.get_overrides(locale)
            if self.__parent is not None and self.__parent is not self
            else self._get_file_overrides(locale)
        )
        self_transl = (
            (self.translations.get(locale) or Overrides())
            if isinstance(self.translations, dict)
            else self.translations
            if self.translations is not None and self.locale == locale
            else Overrides()
        )

        return parent_base.merge(self.base or Overrides()), parent_transl.merge(
            self_transl
        )

    @property
    def overrides(self) -> Overrides:
        """Return merger of all the parents' overrides and self's overrides."""
        base, transl = self.get_overrides(self.locale)
        return base.merge(transl)

    def __post_init__(self):  # noqa: D105
        self.__parent = None
        self.__token = None
        self.__translator = GoogleTranslator(source="en", target=self.locale.language)

    def activate(self):
        """Set this the as current localization."""
        self.__parent = active_localization.get()
        self.__token = active_localization.set(self)
        setlocale(LC_ALL, str(self.locale))

    def __enter__(self):  # noqa: D105
        self.activate()
        return self

    def deactivate(self, *_):
        """Reset locale to previous value."""
        if self.__token is not None:
            active_localization.reset(self.__token)
        if self.__parent is not None:
            setlocale(LC_ALL, str(self.__parent.locale))

    def __exit__(self, *_):  # noqa: D105
        self.deactivate()

    # Define custom deepcopy to avoid error when copying `self.__token`.
    def __deepcopy__(self, memo: dict) -> Self:  # noqa: D105
        c = Localization(
            loc=self.loc,
            translations=self.translations,
            show_raw=self.show_raw,
        )
        memo[id(self)] = c
        return cast(Self, c)

    def __apply_template(
        self,
        tmpl: str,
        non_intl_args: Iterable,
        intl_args: Iterable,
        locale: Locale | None,
        context: str | None,
    ) -> str:
        loc_args = [
            self.text(a, locale=locale, context=context)
            if isinstance(a, str)
            else self.value(a, locale=locale)
            for a in intl_args
        ]

        return tmpl.format(*non_intl_args, *loc_args)

    def text(
        self,
        text: str,
        *extra_args: Any,
        label: str | None = None,
        context: str | None = None,
        locale: Locale | None = None,
    ) -> str:
        """Localize given text.

        Args:
            text: Text to localize.
            extra_args: Extra args to pass, if given label is a template string.
            label: Custom label for referencing this text in translation overrides.
            context: Context in which the label is used.
            locale: Locale to use for localization.

        Returns:
            Localized label.
        """
        label = label or text
        locale = locale or self.locale

        if self.show_raw:
            arg_str = (", " + ", ".join(extra_args)) if len(extra_args) > 0 else ""
            return (
                f"text('{text}'"
                + arg_str
                + (f", ctx={context}" if context is not None else "")
                + ")"
            )

        base, translation = self.get_overrides(locale or self.locale)

        all_texts = _merge_ordered_dicts(
            [
                {
                    k: (Locale("en", "US"), v)
                    for k, v in base.get_texts().get(None, {}).items()
                },
                {
                    k: (locale, v)
                    for k, v in translation.get_texts().get(None, {}).items()
                },
                {
                    k: (Locale("en", "US"), v)
                    for k, v in base.get_texts().get(context, {}).items()
                },
                {
                    k: (locale, v)
                    for k, v in translation.get_texts().get(context, {}).items()
                },
            ]
        )

        all_texts_grouped = {str: [], tuple: [], TextMatch: []}

        for t, g in groupby(all_texts.items(), lambda i: type(i[0])):
            if t in all_texts_grouped:
                all_texts_grouped[t] += list(g)

        all_texts_sorted = [
            *all_texts_grouped.get(str, []),
            *all_texts_grouped.get(tuple, []),
            *all_texts_grouped.get(TextMatch, []),
        ]

        rendered = text
        translated = False
        for search, (loc, replace) in all_texts_sorted:
            matched = (
                label == search
                if isinstance(search, str)
                else label in search
                if isinstance(search, tuple)
                else (
                    search.regex
                    if isinstance(search.regex, re.Pattern)
                    else re.compile(search.regex)
                ).match(rendered if search.match_current else text)
                is not None
                if isinstance(search, TextMatch)
                else False
            )
            if matched:
                rendered = self.__apply_template(
                    replace
                    if loc == locale
                    else self.__machine_translate(replace, locale),
                    [
                        rendered
                        if translated
                        else self.__machine_translate(rendered, locale)
                    ],
                    extra_args,
                    locale,
                    context,
                )
                if loc == locale:
                    translated = True

        return (
            rendered
            if translated
            else self.__apply_template(
                self.__machine_translate(rendered, locale),
                [],
                extra_args,
                locale,
                context,
            )
        )

    def value(
        self,
        v: Any,
        options: Format = Format(),
        context: str | None = None,
        locale: Locale | None = None,
    ) -> str:
        """Return localized string represention of given value.

        Args:
            v: Value to localize.
            options: Options for formatting.
            context: Context in which the value is used.
            locale: Locale to use for localization.

        Returns:
            Localized value.
        """
        if self.show_raw:
            return f"value({v})"

        locale = locale or self.locale
        base, translation = self.get_overrides(locale)
        options = base.merge(translation).format.merge(options)

        match (v):
            case _ if v is None or pd.isna(v):
                return options.na_representation
            case float() | Decimal() | int():
                fixed_digits = options.decimal_digits or options.decimal_min_digits or 0
                max_digits = options.decimal_digits or options.decimal_max_digits or 6
                group_sep = (
                    options.decimal_group_separator
                    if options.decimal_group_separator is not None
                    else False
                )

                match (options.decimal_notation):
                    case "percent":
                        return format_decimal(
                            v,
                            format=f"#,##0.{''.join(['0']*fixed_digits)}"
                            f"{''.join(['#']*(max_digits - fixed_digits))}%",
                            locale=locale,
                            group_separator=group_sep,
                        )
                    case "scientific":
                        return format_decimal(
                            v,
                            format=f"#,##0.{''.join(['0']*fixed_digits)}"
                            f"{''.join(['#']*(max_digits - fixed_digits))}E00",
                            locale=locale,
                            group_separator=group_sep,
                        )
                    case "plain" | _:
                        return format_decimal(
                            v,
                            format=f"#,##0.{''.join(['0']*fixed_digits)}"
                            f"{''.join(['#']*(max_digits - fixed_digits))}",
                            locale=locale,
                            group_separator=group_sep,
                        )
            case datetime() | pd.Timestamp():
                return format_datetime(
                    v,
                    options.datetime_format or options.datetime_auto_format or "medium",
                    locale=locale,
                )
            case date():
                return format_date(
                    v,
                    options.datetime_format or options.datetime_auto_format or "short",
                    locale=locale,
                )
            case time():
                return format_time(
                    v,
                    options.datetime_format or options.datetime_auto_format or "medium",
                    locale=locale,
                )
            case timedelta() | pd.Timedelta() if options.timedelta_auto_format != "raw":
                return format_timedelta(
                    v,
                    format=(
                        options.timedelta_auto_format
                        or options.datetime_auto_format
                        or "short"
                    ),
                    granularity=options.timedelta_resolution or "second",
                    threshold=0.99,
                    locale=locale,
                )
            case pd.Interval() if isinstance(v.left, pd.Timestamp) and isinstance(
                v.right, pd.Timestamp
            ):
                # Only do locale-aware formatting for datetime intervals.
                return format_interval(
                    v.left, v.right, skeleton=options.datetime_format, locale=locale
                )
            case Country():
                if (
                    options.country_format == GeoScheme.country_name
                    or options.country_format is None
                ):
                    return locale.territories.get(
                        str(v.to(GeoScheme.cc_iso2))
                    ) or self.text(
                        str(v.to(GeoScheme.country_name)),
                        context=context,
                        locale=locale,
                    )
                else:
                    return str(v.to(options.country_format))
            case _:
                return (
                    self.text(str(v), context=context, locale=locale)
                    if options.fallback_to_translation
                    else str(v)
                )

    def formatter(
        self, options: Format = Format(), locale: Locale | None = None
    ) -> Callable[[Any], str]:
        """Return a formatting function bound to this locale.

        Args:
            options: Options for formatting.
            locale: Locale to use for localization.

        Returns:
            Formatting function.
        """
        return partial(self.value, options=options, locale=locale)

    def format_spec(
        self, typ: type, options: Format = Format(), locale: Locale | None = None
    ) -> str:
        """Return locale-aware format-string for given type, if applicable.

        Args:
            typ: Type to get format-string for.
            options: Options for formatting.
            locale: Locale to use for localization.
        """
        locale = locale or self.locale
        base, translation = self.get_overrides(locale)
        options = base.merge(translation).format.merge(options)

        if issubclass(typ, int):
            match (options.decimal_notation):
                case "percent":
                    return ".0%"
                case "scientific":
                    return f".{options.decimal_min_digits or 6}e"
                case "plain" | _:
                    return "n"
        elif issubclass(typ, float | Decimal):
            match (options.decimal_notation):
                case "percent":
                    return f".{options.decimal_min_digits or 2}%"
                case "scientific":
                    return f".{options.decimal_min_digits or 6}e"
                case "plain" | _:
                    return f".{options.decimal_min_digits or 3}f"
        elif issubclass(typ, datetime | pd.Timestamp):
            opt = options.datetime_auto_format or "medium"
            return _ldml_to_posix_format(
                options.datetime_format
                or str.format(
                    locale.datetime_formats[opt],
                    locale.date_formats[opt],
                    locale.time_formats[opt],
                )
            )
        elif issubclass(typ, date):
            return _ldml_to_posix_format(
                options.datetime_format
                or locale.date_formats[options.datetime_auto_format or "medium"]
            )
        elif issubclass(typ, time):
            return _ldml_to_posix_format(
                options.datetime_format
                or locale.time_formats[options.datetime_auto_format or "medium"]
            )
        else:
            return ""


active_localization: ContextVar[Localization] = ContextVar(
    "active_localization", default=Localization()
)


def get_localization() -> Localization:
    """Get currently active localization."""
    return active_localization.get()


def iter_locales(
    locales: list[str],
    translations: dict[str, Overrides] | None = None,
    base: Overrides = Overrides(),
) -> Generator[Localization, None, None]:
    """Iterate over localizations for given locales w/ optional overrides.

    Args:
        locales: Locales to iterate over.
        translations: Optional translation overrides for the localizations.
        base: Optional base overrides in English.

    Returns:
        Generator of localizations.
    """
    for loc in locales:
        locz = Localization(
            loc=Locale.parse(loc, sep=("_" if "_" in loc else "-")),
            base=base,
            translations={
                Locale.parse(k, sep=("_" if "_" in k else "-")): v
                for k, v in translations.items()
            }
            if translations is not None
            else None,
        )
        with locz:
            yield locz
