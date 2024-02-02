"""Simple & semi-automatic intl. + localization of data analysis functions."""

from collections.abc import Callable, Generator, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial
from locale import LC_ALL, getlocale, normalize, setlocale
from os import environ
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar, cast

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
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Self
from yaml import CLoader, load

from py_research.caching import get_cache
from py_research.enums import StrEnum
from py_research.geo import Country, CountryScheme, GeoScheme
from py_research.hashing import gen_int_hash

cache = get_cache()

K = TypeVar("K")
V = TypeVar("V")
P = ParamSpec("P")


def _merge_ordered_dicts(d: list[dict[K, V]]) -> dict[K, V]:
    d = d if len(d) > 0 else [{}]
    merged = d[0].copy()

    for di in d[1:]:
        for k, v in di.items():
            merged[k] = v

    return merged


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

    timedelta_auto_format: (
        Literal["narrow", "short", "medium", "long", "raw"] | None
    ) = None
    """How to auto-format timedeltas."""

    timedelta_resolution: (
        Literal["year", "month", "week", "day", "hour", "minute", "second"] | None
    ) = None
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
            Format(
                **{
                    **asdict(self),
                    **{k: v for k, v in asdict(other).items() if v is not None},
                }
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
class Args:
    """Match for args of a text template."""

    values: Any = None
    all: bool = False

    def __hash__(self) -> int:  # noqa: D105
        return gen_int_hash(asdict(self))

    def match(self, args: tuple) -> float:
        """Check whether given args match this match."""
        if self.all:
            return 1

        values = self.values if isinstance(self.values, tuple | list) else [self.values]

        if len(args) != len(values):
            return 0

        matchers = [v if isinstance(v, set) else {v} for v in values]

        return max(
            (1 / len(matcher) if arg in matcher else 0)
            for arg, matcher in zip(args, matchers)
        )


@dataclass
class Template:
    """Override for a text template."""

    replace: str | None = None
    """Replace template with this template string."""

    wrap: str | None = None
    """Wrap template with this template string."""

    args: tuple | dict[int, Any] | None = None
    """Replace (some) args."""

    def apply_tmpl(self, tmpl: str) -> str | None:
        """Apply this template to given template string and args."""
        return (
            self.wrap.format(tmpl)
            if self.wrap is not None
            else self.replace if self.replace is not None else None
        )

    def apply_args(self, args: tuple) -> tuple | None:
        """Apply this template to given args."""
        return (
            self.args
            if isinstance(self.args, tuple)
            else (
                tuple(self.args[i] if i in self.args else a for i, a in enumerate(args))
                if isinstance(self.args, dict)
                else None
            )
        )


@pydantic_dataclass
class Overrides:
    """Custom, language-dependent text overrides."""

    templates: Mapping[
        str, str | Template | dict[str | tuple | Args, str | Template]
    ] = field(default_factory=dict)
    """Template overrides by name."""

    vocabulary: dict[str, str] = field(default_factory=dict)
    """Term overrides."""

    format: Format = field(default_factory=Format)
    """Overrides for formatting."""

    def merge(self, other: Self) -> Self:
        """Merge and override ``self`` with ``other``."""
        texts = {}
        for name in set(self.templates.keys()).union(other.templates.keys()):
            self_texts = self.templates.get(name, {})
            other_texts = other.templates.get(name, {})

            texts[name] = _merge_ordered_dicts(
                [
                    (
                        self_texts
                        if isinstance(self_texts, dict)
                        else {
                            Args(all=True): (
                                Template(self_texts)
                                if isinstance(self_texts, str)
                                else self_texts
                            )
                        }
                    ),
                    (
                        other_texts
                        if isinstance(other_texts, dict)
                        else {
                            Args(all=True): (
                                Template(other_texts)
                                if isinstance(other_texts, str)
                                else other_texts
                            )
                        }
                    ),
                ]
            )

        return cast(
            Self,
            Overrides(
                vocabulary=_merge_ordered_dicts([self.vocabulary, other.vocabulary]),
                templates=texts,
                format=self.format.merge(other.format),
            ),
        )


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

    loc: Locale | str | None = None
    """Locale to use for this localization. If None, use parent context's locale."""

    overrides: Overrides | dict[Locale, Overrides] | dict[str, Overrides] | None = None
    """Optional translation overrides for the current or other locales."""

    show_raw: bool = False
    """Whether to show raw function calls instead of localized text, for debugging."""

    def _machine_translate(self, text: str, locale: Locale | None = None) -> str:
        return _cached_translate(
            locale.language if locale is not None else self.locale.language,
            text,
            self.__translator,
        )

    @property
    def locale(self) -> Locale:
        """Return first locale found up the parent tree or default locale."""
        if self.loc is not None:
            return self.loc if isinstance(self.loc, Locale) else Locale.parse(self.loc)
        elif self.__parent is not None and self.__parent is not self:
            return self.__parent.locale

        return _get_default_locale()

    @property
    def override_dict(self) -> dict[Locale, Overrides]:
        """Return dict of all overrides, keyed by locale string."""
        return (
            {
                (
                    Locale.parse(k, sep=("_" if "_" in k else "-"))
                    if isinstance(k, str)
                    else k
                ): v
                for k, v in self.overrides.items()
            }
            if isinstance(self.overrides, Mapping)
            else {self.locale: self.overrides or Overrides()}
        )

    def get_overrides(self, locale: Locale) -> Overrides:
        """Return merger of all the parents' and self's overrides for given locale."""
        parent_overrd = (
            self.__parent.get_overrides(locale)
            if self.__parent is not None and self.__parent is not self
            else Overrides()
        )
        self_overrd = self.override_dict.get(locale, Overrides())

        return parent_overrd.merge(self_overrd)

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
            overrides=self.overrides,
            show_raw=self.show_raw,
        )
        memo[id(self)] = c
        return cast(Self, c)

    def term(
        self,
        term: str,
        locale: Locale | None = None,
    ) -> str:
        """Localize given term.

        Args:
            term: Term to localize.
            locale:
                Locale to use for localization.
                Defaults to the currently activated locale.

        Returns:
            Localized term.
        """
        if self.show_raw:
            return f"term('{term}')"

        base = self.get_overrides(Locale("en", "US"))
        translation = self.get_overrides(locale or self.locale)

        if term in translation.vocabulary:
            return translation.vocabulary[term]
        elif term in base.vocabulary:
            return self._machine_translate(base.vocabulary[term], locale)

        return self._machine_translate(term, locale)

    @staticmethod
    def _get_template(overrides: Overrides, name: str, args: tuple) -> Template:
        texts = overrides.templates.get(name, {})
        texts = texts if isinstance(texts, dict) else {Args(all=True): texts}

        matches: dict[float, str | Template] = {}
        for match, tmpl in texts.items():
            matcher = (
                match
                if isinstance(match, Args)
                else Args(set(match) if isinstance(match, tuple) else match)
            )
            score = matcher.match(args)
            if score > 0:
                matches[score] = tmpl

        template = matches[max(matches.keys())] if len(matches) > 0 else None

        return (
            template
            if isinstance(template, Template)
            else Template(template) if template is not None else Template()
        )

    def text(
        self,
        context: str,
        content: str | None,
        *args: Any,
        locale: Locale | None = None,
    ) -> str:
        """Localize given text.

        Args:
            context: Label indentifying the context of the text to localize.
            content: Text to localize.
            args: Extra args to pass, if given text content is a template string.
            locale:
                Locale to use for localization.
                Defaults to the currently activated locale.

        Returns:
            Localized text.
        """
        if self.show_raw:
            arg_str = (", " + ", ".join(args)) if len(args) > 0 else ""
            return f"text('{content}'" + arg_str + f", context={context})"

        base = self.get_overrides(Locale("en", "US"))
        translation = self.get_overrides(locale or self.locale)

        tmpl = (
            self._machine_translate(content, locale) if content is not None else "{0}"
        )
        tmpl_args = [self.value(a, locale=locale) for a in args]

        transl_template = self._get_template(translation, context, args)
        base_template = self._get_template(base, context, args)

        if transl_template.replace is not None:
            tmpl = transl_template.replace
        elif base_template.replace is not None:
            tmpl = self._machine_translate(base_template.replace, locale)

        if transl_template.wrap is not None:
            tmpl = transl_template.wrap.format(tmpl)
        elif base_template.wrap is not None:
            tmpl = self._machine_translate(base_template.wrap, locale).format(tmpl)

        if transl_template.args is not None:
            if isinstance(transl_template.args, tuple):
                tmpl_args = transl_template.args
            elif isinstance(transl_template.args, dict):
                tmpl_args = {
                    i: transl_template.args[i] if i in transl_template.args else a
                    for i, a in enumerate(tmpl_args)
                }
        elif base_template.args is not None:
            if isinstance(base_template.args, tuple):
                tmpl_args = [self.value(a, locale=locale) for a in base_template.args]
            elif isinstance(base_template.args, dict):
                tmpl_args = {
                    i: (
                        self.value(base_template.args[i], locale=locale)
                        if i in base_template.args
                        else a
                    )
                    for i, a in enumerate(tmpl_args)
                }

        return tmpl.format(*tmpl_args)

    def label(
        self,
        label: str,
        context: str | None = None,
        locale: Locale | None = None,
    ) -> str:
        """Localize given label.

        Args:
            label: Label to localize.
            context: Context in which the label is used.
            locale:
                Locale to use for localization.
                Defaults to the currently activated locale.

        Returns:
            Localized label.
        """
        if self.show_raw:
            return f"label('{label}', context={context})"

        return (
            self.text(context, None, label, locale=locale)
            if context is not None
            else self.term(label, locale=locale)
        )

    def value(
        self,
        v: Any,
        options: Format = Format(),
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

        base = self.get_overrides(Locale("en", "US"))
        translation = self.get_overrides(locale or self.locale)

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
                    ) or self.term(
                        str(v.to(GeoScheme.country_name)),
                        locale=locale,
                    )
                else:
                    return str(v.to(options.country_format))
            case _:
                return (
                    self.term(str(v), locale=locale)
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
            Formatting function for generating localized values.
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

        base = self.get_overrides(Locale("en", "US"))
        translation = self.get_overrides(locale or self.locale)

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
    locales: list[str] | None = None,
    overrides: dict[str, Overrides] | None = None,
) -> Generator[Localization, None, None]:
    """Iterate over localizations for given locales w/ optional overrides.

    Args:
        locales:
            Locales to iterate over.
            If None, iterates over all locales, for which overrides are defined.
        overrides: Optional text overrides for the localizations.

    Returns:
        Generator of localizations.
    """
    if locales is None:
        current_loc = get_localization()
        locales = [str(k) for k in current_loc.override_dict.keys()]
        if overrides is not None:
            ovrd_locz = [str(k) for k in overrides.keys()]
            locales = list(set(locales).union(ovrd_locz))

    for loc in locales:
        locz = Localization(loc=loc, overrides=overrides)
        with locz:
            yield locz


default_file_path = Path(environ.get("TEXTS_FILE_PATH") or (Path.cwd() / "texts.yaml"))
"""Default path to text overrides file."""


def load_from_file(path: Path | str | None = None) -> Path | None:
    """Load text overrides from file.

    Args:
        path: Path to file. If None, the default file path is used.

    Returns:
        Path to the loaded file, if not given as argument.
    """
    res_path = path or default_file_path

    with open(default_file_path, encoding="utf-8") as f:
        override_dict = load(f, Loader=CLoader)
        assert isinstance(override_dict, dict)
        overrides = {
            str(loc): Overrides(**overrides) for loc, overrides in override_dict.items()
        }
        active_localization.set(Localization(overrides=overrides))

    if path is None:
        return Path(res_path)
