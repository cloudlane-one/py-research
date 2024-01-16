"""Simple & semi-automatic intl. + localization of data analysis functions."""

import re
from collections.abc import Callable, Generator, Iterable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial
from locale import LC_ALL, getlocale, normalize, setlocale
from numbers import Rational
from os import environ
from pathlib import Path
from typing import (
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    cast,
    overload,
    runtime_checkable,
)

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
from deepmerge import always_merger
from pydantic import ConfigDict, parse_obj_as
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Self
from yaml import CLoader, load

from py_research.caching import get_cache
from py_research.geo import Country, CountryScheme, GeoScheme

cache = get_cache()


P = ParamSpec("P")


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

    datetime_format: Literal["short", "medium", "long"] | None = None
    """How to format datetimes."""

    country_format: CountryScheme | None = None
    """How to format countries."""

    def merge(self, other: Self) -> Self:
        """Merge and override ``self`` with ``other``."""
        return cast(
            Self,
            parse_obj_as(
                Format,
                always_merger.merge(
                    asdict(self),
                    {k: v for k, v in asdict(other).items() if v is not None},
                ),
            ),
        )

    def spec(self, t: type) -> str:
        """Generate format-spec matching this format usign current locale."""
        return get_localization().format_spec(t, self)

    def auto_digits(
        self, sample: Iterable[Rational], min_sig: int = 3, fixed: bool = True
    ) -> Self:
        """Copy and change options according to numbers in ``sample``.

        Change such that all given numbers have minimum ``min_sig`` significant digits.

        Args:
            sample: Sample of numbers to use for determining the number of digits.
            min_sig: Minimum number of significant digits.
            fixed: Whether to fix the number of digits after the decimal point.

        Returns:
            New instance with changed options.
        """
        min_int_digits = (
            min(
                len(str(i))
                if (i := int(num if self.decimal_notation == "plain" else num * 100))
                != 0
                else 0
                for num in sample
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


@runtime_checkable
class DynamicMessage(Protocol[P]):
    """Dynamically generated message with typed args and a name."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> str:  # noqa: D102
        ...

    @property
    def __name__(self) -> str:  # noqa: D105
        ...


KwdOverride: TypeAlias = (
    dict[str, str] | Format | Callable[[Any, dict[str | int, Any]], str | None]
)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Template:
    """Custom override for formatted message."""

    scaffold: str | DynamicMessage
    """Template string or function to use for formatting."""

    given_params: dict[str, Any] = field(default_factory=dict)
    """Parameters given to the template."""

    param_overrides: dict[str, KwdOverride] = field(default_factory=dict)
    """Overrides for parameters given to the template."""

    context: str | dict[str, str | None] | None = None
    """Context in which the template is used."""


LabelOverride: TypeAlias = (
    dict[str, str | Callable[[str], str]]
    | tuple[tuple[str, ...], str | Callable[[str], str]]
)
"""Override for a label, either as a dict or a tuple of (search, replace).""" ""

MessageOverride: TypeAlias = (
    dict[str, str | Template] | tuple[re.Pattern | tuple[str, ...], str | Template]
)
"""Override for a message, either as a dict or a tuple of (search, replace)."""

RegexReplace: TypeAlias = tuple[re.Pattern | str, str | Callable[[str], str]]
"""Regex search-replace.""" ""


@pydantic_dataclass
class Overrides:
    """Custom, language-dependent text overrides."""

    labels: list[LabelOverride] | dict[str | None, list[LabelOverride]] = field(
        default_factory=list
    )
    """(Context-dependent) overrides for labels."""

    messages: list[MessageOverride] = field(default_factory=list)
    """Overrides for messages."""

    format: Format = field(default_factory=Format)
    """Overrides for formatting."""

    def merge(self, other: Self) -> Self:
        """Merge and override ``self`` with ``other``."""
        return cast(
            Self,
            Overrides(
                labels=always_merger.merge(
                    self.labels
                    if isinstance(self.labels, dict)
                    else {None: self.labels},
                    other.labels
                    if isinstance(other.labels, dict)
                    else {None: other.labels},
                ),
                messages=always_merger.merge(self.messages, other.messages),
                format=self.format.merge(other.format),
            ),
        )

    def get_labels(
        self, context: str | None = None
    ) -> dict[str | None, list[LabelOverride]]:
        """Get all labels applicable to given context. List default ctx first.

        Args:
            context: Context to get translations for.

        Returns:
            Labels for given context.
        """
        return (
            {
                None: self.labels.get(None) or [],
                **{context: self.labels.get(context) or []},
            }
            if isinstance(self.labels, dict)
            else {None: self.labels}
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
    "d": "%e",  # day of month, unpadded
    "dd": "%d",  # day of month, padded
    "D": "%j",  # day of year
    "EEEE": "%A",  # weekday
    "h": "%-I",  # 12-hour time, unpadded
    "hh": "%I",  # 12-hour time, padded
    "H": "%-H",  # 24-hour time, unpadded
    "HH": "%H",  # 24-hour time, padded
    "M": "%-m",  # month, unpadded
    "MM": "%m",  # month, padded
    "MMM": "%b",  # month name, short
    "MMMM": "%B",  # month: name, full
    "m": "%-M",  # minute, unpadded
    "mm": "%M",  # minute, padded
    "s": "%-S",  # second, unpadded
    "ss": "%S",  # second, padded
    "y": "%Y",  # year, full
    "yy": "%y",  # year, two-digit
    "yyyy": "%Y",  # year, full
    "Z": "%Z",  # timezone
    "z": "%Z",  # timezone
    "zzzz": "%Z",  # timezone
}


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

    local_locale: Locale | None = None
    """Locale to use for this localization. If None, use parent context's locale."""

    local_overrides: Overrides | dict[Locale, Overrides] | None = None
    """Optional overrides for this localization."""

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
        if self.local_locale is not None:
            return self.local_locale
        elif self.__parent is not None and self.__parent is not self:
            return self.__parent.locale

        return _get_default_locale()

    def get_overrides(self, locale: Locale) -> Overrides:
        """Return merger of all the parents' and self's overrides for given locale."""
        parent_overrides = (
            self.__parent.get_overrides(locale)
            if self.__parent is not None and self.__parent is not self
            else file_overrides[str(locale)]
            if file_overrides is not None and str(locale) in file_overrides
            else file_overrides[str(locale.language)]
            if file_overrides is not None and str(locale.language) in file_overrides
            else Overrides()
        )
        self_overrides = (
            (self.local_overrides.get(locale) or Overrides())
            if isinstance(self.local_overrides, dict)
            else self.local_overrides
            if self.local_overrides is not None and self.locale == locale
            else Overrides()
        )

        return parent_overrides.merge(self_overrides)

    @property
    def overrides(self) -> Overrides:
        """Return merger of all the parents' overrides and self's overrides."""
        return self.get_overrides(self.locale)

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
            local_locale=self.local_locale,
            local_overrides=self.local_overrides,
            show_raw=self.show_raw,
        )
        memo[id(self)] = c
        return cast(Self, c)

    def __get_label(
        self,
        label: str,
        context: str | None = None,
        locale: Locale | None = None,
    ) -> tuple[str, bool, bool]:
        matched = False

        overrides = self.get_overrides(locale or self.locale)
        translations = overrides.get_labels(context)

        matched_ctx = False
        sub = label
        for ctx, transl in translations.items():
            for override in transl:
                replace = None

                if isinstance(override, dict):
                    replace = override.get(label)
                elif label in override[0]:
                    replace = override[1]

                if replace is not None:
                    if isinstance(replace, str):
                        sub = replace.format(sub)
                    else:
                        sub = replace(sub)

                    matched = True
                    if ctx is not None:
                        matched_ctx = True

        return sub, matched, matched_ctx

    def __apply_template(
        self,
        tmpl: Template,
        combined_args: dict[str | int, Any],
        locale: Locale | None = None,
    ) -> str:
        def _get_ctx(label: str) -> str | None:
            return (
                tmpl.context.get(label)
                if isinstance(tmpl.context, dict)
                else tmpl.context
            )

        intl_args = {}
        for k, v in combined_args.items():
            if k in tmpl.param_overrides:
                o = tmpl.param_overrides[k]
                if isinstance(o, Format):
                    intl_args[k] = self.value(v, o)
                elif isinstance(o, dict):
                    intl_args[k] = o[k]
                else:
                    intl_args[k] = o(v, combined_args)
            else:
                intl_args[k] = None

        # Replace defaulted with defaults.
        intl_args = {
            k: v
            if v is not None or combined_args[k] is None
            else self.label(
                combined_args[k], locale=locale, context=_get_ctx(combined_args[k])
            )
            if isinstance(combined_args[k], str)
            else self.value(combined_args[k], locale=locale)
            for k, v in intl_args.items()
        }

        message = (
            tmpl.scaffold.format if isinstance(tmpl.scaffold, str) else tmpl.scaffold
        )

        return message(
            *(v for k, v in intl_args.items() if isinstance(k, int)),
            **{k: v for k, v in intl_args.items() if isinstance(k, str)},
        )

    def __get_message_template(
        self,
        search: str,
        combined_args: dict[str | int, Any] | None = None,
        locale: Locale | None = None,
    ) -> Template | None:
        combined_args = combined_args or {}

        overrides = self.overrides if locale is None else self.get_overrides(locale)
        translations = overrides.messages

        matched = None
        for transl in translations:
            replace = None

            if isinstance(transl, dict):
                replace = transl.get(search)
            else:
                if isinstance(transl[0], tuple):
                    if search in transl[0]:
                        replace = transl[1]
                else:
                    m = transl[0].search(search)
                    if m is not None:
                        replace = transl[1]

            if replace is not None:
                tmpl = Template(replace) if isinstance(replace, str) else replace
                if len(tmpl.given_params) == 0 or all(
                    (
                        (
                            v(combined_args[k])
                            if isinstance(v, Callable)
                            else combined_args[k] == v
                        )
                        for k, v in tmpl.given_params.items()
                    )
                ):
                    matched = replace

        return Template(matched) if isinstance(matched, str) else matched

    def label(
        self, label: str, context: str | None = None, locale: Locale | None = None
    ) -> str:
        """Localize given text label.

        Args:
            label: Label to localize.
            context: Context in which the label is used.
            locale: Locale to use for localization.

        Returns:
            Localized label.
        """
        if self.show_raw:
            return (
                f"label('{label}{f', ctx={context}' if context is not None else ''}')"
            )

        transl, matched, matched_ctx = self.__get_label(
            label, context=context, locale=locale
        )

        if locale != Locale("en", "US") and (not matched or not matched_ctx):
            transl_en, _, matched_ctx_en = self.__get_label(
                label, context=context, locale=Locale("en", "US")
            )
            if not matched or (not matched_ctx and matched_ctx_en):
                transl = self.__machine_translate(transl_en, locale)

        return transl

    @overload
    def message(self, msg: str | Template, *args: Any, **kwargs: Any) -> str:
        ...

    @overload
    def message(self, msg: DynamicMessage[P], *args: P.args, **kwargs: P.kwargs) -> str:
        ...

    def message(
        self, msg: str | DynamicMessage[P] | Template, *args: Any, **kwargs: Any
    ) -> str:
        """Localize given text message.

        Args:
            msg: Message to localize.
            args: Positional arguments to pass to the message.
            kwargs: Keyword arguments to pass to the message.

        Returns:
            Localized message.
        """
        tpl = msg if isinstance(msg, Template) else Template(msg)
        name = tpl.scaffold if isinstance(tpl.scaffold, str) else tpl.scaffold.__name__

        if self.show_raw:
            kwd_str = (
                (", " + ", ".join(f"{k}={v}" for k, v in kwargs.items()))
                if len(kwargs) > 0
                else ""
            )
            return f"msg('{name}'{kwd_str})"

        combined_args = {**dict(enumerate(args)), **kwargs}

        matched_tpl = self.__get_message_template(name, combined_args)

        if self.locale != Locale("en", "US") and not matched_tpl:
            matched_tpl = self.__get_message_template(
                name, combined_args, locale=Locale("en", "US")
            )

            return self.__machine_translate(
                self.__apply_template(
                    tpl or matched_tpl, combined_args, locale=Locale("en", "US")
                )
            )

        return self.__apply_template(tpl or matched_tpl, combined_args)

    def value(
        self, v: Any, options: Format = Format(), locale: Locale | None = None
    ) -> str:
        """Return localized string represention of given value.

        Args:
            v: Value to localize.
            options: Options for formatting.
            locale: Locale to use for localization.

        Returns:
            Localized value.
        """
        if self.show_raw:
            return f"value({v})"

        locale = locale or self.locale
        options = self.get_overrides(locale).format.merge(options)

        match (v):
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
                    v, options.datetime_format or "short", locale=locale
                )
            case date():
                return format_date(v, options.datetime_format or "short", locale=locale)
            case time():
                return format_time(v, options.datetime_format or "short", locale=locale)
            case timedelta() | pd.Timedelta():
                return format_timedelta(
                    v, format=(options.datetime_format or "short"), locale=locale
                )
            case pd.Interval():
                return format_interval(v.left, v.right, locale=locale)
            case Country():
                if (
                    options.country_format == GeoScheme.country_name
                    or options.country_format is None
                ):
                    return locale.territories.get(
                        str(v.to(GeoScheme.cc_iso2))
                    ) or self.label(
                        str(v.to(GeoScheme.country_name)), context="country_name"
                    )
                else:
                    return str(v.to(options.country_format))
            case _:
                return str(v)

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
        """Return locale-aware format-string for given type."""
        locale = locale or self.locale
        options = self.get_overrides(locale).format.merge(options)

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
            opt = options.datetime_format or "medium"
            date_format = _ldml_to_posix[locale.date_formats[opt]]
            time_format = _ldml_to_posix[locale.time_formats[opt]]
            return str.format(locale.datetime_formats[opt], date_format, time_format)
        elif issubclass(typ, date):
            return _ldml_to_posix[
                locale.date_formats[options.datetime_format or "medium"]
            ]
        elif issubclass(typ, time):
            return _ldml_to_posix[
                locale.time_formats[options.datetime_format or "medium"]
            ]
        else:
            return ""


active_localization: ContextVar[Localization] = ContextVar(
    "active_localization", default=Localization()
)


def get_localization() -> Localization:
    """Get currently active localization."""
    return active_localization.get()


def iter_locales(
    locales: list[str], overrides: dict[str, Overrides] | None = None
) -> Generator[Localization, None, None]:
    """Iterate over localizations for given locales w/ optional overrides.

    Args:
        locales: Locales to iterate over.
        overrides: Optional overrides for the localizations.

    Returns:
        Generator of localizations.
    """
    for loc in locales:
        locz = Localization(
            local_locale=Locale.parse(loc, sep=("_" if "_" in loc else "-")),
            local_overrides={
                Locale.parse(k, sep=("_" if "_" in k else "-")): v
                for k, v in overrides.items()
            }
            if overrides is not None
            else None,
        )
        with locz:
            yield locz
