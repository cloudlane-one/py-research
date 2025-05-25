"""Default converters for common types."""

from __future__ import annotations

import json
import pickle
import xml.etree.ElementTree as xml  # noqa: N813
from collections.abc import Set
from typing import Any, BinaryIO, Literal, TextIO, cast

import numpy as np
import pandas as pd
import polars as pl
import pydantic as pyd
import sqlalchemy as sqla
import sqlparse
import yaml
from typing_extensions import TypeVar

from py_research.files import File
from py_research.reflect.types import SingleTypeDef, has_type, is_subtype

from .converters import Converter

converters: dict[SingleTypeDef, type[Converter]] = {}


def get_converter[U, T](
    instance: type[U] | U, targets: Set[T | SingleTypeDef[T]]
) -> type[Converter[U, T]] | None:
    """Get the converter for a given type or instance."""
    obj_type = instance if isinstance(instance, type) else type(instance)

    matching_conv = [
        c
        for t, c in converters.items()
        if is_subtype(obj_type, t) and len(c.conv_types().match_targets(targets)) > 0
    ]

    if len(matching_conv) == 0:
        return None

    return matching_conv[0]


# Simple, stringifiable types

type TextFile = File[TextIO, Literal["text/plain", "text/csv"]]
TF = TypeVar(
    "TF",
    bound=TextFile,
    default=TextFile,
)

type StrScalar = str | int | float | complex | bool | None
type Stringifiable = StrScalar | list[StrScalar]


type JSONFile = File[TextIO, Literal["application/yaml"]] | File[
    TextIO, Literal["application/json"]
]
JF = TypeVar(
    "JF",
    bound=JSONFile,
    default=JSONFile,
)

# JSON / YAML dict

type JSONScalar = str | int | float | bool | None
type JSONDict = dict[str, JSONDict | list[JSONScalar]]

J = TypeVar("J", bound=JSONDict, default=JSONDict)


class JSONConv(Converter[JSONDict, JSONFile]):
    """Converter for JSON and YAML dictionaries."""

    @classmethod
    def parse(
        cls,
        source: JSONFile,
        realm: Any,
        annotation: SingleTypeDef[J] | None = None,
    ) -> J:
        """Parse a dictionary from a JSON or YAML file."""
        dct: dict
        if source.mime == "application/json":
            dct = json.load(source.content)
        elif source.mime == "application/yaml":
            dct = yaml.load(source.content, Loader=yaml.CLoader)
        else:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

        if annotation is not None:
            assert has_type(
                dct, annotation
            ), f"Parsed data does not match expected type: {annotation}"

        return cast(J, dct)

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, JF]],
        instance: J,
        targets: Set[JF | SingleTypeDef[JF]],
        realm: Any,
        annotation: SingleTypeDef[J] | None = None,
    ) -> JF:
        """Convert a dictionary to a JSON or YAML file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            if not isinstance(target, File):
                continue

            if target.mime == "application/json":
                target.content.write(json.dumps(instance, indent=2))
                return cast(JF, target)
            elif target.mime == "application/yaml":
                yaml.dump(instance, allow_unicode=True, stream=target.content)
                return cast(JF, target)

        raise ValueError(
            "No suitable target found for converting dictionary to JSON or YAML."
        )


converters[JSONDict] = JSONConv


# XML etree

XMLFile = File[TextIO, Literal["text/xml"]] | File[TextIO, Literal["application/xml"]]

XMLF = TypeVar(
    "XMLF",
    bound=XMLFile,
    default=XMLFile,
)
ET = TypeVar("ET", bound=xml.ElementTree, default=xml.ElementTree)


class XMLConv(Converter[ET, XMLFile]):
    """Converter for XML ElementTree."""

    @classmethod
    def parse(
        cls,
        source: XMLFile,
        realm: Any,
        annotation: SingleTypeDef[ET] | None = None,
    ) -> ET:
        """Parse an XML ElementTree from a file."""
        if source.mime not in {"application/xml", "text/xml"}:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

        tree = xml.parse(source.content)
        if annotation is not None and not has_type(tree, annotation):
            raise ValueError(f"Parsed data does not match expected type: {annotation}")

        return cast(ET, tree)

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, XMLF]],
        instance: ET,
        targets: Set[XMLF | SingleTypeDef[XMLF]],
        realm: Any,
        annotation: SingleTypeDef[ET] | None = None,
    ) -> XMLF:
        """Convert an XML ElementTree to a file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            if not isinstance(target, File):
                continue

            if target.mime in {"application/xml", "text/xml"}:
                instance.write(target.content)
                return cast(XMLF, target)

        raise ValueError("No suitable target found for converting XML ElementTree.")


converters[xml.ElementTree] = XMLConv

# Pydantic models


class PydConv(Converter[pyd.BaseModel, JSONFile]):
    """Converter for Pydantic models."""

    @classmethod
    def parse(
        cls,
        source: JSONFile,
        realm: Any,
        annotation: SingleTypeDef[pyd.BaseModel] | None = None,
    ) -> pyd.BaseModel:
        """Parse a Pydantic model from a JSON or YAML file."""
        if source.mime == "application/json":
            data = json.load(source.content)
        elif source.mime == "application/yaml":
            data = yaml.load(source.content, Loader=yaml.CLoader)
        else:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

        if annotation is None or not has_type(data, annotation):
            raise ValueError(f"Data does not match expected type: {annotation}")

        assert (
            annotation is not None
        ), "Annotation must be provided for Pydantic model parsing."
        adapter = pyd.TypeAdapter(annotation)
        return adapter.validate_python(data)

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, JF]],
        instance: pyd.BaseModel,
        targets: Set[JF | SingleTypeDef[JF]],
        realm: Any,
        annotation: SingleTypeDef[pyd.BaseModel] | None = None,
    ) -> JF:
        """Convert a Pydantic model to a JSON or YAML file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            if not isinstance(target, File):
                continue

            if target.mime == "application/json":
                json.dump(instance.model_dump(), target.content, indent=2)
                return cast(JF, target)
            elif target.mime == "application/yaml":
                yaml.dump(
                    instance.model_dump(), allow_unicode=True, stream=target.content
                )
                return cast(JF, target)

        raise ValueError("No suitable target found for converting Pydantic model.")


converters[pyd.BaseModel] = PydConv

# DataFrames

type DfFile = File[BinaryIO, Literal["application/vnd.ms-excel"]] | File[
    BinaryIO, Literal["application/vnd.apache.parquet"]
] | File[TextIO, Literal["text/csv"]]

DfF = TypeVar("DfF", bound=DfFile, default=DfFile)

# Pandas


class PdConv(Converter[pd.DataFrame, DfFile]):
    """Converter for Pandas DataFrames."""

    @classmethod
    def parse(
        cls,
        source: DfFile,
        realm: Any,
        annotation: SingleTypeDef[pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Parse a DataFrame from a CSV or Parquet file."""
        if source.mime == "text/csv":
            return pd.read_csv(source.content)
        elif source.mime == "application/vnd.apache.parquet":
            return pd.read_parquet(source.content)
        elif source.mime == "application/vnd.ms-excel":
            return pd.read_excel(source.content)
        else:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, DfF]],
        instance: pd.DataFrame,
        targets: Set[DfF | SingleTypeDef[DfF]],
        realm: Any,
        annotation: SingleTypeDef[pd.DataFrame] | None = None,
    ) -> DfF:
        """Convert a DataFrame to a CSV or Parquet file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            if not isinstance(target, File):
                continue

            if target.mime == "text/csv":
                instance.to_csv(target.content, index=False)
                return cast(DfF, target)
            elif target.mime == "application/vnd.apache.parquet":
                assert isinstance(target.content, BinaryIO)
                instance.to_parquet(target.content)
                return cast(DfF, target)
            elif target.mime == "application/vnd.ms-excel":
                instance.to_excel(target.content, index=False)
                return cast(DfF, target)

        raise ValueError("No suitable target found for converting DataFrame.")


converters[pd.DataFrame] = PdConv

# Polars


class PlConv(Converter[pl.DataFrame, DfFile]):
    """Converter for Polars DataFrames."""

    @classmethod
    def parse(
        cls,
        source: DfFile,
        realm: Any,
        annotation: SingleTypeDef[pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Parse a DataFrame from a CSV or Parquet file."""
        if source.mime == "text/csv":
            return pl.read_csv(source.content)
        elif source.mime == "application/vnd.apache.parquet":
            return pl.read_parquet(source.content)
        elif source.mime == "application/vnd.ms-excel":
            return pl.read_excel(source.content)
        else:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, DfF]],
        instance: pl.DataFrame,
        targets: Set[DfF | SingleTypeDef[DfF]],
        realm: Any,
        annotation: SingleTypeDef[pl.DataFrame] | None = None,
    ) -> DfF:
        """Convert a DataFrame to a CSV or Parquet file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            assert isinstance(target, File), "Target must be a File instance."
            if target.mime == "text/csv":
                instance.write_csv(target.content)
                return target
            elif target.mime == "application/vnd.apache.parquet":
                assert isinstance(target.content, BinaryIO)
                instance.write_parquet(target.content)
                return target
            elif target.mime == "application/vnd.ms-excel":
                instance.write_excel(target.content)
                return target

        raise ValueError("No suitable target found for converting DataFrame.")


converters[pl.DataFrame] = PlConv

# NumPy arrays

type NpFile = File[BinaryIO, Literal["application/x-npy"]] | File[
    BinaryIO, Literal["application/x-npz"]
] | File[TextIO, Literal["text/csv"]]

NpF = TypeVar("NpF", bound=NpFile, default=NpFile)


class NpConv(Converter[np.ndarray, NpFile]):
    """Converter for NumPy arrays."""

    @classmethod
    def parse(
        cls,
        source: NpFile,
        realm: Any,
        annotation: SingleTypeDef[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Parse a NumPy array from a .npy, .npz, or CSV file."""
        if source.mime == "application/x-npy":
            return np.load(source.content, allow_pickle=True)
        elif source.mime == "application/x-npz":
            return np.load(source.content, allow_pickle=True)["arr_0"]
        elif source.mime == "text/csv":
            return np.loadtxt(source.content, delimiter=",")
        else:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, NpF]],
        instance: np.ndarray,
        targets: Set[NpF | SingleTypeDef[NpF]],
        realm: Any,
        annotation: SingleTypeDef[np.ndarray] | None = None,
    ) -> NpF:
        """Convert a NumPy array to a .npy, .npz, or CSV file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            assert isinstance(target, File), "Target must be a File instance."
            if target.mime == "application/x-npy":
                np.save(target.content, instance)
                return cast(NpF, target)
            elif target.mime == "application/x-npz":
                np.savez(target.content, arr_0=instance)
                return cast(NpF, target)
            elif target.mime == "text/csv":
                np.savetxt(target.content, instance, delimiter=",")
                return cast(NpF, target)

        raise ValueError("No suitable target found for converting NumPy array.")


converters[np.ndarray] = NpConv

# SQL statements

type SQLFile = File[TextIO, Literal["application/sql"]]

SQLF = TypeVar("SQLF", bound=SQLFile, default=SQLFile)


class SQLConv(Converter[sqla.sql.expression.ClauseElement, SQLFile]):
    """Converter for SQL statements."""

    @classmethod
    def parse(
        cls,
        source: SQLFile,
        realm: Any,
        annotation: SingleTypeDef[sqla.sql.expression.ClauseElement] | None = None,
    ) -> sqla.sql.expression.ClauseElement:
        """Parse a SQL statement from a file."""
        content = source.content.read()
        stmt = sqla.text(content)
        if annotation is not None and not has_type(stmt, annotation):
            raise ValueError(f"Parsed SQL does not match expected type: {annotation}")
        return stmt

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, SQLF]],
        instance: sqla.sql.expression.ClauseElement,
        targets: Set[SQLF | SingleTypeDef[SQLF]],
        realm: Any,
        annotation: SingleTypeDef[sqla.sql.expression.ClauseElement] | None = None,
    ) -> SQLF:
        """Convert a SQL statement to a file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            assert isinstance(target, File), "Target must be a File instance."
            target.content.write(
                sqlparse.format(
                    str(instance),
                    reindent=True,
                    keyword_case="upper",
                )
            )
            return cast(SQLF, target)

        raise ValueError("No suitable target found for converting SQL statement.")


converters[sqla.sql.expression.ClauseElement] = SQLConv

# PIL images, if PIL is installed

try:
    from PIL import Image

    type ImageFile = File[
        BinaryIO,
        Literal["image/png",],
    ] | File[
        BinaryIO,
        Literal["image/jpeg",],
    ] | File[
        BinaryIO,
        Literal["image/gif",],
    ] | File[
        BinaryIO,
        Literal["image/bmp",],
    ] | File[
        BinaryIO,
        Literal["image/tiff",],
    ] | File[
        BinaryIO,
        Literal["image/webp",],
    ]

    ImageF = TypeVar("ImageF", bound=ImageFile, default=ImageFile)

    class ImageConv(Converter[Image.Image, ImageFile]):
        """Converter for PIL Images."""

        @classmethod
        def parse(
            cls,
            source: ImageFile,
            realm: Any,
            annotation: SingleTypeDef[Image.Image] | None = None,
        ) -> Image.Image:
            """Parse an image from a file."""
            img = Image.open(source.content)
            if annotation is not None and not has_type(img, annotation):
                raise ValueError(
                    f"Parsed image does not match expected type: {annotation}"
                )
            return img

        @classmethod
        def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
            cls: type[Converter[Any, ImageF]],
            instance: Image.Image,
            targets: Set[ImageF | SingleTypeDef[ImageF]],
            realm: Any,
            annotation: SingleTypeDef[Image.Image] | None = None,
        ) -> ImageF:
            """Convert an image to a file."""
            matched_targets = cls.conv_types().match_targets(targets)

            for target in matched_targets:
                assert isinstance(target, File), "Target must be a File instance."
                instance.save(target.content, format=target.mime.split("/")[1].upper())
                return cast(ImageF, target)

            raise ValueError("No suitable target found for converting image.")

    converters[Image.Image] = ImageConv

except ImportError:
    # PIL is not installed, so we won't register any converters for images
    pass


# BeautifulSoup HTML, if BeautifulSoup is installed

try:
    from bs4 import BeautifulSoup

    type HTMLFile = File[TextIO, Literal["text/html"]]

    HTMLF = TypeVar("HTMLF", bound=HTMLFile, default=HTMLFile)

    class HTMLConv(Converter[BeautifulSoup, HTMLFile]):
        """Converter for BeautifulSoup HTML."""

        @classmethod
        def parse(
            cls,
            source: HTMLFile,
            realm: Any,
            annotation: SingleTypeDef[BeautifulSoup] | None = None,
        ) -> BeautifulSoup:
            """Parse HTML from a file."""
            content = source.content.read()
            soup = BeautifulSoup(content, "html.parser")
            if annotation is not None and not has_type(soup, annotation):
                raise ValueError(
                    f"Parsed HTML does not match expected type: {annotation}"
                )
            return soup

        @classmethod
        def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
            cls: type[Converter[Any, HTMLF]],
            instance: BeautifulSoup,
            targets: Set[HTMLF | SingleTypeDef[HTMLF]],
            realm: Any,
            annotation: SingleTypeDef[BeautifulSoup] | None = None,
        ) -> HTMLF:
            """Convert HTML to a file."""
            matched_targets = cls.conv_types().match_targets(targets)

            for target in matched_targets:
                assert isinstance(target, File), "Target must be a File instance."
                target.content.write(str(instance))
                return cast(HTMLF, target)

            raise ValueError("No suitable target found for converting HTML.")

    converters[BeautifulSoup] = HTMLConv

except ImportError:
    # BeautifulSoup is not installed, so we won't register any converters for HTML
    pass


# Pickle files

type PickleFile = File[BinaryIO, Literal["application/pickle"]]


class PickleConv(Converter[Any, PickleFile]):
    """Converter for Pickle files."""

    @classmethod
    def parse(
        cls,
        source: PickleFile,
        realm: Any,
        annotation: SingleTypeDef[Any] | None = None,
    ) -> Any:
        """Parse an object from a Pickle file."""
        obj = pickle.load(source.content)
        if annotation is not None and not has_type(obj, annotation):
            raise ValueError(
                f"Parsed object does not match expected type: {annotation}"
            )
        return obj

    @classmethod
    def convert(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[Converter[Any, PickleFile]],
        instance: Any,
        targets: Set[PickleFile | SingleTypeDef[PickleFile]],
        realm: Any,
        annotation: SingleTypeDef[Any] | None = None,
    ) -> PickleFile:
        """Convert an object to a Pickle file."""
        matched_targets = cls.conv_types().match_targets(targets)

        for target in matched_targets:
            assert isinstance(target, File), "Target must be a File instance."
            pickle.dump(instance, target.content)
            return cast(PickleFile, target)

        raise ValueError("No suitable target found for converting to Pickle.")


converters[object] = PickleConv
