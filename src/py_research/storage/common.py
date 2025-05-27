"""Default converters for common types."""

from __future__ import annotations

import json
import pickle
import xml.etree.ElementTree as xml  # noqa: N813
from collections.abc import Iterable, Set
from types import UnionType
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

from .storables import V2, StorageDriver

drivers: dict[SingleTypeDef | UnionType, type[StorageDriver]] = {}


def get_storage_driver[U, T](
    instance: type[U] | U, targets: Set[SingleTypeDef[T]]
) -> type[StorageDriver[U, T]] | None:
    """Get the converter for a given type or instance."""
    obj_type = instance if isinstance(instance, type) else type(instance)

    matching_conv = [
        c
        for t, c in drivers.items()
        if is_subtype(obj_type, t) and len(c.conv_types().match_targets(targets)) > 0
    ]

    if len(matching_conv) == 0:
        return None

    return matching_conv[0]


# Simple, stringifiable types

type TextFile = File[Literal["text/plain", "text/csv"], TextIO]
TF = TypeVar(
    "TF",
    bound=TextFile,
    default=TextFile,
)

type StrScalar = str | int | float | complex | bool | None
type Stringifiable = StrScalar | list[StrScalar]


type JSONFile = File[Literal["application/yaml"], TextIO] | File[
    Literal["application/json"], TextIO
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


def check_type_and_cast[T](obj: Any, annotation: SingleTypeDef[T] | None = None) -> T:
    """Check if the object matches the annotation type and cast it."""
    if annotation is not None and not has_type(obj, annotation):
        raise TypeError(f"Object does not match expected type: {annotation}")
    return cast(T, obj)


class JSONStore(StorageDriver[JSONDict, JSONFile]):
    """Converter for JSON and YAML dictionaries."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, JSONFile]],
        source: JSONFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse a dictionary from a JSON or YAML file."""
        dct: dict

        with source.open("r") as f:
            if source.mime == "application/json":
                dct = yaml.load(f, Loader=yaml.CLoader)
            else:
                dct = json.load(f)

        return check_type_and_cast(dct, annotation)

    @classmethod
    def store(
        cls,
        instance: JSONDict,
        target: JSONFile,
        realm: Any,
        annotation: SingleTypeDef[JSONDict] | None = None,
    ) -> None:
        """Convert a dictionary to a JSON or YAML file."""
        with target.open("w") as f:
            if target.mime == "application/yaml":
                yaml.dump(instance, allow_unicode=True, stream=f)
            else:
                f.write(json.dumps(instance, indent=2))


drivers[JSONDict] = JSONStore


# XML etree

XMLFile = File[Literal["text/xml", "application/xml"], TextIO]

XMLF = TypeVar(
    "XMLF",
    bound=XMLFile,
    default=XMLFile,
)


class XMLStore(StorageDriver[xml.ElementTree, XMLFile]):
    """Converter for XML ElementTree."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, XMLFile]],
        source: XMLFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse an XML ElementTree from a file."""
        if source.mime not in {"application/xml", "text/xml"}:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

        with source.open("r") as f:
            tree = xml.parse(f)

        return check_type_and_cast(tree, annotation)

    @classmethod
    def store(
        cls,
        instance: xml.ElementTree,
        target: XMLFile,
        realm: Any,
        annotation: SingleTypeDef[xml.ElementTree] | None = None,
    ) -> None:
        """Convert an XML ElementTree to a file."""
        assert target.mime in {"application/xml", "text/xml"}
        with target.open("w") as f:
            instance.write(f)


drivers[xml.ElementTree] = XMLStore

# Pydantic models


class PydConv(StorageDriver[pyd.BaseModel, JSONFile]):
    """Converter for Pydantic models."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, JSONFile]],
        source: JSONFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse a Pydantic model from a JSON or YAML file."""
        with source.open("r") as f:
            if source.mime == "application/yaml":
                data = yaml.load(f, Loader=yaml.CLoader)
            else:
                data = json.load(f)

        if annotation is None or not has_type(data, annotation):
            raise ValueError(f"Data does not match expected type: {annotation}")

        assert (
            annotation is not None
        ), "Annotation must be provided for Pydantic model parsing."
        adapter = pyd.TypeAdapter(annotation)
        return adapter.validate_python(data)

    @classmethod
    def store(
        cls,
        instance: pyd.BaseModel,
        target: JSONFile,
        realm: Any,
        annotation: SingleTypeDef[pyd.BaseModel] | None = None,
    ) -> None:
        """Convert a Pydantic model to a JSON or YAML file."""
        with target.open("w") as f:
            if target.mime == "application/yaml":
                yaml.dump(instance.model_dump(), allow_unicode=True, stream=f)
            else:
                json.dump(instance.model_dump(), f, indent=2)


drivers[pyd.BaseModel] = PydConv

# DataFrames

type DfFile = File[
    Literal["application/vnd.ms-excel", "application/vnd.apache.parquet"], BinaryIO
] | File[Literal["text/csv"], TextIO]


# Pandas


class PdConv(StorageDriver[pd.DataFrame, DfFile]):
    """Converter for Pandas DataFrames."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, DfFile]],
        source: DfFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse a DataFrame from a CSV or Parquet file."""
        with source.open("r") as f:
            if source.mime == "text/csv":
                res = pd.read_csv(f)
            elif source.mime == "application/vnd.apache.parquet":
                assert isinstance(f, BinaryIO), "Parquet file must be binary."
                res = pd.read_parquet(f)
            elif source.mime == "application/vnd.ms-excel":
                res = pd.read_excel(f)
            else:
                raise ValueError(f"Unsupported MIME type: {source.mime}")

        return check_type_and_cast(res, annotation)

    @classmethod
    def store(
        cls,
        instance: pd.DataFrame,
        target: DfFile,
        realm: Any,
        annotation: SingleTypeDef[pd.DataFrame] | None = None,
    ) -> None:
        """Convert a DataFrame to a CSV or Parquet file."""
        with target.open("w") as f:
            if target.mime == "text/csv":
                instance.to_csv(f, index=False)
            elif target.mime == "application/vnd.apache.parquet":
                assert isinstance(f, BinaryIO)
                instance.to_parquet(f)
            elif target.mime == "application/vnd.ms-excel":
                instance.to_excel(f, index=False)
            else:
                raise ValueError(f"Unsupported MIME type: {target.mime}")


drivers[pd.DataFrame] = PdConv

# Polars


class PlConv(StorageDriver[pl.DataFrame, DfFile]):
    """Converter for Polars DataFrames."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, DfFile]],
        source: DfFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse a Polars DataFrame from a CSV or Parquet file."""
        with source.open("r") as f:
            if source.mime == "text/csv":
                res = pl.read_csv(f)
            elif source.mime == "application/vnd.apache.parquet":
                assert isinstance(f, BinaryIO), "Parquet file must be binary."
                res = pl.read_parquet(f)
            elif source.mime == "application/vnd.ms-excel":
                assert isinstance(f, BinaryIO), "Excel file must be binary."
                res = pl.read_excel(f)
            else:
                raise ValueError(f"Unsupported MIME type: {source.mime}")

        return check_type_and_cast(res, annotation)

    @classmethod
    def store(
        cls,
        instance: pl.DataFrame,
        target: DfFile,
        realm: Any,
        annotation: SingleTypeDef[pl.DataFrame] | None = None,
    ) -> None:
        """Convert a Polars DataFrame to a CSV or Parquet file."""
        with target.open("w") as f:
            if target.mime == "text/csv":
                instance.write_csv(f)
            elif target.mime == "application/vnd.apache.parquet":
                assert isinstance(f, BinaryIO)
                instance.write_parquet(f)
            elif target.mime == "application/vnd.ms-excel":
                assert isinstance(f, BinaryIO)
                instance.write_excel(f)
            else:
                raise ValueError(f"Unsupported MIME type: {target.mime}")


drivers[pl.DataFrame] = PlConv

# NumPy arrays

type NpFile = File[Literal["application/x-npy", "application/x-npz"], BinaryIO] | File[
    Literal["text/csv"], TextIO
]


class NpConv(StorageDriver[np.ndarray, NpFile]):
    """Converter for NumPy arrays."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, NpFile]],
        source: NpFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse a NumPy array from a file."""
        with source.open("r") as f:
            if source.mime == "application/x-npy":
                assert isinstance(f, BinaryIO), "Numpy file must be binary."
                res = np.load(f)
            elif source.mime == "application/x-npz":
                assert isinstance(f, BinaryIO), "Numpy file must be binary."
                res = np.load(f)["arr_0"]
            elif source.mime == "text/csv":
                res = np.loadtxt(f, delimiter=",")
            else:
                raise ValueError(f"Unsupported MIME type: {source.mime}")

        return check_type_and_cast(res, annotation)

    @classmethod
    def store(
        cls,
        instance: np.ndarray,
        target: NpFile,
        realm: Any,
        annotation: SingleTypeDef[np.ndarray] | None = None,
    ) -> None:
        """Convert a NumPy array to a file."""
        with target.open("w") as f:
            if target.mime == "application/x-npy":
                assert isinstance(f, BinaryIO), "Numpy file must be binary."
                np.save(f, instance)
            elif target.mime == "application/x-npz":
                assert isinstance(f, BinaryIO), "Numpy file must be binary."
                np.savez(f, arr_0=instance)
            elif target.mime == "text/csv":
                np.savetxt(f, instance, delimiter=",")
            else:
                raise ValueError(f"Unsupported MIME type: {target.mime}")


drivers[np.ndarray] = NpConv

# SQL statements

type SQLFile = File[Literal["application/sql"], TextIO]


class SQLConv(StorageDriver[sqla.Executable | Iterable[sqla.Executable], SQLFile]):
    """Converter for SQL statements."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, SQLFile]],
        source: SQLFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse an SQL statement from a file."""
        with source.open("r") as f:
            sql_content = f.read()
            parsed_sql = sqlparse.parse(sql_content)

        stmts = [sqla.text(str(stmt)) for stmt in parsed_sql]

        return check_type_and_cast(stmts[0] if len(stmts) == 1 else stmts, annotation)

    @classmethod
    def store(
        cls,
        instance: sqla.Executable | Iterable[sqla.Executable],
        target: SQLFile,
        realm: Any,
        annotation: (
            SingleTypeDef[sqla.Executable | Iterable[sqla.Executable]] | None
        ) = None,
    ) -> None:
        """Convert an SQL statement to a file."""
        if isinstance(instance, sqla.Executable):
            instance = [instance]

        with target.open("w") as f:
            for stmt in instance:
                f.write(
                    sqlparse.format(
                        str(stmt),
                        reindent=True,
                        keyword_case="upper",
                    )
                )


drivers[sqla.Executable | Iterable[sqla.Executable]] = SQLConv

# PIL images, if PIL is installed

try:
    from PIL import Image

    type ImageFile = File[
        Literal[
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "image/webp",
        ],
        BinaryIO,
    ]

    ImageF = TypeVar("ImageF", bound=ImageFile, default=ImageFile)

    class ImageConv(StorageDriver[Image.Image, ImageFile]):
        """Converter for PIL Images."""

        @classmethod
        def load(
            cls: type[StorageDriver[V2, ImageFile]],
            source: ImageFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse an image from a file."""
            with source.open("r") as f:
                img = Image.open(f)
                img.load()

            return check_type_and_cast(img, annotation)

        @classmethod
        def store(
            cls,
            instance: Image.Image,
            target: ImageFile,
            realm: Any,
            annotation: SingleTypeDef[Image.Image] | None = None,
        ) -> None:
            """Convert an image to a file."""
            with target.open("w") as f:
                instance.save(f, format=instance.format)

    drivers[Image.Image] = ImageConv

except ImportError:
    # PIL is not installed, so we won't register any converters for images
    pass


# BeautifulSoup HTML, if BeautifulSoup is installed

try:
    from bs4 import BeautifulSoup

    type HTMLFile = File[Literal["text/html"], TextIO]

    HTMLF = TypeVar("HTMLF", bound=HTMLFile, default=HTMLFile)

    class HTMLConv(StorageDriver[BeautifulSoup, HTMLFile]):
        """Converter for BeautifulSoup HTML."""

        @classmethod
        def load(
            cls: type[StorageDriver[V2, HTMLFile]],
            source: HTMLFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse HTML from a file."""
            with source.open("r") as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, "html.parser")

            return check_type_and_cast(soup, annotation)

        @classmethod
        def store(
            cls,
            instance: BeautifulSoup,
            target: HTMLFile,
            realm: Any,
            annotation: SingleTypeDef[BeautifulSoup] | None = None,
        ) -> None:
            """Convert BeautifulSoup object to an HTML file."""
            with target.open("w") as f:
                f.write(str(instance))

    drivers[BeautifulSoup] = HTMLConv

except ImportError:
    # BeautifulSoup is not installed, so we won't register any converters for HTML
    pass


# Pickle files

type PickleFile = File[Literal["application/pickle"], BinaryIO]


class PickleConv(StorageDriver[Any, PickleFile]):
    """Converter for Pickle files."""

    @classmethod
    def load(
        cls: type[StorageDriver[V2, PickleFile]],
        source: PickleFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse an object from a Pickle file."""
        with source.open("r") as f:
            obj = pickle.load(f)

        return check_type_and_cast(obj, annotation)

    @classmethod
    def store(
        cls,
        instance: Any,
        target: PickleFile,
        realm: Any,
        annotation: SingleTypeDef[Any] | None = None,
    ) -> None:
        """Convert an object to a Pickle file."""
        with target.open("w") as f:
            pickle.dump(instance, f)


drivers[object] = PickleConv
