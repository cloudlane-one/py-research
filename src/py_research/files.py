"""Helper functions for handling files and directories."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from mimetypes import guess_extension, guess_type
from pathlib import Path
from tempfile import gettempdir
from typing import Any, BinaryIO, Generic, Literal, LiteralString, TextIO, TypeVar, cast

import requests
import yarl


def ensure_dir_exists(path: Path | str):
    """Make sure the given path exists and is a directory.

    Create dir if necessary. Then return the path unchanged.

    Args:
        path: Path to ensure exists.

    Returns:
        The given path.
    """
    path = Path(path).absolute()
    p = Path(path.parts[0])
    for d in path.parts[1:]:
        p = p / d
        if not p.is_dir():
            if not p.exists():
                p.mkdir()
            else:
                raise NotADirectoryError()

    return path


additional_mime = {
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".parquet": "application/vnd.apache.parquet",
    ".xlsx": "application/vnd.ms-excel",
    ".npy": "application/x-npy",
}
additional_mime_inv = {
    g: [k for k, _ in g_items]
    for g, g_items in groupby(additional_mime.items(), key=lambda x: x[1])
}

C = TypeVar("C", bound=TextIO | BinaryIO, covariant=True)
M = TypeVar("M", bound=LiteralString, covariant=True)


@dataclass
class File(Generic[C, M]):
    """A generic, single file (may be within a file system or object storage)."""

    content: C
    mime: M

    @staticmethod
    def from_path[C2: TextIO | BinaryIO, M2: LiteralString](
        path: Path | str,
        mode: Literal["r", "w", "rw"],
        io_kind: type[C2],
        target_mime: Set[M2] | None,
    ) -> File[C2, M2]:
        """Create a File instance from a file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mime_type, encoding = guess_type(path)
        if mime_type is None:
            mime_type = additional_mime.get(path.suffix)

        if target_mime and mime_type not in target_mime:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

        if issubclass(io_kind, TextIO):
            file_mode = mode
        else:
            file_mode = mode + "b"
            encoding = None

        return File(
            content=cast(C2, path.open(file_mode, encoding=encoding)),
            mime=cast(M2, mime_type or "application/octet-stream"),
        )

    def extension(self) -> str:
        """Return the file extension based on the MIME type."""
        ext = guess_extension(self.mime, strict=False)
        if ext is None:
            ext = additional_mime_inv.get(self.mime, [None])[0]

        default = ".txt" if isinstance(self.content, TextIO) else ".bin"
        return ext if ext else default


type Dir = Mapping[str, File[Any, Any]]


@dataclass
class HttpFile:
    """A (writable) file on a WebDAV server."""

    url: str | yarl.URL
    auth: tuple[str, str] | None = None

    def __post_init__(self):  # noqa: D105
        self._local_path = None

    @property
    def url_obj(self) -> yarl.URL:
        """Return the URL as a yarl.URL object."""
        return self.url if isinstance(self.url, yarl.URL) else yarl.URL(self.url)

    def get(self) -> Path:
        """Download the file to a temporary location and return the path."""
        filename = Path(self.url_obj.path).name

        if (
            self._local_path is not None
            and self._local_path.exists()
            and self.last_modified
            <= datetime.fromtimestamp(self._local_path.stat().st_mtime)
        ):
            return self._local_path
        else:
            self._local_path = Path(gettempdir()) / filename

        with requests.get(str(self.url_obj), stream=True, auth=self.auth) as response:
            response.raise_for_status()
            with self._local_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        return self._local_path

    def set(self, file: Path | str | BinaryIO) -> None:
        """Upload the file to the given path."""
        f = file if isinstance(file, BinaryIO) else Path(file).open("rb")
        response = requests.put(str(self.url_obj), data=f, auth=self.auth)
        response.raise_for_status()
        f.close()

    @property
    def last_modified(self) -> datetime:
        """Return the last modified date of the file."""
        response = requests.head(str(self.url_obj), auth=self.auth)
        response.raise_for_status()
        return datetime.fromisoformat(response.headers["Last-Modified"])
