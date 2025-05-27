"""Helper functions for handling files and directories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from mimetypes import guess_extension, guess_type
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Generic,
    Literal,
    LiteralString,
    TextIO,
    cast,
    overload,
)

from identify.identify import tags_from_path
from typing_extensions import TypeVar

from py_research.caching import cached_method, cached_prop


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
    ".pkl": "application/pickle",
}
additional_mime_inv = {
    g: [k for k, _ in g_items]
    for g, g_items in groupby(additional_mime.items(), key=lambda x: x[1])
}

M = TypeVar("M", bound=LiteralString | str, covariant=True)

I = TypeVar(  # noqa: E741
    "I",
    bound=TextIO | BinaryIO,
    default=Any,
)


@dataclass(kw_only=True)
class File(ABC, Generic[M, I]):
    """A generic, single file (may be within a file system or object storage)."""

    mime: M
    io_type: type[I]

    @cached_prop
    def extension(self) -> str:
        """Return the file extension based on the MIME type."""
        ext = guess_extension(self.mime, strict=False)
        if ext is None:
            ext = additional_mime_inv.get(self.mime, [None])[0]

        if ext is None:
            ext = ".txt" if self.io_type == "text" else ".bin"

        assert ext is not None, "Could not determine file extension"
        return ext

    @overload
    def get_io(
        self: File[Any, TextIO], mode: Literal["rw", "r", "w"] = ...
    ) -> TextIO: ...

    @overload
    def get_io(
        self: File[Any, BinaryIO], mode: Literal["rw", "r", "w"] = ...
    ) -> BinaryIO: ...

    @overload
    def get_io(
        self: File[Any, TextIO | BinaryIO], mode: Literal["rw", "r", "w"] = ...
    ) -> TextIO | BinaryIO: ...

    @abstractmethod
    def get_io(self, mode: Literal["rw", "r", "w"] = "rw") -> TextIO | BinaryIO:
        """Return the file's content as an IO stream."""
        ...

    @overload
    def open(
        self: File[Any, TextIO], mode: Literal["rw", "r", "w"] = ...
    ) -> TextIO: ...

    @overload
    def open(
        self: File[Any, BinaryIO], mode: Literal["rw", "r", "w"] = ...
    ) -> BinaryIO: ...

    @overload
    def open(
        self: File[Any, TextIO | BinaryIO], mode: Literal["rw", "r", "w"] = ...
    ) -> TextIO | BinaryIO: ...

    @contextmanager
    def open(
        self, mode: Literal["rw", "r", "w"] = "rw"
    ) -> Generator[TextIO | BinaryIO]:
        """Open the file and return its IO stream."""
        io = self.get_io()
        try:
            yield io
        finally:
            io.close()


@dataclass
class LocalFile(File[M, TextIO | BinaryIO]):
    """A generic, single file (may be within a file system or object storage)."""

    dir: LocalDir
    name: str

    @staticmethod
    def from_path(path: Path | str) -> LocalFile:
        """Create a LocalFile from a given path."""
        path = Path(path).absolute()

        dir_path = path.parent
        name = path.stem
        mime = guess_type(path)[0] or "application/octet-stream"
        tags = tags_from_path(str(path))
        io_type = TextIO if "text" in tags else BinaryIO

        return LocalFile(
            dir=LocalDir(path=dir_path), name=name, mime=mime, io_type=io_type
        )

    @cached_prop
    def path(self) -> Path:
        """Return the absolute path of the file."""
        return (self.dir.abs_path / (self.name + self.extension)).absolute()

    @cached_prop
    def detected_format(self) -> Literal["text", "binary"]:
        """Default encoding based on filename and content."""
        path = Path(self.path)
        file_tags = tags_from_path(str(path))
        return "text" if "text" in file_tags else "binary"

    def get_file_mode(
        self, mode: Literal["rw", "r", "w"], encoding: Literal["text", "binary"]
    ) -> str:
        """Map the file mode to a string representation."""
        if mode == "r":
            return "rb" if encoding == "binary" else "r"
        elif mode == "w":
            return "wb" if encoding == "binary" else "w"
        else:
            return "wb+" if encoding == "binary" else "w+"

    @overload
    def get_io(
        self: File[Any, TextIO], mode: Literal["rw", "r", "w"] = ...
    ) -> TextIO: ...

    @overload
    def get_io(
        self: File[Any, BinaryIO], mode: Literal["rw", "r", "w"] = ...
    ) -> BinaryIO: ...

    @overload
    def get_io(
        self: File[Any, TextIO | BinaryIO], mode: Literal["rw", "r", "w"] = ...
    ) -> TextIO | BinaryIO: ...

    def get_io(self, mode: Literal["rw", "r", "w"] = "rw") -> TextIO | BinaryIO:
        """Return the file's content as a text IO stream."""
        if issubclass(self.io_type, TextIO):
            return cast(TextIO, open(self.path, mode=self.get_file_mode(mode, "text")))

        return cast(BinaryIO, open(self.path, mode=self.get_file_mode(mode, "binary")))


F = TypeVar("F", bound="File[Any, TextIO | BinaryIO]", covariant=True)
D = TypeVar("D", bound="Dir | None", covariant=True)


@dataclass(kw_only=True)
class Dir(ABC, Generic[F, D]):
    """A directory of files or sub-directories."""

    @abstractmethod
    def keys(self) -> Sequence[str]:
        """Iterate over the items in the directory."""
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> F | D:
        """Get an item from the directory."""
        ...

    @abstractmethod
    def __delitem__(self, name: str) -> None:
        """Delete an item from the directory."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the directory."""
        ...

    def values(self) -> Sequence[F | D]:
        """Iterate over the items in the directory."""
        return [self[key] for key in self.keys()]

    def items(self) -> Iterator[tuple[str, F | D]]:
        """Iterate over the items in the directory as (name, item) pairs."""
        for key in self.keys():
            yield key, self[key]

    def __iter__(self) -> Iterator[F | D]:
        """Iterate over the items in the directory."""
        return iter(self.values())

    def __contains__(self, name: str) -> bool:
        """Check if an item exists in the directory."""
        return name in self.keys()

    def file(self, name: str) -> F:
        """Get a file from the directory."""
        item = self[name]
        assert isinstance(item, File)
        return item

    def subdir(self, name: str) -> D:
        """Get a sub-directory from the directory."""
        item = self[name]
        assert isinstance(item, Dir)
        return item


@dataclass
class LocalDir(Dir[LocalFile, "LocalDir"]):
    """A local directory of files or sub-directories."""

    path: Path | str

    @cached_prop
    def abs_path(self) -> Path:
        """Return the absolute path of the directory."""
        return Path(self.path).absolute()

    @cached_method
    def _ensure_exists(self) -> None:
        ensure_dir_exists(self.abs_path)

    def keys(self) -> Sequence[str]:
        """Iterate over the items in the directory."""
        self._ensure_exists()
        return [
            item.name
            for item in self.abs_path.iterdir()
            if item.is_file() or item.is_dir()
        ]

    def __getitem__(self, item: str) -> LocalDir | LocalFile:
        """Get an item from the directory."""
        self._ensure_exists()
        item_path = self.abs_path / item
        if item_path.is_dir():
            return LocalDir(path=item_path)
        elif item_path.is_file():
            return LocalFile.from_path(item_path)

        raise KeyError(f"Item '{item}' not found in directory '{self.abs_path}'.")

    def __delitem__(self, key: str) -> None:
        """Delete an item from the directory."""
        self._ensure_exists()
        item_path = self.abs_path / key
        if item_path.is_file():
            item_path.unlink()
        elif item_path.is_dir():
            item_dir = LocalDir(path=item_path)
            for name in item_dir.keys():
                del item_dir[name]
        else:
            raise KeyError(f"Item '{key}' not found in directory '{self.abs_path}'.")

    def __len__(self) -> int:
        """Return the number of items in the directory."""
        self._ensure_exists()
        return len(list(self.abs_path.iterdir()))


# @dataclass
# class HttpFile:
#     """A (writable) file on a WebDAV server."""

#     url: str | URL
#     auth: tuple[str, str] | None = None

#     def __post_init__(self):  # noqa: D105
#         self._local_path = None

#     @property
#     def url_obj(self) -> URL:
#         """Return the URL as a yarl.URL object."""
#         return self.url if isinstance(self.url, URL) else URL(self.url)

#     def get(self) -> Path:
#         """Download the file to a temporary location and return the path."""
#         filename = Path(self.url_obj.path).name

#         if (
#             self._local_path is not None
#             and self._local_path.exists()
#             and self.last_modified
#             <= datetime.fromtimestamp(self._local_path.stat().st_mtime)
#         ):
#             return self._local_path
#         else:
#             self._local_path = Path(gettempdir()) / filename

#         with requests.get(str(self.url_obj), stream=True, auth=self.auth) as response:
#             response.raise_for_status()
#             with self._local_path.open("wb") as file:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     file.write(chunk)

#         return self._local_path

#     def set(self, file: Path | str | BinaryIO) -> None:
#         """Upload the file to the given path."""
#         f = file if isinstance(file, BinaryIO) else Path(file).open("rb")
#         response = requests.put(str(self.url_obj), data=f, auth=self.auth)
#         response.raise_for_status()
#         f.close()

#     @property
#     def last_modified(self) -> datetime:
#         """Return the last modified date of the file."""
#         response = requests.head(str(self.url_obj), auth=self.auth)
#         response.raise_for_status()
#         return datetime.fromisoformat(response.headers["Last-Modified"])
