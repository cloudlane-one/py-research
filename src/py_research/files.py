"""Helper functions for handling files and directories."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import BinaryIO

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


@dataclass
class HttpFile:
    """A file on a WebDAV server."""

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
