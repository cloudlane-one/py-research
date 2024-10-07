"""Flexible, performant and batteries-included HTTP API client."""

from asyncio import sleep
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine, Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Literal, Self

from aiohttp import (
    BasicAuth,
    ClientConnectionError,
    ClientResponse,
    ClientResponseError,
    ClientSession,
)
from tenacity import Retrying, retry_if_exception, stop_after_delay, wait_exponential
from yarl import URL

from py_research.caching import FileCache, get_cache
from py_research.data import copy_and_override
from py_research.hashing import gen_str_hash

type HttpMethod = Literal[
    "GET", "POST", "PUT", "DELETE", "PATCH", "SEARCH", "HEAD", "OPTIONS"
]


@dataclass
class ApiClient:
    """HTTP API client."""

    api_name: str
    """Name of the API."""

    url: URL | str
    """Base URL of this API client instance."""

    headers: dict[str, str] = field(default_factory=dict)
    """Headers to send with all requests."""

    auth: BasicAuth | None = None
    """Basic authentication credentials."""

    max_concurrency: int = 10
    """Maximum number of concurrent async requests."""

    rate_limit: int = 10
    """Maximum number of requests per second."""

    network_wait: int = 300
    """Maximum seconds to wait before erroring out during network issues."""

    server_error_retries: int = 3
    """Number of retries for server errors (HTTP 5xx)."""

    cache: FileCache | bool = False
    """File-based cache for responses."""

    @property
    def _url(self) -> URL:
        """URL object of the base URL."""
        return self.url if isinstance(self.url, URL) else URL(self.url)

    @property
    def _loc_str(self) -> str:
        """Hash a URL location."""
        string = self._url.path.replace("/", "_").lstrip("_")

        if len(self._url.query) > 0:
            string += "_" + gen_str_hash(dict(self._url.query), 8)
        if len(self.headers) > 0:
            string += "_" + gen_str_hash(self.headers, 8)

        return string

    @cached_property
    def _past_requests(self) -> deque[datetime]:
        """Queue of past requests."""
        return deque(maxlen=self.rate_limit)

    @cached_property
    def _running_requests(self) -> set[datetime]:
        """Set of running requests."""
        return set()

    @asynccontextmanager
    async def _request_session(self) -> AsyncGenerator[ClientSession]:
        """Check the rate limit, concurrency limit, and create a session."""
        sleep_time = 1 / self.rate_limit
        while True:
            now = datetime.now()
            second_ago = now - timedelta(seconds=1)

            if (
                len(self._past_requests) < self.rate_limit
                or self._past_requests[0] < second_ago
            ):
                break

            await sleep(sleep_time)

        self._past_requests.append(now)
        self._running_requests.add(now)

        yield ClientSession()

        self._running_requests.remove(now)
        return

    @staticmethod
    def _is_server_error(e: BaseException) -> bool:
        """Check if the response is a server error."""
        return isinstance(e, ClientResponseError) and 500 <= e.status < 600

    def to(
        self,
        loc: URL | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Self:
        """Create a client on a sub-URL with optional extra headers."""
        url = self.url
        if loc is not None:
            if isinstance(loc, str):
                loc = URL(loc)
            url = self._url / loc.path % loc.query

        return copy_and_override(
            self,
            type(self),
            api_name=self.api_name,
            url=url,
            headers={**self.headers, **(headers or {})},
        )

    async def _request(
        self,
        method: HttpMethod = "GET",
        url: URL | Hashable | None = None,
        body: str | bytes | None = None,
    ) -> ClientResponse:
        """Make an HTTP request."""
        full_url = self._url
        if isinstance(url, URL):
            full_url = full_url / url.path % url.query
        elif isinstance(url, Hashable):
            full_url = full_url / str(url)

        async with self._request_session() as session:
            for net_attempt in Retrying(
                retry=retry_if_exception(
                    lambda e: isinstance(e, ClientConnectionError)
                ),
                wait=wait_exponential(),
                stop=stop_after_delay(self.network_wait),
            ):
                with net_attempt:
                    for server_attempt in Retrying(
                        retry=retry_if_exception(self._is_server_error),
                        wait=wait_exponential(),
                        stop=stop_after_delay(self.network_wait),
                    ):
                        with server_attempt:
                            return await session.request(
                                method,
                                full_url,
                                data=body,
                                headers=self.headers,
                                auth=self.auth,
                                raise_for_status=True,
                            )

        # This should never be reached:
        raise RuntimeError()

    @cached_property
    def _cache(self) -> FileCache:
        """Get the cache."""
        return (
            self.cache
            if isinstance(self.cache, FileCache)
            else get_cache(self.api_name)
        )

    async def _get(self, res_id: URL | Hashable | None = None) -> dict[str, Any]:
        """Make a GET request."""
        res = await self._request("GET", res_id)
        return await res.json()

    def getter(
        self,
    ) -> Callable[[URL | Hashable | None], Coroutine[Any, Any, dict[str, Any]]]:
        """Get a cached GET request callable."""
        return (
            self._get
            if self.cache is False
            else self._cache.function(
                name=self._loc_str, async_func=True, id_arg_subset=["res_id"]
            )(self._get)
        )
