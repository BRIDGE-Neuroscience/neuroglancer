# @license
# Copyright 2026 Allen Institute for Brain Science
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A read-only zarr Store backed by the browser's ``fetch``.

This is the piece ``zarr_vectors.core.aio`` asks the host to supply: "the only
way in is a fetch-backed Store the caller constructs itself." With it, ngpy
reads a zarr-vectors store through the real library rather than a hand-rolled
decoder -- see :mod:`ngpy.zv`.

**Imports zarr**, so it can only be imported where zarr is available (under
Pyodide zarr comes from the Pyodide distribution, which is WASM-patched).

Listing is OPTIONAL. A plain HTTP object store cannot enumerate keys, and the
geometry reader in :mod:`ngpy.zv` never needs to. But attribute *discovery*
(``object_attributes/<name>`` children) does, and zarr-vectors 0.9.2's own
readers stall on a non-listing store whenever a level carries per-vertex
attributes.  So a host that CAN list -- Google Cloud Storage through its JSON
API, or a local directory in tests -- passes ``lister``, and ``list_dir`` is
served through it.  Without one, every listing raises.

Byte ranges are forwarded as HTTP ``Range`` requests: a sharded store packs
many cells into one object and zarr reads the shard index and each cell by
range, so fetching whole objects would download every shard once per cell.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype

from .zvf_pure import byte_range_header

#: ``fetch(url, range_header=None) -> bytes | None``.  ``range_header`` is an
#: HTTP ``Range`` value (``bytes=a-b``, ``bytes=a-``, ``bytes=-n``); the fetch
#: must return exactly that range (slicing locally if the server ignored it --
#: see :func:`zvf_pure.slice_for_range_header`) and None for a missing key.
FetchFn = Callable[..., Awaitable[bytes | None]]
#: ``lister(url_of_a_directory_with_trailing_slash) -> child names`` (or None
#: when the directory does not exist / cannot be listed).
ListFn = Callable[[str], Awaitable[list[str] | None]]


class BrowserFetchStore(Store):
    """Read-only zarr v3 store over an async ``fetch(url) -> bytes | None``.

    ``base_url`` is the store root (with or without a trailing slash); a zarr
    key is appended to it directly. ``fetch`` returns ``None`` for a 404, which
    zarr reads as an absent key.
    """

    def __init__(self, base_url: str, fetch: FetchFn, lister: ListFn | None = None):
        super().__init__(read_only=True)
        self._base = base_url if base_url.endswith("/") else base_url + "/"
        self._fetch = fetch
        self._lister = lister

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BrowserFetchStore) and other._base == self._base

    def __hash__(self) -> int:
        return hash(("BrowserFetchStore", self._base))

    def __repr__(self) -> str:
        return f"BrowserFetchStore({self._base!r})"

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_partial_writes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return self._lister is not None

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        header = byte_range_header(byte_range)
        if header is None:
            data = await self._fetch(self._base + key)
        else:
            data = await self._fetch(self._base + key, header)
        if data is None:
            return None
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        import asyncio

        return list(
            await asyncio.gather(
                *(self.get(key, prototype, br) for key, br in key_ranges)
            )
        )

    async def exists(self, key: str) -> bool:
        return (await self._fetch(self._base + key)) is not None

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("BrowserFetchStore is read-only")

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("BrowserFetchStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("BrowserFetchStore is read-only")

    # Each of these is an async generator.  ``list``/``list_prefix`` (deep
    # enumeration) are never needed; ``list_dir`` is served by the lister.
    async def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("BrowserFetchStore cannot list recursively")
        yield  # pragma: no cover - marks this an async generator

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("BrowserFetchStore cannot list recursively")
        yield  # pragma: no cover

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if self._lister is None:
            raise NotImplementedError("BrowserFetchStore cannot list")
        path = prefix.strip("/")
        url = self._base + (path + "/" if path else "")
        names = await self._lister(url)
        for name in names or ():
            yield name


def make_browser_fetch_store(
    base_url: str, fetch: FetchFn, lister: ListFn | None = None
) -> Any:
    """Construct a :class:`BrowserFetchStore`.

    A thin factory so callers can obtain a store without importing this module
    at their top level (it imports zarr, which is not always present).
    """
    return BrowserFetchStore(base_url, fetch, lister)
