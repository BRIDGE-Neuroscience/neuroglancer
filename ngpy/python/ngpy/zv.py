# @license
# Copyright 2026 The Neuroglancer Authors
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

"""Reading a zarr-vectors store over HTTP, for whole-store dissection.

Everything here goes through zarr-vectors-py's **async** path
(``open_store_async`` + ``read_async``): the reader is an ordinary synchronous
function replayed against an offline snapshot whose I/O was awaited up front,
so nothing ever reaches zarr's blocking ``sync()``.  That is what lets these
reads run inside the Pyodide worker WITHOUT JSPI and concurrently with other
Python work.

**Which geometry is evaluated.**  The viewer draws whatever pyramid level its
GPU budget affords and only has the chunks in view; a dissection, though, must
answer "which objects pass" for the whole store.  ngpy therefore reads ONE
whole pyramid level -- by default the finest whose declared ``vertex_count``
(level metadata, no data read) is at most :data:`DEFAULT_VERTEX_BUDGET` -- and
evaluates against all of it.  On an object-sparse pyramid (``object_sparsity <
1``, e.g. HCP-1065: 503k / 50k / 5k / 503 / 50 tracts at levels 0..4) the
coarse levels are nested *subsets* of the objects, so a dissection evaluated at
level L names only objects present at L.  The level is reported to the GUI and
can be pinned by the user.

**Why not ``read_polylines``.**  In zarr-vectors 0.9.2 it enumerates
``vertex_attributes/`` and therefore stalls on a store that cannot list (any
plain HTTP host) whenever a level carries per-vertex attributes -- and when it
can list, it decodes every attribute column (``tangent`` alone triples the
bytes).  The dissection needs positions and object ids only, so
:func:`_read_level_geometry` replays exactly that part: object manifests, then
the vertex chunks they reference.

**Ids.**  A zarr-vectors object id is ``object_index/object_ids[row]`` under the
V2 layout and the row itself under V1.  Neuroglancer's zarr-vectors datasource
keys segments by ``object_attributes/segment_id[row]`` when the store has that
column (the "standard" object-index convention) and by the dense row otherwise.
:func:`segment_ids_for` applies the same rule, so the ids written into
``segments`` are the ids the viewer draws.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tractography.index import TractIndex

#: ``fetch(url, range_header=None) -> bytes | None`` (see browser_fetch_store).
FetchFn = Callable[..., Awaitable[bytes | None]]
ListFn = Callable[[str], Awaitable[list[str] | None]]

#: Default ceiling on the vertices of the level evaluated for a dissection.
#: ~44 bytes/vertex in a TractIndex, so 2M vertices is ~90 MB of WASM heap.
DEFAULT_VERTEX_BUDGET = 2_000_000

_SCHEME_PREFIX = re.compile(r"^(zarr-vectors|zarr3?)://")
_PIPELINE_SUFFIX = re.compile(r"\|zarr-vectors\b.*$")

_UNIT_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimetre": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "micrometer": 1e-6,
    "micrometre": 1e-6,
    "micron": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometre": 1e-9,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "km": 1e3,
}


def unit_to_meters(unit: str | None, default: float = 1e-3) -> float:
    """Length of one ``unit`` in metres (``default`` when unknown/absent)."""
    if not unit:
        return default
    return _UNIT_METERS.get(str(unit).strip().lower(), default)


# -- URLs ---------------------------------------------------------------------


def store_path_from_url(url: str) -> str:
    """Strip Neuroglancer's datasource decoration from a layer source URL.

    Handles both the ``zarr-vectors://<store>`` prefix form and the KvStore
    pipeline form ``<store>/|zarr-vectors:``.
    """
    url = _SCHEME_PREFIX.sub("", url, count=1)
    url = _PIPELINE_SUFFIX.sub("", url)
    return url


def http_base(url: str) -> str:
    """The fetchable HTTPS base (with trailing slash) for a store URL.

    ``gs://bucket/path`` becomes ``https://storage.googleapis.com/bucket/path/``
    -- ``pyfetch`` cannot read ``gs://``.  Other URLs pass through.
    """
    base = store_path_from_url(url)
    if base.startswith("gs://"):
        base = "https://storage.googleapis.com/" + base[len("gs://") :]
    return base if base.endswith("/") else base + "/"


_GCS_OBJECT_URL = re.compile(r"^https://storage\.googleapis\.com/([^/]+)/(.*)$")


def gcs_list_url(dir_url: str) -> str | None:
    """The GCS JSON-API listing URL for a ``https://storage.googleapis.com``
    directory, or None for any other host."""
    from urllib.parse import quote

    m = _GCS_OBJECT_URL.match(dir_url)
    if m is None:
        return None
    bucket, prefix = m.group(1), m.group(2)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return (
        f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o"
        f"?prefix={quote(prefix, safe='')}&delimiter=%2F"
        f"&fields=prefixes%2Citems(name)%2CnextPageToken"
    )


def gcs_children(page: dict, prefix: str) -> list[str]:
    """Immediate child names from one JSON-API listing page."""
    names: set[str] = set()
    for p in page.get("prefixes", []) or []:
        rest = p[len(prefix) :].strip("/")
        if rest:
            names.add(rest.split("/")[0])
    for item in page.get("items", []) or []:
        rest = item.get("name", "")[len(prefix) :]
        if rest and "/" not in rest:
            names.add(rest)
    return sorted(names)


# -- fetching -------------------------------------------------------------------


async def pyodide_fetch(url: str, range_header: str | None = None) -> bytes | None:
    from pyodide.http import pyfetch  # type: ignore[import-not-found]

    from .tractography.zvf_pure import slice_for_range_header

    try:
        if range_header:
            response = await pyfetch(url, headers={"Range": range_header})
        else:
            response = await pyfetch(url)
    except AttributeError:
        # pyodide.http raises AbortError(reason) on a failed fetch and its
        # __init__ reads `reason.message`, which throws a bare AttributeError
        # when the abort carries no reason (network error, CORS, OOM).
        raise OSError(
            f"fetch of {url} failed (network error, CORS, or the response was "
            f"too large to load into browser memory)"
        ) from None
    if response.status == 404:
        return None
    if response.status in (401, 403) and "storage.googleapis.com" in url:
        # GCS answers a missing object in a non-listable public bucket with 403.
        return None
    if response.status == 416:
        return b""
    if response.status >= 400:
        raise OSError(f"HTTP {response.status} fetching {url}")
    try:
        data = await response.bytes()
    except AttributeError:
        raise OSError(
            f"reading the body of {url} failed -- the fetch was aborted, often "
            f"because the object is too large to hold in browser memory."
        ) from None
    if range_header and response.status != 206:
        data = slice_for_range_header(data, range_header)
    return data


async def local_fetch(path: str, range_header: str | None = None) -> bytes | None:
    """Fetch from the local filesystem (tests / CPython scripts)."""
    import os

    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        if not range_header:
            return fh.read()
        spec = range_header.split("=", 1)[1]
        first, _, last = spec.partition("-")
        if first == "":
            size = os.path.getsize(path)
            fh.seek(max(0, size - int(last)))
            return fh.read()
        fh.seek(int(first))
        if last == "":
            return fh.read()
        return fh.read(int(last) + 1 - int(first))


async def urllib_fetch(url: str, range_header: str | None = None) -> bytes | None:
    """CPython-only fallback (tests, scripts outside the browser)."""
    import urllib.error
    import urllib.request

    from .tractography.zvf_pure import slice_for_range_header

    if "://" not in url:
        return await local_fetch(url, range_header)

    def get():
        request = urllib.request.Request(url)
        if range_header:
            request.add_header("Range", range_header)
        try:
            with urllib.request.urlopen(request) as response:
                data = response.read()
                if range_header and response.status != 206:
                    data = slice_for_range_header(data, range_header)
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    return await asyncio.to_thread(get)


def default_fetch() -> FetchFn:
    try:
        import pyodide.http  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return urllib_fetch
    return pyodide_fetch


def make_lister(fetch: FetchFn) -> ListFn:
    """A lister for GCS (JSON API) and local directories; None elsewhere."""

    async def lister(dir_url: str) -> list[str] | None:
        import os

        if "://" not in dir_url:
            return sorted(os.listdir(dir_url)) if os.path.isdir(dir_url) else None
        list_url = gcs_list_url(dir_url)
        if list_url is None:
            return None
        m = _GCS_OBJECT_URL.match(dir_url)
        assert m is not None
        prefix = m.group(2)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        names: list[str] = []
        token = None
        while True:
            url = list_url + (f"&pageToken={token}" if token else "")
            raw = await fetch(url)
            if raw is None:
                return None
            page = json.loads(raw.decode("utf-8"))
            names.extend(gcs_children(page, prefix))
            token = page.get("nextPageToken")
            if not token:
                break
        return sorted(set(names))

    return lister


# -- store handles --------------------------------------------------------------


@dataclass
class LevelInfo:
    level: int
    vertex_count: int
    object_sparsity: float
    num_objects: int


@dataclass
class StoreInfo:
    base: str
    geometry_types: list[str]
    units: str | None
    unit_meters: float
    axes: list[str]
    bounds: Any
    levels: list[LevelInfo]
    chunk_shape: list[float] | None = None
    object_attributes: list[dict] = field(default_factory=list)
    vertex_attributes: list[str] = field(default_factory=list)
    has_segment_id: bool = False

    def to_json(self) -> dict:
        return {
            "base": self.base,
            "geometryTypes": self.geometry_types,
            "units": self.units,
            "unitMeters": self.unit_meters,
            "axes": self.axes,
            "bounds": self.bounds,
            "chunkShape": self.chunk_shape,
            "levels": [
                {
                    "level": lv.level,
                    "vertexCount": lv.vertex_count,
                    "objectSparsity": lv.object_sparsity,
                    "numObjects": lv.num_objects,
                }
                for lv in self.levels
            ],
            "objectAttributes": self.object_attributes,
            "vertexAttributes": self.vertex_attributes,
            "hasSegmentId": self.has_segment_id,
        }

    def choose_level(self, budget: int = DEFAULT_VERTEX_BUDGET) -> int:
        """The finest level whose vertex count fits ``budget`` (else coarsest)."""
        if not self.levels:
            return 0
        fitting = [lv.level for lv in self.levels if 0 < lv.vertex_count <= budget]
        if fitting:
            return min(fitting)
        return max(lv.level for lv in self.levels)


class ZvStore:
    """One opened store: root handle plus caches, keyed by its HTTP base."""

    def __init__(self, base: str, fetch: FetchFn, lister: ListFn | None):
        self.base = base
        self.fetch = fetch
        self.lister = lister
        self._root = None
        self._info: StoreInfo | None = None
        self._segment_ids: np.ndarray | None | bool = False  # False = unprobed

    async def root(self):
        if self._root is None:
            from zarr_vectors.core.aio import open_store_async

            from .tractography.browser_fetch_store import make_browser_fetch_store

            store = make_browser_fetch_store(self.base, self.fetch, self.lister)
            self._root = await open_store_async(store)
        return self._root

    async def read(self, reader: Callable[..., Any], **kwargs) -> Any:
        from zarr_vectors.core.aio import read_async

        return await read_async(reader, await self.root(), **kwargs)

    async def info(self) -> StoreInfo:
        if self._info is None:
            raw = await self.read(_read_store_info)
            info = StoreInfo(base=self.base, **raw)
            if self.lister is not None:
                try:
                    info.object_attributes = await self.read(
                        _read_object_attribute_catalog, level=0
                    )
                except Exception:  # noqa: BLE001 - discovery is best-effort
                    info.object_attributes = []
                try:
                    info.vertex_attributes = await self.read(
                        _list_children, path="0/vertex_attributes"
                    )
                except Exception:  # noqa: BLE001
                    info.vertex_attributes = []
            info.has_segment_id = (await self.segment_id_table()) is not None
            self._info = info
        return self._info

    async def segment_id_table(self) -> np.ndarray | None:
        """``object_attributes/segment_id`` at level 0 (row -> segment id)."""
        if self._segment_ids is False:
            try:
                table = await self.read(
                    _read_object_attribute_rows, level=0, name="segment_id"
                )
                self._segment_ids = None if table is None else table[1]
            except Exception:  # noqa: BLE001 - absent column
                self._segment_ids = None
        return self._segment_ids  # type: ignore[return-value]

    async def level_geometry(self, level: int) -> tuple[TractIndex, np.ndarray]:
        """``(index, rows)``: one TractIndex row per object (fragments
        concatenated in manifest order) and each object's index row."""
        ids, rows, positions, counts = await self.read(
            _read_level_geometry, level=level
        )
        offsets = np.zeros(counts.size + 1, dtype=np.intp)
        np.cumsum(counts, out=offsets[1:])
        index = TractIndex(positions, offsets, ids.astype(np.uint64))
        # TractIndex sorts/uniques ids; rows follow the same order.
        order = np.argsort(ids, kind="stable")
        return index, rows[order]

    async def polylines(
        self, level: int, ids: list[int] | None = None
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """``(object_ids, polylines)``: one concatenated polyline per object.

        ``ids=None`` reads the whole level; otherwise only those objects'
        manifests and the chunks they reference (ids absent at ``level`` are
        dropped).
        """
        if ids is None:
            found, _rows, positions, counts = await self.read(
                _read_level_geometry, level=level
            )
        else:
            found, positions, counts = await self.read(
                _read_objects_geometry, level=level, ids=[int(i) for i in ids]
            )
        offsets = np.zeros(counts.size + 1, dtype=np.intp)
        np.cumsum(counts, out=offsets[1:])
        return found, [
            positions[offsets[i] : offsets[i + 1]] for i in range(counts.size)
        ]

    async def object_attribute(
        self, level: int, name: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``(object_ids, values)`` for one object attribute, or None."""
        try:
            return await self.read(_read_object_attribute_rows, level=level, name=name)
        except Exception:  # noqa: BLE001
            if level != 0:
                return await self.object_attribute(0, name)
            return None

    async def segment_ids_for(self, object_ids: np.ndarray) -> np.ndarray:
        """Map zarr-vectors object ids to the segment ids the viewer draws."""
        table = await self.segment_id_table()
        object_ids = np.asarray(object_ids, dtype=np.uint64)
        if table is None:
            return object_ids
        rows = await self.read(
            _rows_for_object_ids, level=0, ids=[int(i) for i in object_ids]
        )
        return np.asarray(table, dtype=np.uint64)[rows]


# -- readers (replayed offline by read_async; take the root Group first) ------


def _read_store_info(root) -> dict:
    from zarr_vectors.core.store import (
        get_resolution_level,
        read_level_metadata,
        read_root_metadata,
    )

    meta = read_root_metadata(root)
    crs = getattr(meta, "crs", None) or {}
    units = crs.get("units") if isinstance(crs, dict) else None
    axes_meta = list(getattr(meta, "spatial_index_dims", None) or [])
    axes = []
    for a in axes_meta:
        if isinstance(a, dict):
            axes.append(str(a.get("name", "")))
            if units is None and a.get("unit"):
                units = a.get("unit")
    if not axes:
        axes = ["x", "y", "z"][: int(getattr(meta, "sid_ndim", 3) or 3)]
    levels = []
    level_index = 0
    while True:
        try:
            lm = read_level_metadata(root, level_index)
        except Exception:  # noqa: BLE001 - past the last level
            break
        try:
            num_objects = int(
                get_resolution_level(root, level_index)
                .read_array_meta("object_index")
                .get("num_objects", 0)
            )
        except Exception:  # noqa: BLE001 - point clouds have no object index
            num_objects = 0
        levels.append(
            LevelInfo(
                level=level_index,
                vertex_count=int(getattr(lm, "vertex_count", 0) or 0),
                object_sparsity=float(getattr(lm, "object_sparsity", 1.0) or 1.0),
                num_objects=num_objects,
            )
        )
        level_index += 1
        if level_index > 64:
            break
    bounds = getattr(meta, "bounds", None)
    return {
        "geometry_types": list(getattr(meta, "geometry_types", None) or []),
        "units": units,
        "unit_meters": unit_to_meters(units),
        "axes": axes,
        "bounds": None if bounds is None else [list(map(float, b)) for b in bounds],
        "levels": levels,
        "chunk_shape": [float(v) for v in (getattr(meta, "chunk_shape", None) or ())]
        or None,
    }


def _list_children(root, *, path: str) -> list[str]:
    node = root
    for part in path.split("/"):
        node = node[part]
    return sorted(node.children())


def _read_object_attribute_catalog(root, *, level: int) -> list[dict]:
    from zarr_vectors.core.store import get_resolution_level

    level_group = get_resolution_level(root, level)
    try:
        names = sorted(level_group["object_attributes"].children())
    except Exception:  # noqa: BLE001
        return []
    out = []
    for name in names:
        try:
            meta = level_group.read_array_meta(f"object_attributes/{name}")
        except Exception:  # noqa: BLE001
            continue
        shape = list(meta.get("shape", []) or [])
        out.append(
            {
                "name": name,
                "dtype": str(meta.get("dtype", "")),
                "ncols": int(shape[1]) if len(shape) > 1 else 1,
            }
        )
    return out


def _read_object_attribute_rows(root, *, level: int, name: str):
    from zarr_vectors.core.arrays import object_ids_for_rows, read_object_attributes
    from zarr_vectors.core.store import get_resolution_level

    level_group = get_resolution_level(root, level)
    values = np.asarray(read_object_attributes(level_group, name))
    ids = np.asarray(object_ids_for_rows(level_group), dtype=np.int64)
    n = min(ids.size, values.shape[0])
    return ids[:n], values[:n]


def _rows_for_object_ids(root, *, level: int, ids: list[int]) -> np.ndarray:
    from zarr_vectors.core.arrays import object_rows_for_ids
    from zarr_vectors.core.store import get_resolution_level

    level_group = get_resolution_level(root, level)
    found, rows = object_rows_for_ids(level_group, ids)
    if found.size != len(ids):
        raise KeyError(
            f"{len(ids) - found.size} object ids are not in object_index at level {level}"
        )
    return np.asarray(rows, dtype=np.int64)


def _read_level_geometry(root, *, level: int):
    """Positions of every object at ``level``, fragments joined per object.

    Mirrors the full-read branch of ``read_polylines`` (manifests are the
    authority on which chunks an object's fragments live in) without touching
    attributes, so it needs no listing.
    """
    from zarr_vectors.core.arrays import read_chunk_vertices, read_object_manifest_rows
    from zarr_vectors.core.store import get_resolution_level, read_root_metadata
    from zarr_vectors.exceptions import ArrayError

    meta = read_root_metadata(root)
    ndim = int(getattr(meta, "sid_ndim", 3) or 3)
    level_group = get_resolution_level(root, level)
    ids, manifests = read_object_manifest_rows(level_group)

    needed: set = set()
    for m in manifests:
        for cc, _fi in m or ():
            needed.add(tuple(cc))
    chunk_iter = sorted(needed)
    keys = [".".join(str(c) for c in cc) for cc in chunk_iter]
    chunk_cache: dict = {}
    # Under the offline replay, `batched_reads` records EVERY planned cell the
    # snapshot lacks as a miss in one pass (otherwise each round would discover
    # one cell and a level with many cells would never converge).
    with level_group.batched_reads([("vertices", keys), ("vertex_fragments", keys)]):
        for cc in chunk_iter:
            try:
                chunk_cache[cc] = read_chunk_vertices(level_group, cc, ndim=ndim)
            except ArrayError:
                chunk_cache[cc] = []

    out_ids: list[int] = []
    out_rows: list[int] = []
    blocks: list[np.ndarray] = []
    counts: list[int] = []
    for row, (oid, manifest) in enumerate(zip(ids.tolist(), manifests)):
        if not manifest:
            continue
        frags = []
        for cc, fi in manifest:
            group = chunk_cache.get(tuple(cc))
            if group is not None and 0 <= fi < len(group):
                frags.append(np.asarray(group[fi], dtype=np.float32).reshape(-1, ndim))
        if not frags:
            continue
        joined = frags[0] if len(frags) == 1 else np.concatenate(frags, axis=0)
        if joined.shape[0] == 0:
            continue
        out_ids.append(int(oid))
        out_rows.append(row)
        blocks.append(joined)
        counts.append(joined.shape[0])
    positions = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.zeros((0, ndim), dtype=np.float32)
    )
    return (
        np.asarray(out_ids, dtype=np.int64),
        np.asarray(out_rows, dtype=np.int64),
        positions.astype(np.float32, copy=False),
        np.asarray(counts, dtype=np.intp),
    )


def _read_objects_geometry(root, *, level: int, ids: list[int]):
    """Like :func:`_read_level_geometry` for an explicit object subset."""
    from zarr_vectors.core.arrays import read_chunk_vertices, read_object_manifests
    from zarr_vectors.core.store import get_resolution_level, read_root_metadata
    from zarr_vectors.exceptions import ArrayError

    meta = read_root_metadata(root)
    ndim = int(getattr(meta, "sid_ndim", 3) or 3)
    level_group = get_resolution_level(root, level)
    by_oid = read_object_manifests(level_group, ids=ids)
    needed = sorted({tuple(cc) for m in by_oid.values() for cc, _ in (m or ())})
    keys = [".".join(str(c) for c in cc) for cc in needed]
    chunk_cache: dict = {}
    with level_group.batched_reads([("vertices", keys), ("vertex_fragments", keys)]):
        for cc in needed:
            try:
                chunk_cache[cc] = read_chunk_vertices(level_group, cc, ndim=ndim)
            except ArrayError:
                chunk_cache[cc] = []
    out_ids: list[int] = []
    blocks: list[np.ndarray] = []
    counts: list[int] = []
    for oid in ids:
        manifest = by_oid.get(int(oid))
        if not manifest:
            continue
        frags = []
        for cc, fi in manifest:
            group = chunk_cache.get(tuple(cc))
            if group is not None and 0 <= fi < len(group):
                frags.append(np.asarray(group[fi], dtype=np.float32).reshape(-1, ndim))
        if not frags:
            continue
        joined = np.concatenate(frags, axis=0)
        if joined.shape[0] == 0:
            continue
        out_ids.append(int(oid))
        blocks.append(joined)
        counts.append(joined.shape[0])
    positions = (
        np.concatenate(blocks, axis=0) if blocks else np.zeros((0, ndim), np.float32)
    )
    return (
        np.asarray(out_ids, dtype=np.int64),
        positions.astype(np.float32, copy=False),
        np.asarray(counts, dtype=np.intp),
    )


# -- registry -------------------------------------------------------------------

_stores: dict[str, ZvStore] = {}


def open_store(
    url: str, fetch: FetchFn | None = None, lister: ListFn | None | bool = True
) -> ZvStore:
    """A cached :class:`ZvStore` for ``url`` (any Neuroglancer source spelling).

    ``lister=True`` builds the default GCS/local lister; ``None``/``False``
    disables listing.
    """
    base = http_base(url)
    store = _stores.get(base)
    if store is None or (fetch is not None and store.fetch is not fetch):
        fetch = fetch or default_fetch()
        if lister is True:
            lister_fn: ListFn | None = make_lister(fetch)
        elif lister is False or lister is None:
            lister_fn = None
        else:
            lister_fn = lister
        store = _stores[base] = ZvStore(base, fetch, lister_fn)
    return store


def forget_stores() -> None:
    _stores.clear()
