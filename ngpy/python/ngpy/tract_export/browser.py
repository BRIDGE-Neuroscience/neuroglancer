# @license
# Copyright 2026 Google Inc.
#
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
"""In-browser tract export under Pyodide: select, then write, from one async read.

Ported from ``neuroglancer.tract_export.browser`` (``zarr_vectors_roi_store``).

The read goes through :mod:`ngpy.zv` -- zarr-vectors' async prime-and-replay
path, which never reaches zarr's blocking ``sync()`` -- and the finished file
comes back as *bytes* for the page to download or upload to GCS.

Selection: the Export tab normally hands over each group's passing object ids
(``objectIds``, computed by :mod:`ngpy.filter` at the evaluation level), so
this reads exactly those objects -- at the export level, which may be finer --
through the selective manifest path.  A spec without ``objectIds`` folds its
``rois`` over a whole-level read instead.

``.trk`` needs only an ``await`` for the read and a pure-Python writer
(nibabel, micropip-installed on first use).  ``.zvf`` goes through
zarr-vectors' synchronous writer, i.e. zarr ``sync()``, which under Pyodide can
only suspend via WebAssembly stack switching (JSPI): the worker calls
:func:`export_sync` through ``callPromising`` under a one-at-a-time mutex.
Without JSPI a ``.zvf`` export fails with a clear message; ``.trk`` still works.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import numpy as np

from ngpy import zv
from ngpy.tract_export.job import ExportJob
from ngpy.tract_export.run import (
    ExportRunError,
    per_group_from_object_ids,
    select_from,
    union_ids,
    uses_explicit_ids,
)

_MISSING_LIB = (
    "In-browser export needs the `zarr-vectors` package, which is embedded in "
    "ngpy.html; it failed to import."
)

_ZVF_TMP = "/tmp/ngpy_export.zvf"


def http_base(url: str) -> str:
    """Kept for callers of the old module: see :func:`ngpy.zv.http_base`."""
    return zv.http_base(url)


async def export_async(
    job: ExportJob, fetch: zv.FetchFn | None = None
) -> tuple[bytes, str, dict[str, Any]]:
    """Carry out ``job``; returns ``(body, content_type, summary)``.

    * TRK -> the ``.trk`` bytes, ``application/octet-stream``;
    * ZVF -> the zipped store, ``application/zip``;
    * nothing selected -> the summary as JSON, ``application/json``.
    """
    try:
        store = zv.open_store(job.source_url, fetch=fetch)
    except Exception as e:  # noqa: BLE001
        raise ExportRunError(f"{_MISSING_LIB} ({e})") from e

    per_group: list[tuple[str, np.ndarray]]
    if job.scope == "whole":
        ids_read, streamlines = await store.polylines(job.level)
        ids = np.asarray(ids_read, dtype=np.uint64)
        considered = int(ids.size)
        per_group = []
    elif uses_explicit_ids(job.groups):
        per_group = per_group_from_object_ids(job.groups)
        ids = union_ids(per_group)
        considered = int(ids.size)
        if ids.size:
            found, streamlines = await store.polylines(
                job.level, [int(i) for i in ids]
            )
            ids = np.asarray(found, dtype=np.uint64)
        else:
            streamlines = []
    else:
        ids_read, streamlines_all = await store.polylines(job.level)
        polylines = [[p] for p in streamlines_all]
        considered, per_group = select_from(
            polylines, list(ids_read), job.groups, release=False
        )
        ids = union_ids(per_group)
        position = {int(o): i for i, o in enumerate(ids_read)}
        streamlines = [streamlines_all[position[int(o)]] for o in ids]

    summary: dict[str, Any] = {
        "object_count": int(ids.size),
        "candidate_object_count": considered,
        "groups": [{"name": name, "count": int(v.size)} for name, v in per_group],
        "output_path": job.destination.path,
        "level": job.level,
    }
    if ids.size == 0 or not streamlines:
        summary.update(
            written=False,
            streamline_count=0,
            vertex_count=0,
            message="No streamlines passed this dissection; nothing written.",
        )
        return json.dumps(summary).encode("utf-8"), "application/json", summary

    summary.update(
        written=True,
        streamline_count=len(streamlines),
        vertex_count=int(sum(len(s) for s in streamlines)),
    )
    if job.format == "trk":
        await _ensure_nibabel()
        body, content_type = _write_trk_bytes(streamlines, job.affine)
    else:
        info = await store.info()
        chunk_shape = info.chunk_shape
        if not chunk_shape:
            raise ExportRunError("store metadata has no chunk_shape")
        try:
            body, content_type = _write_zvf_zip_bytes(streamlines, chunk_shape)
        except ExportRunError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(k in msg.lower() for k in ("stack switching", "suspend", "jspi")):
                raise ExportRunError(
                    "ZVF export needs WebAssembly stack switching (JSPI), which "
                    "this browser does not support. Export as TRK instead."
                ) from e
            raise ExportRunError(f"ZVF export failed: {msg}") from e
    return body, content_type, summary


async def _ensure_nibabel() -> None:
    """Make ``nibabel`` importable, micropip-installing it under Pyodide."""
    try:
        import nibabel  # noqa: F401

        return
    except ImportError:
        pass
    try:
        import micropip  # type: ignore[import-not-found]
    except ImportError as e:
        raise ExportRunError(
            "TRK export needs `nibabel`, which is not installed and micropip is "
            "unavailable to fetch it."
        ) from e
    await micropip.install("nibabel")


def _write_trk_bytes(
    streamlines: list[np.ndarray], affine: np.ndarray | None
) -> tuple[bytes, str]:
    """A TrackVis ``.trk`` in memory, with a header tools can place.

    ``affine`` maps the store's coordinates to RAS millimetres (the Export tab
    pre-fills it from the store's unit); identity for a store already in mm.
    """
    try:
        import nibabel as nib
        from nibabel.streamlines.trk import TrkFile
    except ImportError as e:
        raise ExportRunError("TRK export needs `nibabel`.") from e

    aff = (
        np.eye(4, dtype=np.float32)
        if affine is None
        else np.asarray(affine, dtype=np.float32)
    )
    header = _trk_header(aff, streamlines)
    tractogram = nib.streamlines.Tractogram(
        streamlines=streamlines, affine_to_rasmm=aff
    )
    buf = io.BytesIO()
    TrkFile(tractogram=tractogram, header=header).save(buf)
    return buf.getvalue(), "application/octet-stream"


def _trk_header(aff: np.ndarray, streamlines: list[np.ndarray]) -> dict:
    """Non-degenerate voxel size / dimensions / voxel->RAS for the TRK header."""
    from nibabel.streamlines import Field

    voxel_sizes = np.linalg.norm(aff[:3, :3], axis=0).astype(np.float32)
    if streamlines:
        all_pts = np.concatenate(streamlines, axis=0)
        maxc = np.ceil(np.abs(all_pts).max(axis=0)).astype(np.int64) + 1
        dims = np.clip(maxc, 1, 32767).astype(np.int16)
    else:
        dims = np.array([1, 1, 1], dtype=np.int16)
    return {
        Field.VOXEL_SIZES: tuple(float(v) for v in voxel_sizes),
        Field.DIMENSIONS: tuple(int(v) for v in dims),
        Field.VOXEL_TO_RASMM: aff,
    }


def _write_zvf_zip_bytes(
    streamlines: list[np.ndarray], chunk_shape
) -> tuple[bytes, str]:
    """Write a fresh zarr-vectors store to MEMFS and return it zipped."""
    try:
        from zarr_vectors.types.polylines import write_polylines
    except Exception as e:  # noqa: BLE001
        raise ExportRunError(_MISSING_LIB) from e
    _remove(_ZVF_TMP)
    write_polylines(_ZVF_TMP, streamlines, chunk_shape=tuple(chunk_shape))
    try:
        return _zip_dir_bytes(_ZVF_TMP), "application/zip"
    finally:
        _remove(_ZVF_TMP)


def _zip_dir_bytes(path: str) -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, path))
    return buf.getvalue()


def _remove(path: str) -> None:
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
