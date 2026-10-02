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

"""Parcellation (label volume) sampling for the label filter.

The old viewer sampled a linked segmentation layer's *rendered* chunks in
TypeScript.  ngpy cannot reach a viewer's chunk cache (it only uses public
surfaces), so it reads the parcellation itself, over HTTP, as a whole numpy
array -- a parcellation is small (MNI SynthSeg at 1 mm is 189x233x197 uint16 =
17 MB; its 2 mm scale is 2 MB) -- and samples the label under every vertex of
the evaluation index once.  Label ROIs are then a ``np.isin`` over that array.

Supported sources (the Neuroglancer spellings a segmentation layer uses):

- OME-Zarr multiscale, zarr v2 or v3: ``<url>|zarr2:``, ``<url>|zarr:``,
  ``<url>|zarr3:``, ``zarr://``, ``zarr2://``, ``zarr3://``.  Read through
  zarr's *async* API over :class:`BrowserFetchStore`, so no JSPI is needed.
- Neuroglancer precomputed, ``raw`` encoding, unsharded: ``precomputed://<url>``
  / ``<url>|neuroglancer-precomputed:``.

NOT supported (raises :class:`LabelSourceError` naming the reason):
precomputed ``compressed_segmentation`` / ``jpeg`` encodings and sharded
precomputed volumes, n5, nifti.

Coordinates: tract vertices arrive in the zarr-vectors store's units; both they
and the volume's physical frame are converted to metres and matched by AXIS
NAME (``x``/``y``/``z``).  A Neuroglancer layer ``transform`` on either layer is
NOT applied -- the same assumption the old in-viewer label filter made (the
demo pins the global frame to the tracts' x/y/z in mm).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .zv import FetchFn, default_fetch, unit_to_meters

#: Auto-selection ceiling on the voxels read for one parcellation.
DEFAULT_MAX_VOXELS = 32_000_000


class LabelSourceError(ValueError):
    """A parcellation source ngpy cannot read; the message is user-facing."""


@dataclass
class LabelVolume:
    """A dense label array plus its voxel -> physical (metres) mapping."""

    data: np.ndarray
    axes: list[str]
    #: Per array axis: physical metres of voxel ``i`` is ``origin + i * scale``
    #: (``origin`` is the CENTRE of voxel 0).
    scale_m: np.ndarray
    origin_m: np.ndarray
    url: str
    description: str = ""

    def sample(self, positions_m: np.ndarray, axes: list[str]) -> np.ndarray:
        """Label under each point of ``positions_m`` (N, len(axes)); 0 outside."""
        positions_m = np.asarray(positions_m, dtype=np.float64)
        n = positions_m.shape[0]
        idx = np.zeros((n, self.data.ndim), dtype=np.int64)
        inside = np.ones(n, dtype=bool)
        lower_axes = [a.lower() for a in axes]
        for d, name in enumerate(self.axes):
            try:
                src = lower_axes.index(name.lower())
            except ValueError:
                if self.data.shape[d] == 1:
                    continue  # a singleton axis (e.g. channel) the points lack
                raise LabelSourceError(
                    f"parcellation axis {name!r} has no matching tract axis "
                    f"(tract axes: {axes})"
                ) from None
            i = np.rint((positions_m[:, src] - self.origin_m[d]) / self.scale_m[d])
            i = i.astype(np.int64)
            inside &= (i >= 0) & (i < self.data.shape[d])
            idx[:, d] = i
        out = np.zeros(n, dtype=np.int64)
        if inside.any():
            sel = tuple(idx[inside, d] for d in range(self.data.ndim))
            out[inside] = self.data[sel].astype(np.int64, copy=False)
        return out


_PIPE = re.compile(r"\|(zarr2|zarr3|zarr|neuroglancer-precomputed)\b.*$")
_PREFIX = re.compile(r"^(zarr2|zarr3|zarr|precomputed)://")


def parse_volume_source(url: str) -> tuple[str, str]:
    """``(kind, http_base)`` for a Neuroglancer volume source spelling.

    ``kind`` is ``"zarr"`` (OME-Zarr, v2 or v3) or ``"precomputed"``.
    """
    kind = None
    m = _PIPE.search(url)
    if m is not None:
        kind = "precomputed" if m.group(1) == "neuroglancer-precomputed" else "zarr"
        url = url[: m.start()]
    m = _PREFIX.match(url)
    if m is not None:
        kind = "precomputed" if m.group(1) == "precomputed" else "zarr"
        url = url[m.end() :]
    if kind is None:
        if ".zarr" in url:
            kind = "zarr"
        else:
            raise LabelSourceError(
                f"cannot tell the volume format of {url!r}; use a zarr or "
                f"precomputed source"
            )
    if url.startswith("gs://"):
        url = "https://storage.googleapis.com/" + url[len("gs://") :]
    if not url.endswith("/"):
        url += "/"
    return kind, url


async def _fetch_json(fetch: FetchFn, url: str) -> dict | None:
    raw = await fetch(url)
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


async def open_label_volume(
    url: str,
    *,
    fetch: FetchFn | None = None,
    scale_index: int | None = None,
    max_voxels: int = DEFAULT_MAX_VOXELS,
) -> LabelVolume:
    """Read a whole parcellation scale into memory."""
    fetch = fetch or default_fetch()
    kind, base = parse_volume_source(url)
    if kind == "precomputed":
        return await _open_precomputed(base, fetch, scale_index, max_voxels)
    return await _open_ome_zarr(base, fetch, scale_index, max_voxels)


def _pick_scale(shapes: list[tuple[int, ...]], scale_index, max_voxels) -> int:
    if scale_index is not None:
        if not 0 <= scale_index < len(shapes):
            raise LabelSourceError(
                f"scale {scale_index} out of range (have {len(shapes)})"
            )
        return scale_index
    for i, shape in enumerate(shapes):
        if int(np.prod(shape)) <= max_voxels:
            return i
    return len(shapes) - 1


async def _open_ome_zarr(base, fetch, scale_index, max_voxels) -> LabelVolume:
    import zarr

    from .tractography.browser_fetch_store import make_browser_fetch_store

    zarr_format = 2
    attrs = await _fetch_json(fetch, base + ".zattrs")
    if attrs is None:
        meta = await _fetch_json(fetch, base + "zarr.json")
        if meta is None:
            raise LabelSourceError(f"no OME-Zarr metadata at {base}")
        zarr_format = 3
        attrs = meta.get("attributes", {})
        attrs = attrs.get("ome", attrs)
    multiscales = attrs.get("multiscales")
    store = make_browser_fetch_store(base, fetch)
    if not multiscales:
        # A bare array: index space is physical space, unit-less.
        array = await zarr.api.asynchronous.open_array(
            store=store, zarr_format=zarr_format, mode="r"
        )
        data = np.asarray(await array.getitem(tuple(slice(None) for _ in array.shape)))
        names = ["z", "y", "x"][-data.ndim :] if data.ndim <= 3 else []
        return LabelVolume(
            data,
            names,
            np.full(data.ndim, 1e-3),
            np.zeros(data.ndim),
            base,
            "bare array",
        )
    ms = multiscales[0]
    axes_meta = ms.get("axes") or [{"name": n} for n in ("z", "y", "x")]
    axes = [a["name"] if isinstance(a, dict) else str(a) for a in axes_meta]
    units = [
        unit_to_meters(a.get("unit") if isinstance(a, dict) else None, default=1e-6)
        for a in axes_meta
    ]
    datasets = ms.get("datasets", [])
    if not datasets:
        raise LabelSourceError(f"OME-Zarr at {base} lists no datasets")
    arrays = []
    for ds in datasets:
        arr = await zarr.api.asynchronous.open_array(
            store=store, path=ds["path"], zarr_format=zarr_format, mode="r"
        )
        arrays.append(arr)
    choice = _pick_scale([tuple(a.shape) for a in arrays], scale_index, max_voxels)
    array = arrays[choice]
    data = np.asarray(await array.getitem(tuple(slice(None) for _ in array.shape)))
    scale = np.ones(len(axes))
    trans = np.zeros(len(axes))
    for t in datasets[choice].get("coordinateTransformations", []):
        if t.get("type") == "scale":
            scale = scale * np.asarray(t["scale"], dtype=np.float64)
        elif t.get("type") == "translation":
            trans = trans + np.asarray(t["translation"], dtype=np.float64)
    for t in ms.get("coordinateTransformations", []) or []:
        if t.get("type") == "scale":
            s = np.asarray(t["scale"], dtype=np.float64)
            scale, trans = scale * s, trans * s
        elif t.get("type") == "translation":
            trans = trans + np.asarray(t["translation"], dtype=np.float64)
    keep = [
        i
        for i, a in enumerate(axes_meta)
        if not isinstance(a, dict) or a.get("type", "space") == "space"
    ]
    if len(keep) != len(axes):
        # Drop non-space axes (channel/time) by taking index 0 along them.
        sel = tuple(slice(None) if i in keep else 0 for i in range(len(axes)))
        data = data[sel]
        axes = [axes[i] for i in keep]
        scale, trans = scale[keep], trans[keep]
        units = [units[i] for i in keep]
    unit = np.asarray(units)
    return LabelVolume(
        data,
        axes,
        scale * unit,
        trans * unit,
        base,
        f"OME-Zarr v{zarr_format} scale {choice} {tuple(data.shape)}",
    )


async def _open_precomputed(base, fetch, scale_index, max_voxels) -> LabelVolume:
    info = await _fetch_json(fetch, base + "info")
    if info is None:
        raise LabelSourceError(f"no precomputed info at {base}")
    if info.get("@type") == "neuroglancer_segment_properties":
        raise LabelSourceError(
            f"{base} is a segment-properties source (label names), not a volume"
        )
    scales = info.get("scales") or []
    if not scales:
        raise LabelSourceError(f"precomputed info at {base} has no scales")
    choice = _pick_scale(
        [tuple(int(v) for v in s["size"]) for s in scales], scale_index, max_voxels
    )
    s = scales[choice]
    encoding = s.get("encoding", "raw")
    if encoding != "raw":
        raise LabelSourceError(
            f"precomputed encoding {encoding!r} is not supported by the ngpy label "
            f"filter (only 'raw'); convert the parcellation to OME-Zarr"
        )
    if s.get("sharding"):
        raise LabelSourceError("sharded precomputed volumes are not supported")
    dtype = np.dtype(info.get("data_type", "uint32")).newbyteorder("<")
    nc = int(info.get("num_channels", 1))
    size = [int(v) for v in s["size"]]
    offset = [int(v) for v in s.get("voxel_offset", [0, 0, 0])]
    chunk = [int(v) for v in s["chunk_sizes"][0]]
    data = np.zeros((size[2], size[1], size[0]), dtype=dtype)  # z, y, x
    key = s["key"]
    import asyncio

    async def one(x0, y0, z0):
        x1, y1, z1 = (
            min(x0 + chunk[0], size[0]),
            min(y0 + chunk[1], size[1]),
            min(z0 + chunk[2], size[2]),
        )
        name = (
            f"{x0 + offset[0]}-{x1 + offset[0]}_{y0 + offset[1]}-{y1 + offset[1]}_"
            f"{z0 + offset[2]}-{z1 + offset[2]}"
        )
        raw = await fetch(f"{base}{key}/{name}")
        if raw is None:
            return
        block = np.frombuffer(raw, dtype=dtype).reshape(nc, z1 - z0, y1 - y0, x1 - x0)
        data[z0:z1, y0:y1, x0:x1] = block[0]

    jobs = [
        one(x0, y0, z0)
        for x0 in range(0, size[0], chunk[0])
        for y0 in range(0, size[1], chunk[1])
        for z0 in range(0, size[2], chunk[2])
    ]
    await asyncio.gather(*jobs)
    res_m = np.asarray(s["resolution"], dtype=np.float64)[::-1] * 1e-9  # z,y,x
    off = np.asarray(offset, dtype=np.float64)[::-1]
    # Precomputed voxel i spans [i, i+1) * resolution, so its centre is i + 0.5.
    return LabelVolume(
        data,
        ["z", "y", "x"],
        res_m,
        (off + 0.5) * res_m,
        base,
        f"precomputed scale {choice} {tuple(size)}",
    )


async def read_segment_properties(url: str, fetch: FetchFn | None = None) -> dict:
    """Label names/colours from a ``neuroglancer_segment_properties`` source.

    Returns ``{"ids": [...], "names": [...], "colors": [... or None]}``.
    """
    fetch = fetch or default_fetch()
    u = re.sub(r"^precomputed://", "", url)
    u = re.sub(r"\|neuroglancer-precomputed:.*$", "", u)
    if u.startswith("gs://"):
        u = "https://storage.googleapis.com/" + u[len("gs://") :]
    if not u.endswith("/"):
        u += "/"
    info = await _fetch_json(fetch, u + "info")
    if info is None or info.get("@type") != "neuroglancer_segment_properties":
        raise LabelSourceError(f"{url} is not a segment-properties source")
    inline = info.get("inline", {})
    ids = [str(i) for i in inline.get("ids", [])]
    names = list(ids)
    colors: list[str | None] = [None] * len(ids)
    for prop in inline.get("properties", []):
        if prop.get("type") == "label":
            names = [str(v) for v in prop.get("values", names)]
        elif prop.get("type") in ("rgb", "color") or prop.get("id") in ("color", "rgb"):
            values = prop.get("values", [])
            colors = [str(v) if isinstance(v, str) else None for v in values]
    if not any(n != i for n, i in zip(names, ids)):
        for prop in inline.get("properties", []):
            if prop.get("type") == "description":
                names = [str(v) for v in prop.get("values", names)]
                break
    return {"ids": ids, "names": names, "colors": colors}


def sample_vertex_labels(
    volume: LabelVolume,
    positions: np.ndarray,
    store_axes: list[str],
    store_unit_m: float,
) -> np.ndarray:
    """Label under every vertex (store coordinates in store units)."""
    return volume.sample(np.asarray(positions, np.float64) * store_unit_m, store_axes)
