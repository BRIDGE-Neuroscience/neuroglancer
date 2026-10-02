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

"""Whole-store ROI / label / attribute dissection for the wrapper's Filter tab.

The request is the wrapper's group list in the SAME persistence JSON the old
viewer wrote (``groupToJson``: shapes keyed by ``type``, string predicate /
operator names), already converted into the store's coordinate frame by the
page.  The answer is what the page writes into ORDINARY segmentation-layer
state:

- ``segments``  -- union of the objects passing any visible group, as the
  segment ids the viewer draws (decimal strings: ids are uint64);
- ``colors``    -- ``{segment id: "#rrggbb"}``, the colour of the FIRST visible
  group (in list order) each passing object belongs to;
- ``active``    -- whether any visible group selects anything (has a region or
  an attribute predicate).  An inactive filter means "show everything".

Per group, membership is the ROI fold (:func:`streamlines_pass_rois`, which
includes label masks sampled from a parcellation) ANDed with every attribute
range.  A group with predicates and no regions is a pure attribute group; a
group with neither selects nothing and does not make the filter active.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import labels as labels_mod
from . import zv
from .tract_export.job import JobSpecError, parse_roi
from .tractography.index import TractIndex
from .tractography.roi import LabelMask, streamlines_pass_rois


@dataclass
class _LevelCache:
    index: TractIndex
    segment_ids: np.ndarray
    vertex_labels: dict[str, np.ndarray]
    object_attrs: dict[str, np.ndarray]


class FilterEngine:
    """Caches each (store, level) evaluation index between requests.

    The first evaluation of a store reads a whole pyramid level (seconds); each
    later one -- an ROI dragged, a group toggled -- is pure numpy over the
    cached index (tens of milliseconds for a million vertices).
    """

    def __init__(self, fetch=None):
        self._fetch = fetch
        self._levels: dict[tuple[str, int, int | None], _LevelCache] = {}
        self._volumes: dict[str, labels_mod.LabelVolume] = {}

    def forget(self) -> None:
        self._levels.clear()
        self._volumes.clear()

    async def store_info(self, source: str) -> dict:
        store = zv.open_store(source, fetch=self._fetch)
        info = await store.info()
        out = info.to_json()
        out["defaultLevel"] = info.choose_level()
        return out

    async def _level(
        self, source: str, level: int | None, budget: int, max_vertices: int | None
    ) -> tuple[zv.ZvStore, zv.StoreInfo, int, _LevelCache]:
        store = zv.open_store(source, fetch=self._fetch)
        info = await store.info()
        if level is None:
            level = info.choose_level(budget)
        key = (store.base, int(level), max_vertices)
        cache = self._levels.get(key)
        if cache is None:
            index, _rows = await store.level_geometry(int(level))
            if max_vertices:
                index = index.decimate(int(max_vertices))
            segment_ids = await store.segment_ids_for(index.object_ids)
            cache = _LevelCache(index, segment_ids, {}, {})
            # Keep a handful of levels; a new store/level evicts the oldest.
            while len(self._levels) >= 4:
                self._levels.pop(next(iter(self._levels)))
            self._levels[key] = cache
        return store, info, int(level), cache

    async def _vertex_labels(
        self, cache: _LevelCache, info: zv.StoreInfo, parcellation: dict
    ) -> np.ndarray:
        url = parcellation["url"]
        scale = parcellation.get("scale")
        vkey = f"{url}#{scale}"
        labels = cache.vertex_labels.get(vkey)
        if labels is None:
            volume = self._volumes.get(vkey)
            if volume is None:
                volume = await labels_mod.open_label_volume(
                    url, fetch=self._fetch, scale_index=scale
                )
                self._volumes[vkey] = volume
            labels = labels_mod.sample_vertex_labels(
                volume, cache.index.positions, info.axes, info.unit_meters
            )
            cache.vertex_labels[vkey] = labels
        return labels

    async def _object_attr(
        self, store: zv.ZvStore, level: int, cache: _LevelCache, name: str
    ) -> np.ndarray:
        """One value per index object (NaN where the store has none)."""
        values = cache.object_attrs.get(name)
        if values is None:
            column = name
            component = None
            if name.endswith("]") and "[" in name:
                column, _, comp = name[:-1].partition("[")
                component = int(comp)
            got = await store.object_attribute(level, column)
            if got is None:
                raise JobSpecError(f"object attribute {column!r} not found")
            ids, vals = got
            vals = np.asarray(vals, dtype=np.float64)
            if vals.ndim > 1:
                if component is None:
                    raise JobSpecError(
                        f"object attribute {column!r} has {vals.shape[1]} columns; "
                        f"filter one with {column}[i]"
                    )
                vals = vals[:, component]
            order = np.argsort(ids, kind="stable")
            sorted_ids = np.asarray(ids)[order]
            want = cache.index.object_ids.astype(np.int64)
            pos = np.searchsorted(sorted_ids, want)
            pos = np.clip(pos, 0, max(sorted_ids.size - 1, 0))
            found = sorted_ids.size > 0
            hit = (sorted_ids[pos] == want) if found else np.zeros(want.shape, bool)
            values = np.full(want.shape, np.nan)
            values[hit] = vals[order][pos[hit]]
            cache.object_attrs[name] = values
        return values

    async def evaluate(self, request: dict) -> dict:
        t0 = time.time()
        source = request["source"]
        groups = request.get("groups", [])
        level_req = request.get("level")
        budget = int(request.get("vertexBudget") or zv.DEFAULT_VERTEX_BUDGET)
        max_vertices = request.get("maxVerticesPerTract") or None
        store, info, level, cache = await self._level(
            source, level_req, budget, max_vertices
        )
        index = cache.index
        n = len(index)
        passing = np.zeros(n, dtype=bool)
        claimed = np.zeros(n, dtype=bool)
        claimed_color = np.empty(n, dtype=object)
        group_counts = []
        active = False
        for gi, group in enumerate(groups):
            where = f"groups[{gi}]"
            rois_json = group.get("rois") or []
            attr_filters = group.get("attrFilters") or []
            selects = bool(rois_json) or bool(attr_filters)
            visible = group.get("visible", True) is not False
            if not selects:
                group_counts.append({"id": group.get("id"), "count": 0})
                continue
            rois = [parse_roi(r, f"{where}.rois[{i}]") for i, r in enumerate(rois_json)]
            vertex_labels = None
            if any(isinstance(r.shape, LabelMask) for r in rois):
                parcellation = group.get("parcellation") or request.get("parcellation")
                if not parcellation:
                    raise JobSpecError(
                        f"{where}: label regions need a linked parcellation layer"
                    )
                vertex_labels = await self._vertex_labels(cache, info, parcellation)
            member = (
                streamlines_pass_rois(index, rois, vertex_labels=vertex_labels)
                if rois
                else np.ones(n, dtype=bool)
            )
            for f in attr_filters:
                if f.get("scope", "object") != "object":
                    raise JobSpecError(
                        f"{where}: only object-scope attribute filters are "
                        f"supported by the ngpy engine"
                    )
                values = await self._object_attr(store, level, cache, f["name"])
                lo = float(f.get("min", -np.inf))
                hi = float(f.get("max", np.inf))
                with np.errstate(invalid="ignore"):
                    member &= (values >= lo) & (values <= hi)
            group_counts.append({"id": group.get("id"), "count": int(member.sum())})
            if not visible:
                continue
            active = True
            fresh = member & ~claimed
            claimed_color[fresh] = group.get("color", "#ffffff")
            claimed |= fresh
            passing |= member
        seg = cache.segment_ids
        segments = [str(int(s)) for s in np.sort(seg[passing])]
        colors = {
            str(int(s)): str(c) for s, c in zip(seg[claimed], claimed_color[claimed])
        }
        return {
            "level": level,
            "levelObjects": n,
            "levelVertices": int(index.positions.shape[0]),
            "storeObjects": max((lv.num_objects for lv in info.levels), default=0),
            "objectSparsity": next(
                (lv.object_sparsity for lv in info.levels if lv.level == level), 1.0
            ),
            "groups": group_counts,
            "active": active,
            "segments": segments,
            "colors": colors,
            "elapsedMs": int((time.time() - t0) * 1000),
        }

    async def passing_ids(self, request: dict) -> list[dict[str, Any]]:
        """Per-group passing ZARR-VECTORS object ids (for export by id)."""
        result = []
        store, info, level, cache = await self._level(
            request["source"],
            request.get("level"),
            int(request.get("vertexBudget") or zv.DEFAULT_VERTEX_BUDGET),
            request.get("maxVerticesPerTract") or None,
        )
        for group in request.get("groups", []):
            single = dict(request)
            g = dict(group)
            g["visible"] = True
            single["groups"] = [g]
            single["level"] = level
            answer = await self.evaluate(single)
            seg_set = set(answer["segments"])
            seg = cache.segment_ids
            mask = np.array([str(int(s)) in seg_set for s in seg], dtype=bool)
            result.append(
                {
                    "name": group.get("name", ""),
                    "objectIds": [str(int(i)) for i in cache.index.object_ids[mask]],
                }
            )
        return result


_engine: FilterEngine | None = None


def engine() -> FilterEngine:
    global _engine
    if _engine is None:
        _engine = FilterEngine()
    return _engine
