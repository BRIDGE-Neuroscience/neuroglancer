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
"""Selection helpers shared by the in-browser exporter.

Ported from ``neuroglancer.tract_export.run`` (``zarr_vectors_roi_store``).
The native, synchronous exporter that lived here (``run_job``, writing through
``zarr-vectors-tools`` on disk) is NOT part of ngpy -- everything ngpy exports
is produced in the page by :mod:`ngpy.tract_export.browser`.  What remains is
the pure part both paths agreed on: turning polylines into an index, folding a
group over it, and combining per-group id lists.

Endpoints: these helpers concatenate a tract's fragments before testing, so
``either_endpoint`` / ``both_endpoints`` see the tract's real ends.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ngpy.tract_export.job import ExportGroup
from ngpy.tractography.index import TractIndex
from ngpy.tractography.roi import streamlines_pass_rois
from ngpy.zv import store_path_from_url  # noqa: F401  (re-exported)


class ExportRunError(RuntimeError):
    """A job that parsed but could not be carried out."""


def build_index(polylines: list, object_ids: list) -> TractIndex:
    """One index row per polyline, its fragments concatenated."""
    if len(polylines) != len(object_ids):
        raise ExportRunError(
            f"read_polylines returned {len(polylines)} polylines but "
            f"{len(object_ids)} object ids; they must correspond"
        )
    if not polylines:
        return TractIndex(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(1, dtype=np.intp),
            np.zeros(0, dtype=np.uint64),
        )
    rows = [np.concatenate(fragments, axis=0) for fragments in polylines]
    positions = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    offsets = np.zeros(len(rows) + 1, dtype=np.intp)
    offsets[1:] = np.cumsum(np.fromiter((len(r) for r in rows), dtype=np.intp))
    return TractIndex(positions, offsets, np.asarray(object_ids, dtype=np.uint64))


#: Objects folded per batch (TractIndex costs ~44 bytes per vertex).
OBJECT_BATCH = 25_000


def passing_object_ids(index: TractIndex, group: ExportGroup) -> np.ndarray:
    """The object ids of `index` that survive `group`'s fold, ascending."""
    if len(index) == 0:
        return np.zeros(0, dtype=np.uint64)
    return index.object_ids[streamlines_pass_rois(index, list(group.rois))]


def select_from(
    polylines: list,
    object_ids: list,
    groups: Sequence[ExportGroup],
    batch: int | None = None,
    *,
    release: bool = False,
) -> tuple[int, list[tuple[str, np.ndarray]]]:
    """Fold `groups` over geometry that has already been read.

    Returns the number of objects considered and each group's passing ids, as
    a LIST keyed by position (group names are not unique).
    """
    batch = OBJECT_BATCH if batch is None else batch
    per_group: list[list[np.ndarray]] = [[] for _ in groups]
    considered = 0
    for start in range(0, len(polylines), batch):
        stop = min(start + batch, len(polylines))
        index = build_index(polylines[start:stop], object_ids[start:stop])
        considered += len(index)
        for i, group in enumerate(groups):
            hit = passing_object_ids(index, group)
            if hit.size:
                per_group[i].append(hit)
        if release:
            for k in range(start, stop):
                polylines[k] = None
        del index
    return considered, [
        (g.name, np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64))
        for g, parts in zip(groups, per_group)
    ]


def union_ids(per_group: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """The ascending union of every group's passing ids."""
    arrays = [ids for _, ids in per_group if ids.size]
    if not arrays:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(arrays))


def per_group_from_object_ids(
    groups: Sequence[ExportGroup],
) -> list[tuple[str, np.ndarray]]:
    """Each group's explicit id selection as ascending unique uint64 ids."""
    return [
        (g.name, np.unique(np.asarray(g.object_ids or (), dtype=np.uint64)))
        for g in groups
    ]


def uses_explicit_ids(groups: Sequence[ExportGroup]) -> bool:
    """Whether this is an id-based selection, rejecting a mixed spec."""
    flags = [g.object_ids is not None for g in groups]
    if any(flags) and not all(flags):
        raise ExportRunError(
            "Some groups carry explicit objectIds and others do not. Provide "
            "objectIds for all groups or none."
        )
    return bool(flags) and all(flags)
