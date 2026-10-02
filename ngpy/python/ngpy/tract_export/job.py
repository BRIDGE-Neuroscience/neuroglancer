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
"""The export job spec: what the wrapper's Export tab asks the exporter for.

Ported from ``neuroglancer.tract_export.job`` (``zarr_vectors_roi_store``).
Groups arrive in the group **persistence** JSON form (``groupToJson``: shapes
keyed by ``type`` with *string* predicate/operator names) -- the same JSON the
wrapper keeps in its own state and saves to the ROI store, so there is a single
serialisation to keep correct.  :func:`parse_roi` is shared with the Filter
engine (:mod:`ngpy.filter`).

Under ngpy the exporter returns BYTES to the page, which downloads them or
uploads them to GCS; ``destination`` is therefore optional and informational
(the file name), and every kind the page offers is accepted.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from ngpy.tractography.roi import (
    Box,
    Ellipsoid,
    Halfspace,
    LabelMask,
    Roi,
    RoiOperator,
    RoiPredicate,
    RoiShape,
)

#: Bumped when a change would make an older exporter misread a newer spec.
#: v2 added ``scope`` -- a v1 exporter would ignore it and quietly export the
#: dissection when the whole store was asked for, so v2 specs must be rejected
#: by an older exporter rather than silently mis-served.
#: v3 added per-group ``objectIds``: the viewer's on-screen dissection selects
#: the streamlines and hands the exporter their ids directly (WYSIWYG), so the
#: exporter reads exactly those objects instead of re-reading and re-folding the
#: whole level. A v2 exporter would ignore ``objectIds`` and re-fold ``rois``,
#: producing a different (re-evaluated, not on-screen) set -- a misread -- so v3
#: specs must be rejected by an older exporter.
JOB_SCHEMA_VERSION = 3

FORMATS = ("trk", "zvf")
DESTINATION_KINDS = ("local", "download", "gcs")
#: ``selected`` folds the dissection; ``whole`` exports every object at the
#: level with no ROI filter (a straight duplication of the store's geometry).
SCOPES = ("selected", "whole")

_PREDICATE_FROM_JSON = {
    "any_segment": RoiPredicate.ANY_SEGMENT,
    "any_vertex": RoiPredicate.ANY_VERTEX,
    "either_endpoint": RoiPredicate.EITHER_ENDPOINT,
    "both_endpoints": RoiPredicate.BOTH_ENDPOINTS,
}
_OPERATOR_FROM_JSON = {
    "and": RoiOperator.AND,
    "or": RoiOperator.OR,
    "andnot": RoiOperator.ANDNOT,
}


class JobSpecError(ValueError):
    """A malformed job spec. The message is meant to be shown to the user."""


@dataclasses.dataclass(frozen=True)
class ExportGroup:
    """One dissection to export, named so the output can be labelled."""

    name: str
    rois: tuple[Roi, ...]
    #: Explicit object ids for this group, as computed by the viewer's on-screen
    #: dissection (WYSIWYG export). When present, the exporter reads exactly these
    #: objects and does NOT re-fold ``rois`` over the level -- that is the whole
    #: point of v3. ``None`` means "fold ``rois``" (a hand-written spec, or the
    #: retained legacy path); an empty tuple means "the on-screen fold selected
    #: nothing", which is a valid empty selection, not "select everything".
    object_ids: tuple[int, ...] | None = None


@dataclasses.dataclass(frozen=True)
class Destination:
    kind: str
    path: str


@dataclasses.dataclass(frozen=True)
class ExportJob:
    source_url: str
    level: int
    groups: tuple[ExportGroup, ...]
    format: str
    destination: Destination
    #: 4x4 voxel-to-RAS matrix for TRK, or None for identity.
    affine: np.ndarray | None = None
    #: ``"selected"`` folds ``groups``; ``"whole"`` exports every object at the
    #: level, ignoring ``groups`` (which may then be empty).
    scope: str = "selected"


def _require(obj: Any, key: str, where: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise JobSpecError(f"{where}: missing required property {key!r}")
    return obj[key]


def _floats(value: Any, where: str) -> np.ndarray:
    if not isinstance(value, list | tuple):
        raise JobSpecError(f"{where}: expected an array of numbers")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise JobSpecError(f"{where}: expected an array of numbers ({e})") from e
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise JobSpecError(f"{where}: expected a flat array of finite numbers")
    return arr


def _parse_shape(obj: Any, where: str) -> RoiShape:
    kind = _require(obj, "type", where)
    if kind == "ellipsoid":
        return Ellipsoid(
            _floats(_require(obj, "center", where), f"{where}.center"),
            _floats(_require(obj, "radii", where), f"{where}.radii"),
        )
    if kind == "box":
        return Box(
            _floats(_require(obj, "lower", where), f"{where}.lower"),
            _floats(_require(obj, "upper", where), f"{where}.upper"),
        )
    if kind == "halfspace":
        return Halfspace(
            _floats(_require(obj, "origin", where), f"{where}.origin"),
            _floats(_require(obj, "normal", where), f"{where}.normal"),
        )
    if kind == "labelMask":
        labels_raw = _require(obj, "labels", where)
        if not isinstance(labels_raw, list | tuple) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in labels_raw
        ):
            raise JobSpecError(f"{where}.labels: expected an array of integer ids")
        # Folded by ngpy.filter against a sampled parcellation; an export spec
        # carries the passing objectIds, so here it is provenance.
        return LabelMask(labels=tuple(labels_raw))
    raise JobSpecError(f"{where}: unknown ROI shape type {kind!r}")


def parse_roi(obj: Any, where: str = "roi") -> Roi:
    shape = _parse_shape(_require(obj, "shape", where), f"{where}.shape")
    predicate_name = _require(obj, "predicate", where)
    operator_name = _require(obj, "operator", where)
    if predicate_name not in _PREDICATE_FROM_JSON:
        raise JobSpecError(f"{where}: unknown ROI predicate {predicate_name!r}")
    if operator_name not in _OPERATOR_FROM_JSON:
        raise JobSpecError(f"{where}: unknown ROI operator {operator_name!r}")
    # `name` is display-only and deliberately dropped: it never affects which
    # streamlines are selected, so carrying it into the fold would be noise.
    return Roi(
        shape,
        _PREDICATE_FROM_JSON[predicate_name],
        _OPERATOR_FROM_JSON[operator_name],
    )


def _parse_object_ids(value: Any, where: str) -> tuple[int, ...] | None:
    """Parse a group's ``objectIds`` array, or ``None`` if the key is absent.

    Ids arrive as **decimal strings**, not JSON numbers: an object id is a uint64
    and can exceed JavaScript's safe-integer range, where ``JSON.stringify`` would
    silently round it. Plain ints are also accepted for a hand-written spec.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise JobSpecError(
            f"{where}.objectIds: expected an array of non-negative integer ids "
            f"(as decimal strings)"
        )
    out: list[int] = []
    for i, v in enumerate(value):
        if isinstance(v, bool):
            raise JobSpecError(f"{where}.objectIds[{i}]: expected an integer id")
        if isinstance(v, int):
            iv = v
        elif isinstance(v, str):
            try:
                iv = int(v)
            except ValueError:
                raise JobSpecError(
                    f"{where}.objectIds[{i}]: expected a decimal integer string"
                ) from None
        else:
            raise JobSpecError(
                f"{where}.objectIds[{i}]: expected a non-negative integer id "
                f"(or its decimal string)"
            )
        if iv < 0:
            raise JobSpecError(f"{where}.objectIds[{i}]: expected a non-negative id")
        out.append(iv)
    return tuple(out)


def _parse_group(obj: Any, where: str) -> ExportGroup:
    name = _require(obj, "name", where)
    if not isinstance(name, str):
        raise JobSpecError(f"{where}.name: expected a string")
    object_ids = _parse_object_ids(obj.get("objectIds"), where)
    rois_raw = obj.get("rois")
    # ROIs are required only when there is no explicit id selection. A v3 spec
    # carries the viewer's on-screen selection in ``objectIds`` and never folds
    # ``rois`` -- the ids may even be empty (nothing passed on screen), which is a
    # valid empty selection. The "empty rois selects everything" guard therefore
    # applies only to the fold path (``objectIds`` absent).
    if object_ids is None:
        if not isinstance(rois_raw, list):
            raise JobSpecError(f"{where}.rois: expected an array")
        if not rois_raw:
            raise JobSpecError(
                f"{where}: group {name!r} has no ROIs, which would select every "
                f"streamline in the dataset"
            )
    if rois_raw is None:
        rois_raw = []
    elif not isinstance(rois_raw, list):
        raise JobSpecError(f"{where}.rois: expected an array")
    return ExportGroup(
        name=name,
        rois=tuple(parse_roi(r, f"{where}.rois[{i}]") for i, r in enumerate(rois_raw)),
        object_ids=object_ids,
    )


def _parse_affine(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise JobSpecError(f"affine: expected a 4x4 matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise JobSpecError("affine: expected finite numbers")
    return arr


def parse_job(obj: Any) -> ExportJob:
    """Validate a job spec, raising :class:`JobSpecError` with a usable message."""
    if not isinstance(obj, dict):
        raise JobSpecError("Job spec must be a JSON object")

    version = obj.get("schemaVersion", JOB_SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise JobSpecError("schemaVersion: expected an integer")
    # Reject newer rather than silently ignoring fields this build cannot honour
    # -- an exporter that quietly drops a future `subsample` key would produce a
    # plausible file that is not what was asked for.
    if version > JOB_SCHEMA_VERSION:
        raise JobSpecError(
            f"Job schema version {version} is newer than this exporter "
            f"supports ({JOB_SCHEMA_VERSION})"
        )

    source = _require(obj, "source", "spec")
    source_url = _require(source, "url", "source")
    if not isinstance(source_url, str) or not source_url:
        raise JobSpecError("source.url: expected a non-empty string")
    level = source.get("level", 0)
    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise JobSpecError("source.level: expected a non-negative integer")

    fmt = _require(obj, "format", "spec")
    if fmt not in FORMATS:
        raise JobSpecError(f"format: expected one of {FORMATS}, got {fmt!r}")

    # Absent scope means "selected": a v1 spec had no scope and always folded
    # the dissection, so that is the only back-compatible default.
    scope = obj.get("scope", "selected")
    if scope not in SCOPES:
        raise JobSpecError(f"scope: expected one of {SCOPES}, got {scope!r}")

    destination = obj.get("destination") or {"kind": "download", "path": "export"}
    kind = _require(destination, "kind", "destination")
    if kind not in DESTINATION_KINDS:
        raise JobSpecError(
            f"destination.kind: expected one of {DESTINATION_KINDS}, got {kind!r}"
        )
    path = destination.get("path", "export")
    if not isinstance(path, str) or not path:
        raise JobSpecError("destination.path: expected a non-empty string")

    # A whole-store export folds no ROIs, so groups are optional and may be
    # empty; a selected export needs at least one group to have anything to do.
    if scope == "whole":
        groups = obj.get("groups", [])
        if not isinstance(groups, list):
            raise JobSpecError("groups: expected an array")
    else:
        groups = _require(obj, "groups", "spec")
        if not isinstance(groups, list) or not groups:
            raise JobSpecError("groups: expected a non-empty array")

    affine = obj.get("affine")
    return ExportJob(
        source_url=source_url,
        level=level,
        groups=tuple(_parse_group(g, f"groups[{i}]") for i, g in enumerate(groups)),
        format=fmt,
        destination=Destination(kind=kind, path=path),
        affine=None if affine is None else _parse_affine(affine),
        scope=scope,
    )


def roi_bounds(rois: Sequence[Roi]) -> tuple[np.ndarray, np.ndarray] | None:
    """The axis-aligned box enclosing `rois`, or None if it is unbounded.

    Used only to narrow which chunks are read before evaluating the fold; it is
    never the selection itself. A halfspace is unbounded, and an exclusion
    region's *complement* is what passes, so in both cases there is no safe
    finite box and the caller must read everything. Returning None rather than a
    best-effort box keeps that decision explicit at the call site.
    """
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    for roi in rois:
        if roi.operator == RoiOperator.ANDNOT:
            return None
        shape = roi.shape
        if isinstance(shape, Ellipsoid):
            lowers.append(np.asarray(shape.center) - np.abs(shape.radii))
            uppers.append(np.asarray(shape.center) + np.abs(shape.radii))
        elif isinstance(shape, Box):
            lowers.append(np.minimum(shape.lower, shape.upper))
            uppers.append(np.maximum(shape.lower, shape.upper))
        else:
            return None
    if not lowers:
        return None
    # An OR widens the selection, so the union is the only sound envelope; it is
    # also correct (merely looser) for a pure AND chain.
    return (
        np.min(np.stack(lowers), axis=0),
        np.max(np.stack(uppers), axis=0),
    )
