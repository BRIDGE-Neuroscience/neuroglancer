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

"""TrackVis-style region-of-interest tests for streamlines.

A streamline is a polyline: a run of vertices with an implicit edge between
consecutive ones. The tests here answer "does this polyline pass through this
region", and compose several regions into one verdict.

This is a numpy port of the reference implementation that used to run in the
viewer's backend worker (``src/datasource/zarr-vectors/roi.ts`` on the
``zarr_vectors_roi_store`` branch). The semantics are identical, but the unit of
work is inverted: rather than one streamline at a time, every test evaluates
*all* streamlines of a `TractIndex` at once, so the whole dissection is a
handful of vectorised passes over flat arrays. Under ngpy this is the ONLY
implementation: the wrapper evaluates here and writes the answer into ordinary
segmentation-layer state.

The geometry is deliberately pure -- it knows nothing about zarr, the viewer, or
the state channel -- so it can be unit-tested directly.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .index import TractIndex

__all__ = [
    "Ellipsoid",
    "Box",
    "Halfspace",
    "LabelMask",
    "RoiShape",
    "RoiPredicate",
    "RoiOperator",
    "Roi",
    "streamlines_pass_roi",
    "streamlines_pass_rois",
    "combine_roi_verdicts",
]


class Ellipsoid(NamedTuple):
    """Axis-aligned ellipsoid.

    A sphere is ``radii = (r, r, r)``; a disc/slab is one radius much smaller
    than the others. This is the only bounded primitive needed: sphere, disc and
    slab are all choices of its three radii, which avoids a family of
    near-duplicate shapes.
    """

    center: np.ndarray
    radii: np.ndarray


class Box(NamedTuple):
    """Axis-aligned bounding box, ``lower <= upper`` per axis."""

    lower: np.ndarray
    upper: np.ndarray


class Halfspace(NamedTuple):
    """Infinite half-space ``{x : dot(normal, x - origin) >= 0}``.

    ``normal`` need not be unit length. Used as a *filter*; a clipping plane that
    truncates rendering is a separate display concern.
    """

    origin: np.ndarray
    normal: np.ndarray


class LabelMask(NamedTuple):
    """A region defined by anatomical labels in a linked parcellation volume.

    A streamline passes it when a vertex falls in a voxel whose label is one of
    ``labels``. Unlike the geometric primitives this has no closed form: it is
    evaluated against ``vertex_labels`` -- the parcellation label under every
    vertex of the index, sampled once by :mod:`ngpy.labels` and passed to
    :func:`streamlines_pass_roi`. Folding one without that array raises.

    Labels are tested at VERTICES, whatever the predicate: ``ANY_SEGMENT`` is
    treated as ``ANY_VERTEX`` (sampling the voxels an edge passes through is not
    worth its cost at parcellation resolution), and the endpoint predicates test
    the labels under the first/last vertex.
    """

    labels: tuple[int, ...]


RoiShape = Ellipsoid | Box | Halfspace | LabelMask


def _label_mask_unfoldable() -> np.ndarray:
    raise NotImplementedError(
        "label-mask ROIs need the parcellation label under every vertex: pass "
        "vertex_labels= (see ngpy.labels.sample_vertex_labels)."
    )


def _labels_inside(shape: LabelMask, labels: np.ndarray) -> np.ndarray:
    """Boolean mask over `labels` of those belonging to `shape`."""
    if not shape.labels:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, np.asarray(shape.labels))


class RoiPredicate(enum.IntEnum):
    """What it means for a streamline to "pass" a region.

    ``ANY_SEGMENT`` is the default and the only one robust to vertex spacing: it
    tests the polyline itself, not its samples. TrackVis's historical test is
    ``ANY_VERTEX``, which is only safe while the step size is small relative to
    the region -- a track can otherwise step clean over a small sphere without
    ever landing a vertex inside it. (DSI Studio calls this the "leap-across
    artifact"; its documented workaround, enlarging the region, is a symptom of
    the wrong test rather than a fix.) ``ANY_VERTEX`` is kept for compatibility
    with results produced by those tools.
    """

    ANY_SEGMENT = 0
    """Any point of any edge lies within the region. Exact for a polyline."""
    ANY_VERTEX = 1
    """Any vertex lies within the region. Sensitive to vertex spacing."""
    EITHER_ENDPOINT = 2
    """The first or last vertex lies within: tracts *terminating* in a region."""
    BOTH_ENDPOINTS = 3
    """Both endpoints lie within."""


class RoiOperator(enum.IntEnum):
    """How a region combines with the verdict accumulated so far."""

    AND = 0
    """Keep streamlines that pass this region too."""
    OR = 1
    """Also keep streamlines that pass this region."""
    ANDNOT = 2
    """Drop streamlines that pass this region (an exclusion region)."""


class Roi(NamedTuple):
    shape: RoiShape
    predicate: RoiPredicate = RoiPredicate.ANY_SEGMENT
    operator: RoiOperator = RoiOperator.AND
    """How this region folds into the verdict accumulated so far.

    The *first* region in a list seeds the verdict instead of folding into one:
    ``AND`` seeds it to "passes this region", ``ANDNOT`` to its negation, which
    is how a dissection made only of exclusions is expressed.  ``OR`` is
    degenerate in that position and is treated as ``AND``.  See
    :func:`_seed_verdict`.
    """


def _vertices_inside(shape: RoiShape, points: np.ndarray) -> np.ndarray:
    """Boolean mask over ``points`` (..., rank) of those lying inside `shape`."""
    if isinstance(shape, Ellipsoid):
        radii = np.asarray(shape.radii, dtype=np.float64)
        # A non-positive radius makes the ellipsoid degenerate; nothing is
        # inside it. Matching the reference implementation, this is a rejection
        # rather than an error, so a collapsed slider stays well-defined.
        if np.any(radii <= 0):
            return np.zeros(points.shape[:-1], dtype=bool)
        d = (points - np.asarray(shape.center, dtype=np.float64)) / radii
        return np.sum(d * d, axis=-1) <= 1.0
    if isinstance(shape, Box):
        lower = np.asarray(shape.lower, dtype=np.float64)
        upper = np.asarray(shape.upper, dtype=np.float64)
        return np.all((points >= lower) & (points <= upper), axis=-1)
    if isinstance(shape, Halfspace):
        origin = np.asarray(shape.origin, dtype=np.float64)
        normal = np.asarray(shape.normal, dtype=np.float64)
        return np.sum(normal * (points - origin), axis=-1) >= 0.0
    if isinstance(shape, LabelMask):
        return _label_mask_unfoldable()
    raise TypeError(f"unsupported ROI shape: {shape!r}")


def _segments_intersect(shape: RoiShape, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean mask over the segments ``a -> b`` (both (n, rank)) that touch `shape`."""
    if isinstance(shape, Ellipsoid):
        radii = np.asarray(shape.radii, dtype=np.float64)
        if np.any(radii <= 0):
            return np.zeros(a.shape[0], dtype=bool)
        # Map both endpoints into the space where the ellipsoid is the unit
        # sphere centred at the origin, then ask for the distance from the
        # origin to the segment. Exact, and no tolerance to tune.
        center = np.asarray(shape.center, dtype=np.float64)
        pa = (a - center) / radii
        pb = (b - center) / radii
        d = pb - pa
        ab_dot_ab = np.sum(d * d, axis=-1)
        a_dot_ab = np.sum(-pa * d, axis=-1)
        # Project the origin onto the segment, clamping to its ends. A
        # degenerate (zero-length) segment clamps to t = 0, i.e. its endpoint.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                ab_dot_ab > 0, a_dot_ab / np.where(ab_dot_ab > 0, ab_dot_ab, 1.0), 0.0
            )
        np.clip(t, 0.0, 1.0, out=t)
        closest = pa + t[:, None] * d
        return np.sum(closest * closest, axis=-1) <= 1.0

    if isinstance(shape, Box):
        # Slab method: intersect the segment's parametric range with each axis's
        # slab; the segment misses iff the ranges ever become disjoint.
        lower = np.asarray(shape.lower, dtype=np.float64)
        upper = np.asarray(shape.upper, dtype=np.float64)
        d = b - a
        n = a.shape[0]
        t_min = np.zeros(n, dtype=np.float64)
        t_max = np.ones(n, dtype=np.float64)
        missed = np.zeros(n, dtype=bool)
        for i in range(a.shape[1]):
            di = d[:, i]
            ai = a[:, i]
            # Parallel to this slab: inside it for all t, or for none.
            parallel = np.abs(di) < 1e-20
            missed |= parallel & ((ai < lower[i]) | (ai > upper[i]))
            safe_di = np.where(parallel, 1.0, di)
            t0 = (lower[i] - ai) / safe_di
            t1 = (upper[i] - ai) / safe_di
            lo = np.where(parallel, t_min, np.minimum(t0, t1))
            hi = np.where(parallel, t_max, np.maximum(t0, t1))
            t_min = np.maximum(t_min, lo)
            t_max = np.minimum(t_max, hi)
        return ~missed & (t_min <= t_max)

    if isinstance(shape, Halfspace):
        # Convex and unbounded: the segment touches it iff either endpoint does.
        return _vertices_inside(shape, a) | _vertices_inside(shape, b)

    if isinstance(shape, LabelMask):
        return _label_mask_unfoldable()

    raise TypeError(f"unsupported ROI shape: {shape!r}")


def _any_per_object(
    index: TractIndex, flags: np.ndarray, owner: np.ndarray
) -> np.ndarray:
    """OR-reduce a per-element boolean mask into a per-object mask.

    `owner[i]` is the object that element `i` belongs to. This is where a
    tract's fragments are recombined: an element hit anywhere -- in any fragment
    -- marks the whole object as crossing. `bincount` is used rather than
    `reduceat` because it is well-defined for objects that contribute no
    elements at all (a single-vertex fragment has no segments).
    """
    if flags.size == 0:
        return np.zeros(len(index), dtype=bool)
    hits = owner[flags]
    return np.bincount(hits, minlength=len(index)).astype(bool)


def streamlines_pass_roi(
    index: TractIndex,
    shape: RoiShape,
    predicate: RoiPredicate = RoiPredicate.ANY_SEGMENT,
    *,
    vertex_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask over `index`'s objects of those passing one region.

    The mask is per distinct object: where a tract is stored as several
    fragments, a crossing in any one of them counts for the whole tract.

    ``vertex_labels`` (one integer per vertex of ``index``) is required for a
    :class:`LabelMask` and ignored otherwise.
    """
    predicate = RoiPredicate(predicate)
    n = len(index)
    positions = index.positions
    # No tracts, or tracts that are all empty: nothing can pass, and the
    # endpoint gather below would index into an empty array.
    if n == 0 or positions.shape[0] == 0:
        return np.zeros(n, dtype=bool)

    counts = index.counts
    nonempty = counts > 0

    if isinstance(shape, LabelMask):
        if vertex_labels is None:
            _label_mask_unfoldable()
        vertex_labels = np.asarray(vertex_labels)
        if vertex_labels.shape[0] != positions.shape[0]:
            raise ValueError(
                f"vertex_labels has {vertex_labels.shape[0]} entries for "
                f"{positions.shape[0]} vertices"
            )
        if predicate in (RoiPredicate.EITHER_ENDPOINT, RoiPredicate.BOTH_ENDPOINTS):
            first = np.where(nonempty, index.offsets[:-1], 0)
            last = np.where(nonempty, index.offsets[1:] - 1, 0)
            in_first = _labels_inside(shape, vertex_labels[first])
            in_last = _labels_inside(shape, vertex_labels[last])
            if predicate is RoiPredicate.EITHER_ENDPOINT:
                per_row = nonempty & (in_first | in_last)
            else:
                per_row = nonempty & in_first & in_last
            return _any_per_object(index, per_row, index.row_object)
        inside = _labels_inside(shape, vertex_labels)
        return _any_per_object(index, inside, index.vertex_object)

    if predicate in (RoiPredicate.EITHER_ENDPOINT, RoiPredicate.BOTH_ENDPOINTS):
        # An empty row has no endpoints and passes nothing.
        #
        # Endpoints here are a *fragment's* ends, not necessarily the tract's:
        # a tract cut across chunks has interior ends that look like endpoints.
        # This matches the implementation being replaced, and recovering true
        # tract ends would need the stored fragment order, which the chunk grid
        # does not carry. `ANY_SEGMENT` -- the default -- is unaffected.
        first = index.offsets[:-1]
        last = index.offsets[1:] - 1
        safe_first = np.where(nonempty, first, 0)
        safe_last = np.where(nonempty, last, 0)
        in_first = _vertices_inside(shape, positions[safe_first])
        in_last = _vertices_inside(shape, positions[safe_last])
        if predicate is RoiPredicate.EITHER_ENDPOINT:
            per_row = nonempty & (in_first | in_last)
        else:
            per_row = nonempty & in_first & in_last
        return _any_per_object(index, per_row, index.row_object)

    if predicate is RoiPredicate.ANY_VERTEX:
        inside = _vertices_inside(shape, positions)
        return _any_per_object(index, inside, index.vertex_object)

    # ANY_SEGMENT.
    a_idx, b_idx = index.segment_endpoints
    result = np.zeros(n, dtype=bool)
    if a_idx.size:
        hit = _segments_intersect(shape, positions[a_idx], positions[b_idx])
        result = _any_per_object(index, hit, index.segment_object)
    # A single-vertex row has no edge; fall back to the vertex.
    singles = np.flatnonzero(counts == 1)
    if singles.size:
        inside = _vertices_inside(shape, positions[index.offsets[singles]])
        result |= _any_per_object(index, inside, index.row_object[singles])
    return result


def _seed_verdict(passes: np.ndarray, operator: RoiOperator) -> np.ndarray:
    """How the first region in a list seeds the fold.

    ``AND`` seeds the verdict to "passes this region".  ``ANDNOT`` seeds it to
    the negation, which is what makes a dissection of pure exclusions
    expressible: "everything except the tracts crossing here" needs no leading
    include-everything region.  ``OR`` is degenerate in this position -- folding
    it into an empty verdict is ``False or x == x`` -- so it behaves as ``AND``
    rather than being a third seeding mode.

    Both folds below route their seed through this, so they cannot disagree
    about what a leading ``ANDNOT`` means.
    """
    return ~passes if RoiOperator(operator) is RoiOperator.ANDNOT else passes


def streamlines_pass_rois(
    index: TractIndex,
    rois: Sequence[Roi],
    *,
    vertex_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask over `index`'s tracts of those surviving a list of regions.

    The list is evaluated as a left fold, in order: the first region seeds the
    verdict (see :func:`_seed_verdict`) and each subsequent one folds in. This
    is how TrackVis and DSI Studio express dissections -- neither offers a
    nested expression tree, and a left fold covers the dissections found in
    practice -- so region order is the whole of the syntax, and reordering the
    list is the only editing operation a user needs.

    An empty list passes everything: no regions means no filtering, rather than
    filtering everything away.
    """
    if not rois:
        return np.ones(len(index), dtype=bool)
    verdict = _seed_verdict(
        streamlines_pass_roi(
            index, rois[0].shape, rois[0].predicate, vertex_labels=vertex_labels
        ),
        rois[0].operator,
    )
    for roi in rois[1:]:
        op = RoiOperator(roi.operator)
        # The reference implementation short-circuits per streamline; batching
        # moves that decision up to the whole set, which is where it still pays.
        # AND/ANDNOT can only ever clear a verdict and OR can only ever set one,
        # so a region whose result cannot move any tract is skipped outright.
        if op is RoiOperator.OR:
            if verdict.all():
                continue
        elif not verdict.any():
            continue
        crossed = streamlines_pass_roi(
            index, roi.shape, roi.predicate, vertex_labels=vertex_labels
        )
        if op is RoiOperator.AND:
            verdict &= crossed
        elif op is RoiOperator.OR:
            verdict |= crossed
        else:
            verdict &= ~crossed
    return verdict


def combine_roi_verdicts(
    crossed: Sequence[np.ndarray], rois: Sequence[Roi]
) -> np.ndarray:
    """The same left-fold as :func:`streamlines_pass_rois`, but over pre-computed
    per-region crossing masks rather than the geometry.

    ``crossed[i]`` is the mask of tracts that cross ``rois[i]`` under its
    predicate. Keeping the crossing test separate from the fold is what lets a
    caller compute the two on different schedules -- for instance re-testing only
    the one region a user just dragged and reusing the rest.
    """
    if not rois:
        raise ValueError("combine_roi_verdicts: no ROIs to fold")
    if len(crossed) != len(rois):
        raise ValueError(
            f"combine_roi_verdicts: crossed has {len(crossed)} entries, "
            f"expected {len(rois)}"
        )
    # Copy first: the fold mutates `verdict` in place below, so it must not
    # alias the caller's mask.
    verdict = _seed_verdict(
        np.array(crossed[0], dtype=bool, copy=True), rois[0].operator
    )
    for mask, roi in zip(crossed[1:], rois[1:]):
        op = RoiOperator(roi.operator)
        if op is RoiOperator.AND:
            verdict &= mask
        elif op is RoiOperator.OR:
            verdict |= mask
        else:
            verdict &= ~mask
    return verdict
