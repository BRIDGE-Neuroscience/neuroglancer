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

"""A flat, ragged index of streamlines, suitable for vectorised ROI tests.

This is the only representation :mod:`ngpy.tractography.roi` needs, and
it is deliberately much smaller than the geometry the viewer renders: ROI
dissection does not need render-resolution polylines, so a tract can be
decimated to a few dozen vertices without changing any verdict that matters.
Keeping the index small is what makes the whole filter affordable inside the
Pyodide worker, which shares a single thread with the rest of the Python side.

The layout is the standard ragged triple, one *row* per stored run of vertices:

    positions   (M, rank) float32  -- every vertex of every row, concatenated
    offsets     (R + 1,)  intp     -- row i is positions[offsets[i]:offsets[i+1]]
    object_ids  (R,)      uint64   -- the segment id each row belongs to

Every index-like array here is `np.intp`, never a hardcoded `np.int64`. Under
Pyodide the platform is wasm32, where `intp` is 32-bit, and numpy refuses to
cast an int64 `repeats` or index array down to it ("Cannot cast array data from
dtype('int64') to dtype('int32') according to the rule 'safe'"). Since `intp` is
by definition wide enough to index any array the platform can allocate, it is
both the portable and the correct choice.

`object_ids` must be the *same* ids the datasource assigns, since they are what
gets pushed back to the viewer as the visible-segment set.

A row is not necessarily a whole tract. In the stored format a streamline is cut
at chunk boundaries into *fragments*, so one tract is generally several rows
sharing an object id, and `object_ids` repeats. That distinction is load-bearing
rather than cosmetic: a tract may cross region A in one fragment and region B in
another, so per-region crossings have to be OR-ed across a tract's rows *before*
the include/or/exclude fold is applied. Evaluating the fold against a single
fragment would wrongly drop it.

`TractIndex` therefore keeps two granularities. The geometry arrays are per row;
`object_ids`, `len()`, and every mask returned by
:mod:`ngpy.tractography.roi` are per distinct object. `row_object` maps
between them.

Fragments are deliberately *not* concatenated into one polyline per tract. The
stored order of a tract's fragments is not recoverable from the chunk grid
alone, so joining them would invent segments between unrelated ends.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TractIndex"]


class TractIndex:
    """A ragged batch of streamlines, with the derived arrays the ROI tests need.

    The derived arrays (`vertex_tract`, `segment_endpoints`, `segment_tract`) are
    computed once on construction rather than lazily: every one of them is needed
    by the first dissection, and building them eagerly keeps the per-ROI path
    free of branches.
    """

    __slots__ = (
        "positions",
        "offsets",
        "object_ids",
        "row_object",
        "row_ids",
        "counts",
        "vertex_object",
        "segment_object",
        "_segment_a",
        "_segment_b",
    )

    def __init__(
        self,
        positions: np.ndarray,
        offsets: np.ndarray,
        object_ids: np.ndarray,
    ):
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.ndim != 2:
            raise ValueError(f"positions must be (M, rank), got {positions.shape}")
        offsets = np.ascontiguousarray(offsets, dtype=np.intp)
        row_ids = np.ascontiguousarray(object_ids, dtype=np.uint64)
        if offsets.ndim != 1 or offsets.size != row_ids.size + 1:
            raise ValueError(
                f"offsets must have len(object_ids) + 1 = {row_ids.size + 1} "
                f"entries, got {offsets.size}"
            )
        if offsets.size and (offsets[0] != 0 or offsets[-1] != positions.shape[0]):
            raise ValueError(
                f"offsets must span positions exactly: got [{offsets[0]}, "
                f"{offsets[-1]}] for {positions.shape[0]} vertices"
            )
        if np.any(np.diff(offsets) < 0):
            raise ValueError("offsets must be non-decreasing")

        self.positions = positions
        self.offsets = offsets
        self.row_ids = row_ids
        # Rows sharing an object id are fragments of one tract. `unique` both
        # collapses them and gives the row -> object mapping in one pass; the
        # resulting id order is ascending, and every per-object array returned
        # anywhere in this package follows it.
        self.object_ids, self.row_object = np.unique(row_ids, return_inverse=True)
        self.row_object = self.row_object.astype(np.intp, copy=False)

        rows = row_ids.size
        counts = np.diff(offsets) if rows else np.zeros(0, dtype=np.intp)
        self.counts = counts
        vertex_row = np.repeat(np.arange(rows, dtype=np.intp), counts)

        # Segment starts are every vertex except the last of each row. Empty
        # rows are excluded explicitly: `offsets[i + 1] - 1` would otherwise
        # point at the previous row's last vertex and wrongly suppress it.
        m = positions.shape[0]
        is_last = np.zeros(m, dtype=bool)
        if rows:
            nonempty_ends = offsets[1:][counts > 0] - 1
            is_last[nonempty_ends] = True
        segment_a = np.flatnonzero(~is_last)
        self._segment_a = segment_a
        self._segment_b = segment_a + 1
        # Reduce straight from vertices/segments to objects: the intermediate
        # per-row step has no consumer, and skipping it halves the bincounts.
        self.vertex_object = self.row_object[vertex_row]
        self.segment_object = (
            self.vertex_object[segment_a] if m else np.zeros(0, dtype=np.intp)
        )

    def __len__(self) -> int:
        """The number of distinct objects -- not rows. See `n_rows`."""
        return int(self.object_ids.size)

    @property
    def n_rows(self) -> int:
        """The number of stored runs, one per fragment."""
        return int(self.row_ids.size)

    @property
    def rank(self) -> int:
        return int(self.positions.shape[1])

    @property
    def segment_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Vertex-index arrays `(a, b)` for every intra-tract segment `a -> b`."""
        return self._segment_a, self._segment_b

    @property
    def nbytes(self) -> int:
        return int(
            self.positions.nbytes
            + self.offsets.nbytes
            + self.row_ids.nbytes
            + self.object_ids.nbytes
            + self.row_object.nbytes
            + self.vertex_object.nbytes
            + self._segment_a.nbytes
            + self._segment_b.nbytes
            + self.segment_object.nbytes
        )

    def decimate(self, max_vertices: int) -> TractIndex:
        """Return a copy with each tract resampled to at most `max_vertices`.

        Vertices are picked by evenly spaced index, always keeping both
        endpoints, so a tract's extent and its termination points -- the two
        things the ROI predicates actually look at -- are preserved. Tracts
        already short enough are copied through untouched.

        This trades a little exactness in `ANY_SEGMENT` for a large constant
        factor: the chord between two kept vertices cuts any corner the dropped
        vertices took, so a region grazed only by that corner can be missed.
        With a few dozen vertices per tract against a region of appreciable size
        that difference does not arise in practice, but it is a real
        approximation and not merely a subsample.
        """
        if max_vertices < 2:
            raise ValueError("max_vertices must be at least 2")
        counts = self.counts
        rows = self.n_rows
        if not rows or not np.any(counts > max_vertices):
            return self

        new_counts = np.minimum(counts, max_vertices)
        new_offsets = np.zeros(rows + 1, dtype=np.intp)
        np.cumsum(new_counts, out=new_offsets[1:])

        # Build, for each output vertex, the index of the input vertex it takes.
        # `linspace` per row would loop; instead compute the within-row
        # fractional position of every output vertex in one pass.
        out_total = int(new_offsets[-1])
        out_row = np.repeat(np.arange(rows, dtype=np.intp), new_counts)
        pos_in_row = np.arange(out_total, dtype=np.intp) - new_offsets[out_row]
        span = np.maximum(new_counts - 1, 1)[out_row]
        source_span = np.maximum(counts - 1, 0)[out_row]
        picked = np.rint(pos_in_row * (source_span / span)).astype(np.intp)
        source = self.offsets[out_row] + picked

        return TractIndex(self.positions[source], new_offsets, self.row_ids)

    def __repr__(self) -> str:
        return (
            f"TractIndex({len(self)} tracts, {self.n_rows} fragments, "
            f"{self.positions.shape[0]} vertices, {self.nbytes / 1e6:.1f} MB)"
        )
