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

"""Tests for the streamline ROI dissection geometry.

Ported from ``python/tests/tractography_roi_test.py`` on the
``zarr_vectors_roi_store`` branch, which mirrored the original TypeScript
``roi.spec.ts`` case for case; the label-mask cases at the end are new (ngpy
folds label masks against sampled vertex labels).

The batched API takes a whole `TractIndex`, so most tests build a one-tract
index; `test_batching_matches_one_at_a_time` covers the batching itself.
"""

import numpy as np
import pytest
from ngpy.tractography import (
    Box,
    Ellipsoid,
    Halfspace,
    LabelMask,
    Roi,
    RoiOperator,
    RoiPredicate,
    TractIndex,
    streamlines_pass_roi,
    streamlines_pass_rois,
)


def line(*xyz: float) -> TractIndex:
    """A single rank-3 tract from a flat list of xyz triples."""
    positions = np.array(xyz, dtype=np.float32).reshape(-1, 3)
    return TractIndex(
        positions, np.array([0, len(positions)]), np.array([1], dtype=np.uint64)
    )


def lines(*tracts) -> TractIndex:
    """Several tracts, each a flat list of xyz triples, in one index."""
    arrays = [np.array(t, dtype=np.float32).reshape(-1, 3) for t in tracts]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in arrays], out=offsets[1:])
    positions = np.concatenate(arrays) if arrays else np.zeros((0, 3), dtype=np.float32)
    ids = np.arange(1, len(arrays) + 1, dtype=np.uint64)
    return TractIndex(positions, offsets, ids)


def sphere(cx: float, cy: float, cz: float, r: float) -> Ellipsoid:
    return Ellipsoid(np.array([cx, cy, cz]), np.array([r, r, r]))


def passes(index, shape, predicate=RoiPredicate.ANY_SEGMENT) -> bool:
    result = streamlines_pass_roi(index, shape, predicate)
    assert result.shape == (len(index),)
    return bool(result[0])


class TestEllipsoid:
    def test_vertex_inside_sphere(self):
        assert passes(line(0, 0, 0, 10, 0, 0, 20, 0, 0), sphere(10, 0, 0, 2))

    def test_track_staying_outside(self):
        assert not passes(line(0, 0, 0, 10, 0, 0), sphere(0, 50, 0, 2))

    def test_leap_across(self):
        # The motivating case for ANY_SEGMENT: the track passes straight through
        # the sphere, but every vertex is well outside it. TrackVis's ANY_VERTEX
        # test misses this; testing the polyline does not.
        track = line(-10, 0, 0, 10, 0, 0)
        roi = sphere(0, 0, 0, 1)
        assert passes(track, roi, RoiPredicate.ANY_SEGMENT)
        assert not passes(track, roi, RoiPredicate.ANY_VERTEX)

    def test_anisotropic_radii(self):
        shape = Ellipsoid(np.array([0, 0, 0]), np.array([10, 1, 1]))
        # Inside the long axis, outside the short one.
        assert not passes(line(5, 5, 0, 6, 5, 0), shape)
        assert passes(line(5, 0.5, 0, 6, 0.5, 0), shape)

    def test_surface_is_inside(self):
        assert passes(
            line(2, 0, 0, 2, 0, 0), sphere(0, 0, 0, 2), RoiPredicate.ANY_VERTEX
        )

    def test_degenerate_ellipsoid_rejected(self):
        shape = Ellipsoid(np.array([0, 0, 0]), np.array([1, 1, 0]))
        assert not passes(line(0, 0, 0, 1, 0, 0), shape)


class TestBox:
    box = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))

    def test_crossing_between_vertices(self):
        assert passes(line(-10, 0, 0, 10, 0, 0), self.box)

    def test_passing_beside(self):
        assert not passes(line(-10, 5, 0, 10, 5, 0), self.box)

    def test_segment_parallel_to_slab(self):
        # Constant z, inside the z-slab: must not be discarded by the parallel case.
        assert passes(line(-10, 0, 0.5, 10, 0, 0.5), self.box)
        # Constant z, outside the z-slab.
        assert not passes(line(-10, 0, 9, 10, 0, 9), self.box)

    def test_stops_short(self):
        assert not passes(line(-10, 0, 0, -5, 0, 0), self.box)

    def test_wholly_inside(self):
        assert passes(line(-0.5, 0, 0, 0.5, 0, 0), self.box)


class TestHalfspace:
    # Keep x >= 0.
    half = Halfspace(np.array([0, 0, 0]), np.array([1, 0, 0]))

    def test_crosses_plane(self):
        assert passes(line(-5, 0, 0, 5, 0, 0), self.half)

    def test_wholly_positive(self):
        assert passes(line(1, 0, 0, 5, 0, 0), self.half)

    def test_wholly_negative(self):
        assert not passes(line(-5, 0, 0, -1, 0, 0), self.half)

    def test_normal_need_not_be_unit(self):
        scaled = Halfspace(np.array([0, 0, 0]), np.array([7, 0, 0]))
        assert passes(line(1, 0, 0, 5, 0, 0), scaled)
        assert not passes(line(-5, 0, 0, -1, 0, 0), scaled)


class TestPredicates:
    # Enters the sphere at the origin only in the middle of its run.
    roi = sphere(0, 0, 0, 1)

    def test_either_endpoint(self):
        track = line(-10, 0, 0, 0, 0, 0, 10, 0, 0)
        assert not passes(track, self.roi, RoiPredicate.EITHER_ENDPOINT)
        ending = line(-10, 0, 0, -5, 0, 0, 0, 0, 0)
        assert passes(ending, self.roi, RoiPredicate.EITHER_ENDPOINT)

    def test_both_endpoints(self):
        one_end = line(0, 0, 0, 10, 0, 0)
        assert not passes(one_end, self.roi, RoiPredicate.BOTH_ENDPOINTS)
        both_ends = line(0.5, 0, 0, 5, 0, 0, -0.5, 0, 0)
        assert passes(both_ends, self.roi, RoiPredicate.BOTH_ENDPOINTS)

    def test_single_vertex_falls_back_to_vertex(self):
        assert passes(line(0, 0, 0), self.roi, RoiPredicate.ANY_SEGMENT)
        assert not passes(line(9, 9, 9), self.roi, RoiPredicate.ANY_SEGMENT)

    def test_empty_streamline_rejected(self):
        empty = TractIndex(
            np.zeros((0, 3), dtype=np.float32),
            np.array([0, 0]),
            np.array([1], dtype=np.uint64),
        )
        assert not passes(empty, self.roi)
        for predicate in RoiPredicate:
            assert not passes(empty, self.roi, predicate)


class TestAddressing:
    def test_reads_only_its_own_run(self):
        # Two tracks packed back to back; the second is inside the sphere.
        index = lines(
            [50, 50, 50, 51, 50, 50],  # track 0, far away
            [0, 0, 0, 1, 0, 0],  # track 1, at the origin
        )
        result = streamlines_pass_roi(index, sphere(0, 0, 0, 2))
        assert list(result) == [False, True]

    def test_adjacent_tracts_do_not_form_a_segment(self):
        # The gap between the last vertex of one tract and the first of the next
        # is not an edge; a region sitting only in that gap must catch neither.
        index = lines([-10, 0, 0, -9, 0, 0], [9, 0, 0, 10, 0, 0])
        assert list(streamlines_pass_roi(index, sphere(0, 0, 0, 1))) == [
            False,
            False,
        ]

    def test_single_vertex_tract_among_others(self):
        index = lines([0, 0, 0], [50, 0, 0, 51, 0, 0], [0.5, 0, 0])
        assert list(streamlines_pass_roi(index, sphere(0, 0, 0, 1))) == [
            True,
            False,
            True,
        ]


class TestComposition:
    at_origin = sphere(0, 0, 0, 1)
    at_ten = sphere(10, 0, 0, 1)

    @staticmethod
    def roi(shape, operator):
        return Roi(shape, RoiPredicate.ANY_SEGMENT, operator)

    def test_no_regions_passes_everything(self):
        assert list(streamlines_pass_rois(line(99, 99, 99), [])) == [True]

    def test_first_operator_seeds_verdict_or_treated_as_include(self):
        # AND and OR both seed to "passes this region": OR is degenerate at
        # index 0 (``False or x == x``), so it is not a third seeding mode.
        through = line(-5, 0, 0, 5, 0, 0)
        for op in (RoiOperator.AND, RoiOperator.OR):
            assert list(
                streamlines_pass_rois(through, [self.roi(self.at_origin, op)])
            ) == [True]

    def test_leading_andnot_seeds_a_pure_exclusion(self):
        # "Everything except tracts crossing the origin" -- expressible without
        # a leading include-everything region.
        rois = [self.roi(self.at_origin, RoiOperator.ANDNOT)]
        assert not streamlines_pass_rois(line(-5, 0, 0, 5, 0, 0), rois)[0]
        assert streamlines_pass_rois(line(9, 0, 0, 11, 0, 0), rois)[0]

    def test_folds_into_a_leading_exclusion(self):
        # NOT through the origin, AND through x=10.
        rois = [
            self.roi(self.at_origin, RoiOperator.ANDNOT),
            self.roi(self.at_ten, RoiOperator.AND),
        ]
        assert streamlines_pass_rois(line(9, 0, 0, 11, 0, 0), rois)[0]
        assert not streamlines_pass_rois(line(-5, 0, 0, 15, 0, 0), rois)[0]
        assert not streamlines_pass_rois(line(0, 50, 0, 1, 50, 0), rois)[0]

    def test_and(self):
        rois = [
            self.roi(self.at_origin, RoiOperator.AND),
            self.roi(self.at_ten, RoiOperator.AND),
        ]
        both = line(-5, 0, 0, 15, 0, 0)  # crosses origin and x=10
        only_origin = line(-5, 0, 0, 1, 0, 0)
        assert streamlines_pass_rois(both, rois)[0]
        assert not streamlines_pass_rois(only_origin, rois)[0]

    def test_or(self):
        rois = [
            self.roi(self.at_origin, RoiOperator.AND),
            self.roi(self.at_ten, RoiOperator.OR),
        ]
        assert streamlines_pass_rois(line(-5, 0, 0, 1, 0, 0), rois)[0]
        assert streamlines_pass_rois(line(9, 0, 0, 11, 0, 0), rois)[0]
        assert not streamlines_pass_rois(line(0, 50, 0, 10, 50, 0), rois)[0]

    def test_andnot(self):
        # Through the origin but NOT through x=10.
        rois = [
            self.roi(self.at_origin, RoiOperator.AND),
            self.roi(self.at_ten, RoiOperator.ANDNOT),
        ]
        assert streamlines_pass_rois(line(-5, 0, 0, 1, 0, 0), rois)[0]
        assert not streamlines_pass_rois(line(-5, 0, 0, 15, 0, 0), rois)[0]

    def test_folds_left_so_order_matters(self):
        through = line(-5, 0, 0, 15, 0, 0)  # passes both regions
        a = [
            self.roi(self.at_origin, RoiOperator.AND),
            self.roi(self.at_ten, RoiOperator.ANDNOT),
        ]
        b = [
            self.roi(self.at_ten, RoiOperator.AND),
            self.roi(self.at_origin, RoiOperator.OR),
        ]
        assert not streamlines_pass_rois(through, a)[0]
        assert streamlines_pass_rois(through, b)[0]

    def test_mixes_predicates(self):
        rois = [
            Roi(self.at_origin, RoiPredicate.ANY_SEGMENT, RoiOperator.AND),
            Roi(self.at_ten, RoiPredicate.EITHER_ENDPOINT, RoiOperator.AND),
        ]
        # Crosses the origin and terminates at x=10.
        assert streamlines_pass_rois(line(-5, 0, 0, 5, 0, 0, 10, 0, 0), rois)[0]
        # Crosses both but terminates beyond x=10.
        assert not streamlines_pass_rois(line(-5, 0, 0, 15, 0, 0), rois)[0]

    def test_short_circuit_does_not_change_the_verdict(self):
        # The batched fold skips a region when no tract's verdict could move.
        # Whatever it skips, the answer must match the unskipped evaluation.
        index = lines(
            [0, 50, 0, 1, 50, 0],  # passes neither
            [-5, 0, 0, 1, 0, 0],  # passes the origin only
            [-5, 0, 0, 15, 0, 0],  # passes both
        )
        for op in RoiOperator:
            rois = [
                self.roi(self.at_origin, RoiOperator.AND),
                self.roi(self.at_ten, op),
            ]
            batched = streamlines_pass_rois(index, rois)
            expected = [
                bool(streamlines_pass_rois(lines(t), rois)[0])
                for t in (
                    [0, 50, 0, 1, 50, 0],
                    [-5, 0, 0, 1, 0, 0],
                    [-5, 0, 0, 15, 0, 0],
                )
            ]
            assert list(batched) == expected


class TestFragments:
    """A tract stored as several fragments must behave as one tract."""

    @staticmethod
    def fragmented(*runs_with_ids) -> TractIndex:
        arrays = [
            np.array(r, dtype=np.float32).reshape(-1, 3) for r, _ in runs_with_ids
        ]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([len(a) for a in arrays], out=offsets[1:])
        ids = np.array([i for _, i in runs_with_ids], dtype=np.uint64)
        return TractIndex(np.concatenate(arrays), offsets, ids)

    def test_object_ids_are_deduplicated(self):
        index = self.fragmented(
            ([0, 0, 0, 1, 0, 0], 7), ([5, 0, 0, 6, 0, 0], 7), ([0, 9, 0, 1, 9, 0], 3)
        )
        assert len(index) == 2
        assert index.n_rows == 3
        assert list(index.object_ids) == [3, 7]

    def test_crossing_in_any_fragment_counts(self):
        # Object 7's second fragment crosses the sphere; its first does not.
        index = self.fragmented(
            ([0, 0, 0, 1, 0, 0], 7), ([49, 0, 0, 51, 0, 0], 7), ([0, 9, 0, 1, 9, 0], 3)
        )
        result = streamlines_pass_roi(index, sphere(50, 0, 0, 2))
        # object_ids are ascending: [3, 7]
        assert list(result) == [False, True]

    def test_and_fold_spanning_two_fragments(self):
        # The motivating case: region A is crossed by one fragment and region B
        # by another. Folding per fragment would drop the tract; OR-ing the
        # crossings first and then folding keeps it.
        index = self.fragmented(
            ([-1, 0, 0, 1, 0, 0], 7),  # crosses the origin
            ([9, 0, 0, 11, 0, 0], 7),  # crosses x=10
        )
        rois = [
            Roi(sphere(0, 0, 0, 1), RoiPredicate.ANY_SEGMENT, RoiOperator.AND),
            Roi(sphere(10, 0, 0, 1), RoiPredicate.ANY_SEGMENT, RoiOperator.AND),
        ]
        assert list(streamlines_pass_rois(index, rois)) == [True]

    def test_andnot_applies_to_the_whole_tract(self):
        # An exclusion crossed by *any* fragment removes the whole tract.
        index = self.fragmented(
            ([-1, 0, 0, 1, 0, 0], 7),
            ([9, 0, 0, 11, 0, 0], 7),
        )
        rois = [
            Roi(sphere(0, 0, 0, 1), RoiPredicate.ANY_SEGMENT, RoiOperator.AND),
            Roi(sphere(10, 0, 0, 1), RoiPredicate.ANY_SEGMENT, RoiOperator.ANDNOT),
        ]
        assert list(streamlines_pass_rois(index, rois)) == [False]

    def test_single_vertex_fragment_among_fragments(self):
        index = self.fragmented(
            ([50, 0, 0, 51, 0, 0], 7),  # away
            ([0.5, 0, 0], 7),  # a lone vertex inside
        )
        assert list(streamlines_pass_roi(index, sphere(0, 0, 0, 1))) == [True]

    def test_decimate_preserves_fragment_ids(self):
        index = self.fragmented(
            (list(np.stack([np.arange(50), np.zeros(50), np.zeros(50)], 1).ravel()), 7),
            (list(np.stack([np.arange(50), np.ones(50), np.zeros(50)], 1).ravel()), 7),
        )
        small = index.decimate(8)
        assert small.n_rows == 2
        assert len(small) == 1
        assert list(small.object_ids) == [7]


class TestBatching:
    def test_batching_matches_one_at_a_time(self):
        rng = np.random.default_rng(0)
        tracts = [
            rng.normal(scale=6.0, size=(int(rng.integers(1, 12)), 3)) for _ in range(60)
        ]
        index = lines(*[t.ravel() for t in tracts])
        shapes = [
            sphere(0, 0, 0, 3),
            Box(np.array([-4, -4, -4]), np.array([2, 2, 2])),
            Halfspace(np.array([0, 0, 0]), np.array([1, 1, 0])),
            Ellipsoid(np.array([1, 0, 0]), np.array([8, 2, 2])),
        ]
        for shape in shapes:
            for predicate in RoiPredicate:
                batched = streamlines_pass_roi(index, shape, predicate)
                one_by_one = [
                    bool(streamlines_pass_roi(lines(t.ravel()), shape, predicate)[0])
                    for t in tracts
                ]
                assert list(batched) == one_by_one, (shape, predicate)


class TestPlatformIndexDtypes:
    """Index arrays must be `intp`, not a hardcoded 64-bit width.

    Under Pyodide the platform is wasm32, where `intp` is 32-bit, and numpy
    refuses to cast an int64 `repeats` or index array down to it. That failure
    cannot be reproduced on a 64-bit test machine -- there `intp` *is* int64, so
    a hardcoded `np.int64` passes every runtime assertion. The source scan below
    is what actually catches a regression here; the dtype checks document the
    invariant.
    """

    def test_index_arrays_are_intp(self):
        index = lines([0, 0, 0, 1, 0, 0], [2, 0, 0, 3, 0, 0, 4, 0, 0])
        for name in (
            "offsets",
            "counts",
            "row_object",
            "vertex_object",
            "segment_object",
        ):
            assert getattr(index, name).dtype == np.intp, name
        a, b = index.segment_endpoints
        assert a.dtype == np.intp and b.dtype == np.intp
        assert index.decimate(2).offsets.dtype == np.intp

    def test_no_hardcoded_int64_in_package(self):
        import pathlib
        import re
        import tokenize

        import ngpy.tractography as pkg

        offenders = []
        for path in sorted(pathlib.Path(pkg.__file__).parent.glob("*.py")):
            with tokenize.open(path) as f:
                for token in tokenize.generate_tokens(f.readline):
                    # Only real code counts -- prose about int64 is fine, and so
                    # is a `"<i8"` on-disk field width, which is not an index.
                    if token.type in (tokenize.STRING, tokenize.COMMENT):
                        continue
                    # `uint64` is not an index type -- it is the segment id
                    # width neuroglancer itself uses -- so only signed int64,
                    # as its own word, is the mistake being guarded against.
                    if re.search(r"\bint64\b", token.string):
                        offenders.append(
                            f"{path.name}:{token.start[0]}: {token.line.strip()}"
                        )
        assert not offenders, (
            "hardcoded int64 breaks wasm32 (Pyodide), where intp is 32-bit; "
            "use np.intp for index/count arrays:\n  " + "\n  ".join(offenders)
        )


class TestTractIndex:
    def test_rejects_mismatched_offsets(self):
        with pytest.raises(ValueError, match="offsets must have"):
            TractIndex(
                np.zeros((4, 3), dtype=np.float32),
                np.array([0, 4]),
                np.array([1, 2], dtype=np.uint64),
            )

    def test_rejects_offsets_not_spanning_positions(self):
        with pytest.raises(ValueError, match="span positions exactly"):
            TractIndex(
                np.zeros((4, 3), dtype=np.float32),
                np.array([0, 3]),
                np.array([1], dtype=np.uint64),
            )

    def test_decimate_preserves_endpoints_and_ids(self):
        positions = np.stack(
            [np.arange(100, dtype=np.float32), np.zeros(100), np.zeros(100)], axis=1
        )
        index = TractIndex(
            positions, np.array([0, 100]), np.array([7], dtype=np.uint64)
        )
        small = index.decimate(10)
        assert len(small) == 1
        assert small.counts[0] == 10
        assert list(small.object_ids) == [7]
        assert small.positions[0, 0] == 0
        assert small.positions[-1, 0] == 99

    def test_decimate_leaves_short_tracts_alone(self):
        index = lines([0, 0, 0, 1, 0, 0], [2, 0, 0, 3, 0, 0, 4, 0, 0])
        assert index.decimate(8) is index

    def test_decimate_keeps_verdicts_for_a_smooth_tract(self):
        # A densely sampled straight-ish tract: decimation must not change which
        # regions it crosses, which is the assumption the loader relies on.
        t = np.linspace(0, 40, 400, dtype=np.float32)
        positions = np.stack([t, np.sin(t / 8) * 2, np.zeros_like(t)], axis=1)
        index = TractIndex(
            positions, np.array([0, len(t)]), np.array([1], dtype=np.uint64)
        )
        small = index.decimate(24)
        for shape in (
            sphere(20, 0, 0, 3),
            sphere(5, 1, 0, 2),
            Box(np.array([30, -3, -1]), np.array([35, 3, 1])),
            sphere(20, 40, 0, 2),
        ):
            assert bool(streamlines_pass_roi(index, shape)[0]) == bool(
                streamlines_pass_roi(small, shape)[0]
            ), shape


class TestLabelMask:
    def test_folding_without_vertex_labels_raises_a_clear_error(self):
        roi = Roi(LabelMask(labels=(1, 2)), RoiPredicate.ANY_VERTEX, RoiOperator.AND)
        with pytest.raises(NotImplementedError, match="label-mask"):
            streamlines_pass_rois(line(0, 0, 0, 1, 1, 1), [roi])

    def _index_and_labels(self):
        # tract 1 crosses labels 5 then 7; tract 2 stays in 9; tract 3 ends in 5
        index = lines(
            [0, 0, 0, 1, 0, 0, 2, 0, 0],
            [0, 5, 0, 1, 5, 0],
            [0, 9, 0, 1, 9, 0, 2, 9, 0],
        )
        labels = np.array([0, 5, 7, 9, 9, 3, 3, 5], dtype=np.int64)
        return index, labels

    def test_any_vertex_matches_any_listed_label(self):
        index, labels = self._index_and_labels()
        got = streamlines_pass_roi(
            index, LabelMask((7, 9)), RoiPredicate.ANY_VERTEX, vertex_labels=labels
        )
        assert list(got) == [True, True, False]

    def test_any_segment_is_treated_as_any_vertex(self):
        index, labels = self._index_and_labels()
        got = streamlines_pass_roi(index, LabelMask((5,)), vertex_labels=labels)
        assert list(got) == [True, False, True]

    def test_endpoint_predicates_read_the_end_labels(self):
        index, labels = self._index_and_labels()
        either = streamlines_pass_roi(
            index, LabelMask((5,)), RoiPredicate.EITHER_ENDPOINT, vertex_labels=labels
        )
        both = streamlines_pass_roi(
            index, LabelMask((7, 0)), RoiPredicate.BOTH_ENDPOINTS, vertex_labels=labels
        )
        assert list(either) == [False, False, True]
        assert list(both) == [True, False, False]

    def test_include_then_exclude_label_fold(self):
        # "crosses ANY included label AND no excluded one"
        index, labels = self._index_and_labels()
        rois = [
            Roi(LabelMask((5, 9)), RoiPredicate.ANY_VERTEX, RoiOperator.AND),
            Roi(LabelMask((7,)), RoiPredicate.ANY_VERTEX, RoiOperator.ANDNOT),
        ]
        got = streamlines_pass_rois(index, rois, vertex_labels=labels)
        assert list(got) == [False, True, True]

    def test_empty_label_list_matches_nothing(self):
        index, labels = self._index_and_labels()
        got = streamlines_pass_roi(index, LabelMask(()), vertex_labels=labels)
        assert not got.any()

    def test_rejects_misaligned_labels(self):
        index, labels = self._index_and_labels()
        with pytest.raises(ValueError, match="vertex_labels"):
            streamlines_pass_roi(index, LabelMask((5,)), vertex_labels=labels[:-1])
