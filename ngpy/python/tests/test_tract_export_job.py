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
"""Job-spec parsing for the tract exporter."""

import copy

import numpy as np
import pytest
from ngpy.tract_export import (
    JOB_SCHEMA_VERSION,
    JobSpecError,
    parse_job,
    roi_bounds,
)
from ngpy.tractography.roi import RoiOperator, RoiPredicate

# A spec in exactly the shape the Export tab posts: groups are `groupToJson`
# output, i.e. shapes keyed by `type` with string predicate/operator names.
#
# CONTRACT: this literal is duplicated, deliberately, as the expected value in
# `tests/ngpy/export_spec.spec.ts` (the wrapper's Export tab builder). The two
# languages have no shared schema, so the pairing is what keeps them honest --
# change one and the other suite fails.
#
# One asymmetry the pairing alone does not catch, hence the assertion below:
# `parse_job` accepts any version <= JOB_SCHEMA_VERSION, so `VALID` would keep
# parsing after a Python-side bump while the TS suite -- which asserts the
# current version in its *output* -- also stays green. The bump would then ship
# with the two sides silently disagreeing about the current version.
#
# v3 shape: each group also carries `objectIds` -- the viewer's on-screen passing
# ids, as decimal *strings* (an object id is a uint64 and can exceed JS's safe
# integer range). `rois` is retained for provenance but is not folded when
# `objectIds` is present. One id below is deliberately > 2**53 to pin the
# string round-trip.
VALID = {
    "schemaVersion": 3,
    "source": {"url": "zarr-vectors://gs://bucket/tracts.zvf", "level": 0},
    "groups": [
        {
            "name": "Motor CST",
            "color": "#ff0000",
            "rois": [
                {
                    "shape": {
                        "type": "ellipsoid",
                        "center": [10, 20, 30],
                        "radii": [5, 5, 5],
                    },
                    "predicate": "any_segment",
                    "operator": "and",
                    "name": "seed",
                },
                {
                    "shape": {
                        "type": "box",
                        "lower": [0, 0, 0],
                        "upper": [100, 100, 100],
                    },
                    "predicate": "either_endpoint",
                    "operator": "andnot",
                },
            ],
            "objectIds": ["7", "42", "9007199254740993"],
        }
    ],
    "format": "trk",
    "destination": {"kind": "download", "path": "out.trk"},
}


def test_fixture_pins_the_current_schema_version():
    """The fixture must track the version, not merely be accepted by it.

    Without this, bumping `JOB_SCHEMA_VERSION` on the Python side leaves both
    suites green -- see the CONTRACT note above. Bumping should force this
    fixture, and therefore its TypeScript twin, to be updated in the same
    change.
    """
    assert VALID["schemaVersion"] == JOB_SCHEMA_VERSION


def spec(**overrides):
    s = copy.deepcopy(VALID)
    s.update(overrides)
    return s


class TestParseJob:
    def test_parses_a_valid_spec(self):
        job = parse_job(VALID)
        assert job.source_url == "zarr-vectors://gs://bucket/tracts.zvf"
        assert job.level == 0
        assert job.format == "trk"
        assert job.destination.kind == "download"
        assert job.destination.path == "out.trk"
        assert job.affine is None
        assert len(job.groups) == 1
        assert job.groups[0].name == "Motor CST"

    def test_maps_string_enums_onto_the_fold_enums(self):
        rois = parse_job(VALID).groups[0].rois
        assert rois[0].predicate == RoiPredicate.ANY_SEGMENT
        assert rois[0].operator == RoiOperator.AND
        assert rois[1].predicate == RoiPredicate.EITHER_ENDPOINT
        assert rois[1].operator == RoiOperator.ANDNOT

    def test_drops_the_display_only_roi_name(self):
        # `name` exists in the persisted JSON but must not reach the fold.
        roi = parse_job(VALID).groups[0].rois[0]
        assert not hasattr(roi, "name")

    def test_parses_object_ids_as_uint64(self):
        # Ids arrive as decimal strings and must survive past 2**53 exactly.
        ids = parse_job(VALID).groups[0].object_ids
        assert ids == (7, 42, 9007199254740993)

    def test_object_ids_absent_yields_none(self):
        # A legacy (fold-path) group carries no objectIds.
        s = spec()
        del s["groups"][0]["objectIds"]
        assert parse_job(s).groups[0].object_ids is None

    def test_object_ids_present_makes_rois_optional(self):
        # A v3 group need not carry rois at all -- selection is the id list.
        s = spec()
        del s["groups"][0]["rois"]
        job = parse_job(s)
        assert job.groups[0].rois == ()
        assert job.groups[0].object_ids == (7, 42, 9007199254740993)

    def test_empty_object_ids_is_a_valid_empty_selection(self):
        # Empty ids means "nothing passed on screen", NOT "select everything"
        # (which is what an empty rois fold would mean).
        s = spec()
        s["groups"][0]["objectIds"] = []
        del s["groups"][0]["rois"]
        assert parse_job(s).groups[0].object_ids == ()

    def test_accepts_integer_object_ids_too(self):
        # A hand-written spec may use plain ints; strings are only needed to keep
        # the browser's JSON precise past 2**53.
        s = spec()
        s["groups"][0]["objectIds"] = [1, 2, 3]
        assert parse_job(s).groups[0].object_ids == (1, 2, 3)

    def test_rejects_a_non_integer_object_id(self):
        s = spec()
        s["groups"][0]["objectIds"] = ["7", "not-a-number"]
        with pytest.raises(JobSpecError, match="objectIds"):
            parse_job(s)

    def test_rejects_a_negative_object_id(self):
        s = spec()
        s["groups"][0]["objectIds"] = ["-1"]
        with pytest.raises(JobSpecError, match="objectIds"):
            parse_job(s)

    def test_level_defaults_to_zero(self):
        s = spec()
        del s["source"]["level"]
        assert parse_job(s).level == 0

    def test_accepts_a_4x4_affine(self):
        s = spec(affine=np.eye(4).tolist())
        assert parse_job(s).affine.shape == (4, 4)

    def test_all_three_shape_types(self):
        for shape in (
            {"type": "ellipsoid", "center": [0, 0, 0], "radii": [1, 1, 1]},
            {"type": "box", "lower": [0, 0, 0], "upper": [1, 1, 1]},
            {"type": "halfspace", "origin": [0, 0, 0], "normal": [0, 0, 1]},
        ):
            s = spec()
            s["groups"][0]["rois"] = [
                {"shape": shape, "predicate": "any_segment", "operator": "and"}
            ]
            assert len(parse_job(s).groups[0].rois) == 1

    def test_parses_a_label_mask_shape_as_provenance(self):
        # A label dissection is exported by objectIds (which VALID carries); its
        # labelMask rois travel as provenance and must parse without error.
        s = spec()
        s["groups"][0]["rois"] = [
            {
                "shape": {"type": "labelMask", "labels": [17, 53]},
                "predicate": "any_segment",
                "operator": "and",
            }
        ]
        roi = parse_job(s).groups[0].rois[0]
        assert roi.shape.__class__.__name__ == "LabelMask"
        assert roi.shape.labels == (17, 53)

    def test_rejects_non_integer_label_mask_labels(self):
        s = spec()
        s["groups"][0]["rois"] = [
            {
                "shape": {"type": "labelMask", "labels": ["x"]},
                "predicate": "any_segment",
                "operator": "and",
            }
        ]
        with pytest.raises(JobSpecError, match="labels"):
            parse_job(s)


class TestScope:
    def test_defaults_to_selected(self):
        # A v1 spec had no `scope`; it must keep folding the dissection.
        s = spec()
        assert "scope" not in s
        assert parse_job(s).scope == "selected"

    def test_selected_scope_still_requires_groups(self):
        with pytest.raises(JobSpecError, match="groups"):
            parse_job(spec(scope="selected", groups=[]))

    def test_whole_scope_allows_empty_groups(self):
        job = parse_job(spec(scope="whole", groups=[]))
        assert job.scope == "whole"
        assert job.groups == ()

    def test_whole_scope_allows_absent_groups(self):
        s = spec(scope="whole")
        del s["groups"]
        assert parse_job(s).groups == ()

    def test_rejects_an_unknown_scope(self):
        with pytest.raises(JobSpecError, match="scope"):
            parse_job(spec(scope="everything"))


class TestRejections:
    def test_rejects_a_newer_schema_version(self):
        with pytest.raises(JobSpecError, match="newer than this exporter"):
            parse_job(spec(schemaVersion=4))

    def test_rejects_an_unknown_format(self):
        with pytest.raises(JobSpecError, match="format"):
            parse_job(spec(format="obj"))

    def test_rejects_an_unknown_destination_kind(self):
        with pytest.raises(JobSpecError, match="destination.kind"):
            parse_job(spec(destination={"kind": "s3", "path": "x"}))

    def test_accepts_every_page_destination(self):
        # Under ngpy Python returns bytes; the page downloads or uploads them,
        # so download / gcs / local are all just a file name here.
        for kind in ("download", "gcs", "local"):
            job = parse_job(spec(destination={"kind": kind, "path": "x.trk"}))
            assert job.destination.kind == kind

    def test_destination_is_optional(self):
        s = spec()
        del s["destination"]
        assert parse_job(s).destination.kind == "download"

    def test_rejects_a_missing_source_url(self):
        with pytest.raises(JobSpecError, match="url"):
            parse_job(spec(source={"level": 0}))

    def test_rejects_empty_groups(self):
        with pytest.raises(JobSpecError, match="groups"):
            parse_job(spec(groups=[]))

    def test_rejects_a_group_with_no_rois(self):
        # An empty fold passes everything, so this would export the entire
        # dataset under the group's name rather than a dissection. Only applies
        # to the fold path -- an id-based group has no rois by design -- so drop
        # objectIds first.
        s = spec()
        del s["groups"][0]["objectIds"]
        s["groups"][0]["rois"] = []
        with pytest.raises(JobSpecError, match="every streamline"):
            parse_job(s)

    def test_rejects_an_unknown_shape_type(self):
        s = spec()
        s["groups"][0]["rois"][0]["shape"]["type"] = "cone"
        with pytest.raises(JobSpecError, match="cone"):
            parse_job(s)

    def test_rejects_an_unknown_operator(self):
        s = spec()
        s["groups"][0]["rois"][0]["operator"] = "nand"
        with pytest.raises(JobSpecError, match="nand"):
            parse_job(s)

    def test_names_the_offending_roi(self):
        s = spec()
        del s["groups"][0]["rois"][1]["predicate"]
        with pytest.raises(JobSpecError, match=r"groups\[0\].rois\[1\]"):
            parse_job(s)

    def test_rejects_a_non_4x4_affine(self):
        with pytest.raises(JobSpecError, match="4x4"):
            parse_job(spec(affine=[[1, 0], [0, 1]]))

    def test_rejects_a_wire_format_shape(self):
        # The wire encoding keys shapes by `kind`, not `type`. Accepting it here
        # would silently diverge the two encodings.
        s = spec()
        s["groups"][0]["rois"][0]["shape"] = {
            "kind": "ellipsoid",
            "center": [0, 0, 0],
            "radii": [1, 1, 1],
        }
        with pytest.raises(JobSpecError, match="type"):
            parse_job(s)


class TestRoiBounds:
    def _roi(self, shape, operator="and"):
        s = spec()
        s["groups"][0]["rois"] = [
            {"shape": shape, "predicate": "any_segment", "operator": operator}
        ]
        return parse_job(s).groups[0].rois

    def test_encloses_an_ellipsoid(self):
        rois = self._roi(
            {"type": "ellipsoid", "center": [10, 10, 10], "radii": [2, 3, 4]}
        )
        lower, upper = roi_bounds(rois)
        assert list(lower) == [8, 7, 6]
        assert list(upper) == [12, 13, 14]

    def test_unions_several_regions(self):
        rois = (
            parse_job(
                spec(
                    groups=[
                        {
                            "name": "g",
                            "color": "#ffffff",
                            "rois": [
                                {
                                    "shape": {
                                        "type": "box",
                                        "lower": [0, 0, 0],
                                        "upper": [1, 1, 1],
                                    },
                                    "predicate": "any_segment",
                                    "operator": "and",
                                },
                                {
                                    "shape": {
                                        "type": "box",
                                        "lower": [5, 5, 5],
                                        "upper": [7, 7, 7],
                                    },
                                    "predicate": "any_segment",
                                    "operator": "or",
                                },
                            ],
                        }
                    ]
                )
            )
            .groups[0]
            .rois
        )
        lower, upper = roi_bounds(rois)
        assert list(lower) == [0, 0, 0]
        assert list(upper) == [7, 7, 7]

    def test_unbounded_for_a_halfspace(self):
        rois = self._roi(
            {"type": "halfspace", "origin": [0, 0, 0], "normal": [0, 0, 1]}
        )
        assert roi_bounds(rois) is None

    def test_unbounded_when_any_region_excludes(self):
        # What passes an exclusion is its *complement*, which no finite box
        # encloses -- narrowing the read to the region itself would drop exactly
        # the tracts that should survive.
        rois = self._roi(
            {"type": "ellipsoid", "center": [0, 0, 0], "radii": [1, 1, 1]},
            operator="andnot",
        )
        assert roi_bounds(rois) is None
