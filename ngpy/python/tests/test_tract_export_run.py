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
"""The pure selection helpers of ``ngpy.tract_export.run``.

Ported from the pure parts of ``python/tests/tract_export_run_test.py``
(branch ``zarr_vectors_roi_store``); the native ``run_job`` it also covered is
not part of ngpy.
"""

import copy

import numpy as np
import pytest
from ngpy.tract_export import parse_job
from ngpy.tract_export.run import (
    ExportRunError,
    build_index,
    passing_object_ids,
    per_group_from_object_ids,
    select_from,
    store_path_from_url,
    union_ids,
    uses_explicit_ids,
)

BASE = {
    "schemaVersion": 1,
    "source": {"url": "zarr-vectors://gs://bucket/tracts.zvf", "level": 0},
    "groups": [],
    "format": "trk",
    "destination": {"kind": "download", "path": "out.trk"},
}


def job_with(rois, **overrides):
    spec = copy.deepcopy(BASE)
    spec["groups"] = [{"name": "g", "color": "#ff0000", "rois": rois}]
    spec.update(overrides)
    return parse_job(spec)


def sphere(center, radius, operator="and", predicate="any_segment"):
    return {
        "shape": {"type": "ellipsoid", "center": list(center), "radii": [radius] * 3},
        "predicate": predicate,
        "operator": operator,
    }


def line(*points):
    return [np.array(points, dtype=np.float32)]


class TestStorePathFromUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("zarr-vectors://gs://b/x.zvf", "gs://b/x.zvf"),
            ("zarr://gs://b/x.zvf", "gs://b/x.zvf"),
            ("zarr3://gs://b/x.zvf", "gs://b/x.zvf"),
            ("/data/tracts.zvf", "/data/tracts.zvf"),
            ("gs://b/x.zvf", "gs://b/x.zvf"),
            ("https://h/b/store.zarrvectors/|zarr-vectors:", "https://h/b/store.zarrvectors/"),
            ("https://h/b/store/|zarr-vectors:foo=1", "https://h/b/store/"),
        ],
    )
    def test_strips_the_datasource_decoration(self, url, expected):
        assert store_path_from_url(url) == expected

    def test_strips_at_most_one_prefix(self):
        assert store_path_from_url("zarr-vectors://zarr://x") == "zarr://x"


class TestBuildIndex:
    def test_one_row_per_polyline_with_fragments_concatenated(self):
        polylines = [[np.zeros((3, 3), np.float32), np.ones((2, 3), np.float32)]]
        index = build_index(polylines, [7])
        assert index.n_rows == 1
        assert index.offsets.tolist() == [0, 5]
        assert index.object_ids.tolist() == [7]

    def test_rows_align_with_object_ids(self):
        index = build_index(
            [line((0, 0, 0), (1, 0, 0)), line((5, 0, 0), (6, 0, 0))], [11, 4]
        )
        assert index.n_rows == 2
        assert index.object_ids.tolist() == [4, 11]

    def test_empty_input_yields_an_empty_index(self):
        index = build_index([], [])
        assert len(index) == 0 and index.n_rows == 0

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ExportRunError, match="must correspond"):
            build_index([line((0, 0, 0), (1, 0, 0))], [1, 2])


class TestPassingObjectIds:
    def _index(self):
        return build_index(
            [
                line((-5, 0, 0), (5, 0, 0)),
                line((50, 0, 0), (55, 0, 0)),
                line((-5, 0, 0), (55, 0, 0)),
            ],
            [1, 2, 3],
        )

    def test_include_selects_crossing_objects(self):
        ids = passing_object_ids(self._index(), job_with([sphere((0, 0, 0), 2)]).groups[0])
        assert ids.tolist() == [1, 3]

    def test_leading_exclusion_selects_the_complement(self):
        group = job_with([sphere((0, 0, 0), 2, operator="andnot")]).groups[0]
        assert passing_object_ids(self._index(), group).tolist() == [2]

    def test_include_then_exclude(self):
        group = job_with(
            [sphere((0, 0, 0), 2), sphere((52, 0, 0), 5, operator="andnot")]
        ).groups[0]
        assert passing_object_ids(self._index(), group).tolist() == [1]


class TestSelectionAcrossGroups:
    def _two_groups(self, name_a, name_b):
        spec = copy.deepcopy(BASE)
        spec["groups"] = [
            {"name": name_a, "color": "#ff0000", "rois": [sphere((0, 0, 0), 2)]},
            {"name": name_b, "color": "#00ff00", "rois": [sphere((50, 0, 0), 2)]},
        ]
        return parse_job(spec)

    def _geometry(self):
        return [line((-5, 0, 0), (5, 0, 0)), line((45, 0, 0), (55, 0, 0))], [1, 2]

    def test_unions_ids_across_groups(self):
        polylines, ids = self._geometry()
        _, per_group = select_from(polylines, ids, self._two_groups("A", "B").groups)
        assert [name for name, _ in per_group] == ["A", "B"]
        assert union_ids(per_group).tolist() == [1, 2]

    def test_duplicate_group_names_do_not_collapse(self):
        polylines, ids = self._geometry()
        _, per_group = select_from(
            polylines, ids, self._two_groups("Group 3", "Group 3").groups
        )
        assert len(per_group) == 2
        assert union_ids(per_group).tolist() == [1, 2]

    def test_batch_size_does_not_change_the_selection(self):
        polylines, ids = self._geometry()
        groups = self._two_groups("A", "B").groups
        _, one = select_from(list(polylines), ids, groups, batch=1)
        _, all_ = select_from(list(polylines), ids, groups)
        assert [v.tolist() for _, v in one] == [v.tolist() for _, v in all_]

    def test_union_of_no_groups_is_empty(self):
        assert union_ids([]).tolist() == []


class TestExplicitObjectIds:
    def _job(self, groups):
        s = copy.deepcopy(BASE)
        s["schemaVersion"] = 3
        s["groups"] = groups
        return parse_job(s)

    def test_per_group_from_object_ids_dedups_and_sorts(self):
        job = self._job([{"name": "A", "color": "#ff0000", "objectIds": ["5", "5", "3"]}])
        per_group = per_group_from_object_ids(job.groups)
        assert per_group[0][1].tolist() == [3, 5]
        assert uses_explicit_ids(job.groups)

    def test_mixed_id_and_fold_groups_are_rejected(self):
        job = self._job(
            [
                {"name": "A", "color": "#ff0000", "objectIds": ["1"]},
                {"name": "B", "color": "#00ff00", "rois": [sphere((0, 0, 0), 2)]},
            ]
        )
        with pytest.raises(ExportRunError, match="objectIds for all groups or none"):
            uses_explicit_ids(job.groups)
