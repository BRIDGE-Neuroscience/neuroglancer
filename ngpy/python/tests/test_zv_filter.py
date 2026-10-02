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

"""Store reads (ngpy.zv), parcellation sampling (ngpy.labels) and the
whole-store dissection engine (ngpy.filter), against REAL tiny zarr-vectors
stores written with zarr-vectors-py and read back over a local "fetch"."""

import asyncio
import json
import os

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
polylines_mod = pytest.importorskip("zarr_vectors.types.polylines")

from ngpy.filter import FilterEngine  # noqa: E402
from ngpy.tractography import zvf_pure  # noqa: E402

from ngpy import labels, zv  # noqa: E402

write_polylines = polylines_mod.write_polylines


def run(coro):
    return asyncio.run(coro)


# Four tracts in a 40 mm cube with 10 mm chunks:
#   0: along x at y=z=5, x 1..29  (crosses chunks 0..2)
#   1: along x at y=z=25, x 1..9 (one chunk)
#   2: along y at x=z=35, y 1..39
#   3: a short stub at (15, 15, 15)
TRACTS = [
    np.array([[1, 5, 5], [12, 5, 5], [21, 5, 5], [29, 5, 5]], np.float32),
    np.array([[1, 25, 25], [5, 25, 25], [9, 25, 25]], np.float32),
    np.array([[35, 1, 35], [35, 20, 35], [35, 39, 35]], np.float32),
    np.array([[15, 15, 15], [16, 15, 15]], np.float32),
]
LENGTH = np.array([28.0, 8.0, 38.0, 1.0], np.float32)


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "tracts.zarrvectors")
    write_polylines(
        path,
        TRACTS,
        chunk_shape=(10.0, 10.0, 10.0),
        bounds=([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
        # A per-vertex attribute makes zarr-vectors' own read_polylines need a
        # LISTING of vertex_attributes/, which a plain HTTP host cannot give.
        vertex_attributes={"z": [t[:, 2].copy() for t in TRACTS]},
        object_attributes={
            "length": LENGTH,
            "orientation": np.eye(4, 3, dtype=np.float32),
        },
    )
    return path


def engine():
    zv.forget_stores()
    return FilterEngine(fetch=zv.local_fetch)


def box(lo, hi, op="and", predicate="any_segment"):
    return {
        "shape": {"type": "box", "lower": list(lo), "upper": list(hi)},
        "predicate": predicate,
        "operator": op,
    }


def evaluate(store, groups, **extra):
    request = {"source": f"{store}/|zarr-vectors:", "groups": groups, **extra}
    return run(engine().evaluate(request))


# -- ngpy.zv ----------------------------------------------------------------------


class TestUrls:
    def test_store_path_and_http_base(self):
        assert zv.http_base("gs://b/p/x.zarrvectors/|zarr-vectors:") == (
            "https://storage.googleapis.com/b/p/x.zarrvectors/"
        )
        assert zv.http_base("zarr-vectors://https://h/x") == "https://h/x/"
        assert zv.store_path_from_url("/local/dir/|zarr-vectors:") == "/local/dir/"

    def test_gcs_listing_helpers(self):
        url = zv.gcs_list_url("https://storage.googleapis.com/bkt/a/b/")
        assert url.startswith("https://storage.googleapis.com/storage/v1/b/bkt/o?")
        assert "prefix=a%2Fb%2F" in url and "delimiter=%2F" in url
        assert zv.gcs_list_url("https://example.com/a/") is None
        page = {"prefixes": ["a/b/x/", "a/b/y/"], "items": [{"name": "a/b/zarr.json"}]}
        assert zv.gcs_children(page, "a/b/") == ["x", "y", "zarr.json"]


class TestRangedFetch:
    class _R:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def test_byte_request_to_header(self):
        R = self._R
        assert zvf_pure.byte_range_header(None) is None
        assert zvf_pure.byte_range_header(R(start=2, end=5)) == "bytes=2-4"
        assert zvf_pure.byte_range_header(R(offset=7)) == "bytes=7-"
        assert zvf_pure.byte_range_header(R(suffix=3)) == "bytes=-3"

    def test_slicing_a_whole_object_for_a_server_that_ignored_range(self):
        data = bytes(range(10))
        assert zvf_pure.slice_for_range_header(data, "bytes=2-4") == bytes([2, 3, 4])
        assert zvf_pure.slice_for_range_header(data, "bytes=7-") == bytes([7, 8, 9])
        assert zvf_pure.slice_for_range_header(data, "bytes=-3") == bytes([7, 8, 9])

    def test_local_fetch_honours_ranges(self, tmp_path):
        p = tmp_path / "f"
        p.write_bytes(bytes(range(10)))
        assert run(zv.local_fetch(str(p), "bytes=2-4")) == bytes([2, 3, 4])
        assert run(zv.local_fetch(str(p), "bytes=-2")) == bytes([8, 9])
        assert run(zv.local_fetch(str(tmp_path / "missing"))) is None


class TestStoreReads:
    def test_upstream_read_polylines_stalls_without_listing(self, store):
        # Why ngpy has its own geometry reader: zarr-vectors 0.9.2's
        # read_polylines enumerates vertex_attributes/ and so cannot finish on a
        # store that cannot list. (If a later zarr-vectors fixes this, this test
        # fails and the custom reader can be revisited.)
        from zarr_vectors.core.aio import read_async
        from zarr_vectors.exceptions import StoreError

        s = zv.ZvStore(zv.http_base(store), zv.local_fetch, None)

        async def go():
            return await read_async(
                polylines_mod.read_polylines, await s.root(), level=0
            )

        with pytest.raises(StoreError, match="list"):
            run(go())

    def test_level_geometry_needs_no_listing_and_joins_fragments(self, store):
        s = zv.ZvStore(zv.http_base(store), zv.local_fetch, None)
        index, rows = run(s.level_geometry(0))
        assert list(index.object_ids) == [0, 1, 2, 3]
        assert list(rows) == [0, 1, 2, 3]
        # Tract 0 crosses three chunks but comes back as ONE row, in order.
        assert index.n_rows == 4
        np.testing.assert_allclose(index.positions[: index.counts[0]], TRACTS[0])

    def test_polylines_subset(self, store):
        s = zv.ZvStore(zv.http_base(store), zv.local_fetch, None)
        ids, lines = run(s.polylines(0, [2, 0, 99]))
        assert list(ids) == [2, 0]
        np.testing.assert_allclose(lines[0], TRACTS[2])

    def test_store_info_with_and_without_listing(self, store):
        listed = zv.ZvStore(
            zv.http_base(store), zv.local_fetch, zv.make_lister(zv.local_fetch)
        )
        info = run(listed.info())
        assert [lv.level for lv in info.levels] == [0]
        assert info.levels[0].vertex_count == sum(len(t) for t in TRACTS)
        assert info.axes == ["x", "y", "z"]
        assert {a["name"]: a["ncols"] for a in info.object_attributes} == {
            "length": 1,
            "orientation": 3,
        }
        assert info.vertex_attributes == ["z"]
        assert info.choose_level() == 0
        unlisted = zv.ZvStore(zv.http_base(store), zv.local_fetch, None)
        assert run(unlisted.info()).object_attributes == []

    def test_object_attribute_and_identity_segment_ids(self, store):
        s = zv.ZvStore(zv.http_base(store), zv.local_fetch, None)
        ids, values = run(s.object_attribute(0, "length"))
        assert list(ids) == [0, 1, 2, 3]
        np.testing.assert_allclose(values, LENGTH)
        # No object_attributes/segment_id: the segment id IS the object id.
        assert list(run(s.segment_ids_for(np.array([3, 1])))) == [3, 1]


# -- ngpy.filter -------------------------------------------------------------------


class TestEngine:
    def test_no_groups_is_inactive_and_selects_nothing(self, store):
        r = evaluate(store, [])
        assert r["active"] is False and r["segments"] == [] and r["colors"] == {}
        assert r["level"] == 0 and r["levelObjects"] == 4

    def test_box_include_writes_segments_and_group_colour(self, store):
        r = evaluate(
            store,
            [
                {
                    "id": 1,
                    "name": "g",
                    "color": "#ff0000",
                    "rois": [box([10, 0, 0], [20, 10, 10])],
                }
            ],
        )
        assert r["active"] is True
        assert r["segments"] == ["0"]
        assert r["colors"] == {"0": "#ff0000"}
        assert r["groups"] == [{"id": 1, "count": 1}]

    def test_leap_across_is_caught_by_any_segment_but_not_any_vertex(self, store):
        # Tract 2 has vertices at y=1, 20, 39: a thin box at y=10 sits between.
        thin = [30, 9, 30], [40, 11, 40]
        seg = evaluate(store, [{"color": "#fff", "rois": [box(*thin)]}])
        vtx = evaluate(
            store, [{"color": "#fff", "rois": [box(*thin, predicate="any_vertex")]}]
        )
        assert seg["segments"] == ["2"] and vtx["segments"] == []

    def test_exclusion_only_group_selects_the_complement(self, store):
        r = evaluate(
            store,
            [{"color": "#00ff00", "rois": [box([0, 0, 0], [10, 10, 10], op="andnot")]}],
        )
        assert r["segments"] == ["1", "2", "3"]

    def test_first_visible_group_wins_the_colour_and_invisible_groups_do_nothing(
        self, store
    ):
        everything = box([0, 0, 0], [40, 40, 40])
        r = evaluate(
            store,
            [
                {"id": 1, "color": "#111111", "visible": False, "rois": [everything]},
                {"id": 2, "color": "#222222", "rois": [box([0, 20, 20], [10, 30, 30])]},
                {"id": 3, "color": "#333333", "rois": [everything]},
            ],
        )
        assert r["segments"] == ["0", "1", "2", "3"]
        assert r["colors"] == {
            "0": "#333333",
            "1": "#222222",
            "2": "#333333",
            "3": "#333333",
        }
        assert [g["count"] for g in r["groups"]] == [4, 1, 4]

    def test_attribute_only_group_and_attribute_and_roi(self, store):
        r = evaluate(
            store,
            [
                {
                    "color": "#fff",
                    "attrFilters": [{"name": "length", "min": 5, "max": 30}],
                }
            ],
        )
        assert r["segments"] == ["0", "1"]
        r = evaluate(
            store,
            [
                {
                    "color": "#fff",
                    "rois": [box([0, 0, 0], [40, 40, 40])],
                    "attrFilters": [{"name": "orientation[1]", "min": 0.5, "max": 1}],
                }
            ],
        )
        assert r["segments"] == ["1"]

    def test_empty_group_does_not_activate_the_filter(self, store):
        r = evaluate(store, [{"color": "#fff", "rois": [], "attrFilters": []}])
        assert r["active"] is False and r["segments"] == []

    def test_label_group_against_an_ome_zarr_parcellation(self, store, tmp_path):
        vol = _write_ome_labels(tmp_path)
        parc = {"url": f"{vol}/|zarr2:"}
        include = {
            "shape": {"type": "labelMask", "labels": [7]},
            "predicate": "any_vertex",
            "operator": "and",
        }
        exclude = {
            "shape": {"type": "labelMask", "labels": [9]},
            "predicate": "any_vertex",
            "operator": "andnot",
        }
        r = evaluate(
            store, [{"color": "#abcdef", "rois": [include], "parcellation": parc}]
        )
        assert r["segments"] == ["0", "3"]
        r = evaluate(
            store,
            [{"color": "#abcdef", "rois": [include, exclude], "parcellation": parc}],
        )
        assert r["segments"] == ["3"]

    def test_label_group_without_parcellation_is_a_clear_error(self, store):
        lm = {
            "shape": {"type": "labelMask", "labels": [1]},
            "predicate": "any_vertex",
            "operator": "and",
        }
        with pytest.raises(ValueError, match="parcellation"):
            evaluate(store, [{"color": "#fff", "rois": [lm]}])

    def test_passing_ids_per_group(self, store):
        request = {
            "source": store,
            "groups": [
                {"name": "a", "color": "#fff", "rois": [box([0, 0, 0], [10, 10, 10])]},
                {
                    "name": "b",
                    "color": "#fff",
                    "visible": False,
                    "rois": [box([30, 0, 30], [40, 40, 40])],
                },
            ],
        }
        got = run(engine().passing_ids(request))
        assert got == [
            {"name": "a", "objectIds": ["0"]},
            {"name": "b", "objectIds": ["2"]},
        ]


# -- ngpy.labels -------------------------------------------------------------------


def _write_ome_labels(tmp_path):
    """A 40^3 mm label volume at 2 mm voxels, in MICROMETRES, axes z,y,x.

    Label 7 fills x < 18 mm in the slab z,y < 20 mm; label 9 fills x < 10 mm
    in that slab (overriding 7).  Tract 0 (y=z=5, x 1..29) therefore crosses 9
    and 7; tract 3 (15,15,15) sits in 7; tracts 1 and 2 touch neither.
    """
    path = str(tmp_path / "parc.zarr")
    data = np.zeros((20, 20, 20), dtype=np.uint16)  # z, y, x at 2 mm
    data[:10, :10, :9] = 7
    data[:10, :10, :5] = 9
    arr = zarr.open_array(
        store=os.path.join(path, "0"),
        mode="w",
        shape=data.shape,
        chunks=(8, 8, 8),
        dtype="u2",
        zarr_format=2,
    )
    arr[...] = data
    with open(os.path.join(path, ".zgroup"), "w") as f:
        json.dump({"zarr_format": 2}, f)
    with open(os.path.join(path, ".zattrs"), "w") as f:
        json.dump(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [
                            {"name": n, "type": "space", "unit": "micrometer"}
                            for n in ("z", "y", "x")
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [2000.0] * 3},
                                    {
                                        "type": "translation",
                                        "translation": [1000.0] * 3,
                                    },
                                ],
                            }
                        ],
                    }
                ]
            },
            f,
        )
    return path


class TestLabels:
    def test_parse_volume_source_spellings(self):
        assert labels.parse_volume_source("gs://b/p.zarr/|zarr2:") == (
            "zarr",
            "https://storage.googleapis.com/b/p.zarr/",
        )
        assert labels.parse_volume_source("zarr3://https://h/v") == (
            "zarr",
            "https://h/v/",
        )
        assert labels.parse_volume_source("precomputed://https://h/v") == (
            "precomputed",
            "https://h/v/",
        )
        with pytest.raises(labels.LabelSourceError):
            labels.parse_volume_source("https://h/unknown")

    def test_ome_zarr_sampling_matches_axes_by_name_and_units(self, tmp_path):
        vol = run(
            labels.open_label_volume(_write_ome_labels(tmp_path), fetch=zv.local_fetch)
        )
        assert vol.axes == ["z", "y", "x"]
        np.testing.assert_allclose(vol.scale_m, [2e-3] * 3)
        # Points in x,y,z millimetres -> metres.
        pts = np.array([[3, 5, 5], [15, 15, 15], [30, 5, 5], [500, 5, 5]], float) * 1e-3
        assert list(vol.sample(pts, ["x", "y", "z"])) == [9, 7, 0, 0]

    def test_precomputed_raw_volume(self, tmp_path):
        base = tmp_path / "pre"
        (base / "s0").mkdir(parents=True)
        info = {
            "type": "segmentation",
            "data_type": "uint32",
            "num_channels": 1,
            "scales": [
                {
                    "key": "s0",
                    "size": [4, 4, 2],
                    "resolution": [1e6, 1e6, 1e6],  # 1 mm in nm
                    "voxel_offset": [0, 0, 0],
                    "chunk_sizes": [[4, 4, 2]],
                    "encoding": "raw",
                }
            ],
        }
        (base / "info").write_text(json.dumps(info))
        block = np.zeros((2, 4, 4), dtype="<u4")  # z, y, x
        block[1, 2, 3] = 42
        (base / "s0" / "0-4_0-4_0-2").write_bytes(block.tobytes())
        vol = run(
            labels.open_label_volume(f"precomputed://{base}", fetch=zv.local_fetch)
        )
        pts = np.array([[3.5, 2.5, 1.5], [0.5, 0.5, 0.5]]) * 1e-3
        assert list(vol.sample(pts, ["x", "y", "z"])) == [42, 0]

    def test_segment_properties_names(self, tmp_path):
        d = tmp_path / "props"
        d.mkdir()
        (d / "info").write_text(
            json.dumps(
                {
                    "@type": "neuroglancer_segment_properties",
                    "inline": {
                        "ids": ["2", "41"],
                        "properties": [
                            {
                                "id": "label",
                                "type": "label",
                                "values": ["Left WM", "Right WM"],
                            }
                        ],
                    },
                }
            )
        )
        got = run(
            labels.read_segment_properties(f"precomputed://{d}", fetch=zv.local_fetch)
        )
        assert got["ids"] == ["2", "41"] and got["names"] == ["Left WM", "Right WM"]
