# @license
# Copyright 2026 Google Inc.
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

"""In-browser tract export (``ngpy.tract_export.browser``) against a real store.

Ported from ``python/tests/tract_export_browser_test.py`` (branch
``zarr_vectors_roi_store``). Drives ``export_async`` exactly as the Pyodide
worker does -- the async read through :mod:`ngpy.zv` and the in-memory TRK /
zipped-ZVF writers -- over a local-file fetch, so it runs in CPython. What it
cannot cover is the JSPI/promising behaviour of a ``.zvf`` export under real
Pyodide (see the browser check in the README).
"""

import asyncio
import io
import zipfile

import numpy as np
import pytest

pytest.importorskip("zarr")
polylines_mod = pytest.importorskip("zarr_vectors.types.polylines")
pytest.importorskip("zarr_vectors.core.aio")
nib = pytest.importorskip("nibabel")

from ngpy.tract_export import parse_job  # noqa: E402
from ngpy.tract_export.browser import export_async  # noqa: E402

from ngpy import zv  # noqa: E402

read_polylines = polylines_mod.read_polylines
write_polylines = polylines_mod.write_polylines


@pytest.fixture
def store(tmp_path):
    """A three-streamline store: two clustered near (10,10,10), one far."""
    path = str(tmp_path / "tracts.zvf")
    s0 = np.array([[10, 10, 10], [12, 10, 10], [14, 10, 10]], dtype=np.float32)
    s1 = np.array([[11, 11, 11], [13, 11, 11]], dtype=np.float32)
    s2 = np.array([[200, 200, 200], [210, 200, 200]], dtype=np.float32)
    write_polylines(
        path,
        [s0, s1, s2],
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0.0, 0.0, 0.0], [300.0, 300.0, 300.0]),
    )
    return path


@pytest.fixture(autouse=True)
def _fresh_stores():
    zv.forget_stores()
    yield
    zv.forget_stores()


def _spec(store_path, *, fmt="trk", scope="selected", out="dissection"):
    groups = (
        []
        if scope == "whole"
        else [
            {
                "name": "near",
                "color": "#ff0000",
                "rois": [
                    {
                        # A sphere around the clustered pair; the far tract misses.
                        "shape": {
                            "type": "ellipsoid",
                            "center": [11, 11, 11],
                            "radii": [8, 8, 8],
                        },
                        "predicate": "any_segment",
                        "operator": "and",
                    }
                ],
            }
        ]
    )
    return {
        "schemaVersion": 2,
        "source": {"url": f"zarr-vectors://{store_path}", "level": 0},
        "groups": groups,
        "format": fmt,
        "scope": scope,
        "destination": {"kind": "local", "path": out},
    }


def _id_spec(store_path, object_ids, *, fmt="trk", out="dissection"):
    """A v3 spec that hands the exporter explicit ids (the WYSIWYG path)."""
    return {
        "schemaVersion": 3,
        "source": {"url": f"zarr-vectors://{store_path}", "level": 0},
        "groups": [
            {
                "name": "picked",
                "color": "#ff0000",
                "objectIds": [str(i) for i in object_ids],
            }
        ],
        "format": fmt,
        "destination": {"kind": "local", "path": out},
    }


def _run(job):
    return asyncio.run(export_async(job, fetch=zv.local_fetch))


def test_http_base_maps_gs_to_https():
    # A gs:// source (used so the render path can list) must be fetched over
    # HTTPS by the exporter -- pyfetch cannot read gs://.
    from ngpy.tract_export.browser import http_base

    assert http_base("gs://bucket/p/") == "https://storage.googleapis.com/bucket/p/"
    assert (
        http_base("https://storage.googleapis.com/bucket/p/")
        == "https://storage.googleapis.com/bucket/p/"
    )
    assert http_base("/local/dir/") == "/local/dir/"


class TestTrk:
    def test_selected_matches_the_native_selection(self, store, tmp_path):
        body, content_type, summary = _run(parse_job(_spec(store, scope="selected")))
        assert content_type == "application/octet-stream"
        assert summary["written"] is True
        # The sphere selects the two near tracts; the far one misses.
        assert summary["streamline_count"] == 2
        assert summary["object_count"] == 2
        assert summary["candidate_object_count"] == 3

        out = tmp_path / "rt.trk"
        out.write_bytes(body)
        trk = nib.streamlines.load(str(out))
        assert len(trk.streamlines) == 2
        assert sum(len(s) for s in trk.streamlines) == 5  # 3 + 2 vertices

    def test_whole_scope_exports_every_object(self, store, tmp_path):
        body, content_type, summary = _run(parse_job(_spec(store, scope="whole")))
        assert content_type == "application/octet-stream"
        assert summary["streamline_count"] == 3
        assert summary["object_count"] == 3
        out = tmp_path / "whole.trk"
        out.write_bytes(body)
        assert len(nib.streamlines.load(str(out)).streamlines) == 3

    def test_empty_selection_returns_json_and_writes_nothing(self, store):
        # A sphere far from every tract selects nothing.
        spec = _spec(store, scope="selected")
        spec["groups"][0]["rois"][0]["shape"]["center"] = [999, 999, 999]
        body, content_type, summary = _run(parse_job(spec))
        assert content_type == "application/json"
        assert summary["written"] is False
        assert summary["streamline_count"] == 0


class TestExplicitIds:
    """v3: export exactly the objects the viewer selected, via a selective read.

    The store's three tracts have object ids 0 and 1 (the near cluster) and 2
    (the far one).
    """

    def test_reads_only_the_requested_objects(self, store, tmp_path, monkeypatch):
        # Spy on the selective reader to prove the by-id path is taken -- not
        # a whole-level read that folds afterwards.
        seen = {}
        real = zv._read_objects_geometry

        def spy(root, **kwargs):
            seen["kwargs"] = kwargs
            return real(root, **kwargs)

        def no_whole_level(root, **kwargs):
            raise AssertionError("whole-level read on the by-id path")

        monkeypatch.setattr(zv, "_read_objects_geometry", spy)
        monkeypatch.setattr(zv, "_read_level_geometry", no_whole_level)

        body, content_type, summary = _run(parse_job(_id_spec(store, [0, 1])))
        assert content_type == "application/octet-stream"
        assert summary["written"] is True
        assert summary["streamline_count"] == 2
        assert summary["object_count"] == 2
        # candidate == the selection itself (no level was scanned).
        assert summary["candidate_object_count"] == 2
        assert seen["kwargs"].get("ids") == [0, 1]

        out = tmp_path / "ids.trk"
        out.write_bytes(body)
        trk = nib.streamlines.load(str(out))
        assert len(trk.streamlines) == 2
        assert sum(len(s) for s in trk.streamlines) == 5  # tract 0: 3, tract 1: 2

    def test_selects_the_far_tract_alone(self, store, tmp_path):
        body, _ct, summary = _run(parse_job(_id_spec(store, [2])))
        assert summary["streamline_count"] == 1
        out = tmp_path / "far.trk"
        out.write_bytes(body)
        assert len(nib.streamlines.load(str(out)).streamlines) == 1

    def test_empty_ids_returns_json_and_writes_nothing(self, store):
        body, content_type, summary = _run(parse_job(_id_spec(store, [])))
        assert content_type == "application/json"
        assert summary["written"] is False
        assert summary["streamline_count"] == 0

    def test_zvf_by_ids_round_trips(self, store, tmp_path):
        body, content_type, summary = _run(
            parse_job(_id_spec(store, [0, 1], fmt="zvf"))
        )
        assert content_type == "application/zip"
        assert summary["streamline_count"] == 2
        outdir = tmp_path / "ids.zvf"
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            zf.extractall(outdir)
        assert len(read_polylines(str(outdir), level=0)["polylines"]) == 2


class TestTrkHeader:
    """The TRK header must carry a real voxel size / dimensions / voxel->RAS so
    freeview can place the tract, and the millimetre affine must SCALE the stored
    voxel coordinates -- not leave nibabel's degenerate (1,1,1) default that made a
    whole brain read as a sub-millimetre speck and stopped freeview rendering."""

    def _spec_with_affine(self, store_path, object_ids, affine):
        s = _id_spec(store_path, object_ids)
        s["affine"] = [[float(x) for x in row] for row in affine]
        return s

    def test_header_is_non_degenerate_and_scales(self, store, tmp_path):
        from nibabel.streamlines import Field

        aff = np.diag([2.0, 2.0, 2.0, 1.0])  # 2 mm voxels
        body, _ct, _summary = _run(
            parse_job(self._spec_with_affine(store, [0, 1], aff))
        )
        out = tmp_path / "hdr.trk"
        out.write_bytes(body)
        trk = nib.streamlines.load(str(out))

        vs = tuple(float(x) for x in trk.header[Field.VOXEL_SIZES])
        dims = tuple(int(x) for x in trk.header[Field.DIMENSIONS])
        assert vs == (2.0, 2.0, 2.0)  # from the affine, not the (1,1,1) default
        assert dims != (1, 1, 1)  # non-degenerate: freeview has a volume to place
        assert all(d >= 1 for d in dims)

        # Object 0 is s0: voxels (10,10,10)->(14,10,10), a 4-voxel span along x.
        # A 2 mm affine must double it to an 8 mm RAS span, not shrink it.
        pts = np.asarray(trk.streamlines[0])
        assert pts.shape[1] == 3
        x_span = float(pts[:, 0].max() - pts[:, 0].min())
        assert x_span == pytest.approx(8.0)

    def test_identity_affine_still_gives_a_usable_header(self, store, tmp_path):
        from nibabel.streamlines import Field

        # No affine -> identity (1 mm voxels), but dimensions must still bound the
        # data rather than staying at the degenerate default.
        body, _ct, _summary = _run(parse_job(_id_spec(store, [2])))
        out = tmp_path / "id.trk"
        out.write_bytes(body)
        trk = nib.streamlines.load(str(out))
        assert tuple(float(x) for x in trk.header[Field.VOXEL_SIZES]) == (1, 1, 1)
        # s2 reaches ~(210,200,200); the reference grid must span it.
        dims = tuple(int(x) for x in trk.header[Field.DIMENSIONS])
        assert dims[0] > 200


class TestNoJspiReadPath:
    """The aio read must not call flush_prefetch's sync() (which needs JSPI).

    Simulate a JSPI-less runtime by making flush_prefetch raise the same error
    Pyodide throws, then confirm a full in-browser export still succeeds -- the
    offline_reads priming must serve the chunks without the sync prefetch.
    """

    def test_export_succeeds_without_the_sync_prefetch(self, store, monkeypatch):
        import zarr_vectors.core._batch_reader as batch_reader

        def no_jspi(*args, **kwargs):
            raise RuntimeError(
                "WebAssembly stack switching not supported in this JavaScript runtime"
            )

        monkeypatch.setattr(batch_reader, "flush_prefetch", no_jspi)

        body, content_type, summary = _run(parse_job(_spec(store, scope="whole")))
        assert content_type == "application/octet-stream"
        assert summary["streamline_count"] == 3


class TestZvf:
    def test_whole_store_round_trips(self, store, tmp_path):
        body, content_type, summary = _run(
            parse_job(_spec(store, fmt="zvf", scope="whole"))
        )
        assert content_type == "application/zip"
        assert summary["written"] is True
        assert summary["streamline_count"] == 3

        outdir = tmp_path / "rt.zvf"
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            zf.extractall(outdir)
        result = read_polylines(str(outdir), level=0)
        assert len(result["polylines"]) == 3
        # Every vertex survived the export: 3 + 2 + 2.
        total = sum(sum(len(f) for f in obj) for obj in result["polylines"])
        assert total == 7

    def test_selected_group_round_trips(self, store, tmp_path):
        body, content_type, summary = _run(
            parse_job(_spec(store, fmt="zvf", scope="selected"))
        )
        assert content_type == "application/zip"
        assert summary["streamline_count"] == 2
        outdir = tmp_path / "sel.zvf"
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            zf.extractall(outdir)
        assert len(read_polylines(str(outdir), level=0)["polylines"]) == 2
