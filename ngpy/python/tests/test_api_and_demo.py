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

"""The page-facing entry points (ngpy.api) and the shipped demo script."""

import asyncio
import json
import os
import runpy

import pytest

from ngpy import api, bridge, zv

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "..", "examples", "hcp1065_demo.py")


@pytest.fixture(autouse=True)
def _fresh():
    bridge.reset()
    bridge.set_emitter(None)
    zv.forget_stores()
    yield
    bridge.reset()


def test_demo_script_builds_the_scene_and_configures_the_gui():
    runpy.run_path(DEMO, run_name="__main__")
    messages = [(k, json.loads(t)) for k, t in bridge.outbox]
    state = [m for k, m in messages if k == "state" and m["k"] == "s"][-1]["s"]
    names = [layer["name"] for layer in state["layers"]]
    assert names == ["mni_t1", "mni_synthseg", "tracts", "ngpy ROIs"]
    tracts = state["layers"][2]
    sources = (
        tracts["source"] if isinstance(tracts["source"], list) else [tracts["source"]]
    )
    urls = [x if isinstance(x, str) else x["url"] for x in sources]
    assert urls[0].endswith("|zarr-vectors:")
    assert tracts["skeletonRendering"]["mode2d"] == "lines"
    rois = state["layers"][3]
    assert rois["type"] == "annotation"
    assert [p["id"] for p in rois["annotationProperties"]] == ["color", "exclude"]
    assert state["dimensions"] == {
        "x": [0.001, "m"],
        "y": [0.001, "m"],
        "z": [0.001, "m"],
    }
    gui = [m for k, m in messages if k == "gui"]
    assert gui[0] == {
        "filter": {
            "targetLayer": "tracts",
            "roiLayer": "ngpy ROIs",
            "parcellationLayer": "mni_synthseg",
        }
    }
    assert gui[1] == {"tab": "filter"}


def test_label_info_classifies_a_layers_sources(tmp_path, monkeypatch):
    from test_zv_filter import _write_ome_labels

    vol = _write_ome_labels(tmp_path)
    props = tmp_path / "props"
    props.mkdir()
    (props / "info").write_text(
        json.dumps(
            {
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": ["7", "9", "11"],
                    "properties": [
                        {
                            "id": "label",
                            "type": "label",
                            "values": ["seven", "nine", "absent"],
                        }
                    ],
                },
            }
        )
    )
    monkeypatch.setattr(zv, "default_fetch", lambda: zv.local_fetch)
    from ngpy import labels

    monkeypatch.setattr(labels, "default_fetch", lambda: zv.local_fetch)
    from ngpy import filter as filter_mod

    monkeypatch.setattr(
        filter_mod, "_engine", filter_mod.FilterEngine(fetch=zv.local_fetch)
    )
    out = json.loads(
        asyncio.run(
            api.label_info(
                json.dumps({"sources": [f"{vol}/|zarr2:", f"precomputed://{props}"]})
            )
        )
    )
    assert out["volumeUrl"] == f"{vol}/|zarr2:"
    # Label 11 is not in the volume, so it is not offered.
    assert out["labels"] == [
        {"id": 7, "name": "seven", "color": None},
        {"id": 9, "name": "nine", "color": None},
    ]


def test_export_entry_returns_a_clear_error_for_a_bad_spec():
    content_type, body, summary = asyncio.run(api.export(json.dumps({"format": "trk"})))
    assert content_type == "text/plain"
    assert "source" in json.loads(summary)["error"]
