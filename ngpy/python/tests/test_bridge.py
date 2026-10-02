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

"""The Python half of the page <-> Python state protocol (ngpy.bridge)."""

import json

import neuroglancer
import pytest

from ngpy import api, bridge


@pytest.fixture(autouse=True)
def _fresh_bridge():
    bridge.reset()
    bridge.set_emitter(None)
    yield
    bridge.reset()


def _messages(kind=None):
    out = [(k, json.loads(t)) for k, t in bridge.outbox]
    return [m for k, m in out if kind is None or k == kind]


def _client(state, g, pg, c="client1"):
    return json.loads(
        bridge.handle_client_state(json.dumps({"s": state, "g": g, "pg": pg, "c": c}))
    )


LAYER_STATE = {
    "layers": [{"type": "segmentation", "name": "tracts", "source": "precomputed://x"}],
    "layout": "xy",
}


def test_neuroglancer_viewer_is_the_bridge_viewer():
    assert neuroglancer.Viewer is bridge.Viewer
    assert neuroglancer.UnsynchronizedViewer is bridge.UnsynchronizedViewer
    # The server stub keeps habitual calls harmless.
    neuroglancer.set_server_bind_address("127.0.0.1")
    assert neuroglancer.is_server_running()


def test_client_state_is_cached_before_any_viewer_exists():
    reply = _client(LAYER_STATE, 3, "")
    assert reply == {"status": 200, "g": "client1/3"}
    assert bridge.client_state() == LAYER_STATE


def test_new_viewer_adopts_the_page_state_without_echoing_it():
    _client(LAYER_STATE, 3, "")
    viewer = neuroglancer.Viewer()
    assert [layer.name for layer in viewer.state.layers] == ["tracts"]
    # Adoption is page-originated (generation has "/"): no "s" state push...
    assert not [m for m in _messages("state") if m["k"] == "s"]
    # ...but the page is told the generation to quote as `pg`.
    assert _messages("generation") == [{"k": "s", "g": "client1/3"}]
    # The config state is always pushed (Python -> page only).
    assert [m["k"] for m in _messages("state")] == ["c"]


def test_python_edits_are_pushed_with_a_python_generation():
    _client(LAYER_STATE, 1, "")
    viewer = neuroglancer.Viewer()
    bridge.outbox.clear()
    with viewer.txn() as s:
        s.layers["img"] = neuroglancer.ImageLayer(source="precomputed://img")
    pushes = [m for m in _messages("state") if m["k"] == "s"]
    assert len(pushes) == 1
    assert "/" not in pushes[0]["g"]
    assert [layer["name"] for layer in pushes[0]["s"]["layers"]] == ["tracts", "img"]


def test_page_updates_apply_with_generation_check_and_never_echo():
    _client(LAYER_STATE, 1, "")
    viewer = neuroglancer.Viewer()
    bridge.outbox.clear()
    new_state = dict(LAYER_STATE, layout="3d")
    reply = _client(new_state, 2, "client1/1")
    assert reply == {"status": 200, "g": "client1/2"}
    assert viewer.state.layout.type == "3d"
    assert not [m for m in _messages("state") if m["k"] == "s"]


def test_a_stale_page_update_is_refused_with_412():
    _client(LAYER_STATE, 1, "")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:  # Python moves the generation on
        s.layout = "4panel"
    reply = _client(dict(LAYER_STATE, layout="3d"), 2, "client1/1")
    assert reply == {"status": 412}
    assert viewer.state.layout.type == "4panel"


def test_adopt_false_pushes_pythons_state_to_the_page():
    _client(LAYER_STATE, 1, "")
    neuroglancer.Viewer(adopt=False)
    pushes = [m for m in _messages("state") if m["k"] == "s"]
    assert pushes and pushes[0]["s"] == {}


def test_actions_round_trip_through_config_and_handle_action():
    viewer = neuroglancer.Viewer()
    seen = []
    viewer.actions.add("my-action", lambda s: seen.append(s))
    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer["keyt"] = "my-action"
    configs = [m["s"] for m in _messages("state") if m["k"] == "c"]
    assert "my-action" in configs[-1]["actions"]
    assert configs[-1]["inputEventBindings"]["viewer"] == {"keyt": "my-action"}
    api.action(
        json.dumps(
            {
                "action": "my-action",
                "state": {"mousePosition": [1, 2, 3], "selectedValues": {}},
            }
        )
    )
    assert len(seen) == 1
    assert list(seen[0].mouse_voxel_coordinates) == [1, 2, 3]


def test_a_second_viewer_replaces_the_first():
    first = neuroglancer.Viewer()
    second = neuroglancer.Viewer()
    assert bridge.active_viewer() is second
    bridge.outbox.clear()
    with first.txn() as s:
        s.layout = "3d"
    assert not _messages("state")  # a detached handle no longer talks


def test_unsynchronized_viewer_pushes_one_way():
    viewer = neuroglancer.UnsynchronizedViewer()
    bridge.outbox.clear()
    with viewer.txn() as s:
        s.layout = "xz"
    pushes = [m for m in _messages("state") if m["k"] == "s"]
    assert pushes[-1]["s"]["layout"] == "xz"


def test_unsupported_features_fail_loudly():
    viewer = neuroglancer.Viewer()
    with pytest.raises(NotImplementedError):
        viewer.screenshot()
    with pytest.raises(NotImplementedError):
        viewer.volume("x")


def test_the_viewers_new_layer_placeholder_is_not_passed_to_python():
    # A Neuroglancer page with no state opens with this placeholder layer,
    # which upstream's make_layer rejects (ValueError on every txn).
    _client(
        {
            "layers": [
                {"type": "new", "source": "", "tab": "source", "name": "new layer"}
            ],
            "selectedLayer": {"visible": True, "layer": "new layer"},
            "layout": "4panel-alt",
        },
        1,
        "",
    )
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["img"] = neuroglancer.ImageLayer(source="precomputed://img")
    assert [layer.name for layer in viewer.state.layers] == ["img"]
    assert viewer.state.selected_layer.layer is None


def test_unknown_state_keys_survive_the_round_trip():
    # A fork-specific key (e.g. a zarr-vectors layer option) must not be lost
    # when Python edits an unrelated part of the state.
    state = {
        "layers": [
            {
                "type": "segmentation",
                "name": "t",
                "source": "x",
                "forkOnlyOption": {"a": 1},
            }
        ],
        "forkTopLevel": True,
    }
    _client(state, 1, "")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layout = "3d"
    pushed = [m for m in _messages("state") if m["k"] == "s"][-1]["s"]
    assert pushed["forkTopLevel"] is True
    assert pushed["layers"][0]["forkOnlyOption"] == {"a": 1}


def test_boot_imports_the_vendored_package():
    info = json.loads(api.boot(json.dumps({"pageUrl": "http://x/ngpy.html"})))
    assert info["zarr_vectors"]
    assert neuroglancer.Viewer().get_viewer_url() == "http://x/ngpy.html"
