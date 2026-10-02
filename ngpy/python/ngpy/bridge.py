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

"""The Python side of the ngpy page <-> Pyodide worker state bridge.

Replaces the old service-worker SSE emulation (``browser_server.py`` +
``viewer_pyodide.py`` + ``pyodide_service_worker.ts``) with plain messages.
The page owns the one hosted Neuroglancer viewer; this module makes
``neuroglancer.Viewer()`` a handle on it, speaking the same protocol upstream's
tornado server and ``src/python_integration/api.ts`` speak, minus HTTP:

* page -> Python, shared state ``"s"``:
  :func:`handle_client_state` receives ``{s, g, pg, c}`` (state, client
  generation, previous server generation, client id), applies it with
  ``set_state(s, "<c>/<g>", existing_generation=pg)`` and answers
  ``{"status": 200, "g": <new generation>}`` or ``{"status": 412}`` on a
  concurrent Python edit -- exactly upstream's ``_handle_set_state``.
* Python -> page: every change of ``shared_state`` (``k = "s"``) or
  ``config_state`` (``k = "c"``) is emitted as ``("state", {k, s, g})``,
  EXCEPT shared-state changes whose generation contains ``"/"``: those
  originated from the page, and echoing them back would interfere with an
  in-progress gesture (the no-echo rule of the old ``viewer_pyodide``).
* page -> Python, actions: :func:`handle_action` (``{action, state}``) invokes
  ``viewer.actions`` handlers with an ``ActionState``.

The page keeps sending the viewer state even before a script creates a
``Viewer``; it is cached here, and a new ``Viewer`` ADOPTS it (``adopt=True``,
the default) so a script extends what is on screen.  ``adopt=False`` gives
upstream's behaviour instead: the new Python state (empty) replaces the page's.

``emit(kind, json_text)`` is installed by the worker (``set_emitter``); without
one (CPython tests) messages accumulate in :data:`outbox`.

Out of scope (they need a service worker or a blocking wait the worker cannot
do): ``python://`` LocalVolume / SkeletonSource data sources, ``screenshot()``
and ``volume()`` / ``volume_info()`` -- those raise ``NotImplementedError``.
"""

from __future__ import annotations

import json
import typing

# NOTE: no top-level `neuroglancer` import.  The vendored package's
# `neuroglancer/viewer.py` imports THIS module to define `Viewer`, so this one
# must be importable first (the worker imports it before anything else).

Emitter = typing.Callable[[str, str], None]

_emitter: Emitter | None = None
#: Messages emitted while no emitter is installed (tests read these).
outbox: list[tuple[str, str]] = []

#: Filled in by the worker at boot: the wrapper page URL and the viewer URL.
page_info: dict[str, str] = {"pageUrl": "", "viewerUrl": ""}


def set_emitter(fn: Emitter | None) -> None:
    global _emitter
    _emitter = fn


def emit(kind: str, payload: typing.Any) -> None:
    from neuroglancer.json_utils import encode_json

    text = payload if isinstance(payload, str) else encode_json(payload)
    if _emitter is None:
        outbox.append((kind, text))
    else:
        _emitter(kind, text)


class _ClientState:
    __slots__ = ("state", "generation")

    def __init__(self) -> None:
        self.state: typing.Any = None
        self.generation: str = ""


_client = _ClientState()
_active: _BridgeMixin | None = None


def __getattr__(name: str):
    # `ngpy.bridge.Viewer` -- the classes live in the vendored package's
    # `neuroglancer/viewer.py` (see the note on imports above).
    if name in ("Viewer", "UnsynchronizedViewer"):
        from neuroglancer import viewer

        return getattr(viewer, name)
    raise AttributeError(name)


def active_viewer() -> typing.Any:
    """The viewer handle currently attached to the page, if any."""
    return _active  # type: ignore[return-value]


def client_state() -> typing.Any:
    """The last viewer state the page reported (JSON), or None."""
    return _client.state


def reset() -> None:
    """Forget the attached viewer and cached state (tests)."""
    global _active
    _active = None
    _client.state = None
    _client.generation = ""
    outbox.clear()


def sanitize_client_state(state: typing.Any) -> typing.Any:
    """Drop the viewer's transient ``{"type": "new"}`` placeholder layers.

    A Neuroglancer page with no state opens an "add layer" dialog backed by a
    layer of type ``new`` (empty source).  Upstream's Python ``make_layer``
    raises on that type, which would break every ``txn()``; it is a UI
    placeholder, not data, so it is not passed to Python.
    """
    if not isinstance(state, dict):
        return state
    layers = state.get("layers")
    if isinstance(layers, list) and any(
        isinstance(layer, dict) and layer.get("type") == "new" for layer in layers
    ):

        def is_placeholder(layer) -> bool:
            return isinstance(layer, dict) and layer.get("type") == "new"

        kept = [layer for layer in layers if not is_placeholder(layer)]
        dropped = {layer.get("name") for layer in layers if is_placeholder(layer)}
        state = dict(state, layers=kept)
        selected = state.get("selectedLayer")
        if isinstance(selected, dict) and selected.get("layer") in dropped:
            state["selectedLayer"] = {k: v for k, v in selected.items() if k != "layer"}
    return state


def handle_client_state(message: str) -> str:
    """Apply one state update from the page.  Returns the JSON reply."""
    from neuroglancer.trackable_state import ConcurrentModificationError

    msg = json.loads(message)
    state = sanitize_client_state(msg["s"])
    generation = f"{msg['c']}/{msg['g']}"
    previous = msg.get("pg") or None
    viewer = _active
    if viewer is None:
        _client.state = state
        _client.generation = generation
        return json.dumps({"status": 200, "g": generation})
    try:
        new_generation = viewer.set_state(  # type: ignore[attr-defined]
            state, generation, existing_generation=previous
        )
    except ConcurrentModificationError:
        return json.dumps({"status": 412})
    _client.state = state
    _client.generation = new_generation
    return json.dumps({"status": 200, "g": new_generation})


def handle_action(message: str) -> None:
    """Invoke the attached viewer's handlers for one page action."""
    msg = json.loads(message)
    viewer = _active
    if viewer is None:
        return
    viewer.actions.invoke(msg["action"], msg.get("state") or {})  # type: ignore[attr-defined]


class _BridgeMixin:
    """Wires a ``ViewerCommonBase`` subclass to the page.

    ``ViewerCommonBase.__init__`` does not chain to ``super().__init__``, so
    subclasses call :meth:`_bridge_attach` explicitly after their base init.
    """

    if typing.TYPE_CHECKING:
        token: str
        config_state: typing.Any
        shared_state: typing.Any

    def _bridge_attach(self, adopt: bool) -> None:
        global _active
        _active = self
        self.config_state.add_changed_callback(self._bridge_push_config)
        shared = getattr(self, "shared_state", None)
        if shared is not None:
            shared.add_changed_callback(self._bridge_push_shared)
            if adopt and _client.state is not None:
                shared.set_state(_client.state, _client.generation or None)
            else:
                # Python's state wins (upstream semantics): push it.
                self._bridge_push_shared(force=True)
            # Tell the page which generation it must quote as `pg`.
            emit("generation", {"k": "s", "g": shared.state_generation})
        self._bridge_push_config()

    def _bridge_is_active(self) -> bool:
        return _active is self

    def _bridge_push_shared(self, force: bool = False) -> None:
        if not self._bridge_is_active():
            return
        raw_state, generation = self.shared_state.raw_state_and_generation
        # Page-originated updates carry "<client>/<gen>"; never echo those.
        if "/" in generation and not force:
            return
        emit("state", {"k": "s", "s": raw_state, "g": generation})

    def _bridge_push_config(self) -> None:
        if not self._bridge_is_active():
            return
        raw_state, generation = self.config_state.raw_state_and_generation
        emit("state", {"k": "c", "s": raw_state, "g": generation})

    def defer_callback(self, callback, *args, **kwargs) -> None:
        """Run ``callback`` on a later turn of the worker's event loop."""
        try:
            import js  # type: ignore[import-not-found]
            from pyodide.ffi import (
                create_once_callable,  # type: ignore[import-not-found]
            )
        except ImportError:
            callback(*args, **kwargs)
            return
        js.setTimeout(create_once_callable(lambda: callback(*args, **kwargs)), 0)

    def get_viewer_url(self) -> str:
        return page_info.get("pageUrl", "")

    def _repr_html_(self) -> str:
        return f"<b>ngpy viewer</b> (token: {self.token})"

    # -- unsupported upstream features -----------------------------------
    def screenshot(self, *args, **kwargs):  # noqa: ARG002
        raise NotImplementedError(
            "ngpy: screenshot() needs a blocking wait the Pyodide worker cannot "
            "do; use the viewer's own screenshot button."
        )

    def async_screenshot(self, *args, **kwargs):  # noqa: ARG002
        raise NotImplementedError("ngpy: screenshots are not supported")

    def volume_info(self, *args, **kwargs):  # noqa: ARG002
        raise NotImplementedError("ngpy: volume requests are not supported")

    def volume(self, *args, **kwargs):  # noqa: ARG002
        raise NotImplementedError("ngpy: volume requests are not supported")
