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

"""ngpy replacement for upstream ``neuroglancer.viewer`` (NOT vendored).

``neuroglancer.Viewer`` is a handle on the Neuroglancer build hosted in the
ngpy page, driven over ``postMessage`` by :mod:`ngpy.bridge` instead of being
served by tornado.  The classes are defined HERE (not in ``ngpy.bridge``) so
that importing either package first works without a circular import.
"""

from __future__ import annotations

import collections.abc
import contextlib

from ngpy.bridge import _BridgeMixin, emit

from . import viewer_base, viewer_state

__all__ = ["UnsynchronizedViewer", "Viewer"]


class Viewer(_BridgeMixin, viewer_base.ViewerBase):
    """``neuroglancer.Viewer`` under ngpy: a handle on the page's viewer.

    Creating one attaches it (replacing any earlier handle).  With
    ``adopt=True`` (default) it starts from the viewer's current state;
    ``adopt=False`` pushes Python's (empty) state instead, as upstream does.
    """

    def __init__(self, token: str = "ngpy", *, adopt: bool = True, **kwargs):
        viewer_base.ViewerBase.__init__(self, token=token, **kwargs)
        self._bridge_attach(adopt)

    def __repr__(self) -> str:
        return f"ngpy.Viewer(token={self.token!r})"


class UnsynchronizedViewer(_BridgeMixin, viewer_base.UnsynchronizedViewerBase):
    """One-way handle: Python's state is pushed, page edits are not read back."""

    def __init__(self, token: str = "ngpy", **kwargs):
        viewer_base.UnsynchronizedViewerBase.__init__(self, token=token, **kwargs)
        self._bridge_attach(adopt=False)
        self._push_unsynchronized()

    def _push_unsynchronized(self) -> None:
        if self._bridge_is_active():
            from .random_token import make_random_token

            emit("state", {"k": "s", "s": self.raw_state, "g": make_random_token()})

    def set_state(self, new_state) -> None:  # type: ignore[override]
        super().set_state(new_state)
        self._push_unsynchronized()

    @contextlib.contextmanager
    def txn(self) -> collections.abc.Iterator[viewer_state.ViewerState]:  # type: ignore[override]
        yield self.state
        self._push_unsynchronized()
