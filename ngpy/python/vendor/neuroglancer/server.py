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

"""ngpy replacement for upstream ``neuroglancer.server`` (NOT vendored).

Upstream's module starts a tornado HTTP server that the viewer page connects
back to.  Inside ngpy there is no server: the Python runtime lives in a Pyodide
worker of the wrapper page and talks to the viewer over ``postMessage`` (see
``ngpy.bridge``).  This stub keeps the public functions scripts habitually call
(``neuroglancer.set_server_bind_address(...)`` etc.) importable and harmless, so
the vendored upstream ``__init__`` imports unchanged without tornado.
"""

from __future__ import annotations

__all__ = [
    "is_server_running",
    "set_dev_server_content_source",
    "set_server_bind_address",
    "set_static_content_source",
    "stop",
]


def set_server_bind_address(bind_address=None, bind_port=0):  # noqa: ARG001
    """No-op: ngpy has no HTTP server; the viewer is the wrapper's iframe."""


def set_static_content_source(*args, **kwargs):  # noqa: ARG001
    """No-op: the Neuroglancer build is chosen with the wrapper's ``?ng=`` URL."""


def set_dev_server_content_source():
    """No-op (see :func:`set_static_content_source`)."""


def is_server_running() -> bool:
    """Always True: the bridge is up for as long as the wrapper page is."""
    return True


def stop() -> None:
    """No-op: closing the wrapper page is what stops ngpy."""
