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

"""ngpy: a standalone Pyodide GUI wrapper around any Neuroglancer build.

The Python half.  It runs in the wrapper page's Pyodide worker and provides:

- :mod:`ngpy.bridge`        ``neuroglancer.Viewer`` over postMessage;
- :mod:`ngpy.filter`        whole-store ROI / label / attribute dissection;
- :mod:`ngpy.zv`            zarr-vectors reads over HTTP (async, no JSPI);
- :mod:`ngpy.labels`        parcellation sampling;
- :mod:`ngpy.tract_export`  TRK / ZVF export to bytes;
- :mod:`ngpy.api`           the entry points the page calls.

Nothing here imports Neuroglancer's TypeScript or the viewer's internals; the
page applies results through the viewer's public state.
"""

__version__ = "0.1.0"


def viewer():
    """The ``neuroglancer.Viewer`` handle attached to the page (or None)."""
    from .bridge import active_viewer

    return active_viewer()
