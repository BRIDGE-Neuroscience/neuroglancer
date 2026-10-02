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

``neuroglancer.Viewer`` resolves to :class:`ngpy.bridge.Viewer`, which drives
the Neuroglancer build hosted in the wrapper page's iframe over ``postMessage``
instead of serving a page from tornado.
"""

from ngpy.bridge import UnsynchronizedViewer, Viewer

__all__ = ["UnsynchronizedViewer", "Viewer"]
