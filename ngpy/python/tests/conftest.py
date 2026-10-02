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

"""Test setup: import ``ngpy`` and the VENDORED upstream ``neuroglancer``.

The vendored package (``ngpy/python/vendor/neuroglancer``) is what ships in
ngpy.html; it must win over any other ``neuroglancer`` on the path (in
particular this repository's own fork of the Python package).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON = os.path.dirname(_HERE)
for path in (os.path.join(_PYTHON, "vendor"), _PYTHON):
    if path not in sys.path:
        sys.path.insert(0, path)

for name in list(sys.modules):
    if name == "neuroglancer" or name.startswith("neuroglancer."):
        del sys.modules[name]
