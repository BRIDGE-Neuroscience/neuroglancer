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

"""``ngpy.bridge`` and the vendored ``neuroglancer`` import in ANY order.

The worker imports ``ngpy.bridge`` before anything else; user scripts import
``neuroglancer`` first.  Each order runs in a fresh interpreter.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.dirname(HERE)


@pytest.mark.parametrize(
    "first",
    ["ngpy.bridge", "ngpy.api", "neuroglancer", "neuroglancer.viewer", "ngpy.gui"],
)
def test_import_order(first):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([PYTHON, os.path.join(PYTHON, "vendor")])
    code = (
        f"import {first}\n"
        "import neuroglancer, ngpy.bridge\n"
        "assert ngpy.bridge.Viewer is neuroglancer.Viewer\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
