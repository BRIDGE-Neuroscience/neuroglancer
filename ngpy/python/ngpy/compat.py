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

"""wasm32 workarounds for zarr-vectors-py under Pyodide.

Pyodide is wasm32: ``np.intp`` is 32-bit, and numpy refuses to cast an int64
``repeats`` (``np.repeat``) or input (``np.bincount``) array to it --
``TypeError: Cannot cast array data from dtype('int64') to dtype('int32')
according to the rule 'safe'``.  zarr-vectors 0.9.2 (and its current dev
branch) passes int64 there, e.g. ``encoding/fragments.py``
``decode_object_manifests_many``: ``np.repeat(np.arange(n, dtype=np.int64),
expected)`` with int64 ``expected`` -- which every whole-level read reaches
through ``read_object_manifest_rows``.  It cannot show up on a 64-bit test
machine, where ``intp`` is int64.

Rather than patch numpy globally, the ``np`` name of just the affected
zarr-vectors modules is pointed at a thin proxy whose ``repeat`` / ``bincount``
narrow integer arrays to ``intp`` first.  Installed only where ``intp`` is
narrower than 64 bits (or when forced, for tests).  The real fix belongs in
zarr-vectors-py (cast to ``np.intp``).
"""

from __future__ import annotations

import importlib
import types

import numpy as np

#: zarr-vectors modules that call np.repeat / np.bincount with int64 arrays.
AFFECTED_MODULES = (
    "zarr_vectors.encoding.fragments",
    "zarr_vectors.core.arrays",
    "zarr_vectors.spatial.boundary",
    "zarr_vectors.types.lines",
    "zarr_vectors.multiresolution.coarsen",
)


def _to_intp(x):
    if isinstance(x, np.ndarray) and x.dtype.kind in "iu" and x.dtype != np.intp:
        return x.astype(np.intp)
    return x


class _NumpyIntpShim(types.ModuleType):
    """``numpy`` with intp-narrowing ``repeat`` and ``bincount``."""

    def __init__(self):
        super().__init__("numpy")

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def repeat(a, repeats, axis=None):
        return np.repeat(a, _to_intp(repeats), axis=axis)

    @staticmethod
    def bincount(x, weights=None, minlength=0):
        return np.bincount(_to_intp(x), weights=weights, minlength=minlength)


SHIM = _NumpyIntpShim()


def needs_wasm32_shims() -> bool:
    return np.dtype(np.intp).itemsize < 8


def install_wasm32_numpy_shims(force: bool = False) -> list[str]:
    """Point the affected modules' ``np`` at :data:`SHIM`; returns their names."""
    if not (force or needs_wasm32_shims()):
        return []
    installed = []
    for name in AFFECTED_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - module absent in this version
            continue
        if getattr(module, "np", None) is np:
            module.np = SHIM
            installed.append(name)
    return installed
