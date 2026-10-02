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

"""The wasm32 numpy shims for zarr-vectors (ngpy.compat).

The failure itself (int64 ``repeats`` on a 32-bit ``intp``) only reproduces
under Pyodide; here the shim is forced on and must not change any result.
"""

import numpy as np
import pytest
from ngpy import compat


def test_shim_narrows_repeat_and_bincount_inputs():
    repeats = np.array([2, 0, 3], dtype=np.int64)
    out = compat.SHIM.repeat(np.arange(3, dtype=np.int64), repeats)
    assert out.tolist() == [0, 0, 2, 2, 2]
    assert compat.SHIM.bincount(np.array([1, 1, 3], dtype=np.int64), minlength=5).tolist() == [0, 2, 0, 1, 0]
    assert compat.SHIM.float32 is np.float32  # everything else is numpy


def test_forced_shims_keep_store_reads_identical(tmp_path, monkeypatch):
    pytest.importorskip("zarr_vectors.encoding.fragments")
    import asyncio

    from ngpy import zv
    from test_zv_filter import TRACTS

    path = str(tmp_path / "s.zarrvectors")
    from zarr_vectors.types.polylines import write_polylines

    write_polylines(path, TRACTS, chunk_shape=(10.0, 10.0, 10.0), bounds=([0.0] * 3, [40.0] * 3))
    import importlib

    modules = {name: importlib.import_module(name) for name in compat.AFFECTED_MODULES}
    for module in modules.values():
        monkeypatch.setattr(module, "np", np)
    installed = compat.install_wasm32_numpy_shims(force=True)
    assert "zarr_vectors.encoding.fragments" in installed
    zv.forget_stores()
    s = zv.ZvStore(zv.http_base(path), zv.local_fetch, None)
    index, _ = asyncio.run(s.level_geometry(0))
    assert list(index.object_ids) == [0, 1, 2, 3]
    np.testing.assert_allclose(index.positions[: index.counts[0]], TRACTS[0])
