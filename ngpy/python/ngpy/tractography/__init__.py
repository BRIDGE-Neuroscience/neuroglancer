# @license
# Copyright 2026 Allen Institute for Brain Science
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

"""Streamline (tractography) dissection geometry, evaluated in Python/WASM.

Ported from ``neuroglancer.tractography`` on the ``zarr_vectors_roi_store``
branch. There, the viewer's chunk worker uploaded resident geometry and asked
this package which streamlines pass; under ngpy the wrapper reads a whole
pyramid level itself (:mod:`ngpy.zv`), so the chunk-upload service and its wire
format are gone and only the geometry remains.

- ``roi``     does this polyline cross this region, and the include/or/exclude
              fold that composes several regions.
- ``index``   the flat, ragged form the tests evaluate against.
"""

from .index import TractIndex
from .roi import (
    Box,
    Ellipsoid,
    Halfspace,
    LabelMask,
    Roi,
    RoiOperator,
    RoiPredicate,
    RoiShape,
    combine_roi_verdicts,
    streamlines_pass_roi,
    streamlines_pass_rois,
)

__all__ = [
    "Box",
    "Ellipsoid",
    "Halfspace",
    "LabelMask",
    "Roi",
    "RoiOperator",
    "RoiPredicate",
    "RoiShape",
    "TractIndex",
    "combine_roi_verdicts",
    "streamlines_pass_roi",
    "streamlines_pass_rois",
]
