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

"""ngpy demo: the HCP-1065 whole-brain tractogram over an MNI parcellation.

Ported from ``python/examples/pyodide/user_script.py`` on the
``zarr_vectors_roi_store`` branch.  Run it from the Python tab ("Load demo"),
or open ``ngpy.html?script=examples/hcp1065_demo.py`` when ngpy.html is served
next to this examples/ directory.

It runs in ngpy's Pyodide worker: ``neuroglancer.Viewer()`` is a handle on the
Neuroglancer build hosted in the page (``ngpy.bridge``), not a tornado server.
Top-level ``await`` works; ``input()``, threads and ``webbrowser`` do not.

What changed from the old script: the parcellation link (``roi_label_layer``)
and the ROI overlay are no longer Neuroglancer layer options -- the dissection
runs in the wrapper -- so the script creates an ordinary local annotation layer
for ROIs and points the wrapper's Filter tab at the three layers with
``ngpy.gui.configure_filter``.  ``show_cross_section_outline_3d`` (a fork-only
option) is gone.
"""

import neuroglancer
import ngpy.gui

# A zarr-vectors source is `<kvstore-url>|zarr-vectors:`.  The `gs://` form reads
# the bucket anonymously (public `allUsers` read + list, and CORS).
TEST_TRACTOGRAM = (
    "gs://hip_ct_zarr_vector_03987646472fethdsvdvdfg/"
    "zarr_vectors_test/hcp1065_whole_brain.zarrvectors/|zarr-vectors:"
)

# MNI reference volumes, same bucket.  `mni_t1.zarr` / `mni_synthseg.zarr` are
# zarr v2 OME-NGFF multiscales, hence `|zarr2:`; the SynthSeg label names /
# colours come from a `segment_properties` source on the same layer.
MNI_BASE = (
    "gs://hip_ct_zarr_vector_03987646472fethdsvdvdfg/zarr_vectors_test/mni_images/"
)

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    # Pin the global coordinate space to the TRACTS' frame (x, y, z in mm).  The
    # MNI volumes declare z, y, x in micrometres; with the pin every layer maps
    # in by axis NAME and unit, so ROIs drawn in the global frame line up with
    # the tracts -- which is what the wrapper's Filter assumes.
    s.dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="mm", scales=[1, 1, 1]
    )
    s.layers["mni_t1"] = neuroglancer.ImageLayer(
        source=MNI_BASE + "mni_t1.zarr/|zarr2:",
    )
    s.layers["mni_synthseg"] = neuroglancer.SegmentationLayer(
        source=[
            MNI_BASE + "mni_synthseg.zarr/|zarr2:",
            "precomputed://" + MNI_BASE + "mni_synthseg_segment_properties/",
        ],
    )
    tracts = neuroglancer.SegmentationLayer(source=TEST_TRACTOGRAM)
    # Thin lines in the 2-d slices, no per-node dots; passing streamlines fully
    # opaque there (the slice default is 0.5).
    tracts.skeleton_rendering.mode2d = "lines"
    tracts.skeleton_rendering.line_width2d = 2
    tracts.selected_alpha = 1
    s.layers["tracts"] = tracts

    # Where the Filter tab's ROIs are drawn: an ordinary local annotation layer
    # whose `color` / `exclude` properties the wrapper sets per group.
    s.layers["ngpy ROIs"] = ngpy.gui.roi_layer(s.dimensions)
    s.selected_layer.layer = "ngpy ROIs"
    s.layout = "4panel"

    # Load sparse-first: the pyramid holds ~503k / 50k / 5k / 503 / 50 whole
    # streamlines at levels 0..4 and the GPU budget picks the finest that fits
    # (~20 MB -> level 3, ~50 MB -> level 2).
    s.gpu_memory_limit = 50_000_000

    # Let the tracts show through the 3-d cross-section planes, and match the
    # 2-d background to the 3-d one.
    s.hide_cross_section_background_3d = True
    s.cross_section_background_color = "#000000"

# Point the wrapper's Filter tab at the layers just created.
ngpy.gui.configure_filter(
    target="tracts", roi_layer="ngpy ROIs", parcellation="mni_synthseg"
)
ngpy.gui.select_tab("filter")

print("Viewer ready:", viewer.get_viewer_url())
