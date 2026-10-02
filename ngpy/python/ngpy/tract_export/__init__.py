# @license
# Copyright 2026 Google Inc.
#
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
"""Export a dissection as a TrackVis ``.trk`` or a new zarr-vectors store.

Ported from ``neuroglancer.tract_export`` (branch ``zarr_vectors_roi_store``);
in ngpy the whole export runs in the page's Pyodide worker and returns bytes.
"""

from ngpy.tract_export.job import (
    JOB_SCHEMA_VERSION,
    Destination,
    ExportGroup,
    ExportJob,
    JobSpecError,
    parse_job,
    parse_roi,
    roi_bounds,
)

__all__ = [
    "JOB_SCHEMA_VERSION",
    "Destination",
    "ExportGroup",
    "ExportJob",
    "JobSpecError",
    "parse_job",
    "parse_roi",
    "roi_bounds",
]
