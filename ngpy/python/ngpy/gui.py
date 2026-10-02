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

"""Drive the ngpy wrapper GUI (not the viewer) from a Python script.

    import ngpy.gui
    ngpy.gui.configure_filter(target="tracts", roi_layer="ngpy ROIs",
                              parcellation="mni_synthseg")
    ngpy.gui.select_tab("filter")

These are requests to the page (emitted as ``("gui", {...})``); they take
effect once the current script yields.
"""

from __future__ import annotations

from .bridge import emit


def configure_filter(
    target: str | None = None,
    roi_layer: str | None = None,
    parcellation: str | None = None,
    level: int | None = None,
) -> None:
    """Point the Filter tab at layers of the viewer (by name)."""
    settings: dict = {}
    if target is not None:
        settings["targetLayer"] = target
    if roi_layer is not None:
        settings["roiLayer"] = roi_layer
    if parcellation is not None:
        settings["parcellationLayer"] = parcellation
    if level is not None:
        settings["level"] = int(level)
    emit("gui", {"filter": settings})


def select_tab(name: str) -> None:
    """Show a tab: ``python``, ``filter``, ``export``, ``store`` or ``guide``."""
    emit("gui", {"tab": name})


#: The ROI-layer annotation properties ngpy's Filter tab colours ROIs with.
ROI_LAYER_PROPERTIES = [
    {"id": "color", "type": "rgb", "default": "#ffff00"},
    {
        "id": "exclude",
        "type": "uint8",
        "default": 0,
        "enum_values": [0, 1],
        "enum_labels": ["include", "exclude"],
    },
]

ROI_LAYER_SHADER = (
    "void main() {\n"
    "  vec3 c = prop_color();\n"
    "  setColor(prop_exclude() == 1u ? c * 0.45 : c);\n"
    "}\n"
)


def roi_layer(dimensions):
    """A ``LocalAnnotationLayer`` set up as an ngpy ROI layer."""
    import neuroglancer

    return neuroglancer.LocalAnnotationLayer(
        dimensions=dimensions,
        annotation_properties=[
            neuroglancer.AnnotationPropertySpec(**p) for p in ROI_LAYER_PROPERTIES
        ],
        shader=ROI_LAYER_SHADER,
    )
