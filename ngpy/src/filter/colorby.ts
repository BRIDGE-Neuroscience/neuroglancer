/**
 * @license
 * Copyright 2026 The Neuroglancer Authors
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file "Colour by" presets: skeleton shaders written to a layer's
 * `skeletonRendering.shader` / `shaderControls` (the old
 * `background_color_by_controls.ts`, minus the in-viewer UI).
 *
 * `segment` keeps Neuroglancer's segment colour -- which is where the Filter
 * tab's per-group colours (`segmentColors`) show.  The other presets colour
 * per vertex and therefore hide group colours.
 */

export interface ColorPreset {
  id: string;
  label: string;
  shader: string;
  controls: Record<string, unknown>;
}

export const SEGMENT_COLOR_SHADER = "void main() {\n  emitDefault();\n}\n";

export const DIRECTION_SHADER =
  "void main() {\n  emitRGB(abs(normalize(prop_tangent())));\n}\n";

/** A vertex attribute through a colour map (vec3 attributes map |xyz| to RGB). */
export function vertexAttributeShader(
  name: string,
  components: number,
  range: [number, number] = [0, 1],
): string {
  if (components === 3) {
    return `void main() {\n  emitRGB(abs(prop_${name}()));\n}\n`;
  }
  const [lo, hi] = range;
  const extent = Math.max(Math.abs(lo), Math.abs(hi), 1) * 10;
  return (
    `#uicontrol float lo slider(min=${-extent}, max=${extent}, default=${lo})\n` +
    `#uicontrol float hi slider(min=${-extent}, max=${extent}, default=${hi})\n` +
    "void main() {\n" +
    `  emitRGB(colormapJet((prop_${name}() - lo) / max(hi - lo, 1e-6)));\n` +
    "}\n"
  );
}

export function colorPresets(
  vertexAttributes: { name: string; components: number }[],
): ColorPreset[] {
  const out: ColorPreset[] = [
    {
      id: "segment",
      label: "Segment / group colour",
      shader: SEGMENT_COLOR_SHADER,
      controls: {},
    },
    {
      id: "direction",
      label: "Direction (tangent)",
      shader: DIRECTION_SHADER,
      controls: {},
    },
  ];
  for (const a of vertexAttributes) {
    if (a.name === "tangent") continue;
    out.push({
      id: `vertex:${a.name}`,
      label: `Vertex: ${a.name}`,
      shader: vertexAttributeShader(a.name, a.components),
      controls: {},
    });
  }
  return out;
}
