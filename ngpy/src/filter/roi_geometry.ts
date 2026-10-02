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
 * @file Neuroglancer annotation JSON <-> ROI shapes in the store's frame.
 *
 * Annotation coordinates are in the annotation layer's dimensions (each a
 * `[scale, unit]` pair, e.g. `{"x": [0.001, "m"]}` = 1 mm per unit); the
 * zarr-vectors store's coordinates are in its own unit (e.g. mm).  Axes are
 * matched BY NAME (store axis `x` <- annotation dimension `x`), falling back
 * to position.  A layer `transform` other than this per-axis scale is not
 * modelled -- the same assumption the old in-viewer filter made (the tracts'
 * model space is the global x/y/z frame).
 */

export type RoiShapeJson =
  | { type: "box"; lower: number[]; upper: number[] }
  | { type: "ellipsoid"; center: number[]; radii: number[] }
  | { type: "halfspace"; origin: number[]; normal: number[] }
  | { type: "labelMask"; labels: number[] };

export interface DimensionScales {
  /** Annotation-layer dimension names, in coordinate order. */
  names: string[];
  /** Metres per annotation unit, per dimension. */
  scalesM: number[];
  /** Store axis names, e.g. ["x", "y", "z"]. */
  storeAxes: string[];
  /** Metres per store unit. */
  storeUnitM: number;
}

/** Parse a Neuroglancer `dimensions` JSON object (`{name: [scale, unit]}`). */
export function parseDimensions(dims: any): { names: string[]; scalesM: number[] } {
  const names: string[] = [];
  const scalesM: number[] = [];
  if (dims && typeof dims === "object") {
    for (const [name, value] of Object.entries(dims)) {
      const [scale, unit] = Array.isArray(value) ? value : [1, ""];
      names.push(name);
      scalesM.push(Number(scale) * unitFactor(String(unit ?? "")));
    }
  }
  return { names, scalesM };
}

function unitFactor(unit: string): number {
  switch (unit) {
    case "m":
    case "":
      return 1;
    case "mm":
      return 1e-3;
    case "um":
    case "µm":
      return 1e-6;
    case "nm":
      return 1e-9;
    default:
      return 1;
  }
}

/** For each store axis: [annotation dimension index, store units per annotation unit]. */
export function axisMapping(scales: DimensionScales): [number, number][] {
  const lower = scales.names.map((n) => n.toLowerCase());
  return scales.storeAxes.map((axis, i) => {
    let d = lower.indexOf(axis.toLowerCase());
    if (d < 0) d = i;
    const scale = scales.scalesM[d] ?? scales.storeUnitM;
    return [d, scale / scales.storeUnitM];
  });
}

/** An annotation's ROI shape in store coordinates, or undefined if not a region. */
export function annotationToShape(
  ann: any,
  scales: DimensionScales,
): RoiShapeJson | undefined {
  const map = axisMapping(scales);
  const pick = (v: number[]) => map.map(([d, f]) => Number(v?.[d] ?? 0) * f);
  switch (ann?.type) {
    case "axis_aligned_bounding_box": {
      const a = pick(ann.pointA);
      const b = pick(ann.pointB);
      return {
        type: "box",
        lower: a.map((x, i) => Math.min(x, b[i])),
        upper: a.map((x, i) => Math.max(x, b[i])),
      };
    }
    case "ellipsoid":
      return {
        type: "ellipsoid",
        center: pick(ann.center),
        radii: pick(ann.radii).map(Math.abs),
      };
    default:
      return undefined;
  }
}

/** Inverse of {@link annotationToShape}: annotation JSON for a stored shape. */
export function shapeToAnnotation(
  shape: RoiShapeJson,
  scales: DimensionScales,
  id: string,
): any | undefined {
  const map = axisMapping(scales);
  const rank = Math.max(scales.names.length, map.length);
  const place = (v: number[]) => {
    const out = new Array(rank).fill(0);
    map.forEach(([d, f], i) => {
      out[d] = Number(v[i]) / f;
    });
    return out;
  };
  switch (shape.type) {
    case "box":
      return {
        type: "axis_aligned_bounding_box",
        id,
        pointA: place(shape.lower),
        pointB: place(shape.upper),
      };
    case "ellipsoid":
      return { type: "ellipsoid", id, center: place(shape.center), radii: place(shape.radii) };
    default:
      return undefined;
  }
}

export function randomAnnotationId(): string {
  const bytes = new Uint8Array(20);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
