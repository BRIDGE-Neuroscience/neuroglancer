/**
 * @license
 * Copyright 2026 Google Inc.
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

/** @file How each zarr-vectors geometry type is drawn. */

export type ZarrVectorsGeometryKind =
  | "point_cloud"
  | "line"
  | "streamline"
  | "polyline"
  | "skeleton"
  | "graph"
  | "mesh";

export interface GeometryKindCapabilities {
  /** What the dense layer draws: points, edges, or (per object) faces. */
  readonly primitive: "points" | "lines" | "triangles";
  /**
   * Where `prop_tangent()` comes from: the walk order of a curve, the edges of
   * a graph or tree, or nowhere.
   */
  readonly tangent: "walk" | "edges" | undefined;
  /**
   * Colour by direction by default. Only for directional curves; a tree's
   * tangent sign is arbitrary and its users colour by object.
   */
  readonly directionDefault: boolean;
}

export const KIND_CAPABILITIES: Record<
  ZarrVectorsGeometryKind,
  GeometryKindCapabilities
> = {
  point_cloud: {
    primitive: "points",
    tangent: undefined,
    directionDefault: false,
  },
  line: { primitive: "lines", tangent: "walk", directionDefault: true },
  streamline: { primitive: "lines", tangent: "walk", directionDefault: true },
  polyline: { primitive: "lines", tangent: "walk", directionDefault: true },
  skeleton: { primitive: "lines", tangent: "edges", directionDefault: false },
  graph: { primitive: "lines", tangent: "edges", directionDefault: true },
  mesh: { primitive: "triangles", tangent: undefined, directionDefault: false },
};

/**
 * Colour by direction: |unit tangent| as RGB, the tractography convention.
 * Swizzle-free, so a store attribute named `z` cannot break it.
 */
export const DIRECTION_SHADER = `void main() {
  emitRGB(abs(prop_tangent()));
}
`;

/**
 * The kind a store's arrays hold, from its declared `geometry_types`.
 *
 * A failed `zarr_vectors.composite.add_geometry()` call records a second type
 * before writing anything, so a store may declare more than it holds; prefer
 * the declared kind its links are consistent with.
 */
export function resolveGeometryKind(
  declared: readonly string[],
  links: { present: boolean; width: number | undefined },
): { kind: ZarrVectorsGeometryKind; ignored: string[] } {
  const known = declared.filter(
    (g): g is ZarrVectorsGeometryKind => g in KIND_CAPABILITIES,
  );
  if (known.length === 0) {
    throw new Error(
      `no recognised geometry type in ${JSON.stringify(declared)}; expected ` +
        `one of ${Object.keys(KIND_CAPABILITIES).join(", ")}`,
    );
  }
  const consistent = known.filter((kind) => {
    const { primitive } = KIND_CAPABILITIES[kind];
    if (
      links.width !== undefined &&
      (primitive === "triangles") !== links.width >= 3
    ) {
      return false;
    }
    if (primitive === "points" && links.present) return false;
    if ((kind === "graph" || kind === "mesh") && !links.present) return false;
    return true;
  });
  const kind = consistent[0] ?? known[0];
  return { kind, ignored: declared.filter((g) => g !== kind) };
}
