/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Canonical zarr-vectors geometry-kind union and per-kind capability
 * table.  Centralises the decisions that were previously scattered
 * through `geometry_chunk.ts`, `geometry_shader_bridge.ts`, and
 * `geometry_frontend.ts` as ad-hoc `geometryKind === "streamline" ||
 * "polyline"` checks.  Adding a new kind (e.g. a future mesh
 * geometry that fits the same chunk layout) is a one-line edit to
 * the {@link KIND_CAPABILITIES} table.
 */

/**
 * Colour by direction: |unit tangent| as RGB, the tractography convention.
 * Swizzle-free on purpose, so a store attribute named `z` cannot break it.
 */
export const DEFAULT_STREAMLINE_FRAGMENT_MAIN = `void main() {
  emitRGB(abs(prop_tangent()));
}
`;

/**
 * The geometry-type strings zarr-vectors emits in its
 * `zarr_vectors.geometry_types` root attribute.  Every kind listed here routes
 * through the same spatially-indexed geometry render path — the on-disk layout
 * (`vertices/` + `vertex_fragments/` + optional `links/`) is shared, and the
 * differences are exactly what {@link KIND_CAPABILITIES} records. */
export type ZarrVectorsGeometryKind =
  | "point_cloud"
  | "line"
  | "streamline"
  | "polyline"
  | "skeleton"
  | "graph"
  | "mesh";

/** Per-kind metadata consumed by the chunk decoder, shader bridge, and
 *  frontend chunk-source classes. */
export interface GeometryKindCapabilities {
  /**
   * Which GPU primitive the render layer draws for this kind.
   *
   * - `"points"`: the store has no connectivity at all, so the layer draws one
   *   circle per vertex and skips the edge pass entirely (see
   *   `SkeletonRenderMode.POINTS`).
   * - `"lines"`: link records are vertex PAIRS, drawn as line segments.
   * - `"triangles"`: link records are faces. Their arity is not fixed here --
   *   the store declares it as `link_width` on the links family (3 for
   *   triangles, 4 for quads) and the reader honours that; what this field says
   *   is only that the records bound a surface rather than a curve.
   */
  readonly primitive: "points" | "lines" | "triangles";
  /**
   * Where a chunk's edges come from.  This — not the store's
   * `links_convention` — decides what `buildGeometryChunk` synthesises:
   *
   * - `"none"`: the kind has no connectivity.  Emitting implicit-sequential
   *   edges here would be actively wrong, not merely redundant: a point-cloud
   *   fragment is a spatial *bin* holding many unrelated points, so vertex
   *   `i → i+1` would wire each bin into a spaghetti polyline.
   * - `"implicit_sequential"`: edges from fragment ranges only.
   * - `"explicit"`: edges come from `links/0/<chunk>`.
   * - `"both"`: fragment ranges plus explicit branch edges.
   */
  readonly edgeSource: "none" | "implicit_sequential" | "explicit" | "both";
  /**
   * Whether the store carries the discrete-object model: `object_index/`,
   * `object_attributes/`, and per-fragment `segment_id`.  False for
   * `point_cloud`, the one ZVF kind the spec defines without objects — the
   * datasource then emits no per-segment (pass-2) subsource and no
   * segment-property map, and the chunk decoder synthesises one segment id per
   * *vertex* rather than one per fragment.
   */
  readonly hasObjectModel: boolean;
  /**
   * Whether the chunk decoder should synthesise a per-vertex
   * `tangent` (vec3) by walking the fragment index in implicit-
   * sequential order.  True only for `streamline` and `polyline`
   * geometries — those have a well-defined walk direction at every
   * vertex, including endpoints, which we need so that cross-chunk
   * ghost-tangent signs match up across bridge edges (see
   * `chunk_pipeline.ts`).
   */
  readonly hasWalkOrderTangent: boolean;
  /**
   * Whether the chunk decoder should synthesise per-vertex tangents
   * from the edge adjacency (degree-2 vertices get the central
   * difference; degree-1 endpoints get the direction to their lone
   * neighbour; branch points pick the central difference of the
   * first two listed neighbours).  True for `graph`: the on-disk
   * `links/0/<chunk>` array carries the full edge structure and
   * walk-order is undefined, but most non-branch vertices still
   * have a meaningful direction.  Skeletons could opt in similarly
   * but currently don't, preserving prior "no tangents for
   * skeletons" behaviour until a use case shows up.
   */
  readonly hasEdgeAdjacencyTangent: boolean;
  /**
   * Renderer's default shader text (paste-in for users); `undefined`
   * falls back to the neuroglancer-built-in segment-coloured default.
   * Every tangent-bearing kind auto-applies the RGB-by-tangent default --
   * matching the segmentation layer's `roiFilter.backgroundColorBy` default of
   * "direction", so a freshly loaded tract layer renders colour-by-direction on
   * the first frame rather than the generic per-object hash colour.  A kind
   * without a tangent (`point_cloud`) nominates `undefined` and keeps the hash
   * default -- `prop_tangent()` does not exist for it, so the direction shader
   * would not even compile.
   *
   * The nomination still passes through {@link resolveSkeletonDefaultShader}'s
   * agreement check on the layer, so a layer that also draws a no-tangent
   * skeleton subsource (sharing the single skeleton shader) safely falls back
   * to the generic default rather than installing an uncompilable
   * `prop_tangent()`.
   */
  readonly defaultFragmentMain: string | undefined;
}

export const KIND_CAPABILITIES: Record<
  ZarrVectorsGeometryKind,
  GeometryKindCapabilities
> = {
  point_cloud: {
    // No connectivity, no objects, no direction: the simplest ZVF geometry.
    // Falls back to the generic per-object hash-colour default shader because
    // there is no tangent for a colour-by-direction default to read.
    primitive: "points",
    edgeSource: "none",
    hasObjectModel: false,
    hasWalkOrderTangent: false,
    hasEdgeAdjacencyTangent: false,
    defaultFragmentMain: undefined,
  },
  line: {
    // Independent 2-vertex segments (contact sites, short connectors).  Each
    // fragment holds one pair, so implicit-sequential edges give exactly the
    // one segment per fragment, and the walk-order tangent is that segment's
    // own direction.
    primitive: "lines",
    edgeSource: "implicit_sequential",
    hasObjectModel: true,
    hasWalkOrderTangent: true,
    hasEdgeAdjacencyTangent: false,
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
  streamline: {
    primitive: "lines",
    edgeSource: "implicit_sequential",
    hasObjectModel: true,
    hasWalkOrderTangent: true,
    hasEdgeAdjacencyTangent: false,
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
  polyline: {
    primitive: "lines",
    edgeSource: "implicit_sequential",
    hasObjectModel: true,
    hasWalkOrderTangent: true,
    hasEdgeAdjacencyTangent: false,
    // Polylines carry a well-defined walk-order tangent at every vertex, just
    // like streamlines (tractography is commonly written as "polyline"), so they
    // auto-apply the same colour-by-direction default rather than the segment
    // hash colour.
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
  skeleton: {
    // Synthesise per-vertex tangents from edge adjacency so shaders can
    // `prop_tangent()` (colour-by-direction).  Branch points get the
    // central difference of their first two neighbours.
    //
    // But does NOT nominate the colour-by-direction default, unlike the
    // walk-order kinds above.  That default is
    // `emitRGB(abs(prop_tangent()))`, which computes the colour outright and
    // so consults NEITHER the fixed segment colour NOR the per-object colour
    // hash.  Installing it as the layer default made every colour control
    // inert on a skeleton store -- "fixed colour doesn't work, per-object
    // colouring doesn't work" -- with no error to explain why, because the
    // shader compiles perfectly; it just never asks about segments.
    //
    // Colour-by-direction earns being the default on `line`/`streamline`/
    // `polyline`, which are directional curves whose walk order gives the
    // tangent a consistent meaning. A skeleton is a branching tree: its
    // tangents come from edge adjacency, whose sign is arbitrary per vertex
    // (see `computeTangentsFromEdges`) and which degree-0 vertices do not
    // have at all. Colour-by-direction is a tractography convention, not a
    // sensible default for a neuron morphology carrying `object_attributes`
    // that the user wants to colour by.
    //
    // `undefined` keeps the generic per-object hash-colour default (as `mesh`
    // and `point_cloud` do), which respects the fixed colour and the segment
    // colour hash. `prop_tangent()` is still synthesised and still available,
    // so the Rendering tab's "Direction" preset remains one click away.
    primitive: "lines",
    edgeSource: "both",
    hasObjectModel: true,
    hasWalkOrderTangent: false,
    hasEdgeAdjacencyTangent: true,
    defaultFragmentMain: undefined,
  },
  mesh: {
    // Faces, not edges: the same links family carries records of arity >= 3
    // (spec: geometry_types/mesh.md). Every face is explicit -- there is no
    // fragment-order rule that could imply one -- and a surface has no
    // per-vertex direction, so no tangent and no direction default shader.
    primitive: "triangles",
    edgeSource: "explicit",
    hasObjectModel: true,
    hasWalkOrderTangent: false,
    hasEdgeAdjacencyTangent: false,
    defaultFragmentMain: undefined,
  },
  graph: {
    primitive: "lines",
    edgeSource: "explicit",
    hasObjectModel: true,
    hasWalkOrderTangent: false,
    hasEdgeAdjacencyTangent: true,
    // Edge-adjacency tangents give graph vertices a meaningful direction, so
    // graphs (a common tractography-as-connectivity encoding) auto-apply the
    // colour-by-direction default too, matching the background colour-by default.
    defaultFragmentMain: DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  },
};

/**
 * True iff the kind carries the discrete-object model (`object_index/`,
 * `object_attributes/`, per-fragment `segment_id`).  False only for
 * `point_cloud`, whose vertices are independent measurements; the datasource
 * uses this to decide whether to emit the per-segment subsource and the
 * segment-property map at all.
 */
export function hasObjectModel(kind: ZarrVectorsGeometryKind): boolean {
  return KIND_CAPABILITIES[kind].hasObjectModel;
}
