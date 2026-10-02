/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Decode one zarr-vectors spatial chunk into a `SkeletonChunk`: per-vertex
 * positions, intra-chunk edges (synthesised and/or explicit), optional
 * per-vertex tangent vectors (streamline/polyline), and per-vertex
 * attributes carried verbatim.
 *
 * Cross-chunk continuity for pass-1 is handled by `appendGhostVertices`
 * (also in this module): after the host chunk is decoded, the backend
 * fetches the neighbor's boundary vertex (position + attribute values)
 * for each incident `cross_chunk_links` record, appends it as a "ghost"
 * vertex, and synthesises one bridge edge per ghost.  Each chunk
 * therefore renders independently with its existing per-chunk-isolated
 * GPU resources, but the visible line is continuous across boundaries.
 *
 * The fragment-index format and per-object manifest format are documented
 * in the zarr-vectors spec §7.3 and §7.6.  This module consumes the
 * decoder in `./fragment_index.ts` and is consumed by the chunk-source
 * backend that downloads the underlying byte blobs.
 */

import type { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";

/**
 * How edges between vertices in a chunk are encoded.  Mirrors the spec's
 * root-level `links_convention` field; this drives whether we synthesise
 * edges from fragment ranges, read them explicitly, or both.
 */
export type LinksConvention =
  | "implicit_sequential"
  | "implicit_sequential_with_branches"
  | "explicit";

/**
 * Geometry type (a subset of the spec's `geometry_types` values that map
 * to this render path).  Aliases the canonical
 * {@link ZarrVectorsGeometryKind} declared in `geometry_kind.ts`.  See
 * the capability table there for which kinds get tangent synthesis
 * (streamlines/polylines: walk-order; graphs: edge-adjacency;
 * skeletons: none).
 */
export type GeometryKind = ZarrVectorsGeometryKind;

/** Backing array for per-vertex attribute data (matches zarr-vectors dtypes). */
export type AttributeTypedArray =
  | Float32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array;

/**
 * One decoded chunk ready for upload to the render layer.
 *
 * - `positions` is flat `(numVertices, rank)`; `rank` is fixed per store.
 * - `edges` is flat `(numEdges, 2)` chunk-local vertex indices.
 * - `tangents` is flat `(numVertices, 3)` for streamline/polyline; absent
 *   for skeletons (no canonical "direction").
 * - `vertexAttributes` is parallel to the caller's `attributeNames`,
 *   already reinterpreted to its declared dtype.
 * - `segmentIds` is a synthesised per-vertex segment column carrying the
 *   FULL uint64 id as two interleaved uint32 components `[lo, hi]` (length
 *   `2 * numVertices`), uploaded as a `uvec2` "segment" attribute so the
 *   spatially-indexed render layer colours each fragment by its owning
 *   segment via `segmentColorHash` (matching the flat segmentation) and a
 *   pick surfaces the global id.  Derived from the per-fragment
 *   `fragment_attributes/segment_id` column when present, else the
 *   fragment's index within the chunk (`[f, 0]`) — see
 *   `downloadGeometryChunk`.  Absent for chunks that don't synthesise it
 *   (e.g. empty chunks).
 * - `fragmentIndex` is retained so pass 2 can extract just the fragments
 *   named by a per-object manifest entry without re-decoding bytes.
 */
export interface SkeletonChunk {
  readonly rank: number;
  readonly numVertices: number;
  readonly positions: Float32Array;
  readonly numEdges: number;
  readonly edges: Uint32Array;
  /**
   * Surface faces as a flat TRIANGLE list, `(numFaces, 3)` chunk-local vertex
   * indices. Present only for `mesh` geometry.
   *
   * Always triangles, whatever the store's `link_width`: a face of arity N is
   * fanned into N-2 triangles when the chunk is built, so the GPU path has one
   * primitive to draw and a picked primitive is always a triangle. The original
   * arity is a property of the store's links family, not of a chunk, so it is
   * not carried here.
   */
  readonly faces?: Uint32Array;
  readonly numFaces?: number;
  readonly tangents?: Float32Array;
  readonly vertexAttributes: AttributeTypedArray[];
  readonly segmentIds?: Uint32Array;
  /**
   * Whether {@link segmentIds} holds the store's GLOBAL object ids, as opposed
   * to a per-chunk stand-in.
   *
   * `segmentIds` is always populated for a geometry kind with an object model,
   * but when `fragment_attributes/segment_id` is missing or short the decoder
   * substitutes the fragment's index WITHIN THE CHUNK (`[f, 0]`) — distinct per
   * fragment, deliberately not unified across chunks. That is fine for
   * colouring and picking, and catastrophic for anything that must agree about
   * an object across chunk boundaries: the same tract would carry a different
   * id in every cell it passes through. Anything reasoning about object
   * IDENTITY must check this first.
   */
  readonly segmentIdsAreGlobal?: boolean;
  /**
   * Stable per-vertex identity, from the column named by the store's
   * `zarr_vectors.vertex_id_attribute`. Absent when the store declares none.
   *
   * This is what lets the viewer pick a NODE rather than only the object a
   * vertex belongs to: `resolveNodePickFromChunk` needs `chunk.nodeIds`, and
   * without it the edit UI can see a tract but never a point on it.
   *
   * Deliberately dropped, not extended, by any transform that appends vertices
   * without knowing their identity (ghosts, boundary faces). A misaligned id
   * array is worse than none: it would name the wrong node under the cursor.
   */
  readonly nodeIds?: Int32Array;
  readonly fragmentIndex: FragmentIndex;
}

/**
 * Synthesise intra-chunk edges from a fragment index using the
 * `implicit_sequential` convention: vertex `i` connects to vertex `i+1`
 * inside each fragment.  Edges never cross fragment boundaries — the
 * next fragment is a separate skeleton / streamline / polyline.
 *
 * For range fragments of length N: emit `N - 1` edges.
 * For explicit fragments of length N: emit `N - 1` edges connecting the
 *   indices in their declared order (so an explicit fragment with rows
 *   `[12, 7, 19]` emits edges `(12, 7)` and `(7, 19)`).
 *
 * Returns a flat `Uint32Array` of `(2 * num_edges)` chunk-local vertex
 * indices.
 */
export function synthesizeSequentialEdges(fi: FragmentIndex): Uint32Array {
  // First pass: count edges to allocate exactly.
  let numEdges = 0;
  for (let f = 0; f < fi.numFragments; ++f) {
    if (fi.isRange(f)) {
      const { count } = fi.range(f);
      if (count > 1) numEdges += count - 1;
    } else {
      const idx = fi.indices(f);
      if (idx.length > 1) numEdges += idx.length - 1;
    }
  }
  const out = new Uint32Array(numEdges * 2);
  let cursor = 0;
  for (let f = 0; f < fi.numFragments; ++f) {
    if (fi.isRange(f)) {
      const { start, count } = fi.range(f);
      for (let i = 0; i < count - 1; ++i) {
        out[cursor++] = start + i;
        out[cursor++] = start + i + 1;
      }
    } else {
      const idx = fi.indices(f);
      for (let i = 0; i < idx.length - 1; ++i) {
        out[cursor++] = idx[i];
        out[cursor++] = idx[i + 1];
      }
    }
  }
  return out;
}

/**
 * Merge two edge arrays (implicit-sequential + explicit branches) into
 * one flat array.  Used by the `implicit_sequential_with_branches`
 * skeleton convention: implicit edges come from the fragment ranges,
 * explicit edges come from `links/0/<chunk>`.
 *
 * Both inputs are flat `Uint32Array` of `(2*E)` chunk-local indices.
 */
export function mergeEdges(...edgeArrays: Uint32Array[]): Uint32Array {
  let total = 0;
  for (const a of edgeArrays) total += a.length;
  const out = new Uint32Array(total);
  let offset = 0;
  for (const a of edgeArrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}

/**
 * Compute per-vertex tangent vectors via central differences inside each
 * fragment.  Inputs:
 *
 * - `positions`: flat `(numVertices, rank)` float positions.
 * - `rank`: spatial-index dimensionality (`positions.length / numVertices`).
 *   Must be 2 or 3.  For rank-2 input the output's Z component is zero.
 * - `fi`: fragment index that partitions the chunk into discrete
 *   skeletons/streamlines.  Tangents are computed independently inside
 *   each fragment; boundaries are never crossed.
 *
 * Output is a flat `Float32Array` of `(numVertices * 3)` unit tangent
 * vectors.  Endpoints use forward / backward differences; interior
 * vertices use central differences.  Singletons (fragments of length 1)
 * get a zero tangent.
 *
 * The output is rank-3 even for rank-2 input — neuroglancer expects 3D
 * directions in shader code and packing always-3D keeps the upload
 * pipeline uniform.
 */
export function computeTangents(
  positions: Float32Array,
  rank: number,
  fi: FragmentIndex,
): Float32Array {
  if (rank !== 2 && rank !== 3) {
    throw new Error(
      `computeTangents: rank ${rank} not supported (expected 2 or 3)`,
    );
  }
  const numVertices = positions.length / rank;
  if (!Number.isInteger(numVertices)) {
    throw new Error(
      `computeTangents: positions.length=${positions.length} is not a multiple of rank=${rank}`,
    );
  }
  const out = new Float32Array(numVertices * 3);

  // Visit each fragment's vertex indices in walking order.  Range
  // fragments are contiguous; explicit fragments may revisit non-
  // contiguous chunk rows but still have a well-defined walk order
  // (the order they were stored in).
  for (let f = 0; f < fi.numFragments; ++f) {
    let walk: ArrayLike<number>;
    if (fi.isRange(f)) {
      const { start, count } = fi.range(f);
      const arr = new Uint32Array(count);
      for (let i = 0; i < count; ++i) arr[i] = start + i;
      walk = arr;
    } else {
      walk = fi.indices(f);
    }
    const n = walk.length;
    if (n === 0) continue;
    if (n === 1) {
      // Singleton fragment — zero tangent.  Already initialised.
      continue;
    }
    for (let i = 0; i < n; ++i) {
      const vi = walk[i];
      let prev: number;
      let next: number;
      if (i === 0) {
        prev = walk[0];
        next = walk[1];
      } else if (i === n - 1) {
        prev = walk[n - 2];
        next = walk[n - 1];
      } else {
        prev = walk[i - 1];
        next = walk[i + 1];
      }
      // Tangent direction = next - prev (un-normalised), then unit-normalise.
      const dx = positions[next * rank] - positions[prev * rank];
      const dy = positions[next * rank + 1] - positions[prev * rank + 1];
      const dz =
        rank === 3
          ? positions[next * rank + 2] - positions[prev * rank + 2]
          : 0;
      const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (norm > 0) {
        out[vi * 3] = dx / norm;
        out[vi * 3 + 1] = dy / norm;
        out[vi * 3 + 2] = dz / norm;
      }
      // else: leave as zero (two coincident neighbours — degenerate).
    }
  }
  return out;
}

/**
 * Compute per-vertex tangent vectors using **edge adjacency** rather
 * than fragment walk order.  Generalises {@link computeTangents} to
 * edge-based geometries that have no canonical walk order:
 *
 * - **Degree 0** (isolated vertex): zero tangent.
 * - **Degree 1** (endpoint): tangent points to the lone neighbour
 *   (sign is arbitrary; the standard RGB shader uses `abs()`).
 * - **Degree 2** (linear interior): central difference of the two
 *   neighbours — same formula as walk-order interior vertices on a
 *   degree-2 chain.
 * - **Degree ≥ 3** (branch point): central difference of the **first
 *   two** listed neighbours (adjacency build order).  Branch points
 *   have no canonical direction; this just gives a non-black colour
 *   instead of singling out one branch.  For visualisation it doesn't
 *   matter which two neighbours win.
 *
 * Inputs:
 * - `positions`: flat `(numVertices, rank)` float positions; `rank` is 2 or 3.
 * - `edges`: flat `(numEdges, 2)` chunk-local vertex-index pairs.
 *   Self-loops `(a, a)` are skipped (they contribute no direction).
 *
 * Output is a flat `Float32Array` of `(numVertices * 3)` unit tangent
 * vectors — rank-3 even for rank-2 input (uniform GPU upload format).
 */
export function computeTangentsFromEdges(
  positions: Float32Array,
  rank: number,
  edges: Uint32Array,
  numVertices: number,
): Float32Array {
  if (rank !== 2 && rank !== 3) {
    throw new Error(
      `computeTangentsFromEdges: rank ${rank} not supported (expected 2 or 3)`,
    );
  }
  if (edges.length % 2 !== 0) {
    throw new Error(
      `computeTangentsFromEdges: edges.length=${edges.length} is not a multiple of 2`,
    );
  }
  const out = new Float32Array(numVertices * 3);
  if (edges.length === 0 || numVertices === 0) return out;

  // Build adjacency: for each vertex, the list of its neighbours in the
  // order edges were encountered.  Branch points later pick the first
  // two from this list, so the order is significant but harmless —
  // it's deterministic given a deterministic `edges` array.
  const adj: number[][] = new Array(numVertices);
  for (let v = 0; v < numVertices; ++v) adj[v] = [];
  for (let e = 0; e < edges.length; e += 2) {
    const a = edges[e];
    const b = edges[e + 1];
    if (a === b) continue;
    if (a < numVertices) adj[a].push(b);
    if (b < numVertices) adj[b].push(a);
  }

  for (let v = 0; v < numVertices; ++v) {
    const nbrs = adj[v];
    const d = nbrs.length;
    if (d === 0) continue;
    let aIdx: number;
    let bIdx: number;
    if (d === 1) {
      aIdx = v;
      bIdx = nbrs[0];
    } else {
      aIdx = nbrs[0];
      bIdx = nbrs[1];
    }
    const dx = positions[bIdx * rank] - positions[aIdx * rank];
    const dy = positions[bIdx * rank + 1] - positions[aIdx * rank + 1];
    const dz =
      rank === 3 ? positions[bIdx * rank + 2] - positions[aIdx * rank + 2] : 0;
    const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (norm > 0) {
      out[v * 3] = dx / norm;
      out[v * 3 + 1] = dy / norm;
      out[v * 3 + 2] = dz / norm;
      continue;
    }
    // Central difference cancelled (the first two neighbours are
    // coincident).  Any vertex that participates in an edge should still
    // get a non-black direction under `abs(prop_tangent())`, so fall back
    // to the direction toward the first non-coincident neighbour.
    for (let k = 0; k < nbrs.length; ++k) {
      const nb = nbrs[k];
      const fx = positions[nb * rank] - positions[v * rank];
      const fy = positions[nb * rank + 1] - positions[v * rank + 1];
      const fz =
        rank === 3 ? positions[nb * rank + 2] - positions[v * rank + 2] : 0;
      const fnorm = Math.sqrt(fx * fx + fy * fy + fz * fz);
      if (fnorm > 0) {
        out[v * 3] = fx / fnorm;
        out[v * 3 + 1] = fy / fnorm;
        out[v * 3 + 2] = fz / fnorm;
        break;
      }
    }
    // Else: every neighbour is coincident — truly degenerate, leave zero.
  }

  // Sign-orient tangents consistently across each connected component.
  // Edge-adjacency tangents have an arbitrary per-vertex sign: on a path
  // A-B-C-D the interior vertices point "forward" but the terminal vertex
  // computes `C - D` (backward).  A line segment interpolates its two
  // endpoint tangents, so opposing signs cross through zero at the
  // midpoint — rendering a black band on every terminal edge under
  // `abs(prop_tangent())`.  Walk-order tangents avoid this by construction;
  // here we replicate it with a flood-fill that flips each newly-reached
  // vertex's tangent to align (dot >= 0) with the vertex it came from, so
  // no edge has opposing endpoint tangents.  Sign is arbitrary anyway
  // (the standard shader uses `abs()`), so only consistency matters.
  const oriented = new Uint8Array(numVertices);
  const stack: number[] = [];
  for (let s = 0; s < numVertices; ++s) {
    if (oriented[s] || adj[s].length === 0) continue;
    oriented[s] = 1;
    stack.push(s);
    while (stack.length > 0) {
      const u = stack.pop()!;
      const ux = out[u * 3];
      const uy = out[u * 3 + 1];
      const uz = out[u * 3 + 2];
      for (const w of adj[u]) {
        if (oriented[w]) continue;
        oriented[w] = 1;
        const dot = out[w * 3] * ux + out[w * 3 + 1] * uy + out[w * 3 + 2] * uz;
        if (dot < 0) {
          // `0 - x` (not `-x`) so a zero component negates to +0, not -0.
          out[w * 3] = 0 - out[w * 3];
          out[w * 3 + 1] = 0 - out[w * 3 + 1];
          out[w * 3 + 2] = 0 - out[w * 3 + 2];
        }
        stack.push(w);
      }
    }
  }
  return out;
}

/**
 * Build a `SkeletonChunk` from already-decoded inputs.  Callers
 * (typically the chunk-source backend) are responsible for fetching the
 * raw bytes and running the dtype-aware reinterpretations.  This
 * function is the pure decode / shape-assembly step that the unit tests
 * can drive without HTTP machinery.
 */
/**
 * Keys already warned about, so a malformed store logs one diagnostic rather
 * than one per chunk per level.
 */
const warnedChunkKeys = new Set<string>();
function warnOnceChunk(key: string, message: string): void {
  if (warnedChunkKeys.has(key)) return;
  warnedChunkKeys.add(key);
  console.warn(message);
}

/**
 * Fan a face list of arity `arity` into a flat triangle list.
 *
 * ZVF face records are `link_width`-wide and the spec allows more than 3 (quads
 * are called out explicitly). A convex-fan triangulation -- `(v0, vi, vi+1)` --
 * is correct for the convex faces meshes are built from and preserves the
 * winding the producer wrote, which is the only orientation information ZVF
 * keeps. `arity === 3` returns the input unchanged.
 */
export function triangulateFaces(
  faces: Uint32Array | undefined,
  arity: number,
): Uint32Array {
  if (faces === undefined || faces.length === 0) return new Uint32Array(0);
  if (!Number.isInteger(arity) || arity < 3) {
    throw new Error(
      `buildGeometryChunk: link_width=${arity} cannot describe a face`,
    );
  }
  if (faces.length % arity !== 0) {
    throw new Error(
      `buildGeometryChunk: ${faces.length} face indices is not a multiple ` +
        `of link_width=${arity}`,
    );
  }
  if (arity === 3) return faces;
  const numFaces = faces.length / arity;
  const trianglesPerFace = arity - 2;
  const out = new Uint32Array(numFaces * trianglesPerFace * 3);
  let cursor = 0;
  for (let f = 0; f < numFaces; ++f) {
    const base = f * arity;
    for (let i = 1; i < arity - 1; ++i) {
      out[cursor++] = faces[base];
      out[cursor++] = faces[base + i];
      out[cursor++] = faces[base + i + 1];
    }
  }
  return out;
}

export function buildGeometryChunk(args: {
  rank: number;
  positions: Float32Array;
  fragmentIndex: FragmentIndex;
  /** From `links/0/<chunk>`, already reinterpreted to a chunk-local uint
   *  index array.  Flat `(E, 2)`.  Empty / undefined for
   *  `implicit_sequential` stores. */
  explicitEdges?: Uint32Array;
  linksConvention: LinksConvention;
  geometryKind: GeometryKind;
  vertexAttributes: AttributeTypedArray[];
  /** Synthesised per-vertex uint32 segment column (see {@link SkeletonChunk.segmentIds}). */
  segmentIds?: Uint32Array;
  /** See {@link SkeletonChunk.segmentIdsAreGlobal}. */
  segmentIdsAreGlobal?: boolean;
  /** See {@link SkeletonChunk.nodeIds}. */
  nodeIds?: Int32Array;
  /**
   * Face records read from the links family, flat and chunk-local, for surface
   * geometry. `faceArity` is the store's declared `link_width`.
   */
  faces?: Uint32Array;
  faceArity?: number;
  /** See `GeometryChunkDownloadOptions.linkedSkeletonLayout`. */
  linkedSkeletonLayout?: boolean;
  /** Children whose implied parent a cross-chunk record replaces. */
  relinkedChildren?: ReadonlySet<number>;
}): SkeletonChunk {
  const {
    rank,
    positions,
    fragmentIndex,
    explicitEdges,
    linksConvention,
    geometryKind,
    vertexAttributes,
    segmentIds,
    segmentIdsAreGlobal,
    nodeIds,
    faces,
    faceArity,
  } = args;

  const numVertices = positions.length / rank;
  if (!Number.isInteger(numVertices)) {
    throw new Error(
      `buildGeometryChunk: positions.length=${positions.length} is not a multiple of rank=${rank}`,
    );
  }

  const caps = KIND_CAPABILITIES[geometryKind];

  if (caps.primitive === "triangles") {
    // A surface's links are faces. There is no edge list to synthesise: the
    // fragment order of a mesh chunk says nothing about connectivity, and the
    // face records are the connectivity.
    const triangles = triangulateFaces(faces, faceArity ?? 3);
    return {
      rank,
      numVertices,
      positions,
      numEdges: 0,
      edges: new Uint32Array(0),
      faces: triangles,
      numFaces: triangles.length / 3,
      tangents: undefined,
      vertexAttributes,
      segmentIds,
      segmentIdsAreGlobal,
      nodeIds,
      fragmentIndex,
    };
  }

  let edges: Uint32Array;
  if (caps.edgeSource === "none") {
    // Point clouds have no connectivity, and their fragments are spatial BINS
    // holding many unrelated points -- so the implicit-sequential rule below
    // would wire every bin into a spaghetti polyline rather than drawing
    // nothing.  The kind wins over the store's `links_convention`, which the
    // spec says a point cloud need not even declare.
    edges = new Uint32Array(0);
    if (explicitEdges !== undefined && explicitEdges.length > 0) {
      warnOnceChunk(
        `edges-ignored-${geometryKind}`,
        `zarr-vectors: ignoring ${explicitEdges.length >> 1} link record(s) on ` +
          `a ${geometryKind} chunk -- the geometry has no connectivity.`,
      );
    }
  } else {
    edges =
      args.linkedSkeletonLayout &&
      linksConvention === "implicit_sequential_with_branches"
        ? linkedSkeletonEdges(
            fragmentIndex,
            explicitEdges ?? new Uint32Array(0),
            args.relinkedChildren,
          )
        : synthesizeEdgesForConvention(
            linksConvention,
            fragmentIndex,
            explicitEdges,
          );
  }

  // Per-vertex tangent synthesis is driven by the capability table:
  //   - `hasWalkOrderTangent` (line / streamline / polyline): central
  //     differences along the fragment walk; sign is consistent across
  //     bridges because every fragment has a well-defined direction.
  //   - `hasEdgeAdjacencyTangent` (graph / skeleton): central differences
  //     along edge adjacency; tangents are well-defined for degree-2 vertices
  //     and a sensible non-zero direction at branch points.
  //   - Neither (point_cloud): no tangent -- there is no direction to have.
  let tangents: Float32Array | undefined;
  if (caps.hasWalkOrderTangent) {
    tangents = computeTangents(positions, rank, fragmentIndex);
  } else if (caps.hasEdgeAdjacencyTangent) {
    tangents = computeTangentsFromEdges(positions, rank, edges, numVertices);
  }

  return {
    rank,
    numVertices,
    positions,
    numEdges: edges.length >> 1,
    edges,
    tangents,
    vertexAttributes,
    segmentIds,
    segmentIdsAreGlobal,
    nodeIds,
    fragmentIndex,
  };
}

/**
 * Edges of a skeleton in zarr-vectors-tools' "linked" layout, where each
 * fragment row's parent is the previous row UNLESS a stored `[child, parent]`
 * record names a different one (a branch, or a parent in another chunk).
 * Unioning the two, as the plain convention does, draws a chord from every
 * branch child to the unrelated row before it.
 */
export function linkedSkeletonEdges(
  fragmentIndex: FragmentIndex,
  records: Uint32Array,
  relinkedChildren: ReadonlySet<number> | undefined,
): Uint32Array {
  const replaced = new Set<number>(relinkedChildren ?? []);
  for (let i = 0; i < records.length; i += 2) replaced.add(records[i]);
  const implied = synthesizeSequentialEdges(fragmentIndex);
  const out: number[] = [];
  for (let i = 0; i < implied.length; i += 2) {
    if (!replaced.has(implied[i + 1])) out.push(implied[i], implied[i + 1]);
  }
  for (let i = 0; i < records.length; i += 2) {
    out.push(records[i + 1], records[i]);
  }
  return Uint32Array.from(out);
}

/**
 * Edges for a kind that HAS connectivity, following the store's declared
 * `links_convention`.  Split out of {@link buildGeometryChunk} so the
 * no-connectivity case reads as the separate decision it is.
 */
function synthesizeEdgesForConvention(
  linksConvention: LinksConvention,
  fragmentIndex: FragmentIndex,
  explicitEdges: Uint32Array | undefined,
): Uint32Array {
  switch (linksConvention) {
    case "implicit_sequential":
      // Line / polyline / streamline: edges come purely from fragment ranges.
      if (explicitEdges && explicitEdges.length > 0) {
        throw new Error(
          "buildGeometryChunk: implicit_sequential convention got " +
            "explicit edges; the writer should not emit links/0/<chunk> " +
            "in this mode",
        );
      }
      return synthesizeSequentialEdges(fragmentIndex);
    case "implicit_sequential_with_branches":
      // Skeleton: implicit sequential edges plus optional explicit
      // branch edges read from links/0/<chunk>.
      return mergeEdges(
        synthesizeSequentialEdges(fragmentIndex),
        explicitEdges ?? new Uint32Array(0),
      );
    case "explicit":
      // General graph: every edge is explicit.
      if (explicitEdges === undefined) {
        throw new Error(
          "buildGeometryChunk: explicit links_convention requires explicitEdges",
        );
      }
      return explicitEdges;
    default: {
      const _exhaustive: never = linksConvention;
      throw new Error(`Unhandled links_convention: ${_exhaustive}`);
    }
  }
}

/**
 * One bridge whose endpoints have both been resolved to chunk-local
 * vertex indices (either real or ghost) within the host chunk.  Used
 * by {@link recomputeTangentsForBridges} to refresh per-vertex
 * tangents at coarser pyramid levels where the writer's "one
 * metavertex per fragment" model leaves `computeTangents` with no
 * in-fragment neighbor to difference against.
 *
 * Convention matches `cross_chunk_links` semantics: `predecessor` is
 * `endpoint[0]` (the "before" vertex in walk order), `successor` is
 * `endpoint[1]` (the "after" vertex).
 */
export interface ResolvedBridge {
  readonly predecessorLocalIdx: number;
  readonly successorLocalIdx: number;
}

/**
 * Re-derive per-vertex tangents for metavertices whose
 * `computeTangents`-supplied value is zero (single-vertex fragments at
 * coarser pyramid levels).  For each bridge `predecessor → successor`,
 * accumulate the step direction `pos[successor] - pos[predecessor]` at
 * BOTH endpoints; normalise the accumulators in-place.
 *
 * Vertices with at least one incident bridge get their tangent
 * overwritten.  Vertices with no incident bridge keep their existing
 * tangent (whatever `computeTangents` produced — usually correct at
 * level 0, zero at coarser levels for isolated metavertices).
 *
 * Returns a new `SkeletonChunk` with updated `tangents`; no mutation
 * of inputs.  Returns the chunk unchanged for non-streamline /
 * non-polyline geometry (no tangents to update) or when `bridges` is
 * empty.
 */
export function recomputeTangentsForBridges(
  chunk: SkeletonChunk,
  bridges: readonly ResolvedBridge[],
): SkeletonChunk {
  if (chunk.tangents === undefined || bridges.length === 0) return chunk;
  const { rank, numVertices, positions, tangents } = chunk;
  // Touch mask: 1 if this vertex has at least one incident bridge.
  const touched = new Uint8Array(numVertices);
  const accum = new Float32Array(numVertices * 3);
  for (const bridge of bridges) {
    const p = bridge.predecessorLocalIdx;
    const s = bridge.successorLocalIdx;
    if (p < 0 || p >= numVertices || s < 0 || s >= numVertices) continue;
    const dx = positions[s * rank] - positions[p * rank];
    const dy = positions[s * rank + 1] - positions[p * rank + 1];
    const dz =
      rank === 3 ? positions[s * rank + 2] - positions[p * rank + 2] : 0;
    // Step direction `p → s` contributes to BOTH endpoints' forward-walk
    // tangent (predecessor sees `s` ahead; successor sees `p` behind).
    accum[p * 3] += dx;
    accum[p * 3 + 1] += dy;
    accum[p * 3 + 2] += dz;
    accum[s * 3] += dx;
    accum[s * 3 + 1] += dy;
    accum[s * 3 + 2] += dz;
    touched[p] = 1;
    touched[s] = 1;
  }
  const newTangents = new Float32Array(tangents);
  for (let v = 0; v < numVertices; ++v) {
    if (touched[v] === 0) continue;
    const dx = accum[v * 3];
    const dy = accum[v * 3 + 1];
    const dz = accum[v * 3 + 2];
    const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (norm > 0) {
      newTangents[v * 3] = dx / norm;
      newTangents[v * 3 + 1] = dy / norm;
      newTangents[v * 3 + 2] = dz / norm;
    }
    // else: accumulator cancelled (predecessor + successor symmetric);
    // keep existing tangent.
  }
  return { ...chunk, tangents: newTangents };
}

/**
 * Append extra edges (flat `(a, b)` chunk-local vertex-index pairs) to
 * the chunk's existing edge list.  Used by the pass-1 backend to add
 * intra-chunk bridge edges from same-chunk `cross_chunk_links` records
 * (the coarser-pyramid-level case where the writer encodes
 * metavertex-to-metavertex transitions inside one chunk).
 *
 * No vertex insertion: both endpoints already live in the host
 * chunk's vertex texture, so the renderer treats these as ordinary
 * intra-chunk edges.
 *
 * Returns the input chunk unchanged when `extraEdges.length === 0`.
 * Throws if `extraEdges.length` isn't a multiple of 2, or if any
 * referenced vertex index is out of range.
 */
export function appendIntraChunkEdges(
  chunk: SkeletonChunk,
  extraEdges: Uint32Array,
): SkeletonChunk {
  if (extraEdges.length === 0) return chunk;
  if (extraEdges.length % 2 !== 0) {
    throw new Error(
      `appendIntraChunkEdges: extraEdges.length=${extraEdges.length} ` +
        `is not a multiple of 2`,
    );
  }
  for (let i = 0; i < extraEdges.length; ++i) {
    if (extraEdges[i] >= chunk.numVertices) {
      throw new Error(
        `appendIntraChunkEdges: edge endpoint ${extraEdges[i]} out of ` +
          `[0, ${chunk.numVertices})`,
      );
    }
  }
  const merged = new Uint32Array(chunk.edges.length + extraEdges.length);
  merged.set(chunk.edges, 0);
  merged.set(extraEdges, chunk.edges.length);
  return {
    ...chunk,
    numEdges: merged.length >> 1,
    edges: merged,
  };
}

/**
 * One "ghost" vertex to append to a `SkeletonChunk`.  A ghost is a copy
 * of a neighbor chunk's boundary vertex placed inside the host chunk's
 * vertex texture so the host can draw one edge from its real boundary
 * endpoint to the neighbor's endpoint without needing the neighbor's
 * GPU buffers bound at draw time.
 *
 * - `position`: length-`rank` world-space coordinates copied verbatim
 *   from the neighbor chunk.
 * - `attributes`: parallel to the host chunk's `vertexAttributes`.
 *   Each element holds the neighbor's stored value at the bridging
 *   vertex.  When the neighbor lacks an attribute file (e.g. pyramid
 *   levels without `vertex_attributes/<name>/`), the caller may emit a
 *   zero-filled typed-array of length 1 — matches the existing
 *   per-chunk zero-fill rule in `downloadGeometryChunk`.
 * - `bridgeFromLocalVertex`: chunk-local index of the host endpoint
 *   that should be connected to this ghost.  Out-of-range indices are
 *   rejected by `appendGhostVertices`.
 */
export interface GhostVertexRecord {
  readonly position: Float32Array;
  readonly attributes: AttributeTypedArray[];
  readonly bridgeFromLocalVertex: number;
  /**
   * Index of the request this record answers.
   *
   * The fetch drops requests whose neighbour data is missing, so the returned
   * array is not positionally aligned with the requests. The line path does not
   * care -- each ghost carries its own host vertex -- but a boundary FACE has to
   * put three specific endpoints back together, so it needs to know which
   * request each surviving ghost came from.
   */
  readonly requestIndex?: number;
  /**
   * True when this ghost represents the **predecessor** of the host in
   * the streamline's walk order (i.e. it sits "before" the host along
   * the polyline).  False (default) when it's the successor.
   *
   * Why this matters: the ghost's synthesised tangent must point in
   * the FORWARD walk direction so it matches the host's
   * fragment-derived tangent across the bridge edge.  When the
   * ghost is the successor (typical X-side of a chunk crossing), the
   * forward direction is `normalize(ghost - host)`.  When it's the
   * predecessor (typical Y-side of the same crossing), the forward
   * direction is `normalize(host - ghost)` — the SIGN-FLIPPED form.
   * Getting this wrong made one side of every bridge interpolate
   * `forward + backward` ≈ `0`, producing visible black gaps in the
   * default RGB-by-tangent streamline shader.
   *
   * Skeletons / polylines without a meaningful walk direction simply
   * leave the synthesised tangent at zero, and this flag has no
   * effect.
   */
  readonly isGhostPredecessor?: boolean;
}

/**
 * Pure function: append ghost vertices + their bridge edges to an
 * existing `SkeletonChunk`, returning a new `SkeletonChunk`.  Does not
 * mutate the input.
 *
 * Ghost vertices are inserted at the end of the chunk's vertex space:
 * positions, vertex attributes, and (for streamline/polyline) tangents
 * all grow by `ghosts.length` entries.  The edge array grows by one
 * entry per ghost — connecting `ghost.bridgeFromLocalVertex` to the
 * newly-appended ghost index.  The fragment index is preserved
 * verbatim: ghosts are not part of any fragment.
 *
 * Ghost tangents (streamline/polyline only) are derived from the bridge
 * direction itself: `normalize(ghost.position - hostPosition)`.  This
 * is the only well-defined choice — a ghost has no neighbor in the
 * host chunk to do a central difference against, and the bridge edge
 * IS the only direction this vertex participates in locally.  Note
 * the host vertex retains its original fragment-derived tangent.
 *
 * Returns the input chunk unchanged when `ghosts.length === 0`.
 */
/**
 * Turn face templates into a flat triangle list once the ghosts have landed,
 * dropping any face whose ghost was not fetched. Faces of arity > 3 are
 * fanned, matching {@link triangulateFaces} for the intra-chunk path.
 */
export function resolveBoundaryFaces(
  faceTemplates: readonly Int32Array[],
  ghosts: readonly GhostVertexRecord[],
  baseGhostIndex: number,
): Uint32Array {
  const ghostIndexByRequest = new Map<number, number>();
  for (let g = 0; g < ghosts.length; ++g) {
    const requestIndex = ghosts[g].requestIndex;
    if (requestIndex !== undefined) {
      ghostIndexByRequest.set(requestIndex, baseGhostIndex + g);
    }
  }
  const out: number[] = [];
  for (const template of faceTemplates) {
    const corners = new Array<number>(template.length);
    let resolved = true;
    for (let i = 0; i < template.length; ++i) {
      const ref = template[i];
      if (ref >= 0) {
        corners[i] = ref;
        continue;
      }
      const ghostIndex = ghostIndexByRequest.get(-ref - 1);
      if (ghostIndex === undefined) {
        resolved = false;
        break;
      }
      corners[i] = ghostIndex;
    }
    if (!resolved) continue;
    for (let i = 1; i < corners.length - 1; ++i) {
      out.push(corners[0], corners[i], corners[i + 1]);
    }
  }
  return Uint32Array.from(out);
}

/**
 * Append boundary-face ghosts to a surface chunk, then the faces that use them.
 *
 * Distinct from {@link appendGhostVertices}, which exists for line geometry: it
 * appends one bridge EDGE per ghost and synthesises a walk-direction tangent.
 * A boundary face needs neither -- its ghosts are corners of triangles the
 * caller has already resolved, and a surface has no tangent -- so the two
 * cannot share an implementation without one of them lying about the geometry.
 *
 * `extraFaces` is a flat triangle list in the POST-append index space: local
 * vertices keep their index, and ghost `g` is at `chunk.numVertices + g`.
 */
export function appendBoundaryFaces(
  chunk: SkeletonChunk,
  ghosts: readonly GhostVertexRecord[],
  extraFaces: Uint32Array,
): SkeletonChunk {
  if (ghosts.length === 0 && extraFaces.length === 0) return chunk;
  const { rank, numVertices, positions, vertexAttributes, segmentIds } = chunk;
  const numGhosts = ghosts.length;
  const newNumVertices = numVertices + numGhosts;

  const newPositions = new Float32Array(newNumVertices * rank);
  newPositions.set(positions, 0);
  for (let g = 0; g < numGhosts; ++g) {
    newPositions.set(ghosts[g].position, (numVertices + g) * rank);
  }

  const newVertexAttributes: AttributeTypedArray[] = [];
  for (let a = 0; a < vertexAttributes.length; ++a) {
    const src = vertexAttributes[a];
    const Ctor = src.constructor as new (n: number) => AttributeTypedArray;
    const dst = new Ctor(newNumVertices);
    (dst as unknown as { set: (a: ArrayLike<number>, o: number) => void }).set(
      src as unknown as ArrayLike<number>,
      0,
    );
    for (let g = 0; g < numGhosts; ++g) {
      const ghostAttr = ghosts[g].attributes[a];
      (dst as unknown as { [k: number]: number })[numVertices + g] = (
        ghostAttr as unknown as { [k: number]: number }
      )[0];
    }
    newVertexAttributes.push(dst);
  }

  // A boundary face joins one surface, so the ghost belongs to the same object
  // as the local corner that pulled it in -- inherit that corner's id, exactly
  // as the line path does across a bridge.
  let newSegmentIds: Uint32Array | undefined;
  if (segmentIds !== undefined) {
    newSegmentIds = new Uint32Array(newNumVertices * 2);
    newSegmentIds.set(segmentIds, 0);
    for (let g = 0; g < numGhosts; ++g) {
      const host = ghosts[g].bridgeFromLocalVertex;
      newSegmentIds[(numVertices + g) * 2] = segmentIds[host * 2];
      newSegmentIds[(numVertices + g) * 2 + 1] = segmentIds[host * 2 + 1];
    }
  }

  const existingFaces = chunk.faces ?? new Uint32Array(0);
  const newFaces = new Uint32Array(existingFaces.length + extraFaces.length);
  newFaces.set(existingFaces, 0);
  newFaces.set(extraFaces, existingFaces.length);

  return {
    ...chunk,
    numVertices: newNumVertices,
    positions: newPositions,
    vertexAttributes: newVertexAttributes,
    segmentIds: newSegmentIds,
    segmentIdsAreGlobal: chunk.segmentIdsAreGlobal,
    // The appended corners come from neighbouring chunks and carry no id of
    // their own; keeping the old array would misname every vertex past the
    // original count. See {@link SkeletonChunk.nodeIds}.
    nodeIds: undefined,
    faces: newFaces,
    numFaces: newFaces.length / 3,
  };
}

export function appendGhostVertices(
  chunk: SkeletonChunk,
  ghosts: readonly GhostVertexRecord[],
): SkeletonChunk {
  if (ghosts.length === 0) return chunk;

  const {
    rank,
    numVertices,
    positions,
    edges,
    tangents,
    vertexAttributes,
    segmentIds,
  } = chunk;
  const numGhosts = ghosts.length;
  const newNumVertices = numVertices + numGhosts;

  // Validate inputs early — easier to debug than a downstream texture-
  // upload mismatch.
  if (
    vertexAttributes.length === 0 &&
    ghosts.some((g) => g.attributes.length > 0)
  ) {
    throw new Error(
      `appendGhostVertices: host chunk has 0 attributes but ghost ` +
        `supplied ${ghosts[0].attributes.length}`,
    );
  }
  for (let g = 0; g < numGhosts; ++g) {
    const ghost = ghosts[g];
    if (ghost.position.length !== rank) {
      throw new Error(
        `appendGhostVertices: ghost ${g} position length ${ghost.position.length} != rank ${rank}`,
      );
    }
    if (ghost.attributes.length !== vertexAttributes.length) {
      throw new Error(
        `appendGhostVertices: ghost ${g} has ${ghost.attributes.length} ` +
          `attributes; host chunk has ${vertexAttributes.length}`,
      );
    }
    if (
      ghost.bridgeFromLocalVertex < 0 ||
      ghost.bridgeFromLocalVertex >= numVertices
    ) {
      throw new Error(
        `appendGhostVertices: ghost ${g} bridgeFromLocalVertex=` +
          `${ghost.bridgeFromLocalVertex} out of [0, ${numVertices})`,
      );
    }
  }

  // Positions: append rank floats per ghost.
  const newPositions = new Float32Array(newNumVertices * rank);
  newPositions.set(positions, 0);
  for (let g = 0; g < numGhosts; ++g) {
    newPositions.set(ghosts[g].position, (numVertices + g) * rank);
  }

  // Tangents (if present): append 3 floats per ghost, computed from the
  // bridge direction `normalize(ghost - host)`.
  let newTangents: Float32Array | undefined;
  if (tangents !== undefined) {
    newTangents = new Float32Array(newNumVertices * 3);
    newTangents.set(tangents, 0);
    for (let g = 0; g < numGhosts; ++g) {
      const ghost = ghosts[g];
      const host = ghost.bridgeFromLocalVertex;
      // Compute the forward walk direction at the ghost's position.
      // Sign depends on whether the ghost sits BEFORE or AFTER the
      // host in walk order — see `isGhostPredecessor` docs above.
      const sign = ghost.isGhostPredecessor === true ? -1 : 1;
      const dx = sign * (ghost.position[0] - positions[host * rank]);
      const dy = sign * (ghost.position[1] - positions[host * rank + 1]);
      const dz =
        rank === 3
          ? sign * (ghost.position[2] - positions[host * rank + 2])
          : 0;
      const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const out = (numVertices + g) * 3;
      if (norm > 0) {
        newTangents[out] = dx / norm;
        newTangents[out + 1] = dy / norm;
        newTangents[out + 2] = dz / norm;
      } else {
        // Coincident host/ghost — the common chunk-boundary case, where
        // the writer duplicates the boundary vertex so the bridge has zero
        // length.  Inherit the host's tangent so the (zero-length) bridge
        // stub colours continuously with the path instead of rendering
        // black under `abs(prop_tangent())`.
        newTangents[out] = newTangents[host * 3];
        newTangents[out + 1] = newTangents[host * 3 + 1];
        newTangents[out + 2] = newTangents[host * 3 + 2];
      }
    }
  }

  // Vertex attributes: each grows by one element per ghost.  Construct
  // a new typed-array of the SAME constructor as the host's so dtype
  // stays consistent across the chunk.
  const newVertexAttributes: AttributeTypedArray[] = [];
  for (let a = 0; a < vertexAttributes.length; ++a) {
    const src = vertexAttributes[a];
    const Ctor = src.constructor as new (n: number) => AttributeTypedArray;
    const dst = new Ctor(newNumVertices);
    (dst as unknown as { set: (a: ArrayLike<number>, o: number) => void }).set(
      src as unknown as ArrayLike<number>,
      0,
    );
    for (let g = 0; g < numGhosts; ++g) {
      const ghostAttr = ghosts[g].attributes[a];
      // Single-element ghost attribute — first slot of the typed-array.
      // Spec note: callers populate ghost.attributes[a].length === 1.
      (dst as unknown as { [k: number]: number })[numVertices + g] = (
        ghostAttr as unknown as { [k: number]: number }
      )[0];
    }
    newVertexAttributes.push(dst);
  }

  // Segment ids (if present): each ghost is the far end of a bridge that
  // connects the SAME segment, so it inherits its host endpoint's id.
  // Keeps the bridge edge a single colour across the chunk boundary.  Two
  // uint32 components per vertex ([lo, hi]) — copy both.
  let newSegmentIds: Uint32Array | undefined;
  if (segmentIds !== undefined) {
    newSegmentIds = new Uint32Array(newNumVertices * 2);
    newSegmentIds.set(segmentIds, 0);
    for (let g = 0; g < numGhosts; ++g) {
      const host = ghosts[g].bridgeFromLocalVertex;
      newSegmentIds[(numVertices + g) * 2] = segmentIds[host * 2];
      newSegmentIds[(numVertices + g) * 2 + 1] = segmentIds[host * 2 + 1];
    }
  }

  // Edges: append one bridge edge per ghost — (hostLocalIdx, ghostIdx).
  const newNumEdges = (edges.length >> 1) + numGhosts;
  const newEdges = new Uint32Array(newNumEdges * 2);
  newEdges.set(edges, 0);
  for (let g = 0; g < numGhosts; ++g) {
    const off = edges.length + g * 2;
    newEdges[off] = ghosts[g].bridgeFromLocalVertex;
    newEdges[off + 1] = numVertices + g;
  }

  return {
    rank,
    numVertices: newNumVertices,
    positions: newPositions,
    numEdges: newNumEdges,
    edges: newEdges,
    tangents: newTangents,
    vertexAttributes: newVertexAttributes,
    segmentIds: newSegmentIds,
    segmentIdsAreGlobal: chunk.segmentIdsAreGlobal,
    fragmentIndex: chunk.fragmentIndex,
  };
}
