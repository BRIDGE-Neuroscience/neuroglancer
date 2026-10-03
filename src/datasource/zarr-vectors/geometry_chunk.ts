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
