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

/**
 * @file Decodes one spatial chunk of a level: its vertices, fragments,
 * attributes and per-fragment segment ids, and the edges, faces and tangents
 * they imply.
 *
 * Edges follow the store's `links_convention`, as zarr-vectors-py's readers
 * apply it: `implicit_sequential` joins consecutive rows of each fragment;
 * `implicit_sequential_with_branches` does too, except that a stored
 * `[child, parent]` record REPLACES the parent the row order implies
 * (`read_graph`); `explicit` uses the stored records only.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/base.js";
import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  attributeDtype,
  decodeFloat32,
  decodeIndices,
  decodeUint64,
  ELEMENT_BYTES,
  isElementType,
} from "#src/datasource/zarr-vectors/dtype.js";
import type { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import { decodeFragments } from "#src/datasource/zarr-vectors/fragment_index.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import type { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import { intraLinksPath } from "#src/datasource/zarr-vectors/links.js";
import { warnOnce } from "#src/datasource/zarr-vectors/util.js";

export interface DecodedChunk {
  numVertices: number;
  /** x, y, z per vertex. */
  positions: Float32Array;
  fragments: FragmentIndex;
  /** Vertex index pairs. */
  edges: Uint32Array;
  /** Vertex index triples (meshes). */
  faces: Uint32Array;
  /** Unit tangent per vertex, for kinds that have one. */
  tangents: Float32Array | undefined;
  /** One array per exposed attribute, `components` values per vertex. */
  attributes: Float32Array[];
  /**
   * Segment id per vertex (two uint32, low word first) in a store without
   * objects: a synthesised id per point (or per fragment).
   */
  segmentIds: Uint32Array | undefined;
  /**
   * A per-fragment id column (`DecodeChunkOptions.fragmentIdColumn`), one
   * value per fragment; `undefined` where the chunk has none or a short
   * one. What the values mean is the caller's to decide.
   */
  fragmentIds: BigUint64Array | undefined;
}

/** Calls `fn` with each vertex of fragment `f`, in walk order. */
export function forEachFragmentVertex(
  fragments: FragmentIndex,
  f: number,
  fn: (vertex: number, i: number) => void,
) {
  if (fragments.isRange(f)) {
    const { start, count } = fragments.range(f);
    for (let i = 0; i < count; ++i) fn(start + i, i);
  } else {
    fragments.indices(f).forEach(fn);
  }
}

/** Edges joining consecutive vertices of each fragment. */
export function sequentialEdges(fragments: FragmentIndex): Uint32Array {
  const out: number[] = [];
  for (let f = 0; f < fragments.numFragments; ++f) {
    let prev = -1;
    forEachFragmentVertex(fragments, f, (v) => {
      if (prev >= 0) out.push(prev, v);
      prev = v;
    });
  }
  return Uint32Array.from(out);
}

/**
 * Row-order edges, except where a `[child, parent]` record (or a link from
 * another chunk, `relinkedChildren`) names the child's parent.
 */
export function branchedEdges(
  fragments: FragmentIndex,
  records: Uint32Array,
  relinkedChildren?: ReadonlySet<number>,
): Uint32Array {
  const replaced = new Set<number>(relinkedChildren);
  for (let i = 0; i < records.length; i += 2) replaced.add(records[i]);
  const implied = sequentialEdges(fragments);
  const out: number[] = [];
  for (let i = 0; i < implied.length; i += 2) {
    if (!replaced.has(implied[i + 1])) out.push(implied[i], implied[i + 1]);
  }
  for (let i = 0; i < records.length; i += 2) {
    out.push(records[i + 1], records[i]);
  }
  return Uint32Array.from(out);
}

function normalizeInto(out: Float32Array, v: number, d: number[]) {
  const norm = Math.hypot(d[0], d[1], d[2]);
  if (norm > 0) {
    out[3 * v] = d[0] / norm;
    out[3 * v + 1] = d[1] / norm;
    out[3 * v + 2] = d[2] / norm;
  }
}

function delta(positions: Float32Array, from: number, to: number) {
  return [0, 1, 2].map((d) => positions[3 * to + d] - positions[3 * from + d]);
}

/** Tangents along each fragment's walk (central differences, one-sided at ends). */
export function walkTangents(
  positions: Float32Array,
  fragments: FragmentIndex,
): Float32Array {
  const out = new Float32Array(positions.length);
  for (let f = 0; f < fragments.numFragments; ++f) {
    const walk: number[] = [];
    forEachFragmentVertex(fragments, f, (v) => walk.push(v));
    const n = walk.length;
    if (n < 2) continue;
    for (let i = 0; i < n; ++i) {
      const prev = walk[Math.max(0, i - 1)];
      const next = walk[Math.min(n - 1, i + 1)];
      normalizeInto(out, walk[i], delta(positions, prev, next));
    }
  }
  return out;
}

/**
 * Tangents of a graph or tree: through the first two neighbours (towards the
 * only one at an end), with signs made consistent along each component so
 * interpolation across an edge never passes through zero.
 */
export function edgeTangents(
  positions: Float32Array,
  edges: Uint32Array,
): Float32Array<ArrayBuffer> {
  const n = positions.length / 3;
  const out = new Float32Array(positions.length);
  const neighbours: number[][] = Array.from({ length: n }, () => []);
  for (let e = 0; e < edges.length; e += 2) {
    const [a, b] = [edges[e], edges[e + 1]];
    if (a === b || a >= n || b >= n) continue;
    neighbours[a].push(b);
    neighbours[b].push(a);
  }
  for (let v = 0; v < n; ++v) {
    const nb = neighbours[v];
    if (nb.length === 0) continue;
    normalizeInto(
      out,
      v,
      nb.length === 1
        ? delta(positions, v, nb[0])
        : delta(positions, nb[0], nb[1]),
    );
  }
  const oriented = new Uint8Array(n);
  for (let s = 0; s < n; ++s) {
    if (oriented[s] || neighbours[s].length === 0) continue;
    oriented[s] = 1;
    const stack = [s];
    while (stack.length > 0) {
      const u = stack.pop()!;
      for (const w of neighbours[u]) {
        if (oriented[w]) continue;
        oriented[w] = 1;
        const dot =
          out[3 * w] * out[3 * u] +
          out[3 * w + 1] * out[3 * u + 1] +
          out[3 * w + 2] * out[3 * u + 2];
        if (dot < 0) for (let d = 0; d < 3; ++d) out[3 * w + d] *= -1;
        stack.push(w);
      }
    }
  }
  return out;
}

/** Fans polygons of `arity` vertices into triangles. */
export function triangulate(faces: Uint32Array, arity: number): Uint32Array {
  if (arity === 3) return faces;
  const out: number[] = [];
  for (let f = 0; f + arity <= faces.length; f += arity) {
    for (let i = 1; i < arity - 1; ++i) {
      out.push(faces[f], faces[f + i], faces[f + i + 1]);
    }
  }
  return Uint32Array.from(out);
}

/**
 * Packs a chunk key into a 32-bit word, so synthesised point ids are
 * distinct across chunks: 10 bits per coordinate in [-512, 511], else a hash.
 */
export function chunkKeyWord(chunkKey: string): number {
  const parts = chunkKey.split(".").map(Number);
  if (
    parts.length <= 3 &&
    parts.every((c) => Number.isInteger(c) && c >= -512 && c < 512)
  ) {
    return parts.reduce((word, c) => (word << 10) | (c + 512), 0) >>> 0;
  }
  let hash = 0x811c9dc5;
  for (let i = 0; i < chunkKey.length; ++i) {
    hash = Math.imul(hash ^ chunkKey.charCodeAt(i), 0x01000193);
  }
  return hash >>> 0;
}

export interface DecodeChunkOptions {
  /**
   * Children whose parent lies in another chunk (linked skeleton layout).
   * May still be loading: it is needed only once the cells are read.
   */
  relinkedChildren?: ReadonlySet<number> | Promise<ReadonlySet<number>>;
  /** Skip the per-fragment id column (per-object reads know the object). */
  skipSegmentIds?: boolean;
  /**
   * Which per-fragment id column to read: `fragment_attributes/segment_id`
   * (the default) or `fragment_attributes/object_id` (always a dense row).
   */
  fragmentIdColumn?: "segment_id" | "object_id";
  /** Skip a mesh's faces (the dense overview draws its vertices only). */
  skipFaces?: boolean;
  /** Skip per-vertex tangents (a caller that assembles objects makes its own). */
  skipTangents?: boolean;
  /** Skip vertex attributes (mesh nodes carry positions only). */
  skipAttributes?: boolean;
  /** The `vertices` cell, when the caller has it (or its read) already. */
  vertices?: Promise<Uint8Array | undefined>;
}

/** Decodes the chunk at `chunkKey`, or `undefined` if it is empty. */
export async function decodeChunk(
  cells: LevelCells,
  description: ZarrVectorsGeometryDescription,
  chunkKey: string,
  signal: AbortSignal,
  options: DecodeChunkOptions = {},
): Promise<DecodedChunk | undefined> {
  const { primitive, tangent } = KIND_CAPABILITIES[description.geometryKind];
  const { linksConvention, linkWidth, attributes } = description;
  const read = (path: string) => cells.readCell(path, chunkKey, signal);
  const linksPath =
    primitive !== "points" &&
    linksConvention !== "implicit_sequential" &&
    !(primitive === "triangles" && options.skipFaces)
      ? intraLinksPath(linkWidth)
      : undefined;
  // Every array is requested in one wave; nothing's address depends on another.
  const reads = {
    vertices: options.vertices ?? read("vertices"),
    fragments: read("vertex_fragments"),
    links: linksPath === undefined ? undefined : read(linksPath),
    attributes: options.skipAttributes
      ? Promise.resolve([])
      : Promise.all(attributes.map((a) => read(`vertex_attributes/${a.name}`))),
    fragmentIds:
      description.hasObjects && !options.skipSegmentIds
        ? read(
            `fragment_attributes/${options.fragmentIdColumn ?? "segment_id"}`,
          )
        : undefined,
  };
  for (const p of Object.values(reads)) p?.catch(() => {});

  const vertexBytes = await reads.vertices;
  if (vertexBytes === undefined || vertexBytes.byteLength === 0)
    return undefined;
  const vertexType =
    (await cells.reader("vertices"))!.array.attributes.dtype ?? "float32";
  const numVertices =
    vertexBytes.byteLength / (3 * ELEMENT_BYTES[vertexType as ElementType]);
  if (!Number.isInteger(numVertices)) {
    throw new Error(
      `vertices/${chunkKey} is not a whole number of ${vertexType} xyz rows`,
    );
  }
  const positions = decodeFloat32(vertexBytes, vertexType, numVertices * 3);
  const fragmentBytes = await reads.fragments;
  if (fragmentBytes === undefined) {
    throw new Error(`chunk ${chunkKey} has vertices but no vertex_fragments`);
  }
  const fragments = decodeFragments(fragmentBytes);

  let records: Uint32Array = new Uint32Array(0);
  if (linksPath !== undefined) {
    const bytes = await reads.links;
    if (bytes !== undefined && bytes.byteLength > 0) {
      const json = (await cells.reader(linksPath))!.array.attributes;
      const type = (json.dtype ?? "int64") as ElementType;
      const width = primitive === "triangles" ? linkWidth : 2;
      const count = bytes.byteLength / ELEMENT_BYTES[type];
      if (count % width !== 0) {
        throw new Error(
          `${linksPath}/${chunkKey} is not a whole number of records`,
        );
      }
      records = decodeIndices(bytes, type, count);
    }
  }

  let edges: Uint32Array = new Uint32Array(0);
  let faces: Uint32Array = new Uint32Array(0);
  if (primitive === "triangles") {
    faces = triangulate(records, linkWidth);
  } else if (primitive === "lines") {
    edges =
      linksConvention === "implicit_sequential"
        ? sequentialEdges(fragments)
        : linksConvention === "implicit_sequential_with_branches"
          ? branchedEdges(fragments, records, await options.relinkedChildren)
          : records;
  }
  const tangents = options.skipTangents
    ? undefined
    : tangent === "walk"
      ? walkTangents(positions, fragments)
      : tangent === "edges"
        ? edgeTangents(positions, edges)
        : undefined;

  const attributeBytes = await reads.attributes;
  const decodedAttributes = await Promise.all(
    (options.skipAttributes ? [] : attributes).map(async (a, i) => {
      const bytes = attributeBytes[i];
      const count = numVertices * a.components;
      const path = `vertex_attributes/${a.name}`;
      if (bytes === undefined) {
        // Not stored here: unknown, not zero (0 can be a real value or code).
        if ((await cells.reader(path)) === undefined) {
          warnOnce(
            `${cells.levelPath} has no vertex attribute ${a.name}; it reads ` +
              "as NaN there",
          );
        }
        return new Float32Array(count).fill(NaN);
      }
      // Each level's own element type; a level may store it differently.
      const json = (await cells.reader(path))?.array.attributes;
      const type = (json ? attributeDtype({ attributes: json }) : a.dtype) as
        | ElementType
        | string;
      const width = isElementType(type)
        ? bytes.byteLength / (numVertices * ELEMENT_BYTES[type])
        : NaN;
      if (width !== a.components) {
        warnOnce(
          `${cells.levelPath}/${path} stores ${type} x${width} per vertex, ` +
            `not x${a.components}; it reads as NaN there`,
        );
        return new Float32Array(count).fill(NaN);
      }
      return decodeFloat32(bytes, type as ElementType, count);
    }),
  );

  let segmentIds: Uint32Array | undefined;
  let fragmentIds: BigUint64Array | undefined;
  const idBytes = await reads.fragmentIds;
  if (idBytes !== undefined) {
    const column = `fragment_attributes/${options.fragmentIdColumn ?? "segment_id"}`;
    const json = (await cells.reader(column))?.array.attributes;
    const type = json?.dtype ?? "uint64";
    const count = isElementType(type)
      ? idBytes.byteLength / ELEMENT_BYTES[type]
      : NaN;
    // One value per fragment; a short column (an edit added fragments
    // without extending it) is left to the manifests.
    if (Number.isInteger(count) && count >= fragments.numFragments) {
      fragmentIds = decodeUint64(idBytes, type as ElementType, count).subarray(
        0,
        fragments.numFragments,
      );
    } else {
      warnOnce(
        `${cells.levelPath}/${column}: ${chunkKey} does not hold one ` +
          `${type} per fragment; objects there are found from the manifests`,
      );
    }
  } else if (!description.hasObjects) {
    // No objects. A point is its own segment, so a pick selects one point; a
    // curve or surface fragment is one, at least within its chunk.
    segmentIds = new Uint32Array(numVertices * 2);
    const word = chunkKeyWord(chunkKey);
    if (primitive === "points") {
      for (let v = 0; v < numVertices; ++v) {
        segmentIds[2 * v] = v;
        segmentIds[2 * v + 1] = word;
      }
    } else {
      for (let f = 0; f < fragments.numFragments; ++f) {
        forEachFragmentVertex(fragments, f, (v) => {
          segmentIds![2 * v] = f;
          segmentIds![2 * v + 1] = word;
        });
      }
    }
  }

  return {
    numVertices,
    positions,
    fragments,
    edges,
    faces,
    tangents,
    attributes: decodedAttributes,
    segmentIds,
    fragmentIds,
  };
}
