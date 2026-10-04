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
 * @file A mesh store's pyramid as Neuroglancer's multiscale mesh octree.
 *
 * Neuroglancer draws each region of an object at the coarsest level of
 * detail fine enough for the view. Level of detail `lod` has cells of
 * `2 ** lod` base chunks, so a pyramid maps onto it while each level's chunks
 * are twice the previous level's: what zarr-vectors-tools builds with
 * `zvtools pyramid --chunk-scale 2,2`. An octree node at `lod` is one chunk of
 * that level; its fragment is the object's faces stored in that chunk.
 */

import type { ZarrVectorsLevel } from "#src/datasource/zarr-vectors/store.js";
import { getOctreeChildIndex, zorder3LessThan } from "#src/util/zorder.js";

/**
 * The levels to use as levels of detail: level 0, then each next level while
 * its chunks are exactly twice the last's and it stores faces.
 */
export async function meshLevels(
  levels: readonly ZarrVectorsLevel[],
  hasFaces: (level: ZarrVectorsLevel) => Promise<boolean>,
): Promise<{ levels: ZarrVectorsLevel[]; unused: number }> {
  const out = levels.slice(0, 1);
  for (let i = 1; i < levels.length; ++i) {
    const prev = out[out.length - 1].chunkShape;
    const doubles = levels[i].chunkShape.every(
      (c, d) => Math.abs(c - 2 * prev[d]) <= 1e-6 * c,
    );
    if (!doubles || !(await hasFaces(levels[i]))) break;
    out.push(levels[i]);
  }
  return { levels: out, unused: levels.length - out.length };
}

/**
 * Base-chunk offset per axis that makes every octree coordinate
 * non-negative while keeping each level's chunks aligned with its parents.
 */
export function meshGridOffset(
  lowerBounds: readonly number[],
  baseChunkShape: readonly number[],
  numLevels: number,
): number[] {
  const step = 2 ** (numLevels - 1);
  return lowerBounds.map((lower, d) => {
    const first = Math.floor(lower / baseChunkShape[d]);
    return Math.max(0, Math.ceil(-first / step)) * step;
  });
}

const EMPTY = 0x80000000;

function zorderCompare(a: readonly number[], b: readonly number[]) {
  if (zorder3LessThan(a[0], a[1], a[2], b[0], b[1], b[2])) return -1;
  if (zorder3LessThan(b[0], b[1], b[2], a[0], a[1], a[2])) return 1;
  return 0;
}

/**
 * The octree of one object from the grid coordinates (non-negative) of the
 * nodes holding its faces at each level of detail. Rows are
 * `[x, y, z, firstChild, childEnd | EMPTY]`, each level in z-order; parents
 * of every node are added (empty where the object has nothing there), and
 * empty levels above the last stored one lead up to a single root.
 */
export function buildMeshOctree(nodesByLod: readonly (readonly number[][])[]): {
  octree: Uint32Array;
  numLods: number;
} {
  const rows: number[] = [];
  let prevStart = 0;
  let prevEnd = 0;
  for (let lod = 0; ; ++lod) {
    const data = new Set((nodesByLod[lod] ?? []).map((c) => c.join()));
    const nodes = new Map<string, number[]>();
    for (const c of nodesByLod[lod] ?? []) nodes.set(c.join(), [...c]);
    for (let row = prevStart; row < prevEnd; ++row) {
      const parent = [0, 1, 2].map((d) => rows[row * 5 + d] >>> 1);
      nodes.set(parent.join(), parent);
    }
    const sorted = [...nodes.values()].sort(zorderCompare);
    const start = rows.length / 5;
    let child = prevStart;
    for (const node of sorted) {
      const firstChild = child;
      while (
        child < prevEnd &&
        [0, 1, 2].every((d) => rows[child * 5 + d] >>> 1 === node[d])
      ) {
        ++child;
      }
      rows.push(
        node[0],
        node[1],
        node[2],
        firstChild,
        (child | (data.has(node.join()) ? 0 : EMPTY)) >>> 0,
      );
    }
    prevStart = start;
    prevEnd = rows.length / 5;
    const stored = lod + 1 >= nodesByLod.length;
    if (stored && prevEnd - prevStart <= 1) {
      if (prevEnd === 0) {
        // Nothing anywhere: a lone empty root.
        return { octree: Uint32Array.of(0, 0, 0, 0, EMPTY), numLods: 1 };
      }
      return { octree: Uint32Array.from(rows), numLods: lod + 1 };
    }
  }
}

/** What one object's multiscale mesh manifest needs. */
export interface ObjectMeshLayout {
  octree: Uint32Array;
  lodScales: Float32Array;
  /** Base-chunk offset of this object's octree grid from the stored one. */
  gridOffset: number[];
  chunkGridSpatialOrigin: number[];
  clipLowerBound: number[];
  clipUpperBound: number[];
}

/**
 * One object's octree and level-of-detail scales, from the stored chunks it
 * has geometry in at each level of detail (finest first).
 *
 * - An object a coarser level dropped stops at the last level that has it:
 *   an empty level of detail would be drawn as nothing when zoomed out.
 *   Levels above are empty parents with scale 0, which Neuroglancer passes
 *   through.
 * - Grid coordinates must be non-negative and each level's chunks must nest
 *   in its parents': `gridOffset` (from the store's bounds) is raised, in
 *   steps that keep every level aligned, until this object's chunks fit.
 */
export function objectMeshLayout(
  chunksPerLevel: readonly (readonly number[][])[],
  baseChunkShape: readonly number[],
  gridOffset: readonly number[],
  scales: Float32Array,
): ObjectMeshLayout {
  const missing = chunksPerLevel.findIndex((c) => c.length === 0);
  const used = missing > 0 ? chunksPerLevel.slice(0, missing) : chunksPerLevel;
  const step = 2 ** (chunksPerLevel.length - 1);
  const offset = gridOffset.map((o, d) => {
    let need = o;
    used.forEach((coords, lod) => {
      for (const c of coords) need = Math.max(need, -c[d] * 2 ** lod);
    });
    return Math.ceil(need / step) * step;
  });
  const origin = baseChunkShape.map((c, d) => -offset[d] * c);
  const nodes = used.map((coords, lod) =>
    coords.map((c) => c.map((x, d) => x + offset[d] / 2 ** lod)),
  );
  const { octree, numLods } = buildMeshOctree(nodes);
  const lodScales = new Float32Array(numLods);
  lodScales.set(scales.subarray(0, Math.min(numLods, used.length)));
  const lower = [Infinity, Infinity, Infinity];
  const upper = [-Infinity, -Infinity, -Infinity];
  nodes.forEach((coords, lod) => {
    for (const c of coords) {
      for (let d = 0; d < 3; ++d) {
        const size = 2 ** lod * baseChunkShape[d];
        lower[d] = Math.min(lower[d], c[d] * size + origin[d]);
        upper[d] = Math.max(upper[d], (c[d] + 1) * size + origin[d]);
      }
    }
  });
  if (!Number.isFinite(lower[0])) (lower.fill(0), upper.fill(0));
  return {
    octree,
    lodScales,
    gridOffset: offset,
    chunkGridSpatialOrigin: origin,
    clipLowerBound: lower,
    clipUpperBound: upper,
  };
}

/**
 * Orders a fragment's triangles by the octant of the node they fall in, as
 * Neuroglancer draws a coarse node's octants only where finer nodes are not
 * drawn. A triangle that crosses a mid-plane is clipped into one piece per
 * octant it covers -- the precomputed format does the same when it is
 * written -- so drawing some octants coarse and others fine leaves no strip
 * of the coarse surface out and draws none of it twice. Clipped corners are
 * shared between the pieces either side of a plane, so normals stay smooth.
 * Level-of-detail-0 fragments have a single part.
 */
export function partitionMeshFragment(
  positions: Float32Array,
  indices: Uint32Array,
  nodeLower: readonly number[],
  nodeSize: readonly number[],
  octants: boolean,
): {
  vertexPositions: Float32Array;
  indices: Uint32Array;
  subChunkOffsets: Uint32Array;
} {
  if (!octants) {
    return {
      vertexPositions: positions,
      indices,
      subChunkOffsets: Uint32Array.of(0, indices.length),
    };
  }
  const mid = [0, 1, 2].map((d) => nodeLower[d] + nodeSize[d] / 2);
  const numVertices = positions.length / 3;
  const added: number[] = [];
  const coord = (v: number, d: number) =>
    v < numVertices ? positions[3 * v + d] : added[3 * (v - numVertices) + d];
  const side = (v: number, d: number) => (coord(v, d) >= mid[d] ? 1 : 0);
  // Where edge (u, v) meets plane `d`: one vertex per edge and plane, so the
  // triangles either side of an edge share it.
  const cuts = new Map<string, number>();
  const cut = (u: number, v: number, d: number) => {
    const a = Math.min(u, v);
    const b = Math.max(u, v);
    const key = `${a},${b},${d}`;
    let out = cuts.get(key);
    if (out === undefined) {
      const t = (mid[d] - coord(a, d)) / (coord(b, d) - coord(a, d));
      out = numVertices + added.length / 3;
      for (let k = 0; k < 3; ++k) {
        const pa = coord(a, k);
        added.push(k === d ? mid[d] : pa + t * (coord(b, k) - pa));
      }
      cuts.set(key, out);
    }
    return out;
  };
  // The part of a convex polygon on one side of plane `d` (Sutherland-
  // Hodgman). A corner exactly on the plane is on side 1 and is its own cut.
  const clip = (polygon: number[], d: number, keep: number) => {
    const out: number[] = [];
    for (let i = 0; i < polygon.length; ++i) {
      const u = polygon[i];
      const v = polygon[(i + 1) % polygon.length];
      const su = side(u, d);
      if (su === keep) out.push(u);
      if (su !== side(v, d)) {
        out.push(
          coord(u, d) === mid[d]
            ? u
            : coord(v, d) === mid[d]
              ? v
              : cut(u, v, d),
        );
      }
    }
    const deduped = out.filter((v, i) => v !== out[(i + 1) % out.length]);
    return deduped;
  };

  const numTriangles = indices.length / 3;
  const octantOf = new Int8Array(numTriangles);
  const counts = new Uint32Array(8);
  const pieces: number[] = [];
  const pieceOctants: number[] = [];
  for (let t = 0; t < numTriangles; ++t) {
    const a = indices[3 * t];
    const b = indices[3 * t + 1];
    const c = indices[3 * t + 2];
    let crosses = false;
    const bits = [0, 0, 0];
    for (let d = 0; d < 3; ++d) {
      bits[d] = side(a, d);
      if (side(b, d) !== bits[d] || side(c, d) !== bits[d]) crosses = true;
    }
    if (!crosses) {
      const octant = getOctreeChildIndex(bits[0], bits[1], bits[2]);
      octantOf[t] = octant;
      ++counts[octant];
      continue;
    }
    octantOf[t] = -1;
    let parts: { polygon: number[]; bits: number[] }[] = [
      { polygon: [a, b, c], bits: [0, 0, 0] },
    ];
    for (let d = 0; d < 3; ++d) {
      const next: typeof parts = [];
      for (const { polygon, bits } of parts) {
        for (const keep of [0, 1]) {
          const piece = clip(polygon, d, keep);
          if (piece.length < 3) continue;
          const pieceBits = bits.slice();
          pieceBits[d] = keep;
          next.push({ polygon: piece, bits: pieceBits });
        }
      }
      parts = next;
    }
    for (const { polygon, bits } of parts) {
      const octant = getOctreeChildIndex(bits[0], bits[1], bits[2]);
      for (let k = 1; k + 1 < polygon.length; ++k) {
        pieces.push(polygon[0], polygon[k], polygon[k + 1]);
        pieceOctants.push(octant);
        ++counts[octant];
      }
    }
  }
  const subChunkOffsets = new Uint32Array(9);
  for (let o = 0; o < 8; ++o) {
    subChunkOffsets[o + 1] = subChunkOffsets[o] + 3 * counts[o];
  }
  const next = subChunkOffsets.slice(0, 8);
  const out = new Uint32Array(subChunkOffsets[8]);
  for (let t = 0; t < numTriangles; ++t) {
    const octant = octantOf[t];
    if (octant < 0) continue;
    out.set(indices.subarray(3 * t, 3 * t + 3), next[octant]);
    next[octant] += 3;
  }
  for (let p = 0; p < pieceOctants.length; ++p) {
    const octant = pieceOctants[p];
    out.set(pieces.slice(3 * p, 3 * p + 3), next[octant]);
    next[octant] += 3;
  }
  let vertexPositions = positions;
  if (added.length > 0) {
    vertexPositions = new Float32Array(positions.length + added.length);
    vertexPositions.set(positions);
    vertexPositions.set(added, positions.length);
  }
  return { vertexPositions, indices: out, subChunkOffsets };
}
