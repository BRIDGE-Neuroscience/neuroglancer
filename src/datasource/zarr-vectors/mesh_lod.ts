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

/**
 * Orders a fragment's triangles by the octant of the node they fall in (by
 * centroid), as Neuroglancer draws a coarse node's octants only where finer
 * nodes are not drawn. Level-of-detail-0 fragments have a single part.
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
  const numTriangles = indices.length / 3;
  const octantOf = new Uint8Array(numTriangles);
  const counts = new Uint32Array(8);
  for (let t = 0; t < numTriangles; ++t) {
    const bits = [0, 1, 2].map((d) => {
      let sum = 0;
      for (let k = 0; k < 3; ++k) sum += positions[3 * indices[3 * t + k] + d];
      return sum / 3 >= nodeLower[d] + nodeSize[d] / 2 ? 1 : 0;
    });
    const octant = getOctreeChildIndex(bits[0], bits[1], bits[2]);
    octantOf[t] = octant;
    ++counts[octant];
  }
  const subChunkOffsets = new Uint32Array(9);
  for (let o = 0; o < 8; ++o) {
    subChunkOffsets[o + 1] = subChunkOffsets[o] + 3 * counts[o];
  }
  const next = subChunkOffsets.slice(0, 8);
  const out = new Uint32Array(indices.length);
  for (let t = 0; t < numTriangles; ++t) {
    const at = next[octantOf[t]];
    out.set(indices.subarray(3 * t, 3 * t + 3), at);
    next[octantOf[t]] = at + 3;
  }
  return { vertexPositions: positions, indices: out, subChunkOffsets };
}
