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

import { describe, expect, it } from "vitest";
import {
  buildMeshOctree,
  meshGridOffset,
  meshLevels,
  partitionMeshFragment,
} from "#src/datasource/zarr-vectors/mesh_lod.js";
import { ObjectReader } from "#src/datasource/zarr-vectors/object_reader.js";
import { openZarrVectorsStore } from "#src/datasource/zarr-vectors/store.js";
import {
  fixtureExpected,
  fixtureListDirectories,
  fixtureRead,
} from "#src/datasource/zarr-vectors/test_fixtures.js";
import { ShardIndexCache } from "#src/datasource/zarr-vectors/zarr_array.js";
import { validateOctree } from "#src/mesh/multiscale.js";

const signal = new AbortController().signal;

/** `validateOctree` without the empty flag (bit 31), which it does not mask. */
function validate(octree: Uint32Array) {
  const copy = octree.slice();
  for (let r = 0; r < copy.length / 5; ++r) copy[5 * r + 4] &= 0x7fffffff;
  validateOctree(copy);
}

function rowsOf(octree: Uint32Array) {
  return Array.from({ length: octree.length / 5 }, (_, r) =>
    Array.from(octree.subarray(5 * r, 5 * r + 5)),
  );
}

describe("buildMeshOctree", () => {
  it("adds parents and climbs to a single root", () => {
    const { octree, numLods } = buildMeshOctree([
      [
        [0, 0, 0],
        [3, 1, 0],
      ],
      [[1, 0, 0]],
    ]);
    validate(octree);
    const rows = rowsOf(octree);
    // lod 1 holds the stored node and [0,0,0] (parent of lod 0's first node,
    // empty); lod 2 is the empty root above them.
    expect(numLods).toBe(3);
    expect(rows.length).toBe(5);
    const root = rows[rows.length - 1];
    expect(root.slice(0, 3)).toEqual([0, 0, 0]);
    expect(root[4] >>> 31).toBe(1);
    const lod1 = rows.slice(2, 4);
    expect(lod1.map((r) => [r.slice(0, 3), r[4] >>> 31])).toEqual([
      [[0, 0, 0], 1],
      [[1, 0, 0], 0],
    ]);
  });

  it("keeps each level in z-order with contiguous children", () => {
    const nodes = [[], [], []] as number[][][];
    for (let x = 0; x < 4; ++x) {
      for (let y = 0; y < 4; ++y) nodes[0].push([x, y, 1]);
    }
    nodes[1].push([1, 1, 0]);
    const { octree } = buildMeshOctree(nodes);
    validate(octree);
  });

  it("gives an object with no geometry a lone empty root", () => {
    const { octree, numLods } = buildMeshOctree([[], []]);
    expect(Array.from(octree)).toEqual([0, 0, 0, 0, 0x80000000]);
    expect(numLods).toBe(1);
  });
});

describe("partitionMeshFragment", () => {
  // One triangle near each corner of the unit box.
  const positions: number[] = [];
  const indices: number[] = [];
  for (let o = 7; o >= 0; --o) {
    const c = [o & 1, (o >> 1) & 1, (o >> 2) & 1].map((b) => 0.2 + 0.6 * b);
    const base = positions.length / 3;
    positions.push(...c, c[0] + 0.01, c[1], c[2], c[0], c[1] + 0.01, c[2]);
    indices.push(base, base + 1, base + 2);
  }
  const p = Float32Array.from(positions);
  const i = Uint32Array.from(indices);

  it("groups triangles by octant, in Neuroglancer's child order", () => {
    const out = partitionMeshFragment(p, i, [0, 0, 0], [1, 1, 1], true);
    expect(Array.from(out.subChunkOffsets)).toEqual([
      0, 3, 6, 9, 12, 15, 18, 21, 24,
    ]);
    for (let o = 0; o < 8; ++o) {
      const v = out.indices[3 * o];
      const corner = [0, 1, 2].map((d) => (p[3 * v + d] > 0.5 ? 1 : 0));
      expect(corner[0] | (corner[1] << 1) | (corner[2] << 2)).toBe(o);
    }
  });

  it("keeps level-of-detail-0 fragments whole", () => {
    const out = partitionMeshFragment(p, i, [0, 0, 0], [1, 1, 1], false);
    expect(Array.from(out.subChunkOffsets)).toEqual([0, 24]);
  });
});

describe("meshLevels and meshGridOffset", () => {
  const level = (c: number) => ({ chunkShape: [c, c, c] }) as any;

  it("stops at the first level that does not double or has no faces", async () => {
    const all = async () => true;
    const levels = [level(16), level(32), level(64), level(256)];
    expect((await meshLevels(levels, all)).levels.length).toBe(3);
    const noFaces = async (l: any) => l.chunkShape[0] !== 64;
    expect(await meshLevels(levels, noFaces)).toEqual({
      levels: levels.slice(0, 2),
      unused: 2,
    });
  });

  it("offsets negative chunks to non-negative, parent-aligned coordinates", () => {
    expect(meshGridOffset([-100, 0, 5], [16, 16, 16], 3)).toEqual([8, 0, 0]);
  });
});

describe("multi-resolution meshes from a zarr-vectors-tools pyramid", () => {
  const access = () => ({
    read: fixtureRead("mesh_lod"),
    listDirectories: fixtureListDirectories("mesh_lod"),
    shardIndexes: new ShardIndexCache(),
  });
  const key = (p: ArrayLike<number>, v: number) =>
    [0, 1, 2].map((d) => p[3 * v + d].toFixed(3)).join(",");
  const faceKey = (p: ArrayLike<number>, a: number, b: number, c: number) =>
    [key(p, a), key(p, b), key(p, c)].sort().join("|");

  /** Every face of `object` at a level, from its nodes. */
  async function facesAt(reader: ObjectReader, object: number) {
    const faces: string[] = [];
    const edges = new Map<string, number>();
    for (const chunk of await reader.chunksOf(BigInt(object))) {
      const { positions, indices } = await reader.readMeshNode(
        BigInt(object),
        chunk,
        signal,
      );
      for (let t = 0; t < indices.length; t += 3) {
        const [a, b, c] = indices.subarray(t, t + 3);
        faces.push(faceKey(positions, a, b, c));
        for (const [u, v] of [
          [a, b],
          [b, c],
          [c, a],
        ]) {
          const e = [key(positions, u), key(positions, v)].sort().join("|");
          edges.set(e, (edges.get(e) ?? 0) + 1);
        }
      }
    }
    return { faces, edges };
  }

  it("uses every level as a level of detail", async () => {
    const store = await openZarrVectorsStore(access(), undefined);
    const { levels } = await meshLevels(store.levels, async () => true);
    expect(levels.map((l) => l.chunkShape[0])).toEqual([16, 32, 64]);
  });

  it("reads the writer's surface exactly once across level 0's nodes", async () => {
    const expected = fixtureExpected().mesh_lod;
    const store = await openZarrVectorsStore(access(), undefined);
    const reader = new ObjectReader(
      access(),
      store.description,
      store.levels[0],
    );
    for (const object of [0, 1]) {
      const { faces } = await facesAt(reader, object);
      const want = (expected.faces as number[][])
        .filter(([a]) => expected.object_ids[a] === object)
        .map(([a, b, c]) =>
          faceKey((expected.positions as number[][]).flat(), a, b, c),
        );
      expect(faces.length, `object ${object}`).toBe(new Set(faces).size);
      expect(faces.sort(), `object ${object}`).toEqual(want.sort());
    }
  });

  it("keeps every coarser surface closed: no face lost or doubled across chunks", async () => {
    const store = await openZarrVectorsStore(access(), undefined);
    for (const level of store.levels.slice(1)) {
      const reader = new ObjectReader(access(), store.description, level);
      for (const object of [0, 1]) {
        const { faces, edges } = await facesAt(reader, object);
        expect(faces.length, `level ${level.path}`).toBeGreaterThan(0);
        expect(new Set(faces).size).toBe(faces.length);
        for (const count of edges.values()) expect(count).toBe(2);
      }
    }
  });

  it("builds a valid octree for each object from the level manifests", async () => {
    const store = await openZarrVectorsStore(access(), undefined);
    const offset = meshGridOffset(store.lowerBounds, [16, 16, 16], 3);
    for (const object of [0, 1]) {
      const nodes = await Promise.all(
        store.levels.map(async (level, lod) =>
          (
            await new ObjectReader(access(), store.description, level).chunksOf(
              BigInt(object),
            )
          ).map((c) => c.map((x, d) => x + offset[d] / 2 ** lod)),
        ),
      );
      const { octree, numLods } = buildMeshOctree(nodes);
      validate(octree);
      expect(numLods).toBeGreaterThanOrEqual(3);
      const data = rowsOf(octree).filter((r) => r[4] >>> 31 === 0).length;
      expect(data).toBe(nodes.flat().length);
    }
  });
});
