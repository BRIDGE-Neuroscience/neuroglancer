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
  decodeFragments,
  FRAGMENT_INDEX_MAGIC,
  FRAGMENT_INDEX_VERSION,
} from "#src/datasource/zarr-vectors/fragment_index.js";
import {
  buildMeshOctree,
  meshGridOffset,
  meshLevels,
  objectMeshLayout,
  partitionMeshFragment,
  unionMeshNodes,
} from "#src/datasource/zarr-vectors/mesh_lod.js";
import {
  mergeSkeletons,
  ObjectReader,
} from "#src/datasource/zarr-vectors/object_reader.js";
import { openZarrVectorsStore } from "#src/datasource/zarr-vectors/store.js";
import {
  fixtureExpected,
  fixtureListDirectories,
  fixtureRead,
} from "#src/datasource/zarr-vectors/test_fixtures.js";
import { ShardIndexCache } from "#src/datasource/zarr-vectors/zarr_array.js";
import { validateOctree } from "#src/mesh/multiscale.js";

const signal = new AbortController().signal;

/** A fragment index with one range `[0, rows)` (fragment_index_v1). */
function singleRangeIndex(rows: number) {
  const out = new Uint8Array(16 + 8 + 16 + 4);
  const view = new DataView(out.buffer);
  view.setUint32(0, FRAGMENT_INDEX_MAGIC, true);
  view.setUint16(4, FRAGMENT_INDEX_VERSION, true);
  view.setUint32(8, 1, true); // fragments
  view.setUint32(12, 1, true); // ranges
  out[16] = 1; // fragment 0 is a range
  view.setBigInt64(24, 0n, true);
  view.setBigInt64(32, BigInt(rows), true);
  return out;
}

/** A single-element vlen-bytes cell around `payload`. */
function wrapVlenCell(payload: Uint8Array) {
  const out = new Uint8Array(8 + payload.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, 1, true);
  view.setUint32(4, payload.length, true);
  out.set(payload, 8);
  return out;
}

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

describe("objectMeshLayout", () => {
  const scales = Float32Array.of(1, 2, 4);

  it("stops an object at the first level that dropped it", () => {
    // Neuroglancer draws an empty level of detail as nothing and does not
    // look below it, so the object would vanish when zoomed out.
    for (const chunks of [
      [
        [
          [0, 0, 0],
          [1, 0, 0],
        ],
        [[0, 0, 0]],
        [],
      ],
      [
        [
          [0, 0, 0],
          [1, 0, 0],
        ],
        [],
        [[0, 0, 0]],
      ],
    ]) {
      const layout = objectMeshLayout(chunks, [16, 16, 16], [0, 0, 0], scales);
      validate(layout.octree);
      const used = chunks.findIndex((c) => c.length === 0);
      expect(Array.from(layout.lodScales.subarray(0, used))).toEqual(
        Array.from(scales.subarray(0, used)),
      );
      for (let lod = used; lod < layout.lodScales.length; ++lod) {
        expect(layout.lodScales[lod], `lod ${lod}`).toBe(0);
      }
    }
  });

  it("raises the grid offset for chunks below the store's bounds", () => {
    // Stale bounds: a chunk at x = -1 where the bounds start at 0.
    const layout = objectMeshLayout(
      [
        [
          [-1, 0, 0],
          [0, 0, 0],
        ],
        [[-1, 0, 0]],
      ],
      [16, 16, 16],
      [0, 0, 0],
      scales.subarray(0, 2),
    );
    validate(layout.octree);
    expect(layout.gridOffset).toEqual([2, 0, 0]);
    expect(layout.chunkGridSpatialOrigin).toEqual([-32, -0, -0]);
    expect(layout.clipLowerBound).toEqual([-32, 0, 0]);
    expect(rowsOf(layout.octree).length).toBeLessThan(8);
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

  /** Sum of the triangles' oriented area vectors (cross products / 2). */
  function areaVector(positions: Float32Array, indices: Uint32Array) {
    const sum = [0, 0, 0];
    for (let t = 0; t < indices.length; t += 3) {
      const [a, b, c] = [0, 1, 2].map((k) =>
        Array.from(
          positions.subarray(3 * indices[t + k], 3 * indices[t + k] + 3),
        ),
      );
      const u = [0, 1, 2].map((d) => b[d] - a[d]);
      const v = [0, 1, 2].map((d) => c[d] - a[d]);
      sum[0] += (u[1] * v[2] - u[2] * v[1]) / 2;
      sum[1] += (u[2] * v[0] - u[0] * v[2]) / 2;
      sum[2] += (u[0] * v[1] - u[1] * v[0]) / 2;
    }
    return sum;
  }

  it("clips triangles that cross a mid-plane into their octants", () => {
    let seed = 7;
    const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const positions: number[] = [];
    const indices: number[] = [];
    for (let t = 0; t < 300; ++t) {
      // Corners on a coarse grid so some land exactly on a mid-plane.
      for (let k = 0; k < 3; ++k) {
        for (let d = 0; d < 3; ++d)
          positions.push(Math.round(random() * 8) / 8);
      }
      indices.push(3 * t, 3 * t + 1, 3 * t + 2);
    }
    const p = Float32Array.from(positions);
    const i = Uint32Array.from(indices);
    const out = partitionMeshFragment(p, i, [0, 0, 0], [1, 1, 1], true);
    // Nothing gained or lost, and every face keeps its orientation.
    const before = areaVector(p, i);
    const after = areaVector(out.vertexPositions, out.indices);
    for (let d = 0; d < 3; ++d) expect(after[d]).toBeCloseTo(before[d], 4);
    let unsigned = 0;
    for (let t = 0; t < i.length; t += 3) {
      unsigned += Math.hypot(...areaVector(p, i.subarray(t, t + 3)));
    }
    let unsignedAfter = 0;
    for (let t = 0; t < out.indices.length; t += 3) {
      unsignedAfter += Math.hypot(
        ...areaVector(out.vertexPositions, out.indices.subarray(t, t + 3)),
      );
    }
    expect(unsignedAfter).toBeCloseTo(unsigned, 4);
    // Every piece lies in its own octant.
    for (let o = 0; o < 8; ++o) {
      const bits = [o & 1, (o >> 1) & 1, (o >> 2) & 1];
      for (
        let at = out.subChunkOffsets[o];
        at < out.subChunkOffsets[o + 1];
        ++at
      ) {
        const v = out.indices[at];
        for (let d = 0; d < 3; ++d) {
          const x = out.vertexPositions[3 * v + d];
          if (bits[d]) expect(x).toBeGreaterThanOrEqual(0.5 - 1e-6);
          else expect(x).toBeLessThanOrEqual(0.5 + 1e-6);
        }
      }
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

  it("does not use a level that adds to the next as a level of detail", async () => {
    const all = async () => true;
    const levels = [level(16), { ...level(32), refinement: "add" }, level(64)];
    expect((await meshLevels(levels, all)).levels.length).toBe(1);
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

  /** The fixture as seen through `edit`, which may rewrite any whole-file read. */
  function edited(
    edit: (path: string, bytes: Uint8Array) => Uint8Array | undefined,
  ) {
    const base = access();
    const counter = { bytes: 0 };
    return {
      counter,
      access: {
        ...base,
        read: async (path: string, options: any) => {
          const bytes = await base.read(path, options);
          // Cell data only: metadata is read once per level by either path.
          if (path.includes("/c/")) counter.bytes += bytes?.byteLength ?? 0;
          return bytes === undefined || options?.byteRange !== undefined
            ? bytes
            : edit(path, bytes);
        },
      },
    };
  }
  /** Each level's `zarr.json`, without its `fragment_link_groups` stamp. */
  const unstamped = (path: string, bytes: Uint8Array) => {
    if (!/^\d+\/zarr\.json$/.test(path)) return bytes;
    const json = JSON.parse(new TextDecoder().decode(bytes));
    delete json.attributes.zarr_vectors_level.fragment_link_groups;
    return new TextEncoder().encode(JSON.stringify(json));
  };

  async function surfaces(view: ReturnType<typeof edited>) {
    const store = await openZarrVectorsStore(view.access, undefined);
    const out: string[][] = [];
    for (const level of store.levels) {
      const reader = new ObjectReader(view.access, store.description, level);
      for (const object of [0, 1]) {
        out.push((await facesAt(reader, object)).faces.sort());
      }
    }
    return out;
  }

  it("is stamped one face group per object at every level", async () => {
    const store = await openZarrVectorsStore(access(), undefined);
    expect(store.levels.map((level) => level.fragmentLinkGroups)).toEqual([
      true,
      true,
      true,
    ]);
    const unstampedStore = await openZarrVectorsStore(
      edited(unstamped).access,
      undefined,
    );
    expect(
      unstampedStore.levels.map((level) => level.fragmentLinkGroups),
    ).toEqual([false, false, false]);
  });

  it("gives the same surfaces whether it reads rows or whole cells", async () => {
    expect(await surfaces(edited((_, b) => b))).toEqual(
      await surfaces(edited(unstamped)),
    );
  });

  it("reads only its own rows of a chunk another object fills", async () => {
    /** Bytes read to fetch `object`'s node in `chunk` of level `index`. */
    async function cost(
      view: ReturnType<typeof edited>,
      index: number,
      object: bigint,
      chunk: number[],
    ) {
      const store = await openZarrVectorsStore(view.access, undefined);
      const reader = new ObjectReader(
        view.access,
        store.description,
        store.levels[index],
      );
      await reader.chunksOf(object);
      view.counter.bytes = 0;
      const node = await reader.readMeshNode(object, chunk, signal);
      return { bytes: view.counter.bytes, faces: node.indices.length / 3 };
    }
    let compared = 0;
    const store = await openZarrVectorsStore(access(), undefined);
    for (const level of store.levels) {
      const reader = new ObjectReader(access(), store.description, level);
      const of1 = new Set((await reader.chunksOf(1n)).map((c) => c.join()));
      for (const chunk of await reader.chunksOf(0n)) {
        if (!of1.has(chunk.join())) continue;
        // The object holding less of the chunk is where rows pay off.
        const costs = await Promise.all(
          [0n, 1n].map(async (object) => ({
            rows: await cost(
              edited((_, b) => b),
              level.index,
              object,
              chunk,
            ),
            whole: await cost(edited(unstamped), level.index, object, chunk),
          })),
        );
        const minor =
          costs[0].rows.bytes < costs[1].rows.bytes ? costs[0] : costs[1];
        for (const c of costs) expect(c.rows.faces).toBe(c.whole.faces);
        expect(
          minor.rows.bytes,
          `level ${level.path} chunk ${chunk}`,
        ).toBeLessThan(minor.whole.bytes);
        ++compared;
      }
    }
    expect(compared).toBeGreaterThan(0);
  });

  it("falls back to whole cells when the stamp does not hold", async () => {
    const want = await surfaces(edited(unstamped));
    const sidecar = /\/link_fragments\/c\//;
    // Garbage where the face groups should be.
    const garbage = edited((path, bytes) =>
      sidecar.test(path) ? bytes.slice().reverse() : bytes,
    );
    expect(await surfaces(garbage)).toEqual(want);
    // Valid, but one group for the whole chunk: wrong wherever two objects
    // share a chunk.
    const oneGroup = edited((path, bytes) => {
      if (!sidecar.test(path)) return bytes;
      const real = decodeFragments(bytes.subarray(8));
      let rows = 0;
      for (let f = 0; f < real.numFragments; ++f) rows += real.range(f).count;
      return wrapVlenCell(singleRangeIndex(rows));
    });
    expect(await surfaces(oneGroup)).toEqual(want);
    // No face groups at all.
    const missing = edited((path, bytes) =>
      sidecar.test(path) ? undefined : bytes,
    );
    expect(await surfaces(missing)).toEqual(want);
  });

  it("measures a level's edge length from a sample of its faces", async () => {
    const store = await openZarrVectorsStore(access(), undefined);
    for (const level of store.levels) {
      const sampled = await new ObjectReader(
        access(),
        store.description,
        level,
      ).meanEdgeLength();
      // Compare with object 0's own faces in one of its chunks: the same
      // surfaces at the same resolution.
      const reader = new ObjectReader(access(), store.description, level);
      const chunk = (await reader.chunksOf(0n))[0];
      expect(sampled, `level ${level.path}`).toBeGreaterThan(0);
      const { positions: p, indices: f } = await reader.readMeshNode(
        0n,
        chunk,
        signal,
      );
      let total = 0;
      for (let t = 0; t < f.length; t += 3) {
        for (let k = 0; k < 3; ++k) {
          const a = f[t + k];
          const b = f[t + ((k + 1) % 3)];
          total += Math.hypot(
            p[3 * a] - p[3 * b],
            p[3 * a + 1] - p[3 * b + 1],
            p[3 * a + 2] - p[3 * b + 2],
          );
        }
      }
      const object0 = total / f.length;
      // Same surface, same resolution: within a factor of two.
      expect(sampled! / object0).toBeGreaterThan(0.5);
      expect(sampled! / object0).toBeLessThan(2);
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

describe("mergeSkeletons", () => {
  it("joins an object's parts from several levels", () => {
    const merged = mergeSkeletons([
      {
        positions: Float32Array.of(0, 0, 0, 1, 0, 0),
        edges: Uint32Array.of(0, 1),
        attributes: [Float32Array.of(5, 6)],
      },
      {
        positions: new Float32Array(0),
        edges: new Uint32Array(0),
        attributes: [new Float32Array(0)],
      },
      {
        positions: Float32Array.of(2, 0, 0, 3, 0, 0, 4, 0, 0),
        edges: Uint32Array.of(0, 1, 1, 2),
        attributes: [Float32Array.of(7, 8, 9)],
      },
    ]);
    expect(Array.from(merged.positions)).toEqual([
      0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0,
    ]);
    expect(Array.from(merged.edges)).toEqual([0, 1, 2, 3, 3, 4]);
    expect(Array.from(merged.attributes[0])).toEqual([5, 6, 7, 8, 9]);
  });
});

describe("an additive mesh pyramid", () => {
  const access = () => ({
    read: fixtureRead("mesh_add"),
    listDirectories: fixtureListDirectories("mesh_add"),
    shardIndexes: new ShardIndexCache(),
  });

  it("draws each object whole, from the level that stores it", async () => {
    const store = await openZarrVectorsStore(access(), undefined);
    expect(store.levels.map((l) => l.refinement)).toEqual(["add", "replace"]);
    const base = store.levels[1].chunkShape;
    const readers = store.levels.map(
      (level) => new ObjectReader(access(), store.description, level),
    );
    const levelsUsed = new Set<number>();
    for (const object of [0n, 1n]) {
      const all = await Promise.all(readers.map((r) => r.chunksOf(object)));
      const nodes = unionMeshNodes(
        all,
        store.levels.map((l) => l.chunkShape),
        base,
      );
      let faces = 0;
      const edges = new Map<string, number>();
      for (const [key, members] of nodes) {
        const node = key.split(",").map(Number);
        for (const [level, chunk] of members) {
          levelsUsed.add(level);
          // Each part lies in its node.
          const size = store.levels[level].chunkShape;
          for (let d = 0; d < 3; ++d) {
            expect(chunk[d] * size[d]).toBeGreaterThanOrEqual(
              node[d] * base[d],
            );
            expect((chunk[d] + 1) * size[d]).toBeLessThanOrEqual(
              (node[d] + 1) * base[d],
            );
          }
          const { positions, indices } = await readers[level].readMeshNode(
            object,
            chunk,
            signal,
          );
          faces += indices.length / 3;
          for (let t = 0; t < indices.length; t += 3) {
            for (const [u, v] of [
              [indices[t], indices[t + 1]],
              [indices[t + 1], indices[t + 2]],
              [indices[t + 2], indices[t]],
            ]) {
              const e = [key3(positions, u), key3(positions, v)]
                .sort()
                .join("|");
              edges.set(e, (edges.get(e) ?? 0) + 1);
            }
          }
        }
      }
      // A whole closed surface: every edge shared by exactly two faces.
      expect(faces, `object ${object}`).toBeGreaterThan(0);
      for (const count of edges.values()) expect(count).toBe(2);
    }
    // One object at each level.
    expect([...levelsUsed].sort()).toEqual([0, 1]);
  });
});

function key3(p: ArrayLike<number>, v: number) {
  return [0, 1, 2].map((d) => p[3 * v + d].toFixed(3)).join(",");
}
