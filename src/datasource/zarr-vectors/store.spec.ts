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
 * End-to-end checks against stores written by zarr-vectors-py v0.9.2 (see
 * testdata/datasource/zarr-vectors/generate.py): every codec and layout must
 * decode to the geometry the writer was given.
 */

import { describe, expect, it } from "vitest";
import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import { LevelPipeline } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import { ObjectReader } from "#src/datasource/zarr-vectors/object_reader.js";
import {
  readObjectTable,
  readSegmentProperties,
} from "#src/datasource/zarr-vectors/objects.js";
import type { ZarrVectorsStore } from "#src/datasource/zarr-vectors/store.js";
import {
  openZarrVectorsStore,
  safeAttributeId,
} from "#src/datasource/zarr-vectors/store.js";
import {
  fixtureExpected,
  fixtureListDirectories,
  fixtureRead,
} from "#src/datasource/zarr-vectors/test_fixtures.js";
import { ShardIndexCache } from "#src/datasource/zarr-vectors/zarr_array.js";

const expected = fixtureExpected();

function access(store: string) {
  return {
    read: fixtureRead(store),
    listDirectories: fixtureListDirectories(store),
    shardIndexes: new ShardIndexCache(),
  };
}

async function open(store: string, attributes?: string[]) {
  return openZarrVectorsStore(access(store), store, attributes);
}

function description(store: ZarrVectorsStore): ZarrVectorsGeometryDescription {
  return {
    rank: store.rank,
    geometryKind: store.geometryKind,
    linksConvention: store.linksConvention,
    linkWidth: store.linkWidth,
    linkedSkeletonLayout: store.linkedSkeletonLayout,
    attributes: store.attributes,
    vertexIdAttribute: store.vertexIdAttribute,
    hasObjects: store.hasObjects,
  };
}

function objectReader(name: string, store: ZarrVectorsStore) {
  const a = access(name);
  return new ObjectReader({
    ...a,
    description: description(store),
    level: store.levels[0],
    warn: () => {},
  });
}

const signal = new AbortController().signal;

/** Every polyline read back per object, as flat [x,y,z,...] in walk order. */
async function readAllPolylines(name: string) {
  const store = await open(name);
  const reader = objectReader(name, store);
  const out: number[][] = [];
  for (let id = 0; id < expected.polylines.length; ++id) {
    const skeleton = await reader.readSkeleton(BigInt(id), signal);
    // Walk the chain from the vertex with a single edge.
    const n = skeleton.positions.length / 3;
    const adjacency = Array.from({ length: n }, () => [] as number[]);
    for (let i = 0; i < skeleton.edges.length; i += 2) {
      adjacency[skeleton.edges[i]].push(skeleton.edges[i + 1]);
      adjacency[skeleton.edges[i + 1]].push(skeleton.edges[i]);
    }
    const ends = adjacency.flatMap((a, v) => (a.length === 1 ? [v] : []));
    expect(ends.length, `object ${id} is one open chain`).toBe(2);
    const order = [ends[0]];
    for (let prev = -1, cur = ends[0]; order.length < n; ) {
      const next = adjacency[cur].find((v) => v !== prev)!;
      prev = cur;
      cur = next;
      order.push(cur);
    }
    out.push(
      order.flatMap((v) =>
        Array.from(skeleton.positions.subarray(v * 3, v * 3 + 3)),
      ),
    );
  }
  return out;
}

function expectSamePolylines(actual: number[][]) {
  expected.polylines.forEach((polyline: number[][], id: number) => {
    const flat = polyline.flat();
    const reversed = [...polyline].reverse().flat();
    const got = actual[id];
    const matches = (ref: number[]) =>
      ref.length === got.length &&
      ref.every((v, i) => Math.abs(v - got[i]) < 1e-5);
    expect(matches(flat) || matches(reversed), `object ${id}`).toBe(true);
  });
}

describe("openZarrVectorsStore", () => {
  it("reads a raw polyline store", async () => {
    const store = await open("poly_raw");
    expect(store.geometryKind).toBe("streamline");
    expect(store.levels.length).toBe(1);
    expect(store.attributes.map((a) => [a.name, a.components])).toEqual([
      ["fa", 1],
    ]);
  });

  it("reads multi-column attributes as vectors", async () => {
    const store = await open("poly_mc");
    expect(store.attributes).toEqual([
      expect.objectContaining({ name: "rgb", components: 3 }),
    ]);
  });

  it("keeps attribute ids clear of GLSL names", () => {
    const used = new Set<string>();
    expect(safeAttributeId("z", used)).toBe("attr_z");
    expect(safeAttributeId("length", used)).toBe("attr_length");
    expect(safeAttributeId("gene_H2-Q2", used)).toBe("gene_H2_Q2");
    expect(safeAttributeId("Gad1", used)).toBe("a_Gad1");
  });

  it("rejects unknown #attributes names", async () => {
    await expect(open("poly_raw", ["nope"])).rejects.toThrow(/nope/);
  });
});

describe("per-object reads reproduce the writer's polylines", () => {
  for (const name of [
    "poly_raw",
    "poly_zstd",
    "poly_gzip",
    "poly_raw_shard",
    "poly_zstd_shard",
    "poly_multichunk",
    "poly_f64",
  ]) {
    it(name, async () => {
      expectSamePolylines(await readAllPolylines(name));
    });
  }
});

describe("dense chunks", () => {
  it("bridge every curve across chunk faces with global ids", async () => {
    for (const name of ["poly_raw", "poly_zstd_shard"]) {
      const store = await open(name);
      const pipeline = new LevelPipeline({
        ...access(name),
        description: description(store),
        level: store.levels[0],
        warn: () => {},
      });
      const vertices = store.levels[0].arrays.vertices;
      const origin = vertices.attributes.chunk_grid_origin;
      const shape = vertices.shape;
      let ownVertices = 0;
      let bridges = 0;
      const idsSeen = new Set<number>();
      for (let i = 0; i < shape[0]; ++i) {
        for (let j = 0; j < shape[1]; ++j) {
          for (let k = 0; k < shape[2]; ++k) {
            const chunk = await pipeline.download(
              [i + origin[0], j + origin[1], k + origin[2]],
              signal,
            );
            if (chunk === undefined) continue;
            ownVertices += chunk.numOwnVertices;
            bridges += chunk.numVertices - chunk.numOwnVertices;
            for (let v = 0; v < chunk.numVertices; ++v) {
              idsSeen.add(chunk.segmentIds[v * 2]);
            }
            for (const value of chunk.positions) {
              expect(Number.isFinite(value)).toBe(true);
            }
          }
        }
      }
      const total = expected.polylines.reduce(
        (n: number, p: number[][]) => n + p.length,
        0,
      );
      expect(ownVertices, name).toBe(total);
      // Every cross-chunk link becomes exactly one bridge.
      expect(bridges, name).toBe(5);
      // Ids come from the manifests (the writer stores no fragment ids).
      expect([...idsSeen].sort(), name).toEqual([0, 1, 2, 3, 4, 5]);
    }
  });
});

describe("object layer", () => {
  it("maps sparse object ids and reads multi-chunk properties", async () => {
    const table = await readObjectTable(access("pts_sparse_ids"), "0");
    expect(Array.from(table!.segmentIds, Number)).toEqual(
      expected.point_object_ids,
    );
    const polyTable = await readObjectTable(access("poly_multichunk"), "0");
    expect(polyTable!.numObjects).toBe(6);
    const warnings: string[] = [];
    const properties = await readSegmentProperties(
      access("poly_multichunk"),
      "0",
      polyTable!,
      warnings,
    );
    const length = properties!.properties.find((p) => p.id === "length") as any;
    expect(Array.from(length.values)).toEqual(
      expected.object_attributes.length,
    );
  });

  it("turns groups into tags", async () => {
    const table = await readObjectTable(access("poly_zstd"), "0");
    const properties = await readSegmentProperties(
      access("poly_zstd"),
      "0",
      table!,
      [],
    );
    const group = properties!.properties.find((p) => p.id === "group") as any;
    expect(group.values.map((v: string) => v.charCodeAt(0))).toEqual([
      0, 0, 0, 1, 1, 1,
    ]);
  });
});
