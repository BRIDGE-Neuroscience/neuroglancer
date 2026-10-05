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
import {
  decodeChunk,
  forEachFragmentVertex,
} from "#src/datasource/zarr-vectors/chunk_decode.js";
import { LevelPipeline } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import { levelDensities } from "#src/datasource/zarr-vectors/dense_lod.js";
import { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import {
  mergeSkeletons,
  ObjectReader,
} from "#src/datasource/zarr-vectors/object_reader.js";
import {
  chooseShaderObjectValues,
  readObjectTable,
  readSegmentProperties,
} from "#src/datasource/zarr-vectors/objects.js";
import {
  chunkIndexBounds,
  kvStoreAccess,
  levelChain,
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
  return openZarrVectorsStore(access(store), attributes);
}

async function objectReader(name: string) {
  const store = await open(name);
  return new ObjectReader(access(name), store.description, store.levels[0]);
}

const signal = new AbortController().signal;

/** Every polyline read back per object, as flat [x,y,z,...] in walk order. */
async function readAllPolylines(name: string) {
  const reader = await objectReader(name);
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
    expect(store.description.geometryKind).toBe("streamline");
    expect(store.levels.length).toBe(1);
    expect(
      store.description.attributes.map((a) => [a.name, a.components]),
    ).toEqual([["fa", 1]]);
  });

  it("reads multi-column attributes as vectors", async () => {
    const store = await open("poly_mc");
    expect(store.description.attributes).toEqual([
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
      const pipeline = new LevelPipeline(
        access(name),
        store.description,
        store.levels[0],
      );
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

describe("vertex cell reads", () => {
  /** Every chunk of a level 0, downloaded by one pipeline. */
  async function downloadAll(name: string, requested: boolean) {
    const reads: string[] = [];
    const base = access(name);
    const counted = {
      ...base,
      read: (path: string, options: any) => {
        // One entry per cell read: a shard's cells share its path.
        const range = options?.byteRange;
        if (
          path.startsWith("0/vertices/c/") &&
          !(range && "suffixLength" in range)
        ) {
          reads.push(`${path}@${range?.offset ?? ""}`);
        }
        return base.read(path, options);
      },
    };
    const store = await openZarrVectorsStore(counted, undefined);
    // Opening samples a cell to measure attribute widths; count only the pipeline.
    reads.length = 0;
    const level = store.levels[0];
    const pipeline = new LevelPipeline(
      counted,
      store.description,
      level,
      () => requested,
    );
    const { shape, attributes } = level.arrays.vertices;
    const origin = attributes.chunk_grid_origin ?? [0, 0, 0];
    let chunks = 0;
    for (let i = 0; i < shape[0]; ++i) {
      for (let j = 0; j < shape[1]; ++j) {
        for (let k = 0; k < shape[2]; ++k) {
          const chunk = [i + origin[0], j + origin[1], k + origin[2]];
          if ((await pipeline.download(chunk, signal)) !== undefined) ++chunks;
        }
      }
    }
    return { reads, chunks };
  }

  it("fetches each cell once when the view draws its neighbours too", async () => {
    for (const name of ["poly_raw", "poly_zstd", "poly_zstd_shard"]) {
      const { reads, chunks } = await downloadAll(name, true);
      expect(chunks, name).toBeGreaterThan(1);
      expect(reads.length, name).toBe(new Set(reads).size);
    }
  });

  it("reads only the rows it needs of a raw neighbour the view does not draw", async () => {
    const { reads } = await downloadAll("poly_raw", false);
    // Whole-cell reads carry no offset; row-range reads of neighbours do.
    expect(reads.filter((r) => !r.endsWith("@")).length).toBeGreaterThan(0);
  });

  it("leaves a mesh's faces unread, since the overview draws its vertices", async () => {
    const reads: string[] = [];
    const base = access("mesh_raw");
    const counted = {
      ...base,
      read: (path: string, options: any) => {
        reads.push(path);
        return base.read(path, options);
      },
    };
    const store = await openZarrVectorsStore(counted, undefined);
    const pipeline = new LevelPipeline(
      counted,
      store.description,
      store.levels[0],
    );
    const data = await pipeline.download([0, 0, 0], signal);
    expect(data?.numVertices).toBeGreaterThan(0);
    expect(reads.filter((r) => /^0\/links\/.*\/c\//.test(r))).toEqual([]);
  });

  it("never takes ghost positions from another store's cells", async () => {
    // pts_raw has cells at the same keys as skel_raw's neighbours.
    await downloadAll("pts_raw", true);
    const store = await open("skel_raw");
    const pipeline = new LevelPipeline(
      access("skel_raw"),
      store.description,
      store.levels[0],
    );
    const data = await pipeline.download([0, 0, 0], signal);
    // Vertex 4's bridge to its parent in chunk 1.0.0 ends at (20, 12, 4).
    const ghost = data!.positions.subarray(3 * data!.numOwnVertices);
    expect(Array.from(ghost)).toEqual([20, 12, 4]);
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

describe("skeletons", () => {
  const {
    positions,
    edges,
    object_ids: objectIds,
  } = expected.skeleton as {
    positions: number[][];
    edges: number[][];
    object_ids: number[];
  };
  const key = (p: ArrayLike<number>) =>
    Array.from(p, (v) => v.toFixed(3)).join(",");
  const edgeKey = (a: ArrayLike<number>, b: ArrayLike<number>) =>
    [key(a), key(b)].sort().join("|");
  const expectedEdges = (id?: number) =>
    edges
      .filter(([child]) => id === undefined || objectIds[child] === id)
      .map(([child, parent]) => edgeKey(positions[child], positions[parent]))
      .sort();
  const at = (p: Float32Array, v: number) => p.subarray(3 * v, 3 * v + 3);

  it("are detected as the linked layout", async () => {
    const store = await open("skel_raw");
    expect(store.description.geometryKind).toBe("skeleton");
    expect(store.description.linksConvention).toBe(
      "implicit_sequential_with_branches",
    );
    expect(store.description.skeletonLayout).toBe("linked");
  });

  it("read per object with exactly the writer's edges", async () => {
    const reader = await objectReader("skel_raw");
    for (const id of [0, 1]) {
      const skeleton = await reader.readSkeleton(BigInt(id), signal);
      const got: string[] = [];
      for (let i = 0; i < skeleton.edges.length; i += 2) {
        got.push(
          edgeKey(
            at(skeleton.positions, skeleton.edges[i]),
            at(skeleton.positions, skeleton.edges[i + 1]),
          ),
        );
      }
      expect(got.sort(), `object ${id}`).toEqual(expectedEdges(id));
    }
  });

  it("draw every edge once across dense chunks, with no chords", async () => {
    const store = await open("skel_raw");
    const pipeline = new LevelPipeline(
      access("skel_raw"),
      store.description,
      store.levels[0],
    );
    const got: string[] = [];
    const ids = new Map<string, number>();
    for (const chunk of [
      [0, 0, 0],
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 1],
    ]) {
      const data = await pipeline.download(chunk, signal);
      expect(data, chunk.join(".")).toBeDefined();
      for (let v = 0; v < data!.numOwnVertices; ++v) {
        ids.set(key(at(data!.positions, v)), data!.segmentIds[2 * v]);
      }
      for (let i = 0; i < data!.edges.length; i += 2) {
        got.push(
          edgeKey(
            at(data!.positions, data!.edges[i]),
            at(data!.positions, data!.edges[i + 1]),
          ),
        );
      }
    }
    expect(got.sort()).toEqual(expectedEdges());
    positions.forEach((p, v) => expect(ids.get(key(p))).toBe(objectIds[v]));
  });
});

describe("edge cases", () => {
  it("falls back to row ids when an id column is shorter than the objects", async () => {
    // An edit that added objects without extending the column.
    const base = access("pts_sparse_ids");
    const short = {
      ...base,
      read: async (path: string, options: any) => {
        const bytes = await base.read(path, options);
        if (
          bytes === undefined ||
          !path.endsWith("object_index/object_ids/zarr.json")
        ) {
          return bytes;
        }
        const json = JSON.parse(new TextDecoder().decode(bytes));
        json.shape = [json.shape[0] - 1];
        return new TextEncoder().encode(JSON.stringify(json));
      },
    };
    const table = await readObjectTable(short, "0");
    expect(table!.idSource).toBe("row");
    expect(table!.segmentIds.length).toBe(expected.point_object_ids.length);
  });

  it("refuses a store that requires what it does not implement", async () => {
    const withRequired = (required: string[]) => {
      const base = access("poly_raw");
      return {
        ...base,
        read: async (path: string, options: any) => {
          const bytes = await base.read(path, options);
          if (bytes === undefined || path !== "zarr.json") return bytes;
          const json = JSON.parse(new TextDecoder().decode(bytes));
          json.attributes.zarr_vectors.required_capabilities = required;
          return new TextEncoder().encode(JSON.stringify(json));
        },
      };
    };
    await expect(
      openZarrVectorsStore(withRequired(["teleportation"]), undefined),
    ).rejects.toThrow(/requires teleportation/);
    const store = await openZarrVectorsStore(
      withRequired(["additive_levels"]),
      undefined,
    );
    expect(store.levels[0].refinement).toBe("replace");
  });

  it("counts a vertex on the upper bound as inside", () => {
    const { lowerChunkBound, upperChunkBound } = chunkIndexBounds(
      [-16, 0, 1],
      [32, 31.5, 1],
      [16, 16, 16],
    );
    expect(Array.from(lowerChunkBound)).toEqual([-1, 0, 0]);
    // 32 is stored in cell 2; 31.5 in cell 1; a flat axis still has a cell.
    expect(Array.from(upperChunkBound)).toEqual([3, 2, 1]);
  });

  it("keeps only the asked-for bytes when a server ignores the range", async () => {
    const file = Uint8Array.from({ length: 100 }, (_, i) => i);
    for (const honours of [false, true]) {
      const context = {
        read: async (_url: string, options: any) => {
          const range = options.byteRange;
          if (!honours || range === undefined) {
            return { response: new Response(file), offset: 0 };
          }
          const offset =
            "suffixLength" in range
              ? file.length - range.suffixLength
              : range.offset;
          const length =
            "suffixLength" in range ? range.suffixLength : range.length;
          return {
            response: new Response(file.slice(offset, offset + length)),
            offset,
          };
        },
        list: async () => ({ directories: [] }),
      };
      const access = kvStoreAccess(context, "http://example/store/");
      const [a, b, suffix] = await Promise.all([
        access.read("cell", { byteRange: { offset: 10, length: 5 } }),
        access.read("cell", { byteRange: { offset: 50, length: 3 } }),
        access.read("cell", { byteRange: { suffixLength: 4 } }),
      ]);
      expect(Array.from(a!), `honours ${honours}`).toEqual([
        10, 11, 12, 13, 14,
      ]);
      expect(Array.from(b!)).toEqual([50, 51, 52]);
      expect(Array.from(suffix!)).toEqual([96, 97, 98, 99]);
    }
  });
});

describe("additive pyramids", () => {
  // The same streamlines as a replacement pyramid and as an additive one
  // (zvtools pyramid --refinement add): each additive level's chain must
  // show exactly what the replacement level shows.
  const point = (p: ArrayLike<number>, v: number) =>
    [0, 1, 2].map((d) => p[3 * v + d].toFixed(4)).join(",");
  const edgeSet = (positions: ArrayLike<number>, edges: ArrayLike<number>) => {
    const out: string[] = [];
    for (let e = 0; e < edges.length; e += 2) {
      out.push(
        [point(positions, edges[e]), point(positions, edges[e + 1])]
          .sort()
          .join("|"),
      );
    }
    return out.sort();
  };

  it("are read as chains of levels", async () => {
    const store = await open("add_additive");
    expect(store.levels.map((l) => l.refinement)).toEqual([
      "add",
      "add",
      "replace",
    ]);
    expect(store.levels.map((_, i) => levelChain(store.levels, i))).toEqual([
      [0, 1, 2],
      [1, 2],
      [2],
    ]);
    const replace = await open("add_replace");
    const densities = (s: typeof store) =>
      levelDensities(s.levels, s.lowerBounds, s.upperBounds);
    expect(densities(store)).toEqual(densities(replace));
  });

  it("give every object whole, from whichever level stores it", async () => {
    const additive = await open("add_additive");
    const replace = await open("add_replace");
    const readers = additive.levels.map(
      (level) =>
        new ObjectReader(access("add_additive"), additive.description, level),
    );
    const whole = new ObjectReader(
      access("add_replace"),
      replace.description,
      replace.levels[0],
    );
    const stored = new Set<number>();
    for (let id = 0; id < 24; ++id) {
      const parts = await Promise.all(
        readers.map((r) => r.readSkeleton(BigInt(id), signal)),
      );
      parts.forEach((p, level) => {
        if (p.positions.length > 0) stored.add(level);
      });
      expect(parts.filter((p) => p.positions.length > 0).length, `${id}`).toBe(
        1,
      );
      const merged = mergeSkeletons(parts);
      const want = await whole.readSkeleton(BigInt(id), signal);
      expect(edgeSet(merged.positions, merged.edges), `object ${id}`).toEqual(
        edgeSet(want.positions, want.edges),
      );
    }
    expect([...stored].sort()).toEqual([0, 1, 2]);
  });

  it("draw each level's view from its chain, as the replacement level", async () => {
    const additive = await open("add_additive");
    const replace = await open("add_replace");
    const dense = async (
      name: string,
      store: typeof additive,
      levels: number[],
    ) => {
      const edges: string[] = [];
      let own = 0;
      for (const index of levels) {
        const level = store.levels[index];
        const pipeline = new LevelPipeline(
          access(name),
          store.description,
          level,
        );
        const { shape } = level.arrays.vertices;
        const origin = level.arrays.vertices.attributes.chunk_grid_origin ?? [
          0, 0, 0,
        ];
        for (let i = 0; i < shape[0]; ++i) {
          for (let j = 0; j < shape[1]; ++j) {
            for (let k = 0; k < shape[2]; ++k) {
              const chunk = await pipeline.download(
                [i + origin[0], j + origin[1], k + origin[2]],
                signal,
              );
              if (chunk === undefined) continue;
              own += chunk.numOwnVertices;
              edges.push(...edgeSet(chunk.positions, chunk.edges));
            }
          }
        }
      }
      return { own, edges: edges.sort() };
    };
    for (let index = 0; index < 3; ++index) {
      const got = await dense(
        "add_additive",
        additive,
        levelChain(additive.levels, index),
      );
      const want = await dense("add_replace", replace, [index]);
      expect(got.own, `level ${index}`).toBe(want.own);
      expect(got.edges, `level ${index}`).toEqual(want.edges);
    }
  });
});

describe("which object a fragment belongs to", () => {
  /** Every fragment's segment id in the dense view, against the manifests. */
  async function compare(name: string, index: number) {
    const store = await open(name);
    const level = store.levels[index];
    const table = (await readObjectTable(access(name), level.path))!;
    const reader = new ObjectReader(access(name), store.description, level);
    const truth = new Map<string, bigint>();
    for (const id of table.segmentIds) {
      for (const block of await reader.manifest(id)) {
        const key = block.chunkCoords.join(".");
        const ref = block.fragmentRef as any;
        const fragments =
          ref.mode === "single"
            ? [ref.fragmentIndex]
            : ref.mode === "range"
              ? Array.from({ length: ref.count }, (_, i) => ref.start + i)
              : Array.from(ref.indices as number[]);
        for (const f of fragments) truth.set(`${key}:${f}`, id);
      }
    }
    const pipeline = new LevelPipeline(access(name), store.description, level);
    const cells = LevelCells.forLevel(access(name), level, store.description);
    const { shape } = level.arrays.vertices;
    let checked = 0;
    let wrong = 0;
    for (let i = 0; i < shape[0]; ++i) {
      for (let j = 0; j < shape[1]; ++j) {
        for (let k = 0; k < shape[2]; ++k) {
          const key = `${i}.${j}.${k}`;
          const chunk = await pipeline.download([i, j, k], signal);
          if (chunk === undefined) continue;
          const decoded = (await decodeChunk(
            cells,
            store.description,
            key,
            signal,
            { skipSegmentIds: true },
          ))!;
          for (let f = 0; f < decoded.fragments.numFragments; ++f) {
            const want = truth.get(`${key}:${f}`);
            forEachFragmentVertex(decoded.fragments, f, (v) => {
              const got =
                BigInt(chunk.segmentIds[2 * v]) |
                (BigInt(chunk.segmentIds[2 * v + 1]) << 32n);
              ++checked;
              if (got !== want) ++wrong;
            });
          }
        }
      }
    }
    return { checked, wrong };
  }

  it("never takes a stored row for another object's id", async () => {
    // Ids 1..12; zarr-vectors-tools' pyramid wrote rows 0..11 per fragment.
    for (const index of [0, 1]) {
      const { checked, wrong } = await compare("ids_collide", index);
      expect(checked, `level ${index}`).toBeGreaterThan(0);
      expect(wrong, `level ${index}`).toBe(0);
    }
  });

  it("reads a per-fragment id column of any integer type", async () => {
    const { checked, wrong } = await compare("ids_int32", 0);
    expect(checked).toBeGreaterThan(0);
    expect(wrong).toBe(0);
  });

  it("leaves an attribute a level does not store unknown, not zero", async () => {
    const store = await open("attrs_coarse");
    expect(store.description.attributes.map((a) => a.name)).toEqual(["radius"]);
    const decodeLevel = async (index: number) => {
      const level = store.levels[index];
      const cells = LevelCells.forLevel(
        access("attrs_coarse"),
        level,
        store.description,
      );
      const key = [...level.arrays.vertices.attributes.nonempty_chunks][0];
      return (await decodeChunk(
        cells,
        store.description,
        String(key),
        signal,
      ))!;
    };
    const fine = await decodeLevel(0);
    expect(fine.attributes[0].every((x) => x >= 2 && x <= 13)).toBe(true);
    const coarse = await decodeLevel(1);
    expect(coarse.numVertices).toBeGreaterThan(0);
    expect(coarse.attributes[0].every((x) => Number.isNaN(x))).toBe(true);
  });
});

describe("object values in shaders", () => {
  /** poly_raw with its object values added, as the frontend adds them. */
  async function withObjectValues() {
    const store = await open("poly_raw");
    const warnings: string[] = [];
    const chosen = await chooseShaderObjectValues(
      access("poly_raw"),
      "0",
      undefined,
      12 - store.description.attributes.length,
      warnings,
    );
    const description = {
      ...store.description,
      objectValuesPath: "0",
      attributes: [
        ...store.description.attributes,
        ...chosen.map((c) => ({
          name: c.column ?? "group",
          id: `obj_${c.column ?? "group"}`,
          dtype: "float32" as const,
          components: 1,
          objectValue: { column: c.column },
        })),
      ],
    };
    return { store, description, chosen, warnings };
  }

  it("offers each object's group and its numeric attributes", async () => {
    const { chosen, warnings } = await withObjectValues();
    expect(chosen.map((c) => c.column ?? "#group")).toEqual([
      "#group",
      "kind",
      "length",
    ]);
    expect(warnings).toEqual([]);
  });

  it("puts the object's values on every vertex, in both layers", async () => {
    const { store, description } = await withObjectValues();
    const slot = (id: string) =>
      description.attributes.findIndex((a) => a.id === id);
    const want = (row: number) => ({
      group: row < 3 ? 0 : 1,
      kind: expected.object_attributes.kind[row],
      length: expected.object_attributes.length[row],
    });
    // Objects: the skeleton of each object carries its own values.
    const reader = new ObjectReader(
      access("poly_raw"),
      description,
      store.levels[0],
    );
    for (let row = 0; row < 6; ++row) {
      const skeleton = await reader.readSkeleton(BigInt(row), signal);
      const w = want(row);
      for (const [id, value] of Object.entries(w)) {
        const values = skeleton.attributes[slot(`obj_${id}`)];
        expect(values.length).toBeGreaterThan(0);
        expect(
          values.every((x) => x === value),
          `${id} of ${row}`,
        ).toBe(true);
      }
    }
    // Dense: every own vertex carries its object's values.
    const pipeline = new LevelPipeline(
      access("poly_raw"),
      description,
      store.levels[0],
    );
    const { shape } = store.levels[0].arrays.vertices;
    const origin = store.levels[0].arrays.vertices.attributes.chunk_grid_origin;
    let checked = 0;
    for (let i = 0; i < shape[0]; ++i) {
      for (let j = 0; j < shape[1]; ++j) {
        for (let k = 0; k < shape[2]; ++k) {
          const chunk = await pipeline.download(
            [i + origin[0], j + origin[1], k + origin[2]],
            signal,
          );
          if (chunk === undefined) continue;
          for (let v = 0; v < chunk.numOwnVertices; ++v) {
            const w = want(chunk.segmentIds[2 * v]);
            expect(chunk.attributes[slot("obj_kind")][v]).toBe(w.kind);
            expect(chunk.attributes[slot("obj_length")][v]).toBe(w.length);
            expect(chunk.attributes[slot("obj_group")][v]).toBe(w.group);
            ++checked;
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(0);
  });
});
