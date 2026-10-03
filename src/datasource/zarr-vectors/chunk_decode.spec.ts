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
  branchedEdges,
  chunkKeyWord,
  edgeTangents,
  sequentialEdges,
  triangulate,
  walkTangents,
} from "#src/datasource/zarr-vectors/chunk_decode.js";
import { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";

/** Fragments that are contiguous vertex ranges of the given lengths. */
function ranges(...lengths: number[]) {
  const table: bigint[] = [];
  let start = 0;
  for (const n of lengths) {
    table.push(BigInt(start), BigInt(n));
    start += n;
  }
  const bitmap = new Uint8Array(Math.ceil(lengths.length / 8)).fill(0xff);
  return new FragmentIndex(
    lengths.length,
    bitmap,
    BigInt64Array.from(table),
    new Uint32Array([0]),
    new BigInt64Array(0),
  );
}

const pairs = (edges: Uint32Array) => {
  const out: string[] = [];
  for (let i = 0; i < edges.length; i += 2)
    out.push(`${edges[i]}-${edges[i + 1]}`);
  return out.sort();
};

describe("sequentialEdges", () => {
  it("joins consecutive vertices within, never across, fragments", () => {
    expect(pairs(sequentialEdges(ranges(3, 1, 2)))).toEqual([
      "0-1",
      "1-2",
      "4-5",
    ]);
  });
});

describe("branchedEdges", () => {
  // A depth-first tree: 0-1-2 is the trunk, 3 branches off 1, 4 follows 3.
  const fragments = ranges(5);

  it("replaces a child's implied parent with its record", () => {
    const records = new Uint32Array([3, 1]); // [child, parent]
    expect(pairs(branchedEdges(fragments, records))).toEqual([
      "0-1",
      "1-2",
      "1-3",
      "3-4",
    ]);
  });

  it("drops the implied edge of a child relinked from another chunk", () => {
    const edges = branchedEdges(fragments, new Uint32Array(0), new Set([3]));
    expect(pairs(edges)).toEqual(["0-1", "1-2", "3-4"]);
  });

  it("keeps a record even when it repeats the implied parent", () => {
    expect(pairs(branchedEdges(fragments, new Uint32Array([2, 1])))).toEqual([
      "0-1",
      "1-2",
      "2-3",
      "3-4",
    ]);
  });
});

describe("tangents", () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0, 2, 1, 0]);

  it("follow each fragment's walk", () => {
    const t = walkTangents(positions, ranges(3, 1));
    expect(Array.from(t.subarray(0, 9))).toEqual([1, 0, 0, 1, 0, 0, 1, 0, 0]);
    // A single-vertex fragment has no direction.
    expect(Array.from(t.subarray(9))).toEqual([0, 0, 0]);
  });

  it("orient consistently along a graph whatever the edge order", () => {
    // 1→0 and 2→1 point backwards; orientation must still agree.
    const t = edgeTangents(positions, new Uint32Array([1, 0, 2, 1, 2, 3]));
    for (let v = 1; v < 4; ++v) {
      const dot =
        t[3 * v] * t[3 * (v - 1)] +
        t[3 * v + 1] * t[3 * (v - 1) + 1] +
        t[3 * v + 2] * t[3 * (v - 1) + 2];
      expect(dot, `vertex ${v}`).toBeGreaterThanOrEqual(0);
    }
    for (let v = 0; v < 4; ++v) {
      expect(Math.hypot(t[3 * v], t[3 * v + 1], t[3 * v + 2])).toBeCloseTo(1);
    }
  });
});

describe("triangulate", () => {
  it("passes triangles through and fans quads", () => {
    const tris = new Uint32Array([0, 1, 2]);
    expect(triangulate(tris, 3)).toBe(tris);
    expect(Array.from(triangulate(new Uint32Array([0, 1, 2, 3]), 4))).toEqual([
      0, 1, 2, 0, 2, 3,
    ]);
  });
});

describe("chunkKeyWord", () => {
  it("is distinct for every chunk of a 3-d grid", () => {
    const seen = new Set<number>();
    for (const x of [-512, -1, 0, 1, 511]) {
      for (const y of [-1, 0, 7]) {
        for (const z of [0, 300]) seen.add(chunkKeyWord(`${x}.${y}.${z}`));
      }
    }
    expect(seen.size).toBe(30);
  });

  it("hashes keys outside the packed range", () => {
    const word = chunkKeyWord("4096.0.0");
    expect(word).toBe(word >>> 0);
    expect(word).not.toBe(chunkKeyWord("0.0.0"));
  });
});
