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
import { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import {
  CrossChunkLinks,
  ownsLink,
  decodeLinkCell,
  decodeRaggedRows,
  lehmerDecode,
} from "#src/datasource/zarr-vectors/links.js";
import {
  fixtureListDirectories,
  fixtureRead,
} from "#src/datasource/zarr-vectors/test_fixtures.js";
import { ShardIndexCache } from "#src/datasource/zarr-vectors/zarr_array.js";

/** A ragged blob: int64 group count, int64 group offsets, then the rows. */
function raggedBlob(groups: number[][][], elementBytes: 4 | 8 = 8) {
  const rows = groups.flat();
  const ncols = rows[0]?.length ?? 0;
  const header = 8 * (1 + groups.length);
  const out = new Uint8Array(header + rows.length * ncols * elementBytes);
  const view = new DataView(out.buffer);
  view.setBigInt64(0, BigInt(groups.length), true);
  let offset = 0;
  groups.forEach((g, i) => {
    view.setBigInt64(8 + 8 * i, BigInt(offset), true);
    offset += g.length * ncols * elementBytes;
  });
  rows.flat().forEach((v, i) => {
    if (elementBytes === 8) view.setBigInt64(header + 8 * i, BigInt(v), true);
    else view.setInt32(header + 4 * i, v, true);
  });
  return out;
}

describe("decodeRaggedRows", () => {
  it("flattens every group's rows", () => {
    const rows = decodeRaggedRows(
      raggedBlob([
        [[1, 2]],
        [
          [3, 4],
          [5, 6],
        ],
      ]),
      2,
    );
    expect(Array.from(rows)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("reads 32-bit link families", () => {
    const rows = decodeRaggedRows(raggedBlob([[[7, 9]]], 4), 2, 4, true);
    expect(Array.from(rows)).toEqual([7, 9]);
  });

  it("returns nothing for an empty cell", () => {
    expect(decodeRaggedRows(new Uint8Array(8), 2).length).toBe(0);
  });

  it("rejects a cell that is not whole rows", () => {
    const blob = raggedBlob([[[1, 2]]]);
    expect(() =>
      decodeRaggedRows(blob.subarray(0, blob.length - 4), 2),
    ).toThrow();
  });
});

describe("lehmerDecode", () => {
  it("inverts zarr-vectors-py's _lehmer_encode", () => {
    // _lehmer_encode([0,1,2]) == 0, ([2,1,0]) == 5, ([1,0]) == 1.
    expect(lehmerDecode(0, 3)).toEqual([0, 1, 2]);
    expect(lehmerDecode(5, 3)).toEqual([2, 1, 0]);
    expect(lehmerDecode(1, 2)).toEqual([1, 0]);
    expect(lehmerDecode(3, 3)).toEqual([1, 2, 0]);
  });
});

describe("decodeLinkCell", () => {
  const array = {
    offsets: [[0, 0, 1]],
    hasPerm: false,
    elementBytes: 8 as const,
    signed: true,
  };

  it("places the second endpoint at source + offset", () => {
    const [record] = decodeLinkCell(
      raggedBlob([[[4, 9]]]),
      [1, 2, 3],
      array,
      2,
    );
    expect(record.endpoints).toEqual([
      { chunkCoords: [1, 2, 3], vertexIndex: 4 },
      { chunkCoords: [1, 2, 4], vertexIndex: 9 },
    ]);
  });

  it("restores the writer's endpoint order from perm_idx", () => {
    const withPerm = { ...array, hasPerm: true };
    const [record] = decodeLinkCell(
      raggedBlob([[[1, 4, 9]]]),
      [0, 0, 0],
      withPerm,
      2,
    );
    // Stored canonically as (source, neighbour); perm 1 means the writer had
    // the neighbour first.
    expect(record.endpoints.map((e) => e.vertexIndex)).toEqual([9, 4]);
    expect(record.endpoints[0].chunkCoords).toEqual([0, 0, 1]);
  });
});

describe("CrossChunkLinks on a zarr-vectors-py store", () => {
  const make = (listing = true) =>
    new CrossChunkLinks({
      cells: new LevelCells(
        { read: fixtureRead("poly_raw"), shardIndexes: new ShardIndexCache() },
        "0",
      ),
      listDirectories: listing
        ? (path) => fixtureListDirectories("poly_raw")(`0/${path}`)
        : undefined,
    });
  const keys = ["0.1.0", "1.1.0", "0.1.1", "1.1.1", "0.1.2", "1.1.2"];
  const chunks = keys.map((key) => key.split(".").map(Number));
  const linkKey = (r: {
    endpoints: { chunkCoords: number[]; vertexIndex: number }[];
  }) =>
    r.endpoints
      .map((e) => `${e.chunkCoords.join(".")}:${e.vertexIndex}`)
      .join("|");

  it("finds every cross-chunk link exactly once across owning chunks", async () => {
    const links = make();
    const discovery = await links.discover();
    expect(discovery?.arrays.length).toBe(2);
    let total = 0;
    for (const chunk of chunks) {
      const records = await links.linksOwnedBy(chunk);
      for (const record of records) {
        const [a, b] = record.endpoints;
        expect(record.endpoints.map((e) => e.chunkCoords)).toContainEqual(
          chunk,
        );
        const delta = a.chunkCoords.map((c, d) =>
          Math.abs(c - b.chunkCoords[d]),
        );
        expect(delta.reduce((x, y) => x + y, 0)).toBe(1);
      }
      total += records.length;
    }
    // The family group records how many links the writer stored.
    expect(total).toBe(5);
  });

  it("finds each link from both of its chunks", async () => {
    const links = make();
    const seen = new Map<string, number>();
    for (const chunk of chunks) {
      for (const record of await links.linksTouching(chunk)) {
        seen.set(linkKey(record), (seen.get(linkKey(record)) ?? 0) + 1);
      }
    }
    expect(seen.size).toBe(5);
    for (const count of seen.values()) expect(count).toBe(2);
  });

  it("finds the same links by probing neighbours when listing is unsupported", async () => {
    const listed = make(true);
    const probed = make(false);
    for (const chunk of chunks) {
      const a = (await listed.linksOwnedBy(chunk)).map(linkKey).sort();
      const b = (await probed.linksOwnedBy(chunk)).map(linkKey).sort();
      expect(b).toEqual(a);
    }
  });
});

describe("ownsLink", () => {
  const link = (...chunks: number[][]) => ({
    endpoints: chunks.map((chunkCoords, vertexIndex) => ({
      chunkCoords,
      vertexIndex,
    })),
  });

  it("gives every link one owner among the chunks it touches", () => {
    const face = link([1, 0, 2], [0, 5, 0], [0, 4, 9]);
    const touched = [
      [1, 0, 2],
      [0, 5, 0],
      [0, 4, 9],
    ];
    expect(touched.filter((c) => ownsLink(face, c))).toEqual([[0, 4, 9]]);
    expect(ownsLink(link([-1, 0, 0], [0, 0, 0]), [-1, 0, 0])).toBe(true);
    expect(ownsLink(link([-1, 0, 0], [0, 0, 0]), [0, 0, 0])).toBe(false);
  });
});
