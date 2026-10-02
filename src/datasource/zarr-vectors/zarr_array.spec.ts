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
import { fixtureRead } from "#src/datasource/zarr-vectors/test_fixtures.js";
import {
  openZarrArray,
  ShardIndexCache,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";

async function openReader(store: string, path: string) {
  const read = fixtureRead(store);
  const array = await openZarrArray(read, path);
  expect(array).toBeDefined();
  return new ZarrArrayReader(array!, read, new ShardIndexCache());
}

async function allCells(reader: ZarrArrayReader) {
  const out = new Map<string, Uint8Array>();
  const { shape, origin } = reader.array;
  for (let i = 0; i < shape[0]; ++i) {
    for (let j = 0; j < shape[1]; ++j) {
      for (let k = 0; k < shape[2]; ++k) {
        const cell = [i + origin[0], j + origin[1], k + origin[2]];
        const bytes = await reader.readCell(cell);
        if (bytes !== undefined) out.set(cell.join("."), bytes);
      }
    }
  }
  return out;
}

describe("ZarrArrayReader", () => {
  it("decodes every codec and layout to the same cells", async () => {
    const reference = await allCells(
      await openReader("poly_raw", "0/vertices"),
    );
    expect(reference.size).toBeGreaterThan(1);
    // blosc is covered by zarr_array.browser_test.ts: its WASM decoder rejects
    // the cross-realm typed arrays of vitest's node worker polyfill.
    for (const store of [
      "poly_zstd",
      "poly_gzip",
      "poly_raw_shard",
      "poly_zstd_shard",
    ]) {
      const cells = await allCells(await openReader(store, "0/vertices"));
      expect([...cells.keys()].sort(), store).toEqual(
        [...reference.keys()].sort(),
      );
      for (const [key, bytes] of reference) {
        expect(cells.get(key), `${store} ${key}`).toEqual(bytes);
      }
    }
  });

  it("range-reads rows of a raw cell, sharded or not", async () => {
    for (const store of ["poly_raw", "poly_raw_shard"]) {
      const reader = await openReader(store, "0/vertices");
      const cells = await allCells(reader);
      for (const [key, bytes] of cells) {
        if (bytes.length < 24) continue;
        const cell = key.split(".").map(Number);
        const row = await reader.readCellRange(cell, 12, 12);
        expect(row, `${store} ${key}`).toEqual(bytes.subarray(12, 24));
      }
    }
  });

  it("refuses range reads of compressed cells", async () => {
    const reader = await openReader("poly_zstd", "0/vertices");
    const cells = await allCells(reader);
    const cell = [...cells.keys()][0].split(".").map(Number);
    expect(await reader.readCellRange(cell, 0, 12)).toBeNull();
  });

  it("reads fixed-size columns across chunks", async () => {
    const reader = await openReader(
      "poly_multichunk",
      "0/object_attributes/length",
    );
    expect(reader.array.readChunkShape[0]).toBe(4);
    const bytes = await reader.readAllRows();
    const values = new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength / 4,
    );
    expect(Array.from(values)).toEqual([0, 1.5, 3, 4.5, 6, 7.5]);
  });

  it("reads compressed fixed-size columns", async () => {
    for (const store of ["poly_zstd", "poly_gzip"]) {
      const reader = await openReader(store, "0/object_attributes/kind");
      const bytes = await reader.readAllRows();
      const values = new Int32Array(
        bytes.buffer,
        bytes.byteOffset,
        bytes.byteLength / 4,
      );
      expect(Array.from(values), store).toEqual([0, 1, 0, 1, 2, 2]);
    }
  });

  it("reads vlen rows across chunks", async () => {
    const reader = await openReader(
      "poly_multichunk",
      "0/object_index/manifests",
    );
    const rows = await reader.readVlenRows(0, 6);
    expect(rows.length).toBe(6);
    expect(rows.every((r) => r !== undefined && r.length > 0)).toBe(true);
  });

  it("returns undefined for cells outside the grid", async () => {
    const reader = await openReader("poly_raw", "0/vertices");
    expect(await reader.readCell([-100, -100, -100])).toBeUndefined();
  });
});
