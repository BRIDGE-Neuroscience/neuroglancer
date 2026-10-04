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

import { getEventListeners } from "node:events";
import { describe, expect, it } from "vitest";
import { fixtureRead } from "#src/datasource/zarr-vectors/test_fixtures.js";
import {
  coalesceRangeReads,
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

describe("coalesceRangeReads", () => {
  const data = Uint8Array.from({ length: 1 << 20 }, (_, i) => i % 251);
  function source() {
    const calls: { path: string; offset: number; length: number }[] = [];
    const read = async (path: string, options: any) => {
      const { offset, length } = options.byteRange;
      calls.push({ path, offset, length });
      options.signal?.throwIfAborted();
      return path === "missing"
        ? undefined
        : data.slice(offset, offset + length);
    };
    return { calls, read };
  }

  it("merges nearby ranges of one key into one read, each caller keeping its bytes", async () => {
    const { calls, read } = source();
    const merged = coalesceRangeReads(read);
    const ranges = [
      [5000, 100],
      [0, 10],
      [200, 50],
      [5000, 100],
    ];
    const results = await Promise.all(
      ranges.map(([offset, length]) =>
        merged("shard", { byteRange: { offset, length } }),
      ),
    );
    expect(calls).toEqual([{ path: "shard", offset: 0, length: 5100 }]);
    results.forEach((bytes, i) => {
      const [offset, length] = ranges[i];
      expect(bytes).toEqual(data.slice(offset, offset + length));
    });
    // Copies, so one caller's decoder may take over its buffer.
    expect(results[0]!.buffer).not.toBe(results[3]!.buffer);
  });

  it("keeps distant ranges and different keys apart", async () => {
    const { calls, read } = source();
    const merged = coalesceRangeReads(read, 1024);
    await Promise.all([
      merged("a", { byteRange: { offset: 0, length: 10 } }),
      merged("a", { byteRange: { offset: 100_000, length: 10 } }),
      merged("b", { byteRange: { offset: 0, length: 10 } }),
    ]);
    expect(calls.length).toBe(3);
  });

  it("passes whole-object and suffix reads straight through", async () => {
    const calls: unknown[] = [];
    const merged = coalesceRangeReads(async (_path, options) => {
      calls.push(options.byteRange);
      return new Uint8Array(4);
    });
    await merged("a", {});
    await merged("a", { byteRange: { suffixLength: 4 } });
    expect(calls).toEqual([undefined, { suffixLength: 4 }]);
  });

  it("lets one caller abort without failing the others", async () => {
    const { calls, read } = source();
    const merged = coalesceRangeReads(read);
    const controller = new AbortController();
    const aborted = merged("shard", {
      signal: controller.signal,
      byteRange: { offset: 0, length: 10 },
    });
    const kept = merged("shard", { byteRange: { offset: 20, length: 10 } });
    const alsoKept = merged("shard", { byteRange: { offset: 40, length: 10 } });
    controller.abort();
    await expect(aborted).rejects.toBeDefined();
    expect(await kept).toEqual(data.slice(20, 30));
    expect(await alsoKept).toEqual(data.slice(40, 50));
    expect(calls).toEqual([{ path: "shard", offset: 20, length: 30 }]);
  });

  it("reports a missing key to every caller", async () => {
    const { read } = source();
    const merged = coalesceRangeReads(read);
    const results = await Promise.all([
      merged("missing", { byteRange: { offset: 0, length: 4 } }),
      merged("missing", { byteRange: { offset: 8, length: 4 } }),
    ]);
    expect(results).toEqual([undefined, undefined]);
  });

  it("reads a sharded array's cells in a few requests when they are wanted together", async () => {
    for (const store of ["poly_raw_shard", "poly_zstd_shard"]) {
      const base = fixtureRead(store);
      let requests = 0;
      const counted: typeof base = (path, options) => {
        if (path.startsWith("0/vertices/c/")) ++requests;
        return base(path, options);
      };
      const merged = coalesceRangeReads(counted);
      const array = await openZarrArray(merged, "0/vertices");
      const reader = new ZarrArrayReader(array!, merged, new ShardIndexCache());
      const plain = await openReader(store, "0/vertices");
      const { shape, origin } = array!;
      const cells: number[][] = [];
      for (let i = 0; i < shape[0]; ++i) {
        for (let j = 0; j < shape[1]; ++j) {
          for (let k = 0; k < shape[2]; ++k) {
            cells.push([i + origin[0], j + origin[1], k + origin[2]]);
          }
        }
      }
      const together = await Promise.all(cells.map((c) => reader.readCell(c)));
      for (let i = 0; i < cells.length; ++i) {
        expect(together[i], `${store} ${cells[i]}`).toEqual(
          await plain.readCell(cells[i]),
        );
      }
      const shards = new Set(
        cells.map((c) =>
          c
            .map((x, d) =>
              Math.floor((x - origin[d]) / array!.readChunkShape[d] / 2),
            )
            .join(),
        ),
      ).size;
      // One index read and one merged cell read per shard.
      expect(requests, store).toBeLessThanOrEqual(2 * shards);
      expect(requests, store).toBeLessThan(cells.length);
    }
  });
});

describe("reads sharing a signal or a shard", () => {
  it("drops its abort listeners once each read settles", async () => {
    const data = new Uint8Array(1 << 16);
    const merged = coalesceRangeReads(async (_path, options) => {
      const { offset, length } = options.byteRange as any;
      return data.slice(offset, offset + length);
    });
    // A signal that never aborts, as shared loads use: a listener left on
    // it would keep every result reachable.
    const signal = new AbortController().signal;
    await Promise.all(
      [0, 10, 20, 40_000].map((offset) =>
        merged("a", { signal, byteRange: { offset, length: 8 } }),
      ),
    );
    await merged("b", { signal, byteRange: { offset: 0, length: 8 } });
    expect(getEventListeners(signal, "abort").length).toBe(0);
  });

  it("does not fail one cell of a shard when another's read is cancelled", async () => {
    const read = fixtureRead("poly_raw_shard");
    const array = (await openZarrArray(read, "0/vertices"))!;
    const plain = new ZarrArrayReader(array, read, new ShardIndexCache());
    const cells = [...(await allCells(plain)).keys()].map((k) =>
      k.split(".").map(Number),
    );
    const shardOf = (c: number[]) =>
      c
        .map((x, d) =>
          Math.floor((x - array.origin[d]) / array.readChunkShape[d] / 2),
        )
        .join();
    const byShard = new Map<string, number[][]>();
    for (const c of cells) {
      const key = shardOf(c);
      byShard.set(key, [...(byShard.get(key) ?? []), c]);
    }
    const [a, b] = [...byShard.values()].find((group) => group.length > 1)!;
    // Like a real kvstore, a read gives up when its signal aborts.
    const abortable = async (path: string, options: any) => {
      await new Promise((resolve) => setTimeout(resolve, 0));
      options.signal?.throwIfAborted();
      return read(path, options);
    };
    const reader = new ZarrArrayReader(array, abortable, new ShardIndexCache());
    const cancelled = new AbortController();
    const first = reader.readCell(a, cancelled.signal);
    const second = reader.readCell(b, new AbortController().signal);
    cancelled.abort();
    await expect(first).rejects.toThrow();
    expect(await second).toEqual(await plain.readCell(b));
  });
});

describe("cellPayloadLength", () => {
  it("reads a cell's length from its header or shard index alone", async () => {
    for (const store of ["poly_raw", "poly_raw_shard"]) {
      const reader = await openReader(store, "0/vertices");
      for (const [key, bytes] of await allCells(reader)) {
        const cell = key.split(".").map(Number);
        expect(await reader.cellPayloadLength(cell), `${store} ${key}`).toBe(
          bytes.byteLength,
        );
      }
    }
  });

  it("cannot without reading a compressed cell", async () => {
    const reader = await openReader("poly_zstd", "0/vertices");
    const [key] = (await allCells(reader)).keys();
    expect(
      await reader.cellPayloadLength(key.split(".").map(Number)),
    ).toBeNull();
  });
});
