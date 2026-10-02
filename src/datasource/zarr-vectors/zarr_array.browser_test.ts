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
import type { ZarrArrayRead } from "#src/datasource/zarr-vectors/zarr_array.js";
import {
  openZarrArray,
  ShardIndexCache,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";

declare const TEST_DATA_SERVER: string;

function fetchRead(store: string): ZarrArrayRead {
  const root = `${TEST_DATA_SERVER}datasource/zarr-vectors/${store}.zarrvectors/`;
  return async (path, options) => {
    const headers: Record<string, string> = {};
    const range = options.byteRange;
    if (range !== undefined) {
      headers.range =
        "suffixLength" in range
          ? `bytes=-${range.suffixLength}`
          : `bytes=${range.offset}-${range.offset + range.length - 1}`;
    }
    const response = await fetch(root + path, {
      headers,
      signal: options.signal,
    });
    if (response.status === 404) return undefined;
    if (!response.ok) throw new Error(`${response.status} for ${path}`);
    return new Uint8Array(await response.arrayBuffer());
  };
}

async function openReader(store: string, path: string) {
  const read = fetchRead(store);
  const array = await openZarrArray(read, path);
  expect(array).toBeDefined();
  return new ZarrArrayReader(array!, read, new ShardIndexCache());
}

describe("ZarrArrayReader in a browser", () => {
  it("decodes blosc cells like raw ones", async () => {
    const raw = await openReader("poly_raw", "0/vertices");
    const blosc = await openReader("poly_blosc", "0/vertices");
    const { shape, origin } = raw.array;
    let compared = 0;
    for (let i = 0; i < shape[0]; ++i) {
      for (let j = 0; j < shape[1]; ++j) {
        for (let k = 0; k < shape[2]; ++k) {
          const cell = [i + origin[0], j + origin[1], k + origin[2]];
          const expected = await raw.readCell(cell);
          expect(await blosc.readCell(cell)).toEqual(expected);
          if (expected !== undefined) ++compared;
        }
      }
    }
    expect(compared).toBeGreaterThan(1);
  });

  it("decodes blosc columns", async () => {
    const reader = await openReader("poly_blosc", "0/object_attributes/kind");
    const bytes = await reader.readAllRows();
    expect(
      Array.from(new Int32Array(bytes.buffer, bytes.byteOffset, 6)),
    ).toEqual([0, 1, 0, 1, 2, 2]);
  });
});
