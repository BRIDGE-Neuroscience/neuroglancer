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
 * @file Test-only access to the stores in testdata/datasource/zarr-vectors,
 * which `generate.py` there writes with zarr-vectors-py v0.9.2.
 */

import fs from "node:fs";
import path from "node:path";
import type { ZarrArrayRead } from "#src/datasource/zarr-vectors/zarr_array.js";

export const FIXTURE_DIR = path.resolve(
  import.meta.dirname,
  "../../../testdata/datasource/zarr-vectors",
);

export function fixtureExpected(): any {
  return JSON.parse(
    fs.readFileSync(path.join(FIXTURE_DIR, "expected.json"), "utf8"),
  );
}

/** A `ZarrArrayRead` over a fixture store, honouring byte ranges. */
export function fixtureRead(store: string): ZarrArrayRead {
  const root = path.join(FIXTURE_DIR, `${store}.zarrvectors`);
  return async (relPath, options) => {
    let data: Buffer;
    try {
      data = fs.readFileSync(path.join(root, relPath));
    } catch {
      return undefined;
    }
    let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const range = options.byteRange;
    if (range !== undefined) {
      if ("suffixLength" in range) {
        bytes = bytes.subarray(bytes.length - range.suffixLength);
      } else {
        bytes = bytes.subarray(range.offset, range.offset + range.length);
      }
    }
    return new Uint8Array(bytes);
  };
}

/** Lists child directories of a store-relative path, like a kvstore listing. */
export function fixtureListDirectories(store: string) {
  const root = path.join(FIXTURE_DIR, `${store}.zarrvectors`);
  return async (relPath: string): Promise<string[]> => {
    const dir = path.join(root, relPath);
    if (!fs.existsSync(dir)) return [];
    return fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name);
  };
}
