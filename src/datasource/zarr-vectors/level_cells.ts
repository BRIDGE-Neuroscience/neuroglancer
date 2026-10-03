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
 * @file Cell access for one resolution level. Each per-chunk array is opened
 * lazily from its own `zarr.json`, since codecs, sharding and
 * `chunk_grid_origin` differ per array; arrays the store metadata already
 * described are not read again.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/base.js";
import type {
  ZarrVectorsLevel,
  ZarrVectorsStoreAccess,
} from "#src/datasource/zarr-vectors/store.js";
import {
  parseZarrArrayMetadata,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";

function parseChunkKey(chunkKey: string): number[] {
  return chunkKey.split(".").map(Number);
}

export class LevelCells {
  private readers = new Map<string, Promise<ZarrArrayReader | undefined>>();

  constructor(
    private access: Pick<ZarrVectorsStoreAccess, "read" | "shardIndexes">,
    readonly levelPath: string,
    known: ReadonlyMap<string, any> = new Map(),
  ) {
    for (const [path, json] of known) {
      this.readers.set(
        path,
        Promise.resolve(
          json === undefined ? undefined : this.makeReader(path, json),
        ),
      );
    }
  }

  /** The cells of `level`, with the arrays the store metadata described. */
  static forLevel(
    access: Pick<ZarrVectorsStoreAccess, "read" | "shardIndexes">,
    level: ZarrVectorsLevel,
    description: ZarrVectorsGeometryDescription,
  ) {
    const known = new Map<string, any>([
      ["vertices", level.arrays.vertices],
      ["vertex_fragments", level.arrays.vertexFragments],
    ]);
    if (description.hasObjects) {
      known.set(
        "fragment_attributes/segment_id",
        level.arrays.fragmentSegmentIds,
      );
    }
    description.attributes.forEach((a, i) =>
      known.set(`vertex_attributes/${a.name}`, level.arrays.attributes[i]),
    );
    return new LevelCells(access, level.path, known);
  }

  private makeReader(path: string, json: any) {
    return new ZarrArrayReader(
      parseZarrArrayMetadata(`${this.levelPath}/${path}`, json),
      this.access.read,
      this.access.shardIndexes,
    );
  }

  /** A level-relative JSON document, or `undefined` if absent. */
  async readJson(path: string): Promise<any | undefined> {
    const bytes = await this.access.read(`${this.levelPath}/${path}`, {});
    if (bytes === undefined) return undefined;
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  /** The reader of a level-relative array, or `undefined` if absent. */
  reader(path: string): Promise<ZarrArrayReader | undefined> {
    let promise = this.readers.get(path);
    if (promise === undefined) {
      // Opened without any caller's signal: readers are shared.
      const opened = this.readJson(`${path}/zarr.json`).then((json) =>
        json?.node_type === "array" ? this.makeReader(path, json) : undefined,
      );
      promise = opened;
      this.readers.set(path, opened);
      opened.catch(() => {
        if (this.readers.get(path) === opened) this.readers.delete(path);
      });
    }
    return promise;
  }

  /** The payload of a cell, or `undefined` if it is empty. */
  async readCell(
    path: string,
    chunkKey: string,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined> {
    const reader = await this.reader(path);
    const payload = await reader?.readCell(parseChunkKey(chunkKey), signal);
    return payload?.byteLength === 0 ? undefined : payload;
  }

  /** See {@link ZarrArrayReader.readCellRange}. */
  async readCellRange(
    path: string,
    chunkKey: string,
    offset: number,
    length: number,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined | null> {
    const reader = await this.reader(path);
    if (reader === undefined) return undefined;
    return reader.readCellRange(
      parseChunkKey(chunkKey),
      offset,
      length,
      signal,
    );
  }

  /** False when the array's `nonempty_chunks` rules the cell out. */
  async mayHaveCell(path: string, chunkKey: string): Promise<boolean> {
    const reader = await this.reader(path);
    return reader?.mayHaveCell(parseChunkKey(chunkKey)) ?? false;
  }
}
