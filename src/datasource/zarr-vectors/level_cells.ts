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
 * @file Cell access for one resolution level: opens each per-chunk array of
 * the level lazily from its own `zarr.json` and reads cells by spatial chunk
 * key.  Arrays the store metadata already described are not re-read.
 */

import type {
  ShardIndexCache,
  ZarrArrayRead,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import {
  parseZarrArrayMetadata,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";

/**
 * Reads the payload of one per-chunk array cell.  `arrayPath` is relative to
 * the level (`"vertices"`, `"links/0/0.0.+1"`), `chunkKey` the spatial key
 * (`"-2.0.1"`).  Resolves to `undefined` when the cell is empty.
 */
export type CellReader = (
  arrayPath: string,
  chunkKey: string,
  signal: AbortSignal,
) => Promise<Uint8Array | undefined>;

export function parseChunkKey(chunkKey: string): number[] {
  return chunkKey.split(".").map(Number);
}

export class LevelCells {
  private readers = new Map<string, Promise<ZarrArrayReader | undefined>>();

  constructor(
    private read: ZarrArrayRead,
    private shardIndexes: ShardIndexCache,
    readonly levelPath: string,
    known: ReadonlyMap<string, any> = new Map(),
  ) {
    for (const [path, json] of known) {
      if (json === undefined) {
        this.readers.set(path, Promise.resolve(undefined));
        continue;
      }
      this.readers.set(
        path,
        Promise.resolve(
          new ZarrArrayReader(
            parseZarrArrayMetadata(`${levelPath}/${path}`, json),
            read,
            shardIndexes,
          ),
        ),
      );
    }
  }

  /** Parses a level-relative JSON document, or `undefined` if absent. */
  async readJson(path: string): Promise<any | undefined> {
    const bytes = await this.read(`${this.levelPath}/${path}`, {});
    if (bytes === undefined) return undefined;
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  /** The reader of a level-relative array, or `undefined` if it is absent. */
  reader(arrayPath: string): Promise<ZarrArrayReader | undefined> {
    let promise = this.readers.get(arrayPath);
    if (promise === undefined) {
      const fullPath = `${this.levelPath}/${arrayPath}`;
      // Opened without the caller's signal: the reader is shared, and one
      // aborted chunk must not leave every later chunk without it.
      promise = this.read(`${fullPath}/zarr.json`, {}).then((bytes) => {
        if (bytes === undefined) return undefined;
        const json = JSON.parse(new TextDecoder().decode(bytes));
        if (json?.node_type !== "array") return undefined;
        return new ZarrArrayReader(
          parseZarrArrayMetadata(fullPath, json),
          this.read,
          this.shardIndexes,
        );
      });
      this.readers.set(arrayPath, promise);
      promise.catch(() => {
        if (this.readers.get(arrayPath) === promise) {
          this.readers.delete(arrayPath);
        }
      });
    }
    return promise;
  }

  async readCell(
    arrayPath: string,
    chunkKey: string,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined> {
    const reader = await this.reader(arrayPath);
    if (reader === undefined) return undefined;
    return reader.readCell(parseChunkKey(chunkKey), signal);
  }

  /** See {@link ZarrArrayReader.readCellRange}. */
  async readCellRange(
    arrayPath: string,
    chunkKey: string,
    offset: number,
    length: number,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined | null> {
    const reader = await this.reader(arrayPath);
    if (reader === undefined) return undefined;
    return reader.readCellRange(
      parseChunkKey(chunkKey),
      offset,
      length,
      signal,
    );
  }

  /** Whether a cell may hold data, from the array's `nonempty_chunks`. */
  async mayHaveCell(arrayPath: string, chunkKey: string): Promise<boolean> {
    const reader = await this.reader(arrayPath);
    if (reader === undefined) return false;
    return reader.mayHaveCell(parseChunkKey(chunkKey));
  }

  get cellReader(): CellReader {
    return (arrayPath, chunkKey, signal) =>
      this.readCell(arrayPath, chunkKey, signal);
  }
}
