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
import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  decodeFloat32,
  ELEMENT_BYTES,
} from "#src/datasource/zarr-vectors/dtype.js";
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

/** Rows this close together are read together. */
const ROW_GAP = 4096;
/** Past this many reads, the whole cell is cheaper. */
const MAX_ROW_READS = 16;

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

  /** See {@link ZarrArrayReader.cellPayloadLength}. */
  async cellPayloadLength(
    path: string,
    chunkKey: string,
    signal: AbortSignal,
  ): Promise<number | null> {
    const reader = await this.reader(path);
    if (reader === undefined) return 0;
    return reader.cellPayloadLength(parseChunkKey(chunkKey), signal);
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

  /**
   * Positions of vertex rows `rows` (sorted, distinct) of a chunk: read by
   * byte range where the cell allows it (uncompressed, one cell per stored
   * chunk), else decoded from `whole()`, the whole cell.
   */
  async vertexRows(
    chunkKey: string,
    type: ElementType,
    rows: readonly number[],
    signal: AbortSignal,
    whole: () => Promise<Uint8Array | undefined>,
  ): Promise<Map<number, Float32Array>> {
    const found = new Map<number, Float32Array>();
    const rowBytes = 3 * ELEMENT_BYTES[type];
    const spans: [number, number][] = [];
    for (const v of rows) {
      const last = spans[spans.length - 1];
      if (last !== undefined && v - last[1] <= ROW_GAP) last[1] = v;
      else spans.push([v, v]);
    }
    const reads =
      spans.length > MAX_ROW_READS
        ? null
        : await Promise.all(
            spans.map(([first, last]) =>
              this.readCellRange(
                "vertices",
                chunkKey,
                first * rowBytes,
                (last - first + 1) * rowBytes,
                signal,
              ),
            ),
          );
    if (reads !== null && reads.every((r) => r !== null)) {
      reads.forEach((bytes, s) => {
        if (bytes === undefined) return;
        const [first, last] = spans[s];
        const values = decodeFloat32(bytes, type, 3 * (last - first + 1));
        for (let v = first; v <= last; ++v) {
          found.set(v, values.subarray(3 * (v - first), 3 * (v - first) + 3));
        }
      });
      return found;
    }
    const bytes = await whole();
    if (bytes === undefined) return found;
    const all = decodeFloat32(
      bytes,
      type,
      bytes.byteLength / ELEMENT_BYTES[type],
    );
    for (const v of rows) {
      if (3 * v + 3 <= all.length) found.set(v, all.subarray(3 * v, 3 * v + 3));
    }
    return found;
  }

  /** False when the array's `nonempty_chunks` rules the cell out. */
  async mayHaveCell(path: string, chunkKey: string): Promise<boolean> {
    const reader = await this.reader(path);
    return reader?.mayHaveCell(parseChunkKey(chunkKey)) ?? false;
  }
}
