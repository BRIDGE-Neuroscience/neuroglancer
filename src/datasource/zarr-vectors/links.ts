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
 * @file Links within one level (`links/0/<offsets>`, format 0.9).
 *
 * `links/0/0.0.0` (`0.0.0_0.0.0` for faces) holds each chunk's own records,
 * flat. A link whose endpoints lie in different chunks is stored once, in the
 * cell of its first endpoint's chunk, in the array named by the other
 * endpoints' chunk offsets (`links/0/0.0.+1`). Those cells are "ragged
 * blobs": an int64 group count, int64 group offsets, then rows of
 * `[perm_idx?, v0, v1, ...]`. Undirected families store endpoints sorted and
 * prepend `perm_idx`, the Lehmer code of the writer's order.
 */

import type { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import {
  AsyncLru,
  mapConcurrent,
  SHARED_SIGNAL,
} from "#src/datasource/zarr-vectors/util.js";

type Offset = readonly number[];

function formatComponent(c: number) {
  return c > 0 ? `+${c}` : `${c}`;
}

function formatOffsets(offsets: readonly Offset[]) {
  return offsets.map((o) => o.map(formatComponent).join(".")).join("_");
}

/** Path of the array holding each chunk's own records, within a level. */
export function intraLinksPath(linkWidth: number) {
  return `links/0/${formatOffsets(new Array(linkWidth - 1).fill([0, 0, 0]))}`;
}

function parseOffsets(
  segment: string,
  linkWidth: number,
): Offset[] | undefined {
  const parts = segment.split("_");
  if (parts.length !== linkWidth - 1) return undefined;
  const offsets = parts.map((part) => part.split(".").map(Number));
  return offsets.every((o) => o.length === 3 && o.every(Number.isInteger))
    ? offsets
    : undefined;
}

export interface CrossChunkLinkEndpoint {
  readonly chunkCoords: number[];
  readonly vertexIndex: number;
}

/** One link, endpoints in the WRITER's order (`perm_idx` applied). */
export interface CrossChunkLinkRecord {
  readonly endpoints: CrossChunkLinkEndpoint[];
}

interface OffsetArray {
  readonly offsets: Offset[];
  readonly path: string;
  readonly hasPerm: boolean;
  readonly elementBytes: 4 | 8;
  readonly signed: boolean;
}

interface Discovery {
  readonly linkWidth: number;
  readonly arrays: OffsetArray[];
}

/** Inverse of zarr-vectors-py's `_lehmer_encode`. */
export function lehmerDecode(code: number, length: number): number[] {
  const available = Array.from({ length }, (_, i) => i);
  const out = new Array<number>(length);
  let fact = 1;
  for (let i = 2; i <= length; ++i) fact *= i;
  if (!(code >= 0 && code < fact)) {
    throw new Error(`perm_idx ${code} out of range for ${length} endpoints`);
  }
  for (let i = 0; i < length; ++i) {
    fact /= length - i;
    const idx = Math.floor(code / fact);
    code %= fact;
    out[i] = available[idx];
    available.splice(idx, 1);
  }
  return out;
}

function readInt(
  view: DataView,
  offset: number,
  bytes: 4 | 8,
  signed: boolean,
) {
  if (bytes === 4) {
    return signed ? view.getInt32(offset, true) : view.getUint32(offset, true);
  }
  const lo = view.getUint32(offset, true);
  const hi = view.getInt32(offset + 4, true);
  if (hi < -0x200000 || hi >= 0x200000) {
    throw new Error("link index outside the safe integer range");
  }
  return hi * 0x100000000 + lo;
}

/** Rows of a ragged-blob cell, flattened (`ncols` integers each). */
export function decodeRaggedRows(
  blob: Uint8Array,
  ncols: number,
  elementBytes: 4 | 8 = 8,
  signed = true,
): Float64Array {
  if (blob.byteLength < 8) return new Float64Array(0);
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const groups = readInt(view, 0, 8, true);
  if (groups <= 0) return new Float64Array(0);
  const header = 8 * (1 + groups);
  const dataLength = blob.byteLength - header;
  if (dataLength < 0 || dataLength % (ncols * elementBytes) !== 0) {
    throw new Error("links: cell is not a whole number of rows");
  }
  // Groups are contiguous and in order, so the rows are all the data.
  const out = new Float64Array(dataLength / elementBytes);
  for (let i = 0; i < out.length; ++i) {
    out[i] = readInt(view, header + i * elementBytes, elementBytes, signed);
  }
  return out;
}

/** Records of one cross-chunk cell, endpoints in the writer's order. */
export function decodeLinkCell(
  blob: Uint8Array,
  sourceChunk: readonly number[],
  array: Pick<OffsetArray, "offsets" | "hasPerm" | "elementBytes" | "signed">,
  linkWidth: number,
): CrossChunkLinkRecord[] {
  const permColumn = array.hasPerm ? 1 : 0;
  const ncols = permColumn + linkWidth;
  const rows = decodeRaggedRows(blob, ncols, array.elementBytes, array.signed);
  const chunkOf = [[...sourceChunk]];
  for (const offset of array.offsets) {
    chunkOf.push(sourceChunk.map((c, d) => c + offset[d]));
  }
  const records: CrossChunkLinkRecord[] = [];
  for (let base = 0; base < rows.length; base += ncols) {
    const sorted = chunkOf.map((chunkCoords, k) => ({
      chunkCoords,
      vertexIndex: rows[base + permColumn + k],
    }));
    const code = array.hasPerm ? rows[base] : 0;
    if (code === 0) {
      records.push({ endpoints: sorted });
      continue;
    }
    // The writer recorded `input[perm[i]] === sorted[i]`.
    const perm = lehmerDecode(code, linkWidth);
    const original = new Array<CrossChunkLinkEndpoint>(linkWidth);
    for (let i = 0; i < linkWidth; ++i) original[perm[i]] = sorted[i];
    records.push({ endpoints: original });
  }
  return records;
}

function neighbourOffsets(): Offset[] {
  const out: Offset[] = [];
  for (let x = -1; x <= 1; ++x) {
    for (let y = -1; y <= 1; ++y) {
      for (let z = -1; z <= 1; ++z) {
        if (x !== 0 || y !== 0 || z !== 0) out.push([x, y, z]);
      }
    }
  }
  return out;
}

export interface CrossChunkLinksOptions {
  cells: LevelCells;
  /** Child directory names of a level-relative path; throws if unsupported. */
  listDirectories?: (path: string) => Promise<string[]>;
  warn?: (message: string) => void;
}

export class CrossChunkLinks {
  private discovery: Promise<Discovery | null> | undefined;
  private cellCache = new AsyncLru<CrossChunkLinkRecord[]>(2048);
  constructor(private options: CrossChunkLinksOptions) {}

  /**
   * The level's cross-chunk arrays, or `null` when it has no links. Shared by
   * every chunk, so it runs without any one request's abort signal.
   */
  discover(): Promise<Discovery | null> {
    if (this.discovery === undefined) {
      const promise = this.runDiscovery();
      this.discovery = promise;
      promise.catch(() => {
        if (this.discovery === promise) this.discovery = undefined;
      });
    }
    return this.discovery;
  }

  private async runDiscovery(): Promise<Discovery | null> {
    const { cells, listDirectories, warn } = this.options;
    const family = await cells.readJson("links/0/zarr.json");
    if (family === undefined) return null;
    const attrs = family.attributes ?? {};
    const linkWidth = Number(attrs.link_width ?? 2);
    const directed = attrs.directed === true;
    const duplicated = attrs.store === "duplicate";
    let segments: string[] | undefined;
    try {
      segments = await listDirectories?.("links/0");
    } catch {
      segments = undefined;
    }
    if (segments === undefined) {
      if (linkWidth !== 2) {
        warn?.(
          "faces spanning chunks need directory listing to be found; " +
            "they are missing from this store's meshes",
        );
        return { linkWidth, arrays: [] };
      }
      // Without listing, probe the 26 neighbours (the writer's offsets).
      segments = neighbourOffsets().map((o) => formatOffsets([o]));
    }
    const arrays: (OffsetArray | undefined)[] = new Array(segments.length);
    await mapConcurrent(segments, 16, async (segment, i) => {
      const path = `links/0/${segment}`;
      const reader = await cells.reader(path).catch(() => undefined);
      if (reader === undefined) return;
      const a = reader.array.attributes;
      const offsets: Offset[] | undefined = Array.isArray(a.offsets)
        ? a.offsets.map((o: unknown[]) => o.map(Number))
        : parseOffsets(segment, linkWidth);
      if (offsets === undefined) return;
      if (offsets.every((o) => o.every((c) => c === 0))) return;
      const dtype = String(a.dtype ?? "int64");
      arrays[i] = {
        offsets,
        path,
        // Sorted (undirected) and duplicated families need the writer's order.
        hasPerm:
          a.has_perm !== undefined
            ? Boolean(a.has_perm)
            : duplicated || !directed,
        elementBytes: dtype.endsWith("32") ? 4 : 8,
        signed: !dtype.startsWith("u"),
      };
    });
    return {
      linkWidth,
      arrays: arrays.filter((a): a is OffsetArray => a !== undefined),
    };
  }

  private cellRecords(
    discovery: Discovery,
    array: OffsetArray,
    sourceChunk: readonly number[],
  ): Promise<CrossChunkLinkRecord[]> {
    const key = sourceChunk.join(".");
    return this.cellCache.get(`${array.path}|${key}`, async () => {
      const { cells } = this.options;
      if (!(await cells.mayHaveCell(array.path, key))) return [];
      const blob = await cells.readCell(array.path, key, SHARED_SIGNAL);
      if (blob === undefined) return [];
      return decodeLinkCell(blob, sourceChunk, array, discovery.linkWidth);
    });
  }

  /**
   * Links stored in `chunk`'s cells. Each cross-chunk link is stored by
   * exactly one of its chunks, so bridges are drawn once.
   */
  async linksOwnedBy(
    chunk: readonly number[],
  ): Promise<CrossChunkLinkRecord[]> {
    const discovery = await this.discover();
    if (discovery === null) return [];
    const perArray = await Promise.all(
      discovery.arrays.map((a) => this.cellRecords(discovery, a, chunk)),
    );
    return perArray.flat();
  }

  /** Links with any endpoint in `chunk`, from its cells and its neighbours'. */
  async linksTouching(
    chunk: readonly number[],
  ): Promise<CrossChunkLinkRecord[]> {
    const discovery = await this.discover();
    if (discovery === null) return [];
    const inChunk = (e: CrossChunkLinkEndpoint) =>
      e.chunkCoords.every((c, d) => c === chunk[d]);
    const reads: Promise<CrossChunkLinkRecord[]>[] = [];
    for (const array of discovery.arrays) {
      reads.push(this.cellRecords(discovery, array, chunk));
      for (const offset of array.offsets) {
        const source = chunk.map((c, d) => c - offset[d]);
        reads.push(
          this.cellRecords(discovery, array, source).then((records) =>
            records.filter((r) => r.endpoints.some(inChunk)),
          ),
        );
      }
    }
    const seen = new Set<string>();
    return (await Promise.all(reads)).flat().filter((r) => {
      const key = r.endpoints
        .map((e) => `${e.chunkCoords.join(".")}:${e.vertexIndex}`)
        .join("|");
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }
}
