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
 * @file Cross-chunk links (zarr-vectors 0.9 `links/<delta>/<offsets>`).
 *
 * A link whose endpoints lie in different chunks is stored once, in the cell
 * of its first endpoint's chunk (the source), in the array named by the other
 * endpoints' chunk offsets.  Each cell is a "ragged blob": an int64 group
 * count, int64 group offsets, then rows of `[perm_idx?, v0, v1, ...]`.
 * Undirected families store endpoints in canonical (lexicographic) order and
 * prepend `perm_idx`, the Lehmer code that restores the writer's order.
 *
 * Discovery of the offset arrays is shared by every chunk of a level, so it
 * runs without any one caller's abort signal: an early pan must not leave the
 * rest of the session without links.
 */

import type { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import type { LinkOffset } from "#src/datasource/zarr-vectors/links_paths.js";
import {
  formatOffsets,
  isIntra,
  linksGroupPath,
  linksHasPerm,
  parseOffsets,
} from "#src/datasource/zarr-vectors/links_paths.js";
import { mapConcurrent } from "#src/datasource/zarr-vectors/zarr_array.js";

export interface CrossChunkLinkEndpoint {
  readonly chunkCoords: number[];
  readonly vertexIndex: number;
}

/** One link, endpoints in the WRITER's order (any `perm_idx` applied). */
export interface CrossChunkLinkRecord {
  readonly endpoints: CrossChunkLinkEndpoint[];
}

export interface CrossChunkLinksTable {
  readonly linkWidth: number;
  readonly sidNdim: number;
  readonly records: CrossChunkLinkRecord[];
}

interface LinksFamily {
  readonly linkWidth: number;
  readonly sidNdim: number;
  readonly directed: boolean;
  readonly store: string;
}

interface OffsetArray {
  readonly offsets: LinkOffset[];
  readonly arrayPath: string;
  readonly hasPerm: boolean;
  readonly elementBytes: 4 | 8;
  readonly signed: boolean;
}

interface Discovery {
  readonly family: LinksFamily;
  readonly arrays: OffsetArray[];
}

export interface CrossChunkLinksOptions {
  cells: LevelCells;
  /** Child directory names of a level-relative path; throws if unsupported. */
  listDirectories?: (path: string) => Promise<string[]>;
  /** Level delta (0 = links within a level). */
  delta?: number;
  /** Chebyshev radius to probe for offset arrays when listing fails. */
  fallbackOffsetRadius?: number;
  warn?: (message: string) => void;
}

/** Inverse of the writer's `_lehmer_encode`. */
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
  elementBytes: 4 | 8,
  signed: boolean,
): number {
  if (elementBytes === 4) {
    return signed ? view.getInt32(offset, true) : view.getUint32(offset, true);
  }
  const lo = view.getUint32(offset, true);
  const hi = view.getInt32(offset + 4, true);
  if (hi < -0x200000 || hi >= 0x200000) {
    throw new Error("link index outside the safe integer range");
  }
  return hi * 0x100000000 + lo;
}

/**
 * Decodes a ragged-blob cell into flat rows of `ncols` integers.  The header
 * (group count and offsets) is always int64; rows use the array's dtype.
 */
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
  if (header > blob.byteLength) {
    throw new Error("links: ragged blob header exceeds the cell");
  }
  const dataLength = blob.byteLength - header;
  const rowBytes = ncols * elementBytes;
  if (dataLength % rowBytes !== 0) {
    throw new Error(
      `links: ${dataLength} data bytes is not a whole number of ${ncols}-column rows`,
    );
  }
  // Groups are contiguous and in order, so the rows are simply all the data.
  const numRows = dataLength / rowBytes;
  const out = new Float64Array(numRows * ncols);
  for (
    let i = 0, offset = header;
    i < out.length;
    ++i, offset += elementBytes
  ) {
    out[i] = readInt(view, offset, elementBytes, signed);
  }
  return out;
}

/** Decodes one cell of an offset array into records in the writer's order. */
export function decodeLinkCell(
  blob: Uint8Array,
  sourceChunk: readonly number[],
  array: Pick<OffsetArray, "offsets" | "hasPerm" | "elementBytes" | "signed">,
  linkWidth: number,
): CrossChunkLinkRecord[] {
  const permColumn = array.hasPerm ? 1 : 0;
  const ncols = permColumn + linkWidth;
  const rows = decodeRaggedRows(blob, ncols, array.elementBytes, array.signed);
  const numRows = rows.length / ncols;
  const chunkOf = new Array<number[]>(linkWidth);
  chunkOf[0] = [...sourceChunk];
  for (let k = 1; k < linkWidth; ++k) {
    const offset = array.offsets[k - 1];
    chunkOf[k] = sourceChunk.map((c, d) => c + (offset?.[d] ?? 0));
  }
  const records: CrossChunkLinkRecord[] = new Array(numRows);
  const permCache = new Map<number, number[]>();
  for (let r = 0; r < numRows; ++r) {
    const base = r * ncols;
    const sorted: CrossChunkLinkEndpoint[] = new Array(linkWidth);
    for (let k = 0; k < linkWidth; ++k) {
      sorted[k] = {
        chunkCoords: chunkOf[k],
        vertexIndex: rows[base + permColumn + k],
      };
    }
    if (!array.hasPerm || rows[base] === 0) {
      records[r] = { endpoints: sorted };
      continue;
    }
    const code = rows[base];
    let perm = permCache.get(code);
    if (perm === undefined) {
      perm = lehmerDecode(code, linkWidth);
      permCache.set(code, perm);
    }
    // The writer recorded `input[perm[i]] === sorted[i]`.
    const original = new Array<CrossChunkLinkEndpoint>(linkWidth);
    for (let i = 0; i < linkWidth; ++i) original[perm[i]] = sorted[i];
    records[r] = { endpoints: original };
  }
  return records;
}

function boundedOffsets(sidNdim: number, radius: number): LinkOffset[] {
  const out: LinkOffset[] = [];
  const rec = (dim: number, acc: number[]) => {
    if (dim === sidNdim) {
      if (acc.some((c) => c !== 0)) out.push(acc.slice());
      return;
    }
    for (let c = -radius; c <= radius; ++c) rec(dim + 1, [...acc, c]);
  };
  rec(0, []);
  return out;
}

/** Bounded cache of decoded link cells, keyed by array and cell. */
class LinkCellCache {
  private entries = new Map<string, Promise<CrossChunkLinkRecord[]>>();
  constructor(private maxEntries: number) {}
  get(key: string, load: () => Promise<CrossChunkLinkRecord[]>) {
    let entry = this.entries.get(key);
    if (entry !== undefined) {
      this.entries.delete(key);
      this.entries.set(key, entry);
      return entry;
    }
    entry = load();
    this.entries.set(key, entry);
    const self = entry;
    entry.catch(() => {
      if (this.entries.get(key) === self) this.entries.delete(key);
    });
    if (this.entries.size > this.maxEntries) {
      this.entries.delete(this.entries.keys().next().value!);
    }
    return entry;
  }
}

export class CrossChunkLinks {
  private discovery: Promise<Discovery | null> | undefined;
  private cellCache = new LinkCellCache(2048);
  constructor(private options: CrossChunkLinksOptions) {}

  private get delta() {
    return this.options.delta ?? 0;
  }

  /** Offset arrays of the family, or `null` when the level has none. */
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
    const { cells } = this.options;
    const groupPath = linksGroupPath(this.delta);
    const groupJson = await cells.readJson(`${groupPath}/zarr.json`);
    if (groupJson === undefined) return null;
    const attrs = groupJson.attributes ?? {};
    const family: LinksFamily = {
      linkWidth: Number(attrs.link_width ?? 2),
      sidNdim: Number(attrs.sid_ndim ?? 3),
      directed: attrs.directed === true,
      store: String(attrs.store ?? "canonical"),
    };
    let segments: string[] | undefined;
    if (this.options.listDirectories !== undefined) {
      try {
        segments = await this.options.listDirectories(groupPath);
      } catch {
        segments = undefined;
      }
    }
    if (segments === undefined) {
      if (family.linkWidth !== 2) {
        this.options.warn?.(
          `cannot discover link_width=${family.linkWidth} cross-chunk links ` +
            "without directory listing; faces spanning chunks will be missing",
        );
        return { family, arrays: [] };
      }
      const radius = this.options.fallbackOffsetRadius ?? 1;
      segments = boundedOffsets(family.sidNdim, radius).map((o) =>
        formatOffsets([o]),
      );
      this.options.warn?.(
        "object listing is unavailable, so cross-chunk links were found by " +
          `probing offsets within ${radius} chunk(s); longer links are missed`,
      );
    }
    const arrays: (OffsetArray | undefined)[] = new Array(segments.length);
    await mapConcurrent(segments, 16, async (segment, i) => {
      const arrayPath = `${groupPath}/${segment}`;
      const reader = await cells.reader(arrayPath).catch(() => undefined);
      if (reader === undefined) return;
      const a = reader.array.attributes;
      let offsets: LinkOffset[];
      if (Array.isArray(a.offsets)) {
        offsets = a.offsets.map((o: unknown[]) => o.map(Number));
      } else {
        try {
          offsets = parseOffsets(segment, {
            sidNdim: family.sidNdim,
            linkWidth: family.linkWidth,
          });
        } catch {
          return;
        }
      }
      if (isIntra(offsets)) return;
      const dtype = String(a.dtype ?? "int64");
      const hasPerm =
        a.has_perm !== undefined
          ? Boolean(a.has_perm)
          : linksHasPerm(offsets, {
              delta: this.delta,
              directed: family.directed,
              store: family.store,
            });
      arrays[i] = {
        offsets,
        arrayPath,
        hasPerm,
        elementBytes: dtype.endsWith("32") ? 4 : 8,
        signed: !dtype.startsWith("u"),
      };
    });
    return {
      family,
      arrays: arrays.filter((a): a is OffsetArray => a !== undefined),
    };
  }

  private cellRecords(
    discovery: Discovery,
    array: OffsetArray,
    sourceChunk: readonly number[],
    signal: AbortSignal,
  ): Promise<CrossChunkLinkRecord[]> {
    const key = `${array.arrayPath}|${sourceChunk.join(".")}`;
    return this.cellCache.get(key, async () => {
      const { cells } = this.options;
      if (!(await cells.mayHaveCell(array.arrayPath, sourceChunk.join(".")))) {
        return [];
      }
      // Shared between chunks via the cache, so not tied to `signal` beyond
      // the first reader; a cancelled read is evicted and retried.
      const blob = await cells.readCell(
        array.arrayPath,
        sourceChunk.join("."),
        signal,
      );
      if (blob === undefined) return [];
      return decodeLinkCell(
        blob,
        sourceChunk,
        array,
        discovery.family.linkWidth,
      );
    });
  }

  /**
   * Links whose FIRST endpoint lies in `chunk` (one read per offset array, all
   * concurrent).  Every cross-chunk link is owned by exactly one chunk this
   * way, so bridging edges are never drawn twice.
   */
  async linksOwnedBy(
    chunk: readonly number[],
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    const discovery = await this.discover();
    if (discovery === null) return undefined;
    const perArray = await Promise.all(
      discovery.arrays.map((array) =>
        this.cellRecords(discovery, array, chunk, signal),
      ),
    );
    return {
      linkWidth: discovery.family.linkWidth,
      sidNdim: discovery.family.sidNdim,
      records: perArray.flat(),
    };
  }

  /**
   * Links with ANY endpoint in `chunk`: its own cells plus the cells of each
   * neighbour whose offset points back at it.  Needed where the non-owning
   * chunk must also know about a link (the SWC linked layout, where a stored
   * link removes an implied edge on the child's side).
   */
  async linksTouching(
    chunk: readonly number[],
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    const discovery = await this.discover();
    if (discovery === null) return undefined;
    const reads: Promise<CrossChunkLinkRecord[]>[] = [];
    for (const array of discovery.arrays) {
      reads.push(this.cellRecords(discovery, array, chunk, signal));
      for (const offset of array.offsets) {
        const source = chunk.map((c, d) => c - offset[d]);
        reads.push(
          this.cellRecords(discovery, array, source, signal).then((records) =>
            records.filter((r) =>
              r.endpoints.some((e) =>
                e.chunkCoords.every((c, d) => c === chunk[d]),
              ),
            ),
          ),
        );
      }
    }
    const records = (await Promise.all(reads)).flat();
    // A record can arrive through several offsets of the same array; dedupe.
    const seen = new Set<string>();
    const unique = records.filter((r) => {
      const key = r.endpoints
        .map((e) => `${e.chunkCoords.join(".")}:${e.vertexIndex}`)
        .join("|");
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    return {
      linkWidth: discovery.family.linkWidth,
      sidNdim: discovery.family.sidNdim,
      records: unique,
    };
  }
}
