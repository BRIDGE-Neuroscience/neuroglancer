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
 * @file Reads one zarr v3 array of a zarr-vectors store, decoding each chunk
 * through the codec chain the array itself declares.
 *
 * A zarr-vectors store mixes array kinds: per-chunk geometry arrays are
 * variable-length (`vlen-bytes`, one element per spatial cell), object-level
 * arrays are fixed-size columns, and either may be compressed (zstd, blosc,
 * gzip, zlib) and/or sharded.  Codecs and sharding are per ARRAY, not per
 * store or level: `zvtools attach` and `build_pyramid` routinely leave one
 * level holding sharded and unsharded, compressed and raw arrays side by
 * side.  So every array is opened from its own `zarr.json`.
 *
 * Decoding reuses Neuroglancer's zarr codec registry.  The array -> bytes
 * step (`bytes`, `vlen-bytes`) is replaced by a pass-through codec so the
 * registry only runs the bytes -> bytes stages (decompression, checksums);
 * this module then interprets the raw element bytes itself, which is what lets
 * one code path read 64-bit columns (no Neuroglancer `DataType`) and vlen
 * cells alike.  Sharding is resolved here rather than through the registry's
 * sharded kvstore so the same reader runs on the main thread and in the chunk
 * worker, and so a raw cell can be range-read row by row.
 *
 * Everything here is context-free: callers supply a `read(path, options)`
 * function bound to the store root.
 */

import "#src/datasource/zarr/codec/blosc/decode.js";
import "#src/datasource/zarr/codec/blosc/resolve.js";
import "#src/datasource/zarr/codec/bytes/decode.js";
import "#src/datasource/zarr/codec/bytes/resolve.js";
import "#src/datasource/zarr/codec/crc32c/decode.js";
import "#src/datasource/zarr/codec/crc32c/resolve.js";
import "#src/datasource/zarr/codec/gzip/decode.js";
import "#src/datasource/zarr/codec/gzip/resolve.js";
import "#src/datasource/zarr/codec/zstd/decode.js";
import "#src/datasource/zarr/codec/zstd/resolve.js";
import {
  decodeArray,
  registerCodec as registerDecodeCodec,
} from "#src/datasource/zarr/codec/decode.js";
import type { CodecChainSpec } from "#src/datasource/zarr/codec/index.js";
import { CodecKind } from "#src/datasource/zarr/codec/index.js";
import {
  parseCodecChainSpec,
  registerCodec as registerResolveCodec,
} from "#src/datasource/zarr/codec/resolve.js";
import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  ELEMENT_BYTES,
  isElementType,
} from "#src/datasource/zarr-vectors/dtype.js";
import { AsyncLru, mapConcurrent } from "#src/datasource/zarr-vectors/util.js";
import type { ByteRangeRequest } from "#src/kvstore/index.js";
import { DataType } from "#src/util/data_type.js";

/**
 * Name of the pass-through array -> bytes codec.  Namespaced so registering
 * it cannot change how the generic zarr datasource treats real arrays.
 */
const PASSTHROUGH_CODEC = "zarr-vectors.passthrough";

registerResolveCodec({
  name: PASSTHROUGH_CODEC,
  kind: CodecKind.arrayToBytes,
  resolve(configuration: unknown) {
    return { configuration };
  },
  getDecodedArrayLayoutInfo(_configuration, decodedArrayInfo) {
    return {
      physicalToLogicalDimension: Array.from(
        decodedArrayInfo.chunkShape,
        (_, i) => i,
      ),
      readChunkShape: decodedArrayInfo.chunkShape,
    };
  },
});

registerDecodeCodec({
  name: PASSTHROUGH_CODEC,
  kind: CodecKind.arrayToBytes,
  async decode(_configuration, _decodedArrayInfo, encoded) {
    return encoded;
  },
});

/** zarr-python spells some codecs with a `numcodecs.` prefix. */
const CODEC_ALIASES: Record<string, string> = {
  "numcodecs.zlib": "zlib",
  "numcodecs.gzip": "gzip",
  "numcodecs.zstd": "zstd",
  "numcodecs.blosc": "blosc",
};

/** Codecs zarr-python can emit that no decoder in this tree implements. */
const UNSUPPORTED_CODECS = new Set([
  "numcodecs.lz4",
  "numcodecs.bz2",
  "numcodecs.lzma",
  "numcodecs.shuffle",
  "numcodecs.delta",
  "numcodecs.quantize",
  "numcodecs.bitround",
  "numcodecs.pcodec",
  "transpose",
]);

export type ZarrArrayRead = (
  path: string,
  options: { signal?: AbortSignal; byteRange?: ByteRangeRequest },
) => Promise<Uint8Array | undefined>;

/**
 * Element type of an array.  Fixed-size dtypes are interpreted by callers
 * from `elementBytes`; `vlen` marks variable-length bytes or strings.
 */
export type ZarrElementType = ElementType | "vlen";

function parseElementType(dataType: unknown): ZarrElementType {
  if (typeof dataType !== "string") {
    throw new Error(`unsupported zarr data_type ${JSON.stringify(dataType)}`);
  }
  switch (dataType) {
    case "variable_length_bytes":
    case "bytes":
    case "string":
    case "variable_length_utf8":
      return "vlen";
  }
  if (isElementType(dataType)) return dataType;
  throw new Error(`unsupported zarr data_type ${JSON.stringify(dataType)}`);
}

export interface ZarrShardingSpec {
  /** Shape of the read (inner) chunk, in elements. */
  innerChunkShape: number[];
  /** Number of inner chunks per shard along each dimension. */
  subGridShape: number[];
  indexCodecs: CodecChainSpec;
  indexAtStart: boolean;
  indexByteLength: number;
}

export interface ZarrArray {
  /** Path of the array within the store, without a trailing slash. */
  path: string;
  shape: number[];
  elementType: ZarrElementType;
  /** Bytes per element; 0 for variable-length arrays. */
  elementBytes: number;
  bigEndian: boolean;
  /** Shape of the unit the codec chain decodes (the inner chunk if sharded). */
  readChunkShape: number[];
  /** Shape of one stored key (the shard if sharded). */
  storedChunkShape: number[];
  keyPrefix: string;
  separator: string;
  /**
   * zarr-vectors `chunk_grid_origin`: the spatial cell of element 0.  Lets a
   * store's grid start at negative cell coordinates.
   */
  origin: number[];
  /** Bytes -> bytes stages applied to each read chunk. */
  codecs: CodecChainSpec;
  /** True when decoding a read chunk is the identity (range reads allowed). */
  raw: boolean;
  sharding: ZarrShardingSpec | undefined;
  fillValue: unknown;
  attributes: Record<string, any>;
  /**
   * zarr-vectors `nonempty_chunks`: cell keys (`"i.j.k"`, spatial coordinates)
   * that hold data.  `undefined` when the writer did not record it.
   */
  nonemptyCells: Set<string> | undefined;
}

function verifyIntArray(value: unknown, what: string): number[] {
  if (
    !Array.isArray(value) ||
    !value.every((x) => Number.isInteger(x) && x >= 0)
  ) {
    throw new Error(`invalid ${what}: ${JSON.stringify(value)}`);
  }
  return value as number[];
}

/**
 * Rewrites a codec list so its array -> bytes codec is the pass-through and
 * names are canonical.  Recurses into `sharding_indexed` so the caller can
 * pull its configuration apart separately.
 */
function rewriteCodecs(codecs: unknown, where: string): any[] {
  if (!Array.isArray(codecs) || codecs.length === 0) {
    throw new Error(`${where}: missing codecs`);
  }
  return codecs.map((codec: any) => {
    const name: unknown = typeof codec === "string" ? codec : codec?.name;
    if (typeof name !== "string") {
      throw new Error(`${where}: invalid codec ${JSON.stringify(codec)}`);
    }
    if (UNSUPPORTED_CODECS.has(name)) {
      throw new Error(
        `${where}: codec ${JSON.stringify(name)} is not supported by this ` +
          "viewer; re-encode the store with zstd, blosc, gzip or no compressor",
      );
    }
    const configuration = typeof codec === "string" ? {} : codec.configuration;
    switch (name) {
      case "bytes":
      case "vlen-bytes":
      case "vlen-utf8":
        return {
          name: PASSTHROUGH_CODEC,
          configuration: { codec: name, ...configuration },
        };
    }
    return {
      name: CODEC_ALIASES[name] ?? name,
      configuration: configuration ?? {},
    };
  });
}

function parseEndian(codecs: unknown[]): boolean {
  for (const codec of codecs as any[]) {
    if (codec?.name === "bytes") {
      return codec.configuration?.endian === "big";
    }
  }
  return false;
}

function parseChain(codecs: any[], chunkShape: number[]): CodecChainSpec {
  return parseCodecChainSpec(codecs, {
    dataType: DataType.UINT8,
    chunkShape,
  });
}

function parseNonemptyCells(raw: unknown): Set<string> | undefined {
  if (!Array.isArray(raw)) return undefined;
  const out = new Set<string>();
  for (const entry of raw) {
    if (typeof entry === "string") {
      out.add(entry);
    } else if (Array.isArray(entry)) {
      out.add(entry.join("."));
    }
  }
  return out;
}

/** Parses an array's `zarr.json`.  Throws on anything this reader can't decode. */
export function parseZarrArrayMetadata(path: string, json: any): ZarrArray {
  const where = `zarr-vectors array ${JSON.stringify(path)}`;
  if (json?.zarr_format !== 3 || json?.node_type !== "array") {
    throw new Error(`${where} is not a zarr v3 array`);
  }
  const shape = verifyIntArray(json.shape, `${where} shape`);
  const rank = shape.length;
  const elementType = parseElementType(json.data_type);
  const elementBytes = elementType === "vlen" ? 0 : ELEMENT_BYTES[elementType];
  if (json.chunk_grid?.name !== "regular") {
    throw new Error(`${where}: only regular chunk grids are supported`);
  }
  const storedChunkShape = verifyIntArray(
    json.chunk_grid?.configuration?.chunk_shape,
    `${where} chunk_shape`,
  );
  if (storedChunkShape.length !== rank) {
    throw new Error(`${where}: chunk_shape rank differs from shape`);
  }

  let keyPrefix = "c/";
  let separator = "/";
  const keyEncoding = json.chunk_key_encoding;
  if (keyEncoding !== undefined) {
    const sep = keyEncoding.configuration?.separator;
    if (keyEncoding.name === "v2") {
      keyPrefix = "";
      separator = typeof sep === "string" ? sep : ".";
    } else {
      separator = typeof sep === "string" ? sep : "/";
      keyPrefix = `c${separator}`;
    }
  }

  const rawCodecs = json.codecs;
  let sharding: ZarrShardingSpec | undefined;
  let chunkCodecs: any[];
  let readChunkShape = storedChunkShape;
  let bigEndian: boolean;
  if (
    Array.isArray(rawCodecs) &&
    rawCodecs.length > 0 &&
    rawCodecs[0]?.name === "sharding_indexed"
  ) {
    if (rawCodecs.length !== 1) {
      throw new Error(`${where}: codecs after sharding_indexed`);
    }
    const config = rawCodecs[0].configuration ?? {};
    const innerChunkShape = verifyIntArray(
      config.chunk_shape,
      `${where} sharding chunk_shape`,
    );
    if (
      innerChunkShape.length !== rank ||
      innerChunkShape.some((n, i) => n === 0 || storedChunkShape[i] % n !== 0)
    ) {
      throw new Error(`${where}: shard shape is not a multiple of its chunks`);
    }
    if (config.codecs?.[0]?.name === "sharding_indexed") {
      throw new Error(`${where}: nested sharding is not supported`);
    }
    const subGridShape = storedChunkShape.map((n, i) => n / innerChunkShape[i]);
    const indexCodecs = parseCodecChainSpec(
      config.index_codecs ?? [
        { name: "bytes", configuration: { endian: "little" } },
        { name: "crc32c" },
      ],
      { dataType: DataType.UINT64, chunkShape: [...subGridShape, 2] },
    );
    const indexByteLength =
      indexCodecs.encodedSize[indexCodecs.encodedSize.length - 1];
    if (indexByteLength === undefined) {
      throw new Error(`${where}: shard index codecs must have a fixed size`);
    }
    const location = config.index_location ?? "end";
    if (location !== "start" && location !== "end") {
      throw new Error(`${where}: invalid index_location ${location}`);
    }
    sharding = {
      innerChunkShape,
      subGridShape,
      indexCodecs,
      indexAtStart: location === "start",
      indexByteLength,
    };
    readChunkShape = innerChunkShape;
    bigEndian = parseEndian(config.codecs ?? []);
    chunkCodecs = rewriteCodecs(config.codecs, where);
  } else {
    bigEndian = parseEndian(rawCodecs ?? []);
    chunkCodecs = rewriteCodecs(rawCodecs, where);
  }
  const codecs = parseChain(chunkCodecs, readChunkShape);
  if (codecs[CodecKind.arrayToArray].length !== 0) {
    throw new Error(`${where}: array -> array codecs are not supported`);
  }
  const attributes = json.attributes ?? {};
  const originRaw = attributes.chunk_grid_origin;
  const origin =
    Array.isArray(originRaw) && originRaw.length === rank
      ? originRaw.map((x: unknown) => Number(x))
      : new Array<number>(rank).fill(0);
  return {
    path: path.replace(/\/+$/, ""),
    shape,
    elementType,
    elementBytes,
    bigEndian,
    readChunkShape,
    storedChunkShape,
    keyPrefix,
    separator,
    origin,
    codecs,
    raw: codecs[CodecKind.bytesToBytes].length === 0,
    sharding,
    fillValue: json.fill_value,
    attributes,
    nonemptyCells: parseNonemptyCells(attributes.nonempty_chunks),
  };
}

function joinPath(base: string, rel: string) {
  if (base === "") return rel;
  return `${base}/${rel}`;
}

/** Reads and parses `<path>/zarr.json`; `undefined` if the array is absent. */
export async function openZarrArray(
  read: ZarrArrayRead,
  path: string,
  signal?: AbortSignal,
): Promise<ZarrArray | undefined> {
  const bytes = await read(joinPath(path, "zarr.json"), { signal });
  if (bytes === undefined) return undefined;
  const json = JSON.parse(new TextDecoder().decode(bytes));
  if (json?.node_type !== "array") return undefined;
  return parseZarrArrayMetadata(path, json);
}

/**
 * Decoded shard indexes, shared by every array of a store. An index is 16
 * bytes per inner chunk (1 KiB for a 4x4x4 shard).
 */
export class ShardIndexCache extends AsyncLru<BigUint64Array | undefined> {
  constructor() {
    super(4096);
  }
}

/** Reads this close together are merged; the bytes between are discarded. */
const COALESCE_MAX_GAP = 8 * 1024;
/** A merged read stops growing here, so no cell waits on a huge one. */
const COALESCE_MAX_LENGTH = 8 * 1024 * 1024;

interface PendingRange {
  offset: number;
  length: number;
  signal: AbortSignal | undefined;
  resolve: (bytes: Uint8Array | undefined) => void;
  reject: (reason: unknown) => void;
}

/**
 * Merges byte-range reads of one key issued in the same task, such as the
 * cells of one shard that several chunks ask for at once, into fewer
 * requests.  Each caller gets its own copy: decoders take over their input's
 * buffer.
 */
export function coalesceRangeReads(
  read: ZarrArrayRead,
  maxGap = COALESCE_MAX_GAP,
  maxLength = COALESCE_MAX_LENGTH,
): ZarrArrayRead {
  let pending = new Map<string, PendingRange[]>();
  let scheduled = false;

  const issue = (path: string, group: PendingRange[]) => {
    if (group.length === 1) {
      const [r] = group;
      read(path, {
        signal: r.signal,
        byteRange: { offset: r.offset, length: r.length },
      }).then(r.resolve, r.reject);
      return;
    }
    const start = group[0].offset;
    const end = Math.max(...group.map((r) => r.offset + r.length));
    // Aborted only once every caller has given up.
    const controller = new AbortController();
    let live = group.length;
    for (const r of group) {
      r.signal?.addEventListener(
        "abort",
        () => {
          if (--live === 0) controller.abort();
        },
        { once: true },
      );
    }
    read(path, {
      signal: controller.signal,
      byteRange: { offset: start, length: end - start },
    }).then(
      (bytes) => {
        for (const r of group) {
          const from = r.offset - start;
          if (bytes === undefined) {
            r.resolve(undefined);
          } else if (from + r.length > bytes.length) {
            r.reject(new Error(`${path}: short range read`));
          } else {
            r.resolve(bytes.slice(from, from + r.length));
          }
        }
      },
      (e) => {
        for (const r of group) r.reject(e);
      },
    );
  };

  const flush = () => {
    scheduled = false;
    const batch = pending;
    pending = new Map();
    for (const [path, ranges] of batch) {
      const live = ranges
        .filter((r) => !r.signal?.aborted)
        .sort((a, b) => a.offset - b.offset);
      let group: PendingRange[] = [];
      let start = 0;
      let end = 0;
      for (const r of live) {
        const newEnd = Math.max(end, r.offset + r.length);
        if (
          group.length > 0 &&
          r.offset <= end + maxGap &&
          newEnd - start <= maxLength
        ) {
          group.push(r);
          end = newEnd;
          continue;
        }
        if (group.length > 0) issue(path, group);
        group = [r];
        start = r.offset;
        end = r.offset + r.length;
      }
      if (group.length > 0) issue(path, group);
    }
  };

  return (path, options) => {
    const range = options.byteRange;
    if (range === undefined || "suffixLength" in range) {
      return read(path, options);
    }
    const { signal } = options;
    signal?.throwIfAborted();
    return new Promise((resolve, reject) => {
      signal?.addEventListener("abort", () => reject(signal.reason), {
        once: true,
      });
      let list = pending.get(path);
      if (list === undefined) pending.set(path, (list = []));
      list.push({ ...range, signal, resolve, reject });
      if (!scheduled) {
        scheduled = true;
        setTimeout(flush, 0);
      }
    });
  };
}

const MISSING = 0xffffffffffffffffn;

/** Byte range of one stored chunk inside its key, if it exists. */
interface ChunkLocation {
  key: string;
  /** Offset/length inside `key`; `undefined` means the whole object. */
  range: { offset: number; length: number } | undefined;
}

/** Inputs to decode one element of a variable-length array. */
interface VlenElement {
  chunkIndex: number[];
  elementInChunk: number;
}

export class ZarrArrayReader {
  constructor(
    readonly array: ZarrArray,
    private readonly read: ZarrArrayRead,
    private readonly shardIndexes: ShardIndexCache,
  ) {}

  get rank() {
    return this.array.shape.length;
  }

  /** Converts spatial cell coordinates to an element index, or undefined if outside. */
  cellToElement(cell: ArrayLike<number>): number[] | undefined {
    const { shape, origin } = this.array;
    const out = new Array<number>(shape.length);
    for (let i = 0; i < shape.length; ++i) {
      const e = cell[i] - origin[i];
      if (!(e >= 0 && e < shape[i])) return undefined;
      out[i] = e;
    }
    return out;
  }

  /** True unless the array records `nonempty_chunks` and `cell` is not among them. */
  mayHaveCell(cell: ArrayLike<number>): boolean {
    const { nonemptyCells } = this.array;
    if (nonemptyCells === undefined) return true;
    return nonemptyCells.has(Array.prototype.join.call(cell, "."));
  }

  private storedKey(storedIndex: number[]) {
    const { array } = this;
    return joinPath(
      array.path,
      array.keyPrefix + storedIndex.join(array.separator),
    );
  }

  private async locate(
    chunkIndex: number[],
    signal: AbortSignal | undefined,
  ): Promise<ChunkLocation | undefined> {
    const { sharding } = this.array;
    if (sharding === undefined) {
      return { key: this.storedKey(chunkIndex), range: undefined };
    }
    const { subGridShape } = sharding;
    const rank = chunkIndex.length;
    const shardIndex = new Array<number>(rank);
    let entry = 0;
    for (let i = 0; i < rank; ++i) {
      shardIndex[i] = Math.floor(chunkIndex[i] / subGridShape[i]);
      entry = entry * subGridShape[i] + (chunkIndex[i] % subGridShape[i]);
    }
    const key = this.storedKey(shardIndex);
    const index = await this.shardIndexes.get(key, () =>
      this.readShardIndex(key, signal),
    );
    if (index === undefined) return undefined;
    const offset = index[2 * entry];
    const length = index[2 * entry + 1];
    if (offset === MISSING && length === MISSING) return undefined;
    return { key, range: { offset: Number(offset), length: Number(length) } };
  }

  private async readShardIndex(
    key: string,
    signal: AbortSignal | undefined,
  ): Promise<BigUint64Array | undefined> {
    const sharding = this.array.sharding!;
    const byteRange: ByteRangeRequest = sharding.indexAtStart
      ? { offset: 0, length: sharding.indexByteLength }
      : { suffixLength: sharding.indexByteLength };
    const encoded = await this.read(key, { signal, byteRange });
    if (encoded === undefined) return undefined;
    const decoded = await decodeArray(
      sharding.indexCodecs,
      copyIfShared(encoded),
      signal ?? new AbortController().signal,
    );
    return new BigUint64Array(
      decoded.buffer,
      decoded.byteOffset,
      decoded.byteLength / 8,
    );
  }

  /** Reads and decodes one read chunk; `undefined` if absent. */
  async readChunk(
    chunkIndex: number[],
    signal?: AbortSignal,
  ): Promise<Uint8Array | undefined> {
    const location = await this.locate(chunkIndex, signal);
    if (location === undefined) return undefined;
    const encoded = await this.read(location.key, {
      signal,
      byteRange: location.range,
    });
    if (encoded === undefined) return undefined;
    if (this.array.raw) return encoded;
    const decoded = await decodeArray(
      this.array.codecs,
      copyIfShared(encoded),
      signal ?? new AbortController().signal,
    );
    return new Uint8Array(
      decoded.buffer,
      decoded.byteOffset,
      decoded.byteLength,
    );
  }

  private vlenElement(element: number[]): VlenElement {
    const { readChunkShape } = this.array;
    const chunkIndex = new Array<number>(element.length);
    let elementInChunk = 0;
    for (let i = 0; i < element.length; ++i) {
      chunkIndex[i] = Math.floor(element[i] / readChunkShape[i]);
      elementInChunk =
        elementInChunk * readChunkShape[i] + (element[i] % readChunkShape[i]);
    }
    return { chunkIndex, elementInChunk };
  }

  /**
   * Payload of the variable-length element at spatial cell `cell`, or
   * `undefined` when the cell is empty.
   */
  async readCell(
    cell: ArrayLike<number>,
    signal?: AbortSignal,
  ): Promise<Uint8Array | undefined> {
    if (this.array.elementType !== "vlen") {
      throw new Error(`${this.array.path} is not a variable-length array`);
    }
    if (!this.mayHaveCell(cell)) return undefined;
    const element = this.cellToElement(cell);
    if (element === undefined) return undefined;
    const { chunkIndex, elementInChunk } = this.vlenElement(element);
    const bytes = await this.readChunk(chunkIndex, signal);
    if (bytes === undefined) return undefined;
    const payload = vlenElementAt(bytes, elementInChunk);
    return payload.length === 0 ? undefined : payload;
  }

  /**
   * Reads bytes `[offset, offset + length)` of the payload of the
   * variable-length element at `cell` WITHOUT fetching the rest of the cell.
   * Only possible for a raw array whose read chunk holds exactly one element
   * (then the payload starts 8 bytes in: element count, then its length).
   * Returns `null` when the array is not range-addressable, so callers fall
   * back to {@link readCell}; `undefined` when the cell is empty.
   */
  async readCellRange(
    cell: ArrayLike<number>,
    offset: number,
    length: number,
    signal?: AbortSignal,
  ): Promise<Uint8Array | undefined | null> {
    const { array } = this;
    if (
      array.elementType !== "vlen" ||
      !array.raw ||
      array.readChunkShape.some((n) => n !== 1)
    ) {
      return null;
    }
    if (!this.mayHaveCell(cell)) return undefined;
    const element = this.cellToElement(cell);
    if (element === undefined) return undefined;
    const location = await this.locate(element, signal);
    if (location === undefined) return undefined;
    const start = (location.range?.offset ?? 0) + VLEN_SINGLE_HEADER + offset;
    if (
      location.range !== undefined &&
      VLEN_SINGLE_HEADER + offset + length > location.range.length
    ) {
      throw new Error(
        `${array.path}: range read past the end of cell ${Array.from(cell)}`,
      );
    }
    const bytes = await this.read(location.key, {
      signal,
      byteRange: { offset: start, length },
    });
    if (bytes === undefined) return undefined;
    if (bytes.length !== length) {
      throw new Error(
        `${array.path}: short range read (${bytes.length} of ${length} bytes)`,
      );
    }
    return bytes;
  }

  /**
   * Reads elements `[start, end)` of a 1-D (or row-major N-D, first axis)
   * fixed-size array as raw little-endian bytes, filling absent chunks with
   * the fill value.  `rowElements` is the product of the trailing dimensions.
   */
  async readRows(
    start: number,
    end: number,
    signal?: AbortSignal,
    concurrency = 8,
  ): Promise<Uint8Array> {
    const { array } = this;
    if (array.elementType === "vlen") {
      throw new Error(`${array.path}: readRows needs a fixed-size array`);
    }
    const rowElements = array.shape.slice(1).reduce((a, b) => a * b, 1);
    for (let i = 1; i < array.shape.length; ++i) {
      if (array.readChunkShape[i] !== array.shape[i]) {
        throw new Error(
          `${array.path}: chunked along a trailing dimension; unsupported`,
        );
      }
    }
    const rowBytes = rowElements * array.elementBytes;
    end = Math.min(end, array.shape[0]);
    const out = new Uint8Array(Math.max(0, end - start) * rowBytes);
    if (end <= start) return out;
    const fill = fillBytes(array);
    const chunkRows = array.readChunkShape[0];
    const first = Math.floor(start / chunkRows);
    const last = Math.floor((end - 1) / chunkRows);
    const chunkIndices: number[] = [];
    for (let c = first; c <= last; ++c) chunkIndices.push(c);
    const trailing = new Array<number>(array.shape.length - 1).fill(0);
    await mapConcurrent(chunkIndices, concurrency, async (c) => {
      const bytes = await this.readChunk([c, ...trailing], signal);
      const chunkStart = c * chunkRows;
      const lo = Math.max(start, chunkStart);
      const hi = Math.min(end, chunkStart + chunkRows);
      const dst = (lo - start) * rowBytes;
      if (bytes === undefined) {
        fillRange(out, dst, (hi - lo) * rowBytes, fill);
        return;
      }
      const expected = chunkRows * rowBytes;
      if (bytes.length !== expected) {
        throw new Error(
          `${array.path}: chunk ${c} decoded to ${bytes.length} bytes, ` +
            `expected ${expected}`,
        );
      }
      out.set(
        bytes.subarray(
          (lo - chunkStart) * rowBytes,
          (hi - chunkStart) * rowBytes,
        ),
        dst,
      );
    });
    if (array.bigEndian && array.elementBytes > 1) {
      swapEndian(out, array.elementBytes);
    }
    return out;
  }

  /** All rows of a fixed-size array. */
  readAllRows(signal?: AbortSignal): Promise<Uint8Array> {
    return this.readRows(0, this.array.shape[0], signal);
  }

  /** Decodes every element of a 1-D variable-length array chunk. */
  async readVlenRows(
    start: number,
    end: number,
    signal?: AbortSignal,
  ): Promise<(Uint8Array | undefined)[]> {
    const { array } = this;
    if (array.elementType !== "vlen" || array.shape.length !== 1) {
      throw new Error(`${array.path}: readVlenRows needs a 1-D vlen array`);
    }
    end = Math.min(end, array.shape[0]);
    const out = new Array<Uint8Array | undefined>(Math.max(0, end - start));
    if (end <= start) return out;
    const chunkRows = array.readChunkShape[0];
    const first = Math.floor(start / chunkRows);
    const last = Math.floor((end - 1) / chunkRows);
    const chunkIndices: number[] = [];
    for (let c = first; c <= last; ++c) chunkIndices.push(c);
    await mapConcurrent(chunkIndices, 8, async (c) => {
      const bytes = await this.readChunk([c], signal);
      if (bytes === undefined) return;
      const elements = decodeVlenElements(bytes);
      const chunkStart = c * chunkRows;
      for (let i = 0; i < elements.length; ++i) {
        const row = chunkStart + i;
        if (row < start || row >= end) continue;
        out[row - start] = elements[i].length === 0 ? undefined : elements[i];
      }
    });
    return out;
  }
}

/** Header before the payload of a single-element vlen chunk. */
const VLEN_SINGLE_HEADER = 8;

function copyIfShared(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  if (bytes.buffer instanceof ArrayBuffer) {
    return bytes as Uint8Array<ArrayBuffer>;
  }
  return new Uint8Array(bytes);
}

/** Element `index` of a decoded vlen chunk (`u32 count`, then `u32 len` + bytes each). */
export function vlenElementAt(chunk: Uint8Array, index: number): Uint8Array {
  const view = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  if (chunk.length < 4) throw new Error("vlen chunk shorter than its header");
  const count = view.getUint32(0, true);
  if (index >= count) return new Uint8Array(0);
  let offset = 4;
  for (let i = 0; i < count; ++i) {
    if (offset + 4 > chunk.length) throw new Error("vlen chunk truncated");
    const length = view.getUint32(offset, true);
    offset += 4;
    if (offset + length > chunk.length) throw new Error("vlen chunk truncated");
    if (i === index) return chunk.subarray(offset, offset + length);
    offset += length;
  }
  return new Uint8Array(0);
}

/** Every element of a decoded vlen chunk, as views into it. */
export function decodeVlenElements(chunk: Uint8Array): Uint8Array[] {
  const view = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  if (chunk.length < 4) throw new Error("vlen chunk shorter than its header");
  const count = view.getUint32(0, true);
  const out = new Array<Uint8Array>(count);
  let offset = 4;
  for (let i = 0; i < count; ++i) {
    if (offset + 4 > chunk.length) throw new Error("vlen chunk truncated");
    const length = view.getUint32(offset, true);
    offset += 4;
    if (offset + length > chunk.length) throw new Error("vlen chunk truncated");
    out[i] = chunk.subarray(offset, offset + length);
    offset += length;
  }
  return out;
}

/** Little-endian bytes of one fill-value element. */
function fillBytes(array: ZarrArray): Uint8Array {
  const out = new Uint8Array(array.elementBytes);
  const view = new DataView(out.buffer);
  let value = array.fillValue;
  if (value === null || value === undefined) return out;
  if (value === "NaN") value = Number.NaN;
  else if (value === "Infinity") value = Number.POSITIVE_INFINITY;
  else if (value === "-Infinity") value = Number.NEGATIVE_INFINITY;
  if (typeof value === "boolean") value = value ? 1 : 0;
  if (typeof value !== "number" && typeof value !== "string") return out;
  switch (array.elementType) {
    case "float32":
      view.setFloat32(0, Number(value), true);
      break;
    case "float64":
      view.setFloat64(0, Number(value), true);
      break;
    case "int64":
      view.setBigInt64(0, BigInt(value), true);
      break;
    case "uint64":
      view.setBigUint64(0, BigInt.asUintN(64, BigInt(value)), true);
      break;
    case "int32":
      view.setInt32(0, Number(value), true);
      break;
    case "uint32":
      view.setUint32(0, Number(value), true);
      break;
    case "int16":
      view.setInt16(0, Number(value), true);
      break;
    case "uint16":
    case "float16":
      view.setUint16(0, Number(value), true);
      break;
    default:
      view.setUint8(0, Number(value));
  }
  return out;
}

function fillRange(
  out: Uint8Array,
  offset: number,
  length: number,
  element: Uint8Array,
) {
  if (element.every((b) => b === 0)) return;
  for (let i = 0; i < length; i += element.length) {
    out.set(element, offset + i);
  }
}

function swapEndian(bytes: Uint8Array, width: number) {
  for (let i = 0; i < bytes.length; i += width) {
    for (let a = i, b = i + width - 1; a < b; ++a, --b) {
      const t = bytes[a];
      bytes[a] = bytes[b];
      bytes[b] = t;
    }
  }
}
