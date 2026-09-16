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
 * Reads one spatial chunk's raw `vlen-bytes` container out of a per-chunk
 * zarr-vectors array (`vertices`, `vertex_fragments`, `fragment_attributes/*`,
 * `vertex_attributes/*`, `links/<delta>/<offsets>`), honoring the two ways the
 * v0.9.0 "single-array" format can differ from the layout the skeleton reader
 * historically assumed:
 *
 *   1. **`chunk_grid_origin`** — the array is a plain zarr v3 array whose cell
 *      `(0,0,0)` corresponds to the ABSOLUTE spatial chunk coord `origin`
 *      (e.g. `[-4,-5,-2]` for RASmm brain data straddling the world origin).
 *      The on-disk 0-based array index of an absolute coord `C` is `C - origin`.
 *
 *   2. **`sharding_indexed`** — the (opt-in) packing wraps the array in a zarr v3
 *      sharding codec, so many spatial chunks live in one shard object addressed
 *      by a byte-range via the shard index. We reuse neuroglancer's
 *      {@link ShardedKvStore} (suffix index read + crc32c decode + inner
 *      byte-range GET) rather than re-implement it.
 *
 * Both cases decompress Zstd frames before returning the numcodecs
 * `vlen-bytes` container; uncompressed cells pass through unchanged. Detection
 * is per-array from its own `zarr.json` (see `ChunkGridDescriptor`), so an unsharded,
 * zero-origin store reduces to the historical behaviour.
 *
 * Backend-only: instantiate one {@link ShardCellReader} per source so its
 * shard-index cache (keyed by shard) is amortized across all cells of a shard.
 */

// Codec self-registration (resolve side, for parseCodecChainSpec; decode side,
// for the shard-index decode inside ShardedKvStore). crc32c here is Castagnoli.
import "#src/datasource/zarr/codec/bytes/resolve.js";
import "#src/datasource/zarr/codec/bytes/decode.js";
import "#src/datasource/zarr/codec/crc32c/resolve.js";
import "#src/datasource/zarr/codec/crc32c/decode.js";

import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import type { ChunkManager } from "#src/chunk_manager/backend.js";
import { parseCodecChainSpec } from "#src/datasource/zarr/codec/resolve.js";
import { ShardedKvStore } from "#src/datasource/zarr/codec/sharding_indexed/decode.js";
import type { Configuration } from "#src/datasource/zarr/codec/sharding_indexed/resolve.js";
import { ShardIndexLocation } from "#src/datasource/zarr/codec/sharding_indexed/resolve.js";
import type {
  DriverReadOptions,
  ReadableKvStore,
  ReadResponse,
  StatOptions,
  StatResponse,
} from "#src/kvstore/index.js";
import { joinBaseUrlAndPath } from "#src/kvstore/url.js";
import { DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";

/**
 * Per-array chunk-grid geometry needed to map an absolute spatial chunk coord to
 * its physical (possibly sharded) on-disk cell. Read from the array's `zarr.json`
 * on the frontend (see `buildGeometryMetadata`) and threaded via source params.
 */
export interface ChunkGridDescriptor {
  /** `attributes.chunk_grid_origin` (absent ⇒ zeros). arrayIndex = C − origin. */
  origin: number[];
  /** True iff `codecs[0].name === "sharding_indexed"`. */
  sharded: boolean;
  /** Shard shape (`chunk_grid.configuration.chunk_shape`); `[]` when unsharded. */
  shardShape: number[];
  /** `chunk_key_encoding.configuration.separator` (default `/`). */
  separator: string;
}

/** Reads one per-chunk array cell; `undefined` ⇒ the chunk is absent (missing
 * shard/cell, or an empty shard-index entry). Returns the raw vlen container. */
export type CellReader = (
  arrayPath: string,
  chunkKey: string,
  signal: AbortSignal,
) => Promise<Uint8Array | undefined>;

/** Whole-file read of a subpath relative to the source's base URL (used for the
 * unsharded branch; matches the historical `<array>/c/<idx>` read). */
export type WholeFileRead = (
  subpath: string,
  signal: AbortSignal,
) => Promise<Uint8Array | undefined>;

interface KvStoreContextLike {
  read(
    url: string,
    options: DriverReadOptions,
  ): Promise<ReadResponse | undefined>;
  stat?(url: string, options: StatOptions): Promise<StatResponse | undefined>;
}

/**
 * Pure coordinate math: absolute spatial chunk coord → 0-based array index and
 * (when sharded) shard + inner sub-chunk coords. Kept pure for unit testing.
 */
export function resolveChunkCell(
  grid: ChunkGridDescriptor,
  chunkKey: string,
): { arrayIndex: number[]; shard?: number[]; inner?: number[] } {
  const abs = chunkKey.split(".").map(Number);
  const arrayIndex = abs.map((c, i) => c - (grid.origin[i] ?? 0));
  if (!grid.sharded) return { arrayIndex };
  const { shardShape } = grid;
  const shard = arrayIndex.map((v, i) => Math.floor(v / shardShape[i]));
  const inner = arrayIndex.map((v, i) => v - shard[i] * shardShape[i]);
  return { arrayIndex, shard, inner };
}

/** Build the {@link Configuration} for a zarr-vectors sharded array. The inner
 * chunk is always `[1,1,1]` (one spatial chunk per inner cell), so the sharding
 * sub-chunk grid equals the shard shape. `subChunkCodecs` is a never-read
 * placeholder — the real payload is `vlen-bytes`, which ShardedKvStore returns
 * raw (it does not run the sub-chunk codec). */
function buildShardConfiguration(shardShape: number[]): Configuration {
  const rank = shardShape.length;
  const subChunkShape = new Array<number>(rank).fill(1);
  return {
    indexCodecs: parseCodecChainSpec(
      [
        { name: "bytes", configuration: { endian: "little" } },
        { name: "crc32c" },
      ],
      { dataType: DataType.UINT64, chunkShape: [...shardShape, 2] },
    ),
    subChunkCodecs: parseCodecChainSpec(
      [{ name: "bytes", configuration: { endian: "little" } }],
      { dataType: DataType.UINT8, chunkShape: subChunkShape },
    ),
    subChunkShape,
    subChunkGridShape: [...shardShape],
    indexLocation: ShardIndexLocation.END,
  };
}

export class ShardCellReader extends RefCounted {
  private shardedKvStore: ShardedKvStore<string> | undefined;

  constructor(
    chunkManager: ChunkManager,
    private baseUrl: string,
    private kvStoreContext: KvStoreContextLike,
    private wholeFileRead: WholeFileRead,
    private grid: ChunkGridDescriptor,
  ) {
    super();
    if (grid.sharded) {
      this.shardedKvStore = this.registerDisposer(
        new ShardedKvStore<string>(
          buildShardConfiguration(grid.shardShape),
          chunkManager,
          this.makeBaseStore(),
        ),
      );
    }
  }

  /**
   * Adapter exposing the shared kvstore context as a `ReadableKvStore<string>`
   * whose keys are subpaths relative to `baseUrl`. Reuses `joinBaseUrlAndPath` +
   * `kvStoreContext.read` (the exact path the rest of the reader uses), so URL
   * encoding, pipeline handling, and byte-range support are inherited.
   */
  private makeBaseStore(): ReadableKvStore<string> {
    const { baseUrl, kvStoreContext } = this;
    return {
      read: (key, options) =>
        kvStoreContext.read(joinBaseUrlAndPath(baseUrl, key), options),
      stat: (key, options) =>
        kvStoreContext.stat === undefined
          ? Promise.resolve(undefined)
          : kvStoreContext.stat(joinBaseUrlAndPath(baseUrl, key), options),
      getUrl: (key) => joinBaseUrlAndPath(baseUrl, key),
      supportsOffsetReads: true,
      supportsSuffixReads: true,
    };
  }

  read: CellReader = async (arrayPath, chunkKey, signal) => {
    const { separator } = this.grid;
    const { arrayIndex, shard, inner } = resolveChunkCell(this.grid, chunkKey);
    if (shard === undefined) {
      // Unsharded: one whole file per 0-based cell.
      const bytes = await this.wholeFileRead(
        `${arrayPath}/c/${arrayIndex.join(separator)}`,
        signal,
      );
      return decodeCellCompression(bytes, signal);
    }
    const response = await this.shardedKvStore!.read(
      { base: `${arrayPath}/c/${shard.join(separator)}`, subChunk: inner! },
      { signal },
    );
    if (response === undefined) return undefined;
    return decodeCellCompression(
      new Uint8Array(await response.response.arrayBuffer()),
      signal,
    );
  };
}

/** Remove the outer Zstd frame before callers read the vlen-bytes header. */
async function decodeCellCompression(
  bytes: Uint8Array | undefined,
  signal: AbortSignal,
): Promise<Uint8Array | undefined> {
  if (
    bytes === undefined ||
    bytes.length < 4 ||
    bytes[0] !== 0x28 ||
    bytes[1] !== 0xb5 ||
    bytes[2] !== 0x2f ||
    bytes[3] !== 0xfd
  )
    return bytes;
  // Own the buffer before transferring it: a whole-file reader may cache bytes
  // or return a view into a larger buffer.
  const input = new Uint8Array(bytes);
  return requestAsyncComputation(decodeZstd, signal, [input.buffer], input);
}
