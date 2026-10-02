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
 * @file Whole objects at full resolution, for Neuroglancer's own skeleton and
 * mesh layers: segment id -> row -> manifest -> the fragments it owns in each
 * chunk.
 *
 * Chunks are decoded once and shared between the objects that pass through
 * them (a bundle of tracts shares most of its chunks), and manifest chunks
 * are decoded once and shared between rows.
 */

import { ChunkCoalescingCache } from "#src/datasource/zarr-vectors/chunk_coalescing_cache.js";
import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import { hasTangentAttribute } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import type { SkeletonChunk as DecodedChunk } from "#src/datasource/zarr-vectors/geometry_chunk.js";
import { downloadGeometryChunk } from "#src/datasource/zarr-vectors/geometry_chunk_download.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import { CrossChunkLinks } from "#src/datasource/zarr-vectors/links.js";
import type { ManifestBlock } from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  decodeObjectManifest,
  resolveFragmentRef,
} from "#src/datasource/zarr-vectors/object_manifest.js";
import type { ZarrVectorsObjectTable } from "#src/datasource/zarr-vectors/objects.js";
import {
  readObjectTable,
  SegmentIdIndex,
} from "#src/datasource/zarr-vectors/objects.js";
import type { ZarrVectorsLevel } from "#src/datasource/zarr-vectors/store.js";
import type { VertexAttributeDtype } from "#src/datasource/zarr-vectors/vertex_attribute_float.js";
import type {
  ShardIndexCache,
  ZarrArrayRead,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import { decodeVlenElements } from "#src/datasource/zarr-vectors/zarr_array.js";

export interface ObjectReaderOptions {
  read: ZarrArrayRead;
  shardIndexes: ShardIndexCache;
  listDirectories?: (path: string) => Promise<string[]>;
  description: ZarrVectorsGeometryDescription;
  level: ZarrVectorsLevel;
  warn: (message: string) => void;
}

export interface ObjectSkeleton {
  positions: Float32Array;
  edges: Uint32Array;
  /** One array per exposed attribute, then the tangent if the kind has one. */
  attributes: Float32Array[];
}

/** A signal that never aborts, for loads shared between requests. */
const SHARED_SIGNAL = new AbortController().signal;

interface ObjectIndex {
  table: ZarrVectorsObjectTable;
  rows: SegmentIdIndex;
  chunkRows: number;
}

export class ObjectReader {
  readonly cells: LevelCells;
  private links: CrossChunkLinks | undefined;
  private index: Promise<ObjectIndex | undefined> | undefined;
  // Decoded chunks are large (tens of MB at a whole-brain store's finest
  // level) and not charged to the chunk manager, so keep only a few.
  private chunks = new ChunkCoalescingCache<DecodedChunk | undefined>(8);
  private manifestChunks = new ChunkCoalescingCache<Uint8Array[]>(16);

  constructor(private options: ObjectReaderOptions) {
    const { level, description } = options;
    const known = new Map<string, any>();
    known.set("vertices", level.arrays.vertices);
    known.set("vertex_fragments", level.arrays.vertexFragments);
    known.set(
      "fragment_attributes/segment_id",
      level.arrays.fragmentSegmentIds,
    );
    description.attributes.forEach((a, i) =>
      known.set(`vertex_attributes/${a.name}`, level.arrays.attributes[i]),
    );
    this.cells = new LevelCells(
      options.read,
      options.shardIndexes,
      level.path,
      known,
    );
    if (KIND_CAPABILITIES[description.geometryKind].edgeSource !== "none") {
      this.links = new CrossChunkLinks({
        cells: this.cells,
        listDirectories:
          options.listDirectories === undefined
            ? undefined
            : (path) => options.listDirectories!(`${level.path}/${path}`),
        warn: options.warn,
      });
    }
  }

  private objectIndex(): Promise<ObjectIndex | undefined> {
    if (this.index === undefined) {
      const promise = (async () => {
        const table = await readObjectTable(
          {
            read: this.options.read,
            shardIndexes: this.options.shardIndexes,
            listDirectories: async () => [],
          },
          this.options.level.path,
        );
        if (table === undefined) return undefined;
        const manifests = await this.cells.reader("object_index/manifests");
        return {
          table,
          rows: new SegmentIdIndex(table.segmentIds),
          chunkRows: manifests?.array.readChunkShape[0] ?? 1,
        };
      })();
      this.index = promise;
      promise.catch(() => {
        if (this.index === promise) this.index = undefined;
      });
    }
    return this.index;
  }

  /** The manifest blocks of the object shown as `segmentId`. */
  async manifest(segmentId: bigint): Promise<ManifestBlock[]> {
    const index = await this.objectIndex();
    if (index === undefined) {
      throw new Error("this store has no object index");
    }
    const row = index.rows.rowOf(segmentId);
    if (row === undefined) return [];
    const reader = await this.cells.reader("object_index/manifests");
    if (reader === undefined)
      throw new Error("object_index/manifests is missing");
    const chunk = Math.floor(row / index.chunkRows);
    const elements = await this.manifestChunks.get(String(chunk), async () => {
      const bytes = await reader.readChunk([chunk], SHARED_SIGNAL);
      return bytes === undefined ? [] : decodeVlenElements(bytes);
    });
    const blob = elements[row - chunk * index.chunkRows];
    if (blob === undefined || blob.byteLength === 0) return [];
    return decodeObjectManifest(blob, this.options.description.rank);
  }

  private decodedChunk(chunkKey: string): Promise<DecodedChunk | undefined> {
    const { description, level } = this.options;
    return this.chunks.get(chunkKey, () =>
      downloadGeometryChunk(
        {
          chunkKey,
          rank: description.rank,
          linkDtype: String(
            level.arrays.intraLinks?.attributes?.dtype ?? "int64",
          ) as any,
          attributeNames: description.attributes.map((a) => a.name),
          attributeDtypes: description.attributes.map(
            (a) => a.dtype as VertexAttributeDtype,
          ),
          attributeComponents: description.attributes.map((a) => a.components),
          vertexDtype: String(
            level.arrays.vertices.attributes?.dtype ?? "float32",
          ) as VertexAttributeDtype,
          linksConvention: description.linksConvention,
          geometryKind: description.geometryKind,
          // Objects are already known by manifest; per-fragment ids are not needed.
          hasFragmentSegmentIds: false,
          vertexIdAttribute: undefined,
          linkWidth: description.linkWidth,
          linkedSkeletonLayout: description.linkedSkeletonLayout,
          cellRead: this.cells.cellReader,
        },
        SHARED_SIGNAL,
      ),
    );
  }

  /**
   * Vertices of the object in each chunk it visits, in manifest (walk) order.
   * Keys are chunk keys; values the chunk-local vertex indices.
   */
  private async objectVertices(
    blocks: ManifestBlock[],
    signal: AbortSignal,
  ): Promise<{
    order: { chunkKey: string; vertices: Uint32Array }[];
    chunks: Map<string, DecodedChunk>;
  }> {
    const keys = [...new Set(blocks.map((b) => b.chunkCoords.join(".")))];
    const decoded = await Promise.all(keys.map((k) => this.decodedChunk(k)));
    signal.throwIfAborted();
    const chunks = new Map<string, DecodedChunk>();
    keys.forEach((k, i) => {
      if (decoded[i] !== undefined) chunks.set(k, decoded[i]!);
    });
    const order: { chunkKey: string; vertices: Uint32Array }[] = [];
    for (const block of blocks) {
      const chunkKey = block.chunkCoords.join(".");
      const chunk = chunks.get(chunkKey);
      if (chunk === undefined) continue;
      for (const f of resolveFragmentRef(block.fragmentRef)) {
        if (f >= chunk.fragmentIndex.numFragments) continue;
        order.push({ chunkKey, vertices: chunk.fragmentIndex.indices(f) });
      }
    }
    return { order, chunks };
  }

  async readSkeleton(
    segmentId: bigint,
    signal: AbortSignal,
  ): Promise<ObjectSkeleton> {
    const { description } = this.options;
    const blocks = await this.manifest(segmentId);
    const { order, chunks } = await this.objectVertices(blocks, signal);
    // Global vertex numbering: one entry per (chunk, local vertex).
    const globalOf = new Map<string, Map<number, number>>();
    const sources: { chunk: DecodedChunk; vertex: number }[] = [];
    const add = (chunkKey: string, vertex: number) => {
      let map = globalOf.get(chunkKey);
      if (map === undefined) globalOf.set(chunkKey, (map = new Map()));
      let g = map.get(vertex);
      if (g === undefined) {
        g = sources.length;
        map.set(vertex, g);
        sources.push({ chunk: chunks.get(chunkKey)!, vertex });
      }
      return g;
    };
    const edges: number[] = [];
    const sequential =
      description.linksConvention === "implicit_sequential" &&
      KIND_CAPABILITIES[description.geometryKind].edgeSource !== "none";
    let previousTail: number | undefined;
    for (const { chunkKey, vertices } of order) {
      let prev: number | undefined;
      for (const v of vertices) {
        const g = add(chunkKey, v);
        if (sequential && prev !== undefined) edges.push(prev, g);
        prev = g;
      }
      if (sequential && previousTail !== undefined && vertices.length > 0) {
        // Consecutive manifest blocks continue the same curve.
        edges.push(previousTail, globalOf.get(chunkKey)!.get(vertices[0])!);
      }
      if (vertices.length > 0) previousTail = prev;
    }
    if (!sequential) {
      // Explicit / branched connectivity: the chunk's own edges between this
      // object's vertices, plus cross-chunk links between them.
      for (const [chunkKey, map] of globalOf) {
        const chunk = chunks.get(chunkKey)!;
        const e = chunk.edges;
        for (let i = 0; i < e.length; i += 2) {
          const a = map.get(e[i]);
          const b = map.get(e[i + 1]);
          if (a !== undefined && b !== undefined) edges.push(a, b);
        }
      }
      if (this.links !== undefined) {
        const tables = await Promise.all(
          [...globalOf.keys()].map((k) =>
            this.links!.linksOwnedBy(k.split(".").map(Number), signal),
          ),
        );
        for (const table of tables) {
          for (const record of table?.records ?? []) {
            if (record.endpoints.length !== 2) continue;
            const [a, b] = record.endpoints.map((e) =>
              globalOf.get(e.chunkCoords.join("."))?.get(e.vertexIndex),
            );
            if (a !== undefined && b !== undefined) edges.push(a, b);
          }
        }
      }
    }
    const n = sources.length;
    const positions = new Float32Array(n * 3);
    const rank = description.rank;
    const widths = description.attributes.map((a) => a.components);
    const attributes = widths.map((w) => new Float32Array(n * w));
    sources.forEach(({ chunk, vertex }, g) => {
      for (let d = 0; d < 3; ++d) {
        positions[g * 3 + d] =
          d < rank ? chunk.positions[vertex * rank + d] : 0;
      }
      widths.forEach((w, i) => {
        const src = chunk.vertexAttributes[i] as Float32Array<ArrayBuffer>;
        attributes[i].set(src.subarray(vertex * w, (vertex + 1) * w), g * w);
      });
    });
    if (hasTangentAttribute(description)) {
      attributes.push(tangentsFromEdges(positions, edges, n));
    }
    return { positions, edges: Uint32Array.from(edges), attributes };
  }

  /** Mesh fragment ids of an object: the chunks it occupies. */
  async meshFragmentKeys(
    segmentId: bigint,
    signal: AbortSignal,
  ): Promise<string[]> {
    signal.throwIfAborted();
    const blocks = await this.manifest(segmentId);
    return [...new Set(blocks.map((b) => b.chunkCoords.join(".")))];
  }

  /** Triangles of one object within one chunk, including faces it shares with neighbours. */
  async readMeshFragment(
    segmentId: bigint,
    chunkKey: string,
    signal: AbortSignal,
  ): Promise<{ positions: Float32Array; indices: Uint32Array }> {
    const blocks = await this.manifest(segmentId);
    const { order, chunks } = await this.objectVertices(blocks, signal);
    const member = new Map<string, Set<number>>();
    for (const { chunkKey: key, vertices } of order) {
      let set = member.get(key);
      if (set === undefined) member.set(key, (set = new Set()));
      for (const v of vertices) set.add(v);
    }
    const chunk = chunks.get(chunkKey);
    const local = member.get(chunkKey);
    if (chunk === undefined || local === undefined) {
      return { positions: new Float32Array(0), indices: new Uint32Array(0) };
    }
    const rank = this.options.description.rank;
    const positions: number[] = [];
    const remap = new Map<string, number>();
    const vertexOf = (key: string, v: number) => {
      const id = `${key}:${v}`;
      let out = remap.get(id);
      if (out === undefined) {
        const source = key === chunkKey ? chunk : chunks.get(key)!;
        out = positions.length / 3;
        for (let d = 0; d < 3; ++d) {
          positions.push(d < rank ? source.positions[v * rank + d] : 0);
        }
        remap.set(id, out);
      }
      return out;
    };
    const indices: number[] = [];
    const faces = chunk.faces ?? new Uint32Array(0);
    for (let i = 0; i < faces.length; i += 3) {
      if (
        local.has(faces[i]) &&
        local.has(faces[i + 1]) &&
        local.has(faces[i + 2])
      ) {
        indices.push(
          vertexOf(chunkKey, faces[i]),
          vertexOf(chunkKey, faces[i + 1]),
          vertexOf(chunkKey, faces[i + 2]),
        );
      }
    }
    if (this.links !== undefined) {
      const table = await this.links.linksOwnedBy(
        chunkKey.split(".").map(Number),
        signal,
      );
      for (const record of table?.records ?? []) {
        const corners = record.endpoints;
        if (corners.length < 3) continue;
        const keys = corners.map((e) => e.chunkCoords.join("."));
        if (
          !corners.every(
            (e, i) =>
              member.get(keys[i])?.has(e.vertexIndex) === true &&
              chunks.has(keys[i]),
          )
        ) {
          continue;
        }
        const ids = corners.map((e, i) => vertexOf(keys[i], e.vertexIndex));
        for (let k = 1; k + 1 < ids.length; ++k) {
          indices.push(ids[0], ids[k], ids[k + 1]);
        }
      }
    }
    return {
      positions: Float32Array.from(positions),
      indices: Uint32Array.from(indices),
    };
  }
}

/** Unit tangent per vertex: the mean direction of its edges, walk-oriented. */
function tangentsFromEdges(
  positions: Float32Array,
  edges: number[],
  n: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < edges.length; i += 2) {
    const a = edges[i];
    const b = edges[i + 1];
    for (let d = 0; d < 3; ++d) {
      const delta = positions[b * 3 + d] - positions[a * 3 + d];
      out[a * 3 + d] += delta;
      out[b * 3 + d] += delta;
    }
  }
  for (let v = 0; v < n; ++v) {
    const x = out[v * 3];
    const y = out[v * 3 + 1];
    const z = out[v * 3 + 2];
    const norm = Math.hypot(x, y, z);
    if (norm > 0) {
      out[v * 3] = x / norm;
      out[v * 3 + 1] = y / norm;
      out[v * 3 + 2] = z / norm;
    }
  }
  return out;
}
