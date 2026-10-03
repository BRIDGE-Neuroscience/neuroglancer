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
 * chunk. Decoded chunks and manifest chunks are shared between the objects
 * that pass through them.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/base.js";
import type { DecodedChunk } from "#src/datasource/zarr-vectors/chunk_decode.js";
import {
  decodeChunk,
  edgeTangents,
  forEachFragmentVertex,
} from "#src/datasource/zarr-vectors/chunk_decode.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import { CrossChunkLinks } from "#src/datasource/zarr-vectors/links.js";
import type { ManifestBlock } from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  decodeObjectManifest,
  resolveFragmentRef,
} from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  readObjectTable,
  SegmentIdIndex,
} from "#src/datasource/zarr-vectors/objects.js";
import type {
  ZarrVectorsLevel,
  ZarrVectorsStoreAccess,
} from "#src/datasource/zarr-vectors/store.js";
import {
  AsyncLru,
  SHARED_SIGNAL,
  warnOnce,
} from "#src/datasource/zarr-vectors/util.js";
import { decodeVlenElements } from "#src/datasource/zarr-vectors/zarr_array.js";

export interface ObjectSkeleton {
  positions: Float32Array;
  edges: Uint32Array;
  /** One array per exposed attribute, then the tangent if the kind has one. */
  attributes: Float32Array[];
}

/** One object's vertices in each chunk it visits, in manifest (walk) order. */
interface ObjectParts {
  order: { chunkKey: string; vertices: number[] }[];
  chunks: Map<string, DecodedChunk>;
}

export class ObjectReader {
  private cells: LevelCells;
  private links: CrossChunkLinks | undefined;
  private index:
    | Promise<{ rows: SegmentIdIndex; chunkRows: number } | undefined>
    | undefined;
  // Decoded chunks are large (tens of MB at a whole-brain store's finest
  // level) and not charged to the chunk manager, so keep only a few.
  private chunks = new AsyncLru<DecodedChunk | undefined>(8);
  private manifestChunks = new AsyncLru<Uint8Array[]>(16);

  constructor(
    private access: ZarrVectorsStoreAccess,
    private description: ZarrVectorsGeometryDescription,
    level: ZarrVectorsLevel,
  ) {
    this.cells = LevelCells.forLevel(access, level, description);
    if (KIND_CAPABILITIES[description.geometryKind].primitive !== "points") {
      this.links = new CrossChunkLinks({
        cells: this.cells,
        listDirectories: (path) =>
          access.listDirectories(`${level.path}/${path}`),
        warn: warnOnce,
      });
    }
  }

  private objectIndex() {
    if (this.index === undefined) {
      const promise = (async () => {
        const table = await readObjectTable(this.access, this.cells.levelPath);
        if (table === undefined) return undefined;
        const manifests = await this.cells.reader("object_index/manifests");
        return {
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

  /** The manifest of the object shown as `segmentId`. */
  async manifest(segmentId: bigint): Promise<ManifestBlock[]> {
    const index = await this.objectIndex();
    if (index === undefined) throw new Error("this store has no object index");
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
    return blob === undefined || blob.byteLength === 0
      ? []
      : decodeObjectManifest(blob, 3);
  }

  private async parts(
    segmentId: bigint,
    signal: AbortSignal,
  ): Promise<ObjectParts> {
    const blocks = await this.manifest(segmentId);
    const keys = [...new Set(blocks.map((b) => b.chunkCoords.join(".")))];
    const decoded = await Promise.all(
      keys.map((key) =>
        this.chunks.get(key, () =>
          decodeChunk(this.cells, this.description, key, SHARED_SIGNAL, {
            skipSegmentIds: true,
          }),
        ),
      ),
    );
    signal.throwIfAborted();
    const chunks = new Map<string, DecodedChunk>();
    keys.forEach((key, i) => {
      if (decoded[i] !== undefined) chunks.set(key, decoded[i]!);
    });
    const order: ObjectParts["order"] = [];
    for (const block of blocks) {
      const chunkKey = block.chunkCoords.join(".");
      const chunk = chunks.get(chunkKey);
      if (chunk === undefined) continue;
      for (const f of resolveFragmentRef(block.fragmentRef)) {
        if (f >= chunk.fragments.numFragments) continue;
        const vertices: number[] = [];
        forEachFragmentVertex(chunk.fragments, f, (v) => vertices.push(v));
        order.push({ chunkKey, vertices });
      }
    }
    return { order, chunks };
  }

  async readSkeleton(
    segmentId: bigint,
    signal: AbortSignal,
  ): Promise<ObjectSkeleton> {
    const { description } = this;
    const { order, chunks } = await this.parts(segmentId, signal);
    // One output vertex per (chunk, local vertex).
    const indexOf = new Map<string, Map<number, number>>();
    const sources: { chunk: DecodedChunk; vertex: number }[] = [];
    const add = (chunkKey: string, vertex: number) => {
      let map = indexOf.get(chunkKey);
      if (map === undefined) indexOf.set(chunkKey, (map = new Map()));
      let i = map.get(vertex);
      if (i === undefined) {
        i = sources.length;
        map.set(vertex, i);
        sources.push({ chunk: chunks.get(chunkKey)!, vertex });
      }
      return i;
    };
    const edges: number[] = [];
    const primitive = KIND_CAPABILITIES[description.geometryKind].primitive;
    if (
      primitive === "lines" &&
      description.linksConvention === "implicit_sequential"
    ) {
      // A curve: its fragments, in manifest order, continue one another.
      let tail: number | undefined;
      for (const { chunkKey, vertices } of order) {
        for (const v of vertices) {
          const i = add(chunkKey, v);
          if (tail !== undefined) edges.push(tail, i);
          tail = i;
        }
      }
    } else {
      for (const { chunkKey, vertices } of order) {
        for (const v of vertices) add(chunkKey, v);
      }
      // The links between its chunks...
      const linked = description.skeletonLayout === "linked";
      const relinked = new Set<string>();
      for (const chunkKey of indexOf.keys()) {
        const coords = chunkKey.split(".").map(Number);
        const links = linked
          ? this.links?.linksTouching(coords)
          : this.links?.linksOwnedBy(coords);
        for (const { endpoints } of (await links) ?? []) {
          if (endpoints.length !== 2) continue;
          const keys = endpoints.map((e) => e.chunkCoords.join("."));
          const [a, b] = endpoints.map((e, i) =>
            indexOf.get(keys[i])?.get(e.vertexIndex),
          );
          if (a === undefined || b === undefined) continue;
          if (keys[0] !== chunkKey && keys[1] !== chunkKey) continue;
          if (linked && keys[0] !== chunkKey) continue; // the child's chunk keeps it
          edges.push(a, b);
          // In a linked skeleton the link replaces the child's implied parent.
          if (linked) relinked.add(`${keys[0]}:${endpoints[0].vertexIndex}`);
        }
      }
      // ...and each chunk's own edges between its vertices.
      for (const [chunkKey, map] of indexOf) {
        const e = chunks.get(chunkKey)!.edges;
        for (let k = 0; k < e.length; k += 2) {
          const a = map.get(e[k]);
          const b = map.get(e[k + 1]);
          if (a === undefined || b === undefined) continue;
          const implied = e[k] + 1 === e[k + 1];
          if (implied && relinked.has(`${chunkKey}:${e[k + 1]}`)) continue;
          edges.push(a, b);
        }
      }
    }
    const n = sources.length;
    const positions = new Float32Array(n * 3);
    const attributes = description.attributes.map(
      (a) => new Float32Array(n * a.components),
    );
    sources.forEach(({ chunk, vertex }, i) => {
      positions.set(
        chunk.positions.subarray(3 * vertex, 3 * vertex + 3),
        3 * i,
      );
      description.attributes.forEach(({ components: w }, k) => {
        attributes[k].set(
          chunk.attributes[k].subarray(w * vertex, w * (vertex + 1)),
          w * i,
        );
      });
    });
    const edgeArray = Uint32Array.from(edges);
    if (KIND_CAPABILITIES[description.geometryKind].tangent !== undefined) {
      attributes.push(edgeTangents(positions, edgeArray));
    }
    return { positions, edges: edgeArray, attributes };
  }

  /** Mesh fragment ids of an object: the chunks it occupies. */
  async meshFragmentKeys(segmentId: bigint): Promise<string[]> {
    const blocks = await this.manifest(segmentId);
    return [...new Set(blocks.map((b) => b.chunkCoords.join(".")))];
  }

  /** An object's triangles in one chunk, including faces shared with neighbours. */
  async readMeshFragment(
    segmentId: bigint,
    chunkKey: string,
    signal: AbortSignal,
  ): Promise<{ positions: Float32Array; indices: Uint32Array }> {
    const { order, chunks } = await this.parts(segmentId, signal);
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
    const positions: number[] = [];
    const remap = new Map<string, number>();
    const vertexOf = (key: string, v: number) => {
      const id = `${key}:${v}`;
      let out = remap.get(id);
      if (out === undefined) {
        out = positions.length / 3;
        positions.push(
          ...chunks.get(key)!.positions.subarray(3 * v, 3 * v + 3),
        );
        remap.set(id, out);
      }
      return out;
    };
    const indices: number[] = [];
    const { faces } = chunk;
    for (let i = 0; i < faces.length; i += 3) {
      if ([0, 1, 2].every((k) => local.has(faces[i + k]))) {
        indices.push(...[0, 1, 2].map((k) => vertexOf(chunkKey, faces[i + k])));
      }
    }
    for (const { endpoints } of (await this.links?.linksOwnedBy(
      chunkKey.split(".").map(Number),
    )) ?? []) {
      if (endpoints.length < 3) continue;
      const keys = endpoints.map((e) => e.chunkCoords.join("."));
      const ours = endpoints.every(
        (e, i) =>
          member.get(keys[i])?.has(e.vertexIndex) && chunks.has(keys[i]),
      );
      if (!ours) continue;
      const ids = endpoints.map((e, i) => vertexOf(keys[i], e.vertexIndex));
      for (let k = 1; k + 1 < ids.length; ++k)
        indices.push(ids[0], ids[k], ids[k + 1]);
    }
    return {
      positions: Float32Array.from(positions),
      indices: Uint32Array.from(indices),
    };
  }
}
