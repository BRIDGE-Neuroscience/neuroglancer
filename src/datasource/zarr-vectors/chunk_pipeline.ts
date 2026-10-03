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
 * @file Worker-side assembly of one spatial chunk for the dense overview:
 * decode it, give every vertex its object's segment id, and bridge its curves
 * into neighbouring chunks.
 *
 * Bridging: a curve crossing a chunk face is stored as a cross-chunk link
 * between its last vertex on one side and its first on the other. The chunk
 * owning the link (holding its first endpoint) appends the far endpoint as a
 * "ghost" vertex and draws the bridging edge, so each bridge is drawn once.
 *
 * Ghost positions come from the neighbour's vertex cell. Cells are shared
 * through one cache, so a neighbour the view also draws is fetched once,
 * whichever needs it first. A neighbour the view does not draw (a slice
 * view's chunks above and below the plane) is read by row range instead,
 * when its cell is uncompressed.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/base.js";
import type { DecodedChunk } from "#src/datasource/zarr-vectors/chunk_decode.js";
import {
  decodeChunk,
  forEachFragmentVertex,
} from "#src/datasource/zarr-vectors/chunk_decode.js";
import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  decodeFloat32,
  ELEMENT_BYTES,
} from "#src/datasource/zarr-vectors/dtype.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import { CrossChunkLinks } from "#src/datasource/zarr-vectors/links.js";
import { decodeObjectManifest } from "#src/datasource/zarr-vectors/object_manifest.js";
import { readObjectTable } from "#src/datasource/zarr-vectors/objects.js";
import type {
  ZarrVectorsLevel,
  ZarrVectorsStoreAccess,
} from "#src/datasource/zarr-vectors/store.js";
import {
  AsyncLru,
  mapConcurrent,
  SHARED_SIGNAL,
  warnOnce,
} from "#src/datasource/zarr-vectors/util.js";

/** One decoded chunk, ready to post to the frontend. */
export interface DenseChunkData {
  /** Vertices including ghosts. */
  numVertices: number;
  /** Vertices that belong to this chunk; only these get node circles. */
  numOwnVertices: number;
  positions: Float32Array;
  /** Two uint32 per vertex: the segment id, low word first. */
  segmentIds: Uint32Array;
  /** Line segments as vertex index pairs, bridges included. */
  edges: Uint32Array;
  /** Per exposed attribute (`components` floats per vertex), then the tangent. */
  attributes: Float32Array[];
}

/** Rows this close together are read together. */
const RANGE_GAP_VERTICES = 4096;
/** Past this many reads into one neighbour, its whole cell is fetched. */
const MAX_RANGE_READS = 16;

/** Raw vertex cells, by pipeline and cell, shared by chunks and ghosts. */
const vertexCells = new AsyncLru<Uint8Array | undefined>(
  Infinity,
  256 * 1024 * 1024,
  (b) => b?.byteLength ?? 0,
);
let nextPipelineId = 0;

/**
 * Segment id per fragment of each chunk, by inverting a level's manifests.
 * Needed for stores without `fragment_attributes/segment_id`, which
 * zarr-vectors-py's own writers do not write.
 */
async function fragmentOwners(
  access: ZarrVectorsStoreAccess,
  cells: LevelCells,
  level: ZarrVectorsLevel,
): Promise<Map<string, BigUint64Array> | undefined> {
  const table = await readObjectTable(access, level.path);
  const manifests = await cells.reader("object_index/manifests");
  if (table === undefined || manifests === undefined) return undefined;
  const rows = await manifests.readVlenRows(0, table.numObjects);
  const owners = new Map<string, bigint[]>();
  rows.forEach((blob, row) => {
    if (blob === undefined) return;
    for (const { chunkCoords, fragmentRef: ref } of decodeObjectManifest(
      blob,
      3,
    )) {
      const key = chunkCoords.join(".");
      let list = owners.get(key);
      if (list === undefined) owners.set(key, (list = []));
      const fragments =
        ref.mode === "single"
          ? [ref.fragmentIndex]
          : ref.mode === "range"
            ? Array.from({ length: ref.count }, (_, i) => ref.start + i)
            : Array.from(ref.indices);
      for (const f of fragments) list[f] = table.segmentIds[row];
    }
  });
  return new Map(
    [...owners].map(([key, list]) => [
      key,
      BigUint64Array.from(list, (id) => id ?? 0xffffffffffffffffn),
    ]),
  );
}

interface Ghost {
  /** The local vertex the bridge starts from. */
  host: number;
  chunkKey: string;
  vertex: number;
  /** Whether the host comes first in the curve's walk. */
  hostIsPredecessor: boolean;
}

export class LevelPipeline {
  private cells: LevelCells;
  private links: CrossChunkLinks | undefined;
  private owners: Promise<Map<string, BigUint64Array> | undefined> | undefined;
  private id = nextPipelineId++;

  /**
   * `isRequested` says whether the view also wants the chunk at a cell, so a
   * ghost read of it can fetch the whole cell for that chunk to reuse.
   */
  constructor(
    private access: ZarrVectorsStoreAccess,
    private description: ZarrVectorsGeometryDescription,
    private level: ZarrVectorsLevel,
    private isRequested: (chunk: readonly number[]) => boolean = () => false,
  ) {
    this.cells = LevelCells.forLevel(access, level, description);
    if (KIND_CAPABILITIES[description.geometryKind].primitive === "lines") {
      this.links = new CrossChunkLinks({
        cells: this.cells,
        listDirectories: (path) =>
          access.listDirectories(`${level.path}/${path}`),
        warn: warnOnce,
      });
    }
  }

  /** The raw vertex cell at `chunkKey`, fetched once for every reader. */
  private vertexCell(chunkKey: string) {
    return vertexCells.get(`${this.id}|${chunkKey}`, () =>
      this.cells.readCell("vertices", chunkKey, SHARED_SIGNAL),
    );
  }

  private get vertexType(): ElementType {
    return this.level.arrays.vertices.attributes?.dtype ?? "float32";
  }

  private fragmentOwners() {
    if (this.owners === undefined) {
      const promise = fragmentOwners(this.access, this.cells, this.level);
      this.owners = promise;
      promise.catch((e) => {
        if (this.owners === promise) this.owners = undefined;
        warnOnce(
          "could not map fragments to objects " +
            `(${e instanceof Error ? e.message : e}); objects are coloured per chunk`,
        );
      });
    }
    return this.owners;
  }

  /** The chunk at spatial cell `chunk`, or `undefined` if it is empty. */
  async download(
    chunk: number[],
    signal: AbortSignal,
  ): Promise<DenseChunkData | undefined> {
    const { description } = this;
    const chunkKey = chunk.join(".");
    if (!(await this.cells.mayHaveCell("vertices", chunkKey))) return undefined;
    const isLocal = (coords: readonly number[]) =>
      coords.every((c, d) => c === chunk[d]);
    // In "linked" skeletons a cross-chunk link replaces its child's implied
    // parent, so the child's chunk needs the links it does not own.
    const linked = description.skeletonLayout === "linked";
    const linksPromise =
      this.links === undefined
        ? Promise.resolve([])
        : linked
          ? this.links.linksTouching(chunk)
          : this.links.linksOwnedBy(chunk);
    linksPromise.catch(() => {});
    let relinkedChildren: Set<number> | undefined;
    if (linked) {
      relinkedChildren = new Set();
      for (const { endpoints } of await linksPromise) {
        const [child, parent] = endpoints;
        if (isLocal(child.chunkCoords) && !isLocal(parent.chunkCoords)) {
          relinkedChildren.add(child.vertexIndex);
        }
      }
    }
    const [decoded, links] = await Promise.all([
      decodeChunk(this.cells, description, chunkKey, signal, {
        relinkedChildren,
        skipFaces: true,
        vertices: this.vertexCell(chunkKey),
      }),
      linksPromise,
    ]);
    if (decoded === undefined) return undefined;

    let { segmentIds } = decoded;
    if (segmentIds === undefined) {
      const ids = new Uint32Array(decoded.numVertices * 2);
      const owners = await this.fragmentOwners().catch(() => undefined);
      const chunkOwners = owners?.get(chunkKey);
      for (let f = 0; f < decoded.fragments.numFragments; ++f) {
        const id = chunkOwners?.[f] ?? BigInt(f);
        const lo = Number(id & 0xffffffffn);
        const hi = Number(id >> 32n);
        forEachFragmentVertex(decoded.fragments, f, (v) => {
          ids[2 * v] = lo;
          ids[2 * v + 1] = hi;
        });
      }
      segmentIds = ids;
    }

    const ghosts: Ghost[] = [];
    for (const { endpoints } of links) {
      if (endpoints.length !== 2) continue;
      const [a, b] = endpoints;
      const aLocal = isLocal(a.chunkCoords);
      if (aLocal === isLocal(b.chunkCoords)) continue;
      if (linked && !aLocal) continue; // drawn by the child's chunk
      const [host, far] = aLocal ? [a, b] : [b, a];
      if (host.vertexIndex >= decoded.numVertices) continue;
      ghosts.push({
        host: host.vertexIndex,
        chunkKey: far.chunkCoords.join("."),
        vertex: far.vertexIndex,
        hostIsPredecessor: aLocal,
      });
    }
    const ghostPositions = await this.ghostPositions(ghosts, signal);
    return assembleChunk(decoded, segmentIds, ghosts, ghostPositions);
  }

  /** Positions of the requested neighbour vertices (NaN where unreadable). */
  private async ghostPositions(
    ghosts: readonly Ghost[],
    signal: AbortSignal,
  ): Promise<Float32Array> {
    const out = new Float32Array(ghosts.length * 3).fill(Number.NaN);
    const byChunk = new Map<string, number[]>();
    ghosts.forEach((g, i) => {
      let list = byChunk.get(g.chunkKey);
      if (list === undefined) byChunk.set(g.chunkKey, (list = []));
      list.push(i);
    });
    const type = this.vertexType;
    const rowBytes = 3 * ELEMENT_BYTES[type];
    await mapConcurrent([...byChunk], 8, async ([key, indices]) => {
      const wanted = [...new Set(indices.map((i) => ghosts[i].vertex))].sort(
        (x, y) => x - y,
      );
      const found = new Map<number, Float32Array>();
      const takeAll = (all: Float32Array | undefined) => {
        for (const v of wanted) {
          if (all !== undefined && 3 * v + 3 <= all.length) {
            found.set(v, all.subarray(3 * v, 3 * v + 3));
          }
        }
      };
      const decodeAll = (bytes: Uint8Array | undefined) =>
        bytes === undefined
          ? undefined
          : decodeFloat32(bytes, type, bytes.byteLength / ELEMENT_BYTES[type]);
      // The whole cell when it is cached or in flight, or when the view will
      // draw that chunk anyway; otherwise only the rows needed, if possible.
      let whole = vertexCells.peek(`${this.id}|${key}`);
      if (whole === undefined && this.isRequested(key.split(".").map(Number))) {
        whole = this.vertexCell(key);
      }
      let done = false;
      if (whole !== undefined) {
        takeAll(decodeAll(await whole.catch(() => undefined)));
        done = true;
      }
      const spans: [number, number][] = [];
      for (const v of wanted) {
        const last = spans[spans.length - 1];
        if (last !== undefined && v - last[1] <= RANGE_GAP_VERTICES) {
          last[1] = v;
        } else {
          spans.push([v, v]);
        }
      }
      if (!done && spans.length <= MAX_RANGE_READS) {
        const reads = await Promise.all(
          spans.map(([first, last]) =>
            this.cells.readCellRange(
              "vertices",
              key,
              first * rowBytes,
              (last - first + 1) * rowBytes,
              signal,
            ),
          ),
        );
        // `null`: the cell is compressed and must be read whole.
        done = reads.every((r) => r !== null);
        reads.forEach((bytes, s) => {
          if (bytes === null || bytes === undefined) return;
          const [first, last] = spans[s];
          const values = decodeFloat32(bytes, type, 3 * (last - first + 1));
          for (let v = first; v <= last; ++v) {
            const at = 3 * (v - first);
            found.set(v, values.subarray(at, at + 3));
          }
        });
      }
      if (!done) takeAll(decodeAll(await this.vertexCell(key)));
      for (const i of indices) {
        const p = found.get(ghosts[i].vertex);
        if (p !== undefined) out.set(p, 3 * i);
      }
    });
    return out;
  }
}

/**
 * The final chunk arrays, one allocation each: the decoded vertices, then a
 * ghost per bridge carrying its host's segment id and attributes, with a
 * tangent along the bridge in the curve's walk direction.
 */
function assembleChunk(
  decoded: DecodedChunk,
  segmentIds: Uint32Array,
  ghosts: readonly Ghost[],
  ghostPositions: Float32Array,
): DenseChunkData {
  const own = decoded.numVertices;
  const kept = ghosts
    .map((g, i) => ({ ...g, i }))
    .filter((g) => !Number.isNaN(ghostPositions[3 * g.i]));
  const total = own + kept.length;
  const positions = new Float32Array(total * 3);
  positions.set(decoded.positions.subarray(0, own * 3));
  const ids = new Uint32Array(total * 2);
  ids.set(segmentIds.subarray(0, own * 2));
  const edges = new Uint32Array(decoded.edges.length + kept.length * 2);
  edges.set(decoded.edges);
  const sources = [...decoded.attributes];
  if (decoded.tangents !== undefined) sources.push(decoded.tangents);
  const widths = sources.map((src) => (own === 0 ? 1 : src.length / own));
  const attributes = sources.map((src, i) => {
    const out = new Float32Array(total * widths[i]);
    out.set(src);
    return out;
  });
  const tangentIndex = decoded.tangents === undefined ? -1 : sources.length - 1;
  kept.forEach((g, k) => {
    const v = own + k;
    positions.set(ghostPositions.subarray(3 * g.i, 3 * g.i + 3), 3 * v);
    ids[2 * v] = ids[2 * g.host];
    ids[2 * v + 1] = ids[2 * g.host + 1];
    edges[decoded.edges.length + 2 * k] = g.host;
    edges[decoded.edges.length + 2 * k + 1] = v;
    attributes.forEach((out, i) => {
      const width = widths[i];
      if (i !== tangentIndex) {
        out.copyWithin(v * width, g.host * width, (g.host + 1) * width);
        return;
      }
      // Along the bridge, in walk order, so interpolating from the host's
      // tangent never passes through zero.
      const sign = g.hostIsPredecessor ? 1 : -1;
      const d = [0, 1, 2].map(
        (j) => sign * (positions[3 * v + j] - positions[3 * g.host + j]),
      );
      const norm = Math.hypot(d[0], d[1], d[2]);
      for (let j = 0; j < 3; ++j) {
        out[3 * v + j] = norm > 0 ? d[j] / norm : out[3 * g.host + j];
      }
    });
  });
  return {
    numVertices: total,
    numOwnVertices: own,
    positions,
    segmentIds: ids,
    edges,
    attributes,
  };
}
