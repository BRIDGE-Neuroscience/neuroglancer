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
 * @file Worker-side assembly of one spatial chunk for the dense render layer:
 * decode the chunk's own arrays, bridge its curves into neighbouring chunks,
 * and give every vertex the global segment id of the object it belongs to.
 *
 * Bridging: a curve crossing a chunk face is stored as a cross-chunk link
 * between its last vertex on one side and its first on the other.  The chunk
 * that OWNS a link (holds its first endpoint) appends the far endpoint as a
 * "ghost" vertex and draws the bridging edge, so each bridge is drawn once.
 * Ghost positions are range-read row by row from the neighbour's raw cell
 * (12 bytes each) rather than fetching the neighbour's whole cell; compressed
 * cells cannot be range-read and go through a bounded cache instead.
 */

import type { GeometryChunkDownloadOptions } from "#src/datasource/zarr-vectors/geometry_chunk_download.js";
import { downloadGeometryChunk } from "#src/datasource/zarr-vectors/geometry_chunk_download.js";
import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import type { CrossChunkLinkRecord } from "#src/datasource/zarr-vectors/links.js";
import { CrossChunkLinks } from "#src/datasource/zarr-vectors/links.js";
import { decodeObjectManifest } from "#src/datasource/zarr-vectors/object_manifest.js";
import type { ZarrVectorsObjectTable } from "#src/datasource/zarr-vectors/objects.js";
import { readObjectTable } from "#src/datasource/zarr-vectors/objects.js";
import type {
  ZarrVectorsAttribute,
  ZarrVectorsLevel,
  ZarrVectorsLinksConvention,
} from "#src/datasource/zarr-vectors/store.js";
import type { VertexAttributeDtype } from "#src/datasource/zarr-vectors/vertex_attribute_float.js";
import {
  ATTRIBUTE_ELEMENT_BYTES,
  decodeAttributeToFloat32,
} from "#src/datasource/zarr-vectors/vertex_attribute_float.js";
import type {
  ShardIndexCache,
  ZarrArrayRead,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import { mapConcurrent } from "#src/datasource/zarr-vectors/zarr_array.js";

/** Store-wide geometry description shared by every level's pipeline. */
export interface ZarrVectorsGeometryDescription {
  rank: number;
  geometryKind: ZarrVectorsGeometryKind;
  linksConvention: ZarrVectorsLinksConvention;
  linkWidth: number;
  linkedSkeletonLayout: boolean;
  attributes: ZarrVectorsAttribute[];
  vertexIdAttribute: string | undefined;
  /** The store has an object index (see `ZarrVectorsStore.hasObjects`). */
  hasObjects: boolean;
}

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
  /** Per exposed attribute (`components` floats per vertex), then the tangent if any. */
  attributes: Float32Array[];
}

/** Attribute values per vertex of each array a chunk carries. */
function attributeWidths(description: ZarrVectorsGeometryDescription) {
  const widths = description.attributes.map((a) => a.components);
  const caps = KIND_CAPABILITIES[description.geometryKind];
  if (caps.hasWalkOrderTangent || caps.hasEdgeAdjacencyTangent) widths.push(3);
  return widths;
}

/** Whether chunks of this description carry a synthesised `tangent`. */
export function hasTangentAttribute(
  description: ZarrVectorsGeometryDescription,
) {
  const caps = KIND_CAPABILITIES[description.geometryKind];
  return caps.hasWalkOrderTangent || caps.hasEdgeAdjacencyTangent;
}

/** A bounded cache of decoded neighbour positions, by level and cell. */
class PositionCache {
  private entries = new Map<string, Promise<Float32Array | undefined>>();
  private bytes = 0;
  private sizes = new Map<string, number>();
  constructor(private maxBytes: number) {}

  /** The cached entry, if any, without loading. */
  peek(key: string): Promise<Float32Array | undefined> | undefined {
    return this.entries.get(key);
  }

  /** Records positions already decoded elsewhere. */
  put(key: string, positions: Float32Array) {
    if (this.entries.has(key)) return;
    this.get(key, () => Promise.resolve(positions));
  }
  get(key: string, load: () => Promise<Float32Array | undefined>) {
    let entry = this.entries.get(key);
    if (entry !== undefined) {
      this.entries.delete(key);
      this.entries.set(key, entry);
      return entry;
    }
    entry = load();
    this.entries.set(key, entry);
    const self = entry;
    entry.then(
      (value) => {
        if (this.entries.get(key) !== self) return;
        const size = value?.byteLength ?? 0;
        this.sizes.set(key, size);
        this.bytes += size;
        while (this.bytes > this.maxBytes && this.entries.size > 1) {
          const oldest = this.entries.keys().next().value!;
          this.entries.delete(oldest);
          this.bytes -= this.sizes.get(oldest) ?? 0;
          this.sizes.delete(oldest);
        }
      },
      () => {
        if (this.entries.get(key) === self) this.entries.delete(key);
      },
    );
    return entry;
  }
}

const positionCache = new PositionCache(256 * 1024 * 1024);

/** Rows this close together are read together (12 bytes each for float32). */
const RANGE_GAP_VERTICES = 1024;
/** More reads than this into one neighbour, and the whole cell is fetched. */
const MAX_RANGE_READS = 4;
/** Loads shared between requests must not be cancelled by any one of them. */
const SHARED_SIGNAL = new AbortController().signal;

/**
 * Maps each fragment of each chunk to the row of the object that owns it, by
 * inverting the level's manifests.  Only needed for stores that do not write
 * `fragment_attributes/segment_id` (zarr-vectors-py's own writers don't).
 */
class FragmentObjectMap {
  private byChunk = new Map<string, Int32Array>();
  constructor(
    readonly table: ZarrVectorsObjectTable,
    manifests: (Uint8Array | undefined)[],
    sidNdim: number,
    fragmentCounts: Map<string, number>,
  ) {
    for (let row = 0; row < manifests.length; ++row) {
      const blob = manifests[row];
      if (blob === undefined) continue;
      for (const block of decodeObjectManifest(blob, sidNdim)) {
        const key = block.chunkCoords.join(".");
        let map = this.byChunk.get(key);
        if (map === undefined) {
          map = new Int32Array(fragmentCounts.get(key) ?? 64).fill(-1);
          this.byChunk.set(key, map);
        }
        const ref = block.fragmentRef;
        const assign = (f: number) => {
          if (f >= map!.length) {
            const grown = new Int32Array(Math.max(f + 1, map!.length * 2)).fill(
              -1,
            );
            grown.set(map!);
            map = grown;
            this.byChunk.set(key, grown);
          }
          map![f] = row;
        };
        if (ref.mode === "single") assign(ref.fragmentIndex);
        else if (ref.mode === "range") {
          for (let i = 0; i < ref.count; ++i) assign(ref.start + i);
        } else for (const f of ref.indices) assign(f);
      }
    }
  }
  rows(chunkKey: string): Int32Array | undefined {
    return this.byChunk.get(chunkKey);
  }
}

export interface LevelPipelineOptions {
  read: ZarrArrayRead;
  shardIndexes: ShardIndexCache;
  listDirectories?: (path: string) => Promise<string[]>;
  description: ZarrVectorsGeometryDescription;
  level: ZarrVectorsLevel;
  warn: (message: string) => void;
}

export class LevelPipeline {
  readonly cells: LevelCells;
  private links: CrossChunkLinks | undefined;
  private fragmentObjects: Promise<FragmentObjectMap | undefined> | undefined;

  constructor(private options: LevelPipelineOptions) {
    const { level, description } = options;
    const known = new Map<string, any>();
    known.set("vertices", level.arrays.vertices);
    known.set("vertex_fragments", level.arrays.vertexFragments);
    if (description.hasObjects) {
      known.set(
        "fragment_attributes/segment_id",
        level.arrays.fragmentSegmentIds,
      );
    }
    description.attributes.forEach((a, i) =>
      known.set(`vertex_attributes/${a.name}`, level.arrays.attributes[i]),
    );
    this.cells = new LevelCells(
      options.read,
      options.shardIndexes,
      level.path,
      known,
    );
    const caps = KIND_CAPABILITIES[description.geometryKind];
    if (caps.edgeSource !== "none" && caps.primitive === "lines") {
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

  private get vertexDtype(): VertexAttributeDtype {
    return String(
      this.options.level.arrays.vertices.attributes?.dtype ?? "float32",
    ) as VertexAttributeDtype;
  }

  private objectMap(): Promise<FragmentObjectMap | undefined> {
    if (this.fragmentObjects === undefined) {
      const { level, read, shardIndexes } = this.options;
      const promise = (async () => {
        const access = {
          read,
          shardIndexes,
          listDirectories: async () => [],
        };
        const table = await readObjectTable(access, level.path);
        if (table === undefined || table.manifestsJson === undefined) {
          return undefined;
        }
        const manifests = await this.cells.reader("object_index/manifests");
        if (manifests === undefined) return undefined;
        const rows = await manifests.readVlenRows(0, table.numObjects);
        return new FragmentObjectMap(
          table,
          rows,
          this.options.description.rank,
          new Map(),
        );
      })();
      this.fragmentObjects = promise;
      promise.catch(() => {
        if (this.fragmentObjects === promise) this.fragmentObjects = undefined;
      });
    }
    return this.fragmentObjects;
  }

  /** Decodes the chunk at spatial cell `chunk`, or `undefined` if empty. */
  async download(
    chunk: number[],
    signal: AbortSignal,
  ): Promise<DenseChunkData | undefined> {
    const { description, level } = this.options;
    const chunkKey = chunk.join(".");
    if (!(await this.cells.mayHaveCell("vertices", chunkKey))) return undefined;
    const linked = description.linkedSkeletonLayout;
    const linksPromise =
      this.links === undefined
        ? Promise.resolve(undefined)
        : linked
          ? this.links.linksTouching(chunk, signal)
          : this.links.linksOwnedBy(chunk, signal);
    linksPromise.catch(() => {});
    const isLocal = (coords: readonly number[]) =>
      coords.every((c, d) => c === chunk[d]);
    let relinkedChildren: Set<number> | undefined;
    if (linked) {
      relinkedChildren = new Set();
      const table = await linksPromise;
      for (const record of table?.records ?? []) {
        const [child, parent] = record.endpoints;
        if (isLocal(child.chunkCoords) && !isLocal(parent.chunkCoords)) {
          relinkedChildren.add(child.vertexIndex);
        }
      }
    }
    const intraLinksJson = level.arrays.intraLinks;
    const downloadOptions: GeometryChunkDownloadOptions = {
      chunkKey,
      rank: description.rank,
      linkDtype: String(intraLinksJson?.attributes?.dtype ?? "int64") as any,
      attributeNames: description.attributes.map((a) => a.name),
      attributeDtypes: description.attributes.map(
        (a) => a.dtype as VertexAttributeDtype,
      ),
      attributeComponents: description.attributes.map((a) => a.components),
      vertexDtype: this.vertexDtype,
      linksConvention: description.linksConvention,
      geometryKind: description.geometryKind,
      hasFragmentSegmentIds: level.arrays.fragmentSegmentIds !== undefined,
      vertexIdAttribute: description.vertexIdAttribute,
      linkWidth: description.linkWidth,
      linkedSkeletonLayout: linked,
      relinkedChildren,
      cellRead: this.cells.cellReader,
    };
    const [decoded, linksTable] = await Promise.all([
      downloadGeometryChunk(downloadOptions, signal),
      linksPromise,
    ]);
    if (decoded === undefined) return undefined;
    // Neighbours' bridges into this chunk can now be resolved without a read.
    // `decoded.positions` is not transferred (the chunk's arrays are rebuilt
    // below), so it is safe to keep.
    positionCache.put(`${level.path}|${chunkKey}`, decoded.positions);

    // Global ids for stores without per-fragment segment ids.
    let segmentIds =
      decoded.segmentIds ?? new Uint32Array(decoded.numVertices * 2);
    if (description.hasObjects && decoded.segmentIdsAreGlobal !== true) {
      const map = await this.objectMap().catch((e) => {
        this.options.warn(
          `could not map fragments to objects (${e instanceof Error ? e.message : e}); ` +
            "objects are coloured per chunk",
        );
        return undefined;
      });
      const rows = map?.rows(chunkKey);
      if (map !== undefined && rows !== undefined) {
        segmentIds = new Uint32Array(decoded.numVertices * 2);
        const fi = decoded.fragmentIndex;
        for (let f = 0; f < fi.numFragments; ++f) {
          const row = f < rows.length ? rows[f] : -1;
          if (row < 0) continue;
          const id = map.table.segmentIds[row];
          const lo = Number(id & 0xffffffffn) >>> 0;
          const hi = Number(id >> 32n) >>> 0;
          for (const v of fi.indices(f)) {
            segmentIds[v * 2] = lo;
            segmentIds[v * 2 + 1] = hi;
          }
        }
      }
    }

    // Bridges: one ghost per link endpoint outside this chunk.
    interface Ghost {
      host: number;
      chunkKey: string;
      vertex: number;
      hostIsPredecessor: boolean;
    }
    const ghosts: Ghost[] = [];
    for (const record of linksTable?.records ??
      ([] as CrossChunkLinkRecord[])) {
      if (record.endpoints.length !== 2) continue;
      const [a, b] = record.endpoints;
      const aLocal = isLocal(a.chunkCoords);
      const bLocal = isLocal(b.chunkCoords);
      if (aLocal === bLocal) continue;
      if (linked && !aLocal) continue; // drawn by the child's chunk
      const host = aLocal ? a : b;
      const far = aLocal ? b : a;
      if (host.vertexIndex >= decoded.numVertices) continue;
      ghosts.push({
        host: host.vertexIndex,
        chunkKey: far.chunkCoords.join("."),
        vertex: far.vertexIndex,
        hostIsPredecessor: aLocal,
      });
    }
    const ghostPositions = await this.fetchGhostPositions(ghosts, signal);

    return assembleChunk(
      decoded,
      segmentIds,
      ghosts,
      ghostPositions,
      attributeWidths(description),
    );
  }

  /** Positions of the requested neighbour vertices (NaN where unreadable). */
  private async fetchGhostPositions(
    ghosts: readonly { chunkKey: string; vertex: number }[],
    signal: AbortSignal,
  ): Promise<Float32Array> {
    const { rank } = this.options.description;
    const out = new Float32Array(ghosts.length * rank).fill(Number.NaN);
    if (ghosts.length === 0) return out;
    const byChunk = new Map<string, number[]>();
    ghosts.forEach((g, i) => {
      let list = byChunk.get(g.chunkKey);
      if (list === undefined) byChunk.set(g.chunkKey, (list = []));
      list.push(i);
    });
    const dtype = this.vertexDtype;
    const elementBytes = ATTRIBUTE_ELEMENT_BYTES[dtype];
    const vertexBytes = rank * elementBytes;
    await mapConcurrent([...byChunk], 8, async ([key, indices]) => {
      const vertices = [...new Set(indices.map((i) => ghosts[i].vertex))].sort(
        (x, y) => x - y,
      );
      const found = new Map<number, Float32Array>();
      const cacheKey = `${this.options.level.path}|${key}`;
      const fromWhole = (all: Float32Array | undefined) => {
        if (all === undefined) return;
        for (const v of vertices) {
          if ((v + 1) * rank <= all.length) {
            found.set(v, all.subarray(v * rank, (v + 1) * rank));
          }
        }
      };
      const cached = positionCache.peek(cacheKey);
      if (cached !== undefined) {
        fromWhole(await cached.catch(() => undefined));
      } else {
        // Coalesce rows into few reads; past a handful, read the whole cell
        // once (and keep it for the other chunks bridging into it).
        const spans: [number, number][] = [];
        for (const v of vertices) {
          const last = spans[spans.length - 1];
          if (last !== undefined && v - last[1] <= RANGE_GAP_VERTICES) {
            last[1] = v;
          } else {
            spans.push([v, v]);
          }
        }
        let rangeable = spans.length <= MAX_RANGE_READS;
        if (rangeable) {
          const reads = await Promise.all(
            spans.map(([first, last]) =>
              this.cells.readCellRange(
                "vertices",
                key,
                first * vertexBytes,
                (last - first + 1) * vertexBytes,
                signal,
              ),
            ),
          );
          spans.forEach(([first, last], s) => {
            const bytes = reads[s];
            if (bytes === null) {
              rangeable = false;
              return;
            }
            if (bytes === undefined) return;
            const values = decodeAttributeToFloat32(
              bytes,
              dtype,
              (last - first + 1) * rank,
            );
            for (let v = first; v <= last; ++v) {
              found.set(
                v,
                values.subarray((v - first) * rank, (v - first + 1) * rank),
              );
            }
          });
        }
        if (!rangeable) {
          fromWhole(
            await positionCache.get(cacheKey, async () => {
              const bytes = await this.cells.readCell(
                "vertices",
                key,
                SHARED_SIGNAL,
              );
              if (bytes === undefined) return undefined;
              return decodeAttributeToFloat32(
                bytes,
                dtype,
                bytes.byteLength / elementBytes,
              );
            }),
          );
        }
      }
      for (const i of indices) {
        const p = found.get(ghosts[i].vertex);
        if (p !== undefined) out.set(p, i * rank);
      }
    });
    return out;
  }
}

/**
 * Builds the final chunk arrays in one allocation each: the decoded vertices,
 * then one ghost per bridge (copying its host's segment id and attributes,
 * with a tangent along the bridge in the curve's walk direction).
 */
function assembleChunk(
  decoded: NonNullable<Awaited<ReturnType<typeof downloadGeometryChunk>>>,
  segmentIds: Uint32Array,
  ghosts: readonly { host: number; hostIsPredecessor: boolean }[],
  ghostPositions: Float32Array,
  widths: number[],
): DenseChunkData {
  const { rank } = decoded;
  const own = decoded.numVertices;
  const kept: number[] = [];
  for (let g = 0; g < ghosts.length; ++g) {
    if (!Number.isNaN(ghostPositions[g * rank])) kept.push(g);
  }
  const total = own + kept.length;
  const positions = new Float32Array(total * 3);
  for (let v = 0; v < own; ++v) {
    for (let d = 0; d < 3; ++d) {
      positions[v * 3 + d] = d < rank ? decoded.positions[v * rank + d] : 0;
    }
  }
  const ids = new Uint32Array(total * 2);
  ids.set(segmentIds.subarray(0, own * 2));
  const edges = new Uint32Array(decoded.edges.length + kept.length * 2);
  edges.set(decoded.edges);
  const sources = [...decoded.vertexAttributes];
  if (decoded.tangents !== undefined) sources.push(decoded.tangents);
  const attributes = widths.map((width, i) => {
    const out = new Float32Array(total * width);
    const src = sources[i] as Float32Array | undefined;
    if (src !== undefined) out.set(src.subarray(0, own * width));
    return out;
  });
  const tangentIndex = decoded.tangents !== undefined ? widths.length - 1 : -1;
  kept.forEach((g, k) => {
    const v = own + k;
    const { host, hostIsPredecessor } = ghosts[g];
    for (let d = 0; d < 3; ++d) {
      positions[v * 3 + d] = d < rank ? ghostPositions[g * rank + d] : 0;
    }
    ids[v * 2] = ids[host * 2];
    ids[v * 2 + 1] = ids[host * 2 + 1];
    edges[decoded.edges.length + k * 2] = host;
    edges[decoded.edges.length + k * 2 + 1] = v;
    attributes.forEach((out, i) => {
      const width = widths[i];
      if (i === tangentIndex) {
        const sign = hostIsPredecessor ? 1 : -1;
        let dx = 0;
        let norm = 0;
        const dir = [0, 0, 0];
        for (let d = 0; d < 3; ++d) {
          dx = sign * (positions[v * 3 + d] - positions[host * 3 + d]);
          dir[d] = dx;
          norm += dx * dx;
        }
        norm = Math.sqrt(norm);
        for (let d = 0; d < 3; ++d) {
          out[v * 3 + d] = norm > 0 ? dir[d] / norm : out[host * 3 + d];
        }
        return;
      }
      out.copyWithin(v * width, host * width, (host + 1) * width);
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
