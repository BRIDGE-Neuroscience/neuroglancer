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
 * @file Whole objects, for Neuroglancer's own skeleton and mesh layers:
 * segment id -> row -> manifest -> the fragments it owns in each chunk.
 * Skeletons are read whole at the finest level; meshes one chunk (octree node)
 * at a time, at any level. Decoded chunks and manifest chunks are shared
 * between the objects that pass through them.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/base.js";
import type { DecodedChunk } from "#src/datasource/zarr-vectors/chunk_decode.js";
import {
  decodeChunk,
  edgeTangents,
  forEachFragmentVertex,
} from "#src/datasource/zarr-vectors/chunk_decode.js";
import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  decodeFloat32,
  decodeIndices,
  ELEMENT_BYTES,
  isElementType,
} from "#src/datasource/zarr-vectors/dtype.js";
import type { FragmentIndex } from "#src/datasource/zarr-vectors/fragment_index.js";
import { decodeFragments } from "#src/datasource/zarr-vectors/fragment_index.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { LevelCells } from "#src/datasource/zarr-vectors/level_cells.js";
import type { CrossChunkLinkRecord } from "#src/datasource/zarr-vectors/links.js";
import {
  CrossChunkLinks,
  intraLinksPath,
  ownedLinksInRows,
} from "#src/datasource/zarr-vectors/links.js";
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
  mapConcurrent,
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

/**
 * One skeleton from parts read at several levels (an additive pyramid keeps
 * each part of an object at one level): vertices concatenated, edges
 * renumbered, attributes (and the tangent) concatenated alike.
 */
export function mergeSkeletons(
  parts: readonly ObjectSkeleton[],
): ObjectSkeleton {
  const used = parts.filter((p) => p.positions.length > 0);
  if (used.length === 1) return used[0];
  if (used.length === 0) return parts[0];
  const numVertices = used.reduce((n, p) => n + p.positions.length / 3, 0);
  const positions = new Float32Array(3 * numVertices);
  const edges = new Uint32Array(used.reduce((n, p) => n + p.edges.length, 0));
  const attributes = used[0].attributes.map(
    (_, k) =>
      new Float32Array(
        used.reduce((n, p) => n + (p.attributes[k]?.length ?? 0), 0),
      ),
  );
  let vertexAt = 0;
  let edgeAt = 0;
  const attributeAt = attributes.map(() => 0);
  for (const p of used) {
    positions.set(p.positions, 3 * vertexAt);
    for (let i = 0; i < p.edges.length; ++i) {
      edges[edgeAt + i] = p.edges[i] + vertexAt;
    }
    edgeAt += p.edges.length;
    p.attributes.forEach((a, k) => {
      if (k >= attributes.length) return;
      attributes[k].set(a, attributeAt[k]);
      attributeAt[k] += a.length;
    });
    vertexAt += p.positions.length / 3;
  }
  return { positions, edges, attributes };
}

/** One object's vertices in each chunk it visits, in manifest (walk) order. */
interface ObjectParts {
  order: { chunkKey: string; vertices: number[] }[];
  chunks: Map<string, DecodedChunk>;
}

/** Faces sampled to measure a level's edge length. */
const SAMPLE_FACES = 5000;

/** Rows `[start, start + count)` of a chunk and their positions. */
interface RowSpan {
  start: number;
  count: number;
  /** x, y, z per row. */
  positions: Float32Array;
}

/** One object's part of a chunk: its faces there and its vertices. */
interface LocalPart {
  /**
   * Chunk-local vertex triples. A face with a corner outside `spans` is not
   * the object's and is left out.
   */
  faces: Uint32Array;
  /** The object's rows of the chunk, ascending by start. */
  spans: RowSpan[];
}

function decodedBytes(chunk: DecodedChunk | undefined) {
  if (chunk === undefined) return 0;
  let bytes =
    chunk.positions.byteLength +
    chunk.edges.byteLength +
    // Twice: meshes also keep the faces grouped by fragment.
    2 * chunk.faces.byteLength +
    chunk.fragments.byteLength +
    (chunk.tangents?.byteLength ?? 0);
  for (const a of chunk.attributes) bytes += a.byteLength;
  return bytes;
}

/**
 * A decoded chunk's faces grouped by the fragment of their first corner,
 * built once per chunk, so each object's faces are found without scanning
 * every object's.
 */
const facesByFragment = new WeakMap<
  DecodedChunk,
  { offsets: Uint32Array; faces: Uint32Array }
>();

function fragmentFaces(decoded: DecodedChunk, fragment: number) {
  let index = facesByFragment.get(decoded);
  if (index === undefined) {
    const { fragments, faces, numVertices } = decoded;
    const owner = new Int32Array(numVertices).fill(-1);
    for (let f = 0; f < fragments.numFragments; ++f) {
      forEachFragmentVertex(fragments, f, (v) => {
        if (v < numVertices) owner[v] = f;
      });
    }
    const offsets = new Uint32Array(fragments.numFragments + 1);
    for (let i = 0; i + 3 <= faces.length; i += 3) {
      const f = owner[faces[i]];
      if (f >= 0) offsets[f + 1] += 3;
    }
    for (let f = 0; f < fragments.numFragments; ++f) {
      offsets[f + 1] += offsets[f];
    }
    const next = offsets.slice(0, fragments.numFragments);
    const grouped = new Uint32Array(offsets[fragments.numFragments]);
    for (let i = 0; i + 3 <= faces.length; i += 3) {
      const f = owner[faces[i]];
      if (f < 0) continue;
      grouped.set(faces.subarray(i, i + 3), next[f]);
      next[f] += 3;
    }
    index = { offsets, faces: grouped };
    facesByFragment.set(decoded, index);
  }
  return index.faces.subarray(
    index.offsets[fragment],
    index.offsets[fragment + 1],
  );
}

interface StoreCaches {
  chunks: AsyncLru<DecodedChunk | undefined>;
  /** Raw vertex cells, shared by chunk decodes and faces reaching across. */
  vertexCells: AsyncLru<Uint8Array | undefined>;
}

const caches = new WeakMap<ZarrVectorsStoreAccess, StoreCaches>();

/**
 * Decoded chunks are large (tens of MB at a whole-brain store's finest
 * level) and not charged to the chunk manager, so the caches are bounded by
 * bytes -- and shared by every reader of a store (one per level of detail,
 * per kind of source), so the bound is per store, not per reader. Objects
 * drawn together mostly share chunks.
 */
function storeCaches(access: ZarrVectorsStoreAccess): StoreCaches {
  let out = caches.get(access);
  if (out === undefined) {
    out = {
      chunks: new AsyncLru<DecodedChunk | undefined>(
        Infinity,
        384 * 1024 * 1024,
        decodedBytes,
      ),
      vertexCells: new AsyncLru<Uint8Array | undefined>(
        Infinity,
        128 * 1024 * 1024,
        (b) => b?.byteLength ?? 0,
      ),
    };
    caches.set(access, out);
  }
  return out;
}

export class ObjectReader {
  private cells: LevelCells;
  private links: CrossChunkLinks | undefined;
  private index:
    | Promise<{ rows: SegmentIdIndex; chunkRows: number } | undefined>
    | undefined;
  /** Caches shared with the store's other readers, and this one's keys. */
  private caches: StoreCaches;
  private cacheKey: string;
  private manifestChunks = new AsyncLru<Uint8Array[]>(16);
  private fragmentIndexes = new AsyncLru<FragmentIndex | undefined>(1024);

  constructor(
    private access: ZarrVectorsStoreAccess,
    private description: ZarrVectorsGeometryDescription,
    private level: ZarrVectorsLevel,
  ) {
    this.cells = LevelCells.forLevel(access, level, description);
    this.caches = storeCaches(access);
    // A decoded chunk depends on the level and the attributes selected.
    this.cacheKey = `${level.path}|${description.attributes
      .map((a) => a.name)
      .join(",")}|`;
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
    const decoded = await Promise.all(keys.map((key) => this.decode(key)));
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
      // A stored link replaces a row's implied parent only where parents
      // are implied: explicit stores list every edge.
      const relinks =
        linked &&
        description.linksConvention === "implicit_sequential_with_branches";
      const relinked = new Set<string>();
      const chunkKeys = [...indexOf.keys()];
      const linkSets = new Array<CrossChunkLinkRecord[]>(chunkKeys.length);
      await mapConcurrent(chunkKeys, 16, async (chunkKey, i) => {
        const coords = chunkKey.split(".").map(Number);
        linkSets[i] =
          (await (linked
            ? this.links?.linksTouching(coords)
            : this.links?.linksOwnedBy(coords))) ?? [];
      });
      signal.throwIfAborted();
      for (const [i, chunkKey] of chunkKeys.entries()) {
        for (const { endpoints } of linkSets[i]) {
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
          if (relinks) relinked.add(`${keys[0]}:${endpoints[0].vertexIndex}`);
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

  /** The chunks an object has geometry in at this level. */
  async chunksOf(segmentId: bigint): Promise<number[][]> {
    const blocks = await this.manifest(segmentId);
    const keys = new Set(blocks.map((b) => b.chunkCoords.join(".")));
    return [...keys].map((key) => key.split(".").map(Number));
  }

  private vertexCell(chunkKey: string) {
    return this.caches.vertexCells.get(`${this.level.path}|${chunkKey}`, () =>
      this.cells.readCell("vertices", chunkKey, SHARED_SIGNAL),
    );
  }

  /** A chunk decoded once for every object and level of detail that needs it. */
  private decode(chunkKey: string) {
    return this.caches.chunks.get(this.cacheKey + chunkKey, () =>
      decodeChunk(this.cells, this.description, chunkKey, SHARED_SIGNAL, {
        skipSegmentIds: true,
        // Skeletons get tangents once they are assembled; mesh nodes use
        // positions alone.
        skipTangents: true,
        skipAttributes:
          KIND_CAPABILITIES[this.description.geometryKind].primitive ===
          "triangles",
        vertices: this.vertexCell(chunkKey),
      }),
    );
  }

  private get vertexType(): ElementType {
    return this.level.arrays.vertices.attributes?.dtype ?? "float32";
  }

  /**
   * The level's intra-chunk face array, when the level is stamped
   * `fragment_link_groups`: one face group per vertex fragment, in fragment
   * order, as zarr-vectors-py's mesh writers and `zvtools index-faces` leave
   * it.
   */
  private get groupedFaces() {
    const { level, description } = this;
    const array = level.arrays.intraLinks;
    if (
      !level.fragmentLinkGroups ||
      array === undefined ||
      KIND_CAPABILITIES[description.geometryKind].primitive !== "triangles" ||
      // Rows are read as triangles; wider faces go through the whole cell.
      description.linkWidth !== 3
    ) {
      return undefined;
    }
    const type = array.attributes?.dtype ?? "int64";
    return isElementType(type)
      ? { path: intraLinksPath(description.linkWidth), type }
      : undefined;
  }

  /** A chunk's vertex fragments or face groups, decoded. */
  private fragmentIndex(path: string, chunkKey: string) {
    return this.fragmentIndexes.get(`${path}|${chunkKey}`, async () => {
      const bytes = await this.cells.readCell(path, chunkKey, SHARED_SIGNAL);
      return bytes === undefined ? undefined : decodeFragments(bytes);
    });
  }

  private inconsistent(why: string) {
    warnOnce(
      `${this.cells.levelPath}: is stamped fragment_link_groups, ` +
        `but ${why}; those chunks are read whole`,
    );
  }

  /**
   * The object's part of a chunk read from its own rows only: its vertex
   * rows and its face group, by byte range. `undefined` unless the level
   * is stamped `fragment_link_groups` and the chunk bears that out
   * (uncompressed, one group per fragment, each a row range whose faces
   * index only its fragment) -- the caller then reads the whole cell.
   */
  private async partFromRows(
    chunkKey: string,
    fragmentIds: readonly number[],
    signal: AbortSignal,
  ): Promise<LocalPart | undefined> {
    const faceArray = this.groupedFaces;
    if (faceArray === undefined) return undefined;
    try {
      return await this.readRowsPart(chunkKey, fragmentIds, faceArray, signal);
    } catch (e) {
      // An unreadable sidecar, or one pointing past its cell.
      signal.throwIfAborted();
      this.inconsistent("reading a chunk's own rows failed");
      console.debug(e);
      return undefined;
    }
  }

  private async readRowsPart(
    chunkKey: string,
    fragmentIds: readonly number[],
    faceArray: { path: string; type: ElementType },
    signal: AbortSignal,
  ): Promise<LocalPart | undefined> {
    const { cells } = this;
    if (
      !(await cells.rangeAddressable("vertices")) ||
      !(await cells.rangeAddressable(faceArray.path))
    ) {
      return undefined;
    }
    const [fragments, groups, mayHaveFaces] = await Promise.all([
      this.fragmentIndex("vertex_fragments", chunkKey),
      this.fragmentIndex("link_fragments", chunkKey),
      cells.mayHaveCell(faceArray.path, chunkKey),
    ]);
    if (fragments === undefined) return undefined;
    // No face cell where the level lists its cells: every face here
    // crosses into a neighbour, so each object's group is empty.
    const noFaces = groups === undefined && !mayHaveFaces;
    if (groups === undefined && !noFaces) return undefined;
    if (
      groups !== undefined &&
      groups.numFragments !== fragments.numFragments
    ) {
      this.inconsistent("a chunk's face groups do not match its fragments");
      return undefined;
    }
    // Every fragment a row range with a row-range group, before any read.
    for (const f of fragmentIds) {
      if (
        f >= fragments.numFragments ||
        !fragments.isRange(f) ||
        (groups !== undefined && !groups.isRange(f))
      ) {
        return undefined;
      }
    }
    const vertexType = this.vertexType;
    const vertexRow = 3 * ELEMENT_BYTES[vertexType];
    const faceRow = 3 * ELEMENT_BYTES[faceArray.type];
    const parts = await Promise.all(
      fragmentIds.map(async (f) => {
        const { start, count } = fragments.range(f);
        const group = groups?.range(f) ?? { start: 0, count: 0 };
        const [vertexBytes, faceBytes] = await Promise.all([
          count === 0
            ? new Uint8Array(0)
            : cells.readCellRange(
                "vertices",
                chunkKey,
                start * vertexRow,
                count * vertexRow,
                signal,
              ),
          group.count === 0
            ? new Uint8Array(0)
            : cells.readCellRange(
                faceArray.path,
                chunkKey,
                group.start * faceRow,
                group.count * faceRow,
                signal,
              ),
        ]);
        if (vertexBytes == null || faceBytes == null) return undefined;
        const faces = decodeIndices(faceBytes, faceArray.type, 3 * group.count);
        for (const v of faces) {
          if (v < start || v >= start + count) {
            this.inconsistent(
              "a face group indexes another fragment's vertices",
            );
            return undefined;
          }
        }
        const positions = decodeFloat32(vertexBytes, vertexType, 3 * count);
        return { span: { start, count, positions }, faces };
      }),
    );
    if (parts.some((p) => p === undefined)) return undefined;
    const done = parts as NonNullable<(typeof parts)[number]>[];
    const faces = new Uint32Array(done.reduce((n, p) => n + p.faces.length, 0));
    let at = 0;
    for (const p of done) {
      faces.set(p.faces, at);
      at += p.faces.length;
    }
    return {
      faces,
      spans: done.map((p) => p.span).sort((a, b) => a.start - b.start),
    };
  }

  /** The object's part of a chunk, from the whole decoded cell. */
  private async partFromCell(
    chunkKey: string,
    fragmentIds: readonly number[],
  ): Promise<LocalPart | undefined> {
    const decoded = await this.decode(chunkKey);
    if (decoded === undefined) return undefined;
    const { fragments, positions, numVertices } = decoded;
    const spans: RowSpan[] = [];
    const faceParts: Uint32Array[] = [];
    for (const f of new Set(fragmentIds)) {
      if (f >= fragments.numFragments) continue;
      const rows = (start: number, count: number) => {
        count = Math.min(count, numVertices - start);
        if (count > 0) {
          spans.push({
            start,
            count,
            positions: positions.subarray(3 * start, 3 * (start + count)),
          });
        }
      };
      if (fragments.isRange(f)) {
        const { start, count } = fragments.range(f);
        rows(start, count);
      } else {
        forEachFragmentVertex(fragments, f, (v) => rows(v, 1));
      }
      faceParts.push(fragmentFaces(decoded, f));
    }
    const faces = new Uint32Array(faceParts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of faceParts) {
      faces.set(p, at);
      at += p.length;
    }
    return { faces, spans: spans.sort((a, b) => a.start - b.start) };
  }

  /** Positions of some rows of a neighbour, decoded already or read. */
  private async farPositions(
    chunkKey: string,
    rows: readonly number[],
    signal: AbortSignal,
  ): Promise<Map<number, Float32Array>> {
    const cached = this.caches.chunks.peek(this.cacheKey + chunkKey);
    const decoded = cached && (await cached.catch(() => undefined));
    if (decoded !== undefined) {
      const out = new Map<number, Float32Array>();
      for (const v of rows) {
        if (3 * v + 3 <= decoded.positions.length) {
          out.set(v, decoded.positions.subarray(3 * v, 3 * v + 3));
        }
      }
      return out;
    }
    return this.cells.vertexRows(chunkKey, this.vertexType, rows, signal, () =>
      this.vertexCell(chunkKey),
    );
  }

  /**
   * An object's triangles stored in one chunk: its faces inside the chunk,
   * and the faces the chunk stores across its faces (their other corners are
   * read from the neighbours).  Every face of the object is in exactly one
   * chunk's set.  Reads only the object's rows where the store allows it
   * (`partFromRows`), else the whole cell.
   */
  async readMeshNode(
    segmentId: bigint,
    chunk: readonly number[],
    signal: AbortSignal,
  ): Promise<{ positions: Float32Array; indices: Uint32Array }> {
    const empty = {
      positions: new Float32Array(0),
      indices: new Uint32Array(0),
    };
    const chunkKey = chunk.join(".");
    const fragmentIds = [
      ...new Set(
        (await this.manifest(segmentId))
          .filter((b) => b.chunkCoords.join(".") === chunkKey)
          .flatMap((b) => [...resolveFragmentRef(b.fragmentRef)]),
      ),
    ];
    if (fragmentIds.length === 0) return empty;
    const part =
      (await this.partFromRows(chunkKey, fragmentIds, signal)) ??
      (await this.partFromCell(chunkKey, fragmentIds));
    signal.throwIfAborted();
    if (part === undefined) return empty;

    // The object's rows here, numbered in row order: output vertices.
    const { spans, faces } = part;
    let lo = Infinity;
    let hi = -Infinity;
    let total = 0;
    for (const { start, count } of spans) {
      if (count === 0) continue;
      lo = Math.min(lo, start);
      hi = Math.max(hi, start + count);
      total += count;
    }
    if (total === 0) return empty;
    const remap = new Int32Array(hi - lo).fill(-1);
    const local = new Float32Array(3 * total);
    let numLocal = 0;
    for (const { start, count, positions } of spans) {
      for (let i = 0; i < count; ++i) {
        if (remap[start + i - lo] !== -1) continue;
        remap[start + i - lo] = numLocal;
        local.set(positions.subarray(3 * i, 3 * i + 3), 3 * numLocal);
        ++numLocal;
      }
    }
    const localIndex = (v: number) => (v >= lo && v < hi ? remap[v - lo] : -1);

    const intra = new Uint32Array(faces.length);
    let numIntra = 0;
    for (let i = 0; i + 3 <= faces.length; i += 3) {
      const a = localIndex(faces[i]);
      const b = localIndex(faces[i + 1]);
      const c = localIndex(faces[i + 2]);
      if (a < 0 || b < 0 || c < 0) continue;
      intra[numIntra++] = a;
      intra[numIntra++] = b;
      intra[numIntra++] = c;
    }

    // Faces this chunk stores across its faces. One object's: its corners
    // here say which, so they are looked up by the object's rows; the other
    // corners are read from the neighbours.
    const isLocal = (coords: readonly number[]) =>
      coords.every((c, d) => c === chunk[d]);
    const crossing: CrossChunkLinkRecord[] = [];
    if (this.links !== undefined) {
      const owned = await this.links.ownedByLocalVertex(chunk);
      // Merged row ranges, so no record is found twice.
      const ranges: [number, number][] = [];
      for (const { start, count } of spans) {
        const last = ranges[ranges.length - 1];
        if (last !== undefined && start <= last[1]) {
          last[1] = Math.max(last[1], start + count);
        } else if (count > 0) {
          ranges.push([start, start + count]);
        }
      }
      for (const [first, end] of ranges) {
        for (const record of ownedLinksInRows(owned, first, end)) {
          if (record.endpoints.length < 3) continue;
          if (
            record.endpoints.every(
              (e) => !isLocal(e.chunkCoords) || localIndex(e.vertexIndex) >= 0,
            )
          ) {
            crossing.push(record);
          }
        }
      }
    }
    const farRows = new Map<string, Set<number>>();
    for (const { endpoints } of crossing) {
      for (const { chunkCoords, vertexIndex } of endpoints) {
        if (isLocal(chunkCoords)) continue;
        const key = chunkCoords.join(".");
        let rows = farRows.get(key);
        if (rows === undefined) farRows.set(key, (rows = new Set()));
        rows.add(vertexIndex);
      }
    }
    const far = new Map(
      await Promise.all(
        [...farRows].map(
          async ([key, rows]) =>
            [
              key,
              await this.farPositions(
                key,
                [...rows].sort((a, b) => a - b),
                signal,
              ),
            ] as const,
        ),
      ),
    );
    const farIndex = new Map<string, number>();
    const farPositions: number[] = [];
    const across: number[] = [];
    for (const { endpoints } of crossing) {
      const ids: number[] = [];
      for (const { chunkCoords, vertexIndex } of endpoints) {
        if (isLocal(chunkCoords)) {
          ids.push(localIndex(vertexIndex));
          continue;
        }
        const key = `${chunkCoords.join(".")}:${vertexIndex}`;
        let id = farIndex.get(key);
        if (id === undefined) {
          const p = far.get(chunkCoords.join("."))?.get(vertexIndex);
          if (p === undefined) break;
          id = numLocal + farPositions.length / 3;
          farPositions.push(p[0], p[1], p[2]);
          farIndex.set(key, id);
        }
        ids.push(id);
      }
      if (ids.length !== endpoints.length) continue;
      for (let k = 1; k + 1 < ids.length; ++k) {
        across.push(ids[0], ids[k], ids[k + 1]);
      }
    }
    signal.throwIfAborted();
    const positions = new Float32Array(3 * numLocal + farPositions.length);
    positions.set(local.subarray(0, 3 * numLocal));
    positions.set(farPositions, 3 * numLocal);
    const indices = new Uint32Array(numIntra + across.length);
    indices.set(intra.subarray(0, numIntra));
    indices.set(across, numIntra);
    return { positions, indices };
  }

  /**
   * The first faces of a chunk's face cell and their corners: read by byte
   * range (a few thousand faces and just their vertices) where the cell
   * allows it, else from the whole decoded cell.
   */
  private async faceSample(chunkKey: string): Promise<
    | {
        faces: ArrayLike<number>;
        position(vertex: number): ArrayLike<number> | undefined;
      }
    | undefined
  > {
    const path = intraLinksPath(this.description.linkWidth);
    const type =
      (await this.cells.reader(path))?.array.attributes?.dtype ?? "int64";
    const length = isElementType(type)
      ? await this.cells.cellPayloadLength(path, chunkKey, SHARED_SIGNAL)
      : null;
    if (length === null || !isElementType(type)) {
      const decoded = await this.decode(chunkKey);
      if (decoded === undefined || decoded.faces.length < 3) return undefined;
      const { positions } = decoded;
      return {
        faces: decoded.faces.subarray(0, 3 * SAMPLE_FACES),
        position: (v) => positions.subarray(3 * v, 3 * v + 3),
      };
    }
    const rowBytes = 3 * ELEMENT_BYTES[type];
    const rows = Math.min(SAMPLE_FACES, Math.floor(length / rowBytes));
    if (rows === 0) return undefined;
    const bytes = await this.cells.readCellRange(
      path,
      chunkKey,
      0,
      rows * rowBytes,
      SHARED_SIGNAL,
    );
    if (bytes == null) return undefined;
    const faces = decodeIndices(bytes, type, 3 * rows);
    const found = await this.cells.vertexRows(
      chunkKey,
      this.vertexType,
      [...new Set(faces)].sort((a, b) => a - b),
      SHARED_SIGNAL,
      () => this.vertexCell(chunkKey),
    );
    return { faces, position: (v) => found.get(v) };
  }

  /**
   * Mean edge length of some faces in one populated chunk: the level's
   * resolution, for choosing between levels of detail.
   */
  async meanEdgeLength(): Promise<number | undefined> {
    const reader = await this.cells.reader("vertices");
    if (reader === undefined) return undefined;
    const { nonemptyCells, shape, origin } = reader.array;
    const candidates = nonemptyCells
      ? [...nonemptyCells].slice(0, 8)
      : Array.from(
          { length: Math.min(64, shape[0] * shape[1] * shape[2]) },
          (_, i) =>
            [
              Math.floor(i / (shape[1] * shape[2])) + origin[0],
              (Math.floor(i / shape[2]) % shape[1]) + origin[1],
              (i % shape[2]) + origin[2],
            ].join("."),
        );
    const measure = async (key: string) => {
      const sample = await this.faceSample(key);
      if (sample === undefined) return undefined;
      const { faces, position } = sample;
      let total = 0;
      let count = 0;
      for (let i = 0; i + 3 <= faces.length; i += 3) {
        for (let k = 0; k < 3; ++k) {
          const a = position(faces[i + k]);
          const b = position(faces[i + ((k + 1) % 3)]);
          if (a === undefined || b === undefined) continue;
          total += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
          ++count;
        }
      }
      const mean = total / count;
      return count > 0 && Number.isFinite(mean) && mean > 0 ? mean : undefined;
    };
    // Listed cells hold data, so the first usually answers; unlisted ones
    // are guesses, mostly empty, so several are tried at once.
    const batch = nonemptyCells ? 1 : 8;
    for (let i = 0; i < candidates.length; i += batch) {
      const means = await Promise.all(
        candidates.slice(i, i + batch).map(measure),
      );
      const found = means.find((m) => m !== undefined);
      if (found !== undefined) return found;
    }
    return undefined;
  }
}
