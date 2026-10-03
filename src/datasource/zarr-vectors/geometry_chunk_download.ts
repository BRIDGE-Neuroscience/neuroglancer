/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Worker-side orchestrator: fetch the byte blobs for one spatial chunk
 * of a zarr-vectors skeleton / polyline / streamline store and decode
 * them into a `SkeletonChunk` ready for upload to the render layer.
 *
 * This module is deliberately decoupled from neuroglancer's SharedObject
 * / RPC scaffolding so it can be unit-tested with a mock kvstore.  The
 * slice-4 chunk-source backend will provide a real kvstore reader from
 * `SharedKvStoreContext` and forward the result.
 */

import { decodeFragments } from "#src/datasource/zarr-vectors/fragment_index.js";
import {
  buildGeometryChunk,
  type AttributeTypedArray,
  type LinksConvention,
  type SkeletonChunk,
  type GeometryKind,
} from "#src/datasource/zarr-vectors/geometry_chunk.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import type { CellReader } from "#src/datasource/zarr-vectors/level_cells.js";
import {
  intraOffsets,
  linksPath,
} from "#src/datasource/zarr-vectors/links_paths.js";
import type { VertexAttributeDtype } from "#src/datasource/zarr-vectors/vertex_attribute_float.js";
import {
  ATTRIBUTE_ELEMENT_BYTES,
  decodeAttributeExactInts,
  decodeAttributeToFloat32,
  isExactIntDtype,
  zeroAttribute,
} from "#src/datasource/zarr-vectors/vertex_attribute_float.js";

/** Supported on-disk integer dtype for `links/0/<chunk>`. */
export type LinkDtype =
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "int64";

/**
 * Supported on-disk dtype for a per-vertex attribute. Every one of them
 * decodes to float32 before it leaves this module -- see
 * `vertex_attribute_float.ts` for why.
 */
export type AttributeDtype = VertexAttributeDtype;

/**
 * Inputs the orchestrator needs to download a chunk.
 *
 * The caller is responsible for joining `baseUrl + "<sub-path>"` to form
 * the actual fetch URLs and for any decompression (zstd) before handing
 * raw bytes to the decoder via the `cellRead` callback.
 *
 * `cellRead` returns `undefined` for a missing key (sparse chunk
 * presence) — the orchestrator interprets that as "no data here" and
 * returns an empty `SkeletonChunk` when even the vertex blob is absent.
 */
export interface GeometryChunkDownloadOptions {
  /** Spatial chunk key, e.g. `"3.0.2"`. */
  readonly chunkKey: string;
  /** Rank of the position vectors (== sid_ndim). */
  readonly rank: number;
  /**
   * On-disk dtype of `links/0/<chunk>` (per `.zattrs.dtype`); used to
   * reinterpret link bytes correctly.  Not consulted for
   * `implicit_sequential` stores (which don't have a `links/0` array).
   */
  readonly linkDtype: LinkDtype;
  /** Per-vertex attribute names, in the order the render layer expects. */
  readonly attributeNames: readonly string[];
  /** Per-vertex attribute dtypes, parallel to `attributeNames`. */
  readonly attributeDtypes: readonly AttributeDtype[];
  /** Values per vertex of each attribute, parallel to `attributeNames` (default 1). */
  readonly attributeComponents?: readonly number[];
  /** On-disk element type of `vertices` (default float32); decoded to float32. */
  readonly vertexDtype?: AttributeDtype;
  /**
   * zarr-vectors-tools' "linked" skeleton layout: a stored `[child, parent]`
   * record replaces the parent implied by row order instead of adding to it.
   */
  readonly linkedSkeletonLayout?: boolean;
  /**
   * Vertices whose implied parent edge is replaced by a cross-chunk record
   * (linked layout only); see {@link linkedSkeletonLayout}.
   */
  readonly relinkedChildren?: ReadonlySet<number>;
  /** How vertex-to-vertex edges are encoded for this geometry type. */
  readonly linksConvention: LinksConvention;
  /** Geometry kind (drives whether per-vertex tangents are precomputed). */
  readonly geometryKind: GeometryKind;
  /**
   * Whether `fragment_attributes/segment_id` chunks are present in this
   * level.  When false the fetch is skipped entirely (avoids 404 noise for
   * stores that never wrote per-fragment segment ids, e.g. streamline stores).
   * Defaults to true for backward-compatibility with callers that don't set it.
   */
  readonly hasFragmentSegmentIds?: boolean;
  /**
   * Name of a per-vertex attribute (one of {@link attributeNames}) carrying a
   * meaningful integer id for each vertex -- a cell label, a particle id.  Used
   * only by kinds without the discrete-object model (`point_cloud`), where it
   * is the one way to get ids that mean something and stay stable across
   * pyramid levels.  Absent: ids are synthesised from the chunk key and the
   * vertex's index within the chunk.
   */
  readonly vertexIdAttribute?: string;
  /**
   * The links family's declared `link_width`. 2 for edges; >= 3 for surface
   * faces (3 = triangles, 4 = quads). Defaults to 2, which is every non-mesh
   * geometry.
   */
  readonly linkWidth?: number;
  /**
   * Reads one per-chunk array cell, given the array's relative path (e.g.
   * `"vertices"`, `"vertex_fragments"`, `"links/0/<offsets>"`) and the ABSOLUTE
   * spatial chunk key (e.g. `"-2.-1.-1"`).  Resolves `chunk_grid_origin` and the
   * `sharding_indexed` packing internally and returns the raw `vlen-bytes`
   * container (`readChunkBlob` unwraps element 0), or `undefined` when the chunk
   * is absent (missing shard/cell, or an empty shard-index entry).
   */
  readonly cellRead: CellReader;
}

/** Number of bytes per element of an attribute / link dtype. */
const BYTES_PER_ELEMENT: Record<LinkDtype | AttributeDtype, number> = {
  ...ATTRIBUTE_ELEMENT_BYTES,
};

/**
 * Dtypes a typed-array view can reinterpret in place. The 64-bit attribute
 * dtypes are absent by construction: they are downcast to float32 on decode
 * rather than viewed, and links handle `int64` separately.
 */
type ViewableDtype =
  | "float32"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32";

const ATTRIBUTE_CTORS: Record<
  ViewableDtype,
  | Float32ArrayConstructor
  | Uint8ArrayConstructor
  | Uint16ArrayConstructor
  | Uint32ArrayConstructor
  | Int8ArrayConstructor
  | Int16ArrayConstructor
  | Int32ArrayConstructor
> = {
  float32: Float32Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
};

/**
 * Reinterpret a byte blob as a typed array of the given dtype.  Returns
 * a possibly-aligned view (zero-copy when the source buffer is aligned
 * to the element size) or a copy when alignment forbids the in-place
 * view.  Throws if the byte length is not a multiple of the dtype size.
 */
function reinterpretBytes(
  bytes: Uint8Array,
  dtype: ViewableDtype,
  expectedElements: number,
): AttributeTypedArray {
  const elementSize = BYTES_PER_ELEMENT[dtype];
  const expectedBytes = expectedElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors chunk: dtype=${dtype} expected ${expectedBytes} bytes ` +
        `(${expectedElements} elements), got ${bytes.byteLength}`,
    );
  }
  const Ctor = ATTRIBUTE_CTORS[dtype];
  if (bytes.byteOffset % elementSize === 0) {
    return new (Ctor as any)(bytes.buffer, bytes.byteOffset, expectedElements);
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new (Ctor as any)(copy.buffer, 0, expectedElements);
}

/** Widen any integer dtype's link buffer to chunk-local `Uint32Array`. */
function reinterpretLinkBytes(
  bytes: Uint8Array,
  dtype: LinkDtype,
  numEdges: number,
  linkWidth: number,
): Uint32Array {
  const elementSize = BYTES_PER_ELEMENT[dtype];
  const expectedBytes = numEdges * linkWidth * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors links chunk: dtype=${dtype} link_width=${linkWidth} ` +
        `expected ${expectedBytes} bytes (${numEdges} edges), got ${bytes.byteLength}`,
    );
  }
  const total = numEdges * linkWidth;
  const out = new Uint32Array(total);
  if (dtype === "int64") {
    const aligned =
      bytes.byteOffset % 8 === 0
        ? new BigInt64Array(bytes.buffer, bytes.byteOffset, total)
        : (() => {
            const copy = new Uint8Array(bytes.byteLength);
            copy.set(bytes);
            return new BigInt64Array(copy.buffer, 0, total);
          })();
    for (let i = 0; i < total; ++i) out[i] = Number(aligned[i]);
    return out;
  }
  const arr = reinterpretBytes(bytes, dtype, total);
  // Copy across to Uint32Array — narrow uints widen losslessly,
  // signed types may carry negative values (rejected upstream by writer).
  for (let i = 0; i < total; ++i) out[i] = arr[i];
  return out;
}

/**
 * Download and decode one spatial chunk into a `SkeletonChunk`.
 *
 * Reads (per-chunk arrays under the level group; for a chunk key `i.j.k` the
 * cell is at `<array>/c/i/j/k` — see `readChunkBlob`):
 *   - `vertices`         — required for non-empty chunks.
 *   - `vertex_fragments` — required for non-empty chunks.
 *   - `links/0`          — required unless `linksConvention === "implicit_sequential"`.
 *   - `vertex_attributes/<name>` — one per declared attribute name.
 *
 * Returns `undefined` when the chunk's `vertices/` blob is absent (the
 * canonical "this chunk has no data" signal).  Throws on any partially-
 * present chunk (vertices but missing fragments, or fragments but
 * missing edges in the explicit-edge cases).
 */
/**
 * Read one spatial chunk's blob out of a per-chunk array.
 *
 * zarr-vectors (v0.9.0 single-array format) stores each per-chunk array
 * (`vertices`, `vertex_fragments`, `links/<delta>/<offsets>`, attribute arrays)
 * as a *single* multidimensional `vlen-bytes` zarr v3 array whose shape is the
 * spatial chunk grid, with one vlen element per spatial chunk. The physical
 * location of the cell for an absolute chunk coord — accounting for
 * `chunk_grid_origin` and the optional `sharding_indexed` packing — is resolved
 * by `cellRead` (see `shard_cell_reader.ts`), which returns the raw vlen
 * container; this function unwraps element 0.
 *
 * Returns `undefined` when the cell is absent (sparse chunk presence) or
 * present but empty — an empty vlen chunk is `N=0`, which is how the writer
 * represents "no data here" for a populated grid cell.
 */
async function readChunkBlob(
  cellRead: GeometryChunkDownloadOptions["cellRead"],
  arrayPath: string,
  chunkKey: string,
  signal: AbortSignal,
): Promise<Uint8Array | undefined> {
  const payload = await cellRead(arrayPath, chunkKey, signal);
  if (payload === undefined || payload.byteLength === 0) return undefined;
  return payload;
}

/**
 * Pack a spatial chunk key (`"3.0.2"`, components may be negative) into the
 * high word of a synthesised vertex id, so ids stay distinct across chunks.
 *
 * Each component is offset into `[0, 1024)` and packed into 10 bits, which
 * covers grid coordinates in `[-512, 511]` for ranks up to 3 -- comfortably
 * more than any store's chunk grid. Anything outside that (or rank > 3) falls
 * back to an FNV-1a hash of the key: still deterministic, with a vanishing
 * chance that two chunks share an id space.
 */
export function packChunkKeyToIdWord(chunkKey: string): number {
  const parts = chunkKey.split(".");
  if (parts.length <= 3) {
    let packed = 0;
    let exact = true;
    for (let i = 0; i < parts.length; ++i) {
      const c = Number(parts[i]);
      if (!Number.isInteger(c) || c < -512 || c > 511) {
        exact = false;
        break;
      }
      packed = (packed << 10) | (c + 512);
    }
    if (exact) return packed >>> 0;
  }
  let hash = 0x811c9dc5;
  for (let i = 0; i < chunkKey.length; ++i) {
    hash ^= chunkKey.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/**
 * Resolve the per-vertex id column named by `vertexIdAttribute`, or `undefined`
 * when there is none to use.
 *
 * Decoded from the raw blob rather than from the float32 column the render
 * layer gets, because these values become picking ids and must be exact.
 * That also rules out the dtypes float32 cannot hold exactly: `float32`
 * itself, and the 64-bit dtypes, whose float32 form loses exactness above
 * 2^24 -- a MICrONS `synapse_id` silently truncated is worse than a
 * synthesised id that never claimed to mean anything.
 */
function resolveVertexIdColumn(options: {
  vertexIdAttribute: string | undefined;
  attributeNames: readonly string[];
  attributeDtypes: readonly AttributeDtype[];
  attributeBlobs: readonly (Uint8Array | undefined)[];
  numVertices: number;
}): ArrayLike<number> | undefined {
  const {
    vertexIdAttribute,
    attributeNames,
    attributeDtypes,
    attributeBlobs,
    numVertices,
  } = options;
  if (vertexIdAttribute === undefined) return undefined;
  const i = attributeNames.indexOf(vertexIdAttribute);
  if (i < 0) {
    warnOnceDownload(
      `vertex-id-missing-${vertexIdAttribute}`,
      `zarr-vectors: vertex_id_attribute ${JSON.stringify(vertexIdAttribute)} ` +
        "is not one of this level's vertex attributes; synthesising point ids " +
        "from the chunk key instead.",
    );
    return undefined;
  }
  const dtype = attributeDtypes[i];
  if (!isExactIntDtype(dtype)) {
    warnOnceDownload(
      `vertex-id-inexact-${vertexIdAttribute}`,
      `zarr-vectors: vertex_id_attribute ${JSON.stringify(vertexIdAttribute)} ` +
        `is ${dtype}; ids must be exact integers of at most 32 bits, so ` +
        "synthesising point ids from the chunk key instead.",
    );
    return undefined;
  }
  const bytes = attributeBlobs[i];
  if (bytes === undefined) return undefined;
  return decodeAttributeExactInts(bytes, dtype, numVertices);
}

/**
 * Fill the interleaved `[lo, hi]` segment column with one id per vertex.
 *
 * With a declared id column the id is that column's value, so it means
 * something to the user and is stable across pyramid levels.  Otherwise it is
 * `[vertex index within chunk, packed chunk key]`, which is unique within a
 * level but NOT stable across levels -- coarser levels replace points with
 * bin-centroid metanodes, so there is no vertex for an id to follow.
 */
function fillPerVertexSegmentIds(
  segmentIds: Uint32Array,
  numVertices: number,
  chunkKey: string,
  idColumnOptions: {
    vertexIdAttribute: string | undefined;
    attributeNames: readonly string[];
    attributeDtypes: readonly AttributeDtype[];
    attributeBlobs: readonly (Uint8Array | undefined)[];
    numVertices: number;
  },
): void {
  const idValues = resolveVertexIdColumn(idColumnOptions);
  if (idValues !== undefined) {
    for (let v = 0; v < numVertices; ++v) {
      segmentIds[v * 2] = idValues[v] >>> 0;
      segmentIds[v * 2 + 1] = 0;
    }
    return;
  }
  const chunkWord = packChunkKeyToIdWord(chunkKey);
  for (let v = 0; v < numVertices; ++v) {
    segmentIds[v * 2] = v >>> 0;
    segmentIds[v * 2 + 1] = chunkWord;
  }
}

/** Keys already warned about, so one diagnostic does not repeat per chunk. */
const warnedDownloadKeys = new Set<string>();
function warnOnceDownload(key: string, message: string): void {
  if (warnedDownloadKeys.has(key)) return;
  warnedDownloadKeys.add(key);
  console.warn(message);
}

export async function downloadGeometryChunk(
  options: GeometryChunkDownloadOptions,
  signal: AbortSignal,
): Promise<SkeletonChunk | undefined> {
  const {
    chunkKey,
    rank,
    linkDtype,
    attributeNames,
    attributeDtypes,
    linksConvention,
    geometryKind,
    vertexIdAttribute,
    linkWidth = 2,
    cellRead,
  } = options;
  if (attributeNames.length !== attributeDtypes.length) {
    throw new Error(
      `downloadGeometryChunk: attributeNames (${attributeNames.length}) ` +
        `and attributeDtypes (${attributeDtypes.length}) length mismatch`,
    );
  }

  // Every per-chunk blob this function needs is requested HERE, in one wave.
  //
  // These reads used to be awaited in the order they are consumed -- vertices,
  // then vertex_fragments, then links, then the attribute fan-out -- which put
  // three serial round trips in front of a chunk that needs one. Nothing
  // downstream feeds a later read's ADDRESS: every blob is keyed by the same
  // `chunkKey`, and the only thing the vertex blob contributes is
  // `numVertices`, which is needed to DECODE the attributes, not to ask for
  // them. So the ordering bought nothing and cost 3x the latency on every
  // chunk scrolled into view -- multiplied, for a wide store, by the ~4800
  // chunks a whole-brain point cloud spans.
  //
  // An absent chunk still costs nothing extra: `cellRead` answers from the
  // cached shard index without a request when the cell is empty, and every
  // array of a level shares that grid, so the early return below discards
  // resolved `undefined`s rather than cancelling in-flight fetches.
  const vertexBytesPromise = readChunkBlob(
    cellRead,
    "vertices",
    chunkKey,
    signal,
  );
  const fragmentBytesPromise = readChunkBlob(
    cellRead,
    "vertex_fragments",
    chunkKey,
    signal,
  );
  const intraLinksArray = linksPath(0, intraOffsets(rank, linkWidth));
  const needsExplicitLinks =
    linksConvention === "explicit" ||
    linksConvention === "implicit_sequential_with_branches";
  const linkBytesPromise = needsExplicitLinks
    ? readChunkBlob(cellRead, intraLinksArray, chunkKey, signal)
    : undefined;
  const attributeBlobsPromise = Promise.all(
    attributeNames.map((name) =>
      readChunkBlob(cellRead, `vertex_attributes/${name}`, chunkKey, signal),
    ),
  );
  const segFragBytesPromise: Promise<Uint8Array | undefined> =
    (options.hasFragmentSegmentIds ?? true)
      ? readChunkBlob(
          cellRead,
          "fragment_attributes/segment_id",
          chunkKey,
          signal,
        )
      : Promise.resolve(undefined);
  // Mark the companions handled so an early return (absent chunk) or a throw
  // on an earlier await cannot surface one of them as an unhandled rejection.
  // Attaching a no-op catch creates a NEW derived promise; the originals still
  // reject, and the awaits below still see it.
  for (const p of [
    fragmentBytesPromise,
    linkBytesPromise,
    attributeBlobsPromise,
    segFragBytesPromise,
  ]) {
    p?.catch(() => {});
  }

  // 1. Vertices — required.
  const vertexBytes = await vertexBytesPromise;
  if (vertexBytes === undefined || vertexBytes.byteLength === 0) {
    return undefined;
  }
  const vertexDtype = options.vertexDtype ?? "float32";
  const bytesPerVertex = rank * ATTRIBUTE_ELEMENT_BYTES[vertexDtype];
  if (vertexBytes.byteLength % bytesPerVertex !== 0) {
    throw new Error(
      `zarr-vectors vertices/${chunkKey}: ${vertexBytes.byteLength} bytes ` +
        `not a multiple of ${bytesPerVertex} (rank=${rank} * ${vertexDtype})`,
    );
  }
  const numVertices = vertexBytes.byteLength / bytesPerVertex;
  const positions = decodeAttributeToFloat32(
    vertexBytes,
    vertexDtype,
    numVertices * rank,
  );

  // 2. Fragment index — required.
  const fragmentBytes = await fragmentBytesPromise;
  if (fragmentBytes === undefined) {
    throw new Error(
      `zarr-vectors chunk ${chunkKey} has vertices but vertex_fragments is missing`,
    );
  }
  const fragmentIndex = decodeFragments(fragmentBytes);

  // 3. Explicit intra-chunk edges — required for explicit /
  // implicit_sequential_with_branches, absent for pure implicit_sequential.
  //
  // Under ZVF 0.9 `links/0` is a GROUP and the intra-chunk edges live in its
  // all-zero-offsets array `links/0/0.0.0`; the cross-chunk offset arrays
  // (`links/0/0.0.+1`, …) that bridge chunk boundaries are read separately by
  // the backend's links reader. This intra array's cell is a flat int64 pair
  // list (the `delta==0 && is_intra` "flat" encoding), which reinterpretLinkBytes
  // reads directly.
  let intraLinkRecords: Uint32Array | undefined;
  if (linkBytesPromise !== undefined) {
    const linkBytes = await linkBytesPromise;
    if (linkBytes === undefined || linkBytes.byteLength === 0) {
      // implicit_sequential_with_branches with no explicit branches in
      // this chunk is legitimate (a leaf-only sub-skeleton), and so is a Draco
      // mesh chunk, whose intra-chunk faces live in the bitstream rather than
      // here (spec: geometry_types/mesh.md, "Face storage and Draco").
      intraLinkRecords = new Uint32Array(0);
    } else {
      const elementSize = BYTES_PER_ELEMENT[linkDtype];
      const totalElements = linkBytes.byteLength / elementSize;
      if (totalElements % linkWidth !== 0) {
        throw new Error(
          `zarr-vectors ${intraLinksArray}/${chunkKey}: ${totalElements} ` +
            `elements is not a multiple of link_width=${linkWidth}`,
        );
      }
      intraLinkRecords = reinterpretLinkBytes(
        linkBytes,
        linkDtype,
        totalElements / linkWidth,
        linkWidth,
      );
    }
  }
  // A record of arity 2 is an edge; anything wider bounds a face.
  const isSurface = linkWidth > 2;
  const explicitEdges = isSurface ? undefined : intraLinkRecords;
  const faces = isSurface ? intraLinkRecords : undefined;

  // 4. Per-vertex attributes — one fetch per declared attribute name.
  // A missing per-chunk attribute blob is tolerated and degrades to a
  // zero-filled array of the declared dtype.  Two situations trigger
  // this in practice:
  //
  //   - The writer's pyramid coarsening doesn't propagate
  //     `vertex_attributes/<name>/` to higher levels (see
  //     `zarr-vectors-py multiresolution/coarsen.py`); coarser levels
  //     have vertices but no attribute arrays.
  //   - Future writers may emit attributes sparsely (per-chunk
  //     opt-in).
  //
  // The user-visible effect is `prop_<name>()` evaluating to 0 inside
  // the shader for chunks without that attribute.  This matches how
  // the spatially-indexed skeleton shader handles "this segment has no
  // value" elsewhere and avoids cascading layer failures from a
  // single missing optional blob.
  // Requested in the opening wave alongside the vertices, the fragment index
  // and the links, so a chunk costs ONE round trip however many attribute
  // columns the source exposes -- the fan-out is still one blob per column,
  // and on a MERFISH panel that fan-out is the whole download.
  const numFragments = fragmentIndex.numFragments;
  const [attributeBlobs, segFragBytes] = await Promise.all([
    attributeBlobsPromise,
    segFragBytesPromise,
  ]);
  // Every attribute decodes to float32, whatever it is on disk: that is the
  // one representation the render layer packs into a single texture, and it is
  // what lets an int64 category code or a float64 score be colourable at all.
  // The raw blobs stay in scope just below, for the id column's exact decode.
  const vertexAttributes: AttributeTypedArray[] = attributeBlobs.map(
    (bytes, i) =>
      bytes === undefined
        ? zeroAttribute(numVertices * (options.attributeComponents?.[i] ?? 1))
        : decodeAttributeToFloat32(
            bytes,
            attributeDtypes[i],
            numVertices * (options.attributeComponents?.[i] ?? 1),
          ),
  );

  // 5. Per-fragment segment_id → synthesised per-vertex "segment" column.
  // The writer stores one uint64 flywire id per fragment under
  // `fragment_attributes/segment_id/<key>/c/0`.  We expand it to a
  // per-vertex column of the FULL uint64, stored interleaved as two uint32
  // components `[lo, hi, lo, hi, …]` (a `uvec2` on the GPU).  The render
  // layer hashes the full id with the same `segmentColorHash` as the flat
  // segmentation, so dense fragments colour identically to that segment's
  // voxels, and a picked fragment surfaces the global id.  Colours/ids stay
  // consistent across chunks and pyramid levels because the id is intrinsic
  // to the segment.  When the blob is absent or mis-sized (older stores, or
  // streamline/polyline geometries that never wrote it), fall back to the
  // fragment's index within the chunk: still distinct per fragment, just not
  // unified across chunks (`[f, 0]`).
  let fragSegIds: BigUint64Array | undefined;
  if (
    segFragBytes !== undefined &&
    segFragBytes.byteLength >= numFragments * 8
  ) {
    fragSegIds = new BigUint64Array(
      segFragBytes.buffer.slice(
        segFragBytes.byteOffset,
        segFragBytes.byteOffset + numFragments * 8,
      ),
    );
  }
  // The declared per-vertex id column, decoded for EVERY geometry kind rather
  // than only for point clouds. A point cloud spends it as the vertex's segment
  // id (each point is its own object); everything else needs it as node
  // identity, which is what the editing UI picks by.
  const idColumn = resolveVertexIdColumn({
    vertexIdAttribute,
    attributeNames,
    attributeDtypes,
    attributeBlobs,
    numVertices,
  });
  let nodeIds: Int32Array | undefined;
  if (idColumn !== undefined) {
    nodeIds = new Int32Array(numVertices);
    for (let v = 0; v < numVertices; ++v) nodeIds[v] = idColumn[v];
  }

  // Two uint32 per vertex: [lo, hi].
  const segmentIds = new Uint32Array(numVertices * 2);
  if (!KIND_CAPABILITIES[geometryKind].hasObjectModel) {
    // No discrete-object model (point_cloud): the "segment" a vertex belongs to
    // is the vertex itself, so picking selects one point rather than a whole
    // spatial bin.  A fragment here IS a bin holding many unrelated points, so
    // the per-fragment path below would lump them together.
    fillPerVertexSegmentIds(segmentIds, numVertices, chunkKey, {
      vertexIdAttribute,
      attributeNames,
      attributeDtypes,
      attributeBlobs,
      numVertices,
    });
  } else {
    for (let f = 0; f < numFragments; ++f) {
      let lo: number;
      let hi: number;
      if (fragSegIds !== undefined) {
        const id = fragSegIds[f];
        lo = Number(id & 0xffffffffn) >>> 0;
        hi = Number((id >> 32n) & 0xffffffffn) >>> 0;
      } else {
        lo = f;
        hi = 0;
      }
      const idxs = fragmentIndex.indices(f);
      for (let k = 0; k < idxs.length; ++k) {
        const v = idxs[k];
        segmentIds[v * 2] = lo;
        segmentIds[v * 2 + 1] = hi;
      }
    }
  }

  return buildGeometryChunk({
    rank,
    positions,
    fragmentIndex,
    explicitEdges,
    linksConvention,
    geometryKind,
    vertexAttributes,
    segmentIds,
    // Global only when real per-fragment ids were read AND this kind has a
    // discrete-object model; the point-cloud path above assigns one id per
    // VERTEX, which is an identity for a point, not for an object.
    segmentIdsAreGlobal:
      fragSegIds !== undefined &&
      KIND_CAPABILITIES[geometryKind].hasObjectModel,
    nodeIds,
    faces,
    faceArity: linkWidth,
    linkedSkeletonLayout: options.linkedSkeletonLayout,
    relinkedChildren: options.relinkedChildren,
  });
}
