/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  FRAGMENT_INDEX_MAGIC,
  FRAGMENT_INDEX_VERSION,
} from "#src/datasource/zarr-vectors/fragment_index.js";
import {
  downloadGeometryChunk,
  packChunkKeyToIdWord,
  type LinkDtype,
} from "#src/datasource/zarr-vectors/geometry_chunk_download.js";
import type { CellReader } from "#src/datasource/zarr-vectors/level_cells.js";

/** Build a single-range-fragment ZVFG blob covering all `numVertices` rows. */
function singleRangeFragmentBlob(numVertices: number): Uint8Array {
  // Layout: 16-byte header, 8-byte bitmap (1 bit set), 16-byte range entry,
  // 4-byte CSR offsets section (one uint32 zero — E+1 with E=0).
  const buf = new ArrayBuffer(16 + 8 + 16 + 4);
  const view = new DataView(buf);
  view.setUint32(0, FRAGMENT_INDEX_MAGIC, true);
  view.setUint16(4, FRAGMENT_INDEX_VERSION, true);
  view.setUint16(6, 0, true);
  view.setUint32(8, 1, true); // F = 1
  view.setUint32(12, 1, true); // R = 1
  new Uint8Array(buf)[16] = 0x01; // bitmap bit 0 set
  view.setBigInt64(24, 0n, true); // start
  view.setBigInt64(32, BigInt(numVertices), true); // count
  view.setUint32(40, 0, true); // csr_offsets[0] = 0 (E=0)
  return new Uint8Array(buf);
}

/** Pack float32 positions into a vertices blob (rank-3). */
function verticesBlob(positions: number[]): Uint8Array {
  const arr = new Float32Array(positions);
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

/** Pack uint16 edges into a links blob. */
function uint16LinksBlob(edges: number[]): Uint8Array {
  const arr = new Uint16Array(edges);
  return new Uint8Array(arr.buffer);
}

/**
 * Build a ZVFG blob with two contiguous range fragments:
 * fragment 0 = `[0, count0)`, fragment 1 = `[count0, count0 + count1)`.
 */
function twoRangeFragmentsBlob(count0: number, count1: number): Uint8Array {
  // 16 header + 8 bitmap + 2*16 range table + 4 CSR offsets (E=0).
  const buf = new ArrayBuffer(16 + 8 + 32 + 4);
  const view = new DataView(buf);
  view.setUint32(0, FRAGMENT_INDEX_MAGIC, true);
  view.setUint16(4, FRAGMENT_INDEX_VERSION, true);
  view.setUint16(6, 0, true);
  view.setUint32(8, 2, true); // F = 2
  view.setUint32(12, 2, true); // R = 2
  new Uint8Array(buf)[16] = 0x03; // bitmap bits 0 and 1 set
  view.setBigInt64(24, 0n, true); // frag 0 start
  view.setBigInt64(32, BigInt(count0), true); // frag 0 count
  view.setBigInt64(40, BigInt(count0), true); // frag 1 start
  view.setBigInt64(48, BigInt(count1), true); // frag 1 count
  view.setUint32(56, 0, true); // csr_offsets[0] = 0 (E=0)
  return new Uint8Array(buf);
}

/** Pack uint64 per-fragment segment ids into a fragment-attribute blob. */
function uint64SegmentIdBlob(ids: bigint[]): Uint8Array {
  const arr = new BigUint64Array(ids);
  return new Uint8Array(arr.buffer);
}

/** A trailing `"3.0.2"`-style segment marks a per-spatial-chunk array key. */
const CHUNK_KEY_SUFFIX = /\/(\d+(?:\.\d+)*)$/;

/**
 * An already-encoded zarr chunk path (`.../c/0`). These are not per-chunk
 * array keys — `object_index/manifests/c/0` is chunk 0 of the 1-D manifests
 * array — but their trailing `/0` would otherwise look like a chunk key.
 */
const ENCODED_CHUNK_PATH = /\/c\/\d+(?:\/\d+)*$/;

/**
 * Build a kvStore stub from a logical key → blob map (missing key →
 * undefined).
 *
 * Per-chunk arrays are addressed logically as `"<array>/<chunkKey>"`, e.g.
 * `"vertices/0.0.0"`. Those are translated to the on-disk single-vlen-array
 * layout (`vertices/c/0/0/0`) and the blob is wrapped in the vlen framing, so
 * the tests state intent rather than restating the encoding. Any other key is
 * passed through verbatim, which keeps already-correct paths such as
 * `object_index/manifests/c/0` (a 1-D vlen array read by the manifest reader,
 * not a per-chunk array) working unchanged.
 */
function makeKvStore(map: Record<string, Uint8Array | undefined>): CellReader {
  const encoded = new Map<string, Uint8Array | undefined>();
  for (const [key, blob] of Object.entries(map)) {
    const m = ENCODED_CHUNK_PATH.test(key) ? null : key.match(CHUNK_KEY_SUFFIX);
    if (m === null) {
      encoded.set(key, blob);
      continue;
    }
    const arrayPath = key.slice(0, m.index);
    const cellPath = `${arrayPath}/c/${m[1].split(".").join("/")}`;
    encoded.set(cellPath, blob);
  }
  // `cellRead` contract: (arrayPath, chunkKey) -> the cell's payload. Mirrors
  // the real reader, which resolves origin, sharding and codecs itself.
  return async (arrayPath: string, chunkKey: string) =>
    encoded.get(`${arrayPath}/c/${chunkKey.split(".").join("/")}`);
}

describe("downloadGeometryChunk — orchestrator", () => {
  it("returns undefined when the vertices blob is absent", async () => {
    const cellRead = makeKvStore({});
    const result = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "implicit_sequential",
        geometryKind: "streamline",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(result).toBeUndefined();
  });

  it("returns undefined when the vertices blob is zero bytes", async () => {
    const cellRead = makeKvStore({
      "vertices/0.0.0": new Uint8Array(0),
    });
    const result = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "implicit_sequential",
        geometryKind: "streamline",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(result).toBeUndefined();
  });

  it("downloads a streamline chunk and returns positions + edges + tangents", async () => {
    // 3 vertices marching +X by 1 each step → tangents all (1, 0, 0).
    const positions = [0, 0, 0, 1, 0, 0, 2, 0, 0];
    const cellRead = makeKvStore({
      "vertices/1.2.3": verticesBlob(positions),
      "vertex_fragments/1.2.3": singleRangeFragmentBlob(3),
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "1.2.3",
        rank: 3,
        linkDtype: "int64",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "implicit_sequential",
        geometryKind: "streamline",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk).toBeDefined();
    expect(chunk!.numVertices).toBe(3);
    expect(chunk!.numEdges).toBe(2);
    expect(Array.from(chunk!.edges)).toEqual([0, 1, 1, 2]);
    // tangents all (1, 0, 0)
    expect(chunk!.tangents).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(chunk!.tangents![i * 3]).toBeCloseTo(1, 6);
      expect(chunk!.tangents![i * 3 + 1]).toBeCloseTo(0, 6);
      expect(chunk!.tangents![i * 3 + 2]).toBeCloseTo(0, 6);
    }
  });

  it("downloads a polyline chunk with per-vertex attributes", async () => {
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(2),
      "vertex_attributes/radius/0.0.0": new Uint8Array(
        new Float32Array([0.5, 0.7]).buffer,
      ),
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: ["radius"],
        attributeDtypes: ["float32"],
        linksConvention: "implicit_sequential",
        geometryKind: "polyline",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk!.vertexAttributes.length).toBe(1);
    const radius = chunk!.vertexAttributes[0] as Float32Array;
    expect(radius.length).toBe(2);
    expect(radius[0]).toBeCloseTo(0.5);
    expect(radius[1]).toBeCloseTo(0.7);
  });

  it("issues every per-chunk read in ONE round trip", async () => {
    // A chunk of a wide store is dominated by its attribute fan-out, and the
    // reads used to be awaited in consumption order -- vertices, then the
    // fragment index, then the attributes -- so a chunk cost three serial round
    // trips. Nothing downstream feeds a later read's address, so the fan-out
    // belongs in one wave; this pins that, because the regression is invisible
    // (correct data, three times the latency, on every chunk).
    const base = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(2),
      "vertex_attributes/a/0.0.0": new Uint8Array(
        new Float32Array([0.5, 0.7]).buffer,
      ),
      "vertex_attributes/b/0.0.0": new Uint8Array(
        new Float32Array([1.5, 1.7]).buffer,
      ),
    });
    const issued: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const cellRead: CellReader = async (arrayPath, chunkKey, signal) => {
      issued.push(arrayPath);
      // Nothing resolves until the test says so, so `issued` can only grow via
      // a read the orchestrator started WITHOUT waiting for an earlier one.
      await gate;
      return base(arrayPath, chunkKey, signal);
    };
    const pending = downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: ["a", "b"],
        attributeDtypes: ["float32", "float32"],
        linksConvention: "implicit_sequential",
        geometryKind: "point_cloud",
        hasFragmentSegmentIds: false,
        cellRead,
      },
      new AbortController().signal,
    );
    // The fan-out is synchronous, ahead of the first await, so every read is
    // already in flight by the time the call returns its promise.
    expect([...issued].sort()).toEqual([
      "vertex_attributes/a",
      "vertex_attributes/b",
      "vertex_fragments",
      "vertices",
    ]);
    release();
    const chunk = await pending;
    expect(chunk!.numVertices).toBe(2);
    expect((chunk!.vertexAttributes[1] as Float32Array)[0]).toBeCloseTo(1.5);
  });

  it("reports an absent chunk without waiting on its companion reads", async () => {
    // The early return discards resolved `undefined`s rather than leaving the
    // speculative reads dangling; an unhandled rejection here would surface as
    // a process-level warning rather than a test failure, so assert the shape.
    const cellRead: CellReader = async (arrayPath) => {
      if (arrayPath === "vertices") return undefined;
      throw new Error(`companion read failed: ${arrayPath}`);
    };
    const result = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: ["a"],
        attributeDtypes: ["float32"],
        linksConvention: "implicit_sequential",
        geometryKind: "point_cloud",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(result).toBeUndefined();
  });

  it("gives a point-cloud chunk one segment id per vertex, distinct across chunks", async () => {
    // A point cloud's fragment is a spatial bin, not an object, so every vertex
    // has to get its own id or picking would select a whole bin at a time.
    const cellsFor = (key: string) =>
      makeKvStore({
        [`vertices/${key}`]: verticesBlob([0, 0, 0, 1, 0, 0, 2, 0, 0]),
        [`vertex_fragments/${key}`]: singleRangeFragmentBlob(3),
      });
    const read = async (key: string) =>
      (await downloadGeometryChunk(
        {
          chunkKey: key,
          rank: 3,
          linkDtype: "int64",
          attributeNames: [],
          attributeDtypes: [],
          linksConvention: "implicit_sequential",
          geometryKind: "point_cloud",
          cellRead: cellsFor(key),
        },
        new AbortController().signal,
      ))!;

    const a = await read("0.0.0");
    expect(a.numEdges).toBe(0);
    // Low word is the vertex's index within the chunk.
    expect(Array.from(a.segmentIds!.slice(0, 6))).toEqual([
      0,
      packChunkKeyToIdWord("0.0.0"),
      1,
      packChunkKeyToIdWord("0.0.0"),
      2,
      packChunkKeyToIdWord("0.0.0"),
    ]);

    // Same low words in another chunk, but a different high word, so no two
    // points anywhere in the level share an id.
    const b = await read("1.0.2");
    expect(b.segmentIds![1]).not.toBe(a.segmentIds![1]);
    expect(b.segmentIds![0]).toBe(0);
  });

  it("prefers a declared per-vertex id attribute over synthesised point ids", async () => {
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(2),
      "vertex_attributes/cell_label/0.0.0": new Uint8Array(
        new Uint32Array([4242, 99]).buffer,
      ),
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: ["cell_label"],
        attributeDtypes: ["uint32"],
        linksConvention: "implicit_sequential",
        geometryKind: "point_cloud",
        vertexIdAttribute: "cell_label",
        cellRead,
      },
      new AbortController().signal,
    );
    // Ids come straight from the column, so they mean something to the user and
    // stay stable across pyramid levels.
    expect(Array.from(chunk!.segmentIds!)).toEqual([4242, 0, 99, 0]);
  });

  it("falls back to synthesised ids when the declared id attribute is unusable", async () => {
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(2),
      "vertex_attributes/fa/0.0.0": new Uint8Array(
        new Float32Array([0.5, 0.7]).buffer,
      ),
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: ["fa"],
        attributeDtypes: ["float32"],
        linksConvention: "implicit_sequential",
        geometryKind: "point_cloud",
        // float32 cannot hold an exact id past 2^24, so it is refused.
        vertexIdAttribute: "fa",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(Array.from(chunk!.segmentIds!)).toEqual([
      0,
      packChunkKeyToIdWord("0.0.0"),
      1,
      packChunkKeyToIdWord("0.0.0"),
    ]);
  });

  it("downloads a skeleton chunk with implicit_sequential_with_branches and uint16 link dtype", async () => {
    // 5 vertices, single fragment.  Sequential edges plus a branch (1,4)
    // stored in the intra-chunk links array links/0/0.0.0 as uint16.
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([
        0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0,
      ]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(5),
      "links/0/0.0.0/0.0.0": uint16LinksBlob([1, 4]),
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "uint16",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "implicit_sequential_with_branches",
        geometryKind: "skeleton",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk!.numEdges).toBe(5); // 4 sequential + 1 branch
    expect(Array.from(chunk!.edges)).toEqual([0, 1, 1, 2, 2, 3, 3, 4, 1, 4]);
    // Skeletons synthesise edge-adjacency tangents (prop_tangent()).
    expect(chunk!.tangents).toBeDefined();
    expect(chunk!.tangents!.length).toBe(5 * 3);
  });

  it("reads a mesh chunk's faces from the wider links family", async () => {
    // link_width=3, so the intra-chunk array is named by TWO all-zero offsets
    // (one per non-source endpoint) and its records are triangles, not edges.
    const faceBlob = new Uint8Array(new Int32Array([0, 1, 2, 1, 3, 2]).buffer);
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(4),
      "links/0/0.0.0_0.0.0/0.0.0": faceBlob,
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int32",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "explicit",
        geometryKind: "mesh",
        linkWidth: 3,
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk!.numFaces).toBe(2);
    expect(Array.from(chunk!.faces!)).toEqual([0, 1, 2, 1, 3, 2]);
    expect(chunk!.numEdges).toBe(0);
  });

  it("downloads an explicit-only graph chunk with int32 link dtype", async () => {
    const linksBlob = new Uint8Array(new Int32Array([0, 1, 2, 3]).buffer);
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(4),
      "links/0/0.0.0/0.0.0": linksBlob,
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int32",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "explicit",
        geometryKind: "skeleton",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk!.numEdges).toBe(2);
    expect(Array.from(chunk!.edges)).toEqual([0, 1, 2, 3]);
  });

  it("handles a skeleton chunk with no branches (empty links blob)", async () => {
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(2),
      "links/0/0.0.0/0.0.0": new Uint8Array(0),
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "uint16",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "implicit_sequential_with_branches",
        geometryKind: "skeleton",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk!.numEdges).toBe(1); // just the (0,1) sequential edge
    expect(Array.from(chunk!.edges)).toEqual([0, 1]);
  });

  it("rejects when vertex_fragments is missing", async () => {
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0]),
    });
    await expect(
      downloadGeometryChunk(
        {
          chunkKey: "0.0.0",
          rank: 3,
          linkDtype: "int64",
          attributeNames: [],
          attributeDtypes: [],
          linksConvention: "implicit_sequential",
          geometryKind: "streamline",
          cellRead,
        },
        new AbortController().signal,
      ),
    ).rejects.toThrow(/vertex_fragments is missing/);
  });

  it("zero-fills declared attributes when the per-chunk blob is missing", async () => {
    // Used by the pyramid case: coarser levels often have vertices but
    // no `vertex_attributes/<name>/` arrays at all (the writer's
    // metavertex aggregation skips attribute propagation).  Rather
    // than fail the whole chunk, the reader degrades each missing
    // attribute to zero-filled.
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(2),
      // intentionally missing vertex_attributes/radius/...
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "int64",
        attributeNames: ["radius"],
        attributeDtypes: ["float32"],
        linksConvention: "implicit_sequential",
        geometryKind: "polyline",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk).toBeDefined();
    expect(chunk!.vertexAttributes).toHaveLength(1);
    expect(chunk!.vertexAttributes[0]).toBeInstanceOf(Float32Array);
    expect(chunk!.vertexAttributes[0].length).toBe(2); // numVertices
    expect(Array.from(chunk!.vertexAttributes[0] as Float32Array)).toEqual([
      0, 0,
    ]);
  });

  it("widens narrow-int link dtypes to chunk-local Uint32Array", async () => {
    // uint8 link dtype — the widening should preserve indices losslessly.
    const linksBlob = new Uint8Array([0, 1, 1, 2]); // edges (0,1) and (1,2)
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0, 2, 0, 0]),
      "vertex_fragments/0.0.0": singleRangeFragmentBlob(3),
      "links/0/0.0.0/0.0.0": linksBlob,
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "uint8",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "explicit",
        geometryKind: "skeleton",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk!.edges.constructor).toBe(Uint32Array);
    expect(Array.from(chunk!.edges)).toEqual([0, 1, 1, 2]);
  });

  it("rejects attributeNames / attributeDtypes length mismatch", async () => {
    await expect(
      downloadGeometryChunk(
        {
          chunkKey: "0.0.0",
          rank: 3,
          linkDtype: "int64",
          attributeNames: ["radius", "swc_type"],
          attributeDtypes: ["float32"], // mismatched length
          linksConvention: "implicit_sequential",
          geometryKind: "polyline",
          cellRead: makeKvStore({}),
        },
        new AbortController().signal,
      ),
    ).rejects.toThrow(/length mismatch/i);
  });

  /** Reconstruct the per-vertex uint64 ids from the interleaved [lo,hi] column. */
  function segmentIdsAsBigint(segmentIds: Uint32Array): bigint[] {
    const out: bigint[] = [];
    for (let v = 0; v * 2 + 1 < segmentIds.length; ++v) {
      out.push(
        BigInt(segmentIds[v * 2] >>> 0) |
          (BigInt(segmentIds[v * 2 + 1] >>> 0) << 32n),
      );
    }
    return out;
  }

  it("synthesises a per-vertex FULL uint64 segment column from fragment_attributes/segment_id", async () => {
    // Two fragments: frag 0 owns vertices 0..1, frag 1 owns vertices 2..4.
    // Flywire-scale ids (> 2^32) must survive intact — the high 32 bits are
    // what made the old uint32-truncated colour differ from the flat
    // segmentation.
    const id0 = 720575940612786691n;
    const id1 = 720575940606327461n;
    expect(id0 > 0xffffffffn && id1 > 0xffffffffn).toBe(true); // genuinely uint64
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([
        0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0,
      ]),
      "vertex_fragments/0.0.0": twoRangeFragmentsBlob(2, 3),
      "fragment_attributes/segment_id/0.0.0": uint64SegmentIdBlob([id0, id1]),
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "uint16",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "implicit_sequential_with_branches",
        geometryKind: "skeleton",
        cellRead,
      },
      new AbortController().signal,
    );
    expect(chunk!.segmentIds).toBeDefined();
    expect(chunk!.segmentIds!.length).toBe(5 * 2); // interleaved [lo, hi]
    expect(segmentIdsAsBigint(chunk!.segmentIds!)).toEqual([
      id0,
      id0,
      id1,
      id1,
      id1,
    ]);
  });

  it("falls back to the fragment's chunk-local index when segment_id is absent", async () => {
    const cellRead = makeKvStore({
      "vertices/0.0.0": verticesBlob([
        0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0,
      ]),
      "vertex_fragments/0.0.0": twoRangeFragmentsBlob(2, 3),
      // no fragment_attributes/segment_id blob
    });
    const chunk = await downloadGeometryChunk(
      {
        chunkKey: "0.0.0",
        rank: 3,
        linkDtype: "uint16",
        attributeNames: [],
        attributeDtypes: [],
        linksConvention: "implicit_sequential_with_branches",
        geometryKind: "skeleton",
        cellRead,
      },
      new AbortController().signal,
    );
    // Fragment 0 → id 0 for its 2 vertices; fragment 1 → id 1 for its 3.
    expect(segmentIdsAsBigint(chunk!.segmentIds!)).toEqual([
      0n,
      0n,
      1n,
      1n,
      1n,
    ]);
  });
});

describe("downloadGeometryChunk — link dtype matrix", () => {
  // Round-trip parity check: edges (0,1) and (1,2) written in each dtype
  // should produce identical chunk-local Uint32Array output.
  const cases: Array<{ dtype: LinkDtype; blob: () => Uint8Array }> = [
    { dtype: "uint8", blob: () => new Uint8Array([0, 1, 1, 2]) },
    {
      dtype: "uint16",
      blob: () => new Uint8Array(new Uint16Array([0, 1, 1, 2]).buffer),
    },
    {
      dtype: "uint32",
      blob: () => new Uint8Array(new Uint32Array([0, 1, 1, 2]).buffer),
    },
    {
      dtype: "int8",
      blob: () => new Uint8Array(new Int8Array([0, 1, 1, 2]).buffer),
    },
    {
      dtype: "int16",
      blob: () => new Uint8Array(new Int16Array([0, 1, 1, 2]).buffer),
    },
    {
      dtype: "int32",
      blob: () => new Uint8Array(new Int32Array([0, 1, 1, 2]).buffer),
    },
    {
      dtype: "int64",
      blob: () => new Uint8Array(new BigInt64Array([0n, 1n, 1n, 2n]).buffer),
    },
  ];
  for (const { dtype, blob } of cases) {
    it(`reads edges with linkDtype=${dtype}`, async () => {
      const cellRead = makeKvStore({
        "vertices/0.0.0": verticesBlob([0, 0, 0, 1, 0, 0, 2, 0, 0]),
        "vertex_fragments/0.0.0": singleRangeFragmentBlob(3),
        "links/0/0.0.0/0.0.0": blob(),
      });
      const chunk = await downloadGeometryChunk(
        {
          chunkKey: "0.0.0",
          rank: 3,
          linkDtype: dtype,
          attributeNames: [],
          attributeDtypes: [],
          linksConvention: "explicit",
          geometryKind: "skeleton",
          cellRead,
        },
        new AbortController().signal,
      );
      expect(Array.from(chunk!.edges)).toEqual([0, 1, 1, 2]);
    });
  }
});

// ---------------------------------------------------------------------------
// fetchGhostVertices
// ---------------------------------------------------------------------------
