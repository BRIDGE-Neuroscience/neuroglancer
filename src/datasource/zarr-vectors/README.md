# zarr-vectors datasource

Reads [Zarr Vectors](https://github.com/AllenInstitute/zarr_vectors) stores
(format 0.9.x, as written by zarr-vectors-py and zarr-vectors-tools) into a
segmentation layer: point clouds, lines, streamlines, polylines, skeletons,
graphs and meshes.

```
<store url>/|zarr-vectors:                     all subsources, default attributes
<store url>/|zarr-vectors:#attributes=fa,z      choose the vertex attributes
zarr-vectors://<store url>                     older form, still accepted
```

## Merging into another Neuroglancer branch

The handler is self-contained: it modifies no Neuroglancer file and uses only
APIs present in both google/neuroglancer `master` and the MetaCell-based
BRIDGE branches.

1. Copy this directory to `src/datasource/zarr-vectors/` (and, for the tests,
   `testdata/datasource/zarr-vectors/`).
2. Run `npm run update-conditions`. It regenerates
   `src/datasource/enabled_{frontend,backend,async_computation}_modules.ts`
   and the `imports` entries of `package.json` (21 generated lines).

If the build breaks after a merge, look in `compat.ts` first: it holds every
call whose signature differs between the supported Neuroglancer versions.

Verified 2026-10-03 against google/neuroglancer `master` 60c866ee and
Andrew-Keenlyside/neuroglancer `main` 69ed3ef7: typecheck, lint, tests, build
and real-browser rendering, pixel-identical between the two.

## What it registers

- The `zarr-vectors` kvstore-based data source (`register_default.ts`).
- A subclass of `SegmentationUserLayer` under the existing `"segmentation"`
  type name (`layer.ts`). It draws the dense subsource below and passes every
  other subsource to the base class unchanged, so saved states keep their
  layer type and non-zarr-vectors data behaves exactly as before.
- Worker-side zstd and blosc decoders (`async_computation.ts`).

## Modules

Main thread (`frontend.ts` and what it imports) and chunk worker
(`backend.ts`) load different halves; `worker_imports.spec.ts` keeps WebGL and
UI code out of the worker.

| module                                        | runs in | does                                                              |
| --------------------------------------------- | ------- | ----------------------------------------------------------------- |
| `register_default.ts`, `frontend.ts`          | main    | URL parsing, store opening, subsource list                        |
| `layer.ts`                                    | main    | segmentation layer subclass that draws the dense subsource        |
| `dense_frontend.ts`                           | main    | dense render layer: shaders, segment state, picking               |
| `compat.ts`                                   | main    | calls that differ between Neuroglancer versions                   |
| `backend.ts`                                  | worker  | chunk sources: dense chunks, object skeletons, mesh fragments     |
| `chunk_pipeline.ts`                           | worker  | one dense chunk: decode, segment ids, bridges to neighbour chunks |
| `object_reader.ts`                            | worker  | one object, assembled from its manifest's chunks                  |
| `mesh_lod.ts`                                 | both    | mesh levels of detail: level choice, octree, octant split         |
| `dense_lod.ts`, `base.ts`, `geometry_kind.ts` | both    | level choice, shared parameters, per-kind drawing rules           |
| `store.ts`, `objects.ts`                      | both    | root/level metadata; object table and segment properties          |
| `chunk_decode.ts`                             | worker  | a chunk's vertices, edges, faces and tangents                     |
| `links.ts`                                    | both    | cross-chunk links                                                 |
| `zarr_array.ts`                               | both    | zarr v3 arrays: codecs, sharding, byte-range rows                 |
| `level_cells.ts`                              | worker  | a level's arrays, opened lazily                                   |
| `fragment_index.ts`, `object_manifest.ts`     | worker  | binary fragment and manifest encodings                            |
| `dtype.ts`, `util.ts`                         | both    | element decoding; LRU, concurrency, warn-once                     |

## Subsources

| id           | what                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------ |
| `""`         | Dense overview of every object, from spatial chunks (`dense_frontend.ts`)                  |
| `objects`    | Selected objects at full resolution, as Neuroglancer skeletons (curves, skeletons, graphs) |
| `meshes`     | Selected objects as Neuroglancer meshes (mesh stores)                                      |
| `properties` | Object attributes as numeric segment properties; groups as tags and a label                |

The dense layer follows the layer's ordinary segment state: selected segments,
segment colours, colour seed, `selectedAlpha` (2-d) / `objectAlpha` (3-d),
`notSelectedAlpha` for everything else, and `ignoreNullVisibleSet` (an empty
selection shows every object). Selected objects are left to `objects` /
`meshes` while those subsources are enabled. The layer's skeleton shader and
controls apply to both, with `prop_<attribute>()`, `prop_tangent()` and
`segmentColor()`.

One pyramid level is drawn per view: the finest whose vertices, for what the
view would load, fit one vertex per `renderScale`² pixels (`dense_lod.ts`).
The coarsest level stands in while it loads.

## What a store needs

To open, a store needs only what zarr-vectors-py 0.9.x writes for any
geometry: a 3-D root with `zarr_vectors` metadata, and level 0's `vertices`
and `vertex_fragments`. Everything else is optional. The viewer detects each
feature per store (and, where it can vary, per chunk), uses it when present,
and otherwise does what the right-hand column says. Most features change only
cost; an object index and directory listing also change what can be shown.

| Feature                                                              | Gives                                                 | Without it                                                  |
| -------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| `object_index` (`vlen_manifests_v1`/`v2`)                            | `objects` / `meshes` / `properties` subsources        | dense overview only                                         |
| more pyramid levels                                                  | dense overview level choice (`dense_lod.ts`)          | level 0 everywhere                                          |
| mesh levels whose chunks double and that store faces                 | multi-resolution meshes (`mesh_lod.ts`)               | meshes at level 0                                           |
| `"link_groups": "per_vertex_fragment"` on a level's intra face array | one object's faces read alone, by byte range          | whole face cells (`zvtools index-faces` adds it)            |
| uncompressed cells, one per stored chunk (sharded or not)            | byte-range reads of rows (bridges, faces, neighbours) | whole cells                                                 |
| `sharding_indexed`                                                   | ranges of one shard read together are merged          | one request per cell                                        |
| server directory listing                                             | cross-chunk links found by listing `links/0/`         | edges: the 26 neighbours are probed; faces: missing, warned |
| `nonempty_chunks`                                                    | empty cells skipped without a request                 | a request that finds nothing                                |
| zarr-vectors-tools `skeleton_layout`                                 | skeleton layout known                                 | inferred                                                    |
| `fragment_attributes/segment_id`                                     | segment ids read directly                             | taken from the level's manifests                            |

A feature is trusted only where the data bears it out: a declared face group
whose rows index another fragment, or whose count does not match the
fragments, sends that chunk back to the whole-cell read with a console
warning.

## Format support

Read through each array's own `zarr.json` (`zarr_array.ts`):

- Codecs: none, zstd, blosc, gzip, zlib, crc32c; `sharding_indexed` with the
  declared index codecs and location. Codecs and sharding may differ per
  array, as `zvtools attach` and `build_pyramid` produce.
- Byte-range reads of one key issued together (the cells of a shard that
  several chunks need) are merged into one request when they lie within 8 KiB
  of each other (`coalesceRangeReads`).
- `chunk_grid_origin` (negative chunk coordinates), `nonempty_chunks`.
- Positions in any float or integer dtype; vertex attributes of any numeric
  dtype with 1-4 components (the width is measured when `row_shape` is
  missing); dictionary-encoded attributes.
- Object index layouts `vlen_manifests_v1` and `vlen_manifests_v2`
  (`object_ids`), any number of chunks; `object_attributes/segment_id` as the
  segment id; overlapping groups.
- Cross-chunk links of either direction, int32 or int64, with `perm_idx`.
- Skeletons as zarr-vectors-py's `read_graph` reads them: a vertex's parent is
  the previous row of its fragment unless a `[child, parent]` link (in the
  chunk or across chunks) names another. zarr-vectors-tools'
  `skeleton_layout` marker (`linked_across_chunks`, `split_at_chunk_faces`)
  is honoured; without it, the layout is inferred.

Meshes: the `meshes` subsource is a Neuroglancer multiscale mesh
(`mesh_lod.ts`). Level 0 and each following pyramid level whose chunks are
twice the previous level's and which stores faces is a level of detail, as
zarr-vectors-tools builds with `zvtools pyramid --method mesh_decimate
--coarsen 2,2,2 --chunk-scale 2,2,2`; other stores use level 0 alone. An
octree node is one chunk of a level, and its fragment is the object's faces
stored in that chunk, including faces that reach into neighbours. Chunks hold
every object's faces; where a level declares one face group per vertex
fragment, an object's faces and vertices are read alone, by byte range,
otherwise the whole cell is read and filtered. The dense overview draws a mesh
store's vertices and reads no faces.

Not read yet (reported once in the console):

- Draco-encoded vertices (the level is skipped). zarr-vectors-py also reorders
  vertices when it writes Draco, so its cross-chunk faces would be wrong.
- 0.9.4 dense manifests (`dense_manifests` capability): the store opens, but
  without `objects` / `meshes` / `properties`.
- Attribute-chunked levels (`chunk_attribute_values`).
- Rank other than 3.

## Tests

- `npx vitest --run --project node src/datasource/zarr-vectors`: unit tests
  and end-to-end reads of stores written by zarr-vectors-py v0.9.2
  (`testdata/datasource/zarr-vectors/generate.py`), including exact
  reconstruction of every polyline under each codec and layout, and of a
  branching skeleton that leaves a chunk and returns.
- `npx vitest --run --project browser src/datasource/zarr-vectors`: blosc in a
  real browser (its WASM decoder rejects vitest's node worker polyfill).
