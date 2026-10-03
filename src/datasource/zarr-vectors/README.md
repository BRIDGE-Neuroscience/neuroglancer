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

## Format support

Read through each array's own `zarr.json` (`zarr_array.ts`):

- Codecs: none, zstd, blosc, gzip, zlib, crc32c; `sharding_indexed` with the
  declared index codecs and location. Codecs and sharding may differ per
  array, as `zvtools attach` and `build_pyramid` produce.
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
- Fragments without a stored `segment_id` get their object from the level's
  manifests.

Meshes: the `meshes` subsource reads level 0 only, through Neuroglancer's
single-resolution mesh source, one fragment per chunk an object occupies. The
dense overview draws a mesh store's vertices at every level and reads no
faces, even where a zarr-vectors-tools pyramid (`zvtools pyramid`) stores
decimated faces above level 0. Faces that span chunks are found by listing
`links/0/`; on a server without listing they are missing (with a warning).

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
