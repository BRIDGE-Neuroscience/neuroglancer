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

Verified 2026-10-02 against upstream `master` 60c866ee: typecheck, build and
real-browser rendering are unchanged from this branch.

## What it registers

- The `zarr-vectors` kvstore-based data source (`register_default.ts`).
- A subclass of `SegmentationUserLayer` under the existing `"segmentation"`
  type name (`layer.ts`). It draws the dense subsource below and passes every
  other subsource to the base class unchanged, so saved states keep their
  layer type and non-zarr-vectors data behaves exactly as before.
- Worker-side zstd and blosc decoders (`async_computation.ts`).

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
- Cross-chunk links of either direction, int32 or int64, with `perm_idx`; the
  zarr-vectors-tools "linked" SWC layout.
- Fragments without a stored `segment_id` get their object from the level's
  manifests.

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
  reconstruction of every polyline under each codec and layout.
- `npx vitest --run --project browser src/datasource/zarr-vectors`: blosc in a
  real browser (its WASM decoder rejects vitest's node worker polyfill).
