# ngpy — a standalone Pyodide GUI around any Neuroglancer build

`ngpy.html` is ONE self-contained file (~1 MB) that hosts an unmodified
Neuroglancer build in an iframe and wraps it with its own GUI and its own Python
(Pyodide) runtime:

- **Python** tab: run scripts against the viewer with the familiar
  `neuroglancer` Python API (`viewer = neuroglancer.Viewer()`,
  `with viewer.txn() as s: …`, `viewer.actions`, key bindings).
- **Filter** tab: ROI / parcellation-label / attribute dissection of a
  zarr-vectors tractogram (or any polyline store), evaluated in Python and
  written into **ordinary segmentation-layer state** (see the contract below).
- **Export** tab: TrackVis `.trk` or a new zarr-vectors store, downloaded or
  uploaded to GCS.
- **Store** tab: browse / import / save named ROI groups in a GCS bucket.
- **Guide** tab: help.

Nothing is baked into Neuroglancer: ngpy imports nothing from `src/`, and
Neuroglancer's zarr-vectors datasource knows nothing about ngpy.

## Build

```sh
node ngpy/build.ts                 # -> dist/ngpy/ngpy.html (+ dist/ngpy/examples/)
node ngpy/build.ts --no-minify     # readable bundle
node ngpy/build.ts --zarr-vectors /path/to/zarr_vectors-*.whl   # or a package dir
node ngpy/build.ts --offline       # fail instead of downloading the wheel
```

The build bundles the page and the Pyodide worker with esbuild, and embeds the
Python payload as a base64 zip: `ngpy/python/ngpy` → `ngpy/`,
`ngpy/python/vendor/neuroglancer` → `neuroglancer/`, and **zarr-vectors 0.9.2**
(the released wheel from PyPI, sha256-pinned, cached in `dist/ngpy/.cache/`).

## Run

Serve `ngpy.html` from the **same origin** as a Neuroglancer build, e.g. drop
it next to a build's `index.html`:

```sh
npm run build                       # any Neuroglancer build -> dist/client
cp dist/ngpy/ngpy.html dist/client/ && cp -r dist/ngpy/examples dist/client/
# serve dist/client with any static server that sends Content-Type
# (Neuroglancer's worker importScripts needs it) and supports HTTP Range
```

Then open `http://host/ngpy.html`. Query parameters:

| parameter            | meaning                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `ng=<url>`           | the Neuroglancer build to host; default `./index.html`                                                            |
| `pyodide=<indexURL>` | Pyodide distribution; default `https://cdn.jsdelivr.net/pyodide/v314.0.2/full/`                                   |
| `script=<url>`       | a **same-origin** Python script loaded into the editor and run at start (e.g. `?script=examples/hcp1065_demo.py`) |
| `roiStore=<json>`    | ROI-store config, e.g. `{"bucket":"my-groups","clientId":"…apps.googleusercontent.com"}`                          |
| `python=0`           | do not start Python                                                                                               |

The page hash is a Neuroglancer `#!<state>` plus an `ngpy` key holding the
wrapper's own state (the filter groups), so a plain Neuroglancer link opens as
is, and the page URL is shareable. The query string is preserved when the hash
is rewritten (the old build dropped `?script=` when redirecting from `/`).

A **cross-origin** `?ng=` build cannot be scripted (the browser forbids
touching its `window.viewer`); ngpy then only offers write-only `#!` control and
says so in the status bar.

Pasted, opened or dropped scripts ask for confirmation before running — a
script gets full control of the viewer. Only `?script=` on the page's own
origin runs without asking.

## The demo

`examples/hcp1065_demo.py` (ported from the old `user_script.py`) loads the
HCP-1065 tractogram (`gs://hip_ct_zarr_vector_…/hcp1065_whole_brain.zarrvectors`)
over the MNI T1 and SynthSeg parcellation, creates an ROI layer and points the
Filter tab at the three layers with `ngpy.gui.configure_filter(...)`.

## The state contract (what the Filter writes)

On the **target segmentation layer** only, through the layer's own trackables
(never a whole-state restore — upstream's layer-list `restoreState` rebuilds
every layer), falling back to a JSON patch of the full state only on a build
whose layer object lacks those trackables:

| key                    | value                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `segments`             | the union of the objects passing any **visible** group, as decimal segment-id strings, ascending; `[]` when the filter is inactive                                                          |
| `segmentColors`        | `{id: "#rrggbb"}` — the colour of the **first** visible group (list order) each passing object belongs to; replaced wholesale while ngpy manages the layer (off: "group colours" unchecked) |
| `notSelectedAlpha`     | the ghost alpha slider (non-passing objects)                                                                                                                                                |
| `ignoreNullVisibleSet` | `true` while the filter is inactive (empty selection shows everything — upstream semantics); `false` while active, so a dissection that passes nothing shows nothing                        |

Segment ids follow Neuroglancer's zarr-vectors datasource: the store's
`object_attributes/segment_id[row]` when that column exists, otherwise the
zarr-vectors object id (`object_index/object_ids[row]` under the V2 index
layout, the row itself under V1).

Colour-by presets write `skeletonRendering.shader` (+ `shaderControls`) on the
same layer. Group colours only show with the "Segment / group colour" preset.

Standard segment state has no per-segment alpha, so a group's `opacity` is kept
(in documents and the page state) but not rendered; and whether
`notSelectedAlpha` visibly "ghosts" non-passing skeletons is up to the
datasource (upstream applies it to volume rendering).

The wrapper strips the managed layer's `segments`/`segmentColors` from **its
own** page hash (they can be thousands of ids and are regenerated on load); the
hosted viewer's own URL carries them as usual.

## How the dissection is evaluated

- ROIs are boxes / ellipsoids drawn with Neuroglancer's tools in a
  **wrapper-owned local annotation layer** (`ngpy ROIs`, or any local
  annotation layer you pick). Each annotation's group, operator (include / or /
  exclude) and predicate (any segment / any vertex / either / both endpoints)
  live in the wrapper. A new annotation joins the active group as an include.
  If the layer has ngpy's `color` / `exclude` annotation properties, ROIs are
  tinted by group.
- Annotation coordinates are converted into the store's frame by axis **name**
  and unit (e.g. global `x,y,z` in mm → store mm). Layer transforms beyond that
  are not modelled — the same assumption the old in-viewer filter made.
- Python (`ngpy.filter`) reads ONE whole pyramid level through zarr-vectors'
  async read path (`open_store_async` + `read_async`: no JSPI, no zarr
  `sync()`), joins each object's fragments in manifest order, and folds every
  group with the ported `tractography.roi` geometry (left fold, first region
  seeds; `any_segment` is exact for a polyline).
- **Which level**: by default the finest whose `vertex_count` (level metadata)
  is ≤ 2,000,000 (`Evaluate at level` overrides). On an object-sparse pyramid
  coarse levels are SUBSETS of the objects (HCP-1065: 503k / 50k / 5k / 503 /
  50 tracts at levels 0–4, default level 2); a dissection names only objects
  present at the evaluated level. The first evaluation reads the level
  (HCP-1065 level 2 from GCS: ~9.5 s, 13 MB); later ones reuse it
  (~150–300 ms).
- **Labels**: the parcellation layer's sources are classified in Python — a
  `neuroglancer_segment_properties` source gives names/colours (incl. the
  fork's `type: "rgb"` column), the first readable volume (OME-Zarr v2/v3, or
  raw unsharded precomputed) is read whole and sampled under every vertex.
  Label regions are tested at vertices; a group "passes tracts crossing ANY
  included label and NO excluded one". Not supported: precomputed
  `compressed_segmentation`/`jpeg` encodings and sharded precomputed.
- **Attributes**: per-object ranges on `object_attributes/<name>` (or one
  column `name[i]` of a multi-column one), ANDed with the regions. Names are
  listed only where the host can list (GCS JSON API, or local); otherwise type
  the name.
- Point clouds and meshes are not evaluated by this pass (polyline kinds only).

## Export

The visible groups' passing object ids are computed at the evaluation level,
then exactly those objects are read at the chosen export level (pyramid ids are
preserved and nested, so finer levels contain them) and written:

- `.trk`: nibabel (micropip-installed from PyPI on first use), voxel→RAS(mm)
  affine pre-filled from the store's unit; works without JSPI.
- `.zvf` (zipped store): zarr-vectors' **synchronous** writer, run through the
  worker's single JSPI `callPromising` entry under a one-at-a-time promise
  mutex (two concurrent promising calls deadlock the worker). Without JSPI the
  async path reports a clear error.
- Destination: browser download, or GCS upload to `exports/<file>` in the ROI
  store bucket. "Download job spec" gives the v3 job JSON.

## ROI store (GCS)

Same document format as the old `src/roi_store` (`groups/<id>.json`, schema
v1, shapes in the store's frame). Listing and loading are anonymous; saving
signs in with **Google OAuth** (implicit flow in a popup whose redirect target
is ngpy.html itself — register the page URL as an authorised redirect URI) or
**CAVE middleauth** (`/api/v1/authorize` popup; the token is stored under
Neuroglancer's own key, so the hosted viewer shares it). The sign-in flows are
not exercised by the automated tests; the client logic is, against a fetch stub.

## Python API notes

- `neuroglancer` is the **vendored** import closure of upstream
  `viewer_state` / `viewer_base` (google/neuroglancer
  `60c866eee95b0913d48d5d657ec8030f841152c3`, files unmodified) plus three
  ngpy files: `server.py` (no tornado: harmless `set_server_bind_address` etc.),
  `viewer.py` (`Viewer` / `UnsynchronizedViewer` over postMessage) and
  `default_credentials_manager.py`.
- `neuroglancer.Viewer()` attaches to the page's viewer and **adopts** its
  current state; `Viewer(adopt=False)` pushes Python's (empty) state instead,
  as upstream does. The newest `Viewer()` is the attached one.
- Protocol (`ngpy.bridge` ↔ `ngpy/src/host/state_sync.ts`): upstream's
  `ClientStateSynchronizer` semantics — page updates carry
  `{s, g, pg, c}`, Python applies them with a generation check (412 on a
  concurrent Python edit), Python pushes only its own changes (generations
  without `/`), so nothing echoes. ConfigState: `actions`, key bindings
  (`input_event_bindings`, installed as a priority-1000 parent event map),
  `status_messages`, `show*` UI options.
- `ngpy.gui.configure_filter(target=…, roi_layer=…, parcellation=…, level=…)`,
  `ngpy.gui.select_tab(…)`, `ngpy.gui.roi_layer(dimensions)`.
- **Out of scope** (they need a service worker or a blocking wait the worker
  cannot do): `python://` LocalVolume / SkeletonSource data, `screenshot()`,
  `volume()` / `volume_info()` — they raise `NotImplementedError`. Pillow is not
  loaded by default (`await ngpy.api.load_packages(['pillow'])`). Top-level
  `await` works; threads and blocking `input()` do not.
- The viewer's transient `{"type": "new"}` "add layer" placeholder is not passed
  to Python (upstream's `make_layer` rejects that type).

## Known zarr-vectors issues worked around here

- zarr-vectors 0.9.2 `read_polylines` enumerates `vertex_attributes/` and so
  **stalls on any store that cannot list** (plain HTTP) whenever a level has
  per-vertex attributes; when it can list, it decodes every attribute column.
  ngpy reads geometry with its own manifest+vertices reader replayed through
  `read_async` (`ngpy/zv.py`). Pinned by a test.
- zarr-vectors 0.9.2 (and the current dev branch) call `np.repeat` / `np.bincount`
  with int64 arrays (e.g. `decode_object_manifests_many`), which numpy rejects
  on Pyodide's 32-bit `intp`. `ngpy/compat.py` points just those modules' `np`
  at an intp-narrowing proxy on wasm32. Both belong fixed in zarr-vectors-py.

## Tests

```sh
# Python (CPython; the sibling zarr-vectors venv)
PYTHONPATH=ngpy/python /home/andrew/scripts/zarr-vectors-py/.venv/bin/python \
  -m pytest ngpy/python/tests -q -p no:cacheprovider
# TypeScript
npx vitest --run --project node tests/ngpy
npx tsgo --noEmit -p ngpy/tsconfig.json
```

Python tests write tiny real zarr-vectors stores and OME-Zarr parcellations and
read them back through the same async path the browser uses (over a local,
range-aware fetch). What CPython cannot cover — JSPI, the wasm32 numpy
behaviour, the real iframe/viewer — was checked in headless Chromium against a
Neuroglancer build of commit 6e7370d1, with a local tiny store and with
the HCP-1065 demo from GCS (Playwright); that harness is not part of the
repository.
