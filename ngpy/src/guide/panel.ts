/**
 * @license
 * Copyright 2026 The Neuroglancer Authors
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

/** @file The Guide tab: static help (the old streamline guide, updated). */

import { h } from "../ui/dom.js";

const GUIDE = `
<h3>What this is</h3>
<p><b>ngpy</b> wraps an unmodified Neuroglancer build (the right-hand pane) with
its own Python (Pyodide) runtime and tools. Nothing here is part of Neuroglancer:
results are written into ordinary layer state, so they survive in the viewer's
URL and work in any build.</p>

<h3>Dissecting a tractogram (Filter tab)</h3>
<ol>
<li><b>Target</b>: pick the segmentation layer whose source is a zarr-vectors
store (<code>…/|zarr-vectors:</code>). ngpy reads the store's metadata and
chooses an evaluation level.</li>
<li><b>ROI layer</b>: create one (or choose an existing local annotation layer)
and draw <b>bounding boxes</b> or <b>ellipsoids</b> with Neuroglancer's
annotation tools. Each new region joins the <b>active group</b> (the radio
button) as an <i>include</i>.</li>
<li>Per region choose the <b>operator</b> — include (AND), or (OR), exclude
(NOT) — and the <b>predicate</b>: <i>any segment</i> (exact for a polyline, the
default), <i>any vertex</i> (TrackVis's test; can step over a small region),
<i>either</i>/<i>both endpoints</i>. Regions fold left to right, so order
matters; a group that starts with an exclusion selects "everything except".</li>
<li>Every visible group is evaluated independently. A tract passing several
groups takes the colour of the topmost one.</li>
</ol>
<p><b>By segmentation label</b>: choose a parcellation layer (an OME-Zarr or raw
precomputed volume; a <code>segment_properties</code> source on the same layer
supplies names). Click labels to cycle include → exclude → off; tracts crossing
ANY included label and NO excluded one preview in white. <i>Create group from
selection</i> commits it.</p>
<p><b>By attribute</b>: a range on a per-object attribute (e.g.
<code>tortuosity</code>, or <code>orientation[2]</code> for one column), alone or
ANDed with a group's regions.</p>

<h3>What is written to the viewer</h3>
<ul>
<li><code>segments</code>: the passing objects (union of visible groups);</li>
<li><code>segmentColors</code>: each passing object's group colour;</li>
<li><code>notSelectedAlpha</code>: the ghost alpha slider;</li>
<li><code>ignoreNullVisibleSet</code>: off while a filter is active (an empty
result shows nothing), on otherwise (all objects show).</li>
</ul>
<p>Group colours only show while the layer's skeleton shader uses the segment
colour — use <i>Colour by → Segment / group colour</i>.</p>

<h3>Which objects are evaluated</h3>
<p>The browser cannot hold a whole-brain level 0, so the dissection reads ONE
pyramid level — by default the finest with at most 2,000,000 vertices — and
answers for the objects at that level. On an object-sparse pyramid (e.g.
HCP-1065: 503k / 50k / 5k / 503 / 50 tracts) coarse levels are subsets: a
level-2 dissection names only the 5k tracts present there. Pick another level
under <i>Evaluate at level</i>. Export can read the passing objects at a finer
level (they are present there too).</p>

<h3>Export</h3>
<p>TrackVis <code>.trk</code> or a new zarr-vectors store (zipped), for the
visible groups or the whole store, downloaded or uploaded to the ROI-store
bucket under <code>exports/</code>. ZVF needs WebAssembly stack switching (JSPI,
current Chrome/Edge); TRK works everywhere (nibabel is installed on first use).</p>

<h3>Store</h3>
<p>Save groups to, and import them from, a shared GCS bucket. Listing and loading
are anonymous; saving signs in (Google OAuth with this page as the redirect, or
CAVE middleauth). Region shapes are saved in the store's own coordinates.</p>

<h3>Python</h3>
<p><code>import neuroglancer; viewer = neuroglancer.Viewer()</code> gives a
handle on the viewer: <code>with viewer.txn() as s: …</code>,
<code>viewer.state</code>, <code>viewer.actions.add(…)</code> and
<code>viewer.config_state</code> key bindings work as with a local server. A new
<code>Viewer()</code> starts from what is on screen
(<code>Viewer(adopt=False)</code> resets it). Not available: <code>python://</code>
local volumes, <code>screenshot()</code>, <code>volume()</code>. Use
<code>await</code> at top level for async work; there is no blocking
<code>input()</code> or threads.</p>
`;

export function makeGuidePanel(): HTMLElement {
  const el = h("div", { class: "ngpy-panel ngpy-guide" });
  el.innerHTML = GUIDE;
  return el;
}
