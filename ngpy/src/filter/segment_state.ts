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

/**
 * @file Filter result -> ordinary segmentation-layer state.
 *
 * THE STATE CONTRACT (what ngpy writes; nothing else on the layer is touched):
 *
 *   segments              union of the objects passing any visible group, as
 *                         decimal segment-id strings, ascending.  Empty when
 *                         the filter is inactive.
 *   segmentColors         {id: "#rrggbb"} -- the colour of the first visible
 *                         group (in list order) each passing object is in.
 *                         Replaced wholesale (ngpy owns this key while the
 *                         filter manages the layer).
 *   notSelectedAlpha      the "ghost" opacity of non-passing objects.
 *   ignoreNullVisibleSet  `true` while the filter is inactive, so the empty
 *                         selection shows everything (upstream's semantics);
 *                         `false` while active, so a dissection that passes
 *                         nothing shows nothing rather than everything.
 *
 * Two ways to apply it, chosen per layer at run time:
 *
 *  - TARGETED (normal): mutate the layer's own trackables -- the segmentation
 *    group's visible/selected sets, the colour group's stated-colour map, and
 *    the two display-state values -- which is what an ROI drag needs: upstream
 *    `viewer.state.restoreState` clears and rebuilds EVERY layer.
 *  - JSON (fallback, any build whose layer object differs): patch the layer
 *    JSON in the full state and restore it.  Correct but slow.
 */

export interface FilterResult {
  segments: string[];
  colors: Record<string, string>;
  active: boolean;
}

export interface SegmentStateUpdate {
  segments: string[];
  segmentColors: Record<string, string>;
  notSelectedAlpha: number;
  ignoreNullVisibleSet: boolean;
}

export interface SegmentStateOptions {
  /** Opacity of non-passing objects, in [0, 1]. */
  ghostAlpha: number;
  /** Write group colours (false keeps the layer's own colouring). */
  colorByGroup: boolean;
}

export function segmentStateFromResult(
  result: FilterResult | undefined,
  options: SegmentStateOptions,
): SegmentStateUpdate {
  const ghost = Math.min(1, Math.max(0, options.ghostAlpha));
  if (result === undefined || !result.active) {
    return {
      segments: [],
      segmentColors: {},
      notSelectedAlpha: ghost,
      ignoreNullVisibleSet: true,
    };
  }
  const segments = [...result.segments].sort(compareIds);
  const segmentColors: Record<string, string> = {};
  if (options.colorByGroup) {
    for (const id of segments) {
      const c = result.colors[id];
      if (c !== undefined) segmentColors[id] = normalizeColor(c);
    }
  }
  return {
    segments,
    segmentColors,
    notSelectedAlpha: ghost,
    ignoreNullVisibleSet: false,
  };
}

/** Ascending order for decimal uint64 strings. */
export function compareIds(a: string, b: string): number {
  if (a.length !== b.length) return a.length - b.length;
  return a < b ? -1 : a > b ? 1 : 0;
}

export function normalizeColor(c: string): string {
  const m = /^#?([0-9a-fA-F]{6})$/.exec(c.trim());
  if (m !== null) return `#${m[1].toLowerCase()}`;
  const s = /^#?([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])$/.exec(c.trim());
  if (s !== null) return `#${s[1]}${s[1]}${s[2]}${s[2]}${s[3]}${s[3]}`.toLowerCase();
  return c;
}

/** The layer JSON with the update applied (pure; used by the fallback). */
export function applyToLayerJson(layer: any, update: SegmentStateUpdate): any {
  const out = { ...layer };
  out.segments = [...update.segments];
  if (Object.keys(update.segmentColors).length > 0) {
    out.segmentColors = { ...update.segmentColors };
  } else {
    delete out.segmentColors;
  }
  out.notSelectedAlpha = update.notSelectedAlpha;
  out.ignoreNullVisibleSet = update.ignoreNullVisibleSet;
  return out;
}

/** The full viewer state with one layer's segment state replaced (pure). */
export function applyToViewerStateJson(
  state: any,
  layerName: string,
  update: SegmentStateUpdate,
): any {
  const layers = state?.layers;
  if (Array.isArray(layers) && layers.some((l: any) => l?.name === layerName)) {
    return {
      ...state,
      layers: layers.map((l: any) =>
        l?.name === layerName ? applyToLayerJson(l, update) : l,
      ),
    };
  }
  if (layers && typeof layers === "object" && layerName in layers) {
    return {
      ...state,
      layers: { ...layers, [layerName]: applyToLayerJson(layers[layerName], update) },
    };
  }
  throw new Error(`no layer named ${JSON.stringify(layerName)}`);
}

/**
 * Mutate a live segmentation layer's trackables.  `userLayer` is
 * `viewer.layerManager.getLayerByName(name).layer`.  Returns false (having
 * changed nothing) if this build's layer does not expose the expected
 * surfaces, so the caller can fall back to {@link applyToViewerStateJson}.
 */
export function applyTargeted(
  userLayer: any,
  update: SegmentStateUpdate,
): boolean {
  const ds = userLayer?.displayState;
  const group = ds?.segmentationGroupState?.value;
  const colorGroup = ds?.segmentationColorGroupState?.value;
  if (
    group?.visibleSegments === undefined ||
    typeof group.restoreState !== "function" ||
    colorGroup?.segmentStatedColors === undefined ||
    typeof colorGroup.restoreState !== "function" ||
    ds.notSelectedAlpha === undefined ||
    ds.ignoreNullVisibleSet === undefined
  ) {
    return false;
  }
  // Segments.  Clearing `selectedSegments` clears `visibleSegments` too (an
  // upstream hook); the bulk add dispatches once (and makes ONE worker RPC).
  group.selectedSegments?.clear?.();
  group.visibleSegments.clear();
  if (update.segments.length > 0) {
    let bulk = false;
    try {
      group.visibleSegments.add(update.segments.map((s) => BigInt(s)));
      bulk = true;
    } catch {
      bulk = false;
    }
    if (!bulk) group.restoreState({ segments: update.segments });
  }
  // Colours: ngpy owns the stated-colour map while it manages the layer.
  colorGroup.segmentStatedColors.clear();
  if (Object.keys(update.segmentColors).length > 0) {
    colorGroup.restoreState({ segmentColors: update.segmentColors });
  }
  setValue(ds.notSelectedAlpha, update.notSelectedAlpha);
  setValue(ds.ignoreNullVisibleSet, update.ignoreNullVisibleSet);
  return true;
}

function setValue(trackable: any, value: unknown) {
  if ("value" in trackable) {
    trackable.value = value;
  } else if (typeof trackable.restoreState === "function") {
    trackable.restoreState(value);
  }
}

/** Read back what a layer currently shows (for checks / the GUI). */
export function readLayerSegmentState(layerJson: any): SegmentStateUpdate {
  return {
    segments: Array.isArray(layerJson?.segments)
      ? layerJson.segments.map(String)
      : [],
    segmentColors: { ...(layerJson?.segmentColors ?? {}) },
    notSelectedAlpha: Number(layerJson?.notSelectedAlpha ?? 0),
    ignoreNullVisibleSet: layerJson?.ignoreNullVisibleSet !== false,
  };
}
