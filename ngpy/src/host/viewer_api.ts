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
 * @file Every place ngpy touches the hosted viewer object, in one module.
 *
 * Only long-stable public surfaces are used, each duck-typed with a JSON
 * fallback: `viewer.state` (`toJSON` / `restoreState` / `changed`),
 * `viewer.layerManager` (`getLayerByName`, `layersChanged`), a layer's
 * `toJSON()`, a local annotation layer's `localAnnotations`
 * (`toJSON` / `restoreState` / `changed`), and a segmentation layer's display
 * trackables (see `filter/segment_state.ts`).
 */

import type { SegmentStateUpdate } from "../filter/segment_state.js";
import { applyTargeted, applyToViewerStateJson } from "../filter/segment_state.js";

export function stateJson(viewer: any): any {
  return viewer.state.toJSON();
}

export function layerSpecs(viewer: any): any[] {
  const layers = stateJson(viewer)?.layers;
  if (Array.isArray(layers)) return layers;
  if (layers && typeof layers === "object") {
    return Object.entries(layers).map(([name, spec]: [string, any]) => ({ ...spec, name }));
  }
  return [];
}

export function managedLayer(viewer: any, name: string): any {
  return viewer.layerManager?.getLayerByName?.(name);
}

export function layerJson(viewer: any, name: string): any {
  const m = managedLayer(viewer, name);
  if (m !== undefined && typeof m.toJSON === "function") return m.toJSON();
  return layerSpecs(viewer).find((l) => l.name === name);
}

/** Every source URL of a layer spec (string, `{url}` or an array of those). */
export function layerSourceUrls(spec: any): string[] {
  const src = spec?.source;
  const list = Array.isArray(src) ? src : src === undefined ? [] : [src];
  return list
    .map((s: any) => (typeof s === "string" ? s : s?.url))
    .filter((u: any): u is string => typeof u === "string");
}

export function isZarrVectorsUrl(url: string): boolean {
  return /\|zarr-vectors:|^zarr-vectors:\/\//.test(url) || /\.(zarrvectors|zvf|zv)\/?$/.test(url);
}

/** Replace the whole state (rebuilds every layer -- use sparingly). */
export function restoreState(viewer: any, state: any) {
  viewer.state.restoreState(state);
}

/** Add one layer.  Upstream offers no public single-layer add, so this goes
 *  through the whole state once (all layers are rebuilt). */
export function addLayer(viewer: any, spec: any) {
  const state = stateJson(viewer);
  const layers = Array.isArray(state.layers) ? state.layers : [];
  restoreState(viewer, { ...state, layers: [...layers, spec] });
}

export type ApplyPath = "targeted" | "state";

/** Write a filter result into a segmentation layer (targeted, else via state). */
export function applySegmentState(
  viewer: any,
  layerName: string,
  update: SegmentStateUpdate,
): ApplyPath {
  const m = managedLayer(viewer, layerName);
  if (m?.layer !== undefined && m.layer !== null && applyTargeted(m.layer, update)) {
    return "targeted";
  }
  restoreState(viewer, applyToViewerStateJson(stateJson(viewer), layerName, update));
  return "state";
}

/** Set a segmentation layer's skeleton shader (+ controls). */
export function setSkeletonShader(
  viewer: any,
  layerName: string,
  shader: string,
  controls: Record<string, unknown> = {},
): ApplyPath {
  const m = managedLayer(viewer, layerName);
  const opts = m?.layer?.displayState?.skeletonRenderingOptions;
  if (opts?.shader !== undefined && "value" in opts.shader) {
    opts.shader.value = shader;
    opts.shaderControlState?.restoreState?.(controls);
    return "targeted";
  }
  const state = stateJson(viewer);
  state.layers = (state.layers ?? []).map((l: any) =>
    l.name === layerName
      ? {
          ...l,
          skeletonRendering: {
            ...(l.skeletonRendering ?? {}),
            shader,
            shaderControls: controls,
          },
        }
      : l,
  );
  restoreState(viewer, state);
  return "state";
}

// -- local annotation layers (the ROI layer) --------------------------------

export function annotationSource(viewer: any, layerName: string): any {
  return managedLayer(viewer, layerName)?.layer?.localAnnotations;
}

export function readAnnotations(viewer: any, layerName: string): any[] {
  const src = annotationSource(viewer, layerName);
  if (src !== undefined && typeof src.toJSON === "function") return src.toJSON() ?? [];
  return layerJson(viewer, layerName)?.annotations ?? [];
}

/** Replace the annotations of a local annotation layer (references survive). */
export function writeAnnotations(viewer: any, layerName: string, annotations: any[]): boolean {
  const src = annotationSource(viewer, layerName);
  if (src === undefined || typeof src.restoreState !== "function") return false;
  src.restoreState(annotations);
  return true;
}

/** Coordinate dimensions of a layer's annotations (falls back to the global ones). */
export function annotationDimensions(viewer: any, layerName: string): any {
  const spec = layerJson(viewer, layerName);
  const src = Array.isArray(spec?.source) ? spec.source[0] : spec?.source;
  return src?.transform?.outputDimensions ?? stateJson(viewer)?.dimensions ?? {};
}

/**
 * Call `callback` whenever the named annotation layer's annotations change
 * (or the layer itself is replaced).  Returns a disposer.
 */
export function watchAnnotations(
  viewer: any,
  layerName: string,
  callback: () => void,
): () => void {
  let detachSource: (() => void) | undefined;
  let attachedTo: any;
  const attach = () => {
    const src = annotationSource(viewer, layerName);
    if (src === attachedTo) return;
    detachSource?.();
    detachSource = undefined;
    attachedTo = src;
    if (src?.changed?.add !== undefined) {
      const d = src.changed.add(callback);
      detachSource = typeof d === "function" ? d : () => src.changed.remove?.(callback);
    }
    callback();
  };
  const onLayers = () => attach();
  const d = viewer.layerManager?.layersChanged?.add?.(onLayers);
  attach();
  return () => {
    detachSource?.();
    if (typeof d === "function") d();
    else viewer.layerManager?.layersChanged?.remove?.(onLayers);
  };
}
