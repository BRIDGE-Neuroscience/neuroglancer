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
 * @file Page parameters and the wrapper's own URL state.
 *
 * Query parameters (all optional):
 *   ng=<url>        the Neuroglancer build to host (default ./index.html, i.e.
 *                   ngpy.html dropped next to a build's index.html)
 *   pyodide=<url>   Pyodide indexURL (default jsDelivr v314.0.2)
 *   script=<url>    a SAME-ORIGIN Python script, loaded and run at start
 *   roiStore=<json> ROI-store bucket config (see the Store tab)
 *   python=0        do not start Python
 *
 * The page hash is `#!<json>`: a Neuroglancer state, plus an `ngpy` key with
 * the wrapper's own state (the filter model).  Opening the page with a plain
 * Neuroglancer `#!` link therefore just works.  The hash is rewritten with
 * `history.replaceState` and the QUERY IS KEPT -- the old build lost
 * `?script=` when it redirected from `/`.
 */

import { stringifyState } from "./util/signal.js";

export const DEFAULT_NG_URL = "./index.html";

export interface PageParams {
  ngUrl: string;
  pyodideIndexUrl: string | undefined;
  scriptUrl: string | undefined;
  roiStore: string | null;
  startPython: boolean;
}

export function parseParams(search: string): PageParams {
  const q = new URLSearchParams(search);
  return {
    ngUrl: q.get("ng") || DEFAULT_NG_URL,
    pyodideIndexUrl: q.get("pyodide") || undefined,
    scriptUrl: q.get("script") || undefined,
    roiStore: q.get("roiStore"),
    startPython: q.get("python") !== "0",
  };
}

export interface WrapperState {
  viewer: any | undefined;
  ngpy: any | undefined;
}

export function parseWrapperHash(hash: string): WrapperState {
  const raw = hash.replace(/^#/, "");
  if (!raw.startsWith("!")) return { viewer: undefined, ngpy: undefined };
  let json: any;
  try {
    json = JSON.parse(decodeURIComponent(raw.slice(1)));
  } catch {
    return { viewer: undefined, ngpy: undefined };
  }
  if (json === null || typeof json !== "object") return { viewer: undefined, ngpy: undefined };
  const { ngpy, ...viewer } = json;
  return { viewer: Object.keys(viewer).length ? viewer : undefined, ngpy };
}

/**
 * The page hash for a viewer state + wrapper state.  The filter-managed
 * layer's `segments` / `segmentColors` are dropped: they can be thousands of
 * ids, and the wrapper regenerates them from its own state on load.
 */
export function buildWrapperHash(
  viewerState: any,
  ngpy: any,
  managedLayer: string | undefined,
): string {
  let state = viewerState ?? {};
  if (managedLayer !== undefined && Array.isArray(state.layers)) {
    state = {
      ...state,
      layers: state.layers.map((l: any) => {
        if (l?.name !== managedLayer) return l;
        const { segments: _s, segmentColors: _c, ...rest } = l;
        return rest;
      }),
    };
  }
  return `#!${encodeURIComponent(stringifyState({ ...state, ngpy }))}`;
}

/** Resolve a `?script=` URL; only same-origin scripts are allowed. */
export function resolveScriptUrl(script: string, pageUrl: string): URL {
  const url = new URL(script, pageUrl);
  if (url.origin !== new URL(pageUrl).origin) {
    throw new Error(
      `?script= must be on this page's origin (${new URL(pageUrl).origin}); got ${url.origin}`,
    );
  }
  return url;
}
