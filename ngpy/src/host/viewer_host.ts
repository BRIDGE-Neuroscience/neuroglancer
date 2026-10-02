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
 * @file Hosting any Neuroglancer build in an iframe.
 *
 * Same-origin: wait for `iframe.contentWindow.viewer` (every Neuroglancer
 * entry point assigns `window.viewer`) and drive it through its public state.
 * Cross-origin: the browser forbids touching the viewer object, so ngpy
 * degrades to WRITE-ONLY control by setting the iframe's `#!<state>` hash;
 * scripts and filters cannot read the viewer back, and the GUI says so.
 */

import { Signal, stringifyState } from "../util/signal.js";

export function isSameOrigin(url: string, base: string = window.location.href): boolean {
  try {
    return new URL(url, base).origin === new URL(base).origin;
  } catch {
    return false;
  }
}

/** The iframe URL for a Neuroglancer build plus an optional state hash. */
export function viewerUrlWithState(ngUrl: string, state?: unknown): string {
  const base = ngUrl.replace(/#.*$/, "");
  if (state === undefined) return ngUrl;
  return `${base}#!${encodeURIComponent(stringifyState(state))}`;
}

export class ViewerHost {
  readonly iframe: HTMLIFrameElement;
  readonly sameOrigin: boolean;
  readonly viewerChanged = new Signal();
  viewer: any = undefined;
  crossOriginDetected = false;
  readonly ready: Promise<any>;

  constructor(
    container: HTMLElement,
    readonly ngUrl: string,
    initialState?: unknown,
    private timeoutMs = 120_000,
  ) {
    this.sameOrigin = isSameOrigin(ngUrl);
    const iframe = (this.iframe = document.createElement("iframe"));
    iframe.className = "ngpy-viewer-frame";
    iframe.title = "Neuroglancer";
    iframe.allow = "clipboard-read; clipboard-write; fullscreen";
    iframe.src = viewerUrlWithState(ngUrl, initialState);
    container.appendChild(iframe);
    this.ready = this.waitForViewer();
  }

  get contentWindow(): any {
    return this.iframe.contentWindow;
  }

  private waitForViewer(): Promise<any> {
    return new Promise((resolve) => {
      const start = Date.now();
      const poll = () => {
        let viewer: any;
        try {
          viewer = this.contentWindow?.viewer;
          // Touching a cross-origin window's properties throws.
          void this.contentWindow?.document;
        } catch {
          this.crossOriginDetected = true;
          this.viewerChanged.dispatch();
          resolve(undefined);
          return;
        }
        if (viewer?.state !== undefined && viewer?.layerManager !== undefined) {
          this.viewer = viewer;
          this.viewerChanged.dispatch();
          resolve(viewer);
          return;
        }
        if (Date.now() - start > this.timeoutMs) {
          resolve(undefined);
          return;
        }
        setTimeout(poll, 100);
      };
      poll();
    });
  }

  /** Write-only control (cross-origin fallback): reload the iframe with a state hash. */
  setStateViaHash(state: unknown) {
    this.iframe.src = viewerUrlWithState(this.ngUrl, state);
  }
}
