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
 * @file Python's ConfigState ("c") applied to a hosted viewer.
 *
 * Upstream's Python-integrated entry point (`main_python.ts`) wires these keys
 * into the viewer it builds; an arbitrary Neuroglancer build has no such
 * wiring, so ngpy applies the useful subset through public surfaces:
 *
 *  - `actions`: listen for `action:<name>` events on `viewer.element` (the
 *    events Neuroglancer's key/mouse bindings dispatch) and report each to
 *    Python with the upstream action state (mouse position, selected values,
 *    viewer state);
 *  - `inputEventBindings`: a child `EventActionMap` per scope added as a
 *    priority-1000 parent of the viewer's own maps -- exactly what upstream
 *    does -- or, failing that, direct `set`/`delete` on the viewer's maps;
 *  - `statusMessages`: shown in the wrapper's status bar;
 *  - `show*` UI options: `viewer.uiConfiguration[key].value`.
 *
 * Ignored: screenshots, volume requests, prefetch, viewer size, source
 * generations (they need upstream's python datasource or a blocking wait).
 */

import { stringifyState } from "../util/signal.js";

export interface ConfigSyncHooks {
  sendAction(action: string, state: unknown): void;
  setStatusMessages(messages: Record<string, string>): void;
}

const UI_KEYS = [
  "showUIControls",
  "showTopBar",
  "showLocation",
  "showLayerPanel",
  "showHelpButton",
  "showSettingsButton",
  "showLayerSidePanelButton",
  "showScreenshotButton",
  "showToolPaletteButton",
  "showLayerListPanelButton",
  "showSelectionPanelButton",
  "showCopyUrlButton",
  "showPanelBorders",
  "showLayerHoverValues",
];

const BINDING_SCOPES: [string, string[]][] = [
  ["viewer", ["global"]],
  ["sliceView", ["sliceView"]],
  ["perspectiveView", ["perspectiveView"]],
  ["dataView", ["sliceView", "perspectiveView"]],
];

export class ConfigSync {
  private actionListeners = new Map<string, (e: Event) => void>();
  private bindingMaps = new Map<string, any>();
  private directBindings: { map: any; key: string }[] = [];

  constructor(
    private viewer: any,
    private hooks: ConfigSyncHooks,
  ) {}

  apply(config: any) {
    this.applyActions(config?.actions ?? []);
    this.applyBindings(config?.inputEventBindings ?? {});
    this.hooks.setStatusMessages(config?.statusMessages ?? {});
    const ui = this.viewer.uiConfiguration;
    if (ui !== undefined) {
      for (const key of UI_KEYS) {
        const value = config?.[key];
        if (value !== undefined && ui[key] !== undefined && "value" in ui[key]) {
          ui[key].value = value;
        }
      }
    }
  }

  private applyActions(actions: string[]) {
    const element: EventTarget | undefined = this.viewer.element;
    if (element === undefined) return;
    const wanted = new Set(actions);
    for (const [name, listener] of this.actionListeners) {
      if (!wanted.has(name)) {
        element.removeEventListener(`action:${name}`, listener);
        this.actionListeners.delete(name);
      }
    }
    for (const name of wanted) {
      if (this.actionListeners.has(name) || name === "screenshot" || name === "screenshotStatistics") {
        continue;
      }
      const listener = () => this.hooks.sendAction(name, this.actionState());
      element.addEventListener(`action:${name}`, listener);
      this.actionListeners.set(name, listener);
    }
  }

  /** Upstream's `RemoteActionHandler.handleAction` payload. */
  actionState(): any {
    const { mouseState, layerSelectedValues } = this.viewer;
    const state: any = {};
    try {
      if (mouseState?.updateUnconditionally?.() ?? mouseState?.active) {
        state.mousePosition = Array.from(mouseState.position);
      }
    } catch {
      // Older/newer builds: omit the mouse position.
    }
    try {
      state.selectedValues = layerSelectedValues?.toJSON?.() ?? {};
    } catch {
      state.selectedValues = {};
    }
    state.viewerState = this.viewer.state.toJSON();
    return JSON.parse(stringifyState(state));
  }

  private applyBindings(bindings: Record<string, Record<string, string>>) {
    const ieb = this.viewer.inputEventBindings;
    if (ieb === undefined) return;
    // Undo direct (fallback) bindings from the previous config.
    for (const { map, key } of this.directBindings) map.delete?.(key);
    this.directBindings = [];
    for (const [scope, targets] of BINDING_SCOPES) {
      const entries = bindings[scope] ?? {};
      let child = this.bindingMaps.get(scope);
      if (child === undefined) {
        const parentMap = ieb[targets[0]];
        const Ctor = parentMap?.constructor;
        if (Ctor !== undefined && typeof parentMap.addParent === "function") {
          try {
            child = new Ctor();
            for (const t of targets) ieb[t]?.addParent(child, scope === "dataView" ? 999 : 1000);
            this.bindingMaps.set(scope, child);
          } catch {
            child = undefined;
          }
        }
      }
      if (child !== undefined) {
        child.clear?.();
        for (const [key, action] of Object.entries(entries)) child.set(key, action);
      } else {
        for (const t of targets) {
          const map = ieb[t];
          if (map?.set === undefined) continue;
          for (const [key, action] of Object.entries(entries)) {
            map.set(key, action);
            this.directBindings.push({ map, key });
          }
        }
      }
    }
  }
}
