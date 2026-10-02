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
 * @file The ngpy page: layout, the hosted viewer, Python, and the tabs.
 *
 *   ┌ side panel (tabs) ┐┃┌ iframe: any Neuroglancer build ┐
 *   │ Python Filter …   │┃│  window.viewer                 │
 *   └───────────────────┘┃└────────────────────────────────┘
 *   status bar: Python · viewer · Python status messages
 */

import { ExportPanel } from "./export/panel.js";
import { FilterController } from "./filter/controller.js";
import { FilterModel } from "./filter/model.js";
import { FilterPanel } from "./filter/panel.js";
import { makeGuidePanel } from "./guide/panel.js";
import { ConfigSync } from "./host/config_sync.js";
import { SharedStateSync } from "./host/state_sync.js";
import { stateJson } from "./host/viewer_api.js";
import { ViewerHost } from "./host/viewer_host.js";
import type { PageParams } from "./params.js";
import { buildWrapperHash, parseWrapperHash, resolveScriptUrl } from "./params.js";
import { DEFAULT_PYODIDE_INDEX_URL, PythonClient } from "./python/client.js";
import { PythonPanel } from "./python/panel.js";
import { StorePanel } from "./store/panel.js";
import { h } from "./ui/dom.js";
import { debounce } from "./util/signal.js";

export interface Embedded {
  workerSource: string;
  payload: () => ArrayBuffer;
  demoScript: string;
  buildInfo: { version: string; builtAt: string; zarrVectors: string };
}

export class App {
  readonly python: PythonClient;
  readonly host: ViewerHost;
  readonly model = new FilterModel();
  private tabs = new Map<string, { button: HTMLButtonElement; panel: HTMLElement }>();
  private tabBar = h("nav", { class: "ngpy-tabs" });
  private tabBody = h("div", { class: "ngpy-tab-body" });
  private statusPython = h("span", { class: "ngpy-chip" }, "Python: idle");
  private statusViewer = h("span", { class: "ngpy-chip" }, "Viewer: loading");
  private statusMessages = h("span", { class: "ngpy-messages" });
  private sync: SharedStateSync | undefined;
  private configSync: ConfigSync | undefined;
  private pythonPanel: PythonPanel;
  controller: FilterController | undefined;

  constructor(
    private root: HTMLElement,
    private params: PageParams,
    embedded: Embedded,
  ) {
    const initial = parseWrapperHash(window.location.hash);
    if (initial.ngpy?.filter !== undefined) this.model.restoreState(initial.ngpy.filter);

    const side = h("aside", { class: "ngpy-side" }, this.tabBar, this.tabBody);
    const splitter = h("div", { class: "ngpy-splitter", title: "Drag to resize" });
    const main = h("main", { class: "ngpy-main" });
    const statusBar = h(
      "footer",
      { class: "ngpy-statusbar" },
      h("b", { title: `built ${embedded.buildInfo.builtAt}` }, `ngpy ${embedded.buildInfo.version}`),
      this.statusPython,
      this.statusViewer,
      this.statusMessages,
    );
    root.append(h("div", { class: "ngpy-layout" }, side, splitter, main), statusBar);
    this.installSplitter(side, splitter);

    this.host = new ViewerHost(main, params.ngUrl, initial.viewer);
    this.python = new PythonClient(embedded.workerSource, embedded.payload);
    this.pythonPanel = new PythonPanel(this.python, () => this.sync?.flush() ?? Promise.resolve(), embedded.demoScript);
    this.addTab("python", "Python", this.pythonPanel.element);
    this.python.statusChanged.add(() => this.renderPythonStatus());
    this.python.progress.add((m) => (this.statusPython.textContent = `Python: ${m}`));
    this.renderPythonStatus();
  }

  async start() {
    if (this.params.startPython) {
      const indexURL = this.params.pyodideIndexUrl ?? DEFAULT_PYODIDE_INDEX_URL;
      void this.python
        .start(indexURL, { pageUrl: window.location.href, viewerUrl: this.params.ngUrl })
        .catch((e) => console.error("ngpy: Python failed to start", e));
    }
    const viewer = await this.host.ready;
    if (viewer === undefined) {
      this.statusViewer.textContent = this.host.crossOriginDetected || !this.host.sameOrigin
        ? "Viewer: cross-origin — write-only (#! hash) control; Filter and scripts cannot read it"
        : "Viewer: not found (is ?ng= a Neuroglancer build?)";
      this.statusViewer.dataset.kind = "error";
      this.addTab("guide", "Guide", makeGuidePanel());
      this.selectTab("python");
      return;
    }
    this.statusViewer.textContent = `Viewer: ${this.params.ngUrl}`;
    this.statusViewer.dataset.kind = "ok";
    this.attachViewer(viewer);
  }

  private attachViewer(viewer: any) {
    const controller = (this.controller = new FilterController(viewer, this.python, this.model));
    const filterPanel = new FilterPanel(controller);
    const storePanel = new StorePanel(controller, this.params.roiStore, () => window.location.href);
    const exportPanel = new ExportPanel(controller, storePanel);
    this.addTab("filter", "Filter", filterPanel.element);
    this.addTab("export", "Export", exportPanel.element);
    this.addTab("store", "Store", storePanel.element);
    this.addTab("guide", "Guide", makeGuidePanel());
    this.selectTab(this.model.settings.targetLayer ? "filter" : "python");

    // Shared state <-> Python.
    this.sync = new SharedStateSync(viewer.state, async (msg) =>
      JSON.parse(await this.python.call<string>("client_state", [JSON.stringify(msg)])),
    );
    this.configSync = new ConfigSync(viewer, {
      sendAction: (action, state) =>
        void this.python.call("action", [JSON.stringify({ action, state })]),
      setStatusMessages: (messages) => {
        this.statusMessages.textContent = Object.values(messages).join(" · ");
      },
    });
    this.python.emitted.add((kind, text) => {
      const msg = JSON.parse(text);
      if (kind === "state" && msg.k === "s") this.sync?.setServerState(msg.s, msg.g);
      else if (kind === "state" && msg.k === "c") this.configSync?.apply(msg.s);
      else if (kind === "generation") this.sync?.setServerGeneration(msg.g);
      else if (kind === "gui") this.handleGuiRequest(msg);
    });

    // Keep the page URL shareable: viewer state + wrapper state in our hash.
    const writeHash = debounce(() => {
      try {
        const hash = buildWrapperHash(
          stateJson(viewer),
          { filter: this.model.toJSON() },
          this.model.isActive() ? this.model.settings.targetLayer : undefined,
        );
        history.replaceState(null, "", `${window.location.pathname}${window.location.search}${hash}`);
      } catch (e) {
        console.warn("ngpy: could not update the page URL", e);
      }
    }, 1000);
    viewer.state.changed.add(writeHash);
    this.model.changed.add(writeHash);

    const onPythonReady = async () => {
      await this.sync!.enable();
      if (this.model.settings.targetLayer) void controller.evaluate();
      if (this.model.settings.parcellationLayer) void controller.loadLabels();
      await this.runScriptParam();
    };
    if (this.python.status === "ready") void onPythonReady();
    else {
      const d = this.python.statusChanged.add(() => {
        if (this.python.status === "ready") {
          d();
          void onPythonReady();
        }
      });
    }
  }

  /** `ngpy.gui` requests from a Python script. */
  private handleGuiRequest(msg: any) {
    if (msg.filter !== undefined) {
      Object.assign(this.model.settings, msg.filter);
      this.model.dispatch();
      const c = this.controller;
      if (c !== undefined) {
        void c.loadStoreInfo().then(() => c.schedule());
        if (msg.filter.parcellationLayer !== undefined) void c.loadLabels();
      }
    }
    if (typeof msg.tab === "string") this.selectTab(msg.tab);
  }

  private async runScriptParam() {
    const script = this.params.scriptUrl;
    if (script === undefined) return;
    try {
      const url = resolveScriptUrl(script, window.location.href);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);
      this.pythonPanel.load(await response.text(), true);
      this.selectTab("python");
      await this.pythonPanel.run();
    } catch (e) {
      this.pythonPanel.print(`?script=: ${(e as Error).message}\n`, "stderr");
    }
  }

  private renderPythonStatus() {
    const p = this.python;
    this.statusPython.dataset.kind = p.status === "ready" ? "ok" : p.status === "failed" ? "error" : "busy";
    if (p.status === "ready") this.statusPython.textContent = `Python ${p.info?.python ?? ""} ready${p.jspi ? " · JSPI" : ""}`;
    else if (p.status === "failed") this.statusPython.textContent = "Python failed";
    else if (p.status === "idle") this.statusPython.textContent = "Python: off";
  }

  addTab(id: string, label: string, panel: HTMLElement) {
    const button = h("button", { type: "button", class: "ngpy-tab", onclick: () => this.selectTab(id) }, label);
    this.tabBar.append(button);
    panel.hidden = true;
    this.tabBody.append(panel);
    this.tabs.set(id, { button, panel });
    if (this.tabs.size === 1) this.selectTab(id);
  }

  selectTab(id: string) {
    for (const [key, { button, panel }] of this.tabs) {
      const on = key === id;
      button.classList.toggle("active", on);
      panel.hidden = !on;
    }
  }

  private installSplitter(side: HTMLElement, splitter: HTMLElement) {
    const KEY = "ngpy_side_width";
    let width = 380;
    try {
      width = Number(localStorage.getItem(KEY)) || width;
    } catch {
      // ignore
    }
    side.style.width = `${width}px`;
    splitter.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      splitter.setPointerCapture(e.pointerId);
      const frame = this.root.querySelector("iframe");
      if (frame) frame.style.pointerEvents = "none";
      const move = (ev: PointerEvent) => {
        width = Math.min(Math.max(220, ev.clientX), window.innerWidth - 200);
        side.style.width = `${width}px`;
      };
      const up = () => {
        splitter.removeEventListener("pointermove", move);
        splitter.removeEventListener("pointerup", up);
        if (frame) frame.style.pointerEvents = "";
        try {
          localStorage.setItem(KEY, String(width));
        } catch {
          // ignore
        }
      };
      splitter.addEventListener("pointermove", move);
      splitter.addEventListener("pointerup", up);
    });
  }
}
