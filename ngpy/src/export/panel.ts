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
 * @file The Export tab: TRK / ZVF of the current dissection or the whole
 * store, to a browser download or a GCS upload.
 *
 * Selection is two-step: the passing object ids of each visible group are
 * computed at the EVALUATION level (`filter_passing_ids`), then the exporter
 * reads exactly those objects at the EXPORT level (finer levels contain them:
 * pyramid object ids are preserved and nested).  `.zvf` goes through the JSPI
 * promising entry under the worker's mutex.
 */

import type { ExportFormat, ExportScope } from "./spec.js";
import {
  buildJobSpec,
  defaultAffine,
  exportFileName,
  formatAffineText,
  parseAffineText,
} from "./spec.js";
import type { FilterController } from "../filter/controller.js";
import type { StorePanel } from "../store/panel.js";
import {
  button,
  downloadBlob,
  field,
  h,
  section,
  select,
  setStatus,
} from "../ui/dom.js";

export class ExportPanel {
  readonly element = h("div", { class: "ngpy-panel ngpy-export" });
  private status = h("div", { class: "ngpy-status" });
  private format: ExportFormat = "trk";
  private scope: ExportScope = "selected";
  private level: number | undefined;
  private affineText: string | undefined;
  private fileName = "dissection";
  private busy = false;

  constructor(
    private controller: FilterController,
    private store: StorePanel,
  ) {
    controller.storeInfoChanged.add(() => this.render());
    controller.resultChanged.add(() => this.render());
    this.render();
  }

  render() {
    const c = this.controller;
    const info = c.storeInfo;
    this.element.replaceChildren();
    const levels = info?.levels ?? [];
    const evalLevel = c.lastResult?.level ?? info?.defaultLevel ?? 0;
    const level = this.level ?? evalLevel;
    if (this.affineText === undefined && info !== undefined) {
      this.affineText = formatAffineText(defaultAffine(info.unitMeters));
    }
    const visibleGroups = c.model.groups.filter((g) => g.visible);
    this.element.append(
      section(
        "What",
        field(
          "Scope",
          select(
            [
              {
                value: "selected",
                label: `Visible groups (${visibleGroups.length})`,
              },
              {
                value: "whole",
                label: "Whole store (every object at the level)",
              },
            ],
            this.scope,
            (v) => {
              this.scope = v as ExportScope;
              this.render();
            },
          ),
        ),
        field(
          "Format",
          select(
            [
              { value: "trk", label: "TrackVis .trk" },
              {
                value: "zvf",
                label: `zarr-vectors store (.zvf.zip)${c.python.jspi ? "" : " — needs JSPI"}`,
              },
            ],
            this.format,
            (v) => {
              this.format = v as ExportFormat;
              this.render();
            },
          ),
        ),
        field(
          "Read level",
          select(
            levels.map((l) => ({
              value: String(l.level),
              label: `${l.level}: ${l.vertexCount.toLocaleString()} vertices`,
            })),
            String(level),
            (v) => {
              this.level = Number(v);
            },
          ),
          `The dissection is evaluated at level ${evalLevel}; its objects are read at this level (0 = full resolution). The browser holds the selection in memory.`,
        ),
        this.format === "trk"
          ? field(
              "Voxel→RAS (mm)",
              h("textarea", {
                rows: 4,
                class: "ngpy-affine",
                value: this.affineText ?? "",
                onchange: (e: Event) =>
                  (this.affineText = (e.target as HTMLTextAreaElement).value),
              }),
              "4×4 written into the TRK header; pre-filled from the store's unit. Blank = identity.",
            )
          : null,
        field(
          "File name",
          h("input", {
            type: "text",
            value: this.fileName,
            onchange: (e: Event) =>
              (this.fileName = (e.target as HTMLInputElement).value),
          }),
        ),
      ),
      section(
        "Where",
        h(
          "div",
          { class: "ngpy-row" },
          button("Download", () => void this.run("download"), {
            disabled: this.busy,
          }),
          button("Save to GCS", () => void this.run("gcs"), {
            disabled: this.busy || !this.store.configured(),
            title: this.store.configured()
              ? "Upload under exports/ in the ROI-store bucket"
              : "Configure the ROI store first (Store tab)",
          }),
          button("Download job spec", () => void this.downloadSpec()),
        ),
        this.status,
      ),
    );
  }

  private async buildSpec(destination: "download" | "gcs"): Promise<any> {
    const c = this.controller;
    const info = c.storeInfo ?? (await c.loadStoreInfo());
    const source = c.sourceUrl();
    if (info === undefined || source === undefined)
      throw new Error("Choose a target layer on the Filter tab first.");
    const level = this.level ?? c.lastResult?.level ?? info.defaultLevel;
    let groups: any[] = [];
    if (this.scope === "selected") {
      const request = c.buildRequest();
      const visible = (request?.groups ?? []).filter(
        (g: any) => g.id !== 0 && g.visible,
      );
      if (visible.length === 0) throw new Error("No visible group to export.");
      setStatus(this.status, "Selecting objects…", "busy");
      const ids = await c.python.callJson<
        { name: string; objectIds: string[] }[]
      >("filter_passing_ids", { ...request, groups: visible });
      groups = visible.map((g: any, i: number) => ({
        name: g.name,
        color: g.color,
        rois: g.rois,
        objectIds: ids[i]?.objectIds ?? [],
      }));
    }
    return buildJobSpec({
      sourceUrl: source,
      level,
      format: this.format,
      scope: this.scope,
      groups,
      affine:
        this.format === "trk"
          ? parseAffineText(this.affineText ?? "")
          : undefined,
      fileName: this.fileName,
      destination,
    });
  }

  private async run(destination: "download" | "gcs") {
    if (this.busy) return;
    this.busy = true;
    this.render();
    try {
      const spec = await this.buildSpec(destination);
      setStatus(this.status, `Exporting ${this.format.toUpperCase()}…`, "busy");
      const [contentType, body, summaryJson] =
        await this.controller.python.call<[string, Uint8Array, string]>(
          "export",
          [JSON.stringify(spec)],
          this.format === "zvf",
        );
      const summary = JSON.parse(summaryJson);
      if (summary.error) throw new Error(summary.error);
      if (summary.written === false || contentType === "application/json") {
        setStatus(
          this.status,
          summary.message ?? "Nothing to export.",
          "error",
        );
        return;
      }
      const name = exportFileName(this.fileName, this.format);
      const blob = new Blob([body as BlobPart], { type: contentType });
      if (destination === "download") {
        downloadBlob(blob, name);
        setStatus(
          this.status,
          `Wrote ${Number(summary.streamline_count).toLocaleString()} streamlines (${(blob.size / 1e6).toFixed(1)} MB) to ${name}.`,
          "ok",
        );
      } else {
        setStatus(this.status, "Uploading…", "busy");
        const objectName = await this.store.uploadExport(
          name,
          blob,
          contentType,
        );
        setStatus(
          this.status,
          `Uploaded ${Number(summary.streamline_count).toLocaleString()} streamlines as ${objectName}.`,
          "ok",
        );
      }
    } catch (e) {
      setStatus(this.status, `Export failed: ${(e as Error).message}`, "error");
    } finally {
      this.busy = false;
      const text = this.status.textContent;
      const kind = this.status.dataset.kind as any;
      this.render();
      setStatus(this.status, text ?? "", kind);
    }
  }

  private async downloadSpec() {
    try {
      const spec = await this.buildSpec("download");
      downloadBlob(
        new Blob([JSON.stringify(spec, null, 2)], { type: "application/json" }),
        `${this.fileName || "dissection"}.job.json`,
      );
      setStatus(this.status, "Job spec downloaded.", "ok");
    } catch (e) {
      setStatus(this.status, (e as Error).message, "error");
    }
  }
}
