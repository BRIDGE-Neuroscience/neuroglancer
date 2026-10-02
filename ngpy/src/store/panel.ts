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
 * @file The Store tab: browse / import / save / delete ROI-group documents in
 * a GCS bucket, and the sign-in chip writes need.
 *
 * Configured by `?roiStore=<json>` (e.g.
 * `{"bucket":"my-roi-groups","clientId":"…apps.googleusercontent.com"}`) or
 * the form below (remembered in localStorage).  Listing and loading are
 * anonymous; saving signs in (Google: this page is the OAuth redirect target;
 * or CAVE middleauth).
 */

import type { FilterController } from "../filter/controller.js";
import { button, clear, field, h, section, select, setStatus } from "../ui/dom.js";
import type { RoiStoreAuth, RoiStoreConfig } from "./auth.js";
import { makeAuth, redirectUri } from "./auth.js";
import { RoiGroupStore, roiGroupStoreChanged } from "./gcs_client.js";
import type { RoiGroupSummary } from "./schema.js";
import { makeRoiGroupDocument } from "./schema.js";

const CONFIG_KEY = "ngpy_roi_store_config_v1";

export function loadStoreConfig(param: string | null): RoiStoreConfig | undefined {
  const parse = (text: string | null) => {
    if (!text) return undefined;
    try {
      const c = JSON.parse(text);
      return typeof c?.bucket === "string" && c.bucket ? (c as RoiStoreConfig) : undefined;
    } catch {
      return undefined;
    }
  };
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(CONFIG_KEY);
  } catch {
    stored = null;
  }
  return parse(param) ?? parse(stored);
}

export class StorePanel {
  readonly element = h("div", { class: "ngpy-panel ngpy-store" });
  private status = h("div", { class: "ngpy-status" });
  private listEl = h("div", { class: "ngpy-store-list" });
  private config: RoiStoreConfig | undefined;
  private auth: RoiStoreAuth | undefined;
  private store: RoiGroupStore | undefined;
  private summaries: RoiGroupSummary[] = [];

  constructor(
    private controller: FilterController,
    configParam: string | null,
    private pageUrlForScene: () => string,
  ) {
    this.setConfig(loadStoreConfig(configParam), false);
    roiGroupStoreChanged.add(() => void this.refresh());
    controller.model.changed.add(() => this.renderSave());
    this.render();
  }

  configured(): boolean {
    return this.store !== undefined;
  }

  private setConfig(config: RoiStoreConfig | undefined, persist: boolean) {
    this.config = config;
    this.auth = undefined;
    this.store = undefined;
    if (config !== undefined) {
      try {
        this.auth = makeAuth(config);
        this.auth.changed.add(() => this.render());
      } catch (e) {
        setStatus(this.status, (e as Error).message, "error");
      }
      this.store = new RoiGroupStore({
        bucket: config.bucket,
        endpoint: config.endpoint,
        auth: this.auth,
      });
      if (persist) {
        try {
          localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
        } catch {
          // ignore
        }
      }
    }
  }

  private saveSection = h("div");

  render() {
    clear(this.element);
    const c = this.config;
    const bucket = h("input", { type: "text", value: c?.bucket ?? "", placeholder: "bucket name" });
    const endpoint = h("input", { type: "text", value: c?.endpoint ?? "", placeholder: "https://storage.googleapis.com" });
    let provider = c?.provider ?? "google";
    const clientId = h("input", { type: "text", value: c?.clientId ?? "", placeholder: "OAuth client id" });
    const authServer = h("input", { type: "text", value: c?.authServer ?? "", placeholder: "https://global.daf-apis.com" });
    this.element.append(
      section(
        "Bucket",
        field("Bucket", bucket),
        field("Endpoint", endpoint, "Leave blank for Google Cloud Storage."),
        field(
          "Sign-in",
          select(
            [
              { value: "google", label: "Google OAuth" },
              { value: "middleauth", label: "CAVE middleauth" },
            ],
            provider,
            (v) => (provider = v as any),
          ),
        ),
        field("OAuth client id", clientId, `Register ${redirectUri()} as an authorised redirect URI.`),
        field("middleauth server", authServer),
        h(
          "div",
          { class: "ngpy-row" },
          button("Apply", () => {
            if (!bucket.value.trim()) {
              this.setConfig(undefined, false);
            } else {
              this.setConfig(
                {
                  bucket: bucket.value.trim(),
                  endpoint: endpoint.value.trim() || undefined,
                  provider,
                  clientId: clientId.value.trim() || undefined,
                  authServer: authServer.value.trim() || undefined,
                },
                true,
              );
            }
            this.render();
            void this.refresh();
          }),
          this.auth
            ? this.auth.signedIn
              ? button(`Sign out${this.auth.email ? ` (${this.auth.email})` : ""}`, () => this.auth!.signOut())
              : button("Sign in", () =>
                  this.auth!.signIn().catch((e) => setStatus(this.status, (e as Error).message, "error")),
                )
            : null,
        ),
      ),
    );
    if (this.store === undefined) {
      this.element.append(
        h("p", { class: "ngpy-hint" }, "Not configured. Set a bucket above or pass ?roiStore={\"bucket\":…} in the page URL."),
        this.status,
      );
      return;
    }
    this.element.append(
      section("Saved groups", h("div", { class: "ngpy-row" }, button("Refresh", () => void this.refresh())), this.listEl),
      this.saveSection,
      this.status,
    );
    this.renderList();
    this.renderSave();
  }

  private renderList() {
    clear(this.listEl);
    const source = this.controller.sourceUrl();
    for (const s of this.summaries) {
      const mismatch = source !== undefined && s.sourceUrl !== undefined && s.sourceUrl !== source;
      this.listEl.append(
        h(
          "div",
          { class: "ngpy-store-item" },
          h("span", { class: "ngpy-store-name" }, s.name),
          h(
            "span",
            { class: "ngpy-hint" },
            [s.createdBy, s.updated?.slice(0, 10), mismatch ? "drawn on another store" : ""].filter(Boolean).join(" · "),
          ),
          button("Import", () => void this.importDoc(s.id)),
          button("Delete", () => void this.deleteDoc(s)),
        ),
      );
    }
    if (this.summaries.length === 0) this.listEl.append(h("div", { class: "ngpy-hint" }, "No saved groups listed."));
  }

  private renderSave() {
    if (this.store === undefined) return;
    clear(this.saveSection);
    const groups = this.controller.model.groups;
    let chosen = groups[0]?.id;
    this.saveSection.append(
      section(
        "Save a group",
        h(
          "div",
          { class: "ngpy-row" },
          select(
            groups.map((g) => ({ value: String(g.id), label: g.name })),
            chosen === undefined ? undefined : String(chosen),
            (v) => (chosen = Number(v)),
          ),
          button("Save to store", () => {
            if (chosen !== undefined) void this.saveGroup(chosen);
          }),
        ),
      ),
    );
  }

  async refresh() {
    if (this.store === undefined) return;
    setStatus(this.status, "Listing…", "busy");
    try {
      this.summaries = await this.store.list();
      setStatus(this.status, `${this.summaries.length} saved group(s).`, "ok");
    } catch (e) {
      setStatus(this.status, (e as Error).message, "error");
    }
    this.renderList();
  }

  private async importDoc(id: string) {
    try {
      const doc = await this.store!.read(id);
      this.controller.importGroup(doc.group);
      setStatus(this.status, `Imported “${doc.group?.name ?? id}”.`, "ok");
    } catch (e) {
      setStatus(this.status, `Import failed: ${(e as Error).message}`, "error");
    }
  }

  private async deleteDoc(s: RoiGroupSummary) {
    if (!confirm(`Delete the saved group “${s.name}” from the bucket?`)) return;
    try {
      await this.store!.delete(s.id);
    } catch (e) {
      setStatus(this.status, `Delete failed: ${(e as Error).message}`, "error");
    }
  }

  private async saveGroup(groupId: number) {
    try {
      const group = this.controller.exportGroupJson(groupId);
      const source = this.controller.sourceUrl();
      if (source === undefined) throw new Error("Choose a target layer first");
      const doc = makeRoiGroupDocument({
        group,
        source: { url: source },
        scene: { url: this.pageUrlForScene(), layerName: this.controller.model.settings.targetLayer },
        createdBy: this.auth?.email,
      });
      setStatus(this.status, "Saving…", "busy");
      await this.store!.save(doc);
      setStatus(this.status, `Saved “${group.name}” as groups/${doc.id}.json.`, "ok");
    } catch (e) {
      setStatus(this.status, `Save failed: ${(e as Error).message}`, "error");
    }
  }

  /** Used by the Export tab's "Save to GCS". */
  async uploadExport(fileName: string, blob: Blob, contentType: string): Promise<string> {
    if (this.store === undefined) throw new Error("The ROI store is not configured");
    return this.store.putObject(`exports/${fileName}`, blob, contentType);
  }
}
