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
 * @file The Filter tab: target layer, ROI groups, label and attribute
 * dissection, colour-by.  A rewrite of the old in-viewer
 * `streamline_filter_tab` / `label_filter_panel` / `attribute_filter_panel` /
 * `background_color_by_controls` against the wrapper's own model.
 */

import { colorPresets } from "./colorby.js";
import type { FilterController } from "./controller.js";
import type { AttrFilter, Group, GroupRoi, LabelRoi } from "./model.js";
import { OPERATORS, PREDICATES } from "./model.js";
import { setSkeletonShader } from "../host/viewer_api.js";
import {
  button,
  clear,
  field,
  h,
  section,
  select,
  setStatus,
} from "../ui/dom.js";

type LabelState = "include" | "exclude";

export class FilterPanel {
  readonly element = h("div", { class: "ngpy-panel ngpy-filter" });
  private status = h("div", { class: "ngpy-status" });
  private body = h("div");
  private labelState = new Map<number, LabelState>();
  private labelSearch = "";
  private attrDraft: AttrFilter | undefined;
  private renderQueued = false;

  constructor(private controller: FilterController) {
    this.element.append(this.status, this.body);
    controller.statusChanged.add((text, kind) =>
      setStatus(this.status, text, kind as any),
    );
    controller.model.changed.add(() => this.queueRender());
    controller.resultChanged.add(() => this.queueRender());
    controller.storeInfoChanged.add(() => this.queueRender());
    controller.layersChanged.add(() => this.queueRender());
    this.render();
  }

  private queueRender() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      // Keep focus inside text inputs: do not rebuild while the user types.
      const active = document.activeElement;
      if (
        active instanceof HTMLInputElement &&
        this.element.contains(active) &&
        (active.type === "text" ||
          active.type === "number" ||
          active.type === "search")
      ) {
        active.addEventListener("blur", () => this.queueRender(), {
          once: true,
        });
        return;
      }
      this.render();
    });
  }

  render() {
    const c = this.controller;
    const m = c.model;
    clear(this.body);
    if (c.python.status !== "ready") {
      this.body.append(h("p", { class: "ngpy-hint" }, "Waiting for Python…"));
    }
    // --- target -----------------------------------------------------------
    const candidates = c.targetCandidates();
    const target = select(
      [
        { value: "", label: "— choose a zarr-vectors layer —" },
        ...candidates.map((n) => ({ value: n, label: n })),
      ],
      m.settings.targetLayer ?? "",
      (v) => {
        m.settings.targetLayer = v || undefined;
        c.lastResult = undefined;
        m.dispatch();
        void c.loadStoreInfo();
      },
    );
    const info = c.storeInfo;
    const levelOptions = [
      {
        value: "auto",
        label: `auto${info ? ` (level ${info.defaultLevel})` : ""}`,
      },
      ...(info?.levels ?? []).map((l) => ({
        value: String(l.level),
        label: `${l.level}: ${l.vertexCount.toLocaleString()} vertices`,
      })),
    ];
    const level = select(
      levelOptions,
      m.settings.level === undefined ? "auto" : String(m.settings.level),
      (v) => {
        m.settings.level = v === "auto" ? undefined : Number(v);
        m.dispatch();
      },
    );
    this.body.append(
      section(
        "Target",
        field(
          "Layer",
          target,
          candidates.length === 0
            ? "No segmentation layer with a zarr-vectors source."
            : undefined,
        ),
        field(
          "Evaluate at level",
          level,
          "A dissection reads one whole pyramid level. On an object-sparse pyramid coarse levels hold a subset of the objects; objects absent from the level are never selected.",
        ),
        info
          ? h(
              "div",
              { class: "ngpy-hint" },
              `${info.base} · units ${info.units ?? "?"} · axes ${info.axes.join(",")}`,
            )
          : null,
      ),
    );

    // --- ROI layer ---------------------------------------------------------
    const annLayers = c.annotationLayers();
    const roiSelect = select(
      [
        { value: "", label: "— none —" },
        ...annLayers.map((n) => ({ value: n, label: n })),
      ],
      m.settings.roiLayer ?? "",
      (v) => {
        m.settings.roiLayer = v || undefined;
        m.dispatch();
      },
    );
    this.body.append(
      section(
        "ROI layer",
        field("Annotation layer", roiSelect),
        h(
          "div",
          { class: "ngpy-row" },
          button("Create ROI layer", () => c.createRoiLayer(), {
            title:
              "Adds a local annotation layer (reloads the layer list once)",
          }),
        ),
        h(
          "p",
          { class: "ngpy-hint" },
          "Draw bounding boxes or ellipsoids in this layer with Neuroglancer's annotation tools. A new ROI joins the active group as an include; change its operator below.",
        ),
      ),
    );

    // --- groups --------------------------------------------------------------
    const groupsEl = section("Groups");
    const counts = new Map(
      (c.lastResult?.groups ?? []).map((g) => [g.id, g.count]),
    );
    for (const g of m.groups)
      groupsEl.append(this.renderGroup(g, counts.get(g.id)));
    groupsEl.append(
      h(
        "div",
        { class: "ngpy-row" },
        button("+ Group", () => m.addGroup()),
        field(
          "Ghost alpha",
          h("input", {
            type: "range",
            min: "0",
            max: "1",
            step: "0.05",
            value: String(m.settings.ghostAlpha),
            onchange: (e: Event) => {
              m.settings.ghostAlpha = Number(
                (e.target as HTMLInputElement).value,
              );
              m.dispatch();
            },
          }),
        ),
        h(
          "label",
          { class: "ngpy-check" },
          h("input", {
            type: "checkbox",
            checked: m.settings.colorByGroup,
            onchange: (e: Event) => {
              m.settings.colorByGroup = (e.target as HTMLInputElement).checked;
              m.dispatch();
            },
          }),
          "group colours",
        ),
      ),
    );
    this.body.append(groupsEl);
    this.body.append(this.renderLabels());
    this.body.append(this.renderAttributes());
    this.body.append(this.renderColorBy());
  }

  private renderGroup(g: Group, count: number | undefined): HTMLElement {
    const m = this.controller.model;
    const active = m.activeGroupId === g.id;
    const header = h(
      "div",
      { class: "ngpy-group-header" },
      h("input", {
        type: "radio",
        name: "ngpy-active-group",
        checked: active,
        title: "Active group: new ROIs join it",
        onchange: () => {
          m.activeGroupId = g.id;
          m.dispatch();
        },
      }),
      h("input", {
        type: "color",
        value: g.color,
        onchange: (e: Event) =>
          m.updateGroup(g.id, { color: (e.target as HTMLInputElement).value }),
      }),
      h("input", {
        type: "text",
        class: "ngpy-group-name",
        value: g.name,
        onchange: (e: Event) =>
          m.updateGroup(g.id, { name: (e.target as HTMLInputElement).value }),
      }),
      h(
        "label",
        { class: "ngpy-check", title: "Visible" },
        h("input", {
          type: "checkbox",
          checked: g.visible,
          onchange: (e: Event) =>
            m.updateGroup(g.id, {
              visible: (e.target as HTMLInputElement).checked,
            }),
        }),
        "visible",
      ),
      h(
        "span",
        { class: "ngpy-badge" },
        count === undefined ? "–" : count.toLocaleString(),
      ),
      button("↑", () => m.moveGroup(g.id, -1), {
        title: "Move up (first group wins the colour)",
      }),
      button("↓", () => m.moveGroup(g.id, 1)),
      button("✕", () => m.removeGroup(g.id), { title: "Delete group" }),
    );
    const list = h("div", { class: "ngpy-roi-list" });
    g.rois.forEach((roi, i) => list.append(this.renderRoi(g, roi, i)));
    if (g.rois.length === 0 && g.attrFilters.length === 0) {
      list.append(
        h(
          "div",
          { class: "ngpy-hint" },
          "No regions yet: draw one in the ROI layer.",
        ),
      );
    }
    g.attrFilters.forEach((f, i) =>
      list.append(
        h(
          "div",
          { class: "ngpy-roi" },
          h("span", { class: "ngpy-roi-kind" }, "attr"),
          `${f.name} ∈ [${f.min}, ${f.max}]`,
          button("✕", () => {
            g.attrFilters.splice(i, 1);
            m.dispatch();
          }),
        ),
      ),
    );
    return h(
      "div",
      { class: `ngpy-group${active ? " active" : ""}` },
      header,
      list,
    );
  }

  private renderRoi(g: Group, roi: GroupRoi, index: number): HTMLElement {
    const m = this.controller.model;
    const label =
      roi.kind === "labels"
        ? `labels ${roi.labels.slice(0, 6).join(", ")}${roi.labels.length > 6 ? "…" : ""}`
        : (roi.name ?? `ROI ${index + 1} (${roi.annotationId.slice(0, 6)})`);
    return h(
      "div",
      { class: "ngpy-roi" },
      h(
        "span",
        { class: "ngpy-roi-kind" },
        roi.kind === "labels" ? "label" : "shape",
      ),
      h("span", { class: "ngpy-roi-name" }, label),
      select(OPERATORS, roi.operator, (v) => {
        roi.operator = v as any;
        m.dispatch();
      }),
      select(PREDICATES, roi.predicate, (v) => {
        roi.predicate = v as any;
        m.dispatch();
      }),
      button("↑", () => m.moveRoi(g.id, index, -1), {
        title: "Earlier in the fold",
      }),
      button(
        "✕",
        () => {
          g.rois.splice(index, 1);
          m.dispatch();
        },
        {
          title:
            "Remove from group (the annotation stays; delete it in the viewer)",
        },
      ),
    );
  }

  private renderLabels(): HTMLElement {
    const c = this.controller;
    const m = c.model;
    const layers = c
      .segmentationLayers()
      .filter((n) => n !== m.settings.targetLayer);
    const parcel = select(
      [
        { value: "", label: "— none —" },
        ...layers.map((n) => ({ value: n, label: n })),
      ],
      m.settings.parcellationLayer ?? "",
      (v) => {
        m.settings.parcellationLayer = v || undefined;
        this.labelState.clear();
        void c.loadLabels().then(() => this.queueRender());
        m.dispatch();
      },
    );
    const out = section(
      "By segmentation label",
      field("Parcellation layer", parcel),
      button(
        "Reload labels",
        () => void c.loadLabels().then(() => this.queueRender()),
      ),
    );
    const info = c.labelInfo;
    if (info === undefined || m.settings.parcellationLayer === undefined)
      return out;
    const search = h("input", {
      type: "search",
      placeholder: "search labels",
      value: this.labelSearch,
      oninput: (e: Event) => {
        this.labelSearch = (e.target as HTMLInputElement).value;
        renderList();
      },
    });
    const listEl = h("div", { class: "ngpy-label-list" });
    const renderList = () => {
      clear(listEl);
      const q = this.labelSearch.toLowerCase();
      for (const l of info.labels) {
        if (q && !l.name.toLowerCase().includes(q) && !String(l.id).includes(q))
          continue;
        const state = this.labelState.get(l.id);
        const cycle = () => {
          const next: LabelState | undefined =
            state === undefined
              ? "include"
              : state === "include"
                ? "exclude"
                : undefined;
          if (next === undefined) this.labelState.delete(l.id);
          else this.labelState.set(l.id, next);
          this.updateLabelPreview();
          renderList();
        };
        listEl.append(
          h(
            "div",
            {
              class: `ngpy-label ${state ?? ""}`,
              onclick: cycle,
              title: "click: include → exclude → off",
            },
            h("span", {
              class: "ngpy-swatch",
              style: { background: l.color ?? "#888" },
            }),
            h("span", {}, `${l.name}`),
            h("span", { class: "ngpy-hint" }, ` ${l.id}`),
            h("span", { class: "ngpy-label-state" }, state ?? ""),
          ),
        );
      }
    };
    renderList();
    out.append(
      search,
      listEl,
      h(
        "div",
        { class: "ngpy-row" },
        button("Clear", () => {
          this.labelState.clear();
          this.updateLabelPreview();
          this.queueRender();
        }),
        button("Create group from selection", () => {
          const names = [...this.labelState.entries()]
            .filter(([, s]) => s === "include")
            .map(
              ([id]) =>
                info.labels.find((l) => l.id === id)?.name ?? String(id),
            );
          m.commitPreview(
            names.length ? names.slice(0, 3).join(" + ") : "Labels",
          );
          this.labelState.clear();
        }),
      ),
      h(
        "p",
        { class: "ngpy-hint" },
        "Passes tracts crossing ANY included label and NO excluded one. The selection previews live in white.",
      ),
    );
    return out;
  }

  private updateLabelPreview() {
    const m = this.controller.model;
    const include = [...this.labelState]
      .filter(([, s]) => s === "include")
      .map(([id]) => id);
    const exclude = [...this.labelState]
      .filter(([, s]) => s === "exclude")
      .map(([id]) => id);
    const rois: LabelRoi[] = [];
    if (include.length) {
      rois.push({
        kind: "labels",
        labels: include,
        operator: "and",
        predicate: "any_vertex",
      });
    }
    if (exclude.length) {
      rois.push({
        kind: "labels",
        labels: exclude,
        operator: "andnot",
        predicate: "any_vertex",
      });
    }
    m.setPreview(
      rois.length
        ? {
            name: "Label selection",
            color: "#ffffff",
            visible: true,
            opacity: 1,
            rois,
            attrFilters: [],
          }
        : undefined,
    );
  }

  private renderAttributes(): HTMLElement {
    const c = this.controller;
    const m = c.model;
    const info = c.storeInfo;
    const names: string[] = [];
    for (const a of info?.objectAttributes ?? []) {
      if (a.ncols === 1) names.push(a.name);
      else for (let i = 0; i < a.ncols; ++i) names.push(`${a.name}[${i}]`);
    }
    const draft = (this.attrDraft ??= { name: names[0] ?? "", min: 0, max: 1 });
    const nameInput = h("input", {
      type: "text",
      list: "ngpy-attr-names",
      value: draft.name,
      placeholder: "object attribute",
      onchange: (e: Event) =>
        (draft.name = (e.target as HTMLInputElement).value),
    });
    const dl = h(
      "datalist",
      { id: "ngpy-attr-names" },
      names.map((n) => h("option", { value: n })),
    );
    const num = (key: "min" | "max") =>
      h("input", {
        type: "number",
        step: "any",
        value: String(draft[key]),
        class: "ngpy-num",
        onchange: (e: Event) =>
          (draft[key] = Number((e.target as HTMLInputElement).value)),
      });
    return section(
      "By attribute",
      h(
        "div",
        { class: "ngpy-row" },
        nameInput,
        dl,
        "min",
        num("min"),
        "max",
        num("max"),
      ),
      h(
        "div",
        { class: "ngpy-row" },
        button("Preview", () =>
          m.setPreview({
            name: "Attribute selection",
            color: "#ffffff",
            visible: true,
            opacity: 1,
            rois: [],
            attrFilters: [{ ...draft }],
          }),
        ),
        button("Add to active group", () => {
          const g = m.activeGroup ?? m.addGroup();
          g.attrFilters.push({ ...draft });
          m.dispatch();
        }),
        button("Create group", () => {
          m.setPreview(undefined);
          m.addGroup({
            name: `${draft.name} ∈ [${draft.min}, ${draft.max}]`,
            attrFilters: [{ ...draft }],
          });
        }),
        button("Clear preview", () => m.setPreview(undefined)),
      ),
      h(
        "p",
        { class: "ngpy-hint" },
        info && names.length === 0
          ? "No object attributes were listed (the host cannot list); type a name from the store's object_attributes/."
          : "Per-object ranges, ANDed with the group's regions. A multi-column attribute is filtered per column: name[i].",
      ),
    );
  }

  private renderColorBy(): HTMLElement {
    const c = this.controller;
    const target = c.model.settings.targetLayer;
    const attrs = (c.storeInfo?.vertexAttributes ?? []).map((name) => ({
      name,
      components: name === "tangent" ? 3 : 1,
    }));
    const presets = colorPresets(attrs);
    const s = select(
      [
        { value: "", label: "— choose —" },
        ...presets.map((p) => ({ value: p.id, label: p.label })),
      ],
      "",
      (v) => {
        const p = presets.find((x) => x.id === v);
        if (p === undefined || target === undefined) return;
        setSkeletonShader(c.viewer, target, p.shader, p.controls);
      },
    );
    return section(
      "Colour by",
      field(
        "Skeleton shader",
        s,
        "Writes skeletonRendering.shader on the target layer. Group colours only show with “Segment / group colour”.",
      ),
    );
  }
}
