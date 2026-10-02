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
 * @file The wrapper's dissection model: groups of ROIs, labels and attribute
 * ranges -- a rewrite of the old `RoiFilterState` that no longer lives in a
 * Neuroglancer layer.
 *
 * ROI GEOMETRY lives in a wrapper-owned Neuroglancer annotation layer (boxes
 * and ellipsoids drawn with Neuroglancer's own tools); this model only records,
 * per annotation id, which group the ROI belongs to and how it folds
 * (operator, predicate).  An annotation the model has not seen yet is assigned
 * to the active group as an include.  Label regions and attribute ranges have
 * no geometry and live in the group directly.
 *
 * `groupJson` emits the group PERSISTENCE form the old viewer used
 * (`groupToJson`: shapes keyed by `type`, string predicate / operator names),
 * which is what the Python engine, the export job spec and saved ROI-store
 * documents all read -- one serialisation for all three.
 */

import { Signal } from "../util/signal.js";
import type { DimensionScales, RoiShapeJson } from "./roi_geometry.js";
import { annotationToShape } from "./roi_geometry.js";

export type RoiOperator = "and" | "or" | "andnot";
export type RoiPredicate =
  | "any_segment"
  | "any_vertex"
  | "either_endpoint"
  | "both_endpoints";

export const OPERATORS: { value: RoiOperator; label: string }[] = [
  { value: "and", label: "include (AND)" },
  { value: "or", label: "or (OR)" },
  { value: "andnot", label: "exclude (NOT)" },
];

export const PREDICATES: { value: RoiPredicate; label: string }[] = [
  { value: "any_segment", label: "any segment" },
  { value: "any_vertex", label: "any vertex" },
  { value: "either_endpoint", label: "either endpoint" },
  { value: "both_endpoints", label: "both endpoints" },
];

export interface AnnotationRoi {
  kind: "annotation";
  annotationId: string;
  operator: RoiOperator;
  predicate: RoiPredicate;
  name?: string;
}

export interface LabelRoi {
  kind: "labels";
  labels: number[];
  operator: RoiOperator;
  predicate: RoiPredicate;
  name?: string;
}

export type GroupRoi = AnnotationRoi | LabelRoi;

export interface AttrFilter {
  name: string;
  min: number;
  max: number;
  scope?: "object" | "vertex";
}

export interface Group {
  id: number;
  name: string;
  color: string;
  visible: boolean;
  opacity: number;
  rois: GroupRoi[];
  attrFilters: AttrFilter[];
}

export const GROUP_PALETTE = [
  "#ff3b30",
  "#34c759",
  "#0a84ff",
  "#ffcc00",
  "#af52de",
  "#ff9500",
  "#5ac8fa",
  "#ff2d55",
];

export const PREVIEW_COLOR = "#ffffff";
export const DEFAULT_GHOST_ALPHA = 0.3;

export interface FilterSettings {
  targetLayer?: string;
  roiLayer?: string;
  parcellationLayer?: string;
  /** Pyramid level to evaluate; undefined = auto (vertex budget). */
  level?: number;
  ghostAlpha: number;
  colorByGroup: boolean;
}

export class FilterModel {
  readonly changed = new Signal();
  groups: Group[] = [];
  /** Staging dissection (label / attribute panels): evaluated, not persisted. */
  preview: Group | undefined;
  activeGroupId: number | undefined;
  settings: FilterSettings = {
    ghostAlpha: DEFAULT_GHOST_ALPHA,
    colorByGroup: true,
  };
  private nextId = 1;

  dispatch() {
    this.changed.dispatch();
  }

  get activeGroup(): Group | undefined {
    return this.groups.find((g) => g.id === this.activeGroupId);
  }

  addGroup(init: Partial<Omit<Group, "id">> = {}): Group {
    const group: Group = {
      id: this.nextId++,
      name: init.name ?? `Group ${this.groups.length + 1}`,
      color: init.color ?? GROUP_PALETTE[this.groups.length % GROUP_PALETTE.length],
      visible: init.visible ?? true,
      opacity: init.opacity ?? 1,
      rois: init.rois ?? [],
      attrFilters: init.attrFilters ?? [],
    };
    this.groups.push(group);
    this.activeGroupId = group.id;
    this.dispatch();
    return group;
  }

  removeGroup(id: number) {
    this.groups = this.groups.filter((g) => g.id !== id);
    if (this.activeGroupId === id) this.activeGroupId = this.groups.at(-1)?.id;
    this.dispatch();
  }

  updateGroup(id: number, changes: Partial<Omit<Group, "id">>) {
    const g = this.groups.find((x) => x.id === id);
    if (g === undefined) return;
    Object.assign(g, changes);
    this.dispatch();
  }

  moveGroup(id: number, delta: number) {
    const i = this.groups.findIndex((g) => g.id === id);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= this.groups.length) return;
    const [g] = this.groups.splice(i, 1);
    this.groups.splice(j, 0, g);
    this.dispatch();
  }

  moveRoi(groupId: number, index: number, delta: number) {
    const g = this.groups.find((x) => x.id === groupId);
    if (g === undefined) return;
    const j = index + delta;
    if (index < 0 || j < 0 || j >= g.rois.length) return;
    const [r] = g.rois.splice(index, 1);
    g.rois.splice(j, 0, r);
    this.dispatch();
  }

  /** The group (and index) an annotation ROI belongs to. */
  findAnnotation(id: string): { group: Group; index: number } | undefined {
    for (const group of this.groups) {
      const index = group.rois.findIndex(
        (r) => r.kind === "annotation" && r.annotationId === id,
      );
      if (index >= 0) return { group, index };
    }
    return undefined;
  }

  /**
   * Reconcile with the ROI layer's current annotation ids: unknown ids join
   * the active group (creating one if there is none); vanished ids leave their
   * group.  Returns whether anything changed (the caller re-evaluates anyway
   * when geometry moved).
   */
  syncAnnotations(ids: readonly string[]): boolean {
    const present = new Set(ids);
    let changed = false;
    for (const group of this.groups) {
      const before = group.rois.length;
      group.rois = group.rois.filter(
        (r) => r.kind !== "annotation" || present.has(r.annotationId),
      );
      changed ||= group.rois.length !== before;
    }
    for (const id of ids) {
      if (this.findAnnotation(id) !== undefined) continue;
      let group = this.activeGroup;
      if (group === undefined) {
        group = this.addGroup();
      }
      group.rois.push({
        kind: "annotation",
        annotationId: id,
        operator: "and",
        predicate: "any_segment",
      });
      changed = true;
    }
    if (changed) this.dispatch();
    return changed;
  }

  /** Whether any visible group (or the preview) selects something. */
  isActive(): boolean {
    const selects = (g: Group) => g.rois.length > 0 || g.attrFilters.length > 0;
    return (
      (this.preview !== undefined && selects(this.preview)) ||
      this.groups.some((g) => g.visible && selects(g))
    );
  }

  setPreview(preview: Omit<Group, "id"> | undefined) {
    this.preview = preview === undefined ? undefined : { ...preview, id: 0 };
    this.dispatch();
  }

  commitPreview(name?: string): Group | undefined {
    const p = this.preview;
    if (p === undefined || (p.rois.length === 0 && p.attrFilters.length === 0)) {
      return undefined;
    }
    this.preview = undefined;
    return this.addGroup({
      name: name ?? p.name,
      rois: p.rois,
      attrFilters: p.attrFilters,
    });
  }

  /** Persistence JSON for one group (`groupToJson`-compatible). */
  groupJson(
    group: Group,
    annotations: ReadonlyMap<string, any>,
    scales: DimensionScales,
  ): any {
    const rois: any[] = [];
    for (const roi of group.rois) {
      let shape: RoiShapeJson | undefined;
      if (roi.kind === "labels") {
        shape = { type: "labelMask", labels: [...roi.labels] };
      } else {
        const ann = annotations.get(roi.annotationId);
        if (ann === undefined) continue;
        shape = annotationToShape(ann, scales);
        if (shape === undefined) continue;
      }
      const entry: any = { shape, predicate: roi.predicate, operator: roi.operator };
      if (roi.name) entry.name = roi.name;
      rois.push(entry);
    }
    const json: any = { name: group.name, color: group.color, rois };
    if (!group.visible) json.visible = false;
    if (group.opacity !== 1) json.opacity = group.opacity;
    if (group.attrFilters.length > 0) {
      json.attrFilters = group.attrFilters.map((f) => ({ ...f }));
    }
    return json;
  }

  /** Every group to evaluate (committed + preview), as engine requests. */
  evaluationGroups(
    annotations: ReadonlyMap<string, any>,
    scales: DimensionScales,
  ): any[] {
    const out = this.groups.map((g) => ({
      id: g.id,
      ...this.groupJson(g, annotations, scales),
      visible: g.visible,
    }));
    if (this.preview !== undefined) {
      out.push({
        id: 0,
        ...this.groupJson(
          { ...this.preview, color: PREVIEW_COLOR },
          annotations,
          scales,
        ),
        visible: true,
      });
    }
    return out;
  }

  /** The model as persisted in the wrapper's own URL state (not geometry). */
  toJSON(): any {
    return {
      settings: { ...this.settings },
      groups: this.groups.map((g) => ({ ...g, rois: g.rois.map((r) => ({ ...r })) })),
      activeGroupId: this.activeGroupId,
    };
  }

  restoreState(json: any) {
    if (json === undefined || json === null || typeof json !== "object") return;
    this.settings = {
      ghostAlpha: DEFAULT_GHOST_ALPHA,
      colorByGroup: true,
      ...(json.settings ?? {}),
    };
    this.groups = Array.isArray(json.groups)
      ? json.groups.map((g: any) => ({
          id: Number(g.id),
          name: String(g.name ?? "Group"),
          color: String(g.color ?? GROUP_PALETTE[0]),
          visible: g.visible !== false,
          opacity: Number(g.opacity ?? 1),
          rois: Array.isArray(g.rois) ? g.rois : [],
          attrFilters: Array.isArray(g.attrFilters) ? g.attrFilters : [],
        }))
      : [];
    this.nextId = Math.max(0, ...this.groups.map((g) => g.id)) + 1;
    this.activeGroupId =
      json.activeGroupId ?? this.groups.at(-1)?.id ?? undefined;
    this.dispatch();
  }
}
