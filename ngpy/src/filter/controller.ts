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
 * @file Wires the dissection model to the hosted viewer and to Python.
 *
 *   ROI annotation layer --(changed)--> model.syncAnnotations --+
 *   model.changed ---------------------------------------------+--> evaluate
 *   evaluate: groups (store frame) -> ngpy.api.filter_evaluate -> result
 *   result -> segment state -> applySegmentState(target layer)
 *
 * Evaluations are coalesced: at most one is in flight, and only the latest
 * pending request runs after it -- an ROI drag produces one evaluation per
 * settle, not one per frame.
 */

import type { FilterModel } from "./model.js";
import { PREVIEW_COLOR } from "./model.js";
import type { DimensionScales, RoiShapeJson } from "./roi_geometry.js";
import {
  parseDimensions,
  randomAnnotationId,
  shapeToAnnotation,
} from "./roi_geometry.js";
import type { FilterResult } from "./segment_state.js";
import { segmentStateFromResult } from "./segment_state.js";
import type { ApplyPath } from "../host/viewer_api.js";
import {
  annotationDimensions,
  applySegmentState,
  layerJson,
  layerSourceUrls,
  layerSpecs,
  readAnnotations,
  watchAnnotations,
  writeAnnotations,
  isZarrVectorsUrl,
  addLayer,
  stateJson,
} from "../host/viewer_api.js";
import type { PythonClient } from "../python/client.js";
import { Signal, debounce } from "../util/signal.js";

export const ROI_LAYER_NAME = "ngpy ROIs";

/** The annotation-layer spec ngpy creates for drawing ROIs. */
export function roiLayerSpec(dimensions: any, name = ROI_LAYER_NAME): any {
  return {
    type: "annotation",
    name,
    source: {
      url: "local://annotations",
      transform: { outputDimensions: dimensions },
    },
    tool: "annotateBoundingBox",
    annotationProperties: [
      { id: "color", type: "rgb", default: "#ffff00" },
      {
        id: "exclude",
        type: "uint8",
        default: 0,
        enum_values: [0, 1],
        enum_labels: ["include", "exclude"],
      },
    ],
    shader:
      "void main() {\n" +
      "  vec3 c = prop_color();\n" +
      "  setColor(prop_exclude() == 1u ? c * 0.45 : c);\n" +
      "}\n",
    annotations: [],
  };
}

export interface StoreInfo {
  base: string;
  units: string | null;
  unitMeters: number;
  axes: string[];
  levels: {
    level: number;
    vertexCount: number;
    objectSparsity: number;
    numObjects: number;
  }[];
  objectAttributes: { name: string; dtype: string; ncols: number }[];
  vertexAttributes: string[];
  defaultLevel: number;
  chunkShape: number[] | null;
}

export interface EvaluationResult extends FilterResult {
  level: number;
  levelObjects: number;
  levelVertices: number;
  storeObjects: number;
  objectSparsity: number;
  groups: { id: number; count: number }[];
  elapsedMs: number;
}

export interface LabelInfo {
  labels: { id: number; name: string; color: string | null }[];
  volume: string | null;
  volumeUrl: string | null;
  errors: string[];
}

export class FilterController {
  readonly statusChanged = new Signal<[string, string]>();
  readonly resultChanged = new Signal();
  readonly storeInfoChanged = new Signal();
  readonly layersChanged = new Signal();
  storeInfo: StoreInfo | undefined;
  storeInfoFor: string | undefined;
  lastResult: EvaluationResult | undefined;
  lastApplyPath: ApplyPath | undefined;
  labelInfo: LabelInfo | undefined;
  /** The managed target layer's source, once known. */
  private evaluating = false;
  private again = false;
  private unwatchRoi: (() => void) | undefined;
  private watchedRoiLayer: string | undefined;
  private lastAnnotationsJson = "";
  readonly schedule = debounce(() => void this.evaluate(), 120);

  constructor(
    readonly viewer: any,
    readonly python: PythonClient,
    readonly model: FilterModel,
  ) {
    model.changed.add(() => {
      this.ensureRoiWatch();
      this.colorRoiAnnotations();
      this.schedule();
    });
    const lm = viewer.layerManager;
    lm?.layersChanged?.add?.(
      debounce(() => {
        this.layersChanged.dispatch();
        this.ensureRoiWatch();
      }, 200),
    );
  }

  setStatus(text: string, kind = "") {
    this.statusChanged.dispatch(text, kind);
  }

  /** Segmentation layers whose source is a zarr-vectors store. */
  targetCandidates(): string[] {
    return layerSpecs(this.viewer)
      .filter(
        (l) =>
          (l.type === "segmentation" || l.type === undefined) &&
          layerSourceUrls(l).some(isZarrVectorsUrl),
      )
      .map((l) => l.name);
  }

  segmentationLayers(): string[] {
    return layerSpecs(this.viewer)
      .filter((l) => l.type === "segmentation")
      .map((l) => l.name);
  }

  annotationLayers(): string[] {
    return layerSpecs(this.viewer)
      .filter(
        (l) =>
          l.type === "annotation" &&
          layerSourceUrls(l).some((u) => u.startsWith("local://annotations")),
      )
      .map((l) => l.name);
  }

  sourceUrl(): string | undefined {
    const name = this.model.settings.targetLayer;
    if (name === undefined) return undefined;
    return layerSourceUrls(layerJson(this.viewer, name)).find(isZarrVectorsUrl);
  }

  async loadStoreInfo(): Promise<StoreInfo | undefined> {
    const source = this.sourceUrl();
    if (source === undefined) {
      this.storeInfo = undefined;
      this.storeInfoFor = undefined;
      this.storeInfoChanged.dispatch();
      return undefined;
    }
    if (this.storeInfoFor === source && this.storeInfo !== undefined)
      return this.storeInfo;
    this.setStatus("Reading store metadata…", "busy");
    try {
      this.storeInfo = await this.python.callJson<StoreInfo>("store_info", {
        source,
      });
      this.storeInfoFor = source;
      this.setStatus("", "");
    } catch (e) {
      this.storeInfo = undefined;
      this.setStatus(`Cannot read the store: ${(e as Error).message}`, "error");
    }
    this.storeInfoChanged.dispatch();
    return this.storeInfo;
  }

  /** Create the wrapper-owned ROI layer (one full-state restore). */
  createRoiLayer(): string {
    const dims = stateJson(this.viewer)?.dimensions ?? {};
    let name = ROI_LAYER_NAME;
    const existing = new Set(layerSpecs(this.viewer).map((l) => l.name));
    for (let i = 2; existing.has(name); ++i) name = `${ROI_LAYER_NAME} ${i}`;
    addLayer(this.viewer, roiLayerSpec(dims, name));
    this.model.settings.roiLayer = name;
    this.model.dispatch();
    return name;
  }

  private ensureRoiWatch() {
    const name = this.model.settings.roiLayer;
    if (name === this.watchedRoiLayer) return;
    this.unwatchRoi?.();
    this.unwatchRoi = undefined;
    this.watchedRoiLayer = name;
    if (name === undefined) return;
    this.unwatchRoi = watchAnnotations(this.viewer, name, () =>
      this.onAnnotationsChanged(),
    );
  }

  private onAnnotationsChanged() {
    const name = this.model.settings.roiLayer;
    if (name === undefined) return;
    const anns = readAnnotations(this.viewer, name).filter(
      (a) => a.type === "axis_aligned_bounding_box" || a.type === "ellipsoid",
    );
    const json = JSON.stringify(
      anns.map((a) => [a.id, a.pointA, a.pointB, a.center, a.radii]),
    );
    if (json === this.lastAnnotationsJson) return;
    this.lastAnnotationsJson = json;
    // syncAnnotations dispatches `changed` (and so schedules) when membership
    // changed; geometry-only edits need an explicit schedule.
    if (!this.model.syncAnnotations(anns.map((a) => String(a.id))))
      this.schedule();
  }

  /** The annotation-to-store coordinate mapping, or undefined before store info. */
  scales(): DimensionScales | undefined {
    const info = this.storeInfo;
    if (info === undefined) return undefined;
    const roiLayer = this.model.settings.roiLayer;
    const dims =
      roiLayer !== undefined
        ? annotationDimensions(this.viewer, roiLayer)
        : (stateJson(this.viewer)?.dimensions ?? {});
    const { names, scalesM } = parseDimensions(dims);
    return {
      names,
      scalesM,
      storeAxes: info.axes,
      storeUnitM: info.unitMeters,
    };
  }

  annotationsById(): Map<string, any> {
    const name = this.model.settings.roiLayer;
    const map = new Map<string, any>();
    if (name === undefined) return map;
    for (const a of readAnnotations(this.viewer, name))
      map.set(String(a.id), a);
    return map;
  }

  /** The engine request for the current model (store frame). */
  buildRequest(): any | undefined {
    const source = this.sourceUrl();
    const scales = this.scales();
    if (source === undefined || scales === undefined) return undefined;
    const groups = this.model.evaluationGroups(this.annotationsById(), scales);
    const request: any = {
      source,
      groups,
      level: this.model.settings.level,
    };
    if (this.labelInfo?.volumeUrl) {
      request.parcellation = { url: this.labelInfo.volumeUrl };
    }
    return request;
  }

  async evaluate(): Promise<void> {
    if (this.evaluating) {
      this.again = true;
      return;
    }
    this.evaluating = true;
    try {
      do {
        this.again = false;
        await this.evaluateOnce();
      } while (this.again);
    } finally {
      this.evaluating = false;
    }
  }

  private async evaluateOnce(): Promise<void> {
    const target = this.model.settings.targetLayer;
    if (target === undefined || this.python.status !== "ready") return;
    await this.loadStoreInfo();
    const request = this.buildRequest();
    if (request === undefined) return;
    let result: EvaluationResult | undefined;
    if (this.model.isActive()) {
      this.setStatus(
        this.lastResult === undefined
          ? "Reading the store and evaluating…"
          : "Evaluating…",
        "busy",
      );
      try {
        result = await this.python.callJson<EvaluationResult>(
          "filter_evaluate",
          request,
        );
      } catch (e) {
        this.setStatus(`Filter failed: ${(e as Error).message}`, "error");
        return;
      }
    }
    this.lastResult = result;
    const update = segmentStateFromResult(result, {
      ghostAlpha: this.model.settings.ghostAlpha,
      colorByGroup: this.model.settings.colorByGroup,
    });
    try {
      this.lastApplyPath = applySegmentState(this.viewer, target, update);
    } catch (e) {
      this.setStatus(
        `Could not update layer ${target}: ${(e as Error).message}`,
        "error",
      );
      return;
    }
    if (result === undefined) {
      this.setStatus("Filter inactive: all objects shown.", "");
    } else {
      const pct =
        result.storeObjects > 0
          ? (100 * result.levelObjects) / result.storeObjects
          : 100;
      this.setStatus(
        `${result.segments.length.toLocaleString()} of ${result.levelObjects.toLocaleString()} ` +
          `objects pass (level ${result.level}: ${pct.toFixed(pct < 1 ? 2 : 0)}% of ` +
          `${result.storeObjects.toLocaleString()}; ${result.elapsedMs} ms` +
          (this.lastApplyPath === "state" ? "; applied via full state" : "") +
          ")",
        "ok",
      );
    }
    this.resultChanged.dispatch();
  }

  /** Colour each ROI annotation by its group (needs ngpy's ROI layer properties). */
  private colorRoiAnnotations() {
    const name = this.model.settings.roiLayer;
    if (name === undefined) return;
    const spec = layerJson(this.viewer, name);
    const propIds: string[] = (spec?.annotationProperties ?? []).map(
      (p: any) => p.id,
    );
    const ci = propIds.indexOf("color");
    const ei = propIds.indexOf("exclude");
    if (ci < 0) return;
    const anns = readAnnotations(this.viewer, name);
    let changed = false;
    const updated = anns.map((a) => {
      const found = this.model.findAnnotation(String(a.id));
      if (found === undefined) return a;
      const roi = found.group.rois[found.index];
      const props = [...(a.props ?? propIds.map(() => 0))];
      const color = found.group.color;
      const exclude = roi.operator === "andnot" ? 1 : 0;
      if (props[ci] !== color || (ei >= 0 && props[ei] !== exclude)) {
        props[ci] = color;
        if (ei >= 0) props[ei] = exclude;
        changed = true;
        return { ...a, props };
      }
      return a;
    });
    if (changed) writeAnnotations(this.viewer, name, updated);
  }

  /** Label list for the parcellation layer (reads the volume in Python). */
  async loadLabels(): Promise<LabelInfo | undefined> {
    const name = this.model.settings.parcellationLayer;
    if (name === undefined) {
      this.labelInfo = undefined;
      return undefined;
    }
    this.setStatus("Reading the parcellation…", "busy");
    try {
      this.labelInfo = await this.python.callJson<LabelInfo>("label_info", {
        sources: layerSourceUrls(layerJson(this.viewer, name)),
      });
      const err = this.labelInfo.errors.join("; ");
      this.setStatus(
        this.labelInfo.volumeUrl
          ? `Parcellation: ${this.labelInfo.volume}; ${this.labelInfo.labels.length} labels`
          : `No readable parcellation volume${err ? `: ${err}` : ""}`,
        this.labelInfo.volumeUrl ? "ok" : "error",
      );
    } catch (e) {
      this.labelInfo = undefined;
      this.setStatus(
        `Cannot read the parcellation: ${(e as Error).message}`,
        "error",
      );
    }
    return this.labelInfo;
  }

  /**
   * Add a stored group (store-frame shapes) to the model; geometric ROIs become
   * annotations in the ROI layer, label masks stay in the group.
   */
  importGroup(groupJson: any): void {
    const scales = this.scales();
    let roiLayer = this.model.settings.roiLayer;
    const rois: any[] = [];
    const newAnnotations: any[] = [];
    for (const r of groupJson.rois ?? []) {
      const shape: RoiShapeJson = r.shape;
      if (shape?.type === "labelMask") {
        rois.push({
          kind: "labels",
          labels: [...shape.labels],
          operator: r.operator,
          predicate: r.predicate,
          name: r.name,
        });
        continue;
      }
      if (scales === undefined) throw new Error("Select a target layer first");
      const id = randomAnnotationId();
      const ann = shapeToAnnotation(shape, scales, id);
      if (ann === undefined) continue;
      newAnnotations.push(ann);
      rois.push({
        kind: "annotation",
        annotationId: id,
        operator: r.operator,
        predicate: r.predicate,
        name: r.name,
      });
    }
    if (newAnnotations.length > 0) {
      if (roiLayer === undefined) roiLayer = this.createRoiLayer();
      const existing = readAnnotations(this.viewer, roiLayer);
      // Register the group BEFORE the annotations appear, so they are not
      // claimed by the active group.
      this.model.addGroup({
        name: groupJson.name,
        color: groupJson.color,
        visible: groupJson.visible !== false,
        opacity: groupJson.opacity ?? 1,
        rois,
        attrFilters: groupJson.attrFilters ?? [],
      });
      writeAnnotations(this.viewer, roiLayer, [...existing, ...newAnnotations]);
    } else {
      this.model.addGroup({
        name: groupJson.name,
        color: groupJson.color,
        visible: groupJson.visible !== false,
        rois,
        attrFilters: groupJson.attrFilters ?? [],
      });
    }
  }

  /** Persistence JSON of one group, for saving to the ROI store. */
  exportGroupJson(groupId: number): any {
    const scales = this.scales();
    if (scales === undefined) throw new Error("Select a target layer first");
    const g = this.model.groups.find((x) => x.id === groupId);
    if (g === undefined) throw new Error("No such group");
    return this.model.groupJson(g, this.annotationsById(), scales);
  }

  previewColor(): string {
    return PREVIEW_COLOR;
  }
}
