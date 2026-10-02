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

/** The wrapper's dissection model and annotation <-> store-frame geometry. */

import { describe, expect, it } from "vitest";
import { FilterModel, PREVIEW_COLOR } from "../../ngpy/src/filter/model.js";
import {
  annotationToShape,
  parseDimensions,
  shapeToAnnotation,
} from "../../ngpy/src/filter/roi_geometry.js";
import type { DimensionScales } from "../../ngpy/src/filter/roi_geometry.js";

// Global frame x,y,z in mm (the demo's pin); store in mm.
const MM: DimensionScales = {
  ...parseDimensions({ x: [0.001, "m"], y: [0.001, "m"], z: [0.001, "m"] }),
  storeAxes: ["x", "y", "z"],
  storeUnitM: 0.001,
};

const box = (id: string, a: number[], b: number[]) => ({
  type: "axis_aligned_bounding_box",
  id,
  pointA: a,
  pointB: b,
});

describe("roi geometry", () => {
  it("converts boxes and ellipsoids into the store frame", () => {
    expect(annotationToShape(box("a", [10, 0, 5], [0, 20, 1]), MM)).toEqual({
      type: "box",
      lower: [0, 0, 1],
      upper: [10, 20, 5],
    });
    expect(
      annotationToShape({ type: "ellipsoid", id: "e", center: [1, 2, 3], radii: [-4, 5, 6] }, MM),
    ).toEqual({ type: "ellipsoid", center: [1, 2, 3], radii: [4, 5, 6] });
    expect(annotationToShape({ type: "point", point: [1, 2, 3] }, MM)).toBeUndefined();
  });

  it("matches axes by name and converts units", () => {
    // Annotation dims z,y,x in micrometres; store x,y,z in mm.
    const scales: DimensionScales = {
      ...parseDimensions({ z: [1e-6, "m"], y: [1e-6, "m"], x: [1e-6, "m"] }),
      storeAxes: ["x", "y", "z"],
      storeUnitM: 0.001,
    };
    const shape = annotationToShape(box("a", [3000, 2000, 1000], [3000, 2000, 1000]), scales);
    expect(shape).toEqual({ type: "box", lower: [1, 2, 3], upper: [1, 2, 3] });
    // ...and back.
    const ann = shapeToAnnotation(shape!, scales, "id1");
    expect(ann.pointA).toEqual([3000, 2000, 1000]);
  });
});

describe("FilterModel", () => {
  it("assigns new annotations to the active group as includes and drops vanished ones", () => {
    const m = new FilterModel();
    expect(m.syncAnnotations(["a"])).toBe(true); // creates Group 1
    expect(m.groups).toHaveLength(1);
    const g2 = m.addGroup();
    m.syncAnnotations(["a", "b"]);
    expect(m.findAnnotation("b")!.group.id).toBe(g2.id);
    expect(m.findAnnotation("b")!.group.rois[0]).toMatchObject({ operator: "and", predicate: "any_segment" });
    expect(m.syncAnnotations(["a", "b"])).toBe(false);
    m.syncAnnotations(["b"]);
    expect(m.findAnnotation("a")).toBeUndefined();
    expect(m.groups[0].rois).toHaveLength(0);
  });

  it("emits groupToJson-compatible persistence JSON in the store frame", () => {
    const m = new FilterModel();
    const g = m.addGroup({ name: "CST", color: "#ff0000", opacity: 0.5 });
    m.syncAnnotations(["a", "b"]);
    g.rois[1].operator = "andnot";
    g.rois[1].predicate = "either_endpoint";
    g.rois.push({ kind: "labels", labels: [17, 53], operator: "and", predicate: "any_vertex" });
    g.attrFilters.push({ name: "tortuosity", min: 1, max: 2 });
    const annotations = new Map<string, any>([
      ["a", box("a", [0, 0, 0], [10, 10, 10])],
      ["b", { type: "ellipsoid", id: "b", center: [5, 5, 5], radii: [1, 2, 3] }],
    ]);
    expect(m.groupJson(g, annotations, MM)).toEqual({
      name: "CST",
      color: "#ff0000",
      opacity: 0.5,
      rois: [
        {
          shape: { type: "box", lower: [0, 0, 0], upper: [10, 10, 10] },
          predicate: "any_segment",
          operator: "and",
        },
        {
          shape: { type: "ellipsoid", center: [5, 5, 5], radii: [1, 2, 3] },
          predicate: "either_endpoint",
          operator: "andnot",
        },
        { shape: { type: "labelMask", labels: [17, 53] }, predicate: "any_vertex", operator: "and" },
      ],
      attrFilters: [{ name: "tortuosity", min: 1, max: 2 }],
    });
  });

  it("evaluates committed groups plus the white preview", () => {
    const m = new FilterModel();
    m.addGroup({ name: "g", visible: false });
    expect(m.isActive()).toBe(false);
    m.setPreview({
      name: "p",
      color: "#123456",
      visible: true,
      opacity: 1,
      rois: [{ kind: "labels", labels: [1], operator: "and", predicate: "any_vertex" }],
      attrFilters: [],
    });
    expect(m.isActive()).toBe(true);
    const groups = m.evaluationGroups(new Map(), MM);
    expect(groups.map((g) => [g.id, g.visible, g.color])).toEqual([
      [1, false, groups[0].color],
      [0, true, PREVIEW_COLOR],
    ]);
    const committed = m.commitPreview("Labels");
    expect(committed?.rois).toHaveLength(1);
    expect(m.preview).toBeUndefined();
  });

  it("round-trips through toJSON / restoreState", () => {
    const m = new FilterModel();
    m.settings.targetLayer = "tracts";
    m.settings.level = 2;
    m.addGroup({ name: "A" });
    m.syncAnnotations(["x"]);
    const json = JSON.parse(JSON.stringify(m.toJSON()));
    const n = new FilterModel();
    n.restoreState(json);
    expect(n.toJSON()).toEqual(m.toJSON());
    expect(n.addGroup().id).toBe(2);
  });
});
