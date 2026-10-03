/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  DEFAULT_STREAMLINE_FRAGMENT_MAIN,
  KIND_CAPABILITIES,
  hasObjectModel,
  type ZarrVectorsGeometryKind,
} from "#src/datasource/zarr-vectors/geometry_kind.js";

const hasSynthesisedTangent = (kind: ZarrVectorsGeometryKind) =>
  KIND_CAPABILITIES[kind].hasWalkOrderTangent ||
  KIND_CAPABILITIES[kind].hasEdgeAdjacencyTangent;
const isPointGeometry = (kind: ZarrVectorsGeometryKind) =>
  KIND_CAPABILITIES[kind].primitive === "points";
const isSurfaceGeometry = (kind: ZarrVectorsGeometryKind) =>
  KIND_CAPABILITIES[kind].primitive === "triangles";

const ALL_KINDS: readonly ZarrVectorsGeometryKind[] = [
  "point_cloud",
  "line",
  "streamline",
  "polyline",
  "skeleton",
  "graph",
  "mesh",
];

/** Every kind except the one with no connectivity. */
const CONNECTED_KINDS = ALL_KINDS.filter((k) => k !== "point_cloud");

/** Kinds whose links are vertex pairs drawn as line segments. */
const LINE_KINDS = CONNECTED_KINDS.filter((k) => k !== "mesh");

describe("KIND_CAPABILITIES table invariants", () => {
  it("has an entry for every declared kind", () => {
    for (const kind of ALL_KINDS) {
      expect(KIND_CAPABILITIES[kind]).toBeDefined();
    }
  });

  it("at most one tangent algorithm is enabled per kind", () => {
    // Walk-order and edge-adjacency are mutually exclusive: a kind
    // picks one source-of-truth for tangents.  Both true would be
    // ambiguous; both false is fine (skeletons).
    for (const kind of ALL_KINDS) {
      const c = KIND_CAPABILITIES[kind];
      expect(c.hasWalkOrderTangent && c.hasEdgeAdjacencyTangent).toBe(false);
    }
  });

  it("every line kind synthesises a tangent, so prop_tangent() is always available", () => {
    for (const kind of LINE_KINDS) {
      expect(hasSynthesisedTangent(kind)).toBe(true);
    }
  });

  it("auto-applies the RGB-by-tangent default only where direction is meaningful", () => {
    // Nominating the direction default is not free: it is
    // `emitRGB(abs(prop_tangent()))`, which computes the colour outright and
    // therefore consults neither the fixed segment colour nor the per-object
    // colour hash. A kind that nominates it renders colour-by-direction on
    // load and makes every colour control inert until the user edits the
    // shader. That is the right trade only where the tangent actually means
    // something.
    //
    // Walk-order kinds are directional curves -- the tangent follows the walk
    // and is consistent along it -- so tractography's colour-by-direction is
    // the expected default.
    for (const kind of ["line", "streamline", "polyline"] as const) {
      expect(KIND_CAPABILITIES[kind].hasWalkOrderTangent).toBe(true);
      expect(KIND_CAPABILITIES[kind].defaultFragmentMain).toBe(
        DEFAULT_STREAMLINE_FRAGMENT_MAIN,
      );
    }

    // `skeleton` is a branching tree, not a curve. Its tangents come from edge
    // adjacency, whose sign is arbitrary per vertex and which degree-0
    // vertices lack entirely, and its stores carry `object_attributes` the
    // user wants to colour by. It keeps the generic per-object default so the
    // colour controls work; "Direction" stays available as a preset.
    expect(KIND_CAPABILITIES.skeleton.hasEdgeAdjacencyTangent).toBe(true);
    expect(KIND_CAPABILITIES.skeleton.defaultFragmentMain).toBeUndefined();
  });

  it("point clouds nominate no default shader, having no tangent to read", () => {
    // `prop_tangent()` does not exist for a point cloud, so the
    // colour-by-direction default would not compile.
    expect(hasSynthesisedTangent("point_cloud")).toBe(false);
    expect(KIND_CAPABILITIES.point_cloud.defaultFragmentMain).toBeUndefined();
  });

  it("a surface has no direction, so no tangent and no direction default", () => {
    expect(hasSynthesisedTangent("mesh")).toBe(false);
    expect(KIND_CAPABILITIES.mesh.defaultFragmentMain).toBeUndefined();
    expect(isSurfaceGeometry("mesh")).toBe(true);
    for (const kind of LINE_KINDS) expect(isSurfaceGeometry(kind)).toBe(false);
    expect(isSurfaceGeometry("point_cloud")).toBe(false);
  });

  it("only point_cloud is drawn as points, and only it lacks the object model", () => {
    for (const kind of ALL_KINDS) {
      const isPoints = kind === "point_cloud";
      expect(isPointGeometry(kind)).toBe(isPoints);
      expect(hasObjectModel(kind)).toBe(!isPoints);
    }
  });

  it("only the no-connectivity kind suppresses edges", () => {
    expect(KIND_CAPABILITIES.point_cloud.edgeSource).toBe("none");
    for (const kind of CONNECTED_KINDS) {
      expect(KIND_CAPABILITIES[kind].edgeSource).not.toBe("none");
    }
  });

  it("streamline / polyline / graph / skeleton all synthesise tangents", () => {
    expect(hasSynthesisedTangent("streamline")).toBe(true);
    expect(hasSynthesisedTangent("polyline")).toBe(true);
    expect(hasSynthesisedTangent("graph")).toBe(true);
    // Skeletons opt into edge-adjacency tangents so shaders can
    // `prop_tangent()` (colour-by-direction).
    expect(hasSynthesisedTangent("skeleton")).toBe(true);
  });

  it("streamline / polyline use walk-order; graph + skeleton use edge-adjacency", () => {
    expect(KIND_CAPABILITIES.streamline.hasWalkOrderTangent).toBe(true);
    expect(KIND_CAPABILITIES.polyline.hasWalkOrderTangent).toBe(true);
    expect(KIND_CAPABILITIES.graph.hasEdgeAdjacencyTangent).toBe(true);
    expect(KIND_CAPABILITIES.skeleton.hasWalkOrderTangent).toBe(false);
    expect(KIND_CAPABILITIES.skeleton.hasEdgeAdjacencyTangent).toBe(true);
  });
});
