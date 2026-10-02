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
 * The Export tab's job spec.  CONTRACT: `VALID` below is the `VALID` fixture
 * of `ngpy/python/tests/test_tract_export_job.py`; the two suites pin the same
 * literal so the TS builder and the Python parser cannot drift apart.
 */

import { describe, expect, it } from "vitest";
import {
  buildJobSpec,
  defaultAffine,
  exportFileName,
  JOB_SCHEMA_VERSION,
  parseAffineText,
} from "../../ngpy/src/export/spec.js";

const VALID = {
  schemaVersion: 3,
  source: { url: "zarr-vectors://gs://bucket/tracts.zvf", level: 0 },
  groups: [
    {
      name: "Motor CST",
      color: "#ff0000",
      rois: [
        {
          shape: { type: "ellipsoid", center: [10, 20, 30], radii: [5, 5, 5] },
          predicate: "any_segment",
          operator: "and",
          name: "seed",
        },
        {
          shape: { type: "box", lower: [0, 0, 0], upper: [100, 100, 100] },
          predicate: "either_endpoint",
          operator: "andnot",
        },
      ],
      objectIds: ["7", "42", "9007199254740993"],
    },
  ],
  format: "trk",
  destination: { kind: "download", path: "out.trk" },
};

describe("buildJobSpec", () => {
  it("produces the fixture shared with the Python parser", () => {
    const spec = buildJobSpec({
      sourceUrl: VALID.source.url,
      level: 0,
      format: "trk",
      scope: "selected",
      groups: VALID.groups,
      fileName: "out",
      destination: "download",
    });
    expect(spec).toEqual({ ...VALID, scope: "selected" });
    expect(spec.schemaVersion).toBe(JOB_SCHEMA_VERSION);
  });

  it("a whole-store export carries no groups; an affine rides along", () => {
    const spec = buildJobSpec({
      sourceUrl: "s",
      level: 2,
      format: "zvf",
      scope: "whole",
      groups: VALID.groups,
      affine: defaultAffine(1e-6),
      fileName: "x.zvf.zip",
      destination: "gcs",
    });
    expect(spec.groups).toEqual([]);
    expect(spec.affine[0][0]).toBeCloseTo(1e-3);
    expect(spec.destination).toEqual({ kind: "gcs", path: "x.zvf.zip" });
  });
});

describe("affine and file names", () => {
  it("parses 16 numbers, blank as identity, and rejects the rest", () => {
    expect(parseAffineText("")).toBeUndefined();
    expect(parseAffineText("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1")).toEqual(defaultAffine(1e-3));
    expect(() => parseAffineText("1 2 3")).toThrow();
  });

  it("normalises extensions", () => {
    expect(exportFileName("a.trk", "trk")).toBe("a.trk");
    expect(exportFileName("a.zvf.zip", "zvf")).toBe("a.zvf.zip");
    expect(exportFileName("a", "zvf")).toBe("a.zvf.zip");
    expect(exportFileName("  ", "trk")).toBe("dissection.trk");
  });
});
