/**
 * @license
 * Copyright 2026 Google Inc.
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

import { describe, expect, it } from "vitest";
import { levelDensities } from "#src/datasource/zarr-vectors/dense_lod.js";
import { rowSpans } from "#src/datasource/zarr-vectors/level_cells.js";

const level = (vertexCount: number | undefined, chunk: number) => ({
  vertexCount,
  chunkShape: [chunk, chunk, chunk],
});
const box = { lower: [0, 0, 0], upper: [100, 100, 100] };

describe("levelDensities", () => {
  it("never makes a finer level look sparser than a coarser one", () => {
    // A writer's placeholder count of 0 at level 0.
    const d = levelDensities(
      [level(0, 16), level(5e6, 32), level(1e6, 64)],
      box.lower,
      box.upper,
    );
    expect(d[0]).toBeGreaterThanOrEqual(d[1]);
    expect(d[1]).toBeGreaterThanOrEqual(d[2]);
    expect(d[0]).toBeCloseTo(8 * d[1]);
  });

  it("estimates an uncounted coarse level from the one below", () => {
    const d = levelDensities(
      [level(1e6, 16), level(undefined, 32), level(0, 32)],
      box.lower,
      box.upper,
    );
    expect(d[1]).toBeCloseTo(d[0] / 8);
    expect(d[2]).toBeCloseTo(d[1] / 2);
    expect(d.every((x) => Number.isFinite(x) && x > 0)).toBe(true);
  });

  it("gives a flat store a finite density", () => {
    const d = levelDensities(
      [level(4000, 16), level(1000, 32)],
      [0, 0, 0],
      [100, 100, 0],
    );
    expect(d.every((x) => Number.isFinite(x) && x > 0)).toBe(true);
    expect(d[0]).toBeCloseTo(4 * d[1]);
  });
});

describe("rowSpans", () => {
  it("joins the closest spans until few enough remain", () => {
    const rows = Array.from({ length: 41 }, (_, i) => i * 10_000 + (i % 3));
    const spans = rowSpans(rows, 4096, 16);
    expect(spans.length).toBe(16);
    for (let i = 1; i < spans.length; ++i) {
      expect(spans[i][0]).toBeGreaterThan(spans[i - 1][1]);
    }
    for (const r of rows) {
      expect(spans.some(([a, b]) => r >= a && r <= b)).toBe(true);
    }
  });

  it("keeps nearby rows in one span", () => {
    expect(rowSpans([1, 5, 9000, 9001], 4096, 16)).toEqual([
      [1, 5],
      [9000, 9001],
    ]);
  });
});
