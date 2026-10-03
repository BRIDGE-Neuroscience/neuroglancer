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
import {
  formatDelta,
  formatOffsets,
  intraOffsets,
  isIntra,
  linksGroupPath,
  linksHasPerm,
  linksPath,
  parseDelta,
  parseOffsets,
} from "#src/datasource/zarr-vectors/links_paths.js";

describe("formatDelta / parseDelta", () => {
  it("formats the signed convention", () => {
    expect(formatDelta(0)).toBe("0");
    expect(formatDelta(1)).toBe("+1");
    expect(formatDelta(-2)).toBe("-2");
  });

  it("round-trips", () => {
    for (const d of [0, 1, -1, 2, -3, 17]) {
      expect(parseDelta(formatDelta(d))).toBe(d);
    }
  });

  it("rejects malformed segments", () => {
    expect(() => parseDelta("")).toThrow();
    expect(() => parseDelta("1")).toThrow(); // positive must carry the leading +
    expect(() => parseDelta("+")).toThrow();
    expect(() => parseDelta("x")).toThrow();
  });
});

describe("formatOffsets / parseOffsets", () => {
  // The segments actually present on the real HCP1065 store, level 0.
  const REAL = ["0.0.+1", "0.0.-1", "0.+1.0", "+1.0.0", "-1.0.0", "0.+1.+1"];

  it("formats the store's face/edge neighbour offsets", () => {
    expect(formatOffsets([[0, 0, 1]])).toBe("0.0.+1");
    expect(formatOffsets([[0, 0, -1]])).toBe("0.0.-1");
    expect(formatOffsets([[-1, 0, 0]])).toBe("-1.0.0");
    expect(formatOffsets([[0, 1, 1]])).toBe("0.+1.+1");
  });

  it("round-trips every real segment (linkWidth=2, sidNdim=3)", () => {
    for (const seg of REAL) {
      const parsed = parseOffsets(seg, { sidNdim: 3, linkWidth: 2 });
      expect(parsed).toHaveLength(1);
      expect(parsed[0]).toHaveLength(3);
      expect(formatOffsets(parsed)).toBe(seg);
    }
  });

  it("handles a multi-offset (triangle) segment", () => {
    const seg = "0.0.+1_0.+1.0";
    const parsed = parseOffsets(seg, { sidNdim: 3, linkWidth: 3 });
    expect(parsed).toEqual([
      [0, 0, 1],
      [0, 1, 0],
    ]);
    expect(formatOffsets(parsed)).toBe(seg);
  });

  it("uses the 'self' segment for linkWidth 1", () => {
    expect(formatOffsets([])).toBe("self");
    expect(parseOffsets("self", { sidNdim: 3, linkWidth: 1 })).toEqual([]);
  });

  it("rejects arity mismatches (guards against decoding wrong geometry)", () => {
    // Too few offsets for the width.
    expect(() =>
      parseOffsets("0.0.+1", { sidNdim: 3, linkWidth: 3 }),
    ).toThrow();
    // Wrong component count for sidNdim.
    expect(() => parseOffsets("0.0", { sidNdim: 3, linkWidth: 2 })).toThrow();
    // 'self' with width > 1, and a real segment with width 1.
    expect(() => parseOffsets("self", { sidNdim: 3, linkWidth: 2 })).toThrow();
    expect(() => parseOffsets("0.0.0", { sidNdim: 3, linkWidth: 1 })).toThrow();
  });
});

describe("intraOffsets / isIntra", () => {
  it("builds the all-zero offsets for the width", () => {
    expect(intraOffsets(3, 2)).toEqual([[0, 0, 0]]);
    expect(intraOffsets(3, 3)).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    expect(intraOffsets(3, 1)).toEqual([]);
  });

  it("classifies intra vs cross-chunk", () => {
    expect(isIntra([[0, 0, 0]])).toBe(true);
    expect(isIntra([])).toBe(true); // width 1 is intra by definition
    expect(isIntra([[0, 0, 1]])).toBe(false);
    expect(isIntra([[-1, 0, 0]])).toBe(false);
  });

  it("formats the intra segment as the store's 0.0.0 (which is absent under implicit_sequential)", () => {
    expect(formatOffsets(intraOffsets(3, 2))).toBe("0.0.0");
  });
});

describe("linksGroupPath / linksPath", () => {
  it("builds the family group and array paths", () => {
    expect(linksGroupPath(0)).toBe("links/0");
    expect(linksGroupPath(1)).toBe("links/+1");
    expect(linksPath(0, [[0, 0, 1]])).toBe("links/0/0.0.+1");
    expect(linksPath(0, intraOffsets(3, 2))).toBe("links/0/0.0.0");
    expect(linksPath(-1, [[0, 0, 0]])).toBe("links/-1/0.0.0");
  });
});

describe("linksHasPerm", () => {
  const policy = { delta: 0, directed: true, store: "canonical" };

  it("is false for the store's directed-canonical cross-chunk arrays", () => {
    // HCP1065: directed=true, store=canonical, delta=0 -> no perm column.
    expect(linksHasPerm([[0, 0, 1]], policy)).toBe(false);
  });

  it("is false for intra-chunk regardless of policy", () => {
    expect(
      linksHasPerm([[0, 0, 0]], {
        delta: 0,
        directed: false,
        store: "canonical",
      }),
    ).toBe(false);
    expect(
      linksHasPerm([[0, 0, 0]], {
        delta: 0,
        directed: false,
        store: "duplicate",
      }),
    ).toBe(false);
  });

  it("is false for cross-level regardless of policy", () => {
    expect(
      linksHasPerm([[0, 0, 1]], {
        delta: 1,
        directed: false,
        store: "duplicate",
      }),
    ).toBe(false);
  });

  it("is true for undirected same-level canonical (a sort happened)", () => {
    expect(
      linksHasPerm([[0, 0, 1]], {
        delta: 0,
        directed: false,
        store: "canonical",
      }),
    ).toBe(true);
  });

  it("is true for duplicate same-level even when directed", () => {
    expect(
      linksHasPerm([[0, 0, 1]], {
        delta: 0,
        directed: true,
        store: "duplicate",
      }),
    ).toBe(true);
  });
});
