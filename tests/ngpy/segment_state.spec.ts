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

/** Filter result -> segmentation-layer state (the state contract). */

import { describe, expect, it } from "vitest";
import {
  applyTargeted,
  applyToViewerStateJson,
  compareIds,
  readLayerSegmentState,
  segmentStateFromResult,
} from "../../ngpy/src/filter/segment_state.js";

const OPTS = { ghostAlpha: 0.3, colorByGroup: true };

describe("segmentStateFromResult", () => {
  it("an inactive filter shows everything", () => {
    expect(segmentStateFromResult(undefined, OPTS)).toEqual({
      segments: [],
      segmentColors: {},
      notSelectedAlpha: 0.3,
      ignoreNullVisibleSet: true,
    });
    expect(
      segmentStateFromResult(
        { active: false, segments: ["1"], colors: {} },
        OPTS,
      ).ignoreNullVisibleSet,
    ).toBe(true);
  });

  it("an active filter that passes nothing shows nothing", () => {
    const u = segmentStateFromResult(
      { active: true, segments: [], colors: {} },
      OPTS,
    );
    expect(u.segments).toEqual([]);
    expect(u.ignoreNullVisibleSet).toBe(false);
  });

  it("writes the passing ids in uint64 order with their group colours", () => {
    const u = segmentStateFromResult(
      {
        active: true,
        segments: ["10", "9", "18446744073709551615", "100"],
        colors: { "9": "#FF0000", "10": "0f0", "100": "#00ff00" },
      },
      { ghostAlpha: 1.7, colorByGroup: true },
    );
    expect(u.segments).toEqual(["9", "10", "100", "18446744073709551615"]);
    expect(u.segmentColors).toEqual({
      "9": "#ff0000",
      "10": "#00ff00",
      "100": "#00ff00",
    });
    expect(u.notSelectedAlpha).toBe(1);
  });

  it("colorByGroup=false leaves colours alone", () => {
    const u = segmentStateFromResult(
      { active: true, segments: ["1"], colors: { "1": "#123456" } },
      { ghostAlpha: 0, colorByGroup: false },
    );
    expect(u.segmentColors).toEqual({});
  });

  it("compareIds orders decimal uint64 strings numerically", () => {
    expect(["20", "3", "100"].sort(compareIds)).toEqual(["3", "20", "100"]);
  });
});

describe("applying the update", () => {
  const update = segmentStateFromResult(
    {
      active: true,
      segments: ["5", "7"],
      colors: { "5": "#ff0000", "7": "#00ff00" },
    },
    OPTS,
  );

  it("patches only the named layer's JSON (fallback path)", () => {
    const state = {
      layers: [
        {
          name: "tracts",
          type: "segmentation",
          segments: ["1"],
          segmentColors: { "1": "#fff" },
          shader: "x",
        },
        { name: "other", type: "image" },
      ],
      layout: "3d",
    };
    const out = applyToViewerStateJson(state, "tracts", update);
    expect(out.layers[0]).toEqual({
      name: "tracts",
      type: "segmentation",
      shader: "x",
      segments: ["5", "7"],
      segmentColors: { "5": "#ff0000", "7": "#00ff00" },
      notSelectedAlpha: 0.3,
      ignoreNullVisibleSet: false,
    });
    expect(out.layers[1]).toBe(state.layers[1]);
    expect(readLayerSegmentState(out.layers[0])).toEqual(update);
    expect(() => applyToViewerStateJson(state, "missing", update)).toThrow();
  });

  /** A fake of the segmentation layer's trackables (bigint API). */
  function fakeLayer() {
    const visible = new Set<bigint>();
    const selected = new Set<bigint>();
    const colors = new Map<bigint, string>();
    const calls: string[] = [];
    const layer = {
      displayState: {
        segmentationGroupState: {
          value: {
            visibleSegments: {
              clear: () => visible.clear(),
              add: (x: bigint[]) => {
                calls.push(`add ${x.length}`);
                x.forEach((v) => visible.add(v));
              },
            },
            selectedSegments: { clear: () => selected.clear() },
            restoreState: (_: any) => calls.push("group.restoreState"),
          },
        },
        segmentationColorGroupState: {
          value: {
            segmentStatedColors: { clear: () => colors.clear() },
            restoreState: (spec: any) => {
              for (const [k, v] of Object.entries(spec.segmentColors))
                colors.set(BigInt(k), v as string);
            },
          },
        },
        notSelectedAlpha: { value: 0 },
        ignoreNullVisibleSet: { value: true },
      },
    };
    return { layer, visible, colors, calls };
  }

  it("mutates the layer's trackables with ONE bulk add (targeted path)", () => {
    const { layer, visible, colors, calls } = fakeLayer();
    visible.add(99n);
    expect(applyTargeted(layer, update)).toBe(true);
    expect([...visible]).toEqual([5n, 7n]);
    expect(calls).toEqual(["add 2"]);
    expect(colors.get(5n)).toBe("#ff0000");
    expect(layer.displayState.notSelectedAlpha.value).toBe(0.3);
    expect(layer.displayState.ignoreNullVisibleSet.value).toBe(false);
  });

  it("falls back to restoreState when the bulk bigint add is unavailable", () => {
    const { layer, calls } = fakeLayer();
    layer.displayState.segmentationGroupState.value.visibleSegments.add =
      () => {
        throw new TypeError("old Uint64 API");
      };
    expect(applyTargeted(layer, update)).toBe(true);
    expect(calls).toEqual(["group.restoreState"]);
  });

  it("reports false (changing nothing) on a layer without the surfaces", () => {
    expect(applyTargeted({ displayState: {} }, update)).toBe(false);
    expect(applyTargeted(undefined, update)).toBe(false);
  });
});
