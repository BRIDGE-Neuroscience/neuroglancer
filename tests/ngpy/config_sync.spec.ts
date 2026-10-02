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

/** Python's ConfigState applied to a (fake) hosted viewer. */

import { describe, expect, it } from "vitest";
import { ConfigSync } from "../../ngpy/src/host/config_sync.js";

/** Mimics EventActionMap's set/delete/clear/addParent. */
class FakeMap {
  bindings = new Map<string, string>();
  parents: [FakeMap, number][] = [];
  set(k: string, a: string) {
    this.bindings.set(k, a);
  }
  delete(k: string) {
    this.bindings.delete(k);
  }
  clear() {
    this.bindings.clear();
  }
  addParent(p: FakeMap, priority: number) {
    this.parents.push([p, priority]);
  }
}

function fakeViewer() {
  const element = new EventTarget();
  return {
    element,
    state: { toJSON: () => ({ layout: "3d" }) },
    mouseState: {
      updateUnconditionally: () => true,
      position: new Float32Array([1, 2, 3]),
    },
    layerSelectedValues: { toJSON: () => ({ tracts: { value: "5" } }) },
    inputEventBindings: {
      global: new FakeMap(),
      sliceView: new FakeMap(),
      perspectiveView: new FakeMap(),
    },
    uiConfiguration: { showLayerPanel: { value: true } },
  };
}

describe("ConfigSync", () => {
  it("binds actions, reports upstream's action state, and unbinds", () => {
    const viewer = fakeViewer();
    const sent: [string, any][] = [];
    const sync = new ConfigSync(viewer, {
      sendAction: (a, s) => sent.push([a, s]),
      setStatusMessages: () => {},
    });
    sync.apply({ actions: ["my-action", "screenshot"] });
    viewer.element.dispatchEvent(new CustomEvent("action:my-action"));
    viewer.element.dispatchEvent(new CustomEvent("action:screenshot"));
    expect(sent).toEqual([
      [
        "my-action",
        {
          mousePosition: [1, 2, 3],
          selectedValues: { tracts: { value: "5" } },
          viewerState: { layout: "3d" },
        },
      ],
    ]);
    sync.apply({ actions: [] });
    viewer.element.dispatchEvent(new CustomEvent("action:my-action"));
    expect(sent).toHaveLength(1);
  });

  it("adds key bindings as a priority-1000 parent map, replaced on each config", () => {
    const viewer = fakeViewer();
    const sync = new ConfigSync(viewer, {
      sendAction: () => {},
      setStatusMessages: () => {},
    });
    sync.apply({
      inputEventBindings: {
        viewer: { keyt: "my-action" },
        dataView: { "at:dblclick0": "pick" },
      },
    });
    const [child, priority] = viewer.inputEventBindings.global.parents[0];
    expect(priority).toBe(1000);
    expect([...child.bindings]).toEqual([["keyt", "my-action"]]);
    const dataView = viewer.inputEventBindings.sliceView.parents.find(
      ([, p]) => p === 999,
    )![0];
    expect(
      viewer.inputEventBindings.perspectiveView.parents.some(
        ([m]) => m === dataView,
      ),
    ).toBe(true);
    expect([...dataView.bindings]).toEqual([["at:dblclick0", "pick"]]);
    sync.apply({ inputEventBindings: { viewer: {} } });
    expect(child.bindings.size).toBe(0);
    expect(viewer.inputEventBindings.global.parents).toHaveLength(1);
  });

  it("forwards status messages and UI options", () => {
    const viewer = fakeViewer();
    let messages: Record<string, string> = {};
    const sync = new ConfigSync(viewer, {
      sendAction: () => {},
      setStatusMessages: (m) => (messages = m),
    });
    sync.apply({ statusMessages: { a: "hello" }, showLayerPanel: false });
    expect(messages).toEqual({ a: "hello" });
    expect(viewer.uiConfiguration.showLayerPanel.value).toBe(false);
  });
});
