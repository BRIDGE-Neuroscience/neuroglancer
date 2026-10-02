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
 * The page half of the Python <-> viewer state protocol, against a fake
 * trackable and a transport that implements the Python half's semantics
 * (`ngpy.bridge.handle_client_state`: generation check -> 200 / 412).
 */

import { describe, expect, it } from "vitest";
import type {
  SetStateMessage,
  TrackableLike,
} from "../../ngpy/src/host/state_sync.js";
import { SharedStateSync } from "../../ngpy/src/host/state_sync.js";

class FakeTrackable implements TrackableLike {
  value: any = {};
  private handlers: (() => void)[] = [];
  changed = {
    count: 0,
    add: (h: () => void) => {
      this.handlers.push(h);
      return () => {};
    },
  };
  restores = 0;
  resets = 0;
  set(value: any) {
    this.value = value;
    this.changed.count++;
    for (const h of this.handlers) h();
  }
  toJSON() {
    return this.value;
  }
  restoreState(value: any) {
    ++this.restores;
    this.set(structuredClone(value));
  }
  reset() {
    ++this.resets;
    this.set({});
  }
}

/** The Python side: one TrackableState generation, upstream's set_state check. */
class FakePython {
  generation = "py-0";
  state: any = undefined;
  messages: SetStateMessage[] = [];
  transport = async (msg: SetStateMessage) => {
    this.messages.push(msg);
    if (this.state !== undefined && msg.pg !== this.generation) {
      return { status: 412 };
    }
    this.state = msg.s;
    this.generation = `${msg.c}/${msg.g}`;
    return { status: 200, g: this.generation };
  };
}

function setup() {
  const t = new FakeTrackable();
  const py = new FakePython();
  const sync = new SharedStateSync(t, py.transport, 0, "cid");
  return { t, py, sync };
}

describe("SharedStateSync", () => {
  it("sends nothing before it is enabled, then the current state", async () => {
    const { t, py, sync } = setup();
    t.set({ layout: "xy" });
    await new Promise((r) => setTimeout(r, 5));
    expect(py.messages).toHaveLength(0);
    await sync.enable();
    expect(py.messages).toEqual([
      { s: { layout: "xy" }, g: 1, pg: "", c: "cid" },
    ]);
    expect(sync.lastServerGeneration).toBe("cid/1");
  });

  it("quotes the last server generation and skips an unchanged state", async () => {
    const { t, py, sync } = setup();
    await sync.enable();
    t.set({ layout: "3d" });
    await sync.flush();
    expect(py.messages.at(-1)).toMatchObject({
      s: { layout: "3d" },
      pg: "cid/0",
    });
    const n = py.messages.length;
    t.set({ layout: "3d" }); // same JSON, new generation
    await sync.flush();
    expect(py.messages).toHaveLength(n);
  });

  it("applies a Python push with reset+restore and does not echo it", async () => {
    const { t, py, sync } = setup();
    await sync.enable();
    const before = py.messages.length;
    sync.setServerState({ layers: [{ name: "a" }] }, "py-7");
    expect(t.resets).toBe(1);
    expect(t.restores).toBe(1);
    expect(t.value).toEqual({ layers: [{ name: "a" }] });
    await sync.flush();
    expect(py.messages).toHaveLength(before);
    expect(sync.lastServerGeneration).toBe("py-7");
  });

  it("leaves the state alone on 412 (Python's push will win)", async () => {
    const { t, py, sync } = setup();
    await sync.enable();
    py.generation = "py-moved"; // Python edited concurrently
    t.set({ layout: "xz" });
    await sync.flush();
    expect(py.messages.at(-1)).toMatchObject({ pg: "cid/0" });
    expect(sync.lastServerGeneration).toBe("cid/0");
    // Python's push arrives; the next page edit quotes it and succeeds.
    sync.setServerState({ layout: "4panel" }, "py-moved");
    t.set({ layout: "yz" });
    await sync.flush();
    expect(sync.lastServerGeneration).toBe(`cid/${t.changed.count}`);
  });

  it("adopts the generation a newly attached Python viewer reports", async () => {
    const { t, py, sync } = setup();
    await sync.enable();
    sync.setServerGeneration("cid/0");
    t.set({ layout: "3d" });
    await sync.flush();
    expect(py.messages.at(-1)!.pg).toBe("cid/0");
  });

  it("writes uint64 ids as strings", async () => {
    const { t, py, sync } = setup();
    t.set({ segments: [12345678901234567890n] });
    await sync.enable();
    expect(py.messages[0].s).toEqual({ segments: ["12345678901234567890"] });
  });
});
