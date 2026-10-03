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
import { AsyncLru } from "#src/datasource/zarr-vectors/util.js";

describe("AsyncLru", () => {
  it("shares a load in flight even past its entry limit", async () => {
    const lru = new AsyncLru<number>(2);
    let loads = 0;
    const releases: (() => void)[] = [];
    const load = (value: number) => () =>
      new Promise<number>((resolve) => {
        ++loads;
        releases.push(() => resolve(value));
      });
    const first = lru.get("a", load(1));
    lru.get("b", load(2));
    lru.get("c", load(3));
    // "a" is still loading, so it must not have been evicted.
    expect(lru.get("a", load(99))).toBe(first);
    expect(loads).toBe(3);
    for (const release of releases) release();
    expect(await first).toBe(1);
  });

  it("evicts settled entries, oldest first, by size", async () => {
    const lru = new AsyncLru<Uint8Array>(Infinity, 10, (b) => b.byteLength);
    await lru.get("a", async () => new Uint8Array(6));
    await lru.get("b", async () => new Uint8Array(6));
    expect(lru.peek("a")).toBeUndefined();
    expect(lru.peek("b")).toBeDefined();
  });

  it("forgets a failed load", async () => {
    const lru = new AsyncLru<number>(4);
    await expect(
      lru.get("a", () => Promise.reject(new Error("x"))),
    ).rejects.toThrow();
    expect(await lru.get("a", async () => 5)).toBe(5);
  });
});
