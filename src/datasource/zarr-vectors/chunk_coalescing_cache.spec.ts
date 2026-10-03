/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
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
import { ChunkCoalescingCache } from "#src/datasource/zarr-vectors/chunk_coalescing_cache.js";

/** A loader that records how many times it actually ran. */
function countingLoader<T>(value: T) {
  const calls = { n: 0 };
  return {
    calls,
    load: async () => {
      ++calls.n;
      return value;
    },
  };
}

describe("ChunkCoalescingCache", () => {
  it("runs the loader once for concurrent requests of one key", async () => {
    // The case that matters: tracts in a dissection are spatially clustered, so
    // many of them ask for the same chunk at the same moment.
    const cache = new ChunkCoalescingCache<string>();
    const { calls, load } = countingLoader("chunk-a");
    const results = await Promise.all(
      Array.from({ length: 50 }, () => cache.get("a", load)),
    );
    expect(calls.n).toBe(1);
    expect(results.every((r) => r === "chunk-a")).toBe(true);
  });

  it("reuses a settled entry across separate waves", async () => {
    // The chunk queue admits pass-2 downloads in waves, so tracts sharing a
    // chunk often arrive in different ones; coalescing only in-flight work
    // would miss them.
    const cache = new ChunkCoalescingCache<string>();
    const { calls, load } = countingLoader("chunk-a");
    await cache.get("a", load);
    await cache.get("a", load);
    await cache.get("a", load);
    expect(calls.n).toBe(1);
  });

  it("keeps distinct keys distinct", async () => {
    const cache = new ChunkCoalescingCache<string>();
    expect(await cache.get("a", async () => "A")).toBe("A");
    expect(await cache.get("b", async () => "B")).toBe("B");
  });

  it("does not let one caller's abort cancel the shared work", async () => {
    // A shared entry carries no caller signal by design: honouring one waiter's
    // abort would cancel a decode the others are still waiting on, turning one
    // cancelled tract into a cascade of spurious failures.
    const cache = new ChunkCoalescingCache<string>();
    let argCount = -1;
    let resolveLoad: (v: string) => void = () => {};
    const gate = new Promise<string>((resolve) => {
      resolveLoad = resolve;
    });

    // Two waiters on the same in-flight entry.
    const first = cache.get("a", function (this: unknown, ...args: unknown[]) {
      argCount = args.length;
      return gate;
    });
    const second = cache.get("a", async () => "never-called");

    // The first caller gives up; the shared work must be unaffected.
    resolveLoad("chunk-a");
    expect(await first).toBe("chunk-a");
    expect(await second).toBe("chunk-a");
    // The loader is invoked with no arguments, so there is no signal to honour.
    expect(argCount).toBe(0);
  });

  it("evicts a rejected entry so the next caller retries", async () => {
    // A cached failure would otherwise pin a transient error for the life of
    // the entry.
    const cache = new ChunkCoalescingCache<string>();
    let attempt = 0;
    const flaky = async () => {
      if (++attempt === 1) throw new Error("transient");
      return "recovered";
    };
    await expect(cache.get("a", flaky)).rejects.toThrow("transient");
    expect(await cache.get("a", flaky)).toBe("recovered");
    expect(attempt).toBe(2);
  });

  it("bounds its size, evicting oldest first", async () => {
    const cache = new ChunkCoalescingCache<string>(3);
    for (const key of ["a", "b", "c", "d"]) {
      await cache.get(key, async () => key);
    }
    expect(cache.statistics.size).toBe(3);
    // "a" was evicted, so it reloads; "d" is still resident.
    const a = countingLoader("a");
    const d = countingLoader("d");
    await cache.get("a", a.load);
    await cache.get("d", d.load);
    expect(a.calls.n).toBe(1);
    expect(d.calls.n).toBe(0);
  });

  it("reports coalescing effectiveness", async () => {
    const cache = new ChunkCoalescingCache<string>();
    await cache.get("a", async () => "A");
    await cache.get("a", async () => "A");
    await cache.get("b", async () => "B");
    const { hits, misses } = cache.statistics;
    expect(misses).toBe(2);
    expect(hits).toBe(1);
  });
});
