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

import { describe, expect, test } from "vitest";
import { ByteRangeReadCoalescer } from "#src/kvstore/coalesce_reads.js";
import type { ByteRange } from "#src/kvstore/index.js";

// An object whose byte i is i % 251, read with a log of each request.
function setup(maxGap?: number, maxLength?: number) {
  const requests: { key: string; range: ByteRange; signal?: AbortSignal }[] =
    [];
  const gates: (() => void)[] = [];
  const coalescer = new ByteRangeReadCoalescer<string>(
    async (key, range, signal) => {
      requests.push({ key, range, signal });
      await new Promise<void>((resolve) => gates.push(resolve));
      signal?.throwIfAborted();
      const out = new Uint8Array(range.length);
      for (let i = 0; i < range.length; ++i) out[i] = (range.offset + i) % 251;
      return out;
    },
    (key) => key,
    maxGap,
    maxLength,
  );
  const open = async () => {
    await new Promise((resolve) => setTimeout(resolve, 5));
    for (const g of gates.splice(0)) g();
  };
  return { coalescer, requests, open };
}

const expectBytes = (data: Uint8Array, offset: number, length: number) => {
  expect(data.length).toBe(length);
  for (const i of [0, length - 1]) expect(data[i]).toBe((offset + i) % 251);
};

describe("ByteRangeReadCoalescer", () => {
  test("adjacent sub-chunks of one shard are read with one request", async () => {
    const t = setup();
    const reads = Array.from({ length: 7 }, (_, i) =>
      t.coalescer.read(
        "shard",
        { offset: 1000 + i * 5000, length: 5000 },
        undefined,
      ),
    );
    await t.open();
    const results = await Promise.all(reads);
    expect(t.requests.map((r) => r.range)).toEqual([
      { offset: 1000, length: 35000 },
    ]);
    results.forEach((data, i) => expectBytes(data, 1000 + i * 5000, 5000));
    // Each caller owns its buffer: decoders transfer it to a worker.
    expect(new Set(results.map((d) => d.buffer)).size).toBe(7);
  });

  test("distant ranges, other objects and the length cap make separate requests", async () => {
    const t = setup(16, 10000);
    const reads = [
      t.coalescer.read("a", { offset: 0, length: 100 }, undefined),
      t.coalescer.read("a", { offset: 110, length: 100 }, undefined), // gap 10: merged
      t.coalescer.read("a", { offset: 1000, length: 100 }, undefined), // gap 790: not
      t.coalescer.read("b", { offset: 210, length: 100 }, undefined), // another object
      t.coalescer.read("a", { offset: 1100, length: 9950 }, undefined), // over the cap
    ];
    await t.open();
    await Promise.all(reads);
    expect(
      t.requests.map((r) => [r.key, r.range.offset, r.range.length]),
    ).toEqual([
      ["a", 0, 210],
      ["a", 1000, 100],
      ["a", 1100, 9950],
      ["b", 210, 100],
    ]);
  });

  test("a merged request is cancelled only when every caller gives up", async () => {
    const t = setup();
    const c1 = new AbortController();
    const c2 = new AbortController();
    const r1 = t.coalescer.read("s", { offset: 0, length: 10 }, c1.signal);
    const r2 = t.coalescer.read("s", { offset: 10, length: 10 }, c2.signal);
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(t.requests.length).toBe(1);
    const merged = t.requests[0].signal!;
    c1.abort();
    await expect(r1).rejects.toBeDefined();
    expect(merged.aborted).toBe(false);
    await t.open();
    expectBytes(await r2, 10, 10);

    const c3 = new AbortController();
    const c4 = new AbortController();
    const r3 = t.coalescer.read("s", { offset: 0, length: 10 }, c3.signal);
    const r4 = t.coalescer.read("s", { offset: 10, length: 10 }, c4.signal);
    await new Promise((resolve) => setTimeout(resolve, 5));
    c3.abort();
    c4.abort();
    expect(t.requests[1].signal!.aborted).toBe(true);
    await expect(r3).rejects.toBeDefined();
    await expect(r4).rejects.toBeDefined();
  });

  test("a read cancelled before the batch goes out is not requested", async () => {
    const t = setup();
    const c = new AbortController();
    const r = t.coalescer.read("s", { offset: 0, length: 10 }, c.signal);
    c.abort();
    await expect(r).rejects.toBeDefined();
    await t.open();
    expect(t.requests.length).toBe(0);
  });

  test("4096 sub-chunks of a shard: 32 requests of 8 MiB, not 4096", async () => {
    const t = setup();
    // 64^3 uint8 sub-chunks compressed to about 64 KiB, 4096 of them in a
    // 512^3 shard (C order): every read in one go.
    const sub = 64 * 1024;
    const reads = Array.from({ length: 4096 }, (_, i) =>
      t.coalescer.read("shard", { offset: i * sub, length: sub }, undefined),
    );
    await t.open();
    await Promise.all(reads);
    expect(t.coalescer.numRequests).toBe(32);
    expect(t.requests.every((r) => r.range.length <= 8 * 1024 * 1024)).toBe(
      true,
    );
  });
});
