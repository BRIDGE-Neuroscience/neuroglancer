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
import {
  Chunk,
  ChunkManager,
  ChunkQueueManager,
  ChunkSource,
  isDownloadMostlyDone,
  MAX_DOWNLOAD_SIZE_SCALE,
} from "#src/chunk_manager/backend.js";
import {
  ChunkDownloadStatistics,
  ChunkPriorityTier,
  ChunkState,
  getChunkDownloadStatisticIndex,
} from "#src/chunk_manager/base.js";
import { MovementTracker, MOVEMENT_SETTLE_MS } from "#src/sliceview/backend.js";
import { WatchableValue } from "#src/trackable_value.js";

// A stand-in for the worker RPC: shared objects are passed as themselves.
function fakeRpc() {
  const objects = new Map<number, any>();
  let nextId = 1;
  return {
    get: (x: any) => (typeof x === "number" ? objects.get(x) : x),
    set: (id: number, obj: any) => objects.set(id, obj),
    delete: (id: number) => objects.delete(id),
    newId: () => nextId++,
    invoke: () => {},
    promiseInvoke: () => Promise.resolve(),
  } as any;
}

const capacity = (itemLimit: number, sizeLimit: number) => ({
  itemLimit: new WatchableValue(itemLimit),
  sizeLimit: new WatchableValue(sizeLimit),
});

const MiB = 1024 * 1024;

class TestSource extends ChunkSource {
  estimate = 0;
  signals = new Map<string, AbortSignal>();
  estimateDownloadBytes() {
    return this.estimate;
  }
  download(chunk: Chunk, signal: AbortSignal) {
    this.signals.set(chunk.key!, signal);
    // Never finishes: the tests look at what is downloading.
    return new Promise<void>(() => {});
  }
}

function setup(
  options: {
    itemLimit?: number;
    sizeLimit?: number;
    movementAware?: boolean;
  } = {},
) {
  const rpc = fakeRpc();
  const queueManager = new ChunkQueueManager(rpc, {
    gpuMemoryCapacity: capacity(1e6, 1e12),
    systemMemoryCapacity: capacity(1e7, 1e12),
    downloadCapacity: capacity(
      options.itemLimit ?? 100,
      options.sizeLimit ?? 64 * MiB,
    ),
    computeCapacity: capacity(128, 5e8),
    enablePrefetch: new WatchableValue(true),
    enableMovementAwareDownloads: new WatchableValue(
      options.movementAware ?? true,
    ),
  });
  const chunkManager = new ChunkManager(rpc, {
    chunkQueueManager: queueManager,
  });
  const source = new TestSource(rpc, { chunkManager });
  const chunks: Chunk[] = [];
  const getChunk = (key: string) => {
    let chunk = source.chunks.get(key);
    if (chunk === undefined) {
      chunk = source.getNewChunk_(Chunk);
      chunk.initialize(key);
      source.addChunk(chunk);
      chunks.push(chunk);
    }
    return chunk;
  };
  // Asks for the chunks (by key, highest priority first) and lets the
  // queue manager start downloads.
  const request = async (keys: string[]) => {
    keys.forEach((key, i) =>
      chunkManager.requestChunk(getChunk(key), ChunkPriorityTier.VISIBLE, -i),
    );
    chunkManager.updateQueueState([
      ChunkPriorityTier.VISIBLE,
      ChunkPriorityTier.PREFETCH,
    ]);
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
  const downloading = () =>
    chunks.filter((c) => c.state === ChunkState.DOWNLOADING).map((c) => c.key);
  const setMeanDownloadMs = (ms: number) => {
    const { statistics } = source;
    statistics[
      getChunkDownloadStatisticIndex(ChunkDownloadStatistics.totalTime)
    ] = ms;
    statistics[
      getChunkDownloadStatisticIndex(ChunkDownloadStatistics.totalChunks)
    ] = 1;
  };
  return {
    queueManager,
    source,
    getChunk,
    request,
    downloading,
    setMeanDownloadMs,
  };
}

const keys = (n: number, prefix = "c") =>
  Array.from({ length: n }, (_, i) => `${prefix}${i}`);

describe("download capacity by bytes", () => {
  test("a few large chunks download at once, not a hundred", async () => {
    const t = setup();
    t.source.estimate = 16 * MiB; // a 256^3 uint8 chunk
    await t.request(keys(30));
    // 64 MiB / 16 MiB: four at once, the highest priority first.
    expect(t.downloading()).toEqual(["c0", "c1", "c2", "c3"]);
  });

  test("small chunks are limited by the count, as before", async () => {
    const t = setup();
    t.source.estimate = 256 * 1024; // a 64^3 uint8 chunk
    await t.request(keys(300));
    expect(t.downloading().length).toBe(100);
  });

  test("a chunk larger than the whole limit still downloads, alone", async () => {
    const t = setup();
    t.source.estimate = 200 * MiB;
    await t.request(keys(5));
    expect(t.downloading()).toEqual(["c0"]);
  });

  test("an infinite size limit is the old behaviour", async () => {
    const t = setup({ sizeLimit: Number.POSITIVE_INFINITY });
    t.source.estimate = 16 * MiB;
    await t.request(keys(30));
    expect(t.downloading().length).toBe(30);
  });
});

describe("download size limit on a fast link", () => {
  test("downloads that finish quickly raise the limit, slow ones lower it to the setting", async () => {
    const t = setup();
    t.source.estimate = 16 * MiB;
    for (let i = 0; i < 20; ++i) t.queueManager.noteDownloadTime(500);
    expect(t.queueManager.downloadSizeScale).toBe(MAX_DOWNLOAD_SIZE_SCALE);
    await t.request(keys(40));
    // 8 x 64 MiB / 16 MiB.
    expect(t.downloading().length).toBe(32);
    for (let i = 0; i < 20; ++i) t.queueManager.noteDownloadTime(5000);
    expect(t.queueManager.downloadSizeScale).toBe(1);
    // Times in between leave it as it is.
    t.queueManager.noteDownloadTime(2000);
    expect(t.queueManager.downloadSizeScale).toBe(1);
  });
});

describe("mostly done downloads", () => {
  test("isDownloadMostlyDone", () => {
    expect(isDownloadMostlyDone(600, undefined)).toBe(false);
    expect(isDownloadMostlyDone(100, 1000)).toBe(false);
    expect(isDownloadMostlyDone(600, 1000)).toBe(true);
    // Running for more than twice the mean: stalled, may be cancelled.
    expect(isDownloadMostlyDone(5000, 1000)).toBe(false);
  });

  for (const movementAware of [true, false]) {
    test(`a mostly done download is ${movementAware ? "kept" : "cancelled"} for a higher-priority one (movement-aware ${movementAware ? "on" : "off"})`, async () => {
      const t = setup({ movementAware });
      t.source.estimate = 32 * MiB;
      t.setMeanDownloadMs(1000);
      await t.request(["a", "b"]);
      expect(t.downloading()).toEqual(["a", "b"]);
      // "b" is 700 ms into a 1000 ms download; "a" has just begun.
      t.getChunk("b").downloadStartTime = Date.now() - 700;
      // The view moves: a new chunk comes first, "b" is no longer needed.
      await t.request(["n", "a"]);
      const signal = t.source.signals.get("b")!;
      if (movementAware) {
        expect(signal.aborted).toBe(false);
        expect(t.downloading()).toEqual(["a", "b"]);
      } else {
        expect(signal.aborted).toBe(true);
        expect(t.downloading()).toEqual(["a", "n"]);
      }
    });
  }

  test("a download that has just begun is still cancelled", async () => {
    const t = setup();
    t.source.estimate = 32 * MiB;
    t.setMeanDownloadMs(1000);
    await t.request(["a", "b"]);
    await t.request(["n", "a"]);
    expect(t.source.signals.get("b")!.aborted).toBe(true);
    expect(t.downloading()).toEqual(["a", "n"]);
  });
});

describe("MovementTracker", () => {
  test("two changes close together are movement; one is not", () => {
    const m = new MovementTracker();
    m.noteChange(1000);
    expect(m.isMoving(1010)).toBe(false);
    m.noteChange(1100);
    expect(m.isMoving(1110)).toBe(true);
    expect(m.isMoving(1000 + MOVEMENT_SETTLE_MS + 1)).toBe(false);
  });
});
