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

import { zstdCompressSync } from "node:zlib";
import { expect, test } from "vitest";
import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { RETIRE_AFTER_BYTES } from "#src/async_computation/index.js";
import {
  getAsyncComputationPoolStatistics,
  MAX_ASYNC_COMPUTATION_WORKERS,
  requestAsyncComputation,
} from "#src/async_computation/request.js";

const decode = (bytes: number, fill: number) => {
  const input = new Uint8Array(
    zstdCompressSync(new Uint8Array(bytes).fill(fill)),
  );
  return requestAsyncComputation(decodeZstd, undefined, [input.buffer], input);
};

// Note: @vitest/web-worker runs workers in this thread, and workers started
// at the same moment can share module instances; terminating one of them
// then silences the others.  Retirement is therefore tested before the pool
// is filled at once.
test("a worker is replaced after a very large result, and the pool goes on", async () => {
  expect((await decode(1024, 1))[0]).toBe(1);
  const before = getAsyncComputationPoolStatistics();
  const out = await decode(RETIRE_AFTER_BYTES, 7);
  expect(out.length).toBe(RETIRE_AFTER_BYTES);
  const after = getAsyncComputationPoolStatistics();
  expect(after.numRetiredWorkers).toBe(before.numRetiredWorkers + 1);
  expect(after.numWorkers).toBe(before.numWorkers - 1);
  expect((await decode(1024, 3))[0]).toBe(3);
  // A smaller result keeps its worker.
  expect((await decode(RETIRE_AFTER_BYTES - 1024, 4))[0]).toBe(4);
  expect(getAsyncComputationPoolStatistics().numRetiredWorkers).toBe(
    after.numRetiredWorkers,
  );
});

test("the decode pool stays within its cap under many decodes at once", async () => {
  const results = await Promise.all(
    Array.from({ length: 40 }, (_, i) => decode(256 * 1024, i)),
  );
  results.forEach((r, i) => expect(r[0]).toBe(i));
  const { numWorkers } = getAsyncComputationPoolStatistics();
  expect(numWorkers).toBeLessThanOrEqual(MAX_ASYNC_COMPUTATION_WORKERS);
  expect(numWorkers).toBeGreaterThan(1);
});
