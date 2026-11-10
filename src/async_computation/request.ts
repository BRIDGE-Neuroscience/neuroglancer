/**
 * @license
 * Copyright 2019 Google Inc.
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

import type { AsyncComputationSpec } from "#src/async_computation/index.js";
import { getAsyncComputationWorkerFactory } from "#src/worker_factories.js";

let numWorkers = 0;
const freeWorkers: Worker[] = [];
const pendingTasks = new Map<
  number,
  { msg: any; transfer: Transferable[] | undefined; cleanup?: () => void }
>();
const tasks = new Map<
  number,
  {
    resolve: (value: any) => void;
    reject: (error: any) => void;
  }
>();
/**
 * At most this many decode workers.  Each holds its decoders' memory (a
 * WebAssembly heap as large as the largest chunk it has decoded), so more
 * workers cost memory, while downloads rather than decoding set the pace.
 */
export const MAX_ASYNC_COMPUTATION_WORKERS = 6;

// On Safari, `navigator.hardwareConcurrency` is not defined.
const maxWorkers =
  typeof navigator.hardwareConcurrency === "undefined"
    ? 4
    : Math.min(MAX_ASYNC_COMPUTATION_WORKERS, navigator.hardwareConcurrency);

let numRetiredWorkers = 0;

/** The pool's state, for tests and statistics. */
export function getAsyncComputationPoolStatistics() {
  return { numWorkers, maxWorkers, numRetiredWorkers };
}
let nextTaskId = 0;

function returnWorker(worker: Worker) {
  for (const [id, task] of pendingTasks) {
    pendingTasks.delete(id);
    task.cleanup?.();
    worker.postMessage(task.msg, task.transfer as Transferable[]);
    return;
  }
  freeWorkers.push(worker);
}

function launchWorker() {
  ++numWorkers;
  // Note: For compatibility with multiple bundlers, a browser-compatible URL
  // must be used with `new URL`, which means a Node.js subpath import like
  // "#src/async_computation.bundle.js" cannot be used.
  const worker =
    getAsyncComputationWorkerFactory()?.() ??
    new Worker(
      /* webpackChunkName: "neuroglancer_async_computation" */
      new URL("../async_computation.bundle.js", import.meta.url),
      { type: "module" },
    );
  let ready = false;
  worker.onmessage = (msg) => {
    // First message indicates worker is ready.
    if (!ready) {
      ready = true;
      returnWorker(worker);
      return;
    }
    const { id, value, error, retire } = msg.data as {
      id: number;
      value?: any;
      error?: string;
      retire?: boolean;
    };
    if (retire) {
      // Its decoder's memory grew to fit a very large chunk: a fresh worker
      // takes its place (if any work is waiting) with a small heap.
      worker.terminate();
      --numWorkers;
      ++numRetiredWorkers;
      if (pendingTasks.size > 0) launchWorker();
    } else {
      returnWorker(worker);
    }
    const callbacks = tasks.get(id)!;
    tasks.delete(id);
    if (callbacks === undefined) return;
    if (error !== undefined) {
      callbacks.reject(error);
    } else {
      callbacks.resolve(value);
    }
  };
}

export function requestAsyncComputation<
  Signature extends (...args: any) => any,
>(
  request: AsyncComputationSpec<Signature>,
  signal: AbortSignal | undefined,
  transfer: Transferable[] | undefined,
  ...args: Parameters<Signature>
): Promise<ReturnType<Signature>> {
  const id = nextTaskId++;
  const msg = { t: request.id, id, args: args };

  signal?.throwIfAborted();

  const promise = new Promise<ReturnType<Signature>>((resolve, reject) => {
    tasks.set(id, { resolve, reject });
  });

  if (freeWorkers.length !== 0) {
    freeWorkers.pop()!.postMessage(msg, transfer as Transferable[]);
  } else {
    let cleanup: (() => void) | undefined;
    if (signal !== undefined) {
      function abortHandler() {
        pendingTasks.delete(id);
        const task = tasks.get(id)!;
        tasks.delete(id);
        task.reject(signal!.reason);
      }
      signal.addEventListener("abort", abortHandler, { once: true });
      cleanup = () => {
        signal.removeEventListener("abort", abortHandler);
      };
    }
    pendingTasks.set(id, { msg, transfer, cleanup });
    if (tasks.size > numWorkers && numWorkers < maxWorkers) {
      launchWorker();
    }
  }

  return promise;
}
