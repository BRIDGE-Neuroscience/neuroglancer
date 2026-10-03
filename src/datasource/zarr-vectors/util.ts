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

/** @file Small helpers shared by the zarr-vectors frontend and worker. */

const warned = new Set<string>();

/** Logs a diagnostic once per session. */
export function warnOnce(message: string) {
  if (warned.has(message)) return;
  warned.add(message);
  console.warn(`zarr-vectors: ${message}`);
}

/**
 * Least-recently-used cache of promises.  Concurrent requests for one key
 * share a load; a load that fails is forgotten so the next request retries.
 * Bounded by entry count and, when `sizeOf` is given, by total size; loads
 * still in flight are never evicted, so they stay shared.
 */
export class AsyncLru<T> {
  private entries = new Map<string, Promise<T>>();
  private sizes = new Map<string, number>();
  private totalSize = 0;

  constructor(
    private maxEntries: number,
    private maxSize = Infinity,
    private sizeOf?: (value: T) => number,
  ) {}

  /** The cached entry, without loading. */
  peek(key: string): Promise<T> | undefined {
    return this.entries.get(key);
  }

  get(key: string, load: () => Promise<T>): Promise<T> {
    const existing = this.entries.get(key);
    if (existing !== undefined) {
      this.entries.delete(key);
      this.entries.set(key, existing);
      return existing;
    }
    const promise = load();
    this.entries.set(key, promise);
    promise.then(
      (value) => {
        if (this.entries.get(key) !== promise) return;
        const size = this.sizeOf?.(value) ?? 0;
        this.sizes.set(key, size);
        this.totalSize += size;
        this.evict();
      },
      () => {
        if (this.entries.get(key) === promise) this.delete(key);
      },
    );
    this.evict();
    return promise;
  }

  /** Records a value that was computed elsewhere. */
  put(key: string, value: T) {
    if (!this.entries.has(key)) this.get(key, () => Promise.resolve(value));
  }

  private delete(key: string) {
    this.entries.delete(key);
    this.totalSize -= this.sizes.get(key) ?? 0;
    this.sizes.delete(key);
  }

  private evict() {
    // Oldest first; `sizes` holds exactly the settled entries.
    for (const key of this.entries.keys()) {
      if (
        this.entries.size <= 1 ||
        (this.entries.size <= this.maxEntries && this.totalSize <= this.maxSize)
      ) {
        return;
      }
      if (this.sizes.has(key)) this.delete(key);
    }
  }
}

/** Runs `fn` over `items` with at most `limit` in flight. */
export async function mapConcurrent<T>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<void>,
): Promise<void> {
  let next = 0;
  const run = async () => {
    while (next < items.length) {
      const i = next++;
      await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
}

/** A signal that never aborts, for loads shared between requests. */
export const SHARED_SIGNAL = new AbortController().signal;
