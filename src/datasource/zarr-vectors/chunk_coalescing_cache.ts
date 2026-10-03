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

/**
 * Coalesces concurrent decodes of the same spatial chunk.
 *
 * The object-keyed (pass-2) path fetches whole streamlines: one call per tract,
 * each walking that tract's manifest and decoding every chunk it touches. Tracts
 * in a dissection are spatially clustered by construction -- they pass through
 * the same regions -- so the same chunk is decoded once per tract crossing it.
 * At thousands of tracts that redundancy, not the transfer, is the cost.
 *
 * Keyed by chunk, so N tracts over M distinct chunks cost M decodes, not
 * N x blocks.
 *
 * ## Why aborts are not propagated
 *
 * The entry is shared, so honouring one waiter's `AbortSignal` would cancel a
 * decode the others are still waiting on -- turning one cancelled tract into a
 * cascade of spurious failures. Instead the shared work runs to completion and a
 * cancelled caller simply discards the result. That wastes at most one chunk
 * decode per abort, against a redundancy this removes wholesale.
 *
 * ## Why entries are retained after settling
 *
 * Coalescing only in-flight work would miss the common case: the chunk queue
 * admits pass-2 downloads in waves (100 at a time by default), so tracts sharing
 * a chunk frequently arrive in *different* waves. A small bounded set of settled
 * entries catches those; insertion-ordered eviction keeps it simple, since the
 * access pattern is a sweep rather than a working set.
 *
 * Rejections are evicted immediately -- a cached failure would otherwise pin a
 * transient error for the life of the entry.
 */
export class ChunkCoalescingCache<T> {
  private entries = new Map<string, Promise<T>>();
  private hits = 0;
  private misses = 0;

  constructor(private readonly maxEntries: number = 64) {}

  /**
   * Return the shared decode for `key`, starting it via `load` if absent.
   *
   * `load` receives no abort signal by design; see the class comment.
   */
  get(key: string, load: () => Promise<T>): Promise<T> {
    const existing = this.entries.get(key);
    if (existing !== undefined) {
      ++this.hits;
      return existing;
    }
    ++this.misses;
    const promise = load();
    this.entries.set(key, promise);
    // A failure must not be remembered: the next tract through this chunk
    // should retry rather than inherit a transient error.
    promise.catch(() => {
      if (this.entries.get(key) === promise) this.entries.delete(key);
    });
    this.evictToLimit();
    return promise;
  }

  private evictToLimit() {
    while (this.entries.size > this.maxEntries) {
      // Map preserves insertion order, so the first key is the oldest.
      const oldest = this.entries.keys().next();
      if (oldest.done === true) return;
      this.entries.delete(oldest.value);
    }
  }

  /** Coalescing effectiveness, for tests and diagnostics. */
  get statistics(): { hits: number; misses: number; size: number } {
    return { hits: this.hits, misses: this.misses, size: this.entries.size };
  }

  clear() {
    this.entries.clear();
  }
}
