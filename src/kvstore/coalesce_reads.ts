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

/**
 * @file Merges byte-range reads of one object that are issued together, such
 * as the sub-chunks of one shard that a view asks for at once, into fewer
 * requests.
 */

import type { ByteRange } from "#src/kvstore/index.js";

/**
 * Reads separated by at most this many bytes are merged; the bytes between
 * them are read and discarded.
 */
export const COALESCE_MAX_GAP = 16 * 1024;

/**
 * A merged read is at most this long, so that one slow response does not hold
 * up many chunks.
 */
export const COALESCE_MAX_LENGTH = 8 * 1024 * 1024;

interface PendingRead<Key> {
  key: Key;
  offset: number;
  length: number;
  signal: AbortSignal | undefined;
  resolve: (data: Uint8Array<ArrayBuffer>) => void;
  reject: (reason: unknown) => void;
}

export type ReadByteRange<Key> = (
  key: Key,
  byteRange: ByteRange,
  signal: AbortSignal | undefined,
) => Promise<Uint8Array<ArrayBuffer>>;

/**
 * Collects the byte-range reads of each object issued in one task (until a
 * zero-delay timeout), and reads each run of nearby ranges with one request.
 * Each caller gets its own copy of its bytes, since decoders take over their
 * input's buffer.  A merged request is cancelled only once every caller in it
 * has given up; a caller that gives up is rejected at once.
 */
export class ByteRangeReadCoalescer<Key> {
  private pending = new Map<string, PendingRead<Key>[]>();
  private scheduled = false;

  /** Requests made, for tests and statistics. */
  numRequests = 0;

  constructor(
    private readByteRange: ReadByteRange<Key>,
    private keyString: (key: Key) => string,
    public maxGap = COALESCE_MAX_GAP,
    public maxLength = COALESCE_MAX_LENGTH,
  ) {}

  read(
    key: Key,
    byteRange: ByteRange,
    signal: AbortSignal | undefined,
  ): Promise<Uint8Array<ArrayBuffer>> {
    signal?.throwIfAborted();
    return new Promise((resolve, reject) => {
      const onAbort = () => reject(signal!.reason);
      signal?.addEventListener("abort", onAbort, { once: true });
      const settle =
        <T>(f: (value: T) => void) =>
        (value: T) => {
          signal?.removeEventListener("abort", onAbort);
          f(value);
        };
      const id = this.keyString(key);
      let list = this.pending.get(id);
      if (list === undefined) this.pending.set(id, (list = []));
      list.push({
        key,
        offset: byteRange.offset,
        length: byteRange.length,
        signal,
        resolve: settle(resolve),
        reject: settle(reject),
      });
      if (!this.scheduled) {
        this.scheduled = true;
        setTimeout(() => this.flush(), 0);
      }
    });
  }

  private flush() {
    this.scheduled = false;
    const batch = this.pending;
    this.pending = new Map();
    const { maxGap, maxLength } = this;
    for (const reads of batch.values()) {
      const live = reads
        .filter((r) => !r.signal?.aborted)
        .sort((a, b) => a.offset - b.offset);
      let group: PendingRead<Key>[] = [];
      let start = 0;
      let end = 0;
      for (const r of live) {
        const rEnd = r.offset + r.length;
        if (
          group.length > 0 &&
          r.offset <= end + maxGap &&
          Math.max(end, rEnd) - start <= maxLength
        ) {
          group.push(r);
          end = Math.max(end, rEnd);
          continue;
        }
        if (group.length > 0) this.issue(group, start, end);
        group = [r];
        start = r.offset;
        end = rEnd;
      }
      if (group.length > 0) this.issue(group, start, end);
    }
  }

  private issue(group: PendingRead<Key>[], start: number, end: number) {
    ++this.numRequests;
    let signal: AbortSignal | undefined;
    let cleanup = () => {};
    if (group.length === 1) {
      signal = group[0].signal;
    } else if (group.every((r) => r.signal !== undefined)) {
      const controller = new AbortController();
      signal = controller.signal;
      let live = group.length;
      const onAbort = () => {
        if (--live === 0) controller.abort();
      };
      for (const r of group) {
        r.signal!.addEventListener("abort", onAbort, { once: true });
      }
      cleanup = () => {
        for (const r of group) r.signal!.removeEventListener("abort", onAbort);
      };
    }
    this.readByteRange(
      group[0].key,
      { offset: start, length: end - start },
      signal,
    ).then(
      (data) => {
        cleanup();
        for (const r of group) {
          const from = r.offset - start;
          r.resolve(
            group.length === 1 ? data : data.slice(from, from + r.length),
          );
        }
      },
      (error) => {
        cleanup();
        for (const r of group) r.reject(error);
      },
    );
  }
}
