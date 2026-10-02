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
 * @file The page half of the Python <-> viewer shared-state protocol.
 *
 * A re-implementation (not an import) of upstream's
 * `ClientStateSynchronizer` (`src/python_integration/api.ts`), talking to
 * `ngpy.bridge.handle_client_state` over postMessage instead of HTTP:
 *
 *  - the viewer's `state` changes -> after a throttle, the JSON is sent as
 *    `{s, g: changed.count, pg: lastServerGeneration, c: clientId}` unless it
 *    equals the state last received from / sent to Python;
 *  - Python answers 200 `{g}` (remember it) or 412 (Python changed the state
 *    concurrently; its push will arrive and win);
 *  - a push from Python (`setServerState`) does `reset()` + `restoreState()`,
 *    then records the resulting `changed.count` so the echo is not sent back.
 *
 * Only the public `Trackable` surface is used: `changed` (with `add` and
 * `count`), `toJSON`, `restoreState`, `reset`.
 */

import { stringifyState } from "../util/signal.js";

export interface TrackableLike {
  changed: { add(handler: () => void): unknown; count: number };
  toJSON(): unknown;
  restoreState(value: unknown): void;
  reset(): void;
}

export interface SetStateMessage {
  s: unknown;
  g: number;
  pg: string;
  c: string;
}

export interface SetStateReply {
  status: number;
  g?: string;
}

export type SetStateTransport = (
  msg: SetStateMessage,
) => Promise<SetStateReply>;

export function randomClientId(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export class SharedStateSync {
  clientGeneration = -1;
  lastServerState = "";
  lastServerGeneration = "";
  readonly clientId: string;
  private needUpdate = false;
  private updateInProgress = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private enabled = false;
  /** Count of messages sent; for diagnostics and tests. */
  sent = 0;

  constructor(
    public trackable: TrackableLike,
    private transport: SetStateTransport,
    private throttleMs = 100,
    clientId?: string,
  ) {
    this.clientId = clientId ?? randomClientId();
    trackable.changed.add(() => this.schedule());
  }

  /** Start sending (the worker is ready); sends the current state at once. */
  enable(): Promise<void> {
    this.enabled = true;
    return this.handleStateChanged();
  }

  private schedule() {
    if (!this.enabled || this.timer !== undefined) return;
    this.timer = setTimeout(() => {
      this.timer = undefined;
      void this.handleStateChanged();
    }, this.throttleMs);
  }

  /** Send now if anything is pending (e.g. before running a script). */
  async flush(): Promise<void> {
    if (this.timer !== undefined) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
    if (this.enabled) await this.handleStateChanged();
  }

  async handleStateChanged(): Promise<void> {
    this.needUpdate = true;
    if (this.updateInProgress) return;
    try {
      this.updateInProgress = true;
      while (this.needUpdate) {
        this.needUpdate = false;
        const clientGeneration = this.trackable.changed.count;
        if (clientGeneration === this.clientGeneration) return;
        const json = this.trackable.toJSON();
        const encoded = stringifyState(json);
        if (encoded === this.lastServerState) {
          this.clientGeneration = clientGeneration;
          return;
        }
        ++this.sent;
        const reply = await this.transport({
          s: JSON.parse(encoded),
          g: clientGeneration,
          pg: this.lastServerGeneration,
          c: this.clientId,
        });
        if (reply.status === 200) {
          this.lastServerState = encoded;
          this.lastServerGeneration = reply.g ?? "";
          this.clientGeneration = clientGeneration;
        } else if (reply.status !== 412) {
          console.warn("ngpy: state update rejected", reply);
          return;
        }
      }
    } finally {
      this.updateInProgress = false;
    }
  }

  /** Python pushed a new state: replace the viewer's. */
  setServerState(state: unknown, generation: string): void {
    const { trackable } = this;
    trackable.reset();
    trackable.restoreState(state);
    this.lastServerState = stringifyState(state);
    this.clientGeneration = trackable.changed.count;
    this.lastServerGeneration = generation;
  }

  /** Python attached a viewer and reports the generation to quote. */
  setServerGeneration(generation: string): void {
    this.lastServerGeneration = generation;
  }
}
