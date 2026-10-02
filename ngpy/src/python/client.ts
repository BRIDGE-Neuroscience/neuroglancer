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

/** @file Page-side handle on the Pyodide worker: RPC + event fan-out. */

import { Signal } from "../util/signal.js";
import type { FromWorker, ToWorker } from "./protocol.js";

export const DEFAULT_PYODIDE_INDEX_URL =
  "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/";

export type PythonStatus = "idle" | "loading" | "ready" | "failed";

type WithoutId<T> = T extends unknown ? Omit<T, "id"> : never;

export class PythonClient {
  readonly progress = new Signal<[string]>();
  readonly output = new Signal<["stdout" | "stderr", string]>();
  readonly emitted = new Signal<[string, string]>();
  readonly statusChanged = new Signal();
  status: PythonStatus = "idle";
  info: any;
  jspi = false;
  error: string | undefined;
  private worker: Worker | undefined;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (v: any) => void; reject: (e: Error) => void }
  >();
  private readyPromise: Promise<void> | undefined;

  constructor(
    private workerSource: string,
    private payload: () => ArrayBuffer,
  ) {}

  start(
    indexURL: string,
    pageInfo: { pageUrl: string; viewerUrl: string },
  ): Promise<void> {
    if (this.readyPromise !== undefined) return this.readyPromise;
    this.setStatus("loading");
    const blobUrl = URL.createObjectURL(
      new Blob([this.workerSource], { type: "text/javascript" }),
    );
    const worker = (this.worker = new Worker(blobUrl, {
      type: "module",
      name: "ngpy-pyodide",
    }));
    this.readyPromise = new Promise<void>((resolve, reject) => {
      worker.addEventListener("message", (event: MessageEvent<FromWorker>) => {
        const msg = event.data;
        switch (msg.type) {
          case "progress":
            this.progress.dispatch(msg.message);
            break;
          case "ready":
            this.info = msg.info;
            this.jspi = msg.jspi;
            this.setStatus("ready");
            resolve();
            break;
          case "failed":
            this.error = msg.error;
            this.setStatus("failed");
            reject(new Error(msg.error));
            break;
          case "stdout":
          case "stderr":
            this.output.dispatch(msg.type, msg.text);
            break;
          case "emit":
            this.emitted.dispatch(msg.kind, msg.text);
            break;
          case "result": {
            const p = this.pending.get(msg.id);
            if (p === undefined) break;
            this.pending.delete(msg.id);
            if (msg.ok) p.resolve(msg.value);
            else p.reject(new Error(msg.error));
            break;
          }
        }
      });
      worker.addEventListener("error", (e) => {
        this.error = e.message || "worker error";
        this.setStatus("failed");
        reject(new Error(this.error));
      });
    });
    const payload = this.payload();
    this.send({ type: "init", indexURL, payload, pageInfo }, [payload]);
    return this.readyPromise;
  }

  private setStatus(status: PythonStatus) {
    this.status = status;
    this.statusChanged.dispatch();
  }

  private send(msg: ToWorker, transfer: Transferable[] = []) {
    this.worker!.postMessage(msg, transfer);
  }

  private request<T>(
    msg: WithoutId<Extract<ToWorker, { id: number }>>,
  ): Promise<T> {
    if (this.worker === undefined)
      return Promise.reject(new Error("Python is not started"));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send({ ...(msg as any), id });
    });
  }

  /** `ngpy.api.<fn>(...args)`; `args` are already JSON text. */
  call<T = string>(
    fn: string,
    args: string[] = [],
    promising = false,
  ): Promise<T> {
    return this.request<T>({ type: "call", fn, args, promising });
  }

  /** `call` with JSON in and out. */
  async callJson<T = any>(fn: string, request: unknown): Promise<T> {
    const text = await this.call<string>(fn, [JSON.stringify(request)]);
    return JSON.parse(text) as T;
  }

  run(code: string, filename = "<ngpy>"): Promise<string | undefined> {
    return this.request({ type: "run", code, filename });
  }
}
