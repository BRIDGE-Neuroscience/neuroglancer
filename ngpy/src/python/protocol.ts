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

/** @file Messages between the ngpy page and its Pyodide worker. */

export interface InitMessage {
  type: "init";
  indexURL: string;
  payload: ArrayBuffer;
  pageInfo: { pageUrl: string; viewerUrl: string };
}

/** Call `ngpy.api.<fn>(...args)`; args and string results are JSON text. */
export interface CallMessage {
  type: "call";
  id: number;
  fn: string;
  args: string[];
  /** Enter through `callPromising` under the JSPI mutex (`.zvf` export). */
  promising?: boolean;
}

export interface RunMessage {
  type: "run";
  id: number;
  code: string;
  filename: string;
}

export type ToWorker = InitMessage | CallMessage | RunMessage;

export type FromWorker =
  | { type: "progress"; message: string }
  | { type: "ready"; info: any; jspi: boolean }
  | { type: "failed"; error: string }
  | { type: "result"; id: number; ok: true; value: any }
  | { type: "result"; id: number; ok: false; error: string }
  | { type: "stdout"; text: string }
  | { type: "stderr"; text: string }
  | { type: "emit"; kind: string; text: string };
