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

/** @file ngpy.html entry point. */

// `ngpy:embedded` is a virtual module supplied by ngpy/build.ts (typed in
// embedded.d.ts); there is no file for the import resolver to find.
/* eslint-disable import/no-unresolved */
import {
  BUILD_INFO,
  DEMO_SCRIPT,
  PAYLOAD_BASE64,
  WORKER_SOURCE,
} from "ngpy:embedded";
/* eslint-enable import/no-unresolved */
import { App } from "./app.js";
import { parseParams } from "./params.js";
import { handleOAuthRedirect } from "./store/auth.js";

function decodeBase64(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; ++i) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

if (!handleOAuthRedirect()) {
  const root = document.getElementById("ngpy-root") ?? document.body;
  const app = new App(root, parseParams(window.location.search), {
    workerSource: WORKER_SOURCE,
    payload: () => decodeBase64(PAYLOAD_BASE64),
    demoScript: DEMO_SCRIPT,
    buildInfo: BUILD_INFO,
  });
  (window as any).ngpy = app;
  void app.start();
}
