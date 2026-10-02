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
 * @file The Pyodide host: a MODULE worker started from a blob URL.
 *
 * Pyodide >= 314 refuses classic workers and must be `import()`ed from
 * `pyodide.mjs`.  Only numpy, micropip and zarr are loaded (zarr MUST come
 * from Pyodide's own distribution: it is WASM-patched so `sync()` runs on the
 * WebLoop -- never micropip it); scipy (14 MB) is not loaded up front.  The
 * Python payload (ngpy + the vendored neuroglancer subset + zarr-vectors) is
 * an embedded zip unpacked into site-packages with `unpackArchive`.
 *
 * Hard-won pieces kept from the old `pyodide_worker.ts`:
 *
 *  - JSPI `callPromising` is used ONLY for `.zvf` export, whose writer reaches
 *    zarr's synchronous `sync()`.  Everything else is an ordinary call: the
 *    reads go through zarr-vectors' async prime-and-replay path and never
 *    need stack switching.
 *  - EVERY promising entry goes through one promise-chain mutex
 *    (`runExclusive`).  Two concurrent promising calls deadlock this worker
 *    permanently (reproduced 2026-07-19, chromium 141 / pyodide 314.0.2, at
 *    >= ~530 store keys per reader); serialising them is the proven guard.
 *  - Without JSPI the async variant is awaited on the WebLoop instead: TRK
 *    export works, ZVF reports a clear "needs JSPI" error.
 */

/// <reference lib="webworker" />

import type { FromWorker, ToWorker } from "./protocol.js";

declare const self: DedicatedWorkerGlobalScope;

let pyodide: any;
let api: any;

function post(msg: FromWorker, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer);
}

function jspiAvailable(): boolean {
  const wa = WebAssembly as any;
  return typeof wa.Suspending === "function" || typeof wa.promising === "function";
}

let promisingTail: Promise<unknown> = Promise.resolve();
function runExclusive<T>(fn: () => Promise<T>): Promise<T> {
  const result = promisingTail.then(fn);
  promisingTail = result.then(
    () => {},
    () => {},
  );
  return result;
}

async function init(msg: Extract<ToWorker, { type: "init" }>) {
  const indexURL = msg.indexURL.endsWith("/") ? msg.indexURL : msg.indexURL + "/";
  post({ type: "progress", message: "Loading Pyodide runtime…" });
  const mod: any = await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`);
  pyodide = await mod.loadPyodide({ indexURL });
  pyodide.setStdout({ batched: (text: string) => post({ type: "stdout", text }) });
  pyodide.setStderr({ batched: (text: string) => post({ type: "stderr", text }) });
  post({ type: "progress", message: "Loading numpy, micropip and zarr…" });
  await pyodide.loadPackage(["numpy", "micropip", "zarr"], {
    messageCallback: () => {},
  });
  post({ type: "progress", message: "Unpacking the ngpy Python payload…" });
  const sitePackages: string = pyodide.runPython(
    "import site; site.getsitepackages()[0]",
  );
  pyodide.unpackArchive(new Uint8Array(msg.payload), "zip", {
    extractDir: sitePackages,
  });
  (self as any).ngpy_emit = (kind: string, text: string) =>
    post({ type: "emit", kind, text });
  pyodide.runPython(
    [
      "import importlib, js",
      "importlib.invalidate_caches()",
      "import ngpy.bridge as _ngpy_bridge",
      "_ngpy_bridge.set_emitter(lambda k, t: js.ngpy_emit(k, t))",
      "del _ngpy_bridge",
    ].join("\n"),
  );
  api = pyodide.pyimport("ngpy.api");
  const info = JSON.parse(api.boot(JSON.stringify(msg.pageInfo)));
  post({ type: "ready", info, jspi: jspiAvailable() });
}

function toPlain(value: any): any {
  if (value === undefined || value === null) return value;
  if (typeof value === "object" && typeof value.toJs === "function") {
    const converted = value.toJs({ create_proxies: false });
    value.destroy?.();
    return converted;
  }
  return value;
}

async function call(msg: Extract<ToWorker, { type: "call" }>) {
  const fn = api[msg.fn];
  if (fn === undefined) throw new Error(`ngpy.api has no ${msg.fn}`);
  let result: any;
  if (msg.promising && jspiAvailable()) {
    const sync = api[`${msg.fn}_sync`] ?? fn;
    result = await runExclusive(() => sync.callPromising(...msg.args));
  } else if (msg.promising) {
    result = await runExclusive(() => fn(...msg.args));
  } else {
    result = fn(...msg.args);
  }
  if (result && typeof result.then === "function") result = await result;
  return toPlain(result);
}

self.addEventListener("message", async (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  if (msg.type === "init") {
    try {
      await init(msg);
    } catch (e) {
      post({ type: "failed", error: String((e as Error)?.stack ?? e) });
    }
    return;
  }
  if (pyodide === undefined) {
    post({ type: "result", id: msg.id, ok: false, error: "Pyodide is not ready" });
    return;
  }
  try {
    let value: any;
    if (msg.type === "run") {
      value = toPlain(
        await pyodide.runPythonAsync(msg.code, { filename: msg.filename }),
      );
      if (value !== undefined && value !== null && typeof value !== "string") {
        value = String(value);
      }
    } else {
      value = await call(msg);
    }
    const transfer: Transferable[] = [];
    if (Array.isArray(value)) {
      for (const v of value) if (v instanceof Uint8Array) transfer.push(v.buffer);
    }
    post({ type: "result", id: msg.id, ok: true, value }, transfer);
  } catch (e) {
    post({
      type: "result",
      id: msg.id,
      ok: false,
      error: String((e as Error)?.message ?? e),
    });
  }
});
