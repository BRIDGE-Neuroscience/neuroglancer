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
 * @file The chunk worker has no `window` or WebGL: the modules it loads must
 * not import frontend code.  A value import of one pulls the whole UI into the
 * worker bundle, which then fails at load.
 */

import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const DIR = import.meta.dirname;
const PREFIX = "#src/datasource/zarr-vectors/";
/** Handler modules that run on the main thread only. */
const FRONTEND_MODULES = ["compat", "dense_frontend", "frontend", "layer"];
/** Neuroglancer directories that need the DOM or WebGL. */
const FRONTEND_PATHS =
  /^#src\/(webgl|ui|widget|layer|display_context|perspective_view|sliceview\/frontend)/;

/** Non-type imports of a module. */
function valueImports(name: string): string[] {
  const source = fs.readFileSync(path.join(DIR, `${name}.ts`), "utf8");
  const out: string[] = [];
  for (const m of source.matchAll(
    /^import\s+(type\s+)?[^;]*?from\s+"([^"]+)";/gms,
  )) {
    if (m[1] === undefined) out.push(m[2]);
  }
  for (const m of source.matchAll(/^import\s+"([^"]+)";/gm)) out.push(m[1]);
  return out;
}

describe("chunk worker imports", () => {
  it("backend.ts reaches no frontend module", () => {
    const seen = new Set<string>();
    const offending: string[] = [];
    const visit = (name: string, via: string) => {
      if (seen.has(name)) return;
      seen.add(name);
      for (const spec of valueImports(name)) {
        if (spec.startsWith(PREFIX)) {
          const dep = spec.slice(PREFIX.length).replace(/\.js$/, "");
          if (FRONTEND_MODULES.includes(dep)) offending.push(`${via} > ${dep}`);
          else visit(dep, `${via} > ${dep}`);
        } else if (FRONTEND_PATHS.test(spec)) {
          offending.push(`${via} > ${spec}`);
        }
      }
    };
    visit("backend", "backend");
    expect(seen.size).toBeGreaterThan(5);
    expect(offending).toEqual([]);
  });
});
