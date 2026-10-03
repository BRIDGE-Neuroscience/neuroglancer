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
 * @file Everything here differs between Neuroglancer versions the handler
 * supports (google/neuroglancer master and the 2026 BRIDGE branches).  When a
 * merge breaks the build, look here first.
 *
 * Frontend only: it imports WebGL modules, which must stay out of the chunk
 * worker (`worker_imports.spec.ts` checks this).
 */

import type { GL } from "#src/webgl/context.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";
import * as shaderLib from "#src/webgl/shader_lib.js";
import type {
  ShaderControlsParseResult,
  ShaderControlState,
} from "#src/webgl/shader_ui_controls.js";
import { setControlsInShader } from "#src/webgl/shader_ui_controls.js";

/**
 * `setControlsInShader` takes the whole parse result in current Neuroglancer
 * and only its `controls` in older trees.
 */
export function setShaderControls(
  gl: GL,
  shader: ShaderProgram,
  state: ShaderControlState,
  parseResult: ShaderControlsParseResult,
) {
  const set = setControlsInShader as (...args: unknown[]) => void;
  const result = parseResult as ShaderControlsParseResult & {
    preprocessing?: unknown;
  };
  set(
    gl,
    shader,
    state,
    result.preprocessing !== undefined ? result : result.controls,
  );
}

/**
 * String shader controls (newer Neuroglancer) need `glsl_string`, which
 * older trees do not export.  Looked up at run time so bundlers do not flag
 * the missing export.
 */
export function addStringShaderSupport(builder: ShaderBuilder) {
  const code = (shaderLib as Record<string, unknown>)[
    ["glsl", "string"].join("_")
  ];
  if (typeof code === "string") builder.addFragmentCode(code);
}
