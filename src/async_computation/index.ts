/**
 * @license
 * Copyright 2019 Google Inc.
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
 * @file Interface for defining transcode operations.
 */

/**
 * A worker that returns a result at least this large is retired once it has
 * replied, and a fresh one takes its place (see `request.ts`).
 */
export const RETIRE_AFTER_BYTES = 32 * 1024 * 1024;

export interface AsyncComputationSpec<Signature extends (...args: any) => any> {
  id: string;
  t?: Signature;
}

/**
 * Declares an asynchronous operation.
 */
export function asyncComputation<Signature extends (...args: any) => any>(
  id: string,
): AsyncComputationSpec<Signature> {
  return { id };
}
