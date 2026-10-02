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
 * @file Parameters shared by the zarr-vectors frontend and chunk worker.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import type { ZarrVectorsLevel } from "#src/datasource/zarr-vectors/store.js";
import type { SliceViewChunkSpecification } from "#src/sliceview/base.js";

export type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";

/** One level's spatial chunks, for the dense render layer. */
export class ZarrVectorsGeometryChunkSourceParameters {
  /** kvstore URL of the store root, ending in `/`. */
  storeUrl!: string;
  description!: ZarrVectorsGeometryDescription;
  level!: ZarrVectorsLevel;
  static RPC_ID = "zarr-vectors/GeometryChunkSource";
}

/** Whole objects at full resolution, keyed by segment id. */
export class ZarrVectorsObjectSkeletonSourceParameters {
  storeUrl!: string;
  description!: ZarrVectorsGeometryDescription;
  /** The finest level, which per-object reads use. */
  level!: ZarrVectorsLevel;
  static RPC_ID = "zarr-vectors/ObjectSkeletonSource";
}

/** Whole mesh objects, keyed by segment id. */
export class ZarrVectorsMeshSourceParameters {
  storeUrl!: string;
  description!: ZarrVectorsGeometryDescription;
  level!: ZarrVectorsLevel;
  static RPC_ID = "zarr-vectors/MeshSource";
}

/** A spatial chunk grid level, as the slice-view chunk machinery sees it. */
export interface ZarrVectorsChunkSpecification
  extends SliceViewChunkSpecification<Float32Array> {
  /** Index of the level in the store (0 = finest). */
  levelIndex: number;
}

export const ZARR_VECTORS_DENSE_RENDER_LAYER_RPC_ID =
  "zarr-vectors/DenseRenderLayer";
export const ZARR_VECTORS_DENSE_RENDER_LAYER_UPDATE_SOURCES_RPC_ID =
  "zarr-vectors/DenseRenderLayer.updateSources";
