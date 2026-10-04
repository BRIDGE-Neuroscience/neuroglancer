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

/** @file Plain data shared by the zarr-vectors frontend and chunk worker. */

import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";
import type {
  ZarrVectorsAttribute,
  ZarrVectorsLevel,
  ZarrVectorsLinksConvention,
} from "#src/datasource/zarr-vectors/store.js";
import type { SliceViewChunkSpecification } from "#src/sliceview/base.js";

/** What every reader of a store's geometry needs to know about it. */
export interface ZarrVectorsGeometryDescription {
  geometryKind: ZarrVectorsGeometryKind;
  linksConvention: ZarrVectorsLinksConvention;
  /** 2 for edges; 3 or more for faces. */
  linkWidth: number;
  /**
   * zarr-vectors-tools' skeleton layout. In `"linked"` stores a cross-chunk
   * `[child, parent]` link replaces the child's implied parent; in `"split"`
   * stores (precomputed ingests) it joins two copies of one vertex.
   */
  skeletonLayout: "linked" | "split" | undefined;
  /** Exposed vertex attributes. */
  attributes: ZarrVectorsAttribute[];
  /** Whether the store has an object index. */
  hasObjects: boolean;
}

class ZarrVectorsSourceParameters {
  /** kvstore URL of the store root, ending in `/`. */
  storeUrl!: string;
  description!: ZarrVectorsGeometryDescription;
  level!: ZarrVectorsLevel;
}

/** One level's spatial chunks, for the dense overview. */
export class ZarrVectorsGeometryChunkSourceParameters extends ZarrVectorsSourceParameters {
  static RPC_ID = "zarr-vectors/GeometryChunkSource";
}

/**
 * Whole objects at the finest level, as Neuroglancer skeletons. `levels` is
 * level 0's chain (`levelChain`): in an additive pyramid an object's parts
 * are read from every level of it.
 */
export class ZarrVectorsObjectSkeletonSourceParameters extends ZarrVectorsSourceParameters {
  static RPC_ID = "zarr-vectors/ObjectSkeletonSource";
  levels!: ZarrVectorsLevel[];
}

/**
 * Whole objects as Neuroglancer multiscale meshes: level of detail `i` is
 * `levels[i]`, whose chunks are `2 ** i` times level 0's (`mesh_lod.ts`).
 */
export class ZarrVectorsMeshSourceParameters {
  static RPC_ID = "zarr-vectors/MeshSource";
  storeUrl!: string;
  description!: ZarrVectorsGeometryDescription;
  levels!: ZarrVectorsLevel[];
  /** Base chunks added to every octree coordinate (`meshGridOffset`). */
  gridOffset!: number[];
  /**
   * Additive pyramid: `levels` is level 0's chain, and an object is the
   * union of its parts at every one of them, drawn as one level of detail
   * on the coarsest level's chunk grid.
   */
  union!: boolean;
}

/** A level's chunk grid, as Neuroglancer's slice-view machinery sees it. */
export interface ZarrVectorsChunkSpecification
  extends SliceViewChunkSpecification<Float32Array> {
  levelIndex: number;
}

export const ZARR_VECTORS_DENSE_RENDER_LAYER_RPC_ID =
  "zarr-vectors/DenseRenderLayer";
export const ZARR_VECTORS_DENSE_RENDER_LAYER_UPDATE_SOURCES_RPC_ID =
  "zarr-vectors/DenseRenderLayer.updateSources";
