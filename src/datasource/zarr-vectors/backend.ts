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
 * @file Chunk-worker side of the zarr-vectors datasource: spatial chunks for
 * the dense layer, whole objects for Neuroglancer's own skeleton and mesh
 * layers, and the render-layer backend that decides which chunks to load.
 */

import {
  WithParameters,
  withChunkManager,
} from "#src/chunk_manager/backend.js";
import { ChunkState } from "#src/chunk_manager/base.js";
import type { ZarrVectorsChunkSpecification } from "#src/datasource/zarr-vectors/base.js";
import {
  ZARR_VECTORS_DENSE_RENDER_LAYER_RPC_ID,
  ZARR_VECTORS_DENSE_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  ZarrVectorsGeometryChunkSourceParameters,
  ZarrVectorsMeshSourceParameters,
  ZarrVectorsObjectSkeletonSourceParameters,
} from "#src/datasource/zarr-vectors/base.js";
import type { DenseChunkData } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import { LevelPipeline } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import { forEachDenseChunkToLoad } from "#src/datasource/zarr-vectors/dense_lod.js";
import { ObjectReader } from "#src/datasource/zarr-vectors/object_reader.js";
import type { ZarrVectorsStoreAccess } from "#src/datasource/zarr-vectors/store.js";
import { kvStoreAccess } from "#src/datasource/zarr-vectors/store.js";
import type { SharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import { WithSharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import type { FragmentChunk, ManifestChunk } from "#src/mesh/backend.js";
import { assignMeshFragmentData, MeshSource } from "#src/mesh/backend.js";
import type { DisplayDimensionRenderInfo } from "#src/navigation_state.js";
import { validateDisplayDimensionRenderInfoProperty } from "#src/navigation_state.js";
import type {
  RenderedViewBackend,
  RenderLayerBackendAttachment,
} from "#src/render_layer_backend.js";
import { RenderLayerBackend } from "#src/render_layer_backend.js";
import type { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type { SkeletonChunk } from "#src/skeleton/backend.js";
import { SkeletonSource } from "#src/skeleton/backend.js";
import {
  deserializeTransformedSources,
  SCALE_PRIORITY_MULTIPLIER,
  SliceViewChunk,
  SliceViewChunkSourceBackend,
} from "#src/sliceview/backend.js";
import type { TransformedSource } from "#src/sliceview/base.js";
import {
  getBasePriority,
  getPriorityTier,
} from "#src/visibility_priority/backend.js";
import type { RPC } from "#src/worker_rpc.js";
import { registerRPC, registerSharedObject } from "#src/worker_rpc.js";

const accesses = new Map<string, ZarrVectorsStoreAccess>();

/** One access (and shard-index cache) per store, shared by its sources. */
function storeAccess(
  context: SharedKvStoreContextCounterpart,
  storeUrl: string,
): ZarrVectorsStoreAccess {
  let access = accesses.get(storeUrl);
  if (access === undefined) {
    access = kvStoreAccess(context.kvStoreContext, storeUrl);
    accesses.set(storeUrl, access);
  }
  return access;
}

// ------------------------------------------------------------ dense chunks

export class ZarrVectorsDenseChunk extends SliceViewChunk {
  data: DenseChunkData | undefined;

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    const { data } = this;
    if (data !== undefined) {
      msg.data = data;
      transfers.push(
        data.positions.buffer,
        data.segmentIds.buffer,
        data.edges.buffer,
        ...data.attributes.map((a) => a.buffer),
      );
    }
    this.data = undefined;
  }

  downloadSucceeded() {
    const { data } = this;
    let bytes = 0;
    if (data !== undefined) {
      bytes =
        data.positions.byteLength +
        data.segmentIds.byteLength +
        data.edges.byteLength;
      for (const a of data.attributes) bytes += a.byteLength;
    }
    this.systemMemoryBytes = bytes;
    this.gpuMemoryBytes = bytes;
    super.downloadSucceeded();
  }

  freeSystemMemory() {
    this.data = undefined;
  }
}

@registerSharedObject()
export class ZarrVectorsGeometryChunkSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(
    SliceViewChunkSourceBackend<
      ZarrVectorsChunkSpecification,
      ZarrVectorsDenseChunk
    >,
  ),
  ZarrVectorsGeometryChunkSourceParameters,
) {
  private pipeline_: LevelPipeline | undefined;

  private get pipeline() {
    if (this.pipeline_ === undefined) {
      const { storeUrl, description, level } = this.parameters;
      this.pipeline_ = new LevelPipeline(
        storeAccess(this.sharedKvStoreContext, storeUrl),
        description,
        level,
      );
    }
    return this.pipeline_;
  }

  async download(chunk: ZarrVectorsDenseChunk, signal: AbortSignal) {
    chunk.data = await this.pipeline.download(
      Array.from(chunk.chunkGridPosition, Math.round),
      signal,
    );
  }
}
ZarrVectorsGeometryChunkSourceBackend.prototype.chunkConstructor =
  ZarrVectorsDenseChunk;

// ------------------------------------------------------------ objects

@registerSharedObject()
export class ZarrVectorsObjectSkeletonSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(SkeletonSource),
  ZarrVectorsObjectSkeletonSourceParameters,
) {
  private reader_: ObjectReader | undefined;
  private get reader() {
    if (this.reader_ === undefined) {
      const { storeUrl, description, level } = this.parameters;
      this.reader_ = new ObjectReader(
        storeAccess(this.sharedKvStoreContext, storeUrl),
        description,
        level,
      );
    }
    return this.reader_;
  }

  async download(chunk: SkeletonChunk, signal: AbortSignal) {
    const skeleton = await this.reader.readSkeleton(chunk.objectId, signal);
    chunk.vertexPositions = skeleton.positions;
    chunk.indices = skeleton.edges;
    chunk.vertexAttributes = skeleton.attributes;
  }
}

@registerSharedObject()
export class ZarrVectorsMeshSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(MeshSource),
  ZarrVectorsMeshSourceParameters,
) {
  private reader_: ObjectReader | undefined;
  private get reader() {
    if (this.reader_ === undefined) {
      const { storeUrl, description, level } = this.parameters;
      this.reader_ = new ObjectReader(
        storeAccess(this.sharedKvStoreContext, storeUrl),
        description,
        level,
      );
    }
    return this.reader_;
  }

  async downloadFragmentIds(chunk: ManifestChunk, signal: AbortSignal) {
    signal.throwIfAborted();
    chunk.fragmentIds = await this.reader.meshFragmentKeys(chunk.objectId);
  }

  async downloadFragment(chunk: FragmentChunk, signal: AbortSignal) {
    const mesh = await this.reader.readMeshFragment(
      chunk.manifestChunk!.objectId,
      chunk.fragmentId!,
      signal,
    );
    assignMeshFragmentData(chunk, {
      vertexPositions: mesh.positions,
      indices: mesh.indices,
    });
  }

  download(chunk: ManifestChunk, signal: AbortSignal) {
    return this.downloadFragmentIds(chunk, signal);
  }
}

// ------------------------------------------------------------ render layer

interface DenseAttachmentState {
  displayDimensionRenderInfo: DisplayDimensionRenderInfo;
  transformedSources: TransformedSource<
    ZarrVectorsDenseRenderLayerBackend,
    ZarrVectorsGeometryChunkSourceBackend
  >[][];
}

@registerSharedObject(ZARR_VECTORS_DENSE_RENDER_LAYER_RPC_ID)
export class ZarrVectorsDenseRenderLayerBackend extends withChunkManager(
  RenderLayerBackend,
) {
  localPosition: SharedWatchableValue<Float32Array>;
  renderScaleTarget2d: SharedWatchableValue<number>;
  renderScaleTarget3d: SharedWatchableValue<number>;
  densities: number[];

  /** The 3-D target, for code that treats this as a volumetric render layer. */
  get renderScaleTarget() {
    return this.renderScaleTarget3d;
  }

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.localPosition = rpc.get(options.localPosition);
    this.renderScaleTarget2d = rpc.get(options.renderScaleTarget2d);
    this.renderScaleTarget3d = rpc.get(options.renderScaleTarget3d);
    this.densities = options.densities;
    const schedule = () => this.chunkManager.scheduleUpdateChunkPriorities();
    for (const value of [
      this.localPosition,
      this.renderScaleTarget2d,
      this.renderScaleTarget3d,
    ]) {
      this.registerDisposer(value.changed.add(schedule));
    }
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() =>
        this.recomputeChunkPriorities(),
      ),
    );
  }

  attach(
    attachment: RenderLayerBackendAttachment<
      RenderedViewBackend,
      DenseAttachmentState
    >,
  ) {
    const schedule = () => this.chunkManager.scheduleUpdateChunkPriorities();
    const { view } = attachment;
    attachment.registerDisposer(schedule);
    attachment.registerDisposer(
      view.projectionParameters.changed.add(schedule),
    );
    attachment.registerDisposer(view.visibility.changed.add(schedule));
    attachment.state = {
      displayDimensionRenderInfo:
        view.projectionParameters.value.displayDimensionRenderInfo,
      transformedSources: [],
    };
  }

  private recomputeChunkPriorities() {
    this.chunkManager.registerLayer(this);
    for (const attachment of this.attachments.values()) {
      const { view } = attachment;
      const visibility = view.visibility.value;
      if (visibility === Number.NEGATIVE_INFINITY) continue;
      const state = attachment.state as DenseAttachmentState;
      const { transformedSources } = state;
      const projectionParameters = view.projectionParameters.value;
      if (
        transformedSources.length === 0 ||
        !validateDisplayDimensionRenderInfoProperty(
          state,
          projectionParameters.displayDimensionRenderInfo,
        )
      ) {
        continue;
      }
      const priorityTier = getPriorityTier(visibility);
      const basePriority = getBasePriority(visibility);
      // A slice view has an orthographic, zero-depth frustum.
      const is2d = projectionParameters.projectionMat[15] === 1;
      const renderScaleTarget = (
        is2d ? this.renderScaleTarget2d : this.renderScaleTarget3d
      ).value;
      forEachDenseChunkToLoad(
        projectionParameters,
        this.localPosition.value,
        transformedSources[0],
        this.densities,
        renderScaleTarget,
        (tsource, _levelIndex, isTarget) => {
          const chunk = (
            tsource.source as ZarrVectorsGeometryChunkSourceBackend
          ).getChunk(tsource.curPositionInChunks);
          if (isTarget) {
            ++this.numVisibleChunksNeeded;
            if (chunk.state === ChunkState.GPU_MEMORY) {
              ++this.numVisibleChunksAvailable;
            }
          }
          // The coarse stand-in loads first: it is small and fills the view.
          this.chunkManager.requestChunk(
            chunk,
            priorityTier,
            basePriority + (isTarget ? 0 : SCALE_PRIORITY_MULTIPLIER),
          );
        },
      );
    }
  }
}

registerRPC(
  ZARR_VECTORS_DENSE_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  function (x) {
    const view = this.get(x.view) as RenderedViewBackend;
    const layer = this.get(x.layer) as ZarrVectorsDenseRenderLayerBackend;
    const attachment = layer.attachments.get(
      view,
    )! as RenderLayerBackendAttachment<
      RenderedViewBackend,
      DenseAttachmentState
    >;
    attachment.state!.transformedSources = deserializeTransformedSources<
      ZarrVectorsGeometryChunkSourceBackend,
      ZarrVectorsDenseRenderLayerBackend
    >(this, x.sources, layer);
    attachment.state!.displayDimensionRenderInfo = x.displayDimensionRenderInfo;
    layer.chunkManager.scheduleUpdateChunkPriorities();
  },
);
