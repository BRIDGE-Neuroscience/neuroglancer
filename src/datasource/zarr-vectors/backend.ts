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
import { ChunkPriorityTier, ChunkState } from "#src/chunk_manager/base.js";
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
import {
  objectMeshLayout,
  partitionMeshFragment,
} from "#src/datasource/zarr-vectors/mesh_lod.js";
import { ObjectReader } from "#src/datasource/zarr-vectors/object_reader.js";
import type { ZarrVectorsStoreAccess } from "#src/datasource/zarr-vectors/store.js";
import { kvStoreAccess } from "#src/datasource/zarr-vectors/store.js";
import type { SharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import { WithSharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import type {
  MultiscaleFragmentChunk,
  MultiscaleManifestChunk,
} from "#src/mesh/backend.js";
import {
  assignMultiscaleMeshFragmentData,
  MultiscaleMeshSource,
} from "#src/mesh/backend.js";
import { VertexPositionFormat } from "#src/mesh/base.js";
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
import type { RefCounted } from "#src/util/disposable.js";
import { vec3 } from "#src/util/geom.js";
import {
  getBasePriority,
  getPriorityTier,
} from "#src/visibility_priority/backend.js";
import type { RPC } from "#src/worker_rpc.js";
import { registerRPC, registerSharedObject } from "#src/worker_rpc.js";

const accesses = new Map<
  string,
  { access: ZarrVectorsStoreAccess; users: number }
>();

/**
 * One access (and shard-index cache) per store, shared by its sources while
 * any is alive. Dropped with the last, so a store rewritten in place is read
 * afresh once its layers are removed and added again.
 */
function storeAccess(
  owner: RefCounted & { sharedKvStoreContext: SharedKvStoreContextCounterpart },
  storeUrl: string,
): ZarrVectorsStoreAccess {
  let entry = accesses.get(storeUrl);
  if (entry === undefined) {
    entry = {
      access: kvStoreAccess(
        owner.sharedKvStoreContext.kvStoreContext,
        storeUrl,
      ),
      users: 0,
    };
    accesses.set(storeUrl, entry);
  }
  const used = entry;
  ++used.users;
  owner.registerDisposer(() => {
    if (--used.users === 0 && accesses.get(storeUrl) === used) {
      accesses.delete(storeUrl);
    }
  });
  return used.access;
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
        storeAccess(this, storeUrl),
        description,
        level,
        (position) => {
          const chunk = this.chunks.get(position.join());
          return (
            chunk !== undefined &&
            chunk.priorityTier !== ChunkPriorityTier.RECENT
          );
        },
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
        storeAccess(this, storeUrl),
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
  WithSharedKvStoreContextCounterpart(MultiscaleMeshSource),
  ZarrVectorsMeshSourceParameters,
) {
  private readers_: ObjectReader[] | undefined;
  private scales: Promise<Float32Array> | undefined;

  /** One reader per level of detail. */
  private get readers() {
    if (this.readers_ === undefined) {
      const { storeUrl, description, levels } = this.parameters;
      const access = storeAccess(this, storeUrl);
      this.readers_ = levels.map(
        (level) => new ObjectReader(access, description, level),
      );
    }
    return this.readers_;
  }

  private get baseChunk() {
    return this.parameters.levels[0].chunkShape;
  }

  /**
   * Each level of detail's typical edge length: Neuroglancer draws the
   * coarsest level whose edges are about a pixel (times the mesh resolution
   * setting). Measured on the coarsest level, whose chunks every view loads
   * first; finer levels scale with the square root of their vertex counts.
   * One level needs no measurement: any positive scale draws it.
   */
  private lodScales() {
    if (this.scales === undefined) {
      let failed = false;
      const promise = (async () => {
        const { levels } = this.parameters;
        const top = levels.length - 1;
        if (top === 0) return Float32Array.of(1);
        const measured = await this.readers[top].meanEdgeLength().catch(() => {
          failed = true;
          return undefined;
        });
        const topEdge =
          measured !== undefined && Number.isFinite(measured) && measured > 0
            ? measured
            : (Math.min(...this.baseChunk) * 2 ** top) / 64;
        const topCount = levels[top].vertexCount;
        const out = levels.map(({ vertexCount }, lod) =>
          vertexCount && topCount
            ? topEdge * Math.sqrt(topCount / vertexCount)
            : topEdge * 2 ** (lod - top),
        );
        for (let lod = 1; lod < out.length; ++lod) {
          out[lod] = Math.max(out[lod], out[lod - 1]);
        }
        return Float32Array.from(out);
      })();
      this.scales = promise;
      // Not kept when the measurement failed: the next manifest retries.
      const forget = () => {
        if (this.scales === promise) this.scales = undefined;
      };
      promise.then(() => failed && forget(), forget);
    }
    return this.scales;
  }

  async download(chunk: MultiscaleManifestChunk, signal: AbortSignal) {
    const base = this.baseChunk;
    const all = await Promise.all(
      this.readers.map((r) => r.chunksOf(chunk.objectId)),
    );
    const scales = await this.lodScales();
    signal.throwIfAborted();
    const layout = objectMeshLayout(
      all,
      base,
      this.parameters.gridOffset,
      scales,
    );
    gridOffsets.set(chunk, layout.gridOffset);
    const { octree, lodScales } = layout;
    const [origin, lower, upper] = [
      layout.chunkGridSpatialOrigin,
      layout.clipLowerBound,
      layout.clipUpperBound,
    ];
    chunk.manifest = {
      chunkShape: vec3.fromValues(base[0], base[1], base[2]),
      chunkGridSpatialOrigin: vec3.fromValues(origin[0], origin[1], origin[2]),
      clipLowerBound: vec3.fromValues(lower[0], lower[1], lower[2]),
      clipUpperBound: vec3.fromValues(upper[0], upper[1], upper[2]),
      octree,
      lodScales,
      vertexOffsets: new Float32Array(lodScales.length * 3),
    };
  }

  async downloadFragment(chunk: MultiscaleFragmentChunk, signal: AbortSignal) {
    const manifestChunk = chunk.manifestChunk!;
    const { octree } = manifestChunk.manifest!;
    const { lod, chunkIndex: row } = chunk;
    const offset = gridOffsets.get(manifestChunk) ?? this.parameters.gridOffset;
    const base = this.baseChunk;
    const grid = [0, 1, 2].map((d) => octree[row * 5 + d]);
    const size = base.map((c) => c * 2 ** lod);
    const { positions, indices } = await this.readers[lod].readMeshNode(
      manifestChunk.objectId,
      grid.map((x, d) => x - offset[d] / 2 ** lod),
      signal,
    );
    assignMultiscaleMeshFragmentData(
      chunk,
      partitionMeshFragment(
        positions,
        indices,
        grid.map((x, d) => (x - offset[d] / 2 ** lod) * size[d]),
        size,
        lod > 0,
      ),
      VertexPositionFormat.float32,
    );
  }
}

/** The grid offset each object's manifest was built with. */
const gridOffsets = new WeakMap<MultiscaleManifestChunk, number[]>();

// ------------------------------------------------------------ render layer

interface DenseAttachmentState {
  displayDimensionRenderInfo: DisplayDimensionRenderInfo;
  /** A cross-section panel, as the frontend says (it draws with the same). */
  isSliceView: boolean;
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
      isSliceView: false,
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
      const { isSliceView } = state;
      const renderScaleTarget = (
        isSliceView ? this.renderScaleTarget2d : this.renderScaleTarget3d
      ).value;
      forEachDenseChunkToLoad(
        projectionParameters,
        this.localPosition.value,
        transformedSources[0],
        this.densities,
        renderScaleTarget,
        isSliceView,
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
    attachment.state!.isSliceView = x.isSliceView;
    layer.chunkManager.scheduleUpdateChunkPriorities();
  },
);
