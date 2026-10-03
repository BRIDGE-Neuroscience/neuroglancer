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
 * @file The `zarr-vectors:` data source.
 *
 * One store URL yields up to four subsources for a segmentation layer:
 *  - `""`        the dense overview of every object, from spatial chunks;
 *  - `objects`   selected objects at full resolution, through Neuroglancer's
 *                own skeleton layer (curves, skeletons, graphs);
 *  - `meshes`    selected objects through Neuroglancer's mesh layer (meshes);
 *  - `properties` object attributes and groups as segment properties, so the
 *                Seg tab can filter (`length>40`, `#bundle`) and colour.
 */

import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
  makeCoordinateSpace,
  makeIdentityTransform,
  makeIdentityTransformedBoundingBox,
} from "#src/coordinate_transform.js";
import type {
  DataSource,
  DataSourceLookupResult,
  DataSubsourceEntry,
  GetKvStoreBasedDataSourceOptions,
  KvStoreBasedDataSourceProvider,
} from "#src/datasource/index.js";
import type { ZarrVectorsChunkSpecification } from "#src/datasource/zarr-vectors/base.js";
import {
  ZarrVectorsGeometryChunkSourceParameters,
  ZarrVectorsMeshSourceParameters,
  ZarrVectorsObjectSkeletonSourceParameters,
} from "#src/datasource/zarr-vectors/base.js";
import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import { hasTangentAttribute } from "#src/datasource/zarr-vectors/chunk_pipeline.js";
import {
  attributeLayout,
  ZarrVectorsMultiscaleGeometrySource,
} from "#src/datasource/zarr-vectors/dense_frontend.js";
import { levelDensities } from "#src/datasource/zarr-vectors/dense_lod.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  readObjectTable,
  readSegmentProperties,
} from "#src/datasource/zarr-vectors/objects.js";
import type { ZarrVectorsStore } from "#src/datasource/zarr-vectors/store.js";
import { openZarrVectorsStore } from "#src/datasource/zarr-vectors/store.js";
import {
  computeChunkIndexBounds,
  formatAttributesFragment,
  parseAttributesFragment,
} from "#src/datasource/zarr-vectors/store_metadata.js";
import type { ZarrArrayRead } from "#src/datasource/zarr-vectors/zarr_array.js";
import { ShardIndexCache } from "#src/datasource/zarr-vectors/zarr_array.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import {
  joinBaseUrlAndPath,
  kvstoreEnsureDirectoryPipelineUrl,
  parseUrlSuffix,
  pipelineUrlJoin,
} from "#src/kvstore/url.js";
import { MeshSource } from "#src/mesh/frontend.js";
import { SegmentPropertyMap } from "#src/segmentation_display_state/property_map.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import { SkeletonSource } from "#src/skeleton/frontend.js";
import { DataType } from "#src/util/data_type.js";
import * as matrix from "#src/util/matrix.js";

export class ZarrVectorsObjectSkeletonSource extends WithParameters(
  WithSharedKvStoreContext(SkeletonSource),
  ZarrVectorsObjectSkeletonSourceParameters,
) {
  private vertexAttributes_: Map<string, VertexAttributeInfo> | undefined;
  get vertexAttributes() {
    if (this.vertexAttributes_ === undefined) {
      const map = new Map<string, VertexAttributeInfo>();
      for (const a of attributeLayout(this.parameters.description)) {
        map.set(a.id, {
          dataType: DataType.FLOAT32,
          numComponents: a.components,
        });
      }
      this.vertexAttributes_ = map;
    }
    return this.vertexAttributes_;
  }
}

export class ZarrVectorsMeshSource extends WithParameters(
  WithSharedKvStoreContext(MeshSource),
  ZarrVectorsMeshSourceParameters,
) {}

function frontendAccess(context: SharedKvStoreContext, storeUrl: string) {
  const read: ZarrArrayRead = async (path, options) => {
    const response = await context.kvStoreContext.read(
      joinBaseUrlAndPath(storeUrl, path),
      { signal: options.signal, byteRange: options.byteRange },
    );
    if (response === undefined) return undefined;
    return new Uint8Array(await response.response.arrayBuffer());
  };
  const listDirectories = async (path: string, signal?: AbortSignal) => {
    const response = await context.kvStoreContext.list(
      joinBaseUrlAndPath(storeUrl, `${path}/`),
      { responseKeys: "suffix", signal },
    );
    return response.directories
      .map((d) => d.replace(/\/$/, ""))
      .filter((d) => d !== "");
  };
  return { read, listDirectories, shardIndexes: new ShardIndexCache() };
}

function resolveUrl(options: GetKvStoreBasedDataSourceOptions) {
  const { authorityAndPath, query, fragment } = parseUrlSuffix(
    options.url.suffix,
  );
  if (query) {
    throw new Error(
      `Invalid URL ${JSON.stringify(options.url.url)}: query parameters are not supported`,
    );
  }
  let selectedAttributes: string[] | undefined;
  try {
    selectedAttributes = parseAttributesFragment(fragment);
  } catch (e) {
    throw new Error(
      `Invalid URL ${JSON.stringify(options.url.url)}: ${(e as Error).message}`,
    );
  }
  return {
    storeUrl: kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(
        kvstoreEnsureDirectoryPipelineUrl(options.kvStoreUrl),
        authorityAndPath ?? "",
      ),
    ),
    selectedAttributes,
  };
}

function geometryDescription(
  store: ZarrVectorsStore,
): ZarrVectorsGeometryDescription {
  return {
    rank: store.rank,
    geometryKind: store.geometryKind,
    linksConvention: store.linksConvention,
    linkWidth: store.linkWidth,
    linkedSkeletonLayout: store.linkedSkeletonLayout,
    attributes: store.attributes,
    vertexIdAttribute: store.vertexIdAttribute,
    hasObjects: store.hasObjects,
  };
}

function chunkSpecification(
  store: ZarrVectorsStore,
  levelIndex: number,
): ZarrVectorsChunkSpecification {
  const { rank, lowerBounds, upperBounds } = store;
  const chunkShape = store.levels[levelIndex].chunkShape;
  const chunkDataSize = Float32Array.from(chunkShape);
  // Float arithmetic: a sub-unit chunk (0.5 mm) must not truncate to zero.
  const { lowerChunkBound, upperChunkBound } = computeChunkIndexBounds(
    lowerBounds,
    upperBounds,
    chunkShape,
  );
  return {
    rank,
    chunkDataSize,
    lowerChunkBound,
    upperChunkBound,
    lowerVoxelBound: Float32Array.from(lowerBounds),
    upperVoxelBound: Float32Array.from(upperBounds),
    levelIndex,
  };
}

const warned = new Set<string>();
function warnOnce(storeUrl: string, message: string) {
  const key = `${storeUrl}|${message}`;
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(`zarr-vectors (${storeUrl}): ${message}`);
}

async function buildDataSource(
  chunkManager: ChunkManager,
  context: SharedKvStoreContext,
  storeUrl: string,
  selectedAttributes: string[] | undefined,
  signal: AbortSignal | undefined,
): Promise<DataSource> {
  const access = frontendAccess(context, storeUrl);
  const store = await openZarrVectorsStore(
    access,
    storeUrl,
    selectedAttributes,
    signal,
  );
  if (store.levels.length === 0) {
    throw new Error("the store has no level this viewer can read");
  }
  const description = geometryDescription(store);
  const caps = KIND_CAPABILITIES[store.geometryKind];

  const rank = store.rank;
  const space = makeCoordinateSpace({
    rank,
    names: store.axisNames,
    units: store.axisUnits,
    scales: Float64Array.from(store.axisScales),
    boundingBoxes: [
      makeIdentityTransformedBoundingBox({
        lowerBounds: Float64Array.from(store.lowerBounds),
        upperBounds: Float64Array.from(store.upperBounds),
      }),
    ],
  });
  let modelTransform = makeIdentityTransform(space);
  if (store.coordinateOffset !== undefined) {
    // The writer stores `world - coordinate_offset`; put the offset back.
    const transform = matrix.createIdentity(Float64Array, rank + 1);
    for (let i = 0; i < rank; ++i) {
      transform[(rank + 1) * rank + i] = store.coordinateOffset[i];
    }
    modelTransform = { ...modelTransform, transform };
  }

  const subsources: DataSubsourceEntry[] = [];
  const densities = levelDensities(
    store.levels,
    store.lowerBounds,
    store.upperBounds,
  );
  const dense = new ZarrVectorsMultiscaleGeometrySource(
    chunkManager,
    context,
    description,
    store.levels.map((level) => {
      const parameters = new ZarrVectorsGeometryChunkSourceParameters();
      parameters.storeUrl = storeUrl;
      parameters.description = description;
      parameters.level = level;
      return { spec: chunkSpecification(store, level.index), parameters };
    }),
    densities,
  );
  subsources.push({
    id: "",
    default: true,
    // The dense source is not one of Neuroglancer's mesh kinds; the
    // zarr-vectors segmentation layer recognises and draws it.
    subsource: { mesh: dense as any },
  });

  const finest = store.levels[0];
  if (store.hasObjects) {
    const table = await readObjectTable(access, finest.path, signal).catch(
      (e) => {
        warnOnce(
          storeUrl,
          `object index unreadable: ${e instanceof Error ? e.message : e}`,
        );
        return undefined;
      },
    );
    if (table !== undefined) {
      if (caps.primitive === "triangles") {
        const parameters = new ZarrVectorsMeshSourceParameters();
        parameters.storeUrl = storeUrl;
        parameters.description = description;
        parameters.level = finest;
        subsources.push({
          id: "meshes",
          default: true,
          subsource: {
            mesh: chunkManager.getChunkSource(ZarrVectorsMeshSource, {
              sharedKvStoreContext: context,
              parameters,
            }),
          },
        });
      } else {
        const parameters = new ZarrVectorsObjectSkeletonSourceParameters();
        parameters.storeUrl = storeUrl;
        parameters.description = description;
        parameters.level = finest;
        subsources.push({
          id: "objects",
          default: true,
          subsource: {
            mesh: chunkManager.getChunkSource(ZarrVectorsObjectSkeletonSource, {
              sharedKvStoreContext: context,
              parameters,
            }),
          },
        });
      }
      const warnings: string[] = [];
      const properties = await readSegmentProperties(
        access,
        finest.path,
        table,
        warnings,
        signal,
      );
      for (const w of warnings) warnOnce(storeUrl, w);
      if (properties !== undefined) {
        subsources.push({
          id: "properties",
          default: true,
          subsource: {
            segmentPropertyMap: new SegmentPropertyMap({
              inlineProperties: properties,
            }),
          },
        });
      }
    }
  }
  for (const w of store.warnings) warnOnce(storeUrl, w);
  if (!hasTangentAttribute(description) && caps.primitive === "lines") {
    warnOnce(storeUrl, "no direction is available for this geometry");
  }
  return {
    modelTransform,
    subsources,
    canonicalUrl:
      `${storeUrl}|zarr-vectors:` +
      formatAttributesFragment(selectedAttributes),
  };
}

export class ZarrVectorsDataSource implements KvStoreBasedDataSourceProvider {
  get scheme() {
    return "zarr-vectors";
  }
  get expectsDirectory() {
    return true;
  }
  get description() {
    return "Zarr Vectors store (points, lines, skeletons, graphs, meshes)";
  }

  async get(
    options: GetKvStoreBasedDataSourceOptions,
  ): Promise<DataSourceLookupResult> {
    const { storeUrl, selectedAttributes } = resolveUrl(options);
    return options.registry.chunkManager.memoize.getAsync(
      {
        type: "zarr-vectors:get",
        url: storeUrl,
        attributes: selectedAttributes?.join(","),
      },
      options,
      (progressOptions) =>
        buildDataSource(
          options.registry.chunkManager,
          options.registry.sharedKvStoreContext,
          storeUrl,
          selectedAttributes,
          progressOptions.signal,
        ),
    );
  }
}
