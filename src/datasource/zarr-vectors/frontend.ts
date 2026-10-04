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
 * @file The `zarr-vectors:` data source. One store yields, for a
 * segmentation layer:
 *  - `""`         the dense overview of every object (`dense_frontend.ts`);
 *  - `objects`    selected objects at full resolution, as Neuroglancer
 *                 skeletons (curves, skeletons, graphs, points with objects);
 *  - `meshes`     selected objects as Neuroglancer meshes (mesh stores);
 *  - `properties` object attributes and groups as segment properties.
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
import {
  ZarrVectorsGeometryChunkSourceParameters,
  ZarrVectorsMeshSourceParameters,
  ZarrVectorsObjectSkeletonSourceParameters,
} from "#src/datasource/zarr-vectors/base.js";
import {
  attributeLayout,
  ZarrVectorsMultiscaleGeometrySource,
} from "#src/datasource/zarr-vectors/dense_frontend.js";
import { levelDensities } from "#src/datasource/zarr-vectors/dense_lod.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  meshGridOffset,
  meshLevels,
} from "#src/datasource/zarr-vectors/mesh_lod.js";
import {
  readObjectTable,
  readSegmentProperties,
} from "#src/datasource/zarr-vectors/objects.js";
import type {
  ZarrVectorsStore,
  ZarrVectorsStoreAccess,
} from "#src/datasource/zarr-vectors/store.js";
import {
  chunkIndexBounds,
  formatAttributesFragment,
  kvStoreAccess,
  openZarrVectorsStore,
  parseAttributesFragment,
  readJson,
} from "#src/datasource/zarr-vectors/store.js";
import { warnOnce } from "#src/datasource/zarr-vectors/util.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import {
  kvstoreEnsureDirectoryPipelineUrl,
  parseUrlSuffix,
  pipelineUrlJoin,
} from "#src/kvstore/url.js";
import { VertexPositionFormat } from "#src/mesh/base.js";
import { MultiscaleMeshSource } from "#src/mesh/frontend.js";
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
    this.vertexAttributes_ ??= new Map(
      attributeLayout(this.parameters.description).map((a) => [
        a.id,
        { dataType: DataType.FLOAT32, numComponents: a.components },
      ]),
    );
    return this.vertexAttributes_;
  }
}

export class ZarrVectorsMeshSource extends WithParameters(
  WithSharedKvStoreContext(MultiscaleMeshSource),
  ZarrVectorsMeshSourceParameters,
) {}

/** The `meshes` subsource: every usable level as a level of detail. */
async function meshSubsource(
  chunkManager: ChunkManager,
  context: SharedKvStoreContext,
  access: ZarrVectorsStoreAccess,
  storeUrl: string,
  store: ZarrVectorsStore,
  warnings: string[],
  signal: AbortSignal | undefined,
): Promise<DataSubsourceEntry> {
  const { levels, unused } = await meshLevels(
    store.levels,
    async (level) =>
      (await readJson(
        access.read,
        `${level.path}/links/0/zarr.json`,
        signal,
      )) !== undefined,
  );
  if (unused > 0) {
    warnings.push(
      `${unused} pyramid level(s) not used for meshes: each level of detail ` +
        "needs faces and chunks twice the previous level's " +
        "(zvtools pyramid --chunk-scale 2,2)",
    );
  }
  const parameters = Object.assign(new ZarrVectorsMeshSourceParameters(), {
    storeUrl,
    description: store.description,
    levels,
    gridOffset: meshGridOffset(
      store.lowerBounds,
      levels[0].chunkShape,
      levels.length,
    ),
  });
  return {
    id: "meshes",
    default: true,
    subsource: {
      mesh: chunkManager.getChunkSource(ZarrVectorsMeshSource, {
        sharedKvStoreContext: context,
        parameters,
        format: {
          fragmentRelativeVertices: false,
          vertexPositionFormat: VertexPositionFormat.float32,
        },
      }),
    },
  };
}

async function buildDataSource(
  chunkManager: ChunkManager,
  context: SharedKvStoreContext,
  storeUrl: string,
  selectedAttributes: string[] | undefined,
  signal: AbortSignal | undefined,
): Promise<DataSource> {
  const access = kvStoreAccess(context.kvStoreContext, storeUrl);
  const store = await openZarrVectorsStore(access, selectedAttributes, signal);
  const { description, lowerBounds, upperBounds } = store;
  // A coarse level that is known to hold nothing would be drawn as the
  // zoomed-out view and the loading stand-in, showing nothing.
  const levels = store.levels.filter(
    (level, i) =>
      i === 0 ||
      level.vertexCount !== 0 ||
      level.arrays.vertices.attributes?.nonempty_chunks?.length !== 0,
  );
  const warnings = [...store.warnings];

  const space = makeCoordinateSpace({
    rank: 3,
    names: store.axisNames,
    units: store.axisUnits,
    scales: Float64Array.from(store.axisScales),
    boundingBoxes: [
      makeIdentityTransformedBoundingBox({
        lowerBounds: Float64Array.from(lowerBounds),
        upperBounds: Float64Array.from(upperBounds),
      }),
    ],
  });
  let modelTransform = makeIdentityTransform(space);
  if (store.coordinateOffset !== undefined) {
    // The writer stores `world - coordinate_offset`; put the offset back.
    const transform = matrix.createIdentity(Float64Array, 4);
    transform.set(store.coordinateOffset, 12);
    modelTransform = { ...modelTransform, transform };
  }

  const dense = new ZarrVectorsMultiscaleGeometrySource(
    chunkManager,
    context,
    description,
    levels.map((level) => ({
      spec: {
        rank: 3,
        chunkDataSize: Float32Array.from(level.chunkShape),
        ...chunkIndexBounds(lowerBounds, upperBounds, level.chunkShape),
        lowerVoxelBound: Float32Array.from(lowerBounds),
        upperVoxelBound: Float32Array.from(upperBounds),
        levelIndex: level.index,
      },
      parameters: Object.assign(
        new ZarrVectorsGeometryChunkSourceParameters(),
        {
          storeUrl,
          description,
          level,
        },
      ),
    })),
    levelDensities(levels, lowerBounds, upperBounds),
  );
  const subsources: DataSubsourceEntry[] = [
    // Not one of Neuroglancer's mesh kinds: the zarr-vectors segmentation
    // layer (`layer.ts`) recognises and draws it.
    { id: "", default: true, subsource: { mesh: dense as any } },
  ];

  const table = description.hasObjects
    ? await readObjectTable(access, levels[0].path, signal).catch((e) => {
        warnings.push(
          `object index unreadable: ${e instanceof Error ? e.message : e}`,
        );
        return undefined;
      })
    : undefined;
  if (table !== undefined) {
    const params = { storeUrl, description, level: levels[0] };
    subsources.push(
      KIND_CAPABILITIES[description.geometryKind].primitive === "triangles"
        ? await meshSubsource(
            chunkManager,
            context,
            access,
            storeUrl,
            store,
            warnings,
            signal,
          )
        : {
            id: "objects",
            default: true,
            subsource: {
              mesh: chunkManager.getChunkSource(
                ZarrVectorsObjectSkeletonSource,
                {
                  sharedKvStoreContext: context,
                  parameters: Object.assign(
                    new ZarrVectorsObjectSkeletonSourceParameters(),
                    params,
                  ),
                },
              ),
            },
          },
    );
    const properties = await readSegmentProperties(
      access,
      levels[0].path,
      table,
      warnings,
      signal,
    );
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
  for (const w of warnings) warnOnce(`${storeUrl}: ${w}`);
  return {
    modelTransform,
    subsources,
    canonicalUrl: `${storeUrl}|zarr-vectors:${formatAttributesFragment(selectedAttributes)}`,
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
    const storeUrl = kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(
        kvstoreEnsureDirectoryPipelineUrl(options.kvStoreUrl),
        authorityAndPath ?? "",
      ),
    );
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
