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
 * @file Which pyramid level the dense layer draws, shared by the frontend
 * (drawing) and the chunk worker (requesting), so they agree.
 *
 * zarr-vectors pyramids are REPLACEMENT pyramids: each level holds the same
 * objects at a coarser resolution (or a sparser subset), so exactly one level
 * is drawn per region.  The level is the coarsest whose vertex density meets
 * the target set by the render-scale slider (vertices per pixel), computed the
 * way Neuroglancer's spatially indexed annotations do it.  While its chunks
 * load, the coarsest level stands in.
 */

import type { ProjectionParameters } from "#src/projection_parameters.js";
import type { TransformedSource } from "#src/sliceview/base.js";
import { forEachVisibleVolumetricChunk } from "#src/sliceview/base.js";
import type { mat4 } from "#src/util/geom.js";
import { mat3, mat3FromMat4, prod3 } from "#src/util/geom.js";

const tempMat3 = mat3.create();

/**
 * Volume of a view's frustum.  Kept here rather than imported because its
 * name differs between Neuroglancer versions (`getViewFrustrumVolume`, later
 * `getViewFrustumVolume`).
 */
function viewFrustumVolume(projectionMat: mat4) {
  if (projectionMat[15] === 1) {
    // Orthographic.
    return (
      (2 / Math.abs(projectionMat[10])) *
      (2 / Math.abs(projectionMat[0])) *
      (2 / Math.abs(projectionMat[5]))
    );
  }
  const a = projectionMat[10];
  const b = projectionMat[14];
  const near = (2 * b) / (2 * a - 2);
  const far = ((a - 1) * near) / (a + 1);
  const baseArea = 4 / (projectionMat[0] * projectionMat[5]);
  return (baseArea / 3) * (Math.abs(far) ** 3 - Math.abs(near) ** 3);
}

/**
 * Index (0 = finest) of the level to draw for this view.  `densities` are
 * vertices per unit volume of each level, finest first.
 */
export function selectDenseLevel(
  projectionParameters: ProjectionParameters,
  transformedSources: readonly TransformedSource<any, any>[],
  densities: readonly number[],
  renderScaleTarget: number,
): number {
  const numLevels = transformedSources.length;
  if (numLevels <= 1) return 0;
  const { projectionMat, viewMatrix, width, height } = projectionParameters;
  const base = transformedSources[0];
  // Volume and length of one stored unit in display (canonical) units.
  const unitVolume = Math.abs(base.chunkLayout.detTransform);
  const unitLength = Math.cbrt(unitVolume);
  const extent: number[] = [];
  for (let i = 0; i < 3; ++i) {
    extent.push(
      (base.upperClipDisplayBound[i] - base.lowerClipDisplayBound[i]) /
        unitLength,
    );
  }
  const sourceVolume = extent[0] * extent[1] * extent[2];
  // What each level would LOAD for this view, in stored units.  A
  // cross-section loads every chunk the plane cuts, so a slab one chunk deep;
  // a 3-d view loads what its frustum covers.
  // The view matrix scales display units to view units (a slice view's are
  // pixels); undo it to measure the view in display units.
  const viewDet = Math.abs(
    mat3.determinant(mat3FromMat4(tempMat3, viewMatrix)),
  );
  let loadedVolume: (level: number) => number;
  if (projectionMat[15] === 1) {
    const pixelSize =
      2 / Math.abs(projectionMat[0]) / width / Math.cbrt(viewDet) / unitLength;
    const sortedExtent = [...extent].sort((a, b) => b - a);
    const viewArea =
      Math.min(width * pixelSize, sortedExtent[0]) *
      Math.min(height * pixelSize, sortedExtent[1]);
    loadedVolume = (level) => {
      const chunk = transformedSources[level].source.spec.chunkDataSize;
      const depth = Math.min(
        Math.cbrt(chunk[0] * chunk[1] * chunk[2]),
        sortedExtent[2],
      );
      return viewArea * depth;
    };
  } else {
    const frustum = viewFrustumVolume(projectionMat) / viewDet / unitVolume;
    const volume = Math.min(frustum, sourceVolume);
    loadedVolume = () => volume;
  }
  // Vertex budget: one per `renderScaleTarget`^2 pixels.
  const budget = (width * height) / Math.max(renderScaleTarget, 1e-3) ** 2;
  for (let level = 0; level < numLevels - 1; ++level) {
    if (densities[level] * loadedVolume(level) <= budget) return level;
  }
  return numLevels - 1;
}

/**
 * Visits the chunks to load for a view: the target level's visible chunks,
 * then the coarsest level's as the stand-in while they load.
 */
export function forEachDenseChunkToLoad<
  Source extends TransformedSource<any, any>,
>(
  projectionParameters: ProjectionParameters,
  localPosition: Float32Array,
  transformedSources: readonly Source[],
  densities: readonly number[],
  renderScaleTarget: number,
  callback: (source: Source, levelIndex: number, isTarget: boolean) => void,
) {
  if (transformedSources.length === 0) return;
  const target = selectDenseLevel(
    projectionParameters,
    transformedSources,
    densities,
    renderScaleTarget,
  );
  const visit = (levelIndex: number, isTarget: boolean) => {
    const tsource = transformedSources[levelIndex];
    forEachVisibleVolumetricChunk(
      projectionParameters,
      localPosition,
      tsource,
      () => callback(tsource, levelIndex, isTarget),
    );
  };
  visit(target, true);
  const coarsest = transformedSources.length - 1;
  if (coarsest !== target) visit(coarsest, false);
}

/** Vertices per unit (stored) volume of each level, finest first. */
export function levelDensities(
  levels: readonly { vertexCount: number | undefined; chunkShape: number[] }[],
  lowerBounds: readonly number[],
  upperBounds: readonly number[],
): number[] {
  let volume = 1;
  for (let i = 0; i < lowerBounds.length; ++i) {
    volume *= Math.max(upperBounds[i] - lowerBounds[i], 1e-9);
  }
  const out: number[] = [];
  for (let i = 0; i < levels.length; ++i) {
    const { vertexCount, chunkShape } = levels[i];
    if (vertexCount !== undefined && vertexCount > 0) {
      out.push(vertexCount / volume);
      continue;
    }
    // No count: assume density falls with the cube of the chunk growth, and
    // halves per level when chunks do not grow.
    const prev = out[i - 1];
    if (prev === undefined) {
      out.push(1e6 / volume);
      continue;
    }
    const growth =
      prod3(chunkShape as any) / prod3(levels[i - 1].chunkShape as any);
    out.push(prev / (growth > 1 ? growth : 2));
  }
  return out;
}
