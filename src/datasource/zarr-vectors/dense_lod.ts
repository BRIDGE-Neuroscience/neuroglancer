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
 * Volume of a view frustum.  Neuroglancer's helper was renamed
 * (`getViewFrustrumVolume`, later `getViewFrustumVolume`), so a copy lives here.
 */
function viewFrustumVolume(projectionMat: mat4) {
  if (projectionMat[15] === 1) {
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
 * Smallest extent, in stored units, an axis is given when measuring volumes,
 * so a flat store (one plane of points) still has a density and a loaded
 * volume; both sides use it, so the plane's thickness cancels out.
 */
const MIN_EXTENT = 1;

/**
 * What each level would LOAD for a view, in stored units: a cross-section
 * loads every chunk the plane cuts, so a slab one chunk deep; a 3-d view
 * loads what its frustum covers. Also the volume of one stored unit in
 * display (canonical) units.
 */
function loadedVolumes(
  projectionParameters: ProjectionParameters,
  transformedSources: readonly TransformedSource<any, any>[],
  isSliceView: boolean,
): { loadedVolume: (level: number) => number; unitVolume: number } {
  const { projectionMat, viewMatrix, width, height } = projectionParameters;
  const base = transformedSources[0];
  const unitVolume = Math.abs(base.chunkLayout.detTransform);
  const unitLength = Math.cbrt(unitVolume);
  const extent: number[] = [];
  for (let i = 0; i < 3; ++i) {
    extent.push(
      Math.max(
        (base.upperClipDisplayBound[i] - base.lowerClipDisplayBound[i]) /
          unitLength,
        MIN_EXTENT,
      ),
    );
  }
  const sourceVolume = extent[0] * extent[1] * extent[2];
  // The view matrix scales display units to view units (a slice view's are
  // pixels); undo it to measure the view in display units.
  const viewDet = Math.abs(
    mat3.determinant(mat3FromMat4(tempMat3, viewMatrix)),
  );
  // Asked of the panel, not read off the projection: an orthographic 3-d
  // view has the same kind of projection matrix as a cross-section.
  if (isSliceView) {
    const pixelSize =
      2 / Math.abs(projectionMat[0]) / width / Math.cbrt(viewDet) / unitLength;
    const sortedExtent = [...extent].sort((a, b) => b - a);
    const viewArea =
      Math.min(width * pixelSize, sortedExtent[0]) *
      Math.min(height * pixelSize, sortedExtent[1]);
    return {
      unitVolume,
      loadedVolume: (level) => {
        const chunk = transformedSources[level].source.spec.chunkDataSize;
        const depth = Math.min(
          Math.cbrt(chunk[0] * chunk[1] * chunk[2]),
          sortedExtent[2],
        );
        return viewArea * depth;
      },
    };
  }
  const frustum = viewFrustumVolume(projectionMat) / viewDet / unitVolume;
  const volume = Math.min(frustum, sourceVolume);
  return { unitVolume, loadedVolume: () => volume };
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
  isSliceView: boolean,
): number {
  const numLevels = transformedSources.length;
  if (numLevels <= 1) return 0;
  const { width, height } = projectionParameters;
  const { loadedVolume } = loadedVolumes(
    projectionParameters,
    transformedSources,
    isSliceView,
  );
  // Vertex budget: one per `renderScaleTarget`^2 pixels.
  const budget = (width * height) / Math.max(renderScaleTarget, 1e-3) ** 2;
  for (let level = 0; level < numLevels - 1; ++level) {
    if (densities[level] * loadedVolume(level) <= budget) return level;
  }
  return numLevels - 1;
}

/**
 * A level's sample spacing for the layer's resolution histogram, as
 * Neuroglancer's spatially indexed annotations report theirs: physically
 * (the cube root of the volume per vertex), and on screen (pixels per
 * vertex if the vertices the view loads covered it evenly). A level is
 * drawn once its pixel spacing reaches the resolution slider's target.
 */
export function denseLevelSpacing(
  projectionParameters: ProjectionParameters,
  transformedSources: readonly TransformedSource<any, any>[],
  densities: readonly number[],
  level: number,
  isSliceView: boolean,
): { physicalSpacing: number; pixelSpacing: number } {
  const { width, height, displayDimensionRenderInfo } = projectionParameters;
  const { loadedVolume, unitVolume } = loadedVolumes(
    projectionParameters,
    transformedSources,
    isSliceView,
  );
  const density = densities[level];
  const physicalUnitVolume =
    unitVolume * prod3(displayDimensionRenderInfo.voxelPhysicalScales);
  return {
    physicalSpacing: Math.cbrt(physicalUnitVolume / density),
    pixelSpacing: Math.sqrt((width * height) / (density * loadedVolume(level))),
  };
}

/**
 * Visits the chunks to load for a view: the visible chunks of every level
 * the target level's view draws (`chains[target]`: the target itself, and
 * the coarser levels an additive target adds to), coarsest first so what
 * shows first is a whole view; then the coarsest level's as the stand-in
 * while they load, unless the view draws it anyway.
 */
export function forEachDenseChunkToLoad<
  Source extends TransformedSource<any, any>,
>(
  projectionParameters: ProjectionParameters,
  localPosition: Float32Array,
  transformedSources: readonly Source[],
  densities: readonly number[],
  chains: readonly (readonly number[])[],
  renderScaleTarget: number,
  isSliceView: boolean,
  callback: (source: Source, levelIndex: number, isTarget: boolean) => void,
) {
  if (transformedSources.length === 0) return;
  const target = selectDenseLevel(
    projectionParameters,
    transformedSources,
    densities,
    renderScaleTarget,
    isSliceView,
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
  const drawn = chains[target] ?? [target];
  for (const level of [...drawn].reverse()) visit(level, true);
  const coarsest = transformedSources.length - 1;
  if (!drawn.includes(coarsest)) visit(coarsest, false);
}

/**
 * Vertices per unit (stored) volume of what drawing each level shows, finest
 * first: the level's own vertices, and for an additive level the next
 * coarser level's complete content too.
 */
export function levelDensities(
  levels: readonly {
    vertexCount: number | undefined;
    chunkShape: number[];
    refinement?: "replace" | "add";
  }[],
  lowerBounds: readonly number[],
  upperBounds: readonly number[],
): number[] {
  let volume = 1;
  for (let i = 0; i < lowerBounds.length; ++i) {
    volume *= Math.max(upperBounds[i] - lowerBounds[i], MIN_EXTENT);
  }
  const n = levels.length;
  const adds = (i: number) => levels[i].refinement === "add" && i + 1 < n;
  // A count of 0 is a writer's placeholder as often as a fact, so it is
  // treated as unknown.
  const own = levels.map(({ vertexCount }) =>
    vertexCount !== undefined && vertexCount > 0
      ? vertexCount / volume
      : undefined,
  );
  const known: (number | undefined)[] = new Array(n).fill(undefined);
  for (let i = n - 1; i >= 0; --i) {
    if (own[i] === undefined) continue;
    if (!adds(i)) known[i] = own[i];
    else if (known[i + 1] !== undefined) known[i] = own[i]! + known[i + 1]!;
  }
  // An unknown level is estimated from a known neighbour, assuming density
  // falls with the cube of the chunk growth (or halves when chunks do not
  // grow); with no count at all, from a guess.
  const ratio = (finer: number) => {
    const growth =
      prod3(levels[finer + 1].chunkShape as any) /
      prod3(levels[finer].chunkShape as any);
    return growth > 1 ? growth : 2;
  };
  const out = [...known];
  const firstKnown = known.findIndex((d) => d !== undefined);
  if (firstKnown === -1) {
    out[0] = 1e6 / volume;
  } else {
    for (let i = firstKnown - 1; i >= 0; --i) out[i] = out[i + 1]! * ratio(i);
  }
  for (let i = 1; i < n; ++i) {
    out[i] ??= out[i - 1]! / ratio(i - 1);
  }
  // An additive level's own count adds to the next level's, however that
  // was found.
  for (let i = n - 2; i >= 0; --i) {
    if (adds(i) && own[i] !== undefined) out[i] = own[i]! + out[i + 1]!;
  }
  // Coarser levels never hold more per unit volume than finer ones.
  for (let i = n - 2; i >= 0; --i) {
    out[i] = Math.max(out[i]!, out[i + 1]!);
  }
  return out as number[];
}
