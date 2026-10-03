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
 * @file Lets a segmentation layer draw the zarr-vectors dense overview.
 *
 * Neuroglancer's segmentation layer knows a fixed set of subsource kinds and
 * rejects the rest.  Rather than patch it, this module registers a subclass
 * under the same `"segmentation"` type name: it draws zarr-vectors' dense
 * subsource itself and hands everything else to the base class unchanged, so
 * saved states keep their layer type and other data behaves exactly as
 * before.  Registration happens when the zarr-vectors datasource loads, which
 * is the only time the subclass is needed.
 */

import type { ZarrVectorsDenseDisplayState } from "#src/datasource/zarr-vectors/dense_frontend.js";
import {
  PerspectiveViewZarrVectorsDenseLayer,
  SliceViewPanelZarrVectorsDenseLayer,
  ZarrVectorsMultiscaleGeometrySource,
} from "#src/datasource/zarr-vectors/dense_frontend.js";
import {
  DIRECTION_SHADER,
  KIND_CAPABILITIES,
} from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  registerLayerType,
  registerLayerTypeDetector,
  registerVolumeLayerType,
} from "#src/layer/index.js";
import type { LoadedDataSubsource } from "#src/layer/layer_data_source.js";
import { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { VolumeType } from "#src/sliceview/volume/base.js";
import { WatchableValue } from "#src/trackable_value.js";

/** Subsource ids that draw selected objects at full resolution. */
const OBJECT_SUBSOURCE_IDS = new Set(["objects", "meshes"]);

function denseSource(subsource: LoadedDataSubsource) {
  const { mesh } = subsource.subsourceEntry.subsource;
  return mesh instanceof ZarrVectorsMultiscaleGeometrySource ? mesh : undefined;
}

export class ZarrVectorsSegmentationUserLayer extends SegmentationUserLayer {
  private objectsDrawnElsewhere = new WatchableValue(false);

  activateDataSubsources(subsources: Iterable<LoadedDataSubsource>) {
    const rest: LoadedDataSubsource[] = [];
    const dense: LoadedDataSubsource[] = [];
    let objectsElsewhere = false;
    for (const subsource of subsources) {
      if (denseSource(subsource) !== undefined) {
        dense.push(subsource);
        continue;
      }
      if (OBJECT_SUBSOURCE_IDS.has(subsource.subsourceEntry.id)) {
        objectsElsewhere = true;
      }
      rest.push(subsource);
    }
    super.activateDataSubsources(rest);
    this.objectsDrawnElsewhere.value = objectsElsewhere;
    let directionDefault = dense.length > 0;
    for (const loadedSubsource of dense) {
      const source = denseSource(loadedSubsource)!;
      if (
        !KIND_CAPABILITIES[source.description.geometryKind].directionDefault
      ) {
        directionDefault = false;
      }
      loadedSubsource.activate(() => {
        // Inherit everything (including getters) from the layer's display
        // state; override only what differs per subsource.
        const displayState: ZarrVectorsDenseDisplayState = Object.assign(
          Object.create(this.displayState),
          {
            transform: loadedSubsource.getRenderLayerTransform(),
            localPosition: this.localPosition,
            renderScaleTarget2d: this.sliceViewRenderScaleTarget,
            objectsDrawnElsewhere: this.objectsDrawnElsewhere,
          },
        );
        const options = {
          chunkManager: this.manager.chunkManager,
          source,
          displayState,
        };
        loadedSubsource.addRenderLayer(
          new PerspectiveViewZarrVectorsDenseLayer(options),
        );
        loadedSubsource.addRenderLayer(
          new SliceViewPanelZarrVectorsDenseLayer(options),
        );
      }, this.displayState.segmentationGroupState.value);
    }
    if (directionDefault) this.adoptDirectionShader();
  }

  /**
   * Colour curves by direction unless the user has chosen a shader.  Moving
   * the default with the value keeps an untouched shader out of saved state.
   */
  private adoptDirectionShader() {
    const { shader } = this.displayState.skeletonRenderingOptions;
    if (shader.value !== shader.defaultValue) return;
    if (shader.value === DIRECTION_SHADER) return;
    shader.defaultValue = DIRECTION_SHADER;
    shader.value = DIRECTION_SHADER;
  }
}

registerLayerType(ZarrVectorsSegmentationUserLayer, "segmentation");
registerVolumeLayerType(
  VolumeType.SEGMENTATION,
  ZarrVectorsSegmentationUserLayer,
);
registerLayerTypeDetector((subsource) => {
  if (subsource.mesh instanceof ZarrVectorsMultiscaleGeometrySource) {
    return { layerConstructor: ZarrVectorsSegmentationUserLayer, priority: 2 };
  }
  return undefined;
});
