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
 * @file The dense overview of a zarr-vectors store: every object in view,
 * drawn from spatial chunks, inside an ordinary segmentation layer.
 *
 * Neuroglancer's skeleton layer draws one segment per draw call, which cannot
 * show a whole-brain tractogram.  This layer draws a whole chunk per call and
 * looks each vertex's segment up on the GPU, the way the volume segmentation
 * renderer colours voxels: the layer's ordinary segment state (selected
 * segments, segment colours, colour seed, `selectedAlpha` /
 * `notSelectedAlpha` / `objectAlpha`, `ignoreNullVisibleSet`) decides what
 * shows and how.  The layer's skeleton shader and its controls apply, with
 * the same `prop_<attribute>()` and `segmentColor()` API, so one shader serves
 * this layer and the full-resolution per-object skeletons.
 */

import { ChunkState } from "#src/chunk_manager/base.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import {
  ChunkRenderLayerFrontend,
  WithParameters,
} from "#src/chunk_manager/frontend.js";
import type {
  ZarrVectorsChunkSpecification,
  ZarrVectorsGeometryDescription,
} from "#src/datasource/zarr-vectors/base.js";
import {
  ZARR_VECTORS_DENSE_RENDER_LAYER_RPC_ID,
  ZARR_VECTORS_DENSE_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  ZarrVectorsGeometryChunkSourceParameters,
} from "#src/datasource/zarr-vectors/base.js";
import {
  addStringShaderSupport,
  setShaderControls,
} from "#src/datasource/zarr-vectors/compat.js";
import { selectDenseLevel } from "#src/datasource/zarr-vectors/dense_lod.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import type { HashMapUint64 } from "#src/gpu_hash/hash_table.js";
import { GPUHashTable, HashSetShaderManager } from "#src/gpu_hash/shader.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import type {
  LayerView,
  MouseSelectionState,
  VisibleLayerInfo,
} from "#src/layer/index.js";
import type { DisplayDimensionRenderInfo } from "#src/navigation_state.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { RenderLayerTransformOrError } from "#src/render_coordinate_transform.js";
import { get3dModelToDisplaySpaceMatrix } from "#src/render_coordinate_transform.js";
import type { RenderLayer } from "#src/renderlayer.js";
import {
  SegmentColorShaderManager,
  SegmentStatedColorShaderManager,
} from "#src/segment_color.js";
import { getVisibleSegments } from "#src/segmentation_display_state/base.js";
import type { SegmentationDisplayState3D } from "#src/segmentation_display_state/frontend.js";
import { registerRedrawWhenSegmentationDisplayState3DChanged } from "#src/segmentation_display_state/frontend.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type { SkeletonRenderingOptions } from "#src/skeleton/frontend.js";
import { SkeletonRenderMode } from "#src/skeleton/frontend.js";
import { forEachVisibleVolumetricChunk } from "#src/sliceview/base.js";
import type {
  FrontendTransformedSource,
  SliceViewSingleResolutionSource,
} from "#src/sliceview/frontend.js";
import {
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
  SliceViewChunk,
  SliceViewChunkSource,
} from "#src/sliceview/frontend.js";
import type { SliceViewPanelRenderContext } from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { registerNested, WatchableValue } from "#src/trackable_value.js";
import type { Uint64Map } from "#src/uint64_map.js";
import { DataType } from "#src/util/data_type.js";
import type { Borrowed } from "#src/util/disposable.js";
import { RefCounted } from "#src/util/disposable.js";
import { mat4 } from "#src/util/geom.js";
import type { AnyConstructor } from "#src/util/mixin.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import {
  defineCircleShader,
  drawCircles,
  initializeCircleShader,
} from "#src/webgl/circles.js";
import { glsl_COLORMAPS } from "#src/webgl/colormaps.js";
import type { GL } from "#src/webgl/context.js";
import type { WatchableShaderError } from "#src/webgl/dynamic_shader.js";
import {
  parameterizedEmitterDependentShaderGetter,
  shaderCodeWithLineDirective,
} from "#src/webgl/dynamic_shader.js";
import {
  defineLineShader,
  drawLines,
  initializeLineShader,
} from "#src/webgl/lines.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";
import { glsl_uint64 } from "#src/webgl/shader_lib.js";
import type { ShaderControlsBuilderState } from "#src/webgl/shader_ui_controls.js";
import {
  addControlsToBuilder,
  getFallbackBuilderState,
  parseShaderUiControls,
} from "#src/webgl/shader_ui_controls.js";

import {
  computeTextureFormat,
  OneDimensionalTextureAccessHelper,
  setOneDimensionalTextureData,
  TextureFormat,
} from "#src/webgl/texture_access.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

// ------------------------------------------------------------ chunks

const POSITION_FORMAT = computeTextureFormat(
  new TextureFormat(),
  DataType.FLOAT32,
  3,
);
// Segment ids are uploaded as two uint32 words per vertex, low word first,
// which is the layout of a UINT64 texel.
const SEGMENT_FORMAT = computeTextureFormat(
  new TextureFormat(),
  DataType.UINT64,
  1,
);

export class ZarrVectorsDenseChunk extends SliceViewChunk {
  numVertices = 0;
  numOwnVertices = 0;
  numEdges = 0;
  /** Kept on the CPU for picking (two uint32 per vertex). */
  segmentIds: Uint32Array | undefined;
  private uploads:
    | {
        positions: Float32Array;
        attributes: Float32Array[];
        edges: Uint32Array;
      }
    | undefined;
  positionTexture: WebGLTexture | null = null;
  segmentTexture: WebGLTexture | null = null;
  attributeTextures: (WebGLTexture | null)[] = [];
  edgeBuffer: GLBuffer | undefined;

  constructor(source: SliceViewChunkSource, x: any) {
    super(source, x);
    const data = x.data;
    if (data !== undefined) {
      this.numVertices = data.numVertices;
      this.numOwnVertices = data.numOwnVertices;
      this.numEdges = data.edges.length / 2;
      this.segmentIds = data.segmentIds;
      this.uploads = {
        positions: data.positions,
        attributes: data.attributes,
        edges: data.edges,
      };
    }
  }

  copyToGPU(gl: GL) {
    super.copyToGPU(gl);
    const uploads = this.uploads;
    if (uploads === undefined) return;
    const texture = (
      format: TextureFormat,
      data: Float32Array | Uint32Array,
    ) => {
      const t = gl.createTexture();
      gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, t);
      setOneDimensionalTextureData(gl, format, data);
      return t;
    };
    this.positionTexture = texture(POSITION_FORMAT, uploads.positions);
    this.segmentTexture = texture(SEGMENT_FORMAT, this.segmentIds!);
    const formats = (this.source as ZarrVectorsGeometryChunkSource)
      .attributeFormats;
    this.attributeTextures = uploads.attributes.map((a, i) =>
      texture(formats[i], a),
    );
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
    if (this.numEdges > 0) {
      this.edgeBuffer = GLBuffer.fromData(
        gl,
        uploads.edges,
        WebGL2RenderingContext.ARRAY_BUFFER,
        WebGL2RenderingContext.STATIC_DRAW,
      );
    }
  }

  freeGPUMemory(gl: GL) {
    super.freeGPUMemory(gl);
    for (const t of [
      this.positionTexture,
      this.segmentTexture,
      ...this.attributeTextures,
    ]) {
      gl.deleteTexture(t);
    }
    this.positionTexture = this.segmentTexture = null;
    this.attributeTextures = [];
    this.edgeBuffer?.dispose();
    this.edgeBuffer = undefined;
  }
}

class ZarrVectorsGeometryChunkSourceBase extends SliceViewChunkSource<ZarrVectorsChunkSpecification> {
  getChunk(x: any): SliceViewChunk {
    return new ZarrVectorsDenseChunk(this, x);
  }
}

export class ZarrVectorsGeometryChunkSource extends WithParameters(
  WithSharedKvStoreContext(ZarrVectorsGeometryChunkSourceBase),
  ZarrVectorsGeometryChunkSourceParameters,
) {
  private attributeFormats_: TextureFormat[] | undefined;

  get attributeFormats(): TextureFormat[] {
    if (this.attributeFormats_ === undefined) {
      this.attributeFormats_ = attributeLayout(this.parameters.description).map(
        (a) =>
          computeTextureFormat(
            new TextureFormat(),
            DataType.FLOAT32,
            a.components,
          ),
      );
    }
    return this.attributeFormats_;
  }
}

/** The per-vertex attributes a chunk carries, in upload order. */
export function attributeLayout(
  description: ZarrVectorsGeometryDescription,
): { id: string; components: number }[] {
  const layout = description.attributes.map((a) => ({
    id: a.id,
    components: a.components,
  }));
  if (KIND_CAPABILITIES[description.geometryKind].tangent !== undefined) {
    layout.push({ id: "tangent", components: 3 });
  }
  return layout;
}

/** All levels of a store, as one multiscale slice-view source. */
export class ZarrVectorsMultiscaleGeometrySource extends RefCounted {
  constructor(
    public chunkManager: Borrowed<ChunkManager>,
    public sharedKvStoreContext: SharedKvStoreContext,
    public description: ZarrVectorsGeometryDescription,
    public levels: {
      spec: ZarrVectorsChunkSpecification;
      parameters: ZarrVectorsGeometryChunkSourceParameters;
    }[],
    /** Vertices per unit volume per level (finest first). */
    public densities: number[],
  ) {
    super();
  }

  get rank() {
    return 3;
  }

  getSources(): SliceViewSingleResolutionSource<ZarrVectorsGeometryChunkSource>[][] {
    return [
      this.levels.map(({ spec, parameters }) => ({
        chunkSource: this.chunkManager.getChunkSource(
          ZarrVectorsGeometryChunkSource,
          {
            sharedKvStoreContext: this.sharedKvStoreContext,
            spec,
            parameters,
          },
        ),
        chunkToMultiscaleTransform: identityTransform(spec.rank),
      })),
    ];
  }
}

function identityTransform(rank: number) {
  const out = new Float32Array((rank + 1) ** 2);
  for (let i = 0; i <= rank; ++i) out[i * (rank + 1) + i] = 1;
  return out;
}

// ------------------------------------------------------------ render layer

export interface ZarrVectorsDenseDisplayState
  extends Omit<
    SegmentationDisplayState3D,
    "selectedAlpha" | "notSelectedAlpha"
  > {
  skeletonRenderingOptions: SkeletonRenderingOptions;
  shaderError: WatchableShaderError;
  selectedAlpha: WatchableValueInterface<number>;
  notSelectedAlpha: WatchableValueInterface<number>;
  ignoreNullVisibleSet: WatchableValueInterface<boolean>;
  localPosition: WatchableValueInterface<Float32Array>;
  /** `crossSectionRenderScale`: target pixels between vertices in 2-D. */
  renderScaleTarget2d: WatchableValueInterface<number>;
  /**
   * Whether the layer also draws the selected objects at full resolution
   * through Neuroglancer's skeleton layer; then they are left out here.
   */
  objectsDrawnElsewhere: WatchableValueInterface<boolean>;
}

const DEFAULT_FRAGMENT_MAIN = `void main() {
  emitDefault();
}
`;

const SHOW_ALL_FLAG = 1;
const HIDE_VISIBLE_FLAG = 2;

const tempMat4 = mat4.create();

interface DenseAttachmentState {
  transform: RenderLayerTransformOrError;
  displayDimensionRenderInfo: DisplayDimensionRenderInfo;
  modelMatrix: mat4 | undefined;
  sources: { readonly value: FrontendTransformedSource[][] };
}

class DenseRenderHelper extends RefCounted {
  private textureAccess = new OneDimensionalTextureAccessHelper("vertexData");
  private vertexIdHelper: VertexIdHelper;
  private visibleSegments = new HashSetShaderManager("visibleSegments");
  private segmentColor = new SegmentColorShaderManager("segmentColorHash");
  private statedColors = new SegmentStatedColorShaderManager("statedColor");
  readonly attributes: { id: string; components: number }[];
  readonly primitive: "lines" | "points";
  edgeShaderGetter;
  nodeShaderGetter;

  constructor(
    private gl: GL,
    private displayState: ZarrVectorsDenseDisplayState,
    description: ZarrVectorsGeometryDescription,
    private targetIsSliceView: boolean,
    fallbackParameters: WatchableValueInterface<ShaderControlsBuilderState>,
  ) {
    super();
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(gl));
    this.attributes = attributeLayout(description);
    this.primitive =
      KIND_CAPABILITIES[description.geometryKind].primitive === "lines"
        ? "lines"
        : "points";
    const common = {
      fallbackParameters,
      parameters:
        displayState.skeletonRenderingOptions.shaderControlState.builderState,
      shaderError: displayState.shaderError,
    };
    this.edgeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      gl,
      {
        memoizeKey: {
          type: "zarr-vectors/dense/edge",
          attributes: this.attributes,
          slice: targetIsSliceView,
        },
        ...common,
        defineShader: (
          builder: ShaderBuilder,
          state: ShaderControlsBuilderState,
        ) => this.defineShader(builder, state, true),
      },
    );
    this.nodeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      gl,
      {
        memoizeKey: {
          type: "zarr-vectors/dense/node",
          attributes: this.attributes,
          slice: targetIsSliceView,
        },
        ...common,
        defineShader: (
          builder: ShaderBuilder,
          state: ShaderControlsBuilderState,
        ) => this.defineShader(builder, state, false),
      },
    );
  }

  private fadeFactor() {
    return this.targetIsSliceView
      ? "(clamp(1.0 - 2.0 * abs(0.5 - gl_FragCoord.z), 0.0, 1.0))"
      : "(1.0)";
  }

  private defineShader(
    builder: ShaderBuilder,
    state: ShaderControlsBuilderState,
    edges: boolean,
  ) {
    if (state.parseResult.errors.length !== 0) {
      throw new Error("Invalid UI control specification");
    }
    defineVertexId(builder);
    builder.addUniform("highp mat4", "uProjection");
    builder.addUniform("highp uint", "uPickID");
    builder.addUniform("highp uint", "uFlags");
    builder.addUniform("highp float", "uVisibleAlpha");
    builder.addUniform("highp float", "uHiddenAlpha");
    builder.addUniform("highp float", "uSaturation");
    builder.addUniform("highp vec4", "uDefaultColor");
    builder.addUniform("highp uint", "uHasDefaultColor");
    builder.addUniform("highp uint", "uHasStatedColors");
    this.textureAccess.defineShader(builder);
    builder.addTextureSampler("sampler2D", "uPositions", positionSampler);
    builder.addTextureSampler("usampler2D", "uSegments", segmentSampler);
    builder.addVertexCode(
      this.textureAccess.getAccessor(
        "readPosition",
        "uPositions",
        DataType.FLOAT32,
        3,
      ),
    );
    builder.addVertexCode(glsl_uint64);
    builder.addVertexCode(
      this.textureAccess.getAccessor(
        "readSegment",
        "uSegments",
        DataType.UINT64,
        1,
      ),
    );
    this.attributes.forEach((a, i) => {
      builder.addTextureSampler(
        "sampler2D",
        `uAttribute${i}`,
        attributeSampler(i),
      );
      builder.addVertexCode(
        this.textureAccess.getAccessor(
          `readAttribute${i}`,
          `uAttribute${i}`,
          DataType.FLOAT32,
          a.components,
        ),
      );
    });
    builder.addVarying("highp uvec2", "vSegment", "flat");
    builder.addVarying("highp uint", "vPickID", "flat");
    let vertexMain: string;
    if (edges) {
      defineLineShader(builder);
      builder.addAttribute("highp uvec2", "aVertexIndex");
      builder.addUniform("highp float", "uLineWidth");
      vertexMain = `
highp vec3 vertexA = readPosition(aVertexIndex.x);
highp vec3 vertexB = readPosition(aVertexIndex.y);
emitLine(uProjection, vertexA, vertexB, uLineWidth);
highp uint endpoint = getLineEndpointIndex();
highp uint vertexIndex = aVertexIndex.x * (1u - endpoint) + aVertexIndex.y * endpoint;
vSegment = readSegment(aVertexIndex.x).value;
vPickID = uPickID + aVertexIndex.x;
`;
    } else {
      defineCircleShader(builder, this.targetIsSliceView);
      builder.addUniform("highp float", "uNodeDiameter");
      vertexMain = `
highp uint vertexIndex = uint(gl_InstanceID);
emitCircle(uProjection * vec4(readPosition(vertexIndex), 1.0), uNodeDiameter, 0.0);
vSegment = readSegment(vertexIndex).value;
vPickID = uPickID + vertexIndex;
`;
    }
    this.attributes.forEach((a, i) => {
      const type = a.components === 1 ? "float" : `vec${a.components}`;
      builder.addVarying(`highp ${type}`, `vAttribute${i}`);
      vertexMain += `vAttribute${i} = readAttribute${i}(vertexIndex);\n`;
      builder.addFragmentCode(`#define prop_${a.id}() vAttribute${i}\n`);
    });
    builder.setVertexMain(vertexMain);

    builder.addFragmentCode(glsl_uint64);
    this.visibleSegments.defineShader(builder);
    this.segmentColor.defineShader(builder);
    this.statedColors.defineShader(builder);
    const alpha = edges ? `getLineAlpha() * ${this.fadeFactor()}` : "1.0";
    builder.addFragmentCode(`
float zvAlpha;
vec4 zvColor;
void zvResolveSegment() {
  uint64_t id;
  id.value = vSegment;
  bool visible = (uFlags & ${SHOW_ALL_FLAG}u) != 0u ||
      ${this.visibleSegments.hasFunctionName}(id);
  zvAlpha = visible ? uVisibleAlpha : uHiddenAlpha;
  if (visible && (uFlags & ${HIDE_VISIBLE_FLAG}u) != 0u) zvAlpha = 0.0;
  vec4 rgba = vec4(0.0);
  bool stated = uHasStatedColors != 0u && ${this.statedColors.getFunctionName}(id, rgba);
  if (!stated) {
    rgba = uHasDefaultColor != 0u ? vec4(uDefaultColor.rgb, 0.0)
                                 : vec4(segmentColorHash(id), 0.0);
  }
  zvColor = vec4(mix(vec3(1.0), rgba.rgb, uSaturation), zvAlpha);
}
vec4 segmentColor() {
  return zvColor;
}
`);
    if (edges) {
      builder.addFragmentCode(`
void emitRGBA(vec4 color) {
  float a = color.a * ${alpha};
  emit(vec4(color.rgb * a, a), vPickID);
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, zvAlpha));
}
void emitDefault() {
  emitRGBA(zvColor);
}
`);
    } else {
      builder.addFragmentCode(`
void emitRGBA(vec4 color) {
  emit(getCircleColor(color, color), vPickID);
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, zvAlpha));
}
void emitDefault() {
  emitRGBA(zvColor);
}
`);
    }
    builder.addFragmentCode(glsl_COLORMAPS);
    addControlsToBuilder(state, builder);
    addStringShaderSupport(builder);
    builder.addFragmentCode(`
void zvUserMain();
`);
    const userMain = shaderCodeWithLineDirective(
      state.parseResult.code,
    ).replace(/\bvoid\s+main\s*\(\s*\)/, "void zvUserMain()");
    builder.addFragmentCode(userMain);
    builder.setFragmentMain(`
zvResolveSegment();
if (zvAlpha <= 0.0) discard;
zvUserMain();
`);
  }

  private gpuVisible: GPUHashTable<any> | undefined;
  private gpuStated: GPUHashTable<HashMapUint64> | undefined;
  private statedMap: Uint64Map | undefined;

  /** Sets the per-layer uniforms and binds the segment hash tables. */
  beginLayer(
    shader: ShaderProgram,
    modelViewProjection: mat4,
    visibleAlpha: number,
  ) {
    const { gl, displayState } = this;
    gl.uniformMatrix4fv(
      shader.uniform("uProjection"),
      false,
      modelViewProjection,
    );
    this.vertexIdHelper.enable();
    const groupState = displayState.segmentationGroupState.value;
    const visible = getVisibleSegments(groupState);
    let flags = 0;
    if (
      visible.hashTable.size === 0 &&
      displayState.ignoreNullVisibleSet.value
    ) {
      flags |= SHOW_ALL_FLAG;
    } else if (displayState.objectsDrawnElsewhere.value) {
      flags |= HIDE_VISIBLE_FLAG;
    }
    gl.uniform1ui(shader.uniform("uFlags"), flags);
    gl.uniform1f(shader.uniform("uVisibleAlpha"), visibleAlpha);
    gl.uniform1f(
      shader.uniform("uHiddenAlpha"),
      displayState.notSelectedAlpha.value,
    );
    gl.uniform1f(shader.uniform("uSaturation"), displayState.saturation.value);
    if (
      this.gpuVisible === undefined ||
      this.gpuVisible.hashTable !== visible.hashTable
    ) {
      this.gpuVisible?.dispose();
      this.gpuVisible = GPUHashTable.get(gl, visible.hashTable);
    }
    this.visibleSegments.enable(gl, shader, this.gpuVisible);
    this.segmentColor.enable(gl, shader, displayState.segmentColorHash.value);
    const defaultColor = displayState.segmentDefaultColor.value;
    gl.uniform1ui(
      shader.uniform("uHasDefaultColor"),
      defaultColor === undefined ? 0 : 1,
    );
    if (defaultColor !== undefined) {
      gl.uniform4f(
        shader.uniform("uDefaultColor"),
        defaultColor[0],
        defaultColor[1],
        defaultColor[2],
        1,
      );
    }
    const stated = displayState.segmentStatedColors.value;
    gl.uniform1ui(
      shader.uniform("uHasStatedColors"),
      stated.size === 0 ? 0 : 1,
    );
    if (this.gpuStated === undefined || this.statedMap !== stated) {
      this.gpuStated?.dispose();
      this.gpuStated = GPUHashTable.get(gl, stated.hashTable);
      this.statedMap = stated;
    }
    this.statedColors.enable(gl, shader, this.gpuStated);
  }

  endLayer(shader: ShaderProgram) {
    const { gl } = this;
    this.visibleSegments.disable(gl, shader);
    this.statedColors.disable(gl, shader);
    this.vertexIdHelper.disable();
  }

  bindChunk(shader: ShaderProgram, chunk: ZarrVectorsDenseChunk) {
    const { gl } = this;
    const bind = (symbol: symbol, texture: WebGLTexture | null) => {
      gl.activeTexture(
        WebGL2RenderingContext.TEXTURE0 + shader.textureUnit(symbol),
      );
      gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
    };
    bind(positionSampler, chunk.positionTexture);
    bind(segmentSampler, chunk.segmentTexture);
    chunk.attributeTextures.forEach((t, i) => bind(attributeSampler(i), t));
  }

  disposed() {
    this.gpuVisible?.dispose();
    this.gpuStated?.dispose();
    super.disposed();
  }
}

const positionSampler = Symbol("zarr-vectors.positions");
const segmentSampler = Symbol("zarr-vectors.segments");
const attributeSamplers: symbol[] = [];
function attributeSampler(i: number) {
  while (attributeSamplers.length <= i) {
    attributeSamplers.push(
      Symbol(`zarr-vectors.attribute${attributeSamplers.length}`),
    );
  }
  return attributeSamplers[i];
}

export interface ZarrVectorsDenseLayerOptions {
  chunkManager: ChunkManager;
  source: ZarrVectorsMultiscaleGeometrySource;
  displayState: ZarrVectorsDenseDisplayState;
}

/** Shared state of the 2-D and 3-D render layers of one subsource. */
export class ZarrVectorsDenseLayer extends RefCounted {
  backend: ChunkRenderLayerFrontend;
  fallbackShaderParameters = new WatchableValue(
    getFallbackBuilderState(parseShaderUiControls(DEFAULT_FRAGMENT_MAIN)),
  );
  redrawNeeded;

  constructor(
    public options: ZarrVectorsDenseLayerOptions,
    layerChunkProgressInfo: any,
    redrawNeeded: any,
  ) {
    super();
    this.redrawNeeded = redrawNeeded;
    const { chunkManager, displayState } = options;
    const rpc = chunkManager.rpc!;
    const backend = (this.backend = this.registerDisposer(
      new ChunkRenderLayerFrontend(layerChunkProgressInfo),
    ));
    backend.RPC_TYPE_ID = ZARR_VECTORS_DENSE_RENDER_LAYER_RPC_ID;
    backend.initializeCounterpart(rpc, {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, displayState.localPosition),
      ).rpcId,
      renderScaleTarget2d: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(
          rpc,
          displayState.renderScaleTarget2d,
        ),
      ).rpcId,
      renderScaleTarget3d: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(
          rpc,
          displayState.renderScaleTarget,
        ),
      ).rpcId,
      densities: options.source.densities,
    });
  }
}

function DenseRenderLayer<
  TBase extends AnyConstructor<
    PerspectiveViewRenderLayer | SliceViewPanelRenderLayer
  >,
>(Base: TBase, targetIsSliceView: boolean) {
  class C extends (Base as AnyConstructor<PerspectiveViewRenderLayer>) {
    shared: ZarrVectorsDenseLayer;
    helper: DenseRenderHelper;
    declare backend: ChunkRenderLayerFrontend;

    constructor(...args: any[]) {
      super();
      const options = args[0] as ZarrVectorsDenseLayerOptions;
      const { displayState, source, chunkManager } = options;
      this.shared = this.registerDisposer(
        new ZarrVectorsDenseLayer(
          options,
          this.layerChunkProgressInfo,
          this.redrawNeeded,
        ),
      );
      this.backend = this.shared.backend;
      this.helper = this.registerDisposer(
        new DenseRenderHelper(
          chunkManager.chunkQueueManager.gl,
          displayState,
          source.description,
          targetIsSliceView,
          this.shared.fallbackShaderParameters,
        ),
      );
      registerRedrawWhenSegmentationDisplayState3DChanged(
        displayState as unknown as SegmentationDisplayState3D,
        this,
      );
      const redraw = () => this.redrawNeeded.dispatch();
      const options3d = displayState.skeletonRenderingOptions;
      for (const value of [
        options3d.shader,
        options3d.params2d.mode,
        options3d.params2d.lineWidth,
        options3d.params3d.mode,
        options3d.params3d.lineWidth,
        displayState.selectedAlpha,
        displayState.notSelectedAlpha,
        displayState.ignoreNullVisibleSet,
        displayState.objectsDrawnElsewhere,
        displayState.renderScaleTarget2d,
        displayState.renderScaleTarget,
      ]) {
        this.registerDisposer(value.changed.add(redraw));
      }
      this.registerDisposer(
        options3d.shader.changed.add(() => {
          displayState.shaderError.value = undefined;
        }),
      );
    }

    get options() {
      return this.shared.options;
    }

    get gl() {
      return this.options.chunkManager.chunkQueueManager.gl;
    }

    get isTransparent() {
      return true;
    }

    attach(attachment: VisibleLayerInfo<LayerView, DenseAttachmentState>) {
      super.attach(attachment);
      const { displayState, source } = this.options;
      const sources = attachment.registerDisposer(
        registerNested(
          (
            context: RefCounted,
            transform: RenderLayerTransformOrError,
            displayDimensionRenderInfo: DisplayDimensionRenderInfo,
          ) => {
            const transformed = getVolumetricTransformedSources(
              displayDimensionRenderInfo,
              transform,
              () => source.getSources(),
              attachment.messages,
              this,
            );
            for (const scales of transformed) {
              for (const tsource of scales)
                context.registerDisposer(tsource.source);
            }
            attachment.view.flushBackendProjectionParameters();
            this.backend.rpc!.invoke(
              ZARR_VECTORS_DENSE_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
              {
                layer: this.backend.rpcId,
                view: attachment.view.rpcId,
                displayDimensionRenderInfo,
                sources: serializeAllTransformedSources(transformed),
              },
            );
            this.redrawNeeded.dispatch();
            return transformed;
          },
          displayState.transform,
          attachment.view.displayDimensionRenderInfo,
        ),
      );
      attachment.state = {
        transform: displayState.transform.value,
        displayDimensionRenderInfo:
          attachment.view.displayDimensionRenderInfo.value,
        modelMatrix: undefined,
        sources,
      };
    }

    private modelMatrix(
      attachment: VisibleLayerInfo<LayerView, DenseAttachmentState>,
      displayDimensionRenderInfo: DisplayDimensionRenderInfo,
    ): mat4 | undefined {
      const state = attachment.state!;
      const transform = this.options.displayState.transform.value;
      if (
        state.modelMatrix === undefined ||
        state.transform !== transform ||
        state.displayDimensionRenderInfo !== displayDimensionRenderInfo
      ) {
        state.transform = transform;
        state.displayDimensionRenderInfo = displayDimensionRenderInfo;
        state.modelMatrix = undefined;
        if (transform.error !== undefined) return undefined;
        const m = mat4.create();
        try {
          get3dModelToDisplaySpaceMatrix(
            m,
            displayDimensionRenderInfo,
            transform,
          );
        } catch {
          return undefined;
        }
        state.modelMatrix = m;
      }
      return state.modelMatrix;
    }

    /** The chunks to draw: the target level where loaded, else the coarsest. */
    private chunksToDraw(
      projectionParameters: any,
      transformed: FrontendTransformedSource[],
    ): { ready: boolean; chunks: ZarrVectorsDenseChunk[] } {
      const { displayState, source } = this.options;
      const renderScaleTarget = targetIsSliceView
        ? displayState.renderScaleTarget2d.value
        : displayState.renderScaleTarget.value;
      const target = selectDenseLevel(
        projectionParameters,
        transformed,
        source.densities,
        renderScaleTarget,
      );
      const localPosition = displayState.localPosition.value;
      const chunks: ZarrVectorsDenseChunk[] = [];
      const missing: Float32Array[] = [];
      const tsource = transformed[target];
      forEachVisibleVolumetricChunk(
        projectionParameters,
        localPosition,
        tsource,
        () => {
          const key = tsource.curPositionInChunks.join();
          const chunk = (
            tsource.source as unknown as ZarrVectorsGeometryChunkSource
          ).chunks.get(key) as ZarrVectorsDenseChunk | undefined;
          if (chunk !== undefined && chunk.state === ChunkState.GPU_MEMORY) {
            chunks.push(chunk);
          } else {
            missing.push(Float32Array.from(tsource.curPositionInChunks));
          }
        },
      );
      const ready = missing.length === 0;
      if (!ready && target !== transformed.length - 1) {
        // Stand in with the coarsest level's chunk covering each missing one.
        const coarse = transformed[transformed.length - 1];
        const targetSize = tsource.source.spec.chunkDataSize;
        const coarseSource = coarse.source;
        const coarseSize = coarseSource.spec.chunkDataSize;
        const seen = new Set<string>();
        for (const position of missing) {
          const coords = Array.from(position, (c, d) =>
            Math.floor(((c + 0.5) * targetSize[d]) / coarseSize[d]),
          );
          const key = coords.join();
          if (seen.has(key)) continue;
          seen.add(key);
          const chunk = coarseSource.chunks.get(key) as
            | ZarrVectorsDenseChunk
            | undefined;
          if (chunk !== undefined && chunk.state === ChunkState.GPU_MEMORY) {
            chunks.push(chunk);
          }
        }
      }
      return { ready, chunks };
    }

    draw(
      renderContext: PerspectiveViewRenderContext | SliceViewPanelRenderContext,
      attachment: VisibleLayerInfo<any, any>,
    ) {
      if (
        !renderContext.emitColor &&
        (renderContext as any).alreadyEmittedPickID
      ) {
        return;
      }
      const { displayState } = this.options;
      const visibleAlpha = targetIsSliceView
        ? displayState.selectedAlpha.value
        : displayState.objectAlpha.value;
      if (visibleAlpha <= 0 && displayState.notSelectedAlpha.value <= 0) return;
      const { projectionParameters } = renderContext;
      const modelMatrix = this.modelMatrix(
        attachment,
        projectionParameters.displayDimensionRenderInfo,
      );
      if (modelMatrix === undefined) return;
      const transformed = attachment.state!.sources.value;
      if (transformed.length === 0 || transformed[0].length === 0) return;
      const { chunks } = this.chunksToDraw(
        projectionParameters,
        transformed[0],
      );
      void (attachment as VisibleLayerInfo<LayerView, DenseAttachmentState>);
      if (chunks.length === 0) return;

      const { helper, gl } = this;
      const renderOptions = targetIsSliceView
        ? displayState.skeletonRenderingOptions.params2d
        : displayState.skeletonRenderingOptions.params3d;
      const lineWidth = renderOptions.lineWidth.value;
      const pointDiameter =
        helper.primitive === "points" ||
        renderOptions.mode.value === SkeletonRenderMode.LINES_AND_POINTS
          ? Math.max(5, lineWidth * 2)
          : lineWidth;
      const mvp = mat4.multiply(
        tempMat4,
        projectionParameters.viewProjectionMat,
        modelMatrix,
      );
      const { shaderControlState } = displayState.skeletonRenderingOptions;
      const passes: { shader: ShaderProgram; edges: boolean }[] = [];
      if (helper.primitive === "lines") {
        const { shader, parameters } = helper.edgeShaderGetter(
          renderContext.emitter,
        );
        if (shader === null) return;
        shader.bind();
        setShaderControls(
          gl,
          shader,
          shaderControlState,
          parameters.parseResult,
        );
        gl.uniform1f(shader.uniform("uLineWidth"), lineWidth);
        initializeLineShader(
          shader,
          projectionParameters,
          targetIsSliceView ? 1.0 : 0.0,
        );
        passes.push({ shader, edges: true });
      }
      {
        const { shader, parameters } = helper.nodeShaderGetter(
          renderContext.emitter,
        );
        if (shader === null) return;
        shader.bind();
        setShaderControls(
          gl,
          shader,
          shaderControlState,
          parameters.parseResult,
        );
        gl.uniform1f(shader.uniform("uNodeDiameter"), pointDiameter);
        initializeCircleShader(shader, projectionParameters, {
          featherWidthInPixels: targetIsSliceView ? 1.0 : 0.0,
        });
        passes.push({ shader, edges: false });
      }
      for (const { shader, edges } of passes) {
        shader.bind();
        helper.beginLayer(shader, mvp, visibleAlpha);
        for (const chunk of chunks) {
          const pickID = renderContext.emitPickID
            ? renderContext.pickIDs.register(this, chunk.numVertices, 0n, chunk)
            : 0;
          gl.uniform1ui(shader.uniform("uPickID"), pickID);
          helper.bindChunk(shader, chunk);
          if (edges) {
            if (chunk.edgeBuffer === undefined) continue;
            const aVertexIndex = shader.attribute("aVertexIndex");
            chunk.edgeBuffer.bindToVertexAttribI(
              aVertexIndex,
              2,
              WebGL2RenderingContext.UNSIGNED_INT,
            );
            gl.vertexAttribDivisor(aVertexIndex, 1);
            drawLines(gl, 1, chunk.numEdges);
            gl.vertexAttribDivisor(aVertexIndex, 0);
            gl.disableVertexAttribArray(aVertexIndex);
          } else {
            drawCircles(gl, 1, chunk.numOwnVertices);
          }
        }
        helper.endLayer(shader);
      }
    }

    isReady(
      renderContext: { projectionParameters: any },
      attachment: VisibleLayerInfo<any, any>,
    ) {
      const transformed = attachment.state?.sources.value;
      if (transformed === undefined || transformed.length === 0) return true;
      return this.chunksToDraw(
        renderContext.projectionParameters,
        transformed[0],
      ).ready;
    }

    updateMouseState(
      mouseState: MouseSelectionState,
      _pickedValue: bigint,
      pickedOffset: number,
      data: any,
    ) {
      const chunk = data as ZarrVectorsDenseChunk;
      const ids = chunk.segmentIds;
      if (ids === undefined || pickedOffset * 2 + 1 >= ids.length) return;
      mouseState.pickedValue =
        BigInt(ids[pickedOffset * 2]) |
        (BigInt(ids[pickedOffset * 2 + 1]) << 32n);
    }
  }
  return C as unknown as AnyConstructor<RenderLayer> & {
    new (options: ZarrVectorsDenseLayerOptions): InstanceType<TBase> & {
      shared: ZarrVectorsDenseLayer;
    };
  };
}

export const PerspectiveViewZarrVectorsDenseLayer = DenseRenderLayer(
  PerspectiveViewRenderLayer,
  false,
);
export const SliceViewPanelZarrVectorsDenseLayer = DenseRenderLayer(
  SliceViewPanelRenderLayer,
  true,
);
