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
 * @file Opens a zarr-vectors store: root and level metadata, the arrays each
 * level holds, and which vertex attributes to expose.
 *
 * Everything here is plain data that can be posted to the chunk worker.  Each
 * array is described by its own `zarr.json` (see `zarr_array.ts`) because
 * codecs, sharding and `chunk_grid_origin` are per array, not per level.
 */

import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  intraOffsets,
  linksPath,
} from "#src/datasource/zarr-vectors/links_paths.js";
import { resolveDeclaredGeometry } from "#src/datasource/zarr-vectors/store_metadata.js";
import type {
  ZarrArray,
  ZarrArrayRead,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import {
  mapConcurrent,
  parseZarrArrayMetadata,
  ShardIndexCache,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import { allSiPrefixes, supportedUnits } from "#src/util/si_units.js";

/** zarr-vectors format versions this reader understands: 0.9.x. */
export const SUPPORTED_FORMAT = { major: 0, minor: 9, maxPatch: 4 };

/** Object-index layouts this reader can resolve. */
const KNOWN_OBJECT_INDEX_LAYOUTS = new Set([
  undefined,
  "vlen_manifests_v1",
  "vlen_manifests_v2",
]);

/** Format capabilities this reader honours (others are ignored, per spec §8.2). */
const UNSUPPORTED_CAPABILITIES = new Set(["dense_manifests"]);

export type ZarrVectorsLinksConvention =
  | "implicit_sequential"
  | "implicit_sequential_with_branches"
  | "explicit";

/** A vertex attribute exposed to shaders. */
export interface ZarrVectorsAttribute {
  /** Directory name under `vertex_attributes/`. */
  name: string;
  /** GLSL-safe identifier; the shader sees `prop_<id>()`. */
  id: string;
  /** On-disk element type. */
  dtype: string;
  /** Values per vertex (1 for a scalar, C for an `(N, C)` attribute). */
  components: number;
  /** Category labels of a dictionary-encoded attribute. */
  enumLabels?: string[];
}

/** Raw `zarr.json` documents of one level's per-chunk arrays. */
export interface ZarrVectorsLevelArrays {
  vertices: any;
  vertexFragments: any | undefined;
  fragmentSegmentIds: any | undefined;
  intraLinks: any | undefined;
  /** Parallel to the store's attribute list; `undefined` where absent. */
  attributes: (any | undefined)[];
}

export interface ZarrVectorsLevel {
  index: number;
  /** Level directory relative to the store root, e.g. `"0"`. */
  path: string;
  /** Physical size of one spatial cell at this level. */
  chunkShape: number[];
  vertexCount: number | undefined;
  objectSparsity: number | undefined;
  arrays: ZarrVectorsLevelArrays;
}

export interface ZarrVectorsStore {
  url: string;
  zvVersion: string | undefined;
  geometryKind: ZarrVectorsGeometryKind;
  rank: number;
  lowerBounds: number[];
  upperBounds: number[];
  /** World position of stored coordinate 0, when the store declares one. */
  coordinateOffset: number[] | undefined;
  axisNames: string[];
  /** Base SI unit per axis ("m", or "" when undeclared). */
  axisUnits: string[];
  /** Size of one stored coordinate unit, in `axisUnits`. */
  axisScales: number[];
  linksConvention: ZarrVectorsLinksConvention;
  linkWidth: number;
  /** True when the store marks its intra-chunk skeleton links as replacing the implied parent. */
  linkedSkeletonLayout: boolean;
  /** Undirected link families carry a `perm_idx` column on cross-chunk rows. */
  linksDirected: boolean;
  levels: ZarrVectorsLevel[];
  attributes: ZarrVectorsAttribute[];
  /** Name of a vertex attribute holding a stable per-vertex id, if declared. */
  vertexIdAttribute: string | undefined;
  /**
   * Whether the finest level has an object index.  Usually implied by the
   * geometry kind, but a point cloud may carry one too (synapses keyed by the
   * cell they belong to).
   */
  hasObjects: boolean;
  /** Diagnostics worth surfacing once to the user. */
  warnings: string[];
}

/** Reads, lists and parses JSON under a store root. */
export interface ZarrVectorsStoreAccess {
  read: ZarrArrayRead;
  /** Lists the child directory names under `path`; throws if listing is unsupported. */
  listDirectories(path: string, signal?: AbortSignal): Promise<string[]>;
}

export async function readJson(
  read: ZarrArrayRead,
  path: string,
  signal?: AbortSignal,
): Promise<any | undefined> {
  const bytes = await read(path, { signal });
  if (bytes === undefined) return undefined;
  return JSON.parse(new TextDecoder().decode(bytes));
}

/** Like {@link readJson} for an array; checks it parses. */
async function readArrayJson(
  read: ZarrArrayRead,
  path: string,
  signal?: AbortSignal,
): Promise<any | undefined> {
  const json = await readJson(read, `${path}/zarr.json`, signal);
  if (json === undefined || json.node_type !== "array") return undefined;
  parseZarrArrayMetadata(path, json);
  return json;
}

// -------------------------------------------------------------- units

const LONG_UNITS = (() => {
  const m = new Map<string, { unit: string; exponent: number }>();
  for (const baseUnit of ["meter", "second"]) {
    for (const p of allSiPrefixes) {
      if (p.longPrefix === undefined) continue;
      m.set(`${p.longPrefix}${baseUnit}`, {
        unit: baseUnit[0],
        exponent: p.exponent,
      });
    }
  }
  m.set("micron", { unit: "m", exponent: -6 });
  m.set("microns", { unit: "m", exponent: -6 });
  return m;
})();

/**
 * Folds an SI prefix into the scale, so `(1, "mm")` becomes `(0.001, "m")`.
 * An undeclared or unknown unit stays unitless rather than defaulting to
 * metres: a unitless store placed in a metre frame lands 1000x away from a
 * millimetre one, which renders an empty view with no error.
 */
export function normalizeUnitScale(
  scale: number,
  unit: unknown,
): { unit: string; scale: number } {
  if (typeof unit !== "string" || unit === "") return { unit: "", scale };
  const long = LONG_UNITS.get(unit);
  if (long !== undefined) {
    return { unit: long.unit, scale: scale * 10 ** long.exponent };
  }
  const short = supportedUnits.get(unit);
  if (short !== undefined) {
    return { unit: short.unit, scale: scale * 10 ** short.exponent };
  }
  return { unit: "", scale };
}

// -------------------------------------------------------------- versions

function checkFormat(zv: any, warnings: string[]) {
  const raw = zv?.zv_version;
  const match =
    raw === undefined
      ? undefined
      : /^(\d+)\.(\d+)(?:\.(\d+))?/.exec(String(raw));
  if (raw === undefined) {
    warnings.push("store declares no zv_version; reading it as a 0.9 layout");
  } else if (match === null || match === undefined) {
    warnings.push(`unrecognised zv_version ${JSON.stringify(raw)}`);
  } else {
    const major = Number(match[1]);
    const minor = Number(match[2]);
    const patch = Number(match[3] ?? 0);
    const { major: wantMajor, minor: wantMinor, maxPatch } = SUPPORTED_FORMAT;
    if (major < wantMajor || (major === wantMajor && minor < wantMinor)) {
      throw new Error(
        `store zv_version ${raw} predates the ${wantMajor}.${wantMinor} ` +
          "layout this viewer reads; rewrite it with zarr-vectors >= 0.9",
      );
    }
    if (major > wantMajor || minor > wantMinor || patch > maxPatch) {
      warnings.push(
        `store zv_version ${raw} is newer than this viewer's ` +
          `${wantMajor}.${wantMinor}.${maxPatch} reader; newer features may ` +
          "not render",
      );
    }
  }
  const capabilities: unknown = zv?.format_capabilities;
  if (Array.isArray(capabilities)) {
    for (const c of capabilities) {
      if (UNSUPPORTED_CAPABILITIES.has(c)) {
        warnings.push(
          `store uses format capability ${JSON.stringify(c)}, which this ` +
            "viewer cannot read; per-object views will be unavailable",
        );
      }
    }
  }
}

/** Throws unless the object index of a level is a layout this reader resolves. */
export function checkObjectIndexLayout(objectIndexAttrs: any) {
  const layout = objectIndexAttrs?.layout;
  if (!KNOWN_OBJECT_INDEX_LAYOUTS.has(layout)) {
    throw new Error(
      `object_index layout ${JSON.stringify(layout)} is not supported by ` +
        "this viewer",
    );
  }
}

// -------------------------------------------------------------- attributes

/**
 * Names that must not become shader identifiers: the upstream skeleton
 * renderer emits a bare `#define <id>` per attribute, so an attribute called
 * `z` breaks every `.z` swizzle and one called `length` breaks `length()`.
 */
const GLSL_UNSAFE = new Set([
  "x",
  "y",
  "z",
  "w",
  "r",
  "g",
  "b",
  "a",
  "s",
  "t",
  "p",
  "q",
  "abs",
  "all",
  "any",
  "ceil",
  "clamp",
  "cos",
  "cross",
  "degrees",
  "distance",
  "dot",
  "exp",
  "exp2",
  "floor",
  "fract",
  "inverse",
  "length",
  "log",
  "log2",
  "max",
  "min",
  "mix",
  "mod",
  "normalize",
  "not",
  "pow",
  "radians",
  "reflect",
  "refract",
  "round",
  "sign",
  "sin",
  "smoothstep",
  "sqrt",
  "step",
  "tan",
  "transpose",
  "texture",
  "attribute",
  "uniform",
  "varying",
  "in",
  "out",
  "inout",
  "const",
  "float",
  "int",
  "uint",
  "bool",
  "vec2",
  "vec3",
  "vec4",
  "mat2",
  "mat3",
  "mat4",
  "void",
  "if",
  "else",
  "for",
  "while",
  "do",
  "return",
  "break",
  "continue",
  "discard",
  "true",
  "false",
  "highp",
  "mediump",
  "lowp",
  "precision",
  "struct",
  "switch",
  "case",
  "default",
  "layout",
  "flat",
  "smooth",
  "centroid",
  "sampler2D",
  "segment",
  "tangent",
  "position",
  "color",
  "main",
]);

/** Maps an attribute name to a GLSL-safe, unique identifier. */
export function safeAttributeId(name: string, used: Set<string>): string {
  let base = name.replace(/[^a-zA-Z0-9_]/g, "_").replace(/__+/g, "_");
  if (!/^[a-z]/.test(base)) base = `a_${base}`;
  if (GLSL_UNSAFE.has(base) || base.startsWith("gl_")) base = `attr_${base}`;
  let id = base;
  for (let i = 2; used.has(id); ++i) id = `${base}_${i}`;
  used.add(id);
  return id;
}

/** Element types a vertex attribute may have (all decode to float32). */
const ATTRIBUTE_DTYPES = new Set([
  "bool",
  "int8",
  "uint8",
  "int16",
  "uint16",
  "int32",
  "uint32",
  "int64",
  "uint64",
  "float16",
  "float32",
  "float64",
]);

/** Most components one attribute may have (the shader's `vec4`). */
const MAX_ATTRIBUTE_COMPONENTS = 4;

/**
 * Default number of attributes exposed without an `#attributes=` selection.
 * Each one costs a vertex texture and a varying in the skeleton shader, and
 * WebGL2 guarantees only 16 of each.
 */
export const DEFAULT_ATTRIBUTE_LIMIT = 8;

/** Absolute limit, explicit selection included. */
export const MAX_ATTRIBUTES = 12;

function attributeComponents(json: any): number {
  const rowShape = json?.attributes?.row_shape;
  if (Array.isArray(rowShape) && rowShape.length > 0) {
    return rowShape.reduce((a: number, b: number) => a * Number(b), 1);
  }
  // Writers before the `row_shape` fix stamp `[]` on an (N, C) attribute.
  const channels = json?.attributes?.num_channels ?? json?.attributes?.ncols;
  if (Number.isInteger(channels) && channels > 0) return channels;
  return 1;
}

function hasExplicitRowShape(json: any): boolean {
  const rowShape = json?.attributes?.row_shape;
  return Array.isArray(rowShape) && rowShape.length > 0;
}

/**
 * Values per vertex of an attribute whose metadata does not say (writers
 * before zarr-vectors-py's `row_shape` fix stamp `[]` even on `(N, C)`
 * attributes), measured from one populated cell.
 */
type AttributeWidthProbe = (
  name: string,
  json: any,
  dtype: string,
) => Promise<number | undefined>;

const DTYPE_BYTES: Record<string, number> = {
  bool: 1,
  int8: 1,
  uint8: 1,
  int16: 2,
  uint16: 2,
  float16: 2,
  int32: 4,
  uint32: 4,
  float32: 4,
  int64: 8,
  uint64: 8,
  float64: 8,
};

function makeWidthProbe(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  verticesJson: any,
  rank: number,
  signal?: AbortSignal,
): AttributeWidthProbe {
  const shardIndexes = new ShardIndexCache();
  const vertices = new ZarrArrayReader(
    parseZarrArrayMetadata(`${levelPath}/vertices`, verticesJson),
    access.read,
    shardIndexes,
  );
  const vertexBytes =
    rank *
    (DTYPE_BYTES[String(verticesJson.attributes?.dtype ?? "float32")] ?? 4);
  let cellPromise:
    | Promise<{ cell: number[]; count: number } | undefined>
    | undefined;
  const findCell = async () => {
    const candidates: number[][] = [];
    if (vertices.array.nonemptyCells !== undefined) {
      for (const key of vertices.array.nonemptyCells) {
        candidates.push(key.split(".").map(Number));
        if (candidates.length >= 4) break;
      }
    } else {
      const { shape, origin } = vertices.array;
      for (let i = 0; i < shape[0] && candidates.length < 64; ++i) {
        for (let j = 0; j < shape[1] && candidates.length < 64; ++j) {
          for (let k = 0; k < shape[2] && candidates.length < 64; ++k) {
            candidates.push([i + origin[0], j + origin[1], k + origin[2]]);
          }
        }
      }
    }
    for (const cell of candidates) {
      const bytes = await vertices.readCell(cell, signal);
      if (bytes !== undefined && bytes.byteLength > 0) {
        return { cell, count: bytes.byteLength / vertexBytes };
      }
    }
    return undefined;
  };
  return async (name, json, dtype) => {
    cellPromise ??= findCell();
    const found = await cellPromise;
    if (found === undefined) return undefined;
    const reader = new ZarrArrayReader(
      parseZarrArrayMetadata(`${levelPath}/vertex_attributes/${name}`, json),
      access.read,
      shardIndexes,
    );
    const bytes = await reader.readCell(found.cell, signal);
    if (bytes === undefined) return undefined;
    const width = bytes.byteLength / (found.count * DTYPE_BYTES[dtype]);
    return Number.isInteger(width) && width >= 1 ? width : undefined;
  };
}

function attributeDtype(json: any): string | undefined {
  const dtype = json?.attributes?.dtype ?? json?.data_type;
  return typeof dtype === "string" ? dtype : undefined;
}

async function listOrEmpty(
  access: ZarrVectorsStoreAccess,
  path: string,
  warnings: string[],
  what: string,
  signal?: AbortSignal,
): Promise<string[]> {
  try {
    return (await access.listDirectories(path, signal)).sort();
  } catch (e) {
    warnings.push(
      `could not list ${path}/ (${e instanceof Error ? e.message : e}); ` +
        `${what} are unavailable unless named with #attributes=`,
    );
    return [];
  }
}

/** Picks the vertex attributes to expose and reads their metadata. */
async function selectAttributes(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  zv: any,
  selected: readonly string[] | undefined,
  warnings: string[],
  probe: AttributeWidthProbe,
  signal?: AbortSignal,
): Promise<{ attributes: ZarrVectorsAttribute[]; level0Json: any[] }> {
  let candidates: string[];
  if (selected !== undefined) {
    candidates = [...selected];
  } else {
    const declared = zv?.attribute_specs?.vertex;
    // A level without the group simply has no attributes; only list if the
    // group exists, so an absent directory is not reported as a failure.
    const group = await readJson(
      access.read,
      `${levelPath}/vertex_attributes/zarr.json`,
      signal,
    );
    const listed =
      group === undefined
        ? []
        : await listOrEmpty(
            access,
            `${levelPath}/vertex_attributes`,
            warnings,
            "vertex attributes",
            signal,
          );
    const names = new Set(listed);
    if (declared !== null && typeof declared === "object") {
      for (const name of Object.keys(declared)) names.add(name);
    }
    // The renderer synthesises `tangent`; a stored one would shadow it.
    names.delete("tangent");
    candidates = [...names].sort();
    if (candidates.length > DEFAULT_ATTRIBUTE_LIMIT) {
      // A wide panel (one column per gene) would fetch that many cells per
      // chunk for columns nobody asked for; let the user choose.
      warnings.push(
        `store has ${candidates.length} vertex attributes; none are loaded ` +
          "by default. Append #attributes=a,b,c to the source URL to choose " +
          `up to ${MAX_ATTRIBUTES} (e.g. #attributes=${candidates
            .slice(0, 3)
            .map(encodeURIComponent)
            .join(",")})`,
      );
      candidates = [];
    }
  }
  const limit =
    selected !== undefined ? MAX_ATTRIBUTES : DEFAULT_ATTRIBUTE_LIMIT;
  const attributes: ZarrVectorsAttribute[] = [];
  const level0Json: any[] = [];
  const used = new Set<string>();
  const skipped: string[] = [];
  // Read metadata page by page so a 1000-gene panel never reads every column.
  let next = 0;
  while (attributes.length < limit && next < candidates.length) {
    const page = candidates.slice(next, next + (limit - attributes.length));
    next += page.length;
    const jsons = new Array<any>(page.length);
    await mapConcurrent(page, 16, async (name, i) => {
      try {
        jsons[i] = await readArrayJson(
          access.read,
          `${levelPath}/vertex_attributes/${name}`,
          signal,
        );
      } catch (e) {
        jsons[i] = e;
      }
    });
    for (let i = 0; i < page.length; ++i) {
      const name = page[i];
      const json = jsons[i];
      const dtype = attributeDtype(json);
      let components = attributeComponents(json);
      if (
        !(json instanceof Error) &&
        json !== undefined &&
        dtype !== undefined &&
        ATTRIBUTE_DTYPES.has(dtype) &&
        !hasExplicitRowShape(json)
      ) {
        components =
          (await probe(name, json, dtype).catch(() => undefined)) ?? components;
      }
      let problem: string | undefined;
      if (json instanceof Error) problem = json.message;
      else if (json === undefined) problem = "missing";
      else if (dtype === undefined || !ATTRIBUTE_DTYPES.has(dtype)) {
        problem = `dtype ${dtype}`;
      } else if (components > MAX_ATTRIBUTE_COMPONENTS) {
        problem = `${components} components (at most ${MAX_ATTRIBUTE_COMPONENTS})`;
      }
      if (problem !== undefined) {
        if (selected !== undefined) {
          throw new Error(
            `#attributes names ${JSON.stringify(name)}, which this viewer ` +
              `cannot read: ${problem}`,
          );
        }
        skipped.push(`${name} (${problem})`);
        continue;
      }
      let enumLabels: string[] | undefined;
      if (json.attributes?.encoding === "dictionary") {
        const categories = json.attributes.categories;
        if (Array.isArray(categories)) enumLabels = categories.map(String);
      }
      attributes.push({
        name,
        id: safeAttributeId(name, used),
        dtype: dtype!,
        components,
        enumLabels,
      });
      level0Json.push(json);
    }
  }
  if (selected !== undefined && selected.length > MAX_ATTRIBUTES) {
    throw new Error(
      `#attributes names ${selected.length} attributes; at most ` +
        `${MAX_ATTRIBUTES} can be shown at once`,
    );
  }
  if (selected === undefined && candidates.length > attributes.length) {
    const rest = candidates.length - attributes.length - skipped.length;
    if (rest > 0) {
      warnings.push(
        `store has ${candidates.length} vertex attributes; showing ` +
          `${attributes.map((a) => a.name).join(", ")}. Append ` +
          "#attributes=a,b,c to the source URL to choose others",
      );
    }
  }
  if (skipped.length > 0) {
    warnings.push(`skipped vertex attributes: ${skipped.join(", ")}`);
  }
  return { attributes, level0Json };
}

// -------------------------------------------------------------- levels

function enumerateLevelPaths(multiscales: any): string[] {
  const datasets = Array.isArray(multiscales)
    ? multiscales[0]?.datasets
    : undefined;
  if (Array.isArray(datasets)) {
    const paths = datasets
      .map((d: any) => d?.path)
      .filter((p: unknown): p is string => typeof p === "string" && p !== "");
    if (paths.length > 0) return paths;
  }
  return ["0"];
}

async function readLevel(
  access: ZarrVectorsStoreAccess,
  index: number,
  path: string,
  rootChunkShape: number[],
  attributes: ZarrVectorsAttribute[],
  level0AttributeJson: any[] | undefined,
  intraLinksPath: string | undefined,
  needsSegmentIds: boolean,
  signal?: AbortSignal,
): Promise<ZarrVectorsLevel> {
  const { read } = access;
  const levelJson = await readJson(read, `${path}/zarr.json`, signal);
  const levelAttrs = levelJson?.attributes?.zarr_vectors_level ?? {};
  const optional = (p: string) =>
    readArrayJson(read, `${path}/${p}`, signal).catch((e) => {
      throw new Error(`${path}/${p}: ${e instanceof Error ? e.message : e}`);
    });
  const [vertices, vertexFragments, fragmentSegmentIds, intraLinks, ...attrs] =
    await Promise.all([
      readJson(read, `${path}/vertices/zarr.json`, signal),
      optional("vertex_fragments"),
      needsSegmentIds
        ? optional("fragment_attributes/segment_id")
        : Promise.resolve(undefined),
      intraLinksPath !== undefined
        ? optional(intraLinksPath)
        : Promise.resolve(undefined),
      ...attributes.map((a, i) =>
        level0AttributeJson !== undefined
          ? Promise.resolve(level0AttributeJson[i])
          : optional(`vertex_attributes/${a.name}`),
      ),
    ]);
  if (vertices === undefined || vertices.node_type !== "array") {
    throw new Error(
      `${path}/vertices is not a zarr array (node_type ` +
        `${JSON.stringify(vertices?.node_type)}). Stores written before ` +
        "zarr-vectors 0.9.0 must be rewritten.",
    );
  }
  parseZarrArrayMetadata(`${path}/vertices`, vertices);
  const chunkShapeRaw = levelAttrs.chunk_shape;
  const chunkShape =
    Array.isArray(chunkShapeRaw) &&
    chunkShapeRaw.length === rootChunkShape.length
      ? chunkShapeRaw.map(Number)
      : rootChunkShape;
  const vertexCount = Number(levelAttrs.vertex_count);
  const objectSparsity = Number(levelAttrs.object_sparsity);
  return {
    index,
    path,
    chunkShape,
    vertexCount: Number.isFinite(vertexCount) ? vertexCount : undefined,
    objectSparsity: Number.isFinite(objectSparsity)
      ? objectSparsity
      : undefined,
    arrays: {
      vertices,
      vertexFragments,
      fragmentSegmentIds,
      intraLinks,
      attributes: attrs,
    },
  };
}

// -------------------------------------------------------------- store

/** Element type of the vertices array, which must decode to float32 positions. */
export function verticesElementType(array: ZarrArray): string {
  const dtype = array.attributes?.dtype ?? "float32";
  return String(dtype);
}

export async function openZarrVectorsStore(
  access: ZarrVectorsStoreAccess,
  url: string,
  selectedAttributes: readonly string[] | undefined,
  signal?: AbortSignal,
): Promise<ZarrVectorsStore> {
  const { read } = access;
  const root = await readJson(read, "zarr.json", signal);
  const zv = root?.attributes?.zarr_vectors;
  if (zv === undefined) {
    throw new Error(
      "not a zarr-vectors store: the root zarr.json has no zarr_vectors block",
    );
  }
  const warnings: string[] = [];
  checkFormat(zv, warnings);

  const bounds = zv.bounds;
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 2 ||
    !Array.isArray(bounds[0]) ||
    !Array.isArray(bounds[1]) ||
    bounds[0].length !== bounds[1].length
  ) {
    throw new Error("store bounds must be [[lower...], [upper...]]");
  }
  const rank = bounds[0].length;
  if (rank !== 3) {
    throw new Error(
      `this viewer renders 3-D zarr-vectors stores; this one has rank ${rank}`,
    );
  }
  const lowerBounds = bounds[0].map(Number);
  const upperBounds = bounds[1].map(Number);
  const rootChunkShape = zv.chunk_shape;
  if (!Array.isArray(rootChunkShape) || rootChunkShape.length !== rank) {
    throw new Error(`store chunk_shape must have rank ${rank}`);
  }

  const levelPaths = enumerateLevelPaths(root.attributes.multiscales);
  const linksFamily = await readJson(
    read,
    `${levelPaths[0]}/links/0/zarr.json`,
    signal,
  );
  const familyAttrs = linksFamily?.attributes ?? {};
  const declaredWidth = Number(familyAttrs.link_width);
  const geometryTypes: string[] = Array.isArray(zv.geometry_types)
    ? zv.geometry_types
    : [];
  const resolution = resolveDeclaredGeometry(geometryTypes, {
    hasLinks: linksFamily !== undefined,
    linkWidth: Number.isInteger(declaredWidth) ? declaredWidth : undefined,
  });
  const geometryKind = resolution.kind;
  if (resolution.skipped.length > 0) {
    warnings.push(
      `store declares ${JSON.stringify(geometryTypes)}; rendering it as ` +
        `${geometryKind}`,
    );
  }
  const caps = KIND_CAPABILITIES[geometryKind];

  let linksConvention: ZarrVectorsLinksConvention;
  const conventionRaw = zv.links_convention;
  if (conventionRaw === undefined) {
    linksConvention =
      geometryKind === "skeleton"
        ? "implicit_sequential_with_branches"
        : caps.edgeSource === "explicit"
          ? "explicit"
          : "implicit_sequential";
  } else if (
    conventionRaw === "implicit_sequential" ||
    conventionRaw === "implicit_sequential_with_branches" ||
    conventionRaw === "explicit"
  ) {
    linksConvention = conventionRaw;
  } else {
    throw new Error(
      `unknown links_convention ${JSON.stringify(conventionRaw)}`,
    );
  }
  let linkWidth = 2;
  if (Number.isInteger(declaredWidth) && declaredWidth >= 2) {
    linkWidth = declaredWidth;
  } else if (caps.primitive === "triangles") {
    linkWidth = 3;
  }
  const needsIntraLinks =
    caps.edgeSource !== "none" && linksConvention !== "implicit_sequential";
  const intraLinksPath = needsIntraLinks
    ? linksPath(0, intraOffsets(rank, linkWidth))
    : undefined;
  // zarr-vectors-tools' SWC ingest writes the "linked" skeleton layout: a
  // stored [child, parent] record REPLACES the parent implied by row order.
  const skeletonLayout =
    root.attributes?.zarr_vectors_tools?.skeleton_layout ?? zv.skeleton_layout;
  const linkedSkeletonLayout =
    linksConvention === "implicit_sequential_with_branches" &&
    (skeletonLayout === "linked" || skeletonLayout?.name === "linked");

  const [level0Vertices, objectIndex] = await Promise.all([
    readJson(read, `${levelPaths[0]}/vertices/zarr.json`, signal),
    readJson(read, `${levelPaths[0]}/object_index/zarr.json`, signal),
  ]);
  if (level0Vertices?.node_type !== "array") {
    throw new Error(
      `${levelPaths[0]}/vertices is not a zarr array; stores written before ` +
        "zarr-vectors 0.9.0 must be rewritten",
    );
  }
  const hasObjects = objectIndex !== undefined;
  const { attributes, level0Json } = await selectAttributes(
    access,
    levelPaths[0],
    zv,
    selectedAttributes,
    warnings,
    makeWidthProbe(access, levelPaths[0], level0Vertices, rank, signal),
    signal,
  );

  const levels = await Promise.all(
    levelPaths.map((path, index) =>
      readLevel(
        access,
        index,
        path,
        rootChunkShape.map(Number),
        attributes,
        index === 0 ? level0Json : undefined,
        intraLinksPath,
        caps.hasObjectModel || hasObjects,
        signal,
      ),
    ),
  );
  for (const level of levels) {
    const dtype = String(level.arrays.vertices.attributes?.dtype ?? "float32");
    if (
      ![
        "float32",
        "float64",
        "float16",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
      ].includes(dtype)
    ) {
      throw new Error(`${level.path}/vertices has unsupported dtype ${dtype}`);
    }
    if (level.arrays.vertices.attributes?.encoding === "draco") {
      warnings.push(
        `${level.path} stores Draco-encoded vertices, which this viewer ` +
          "cannot decode yet; that level is skipped",
      );
    }
  }

  const multiscale = root.attributes.multiscales?.[0];
  const axes: any[] = multiscale?.axes ?? [];
  const neuroglancerHints = root.attributes.neuroglancer?.coordinate_space;
  const axisNames: string[] = [];
  const axisUnits: string[] = [];
  const axisScales: number[] = [];
  for (let i = 0; i < rank; ++i) {
    const hintUnit = neuroglancerHints?.units?.[i];
    const hintScale = neuroglancerHints?.scales?.[i];
    const name =
      neuroglancerHints?.names?.[i] ?? axes[i]?.name ?? ["x", "y", "z"][i];
    // The per-level NGFF scale is the pyramid's bin ratio, not a unit size,
    // so only the unit is taken from the axes.
    const normalized = normalizeUnitScale(
      hintScale !== undefined ? Number(hintScale) : 1,
      hintUnit ?? axes[i]?.unit ?? zv?.crs?.units,
    );
    axisNames.push(String(name));
    axisUnits.push(normalized.unit);
    axisScales.push(normalized.scale);
  }
  if (axisUnits.every((u) => u === "")) {
    warnings.push(
      "store declares no axis units; it is shown unitless. Declare a unit " +
        "(e.g. zarr-vectors-py create_store(unit=...)) to compose it with " +
        "other data",
    );
  }

  let coordinateOffset: number[] | undefined;
  const offsetRaw = zv.coordinate_offset;
  if (Array.isArray(offsetRaw) && offsetRaw.length === rank) {
    const offset = offsetRaw.map(Number);
    if (offset.every(Number.isFinite) && offset.some((v) => v !== 0)) {
      coordinateOffset = offset;
    }
  }

  return {
    url,
    zvVersion: zv.zv_version,
    geometryKind,
    rank,
    lowerBounds,
    upperBounds,
    coordinateOffset,
    axisNames,
    axisUnits,
    axisScales,
    linksConvention,
    linkWidth,
    linkedSkeletonLayout,
    linksDirected: familyAttrs.directed === true,
    levels: levels.filter(
      (l) => l.arrays.vertices.attributes?.encoding !== "draco",
    ),
    attributes,
    vertexIdAttribute:
      typeof zv.vertex_id_attribute === "string"
        ? zv.vertex_id_attribute
        : undefined,
    hasObjects,
    warnings,
  };
}
