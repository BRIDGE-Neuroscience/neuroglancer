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
 * level holds, and which vertex attributes to expose. The result is plain
 * data, posted as is to the chunk worker.
 */

import type { ZarrVectorsGeometryDescription } from "#src/datasource/zarr-vectors/base.js";
import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  attributeDtype,
  ELEMENT_BYTES,
  isElementType,
} from "#src/datasource/zarr-vectors/dtype.js";
import {
  KIND_CAPABILITIES,
  resolveGeometryKind,
} from "#src/datasource/zarr-vectors/geometry_kind.js";
import { intraLinksPath } from "#src/datasource/zarr-vectors/links.js";
import { mapConcurrent } from "#src/datasource/zarr-vectors/util.js";
import type { ZarrArrayRead } from "#src/datasource/zarr-vectors/zarr_array.js";
import {
  coalesceRangeReads,
  parseZarrArrayMetadata,
  ShardIndexCache,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import { joinBaseUrlAndPath } from "#src/kvstore/url.js";
import { allSiPrefixes, supportedUnits } from "#src/util/si_units.js";

/** Format versions this reader understands: 0.9.0 to 0.9.4. */
const SUPPORTED = { major: 0, minor: 9, maxPatch: 4 };

/** Vertex attributes exposed without an `#attributes=` selection. */
const DEFAULT_ATTRIBUTE_LIMIT = 8;
/** Attributes that can be shown at once (one texture and varying each). */
export const MAX_ATTRIBUTES = 12;

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
  dtype: ElementType;
  /** Values per vertex. */
  components: number;
  /** Category labels of a dictionary-encoded attribute. */
  enumLabels?: string[];
  /**
   * Not stored per vertex: an object's value, the same on every vertex of
   * it (`obj_<column>`, or `obj_group` for its first group's index).
   */
  objectValue?: { column: string | undefined };
}

/** `#attributes=` names with this prefix choose object attributes. */
export const OBJECT_ATTRIBUTE_PREFIX = "obj:";

/** `zarr.json` of each per-chunk array of a level. */
export interface ZarrVectorsLevelArrays {
  vertices: any;
  vertexFragments: any | undefined;
  fragmentSegmentIds: any | undefined;
  intraLinks: any | undefined;
  /** Parallel to the store's attributes; `undefined` where absent. */
  attributes: (any | undefined)[];
}

export interface ZarrVectorsLevel {
  index: number;
  /** Level directory relative to the store root, e.g. `"0"`. */
  path: string;
  /** Physical size of one spatial cell at this level. */
  chunkShape: number[];
  vertexCount: number | undefined;
  /**
   * The level is stamped `fragment_link_groups`: each chunk has one intra
   * link group per vertex fragment, in fragment order.
   */
  fragmentLinkGroups: boolean;
  /**
   * How the level relates to the next coarser one: `"replace"`, a complete
   * representation on its own, or `"add"`, the data the coarser levels do
   * not hold (its complete content is its own and the next level's).
   * `vertexCount` counts the level's own vertices either way.
   */
  refinement: "replace" | "add";
  /**
   * The level's chunk keys lead with an attribute bin (`chunk_dims` of four
   * axes, or `chunk_attribute_values`): its cells are not spatial alone.
   */
  attributeChunked: boolean;
  arrays: ZarrVectorsLevelArrays;
}

export interface ZarrVectorsStore {
  description: ZarrVectorsGeometryDescription;
  lowerBounds: number[];
  upperBounds: number[];
  /** World position of stored coordinate 0, when the store declares one. */
  coordinateOffset: number[] | undefined;
  axisNames: string[];
  /** Base SI unit per axis (`"m"`, or `""` when undeclared). */
  axisUnits: string[];
  /** Size of one stored coordinate unit, in `axisUnits`. */
  axisScales: number[];
  levels: ZarrVectorsLevel[];
  /** Diagnostics worth showing once. */
  warnings: string[];
}

/** Reads and lists under a store root. */
export interface ZarrVectorsStoreAccess {
  read: ZarrArrayRead;
  /** Child directory names of `path`; throws if listing is unsupported. */
  listDirectories(path: string, signal?: AbortSignal): Promise<string[]>;
  shardIndexes: ShardIndexCache;
}

/** Store access through a (frontend or worker) kvstore context. */
export function kvStoreAccess(
  context: {
    read(
      url: string,
      options: any,
    ): Promise<{ response: Response; offset: number } | undefined>;
    list(url: string, options: any): Promise<{ directories: string[] }>;
  },
  storeUrl: string,
  shardIndexes = new ShardIndexCache(),
): ZarrVectorsStoreAccess {
  return {
    // Cells of one shard needed together are fetched together.
    read: coalesceRangeReads(async (path, options) => {
      const { byteRange } = options;
      const response = await context.read(joinBaseUrlAndPath(storeUrl, path), {
        signal: options.signal,
        byteRange,
      });
      if (response === undefined) return undefined;
      const bytes = new Uint8Array(await response.response.arrayBuffer());
      if (byteRange === undefined) return bytes;
      // A server may ignore the range and send more, such as the whole
      // file; keep only what was asked for, wherever it starts.
      const start =
        "suffixLength" in byteRange
          ? response.offset + bytes.length - byteRange.suffixLength
          : byteRange.offset;
      const from = start - response.offset;
      const length =
        "suffixLength" in byteRange ? byteRange.suffixLength : byteRange.length;
      if (from < 0) {
        throw new Error(
          `${path}: response starts at byte ${response.offset}, after ${start}`,
        );
      }
      return from === 0 && bytes.length <= length
        ? bytes
        : bytes.slice(from, from + length);
    }),
    async listDirectories(path, signal) {
      const response = await context.list(
        joinBaseUrlAndPath(storeUrl, `${path}/`),
        { responseKeys: "suffix", signal },
      );
      return response.directories
        .map((d) => d.replace(/\/$/, ""))
        .filter((d) => d !== "");
    },
    shardIndexes,
  };
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

/** An array's `zarr.json`, checked to be readable, or `undefined` if absent. */
async function readArrayJson(
  read: ZarrArrayRead,
  path: string,
  signal?: AbortSignal,
): Promise<any | undefined> {
  const json = await readJson(read, `${path}/zarr.json`, signal);
  if (json?.node_type !== "array") return undefined;
  parseZarrArrayMetadata(path, json);
  return json;
}

// -------------------------------------------------------------- units

const LONG_UNITS = new Map<string, { unit: string; exponent: number }>([
  ["micron", { unit: "m", exponent: -6 }],
  ["microns", { unit: "m", exponent: -6 }],
]);
for (const base of ["meter", "second"]) {
  for (const p of allSiPrefixes) {
    if (p.longPrefix !== undefined) {
      LONG_UNITS.set(`${p.longPrefix}${base}`, {
        unit: base[0],
        exponent: p.exponent,
      });
    }
  }
}

/**
 * Folds an SI prefix into the scale: `(1, "mm")` becomes `(0.001, "m")`. An
 * unknown or missing unit stays unitless rather than defaulting to metres,
 * which would place the data 1000x away from a millimetre frame.
 */
export function normalizeUnitScale(scale: number, unit: unknown) {
  if (typeof unit !== "string" || unit === "") return { unit: "", scale };
  const known = LONG_UNITS.get(unit) ?? supportedUnits.get(unit);
  return known === undefined
    ? { unit: "", scale }
    : { unit: known.unit, scale: scale * 10 ** known.exponent };
}

// -------------------------------------------------------------- versions

function checkFormat(zv: any, warnings: string[]) {
  const raw = zv?.zv_version;
  const match = /^(\d+)\.(\d+)(?:\.(\d+))?/.exec(String(raw ?? ""));
  if (match === null) {
    warnings.push(
      `unrecognised zv_version ${JSON.stringify(raw)}; reading it as 0.9`,
    );
  } else {
    const [major, minor, patch] = [1, 2, 3].map((i) => Number(match[i] ?? 0));
    if (
      major < SUPPORTED.major ||
      (major === SUPPORTED.major && minor < SUPPORTED.minor)
    ) {
      throw new Error(
        `store zv_version ${raw} predates the 0.9 layout this viewer reads; ` +
          "rewrite it with zarr-vectors >= 0.9",
      );
    }
    if (
      major > SUPPORTED.major ||
      minor > SUPPORTED.minor ||
      patch > SUPPORTED.maxPatch
    ) {
      warnings.push(
        `store zv_version ${raw} is newer than this viewer's ` +
          `${SUPPORTED.major}.${SUPPORTED.minor}.${SUPPORTED.maxPatch} reader; ` +
          "newer features may not render",
      );
    }
  }
  // A store lists what a reader must understand to read it correctly; an
  // unknown one would be read wrongly without a word, so it is refused.
  const required = zv?.required_capabilities;
  if (Array.isArray(required)) {
    const unknown = required.filter(
      (c: unknown) => !SUPPORTED_REQUIRED_CAPABILITIES.has(String(c)),
    );
    if (unknown.length > 0) {
      throw new Error(
        `this store requires ${unknown.join(", ")}, which this viewer does ` +
          "not implement",
      );
    }
  }
  const capabilities = zv?.format_capabilities;
  if (Array.isArray(capabilities) && capabilities.includes("dense_manifests")) {
    warnings.push(
      "this store uses dense manifests (0.9.4), which this viewer cannot read; " +
        "selected objects and segment properties are unavailable",
    );
  }
}

/** `required_capabilities` this viewer implements. */
const SUPPORTED_REQUIRED_CAPABILITIES = new Set(["additive_levels"]);

/**
 * The levels whose union is level `index`'s complete content: itself, and,
 * while a level is `refinement: "add"`, the next coarser one too.
 */
export function levelChain(
  levels: readonly { refinement?: ZarrVectorsLevel["refinement"] }[],
  index: number,
): number[] {
  const out = [index];
  for (
    let i = index;
    i + 1 < levels.length && levels[i].refinement === "add";
    ++i
  ) {
    out.push(i + 1);
  }
  return out;
}

/** Throws unless the object index layout is one this reader resolves. */
export function checkObjectIndexLayout(objectIndexAttrs: any) {
  const layout = objectIndexAttrs?.layout;
  if (
    layout !== undefined &&
    layout !== "vlen_manifests_v1" &&
    layout !== "vlen_manifests_v2"
  ) {
    throw new Error(
      `object_index layout ${JSON.stringify(layout)} is not supported`,
    );
  }
}

// -------------------------------------------------------------- attributes

/**
 * Names that must not become shader identifiers: Neuroglancer's skeleton
 * shader `#define`s each attribute's bare name, so one called `z` breaks
 * every `.z` swizzle and one called `length` breaks `length()`.
 */
const GLSL_UNSAFE = new Set(
  (
    "x y z w r g b a s t p q abs all any ceil clamp cos cross degrees " +
    "distance dot exp exp2 floor fract inverse length log log2 max min mix " +
    "mod normalize not pow radians reflect refract round sign sin smoothstep " +
    "sqrt step tan transpose texture attribute uniform varying in out inout " +
    "const float int uint bool vec2 vec3 vec4 mat2 mat3 mat4 void if else for " +
    "while do return break continue discard true false highp mediump lowp " +
    "precision struct switch case default layout flat smooth centroid " +
    "sampler2D segment tangent position color main selectedNodeAttr"
  ).split(" "),
);

/** A GLSL-safe identifier for an attribute name, unique within `used`. */
export function safeAttributeId(name: string, used: Set<string>): string {
  let base = name.replace(/[^a-zA-Z0-9_]/g, "_");
  if (!/^[a-z]/.test(base)) base = `a_${base}`;
  base = base.replace(/__+/g, "_");
  if (GLSL_UNSAFE.has(base) || base.startsWith("gl_")) base = `attr_${base}`;
  let id = base;
  for (let i = 2; used.has(id); ++i) id = `${base}_${i}`;
  used.add(id);
  return id;
}

type CellSample = () => Promise<
  { cell: number[]; vertices: number } | undefined
>;

/** A populated cell of a level's `vertices` and its vertex count. */
function vertexSample(
  access: ZarrVectorsStoreAccess,
  path: string,
  json: any,
): CellSample {
  let found: ReturnType<CellSample> | undefined;
  return () => {
    found ??= (async () => {
      const reader = new ZarrArrayReader(
        parseZarrArrayMetadata(path, json),
        access.read,
        access.shardIndexes,
      );
      const { nonemptyCells, shape, origin } = reader.array;
      const candidates = [...(nonemptyCells ?? [])]
        .slice(0, 4)
        .map((key) => key.split(".").map(Number));
      for (let i = 0; candidates.length === 0 && i < 64; ++i) {
        if (i >= shape[0] * shape[1] * shape[2]) break;
        candidates.push([
          Math.floor(i / (shape[1] * shape[2])) + origin[0],
          (Math.floor(i / shape[2]) % shape[1]) + origin[1],
          (i % shape[2]) + origin[2],
        ]);
      }
      const dtype = (json.attributes?.dtype ?? "float32") as ElementType;
      for (const cell of candidates) {
        const bytes = await reader.readCell(cell);
        if (bytes !== undefined && bytes.byteLength > 0) {
          return {
            cell,
            vertices: bytes.byteLength / (3 * ELEMENT_BYTES[dtype]),
          };
        }
      }
      return undefined;
    })();
    return found;
  };
}

/**
 * Values per vertex. `row_shape` says so, except that zarr-vectors-py stamps
 * `[]` on `(N, C)` attributes, so an empty one is measured from a populated
 * cell.
 */
async function attributeWidth(
  access: ZarrVectorsStoreAccess,
  path: string,
  json: any,
  sample: CellSample,
): Promise<number> {
  const rowShape = json.attributes?.row_shape;
  if (Array.isArray(rowShape) && rowShape.length > 0) {
    return rowShape.reduce((a: number, b: number) => a * Number(b), 1);
  }
  // zarr-vectors-py's own order: row_shape, then the channel names.
  const channelNames = json.attributes?.channel_names;
  if (Array.isArray(channelNames) && channelNames.length > 0) {
    return channelNames.length;
  }
  const found = await sample();
  if (found === undefined) return 1;
  const reader = new ZarrArrayReader(
    parseZarrArrayMetadata(path, json),
    access.read,
    access.shardIndexes,
  );
  const bytes = await reader.readCell(found.cell);
  if (bytes === undefined) return 1;
  const dtype = attributeDtype(json) as ElementType;
  const width = bytes.byteLength / (found.vertices * ELEMENT_BYTES[dtype]);
  return Number.isInteger(width) && width >= 1 ? width : 1;
}

/** Chooses the vertex attributes to expose and reads their metadata. */
async function selectAttributes(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  verticesJson: any,
  zv: any,
  selected: readonly string[] | undefined,
  synthesisesTangent: boolean,
  warnings: string[],
  signal?: AbortSignal,
): Promise<{ attributes: ZarrVectorsAttribute[]; json: any[] }> {
  let names: string[];
  // Declared in the root's attribute_specs but perhaps not written yet:
  // dropped quietly if absent, unlike a listed or chosen name.
  let declaredOnly = new Set<string>();
  if (selected !== undefined) {
    names = [...new Set(selected)];
    if (names.length > MAX_ATTRIBUTES) {
      throw new Error(
        `#attributes names more than ${MAX_ATTRIBUTES} attributes`,
      );
    }
  } else {
    const declared = new Set(Object.keys(zv?.attribute_specs?.vertex ?? {}));
    const found = new Set<string>();
    const group = await readJson(
      access.read,
      `${levelPath}/vertex_attributes/zarr.json`,
      signal,
    );
    if (group !== undefined) {
      try {
        const listed = await access.listDirectories(
          `${levelPath}/vertex_attributes`,
          signal,
        );
        for (const name of listed) found.add(name);
      } catch (e) {
        warnings.push(
          "could not list vertex_attributes/ " +
            `(${e instanceof Error ? e.message : e}); name attributes with ` +
            "#attributes= to show them",
        );
      }
    }
    for (const name of declared) {
      if (!found.has(name)) declaredOnly.add(name);
      found.add(name);
    }
    // Where the renderer synthesises `tangent`, a stored one would shadow it.
    if (synthesisesTangent) found.delete("tangent");
    names = [...found].sort();
    if (names.length > DEFAULT_ATTRIBUTE_LIMIT) {
      // A gene panel would fetch that many cells per chunk; let the user choose.
      warnings.push(
        `store has ${names.length} vertex attributes; none load by default. ` +
          `Append #attributes=a,b,c to the source URL (up to ${MAX_ATTRIBUTES})`,
      );
      names = [];
      declaredOnly = new Set();
    }
  }
  const sample = vertexSample(access, `${levelPath}/vertices`, verticesJson);
  const results = new Array<{
    attribute?: ZarrVectorsAttribute;
    json?: any;
    problem?: string;
  }>(names.length);
  await mapConcurrent(names, 16, async (name, i) => {
    const path = `${levelPath}/vertex_attributes/${name}`;
    try {
      const json = await readArrayJson(access.read, path, signal);
      if (json === undefined) {
        if (declaredOnly.has(name)) return;
        throw new Error("missing");
      }
      const dtype = attributeDtype(json);
      if (!isElementType(dtype)) throw new Error(`dtype ${dtype}`);
      const components = await attributeWidth(access, path, json, sample);
      if (components > 4) {
        throw new Error(`${components} components (at most 4)`);
      }
      const categories = json.attributes?.categories;
      results[i] = {
        json,
        attribute: {
          name,
          id: "",
          dtype,
          components,
          enumLabels:
            json.attributes?.encoding === "dictionary" &&
            Array.isArray(categories)
              ? categories.map(String)
              : undefined,
        },
      };
    } catch (e) {
      results[i] = {
        problem: `${name} (${e instanceof Error ? e.message : e})`,
      };
    }
  });
  const problems = results.flatMap((r) => (r?.problem ? [r.problem] : []));
  if (problems.length > 0) {
    if (selected !== undefined) {
      throw new Error(
        `#attributes names attributes this viewer cannot read: ${problems.join(", ")}`,
      );
    }
    warnings.push(`skipped vertex attributes: ${problems.join(", ")}`);
  }
  const kept = results.filter((r) => r?.attribute !== undefined);
  const wide = kept
    .filter(
      (r) => r.attribute!.dtype === "int64" || r.attribute!.dtype === "uint64",
    )
    .map((r) => r.attribute!.name);
  if (wide.length > 0) {
    warnings.push(
      `vertex attribute(s) ${wide.join(", ")} are 64-bit integers, drawn as ` +
        "float32: values beyond 16,777,216 (such as segment ids) lose precision",
    );
  }
  const used = new Set<string>();
  for (const r of kept)
    r.attribute!.id = safeAttributeId(r.attribute!.name, used);
  return {
    attributes: kept.map((r) => r.attribute!),
    json: kept.map((r) => r.json),
  };
}

// -------------------------------------------------------------- skeletons

/**
 * zarr-vectors-tools' skeleton layout (`skeleton_layout.py`): recorded on the
 * root; otherwise an SWC header means linked, a per-fragment `segment_id`
 * means split (the precomputed ingests), and anything else linked.
 */
async function skeletonLayout(
  access: ZarrVectorsStoreAccess,
  root: any,
  level0: ZarrVectorsLevelArrays,
  signal?: AbortSignal,
): Promise<"linked" | "split"> {
  const marker = root.attributes?.zarr_vectors_tools?.skeleton_layout;
  if (marker === "linked_across_chunks") return "linked";
  if (marker === "split_at_chunk_faces") return "split";
  const swc = await readJson(access.read, "headers/swc/zarr.json", signal);
  if (swc !== undefined) return "linked";
  return level0.fragmentSegmentIds !== undefined ? "split" : "linked";
}

// -------------------------------------------------------------- levels

function levelPaths(multiscales: any): string[] {
  const paths = (multiscales?.[0]?.datasets ?? [])
    .map((d: any) => d?.path)
    .filter((p: unknown): p is string => typeof p === "string" && p !== "");
  return paths.length > 0 ? paths : ["0"];
}

async function readLevel(
  access: ZarrVectorsStoreAccess,
  index: number,
  path: string,
  rootChunkShape: number[],
  attributes: ZarrVectorsAttribute[],
  attributeJson: any[] | undefined,
  intraLinks: string | undefined,
  hasObjects: boolean,
  signal?: AbortSignal,
): Promise<ZarrVectorsLevel> {
  const optional = (p: string) =>
    readArrayJson(access.read, `${path}/${p}`, signal);
  const [
    levelJson,
    vertices,
    vertexFragments,
    fragmentSegmentIds,
    links,
    ...attrs
  ] = await Promise.all([
    readJson(access.read, `${path}/zarr.json`, signal),
    readJson(access.read, `${path}/vertices/zarr.json`, signal),
    optional("vertex_fragments"),
    hasObjects ? optional("fragment_attributes/segment_id") : undefined,
    intraLinks !== undefined ? optional(intraLinks) : undefined,
    ...attributes.map(
      (a, i) => attributeJson?.[i] ?? optional(`vertex_attributes/${a.name}`),
    ),
  ]);
  if (vertices?.node_type !== "array") {
    throw new Error(
      `${path}/vertices is not a zarr array; stores written before ` +
        "zarr-vectors 0.9.0 must be rewritten",
    );
  }
  parseZarrArrayMetadata(`${path}/vertices`, vertices);
  const dtype = vertices.attributes?.dtype ?? "float32";
  if (!isElementType(dtype)) {
    throw new Error(`${path}/vertices has unsupported dtype ${dtype}`);
  }
  const meta = levelJson?.attributes?.zarr_vectors_level ?? {};
  const vertexCount = Number(meta.vertex_count);
  return {
    index,
    path,
    chunkShape:
      Array.isArray(meta.chunk_shape) && meta.chunk_shape.length === 3
        ? meta.chunk_shape.map(Number)
        : rootChunkShape,
    vertexCount: Number.isFinite(vertexCount) ? vertexCount : undefined,
    fragmentLinkGroups: meta.fragment_link_groups === true,
    refinement: meta.refinement === "add" ? "add" : "replace",
    attributeChunked:
      (Array.isArray(meta.chunk_dims) && meta.chunk_dims.length > 3) ||
      Array.isArray(meta.chunk_attribute_values),
    arrays: {
      vertices,
      vertexFragments,
      fragmentSegmentIds,
      intraLinks: links,
      attributes: attrs,
    },
  };
}

// -------------------------------------------------------------- store

export async function openZarrVectorsStore(
  access: ZarrVectorsStoreAccess,
  selectedAttributes: readonly string[] | undefined,
  signal?: AbortSignal,
): Promise<ZarrVectorsStore> {
  const { read } = access;
  const root = await readJson(read, "zarr.json", signal);
  if (root === undefined) {
    throw new Error(
      "no zarr.json found: the URL must be the store's root folder (the one " +
        "holding zarr.json), served by a running server",
    );
  }
  const zv = root?.attributes?.zarr_vectors;
  if (zv === undefined) {
    throw new Error(
      "not a zarr-vectors store: the root zarr.json has no zarr_vectors block",
    );
  }
  const warnings: string[] = [];
  checkFormat(zv, warnings);
  const [lower, upper] = Array.isArray(zv.bounds) ? zv.bounds : [];
  if (lower?.length !== 3 || upper?.length !== 3) {
    throw new Error(
      "this viewer reads 3-D stores; bounds must be [[x, y, z], [x, y, z]]",
    );
  }
  if (!Array.isArray(zv.chunk_shape) || zv.chunk_shape.length !== 3) {
    throw new Error("store chunk_shape must have 3 entries");
  }

  const paths = levelPaths(root.attributes.multiscales);
  const [family, level0Vertices, objectIndex] = await Promise.all([
    readJson(read, `${paths[0]}/links/0/zarr.json`, signal),
    readJson(read, `${paths[0]}/vertices/zarr.json`, signal),
    readJson(read, `${paths[0]}/object_index/zarr.json`, signal),
  ]);
  if (level0Vertices?.node_type !== "array") {
    throw new Error(
      `${paths[0]}/vertices is not a zarr array; stores written before ` +
        "zarr-vectors 0.9.0 must be rewritten",
    );
  }
  const declaredWidth = Number(family?.attributes?.link_width);
  const { kind, ignored } = resolveGeometryKind(zv.geometry_types ?? [], {
    present: family !== undefined,
    width: Number.isInteger(declaredWidth) ? declaredWidth : undefined,
  });
  if (ignored.length > 0) {
    warnings.push(
      `ignoring declared geometry ${ignored.join(", ")}; rendering ${kind}`,
    );
  }
  const { primitive } = KIND_CAPABILITIES[kind];
  const convention =
    zv.links_convention ??
    (kind === "skeleton"
      ? "implicit_sequential_with_branches"
      : kind === "graph" || kind === "mesh"
        ? "explicit"
        : "implicit_sequential");
  if (
    convention !== "implicit_sequential" &&
    convention !== "implicit_sequential_with_branches" &&
    convention !== "explicit"
  ) {
    throw new Error(`unknown links_convention ${JSON.stringify(convention)}`);
  }
  const linkWidth =
    Number.isInteger(declaredWidth) && declaredWidth >= 2
      ? declaredWidth
      : primitive === "triangles"
        ? 3
        : 2;
  const hasObjects = objectIndex !== undefined;
  const intraLinks =
    primitive !== "points" && convention !== "implicit_sequential"
      ? intraLinksPath(linkWidth)
      : undefined;

  const { attributes, json } = await selectAttributes(
    access,
    paths[0],
    level0Vertices,
    zv,
    // `obj:` names choose object attributes; the frontend handles them.
    selectedAttributes?.filter((n) => !n.startsWith(OBJECT_ATTRIBUTE_PREFIX)),
    KIND_CAPABILITIES[kind].tangent !== undefined,
    warnings,
    signal,
  );
  const allLevels = await Promise.all(
    paths.map((path, index) =>
      readLevel(
        access,
        index,
        path,
        zv.chunk_shape.map(Number),
        attributes,
        index === 0 ? json : undefined,
        intraLinks,
        hasObjects,
        signal,
      ),
    ),
  );
  const levels = allLevels.filter((level) => {
    if (level.attributeChunked) {
      warnings.push(
        `${level.path} is chunked by attribute value as well as space ` +
          "(chunk_attribute_values), which this viewer cannot read yet",
      );
      return false;
    }
    if (level.arrays.vertices.attributes?.encoding !== "draco") return true;
    warnings.push(
      `${level.path} stores Draco-encoded vertices, which this viewer ` +
        "cannot read yet",
    );
    return false;
  });
  if (levels.length === 0) {
    throw new Error("the store has no level this viewer can read");
  }

  const axes: any[] = root.attributes.multiscales?.[0]?.axes ?? [];
  const hints = root.attributes.neuroglancer?.coordinate_space;
  const units = [0, 1, 2].map((i) =>
    normalizeUnitScale(
      Number(hints?.scales?.[i] ?? 1),
      hints?.units?.[i] ?? axes[i]?.unit ?? zv.crs?.units,
    ),
  );
  if (units.every((u) => u.unit === "")) {
    warnings.push("store declares no axis units; it is shown unitless");
  }
  const offset: number[] = Array.isArray(zv.coordinate_offset)
    ? zv.coordinate_offset.map(Number)
    : [];

  return {
    description: {
      geometryKind: kind,
      linksConvention: convention,
      linkWidth,
      skeletonLayout:
        convention === "implicit_sequential_with_branches"
          ? await skeletonLayout(access, root, levels[0].arrays, signal)
          : undefined,
      attributes,
      hasObjects,
    },
    lowerBounds: lower.map(Number),
    upperBounds: upper.map(Number),
    coordinateOffset:
      offset.length === 3 &&
      offset.every(Number.isFinite) &&
      offset.some((v) => v !== 0)
        ? offset
        : undefined,
    axisNames: [0, 1, 2].map((i) =>
      String(hints?.names?.[i] ?? axes[i]?.name ?? "xyz"[i]),
    ),
    axisUnits: units.map((u) => u.unit),
    axisScales: units.map((u) => u.scale),
    levels,
    warnings,
  };
}

// -------------------------------------------------------------- URLs

const ATTRIBUTES_PREFIX = "attributes=";

/** The `#attributes=a,b` selection of a source URL, or `undefined`. */
export function parseAttributesFragment(
  fragment: string | undefined,
): string[] | undefined {
  if (!fragment) return undefined;
  if (!fragment.startsWith(ATTRIBUTES_PREFIX)) {
    throw new Error("the only supported fragment is #attributes=<names>");
  }
  // Split before decoding: an encoded comma is part of a name.
  return fragment
    .slice(ATTRIBUTES_PREFIX.length)
    .split(",")
    .map((part) => {
      try {
        return decodeURIComponent(part.trim());
      } catch {
        return part.trim();
      }
    })
    .filter((name) => name !== "");
}

/** The inverse of {@link parseAttributesFragment}. */
export function formatAttributesFragment(
  attributes: readonly string[] | undefined,
): string {
  if (attributes === undefined) return "";
  return `#${ATTRIBUTES_PREFIX}${attributes.map(encodeURIComponent).join(",")}`;
}

/**
 * The cells a level's grid spans. Computed in floats: a sub-unit chunk
 * (0.5 mm) must not truncate to zero.
 */
export function chunkIndexBounds(
  lowerBounds: readonly number[],
  upperBounds: readonly number[],
  chunkShape: readonly number[],
) {
  const lowerChunkBound = new Float32Array(3);
  const upperChunkBound = new Float32Array(3);
  for (let i = 0; i < 3; ++i) {
    if (!(chunkShape[i] > 0)) {
      throw new Error(`chunk_shape[${i}] must be positive`);
    }
    lowerChunkBound[i] = Math.floor(lowerBounds[i] / chunkShape[i]);
    // The upper bound is the largest coordinate, inclusive: a vertex on it
    // is stored in the cell that starts there.
    upperChunkBound[i] = Math.floor(upperBounds[i] / chunkShape[i]) + 1;
  }
  return { lowerChunkBound, upperChunkBound };
}
