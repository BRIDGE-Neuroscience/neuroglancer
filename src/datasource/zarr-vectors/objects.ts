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
 * @file The object layer of one level: which object each row is, the
 * Neuroglancer segment id it is shown as, its attributes and its groups.
 *
 * Three id spaces meet here:
 *  - the ROW of an object in `object_index/manifests` and the
 *    `object_attributes/*` columns;
 *  - its OBJECT ID: `object_index/object_ids[row]` in the 0.9.2 layout
 *    (`vlen_manifests_v2`), the row itself before that;
 *  - its SEGMENT ID: `object_attributes/segment_id[row]` when the store has
 *    one (an EM root id, a 1-based SWC id), else the object id.
 * Neuroglancer shows segment ids, and `fragment_attributes/segment_id` carries
 * the same values per fragment, so geometry and properties agree.
 */

import type { ElementType } from "#src/datasource/zarr-vectors/dtype.js";
import {
  decodeFloat32,
  decodeUint64,
  ELEMENT_BYTES,
} from "#src/datasource/zarr-vectors/dtype.js";
import type { ZarrVectorsStoreAccess } from "#src/datasource/zarr-vectors/store.js";
import {
  checkObjectIndexLayout,
  readJson,
} from "#src/datasource/zarr-vectors/store.js";
import { mapConcurrent, warnOnce } from "#src/datasource/zarr-vectors/util.js";
import {
  decodeUtf32,
  fillBytes,
  parseZarrArrayMetadata,
  parseZarrJsonText,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import type {
  InlineSegmentProperty,
  InlineSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import type { TypedNumberArray } from "#src/util/array.js";
import { DataType } from "#src/util/data_type.js";

export interface ZarrVectorsObjectTable {
  numObjects: number;
  /** Object id per row, or `undefined` when rows are the ids. */
  objectIds: BigUint64Array | undefined;
  /** Neuroglancer segment id per row. */
  segmentIds: BigUint64Array;
  /** Where the segment ids came from, for diagnostics. */
  idSource: "segment_id" | "object_ids" | "row";
  /** `object_index/manifests` metadata, for per-object reads. */
  manifestsJson: any | undefined;
  /** `object_index` group attributes. */
  objectIndexAttrs: any;
}

/** Opens an array, keeping 64-bit fill values exact; `undefined` if absent. */
async function openReader(
  access: ZarrVectorsStoreAccess,
  path: string,
  signal?: AbortSignal,
): Promise<ZarrArrayReader | undefined> {
  const bytes = await access.read(`${path}/zarr.json`, { signal });
  if (bytes === undefined) return undefined;
  const json = parseZarrJsonText(new TextDecoder().decode(bytes));
  if (json?.node_type !== "array") return undefined;
  return new ZarrArrayReader(
    parseZarrArrayMetadata(path, json),
    access.read,
    access.shardIndexes,
  );
}

/** Reads the object table of a level, or `undefined` if it has no objects. */
export async function readObjectTable(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  signal?: AbortSignal,
): Promise<ZarrVectorsObjectTable | undefined> {
  const objectIndex = await readJson(
    access.read,
    `${levelPath}/object_index/zarr.json`,
    signal,
  );
  if (objectIndex === undefined) return undefined;
  const objectIndexAttrs = objectIndex.attributes ?? {};
  checkObjectIndexLayout(objectIndexAttrs);
  const [manifests, objectIdsReader, segmentIdReader] = await Promise.all([
    readJson(
      access.read,
      `${levelPath}/object_index/manifests/zarr.json`,
      signal,
    ),
    // Only the v2 layout has object ids; v1 rows are the ids.
    objectIndexAttrs.layout === "vlen_manifests_v1"
      ? undefined
      : openReader(access, `${levelPath}/object_index/object_ids`, signal),
    openReader(access, `${levelPath}/object_attributes/segment_id`, signal),
  ]);
  let numObjects = Number(objectIndexAttrs.num_objects);
  if (!Number.isInteger(numObjects)) {
    numObjects = Number(manifests?.shape?.[0] ?? 0);
  }
  if (!(numObjects > 0)) return undefined;

  const readIds = (reader: ZarrArrayReader | undefined) => {
    if (reader === undefined) return undefined;
    if (!(reader.array.shape[0] >= numObjects)) {
      // An edit that added objects without extending the column: the next
      // id source names them all.
      warnOnce(
        `${levelPath}: ${reader.array.path} has ${reader.array.shape[0]} ` +
          `rows for ${numObjects} objects; not used`,
      );
      return undefined;
    }
    return reader
      .readRows(0, numObjects, signal)
      .then((bytes) =>
        decodeUint64(
          bytes,
          reader.array.elementType as ElementType,
          numObjects,
        ),
      );
  };
  const [objectIds, segmentColumn] = await Promise.all([
    readIds(objectIdsReader),
    readIds(segmentIdReader),
  ]);
  let segmentIds: BigUint64Array;
  let idSource: ZarrVectorsObjectTable["idSource"];
  if (segmentColumn !== undefined) {
    segmentIds = segmentColumn;
    idSource = "segment_id";
  } else if (objectIds !== undefined) {
    segmentIds = objectIds;
    idSource = "object_ids";
  } else {
    segmentIds = new BigUint64Array(numObjects);
    for (let i = 0; i < numObjects; ++i) segmentIds[i] = BigInt(i);
    idSource = "row";
  }
  return {
    numObjects,
    objectIds,
    segmentIds,
    idSource,
    manifestsJson: manifests,
    objectIndexAttrs,
  };
}

/** Maps a segment id back to its row. */
export class SegmentIdIndex {
  private sortedIds: BigUint64Array;
  private rows: Uint32Array;
  constructor(segmentIds: BigUint64Array) {
    const n = segmentIds.length;
    const order = new Uint32Array(n);
    for (let i = 0; i < n; ++i) order[i] = i;
    let sorted = true;
    for (let i = 1; i < n; ++i) {
      if (segmentIds[i - 1] > segmentIds[i]) {
        sorted = false;
        break;
      }
    }
    if (!sorted) {
      order.sort((a, b) =>
        segmentIds[a] < segmentIds[b]
          ? -1
          : segmentIds[a] > segmentIds[b]
            ? 1
            : 0,
      );
    }
    this.rows = order;
    this.sortedIds = new BigUint64Array(n);
    for (let i = 0; i < n; ++i) this.sortedIds[i] = segmentIds[order[i]];
  }

  rowOf(id: bigint): number | undefined {
    const { sortedIds } = this;
    let lo = 0;
    let hi = sortedIds.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sortedIds[mid] < id) lo = mid + 1;
      else hi = mid;
    }
    return lo < sortedIds.length && sortedIds[lo] === id
      ? this.rows[lo]
      : undefined;
  }
}

// -------------------------------------------------------------- tags

/**
 * Id of the tags property.  Every group and every category of a
 * dictionary-encoded attribute is one tag, so the Seg tab can select
 * `#bundle_name` or `#cell_type=L2IT`.
 */
const TAGS_PROPERTY_ID = "group";

/** Neuroglancer encodes a segment's tags as one UTF-16 code unit each. */
export const MAX_TAGS = 0xffff;

/** One tag before naming: a group or a category, and the rows it holds. */
interface TagSource {
  name: string;
  description: string;
  rows: ArrayLike<number>;
}

/** Rows of `table` whose object id is `id`, without a per-id lookup table. */
function rowLookup(table: ZarrVectorsObjectTable) {
  const { objectIds, numObjects } = table;
  if (objectIds === undefined) {
    const end = BigInt(numObjects);
    return (id: bigint) => (id >= 0n && id < end ? Number(id) : undefined);
  }
  const index = new SegmentIdIndex(objectIds);
  return (id: bigint) => index.rowOf(id);
}

/**
 * Reads a per-group string column, as `group_attributes/<name>` (what
 * zarr-vectors-py writes) or `groupings_attributes/<name>` (the spec's
 * spelling).
 */
async function readGroupColumn(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  name: string,
  count: number,
  signal?: AbortSignal,
): Promise<(string | number)[] | undefined> {
  for (const group of ["group_attributes", "groupings_attributes"]) {
    const reader = await openReader(
      access,
      `${levelPath}/${group}/${name}`,
      signal,
    );
    if (reader === undefined) continue;
    const rows = Math.min(count, reader.array.shape[0]);
    if (reader.array.stringEncoding !== undefined) {
      return readStrings(reader, rows, signal);
    }
    if (reader.array.elementType === "vlen" || reader.array.shape.length > 1) {
      return undefined;
    }
    const bytes = await reader.readRows(0, rows, signal);
    return Array.from(
      decodeNumbers(bytes, reader.array.elementType as ElementType, rows),
    );
  }
  return undefined;
}

/** A level's groups as tag sources, in group id order. */
async function readGroupTags(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  table: ZarrVectorsObjectTable,
  warnings: string[],
  signal?: AbortSignal,
): Promise<TagSource[]> {
  const reader = await openReader(access, `${levelPath}/groups`, signal);
  if (reader === undefined) return [];
  const attrs = reader.array.attributes;
  const numGroups = Number(attrs.num_groups ?? reader.array.shape[0]);
  if (!(numGroups > 0)) return [];
  const columnOrWarn = (name: string) =>
    readGroupColumn(access, levelPath, name, numGroups, signal).catch((e) => {
      warnings.push(
        `could not read group ${name}s: ${e instanceof Error ? e.message : e}`,
      );
      return undefined;
    });
  const [blobs, storedNames, storedCounts, sourceColumns] = await Promise.all([
    reader.readVlenRows(0, numGroups, signal),
    columnOrWarn("name"),
    columnOrWarn("n_objects"),
    columnOrWarn("source_column"),
  ]);
  const listed = attrs.group_names;
  const rowOf = rowLookup(table);
  const { numObjects, objectIds } = table;
  const ranges = attrs.group_ranges;
  const out: TagSource[] = [];
  for (let gid = 0; gid < numGroups; ++gid) {
    let rows: ArrayLike<number>;
    const range = ranges?.[String(gid)];
    if (Array.isArray(range) && range.length === 2) {
      // A contiguous id range, clamped to the objects this level has: one
      // pass over the table, never a loop over the range itself.
      const clamp = (x: unknown, limit: number) =>
        Math.min(limit, Math.max(0, Math.floor(Number(x)) || 0));
      if (objectIds === undefined) {
        const start = clamp(range[0], numObjects);
        const end = clamp(range[1], numObjects);
        const list = new Uint32Array(Math.max(0, end - start));
        for (let i = 0; i < list.length; ++i) list[i] = start + i;
        rows = list;
      } else {
        const lo = BigInt(clamp(range[0], Number.MAX_SAFE_INTEGER));
        const hi = BigInt(clamp(range[1], Number.MAX_SAFE_INTEGER));
        const list: number[] = [];
        for (let r = 0; r < numObjects; ++r) {
          const id = objectIds[r];
          if (id >= lo && id < hi) list.push(r);
        }
        rows = list;
      }
    } else {
      const blob = blobs[gid];
      const list: number[] = [];
      if (blob !== undefined) {
        const view = new DataView(
          blob.buffer,
          blob.byteOffset,
          blob.byteLength,
        );
        for (let i = 0; i + 8 <= blob.byteLength; i += 8) {
          const row = rowOf(view.getBigUint64(i, true));
          if (row !== undefined) list.push(row);
        }
      }
      rows = list;
    }
    const fromAttrs = Array.isArray(listed) ? listed[gid] : listed?.[gid];
    let name =
      typeof fromAttrs === "string" && fromAttrs.trim() !== ""
        ? fromAttrs
        : String(storedNames?.[gid] ?? "");
    if (name.trim() === "") name = `group_${gid}`;
    const stored = Number(storedCounts?.[gid]);
    const members = Number.isFinite(stored) ? stored : rows.length;
    const details = [
      `group ${gid}`,
      `${members} object${members === 1 ? "" : "s"}`,
    ];
    const source = sourceColumns?.[gid];
    if (typeof source === "string" && source !== "") {
      details.push(`from ${source}`);
    }
    out.push({ name, description: details.join(", "), rows });
  }
  return out;
}

/** `true`/`false` as Python spells them, as group names built from them do. */
function categoryText(value: unknown): string {
  if (typeof value === "boolean") return value ? "True" : "False";
  return String(value);
}

/** A tag name Neuroglancer can parse: no spaces, never empty. */
function tagCandidate(name: string): string {
  return name.trim().replace(/\s+/g, "_");
}

interface NamedTags {
  tags: string[];
  tagDescriptions: string[];
  /** Per row, its tags as ascending, distinct character codes. */
  values: string[];
}

/**
 * Names and encodes tags.  Sources with the same stored name are one tag
 * holding the union of their rows (a group `cell_type=L2IT` and the category
 * `L2IT` of `cell_type`).  Neuroglancer matches tags without regard to case,
 * first wins, so a name that differs from an earlier one only in case, or
 * that had to be changed to parse, gets a suffix; names stored exactly win
 * over changed ones.  Only tags that hold a row get a code.
 */
function encodeTags(
  sources: TagSource[],
  numRows: number,
  warnings: string[],
): NamedTags | undefined {
  interface Merged {
    stored: string;
    candidate: string;
    descriptions: string[];
    rows: ArrayLike<number>[];
    count: number;
  }
  const byName = new Map<string, Merged>();
  const merged: Merged[] = [];
  for (const source of sources) {
    let entry = byName.get(source.name);
    if (entry === undefined) {
      entry = {
        stored: source.name,
        candidate: tagCandidate(source.name),
        descriptions: [],
        rows: [],
        count: 0,
      };
      byName.set(source.name, entry);
      merged.push(entry);
    }
    entry.descriptions.push(source.description);
    entry.rows.push(source.rows);
    entry.count += source.rows.length;
  }
  const used = merged.filter((m) => m.count > 0);
  if (used.length === 0) return undefined;
  if (used.length > MAX_TAGS) {
    warnings.push(
      `${used.length} groups and categories hold objects; Neuroglancer can ` +
        `tag at most ${MAX_TAGS}, so none are shown as tags`,
    );
    return undefined;
  }
  const names = new Array<string>(used.length);
  const taken = new Set<string>();
  const claim = (i: number, base: string) => {
    let name = base;
    for (let k = 2; taken.has(name.toLowerCase()); ++k) name = `${base}_${k}`;
    taken.add(name.toLowerCase());
    names[i] = name;
  };
  // Exact names first, so a sanitised name never takes one a store spells.
  used.forEach((m, i) => {
    if (m.candidate === m.stored && !taken.has(m.stored.toLowerCase())) {
      claim(i, m.stored);
    }
  });
  used.forEach((m, i) => {
    if (names[i] === undefined) claim(i, m.candidate);
  });
  const tagDescriptions = used.map((m, i) => {
    const renamed = names[i] !== m.stored ? `${m.stored}: ` : "";
    return renamed + m.descriptions.join("; ");
  });

  // Per row, the codes of its tags (CSR), ascending because tags are
  // visited in code order.
  const counts = new Uint32Array(numRows + 1);
  for (const m of used) {
    for (const rows of m.rows) {
      for (let i = 0; i < rows.length; ++i) ++counts[rows[i] + 1];
    }
  }
  for (let r = 0; r < numRows; ++r) counts[r + 1] += counts[r];
  const codes = new Uint16Array(counts[numRows]);
  const filled = new Uint32Array(numRows);
  used.forEach((m, code) => {
    for (const rows of m.rows) {
      for (let i = 0; i < rows.length; ++i) {
        const r = rows[i];
        const at = counts[r] + filled[r];
        if (filled[r] > 0 && codes[at - 1] === code) continue;
        codes[at] = code;
        ++filled[r];
      }
    }
  });
  const values = new Array<string>(numRows);
  for (let r = 0; r < numRows; ++r) {
    const start = counts[r];
    const end = start + filled[r];
    let s = "";
    for (let i = start; i < end; i += 4096) {
      s += String.fromCharCode(...codes.subarray(i, Math.min(end, i + 4096)));
    }
    values[r] = s;
  }
  return { tags: names, tagDescriptions, values };
}

// -------------------------------------------------------------- columns

/**
 * Decodes values of any numeric type to float64: exact for every float and
 * for integers within ±2^53.
 */
function decodeNumbers(
  bytes: Uint8Array,
  type: ElementType,
  count: number,
): Float64Array {
  if (type === "int64" || type === "uint64") {
    const ids = decodeUint64(bytes, type, count);
    const out = new Float64Array(count);
    for (let i = 0; i < count; ++i) {
      out[i] = Number(type === "int64" ? BigInt.asIntN(64, ids[i]) : ids[i]);
    }
    return out;
  }
  if (type === "float16" || type === "float32") {
    return Float64Array.from(decodeFloat32(bytes, type, count));
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float64Array(count);
  for (let i = 0; i < count; ++i) {
    switch (type) {
      case "float64":
        out[i] = view.getFloat64(8 * i, true);
        break;
      case "int8":
        out[i] = view.getInt8(i);
        break;
      case "uint8":
      case "bool":
        out[i] = view.getUint8(i);
        break;
      case "int16":
        out[i] = view.getInt16(2 * i, true);
        break;
      case "uint16":
        out[i] = view.getUint16(2 * i, true);
        break;
      case "int32":
        out[i] = view.getInt32(4 * i, true);
        break;
      case "uint32":
        out[i] = view.getUint32(4 * i, true);
        break;
    }
  }
  return out;
}

/** The first `count` strings of a string array. */
async function readStrings(
  reader: ZarrArrayReader,
  count: number,
  signal?: AbortSignal,
): Promise<string[]> {
  const { array } = reader;
  const out = new Array<string>(count);
  if (array.elementType === "utf32") {
    const bytes = await reader.readRows(0, count, signal);
    const width = array.elementBytes;
    for (let i = 0; i < count; ++i) {
      out[i] = decodeUtf32(bytes.subarray(i * width, (i + 1) * width));
    }
    return out;
  }
  const blobs = await reader.readVlenRows(0, count, signal);
  const decoder = new TextDecoder();
  for (let i = 0; i < count; ++i) {
    const blob = blobs[i];
    out[i] = blob === undefined ? "" : decoder.decode(blob);
  }
  return out;
}

/** A column ready to become a property, values in row order. */
type ColumnValues =
  | { type: "number"; dataType: DataType; values: TypedNumberArray }
  | { type: "string"; values: string[] };

interface DecodedChannel {
  /** Attribute name, plus the channel's for a vector attribute. */
  name: string;
  /** True when `name` is the attribute's own name (not a derived one). */
  own: boolean;
  description: string;
  column: ColumnValues;
}

interface DecodedAttribute {
  channels: DecodedChannel[];
  tags: TagSource[];
}

const NATIVE_TYPES: Partial<Record<ElementType, DataType>> = {
  bool: DataType.UINT8,
  int8: DataType.INT8,
  uint8: DataType.UINT8,
  int16: DataType.INT16,
  uint16: DataType.UINT16,
  int32: DataType.INT32,
  uint32: DataType.UINT32,
};

const NATIVE_ARRAYS: Partial<
  Record<DataType, { from(values: ArrayLike<number>): TypedNumberArray }>
> = {
  [DataType.UINT8]: Uint8Array,
  [DataType.INT8]: Int8Array,
  [DataType.UINT16]: Uint16Array,
  [DataType.INT16]: Int16Array,
  [DataType.UINT32]: Uint32Array,
  [DataType.INT32]: Int32Array,
};

/** Integers float32 holds exactly. */
const FLOAT32_EXACT = 2 ** 24;

/**
 * The property for one channel: Neuroglancer's own type when the values fit
 * it exactly, float32 with NaN where values are missing, else decimal text
 * (an EM root id has no exact float32 and Neuroglancer has no uint64
 * properties).  `values` is exact within ±2^53, which every bound tested
 * here is; `big` holds 64-bit values for the text.
 */
function channelColumn(
  type: ElementType,
  values: Float64Array,
  big: BigInt64Array | BigUint64Array | undefined,
  missing: Uint8Array | undefined,
): { column: ColumnValues; note?: string } {
  const n = values.length;
  let anyMissing = false;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; ++i) {
    if (missing?.[i]) {
      anyMissing = true;
      continue;
    }
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const number = (dataType: DataType, out: TypedNumberArray) => ({
    column: { type: "number" as const, dataType, values: out },
  });
  const float32 = () => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; ++i) out[i] = missing?.[i] ? Number.NaN : values[i];
    return number(DataType.FLOAT32, out);
  };
  if (type.startsWith("float")) return float32();
  if (!anyMissing) {
    const native =
      NATIVE_TYPES[type] ??
      (min >= 0 && max <= 0xffffffff
        ? DataType.UINT32
        : min >= -(2 ** 31) && max < 2 ** 31
          ? DataType.INT32
          : undefined);
    if (native !== undefined) {
      return number(native, NATIVE_ARRAYS[native]!.from(values));
    }
  }
  if (!(min < -FLOAT32_EXACT) && !(max > FLOAT32_EXACT)) return float32();
  const out = new Array<string>(n);
  for (let i = 0; i < n; ++i) {
    out[i] = missing?.[i] ? "" : String(big?.[i] ?? values[i]);
  }
  return {
    column: { type: "string", values: out },
    note: `${type} values beyond float32 precision, shown as text`,
  };
}

/**
 * Decodes one `object_attributes/<name>` column.  A row is missing when the
 * column says so: every channel equals the array's fill value and the writer
 * declared it the absent sentinel (`fill_sentinel_meaning: "absent"`, as
 * zarr-vectors-py writes), or a legacy `present_mask` sidecar clears it.
 * Missing values are NaN or empty text; the object keeps its other
 * properties.
 */
async function decodeAttribute(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  name: string,
  table: ZarrVectorsObjectTable,
  chainPaths: readonly string[],
  warnings: string[],
  signal?: AbortSignal,
): Promise<DecodedAttribute | undefined> {
  const path = `${levelPath}/object_attributes/${name}`;
  const reader = await openReader(access, path, signal);
  // Not an array: a slot reserved before its first write.
  if (reader === undefined) return undefined;
  const { array } = reader;
  const n = table.numObjects;
  if (!(array.shape[0] >= n)) {
    warnings.push(
      `object attribute ${name} has ${array.shape[0]} rows, expected ${n}`,
    );
    return undefined;
  }
  const attrs = array.attributes ?? {};
  if (array.stringEncoding !== undefined) {
    if (array.shape.length !== 1) {
      throw new Error(`a ${array.shape.length}-D string column`);
    }
    return {
      channels: [
        {
          name,
          own: true,
          description: name,
          column: {
            type: "string",
            values: await readStrings(reader, n, signal),
          },
        },
      ],
      tags: [],
    };
  }
  if (array.elementType === "vlen" || array.elementType === "utf32") {
    throw new Error("variable-length bytes, not values");
  }
  const type = array.elementType;
  const width = ELEMENT_BYTES[type];
  const channels = array.shape.slice(1).reduce((a, b) => a * b, 1);
  const bytes = await reader.readRows(0, n, signal);

  let missing: Uint8Array | undefined;
  if (attrs.fill_sentinel_meaning === "absent") {
    const fill = fillBytes(array);
    const rowBytes = width * channels;
    for (let r = 0; r < n; ++r) {
      let absent = true;
      for (let b = 0; b < rowBytes && absent; ++b) {
        if (bytes[r * rowBytes + b] !== fill[b % width]) absent = false;
      }
      if (absent) (missing ??= new Uint8Array(n))[r] = 1;
    }
  }
  if (attrs.has_present_mask === true) {
    const maskReader = await openReader(access, `${path}/present_mask`, signal);
    const maskArray = maskReader?.array;
    if (
      maskArray !== undefined &&
      maskArray.shape[0] >= n &&
      maskArray.elementBytes > 0 &&
      maskArray.elementType !== "utf32"
    ) {
      const mask = decodeNumbers(
        await maskReader!.readRows(0, n, signal),
        maskArray.elementType as ElementType,
        n,
      );
      for (let r = 0; r < n; ++r) {
        if (mask[r] === 0) (missing ??= new Uint8Array(n))[r] = 1;
      }
    }
  }

  const all = decodeNumbers(bytes, type, n * channels);
  const categories = attrs.categories;
  if (
    attrs.encoding === "dictionary" &&
    Array.isArray(categories) &&
    channels === 1 &&
    !type.startsWith("float")
  ) {
    // Codes become tags `name=category` and the category's text.
    const fillCode = Number(attrs._FillValue);
    const members = categories.map(() => [] as number[]);
    const text = new Array<string>(n);
    const labels = categories.map(categoryText);
    for (let r = 0; r < n; ++r) {
      const code = all[r];
      if (
        missing?.[r] ||
        code === fillCode ||
        !Number.isInteger(code) ||
        code < 0 ||
        code >= categories.length
      ) {
        text[r] = "";
        continue;
      }
      members[code].push(r);
      text[r] = labels[code];
    }
    return {
      channels: [
        {
          name,
          own: true,
          description: name,
          column: { type: "string", values: text },
        },
      ],
      tags: labels.map((label, i) => ({
        name: `${name}=${label}`,
        description: `object attribute ${name}`,
        rows: members[i],
      })),
    };
  }

  // The coarser levels of an additive chain hold the vertices of the
  // objects level 0 does not, and count them there.
  if (name === "vertex_count" && channels === 1 && chainPaths.length > 1) {
    await addChainCounts(access, chainPaths, table, all, missing, signal);
  }
  let big: BigInt64Array | BigUint64Array | undefined;
  if (type === "int64" || type === "uint64") {
    const ids = decodeUint64(bytes, type, n * channels);
    big = type === "int64" ? new BigInt64Array(ids.buffer) : ids;
  }
  const names: unknown = attrs.channel_names;
  const out: DecodedChannel[] = [];
  for (let c = 0; c < channels; ++c) {
    let values = all;
    let bigValues = big;
    if (channels > 1) {
      values = new Float64Array(n);
      for (let r = 0; r < n; ++r) values[r] = all[r * channels + c];
      if (big !== undefined) {
        bigValues =
          big instanceof BigInt64Array
            ? new BigInt64Array(n)
            : new BigUint64Array(n);
        for (let r = 0; r < n; ++r) bigValues[r] = big[r * channels + c];
      }
    }
    const { column, note } = channelColumn(type, values, bigValues, missing);
    const channelName =
      Array.isArray(names) && names.length === channels
        ? String(names[c])
        : String(c);
    const label = channels === 1 ? name : `${name}[${channelName}]`;
    out.push({
      name: channels === 1 ? name : `${name}_${channelName}`,
      own: channels === 1,
      description: note === undefined ? label : `${label}: ${note}`,
      column,
    });
  }
  return { channels: out, tags: [] };
}

/** Adds each coarser chain level's `vertex_count` to `counts`, by object id. */
async function addChainCounts(
  access: ZarrVectorsStoreAccess,
  chainPaths: readonly string[],
  table: ZarrVectorsObjectTable,
  counts: Float64Array,
  missing: Uint8Array | undefined,
  signal?: AbortSignal,
) {
  const n = table.numObjects;
  for (const levelPath of chainPaths.slice(1)) {
    const [levelTable, reader] = await Promise.all([
      readObjectTable(access, levelPath, signal),
      openReader(access, `${levelPath}/object_attributes/vertex_count`, signal),
    ]);
    if (levelTable === undefined || reader === undefined) continue;
    const m = levelTable.numObjects;
    if (reader.array.shape[0] < m || reader.array.shape.length !== 1) continue;
    const type = reader.array.elementType as ElementType;
    const bytes = await reader.readRows(0, m, signal);
    const values = decodeNumbers(bytes, type, m);
    const absent =
      reader.array.attributes?.fill_sentinel_meaning === "absent"
        ? decodeNumbers(fillBytes(reader.array), type, 1)[0]
        : undefined;
    const rowOf = rowLookup(levelTable);
    for (let r = 0; r < n; ++r) {
      const id = table.objectIds?.[r] ?? BigInt(r);
      const row = rowOf(id);
      if (row === undefined) continue;
      const v = values[row];
      if (Number.isNaN(v) || v === absent) continue;
      if (missing?.[r]) {
        missing[r] = 0;
        counts[r] = v;
      } else {
        counts[r] += v;
      }
    }
  }
}

// -------------------------------------------------------------- properties

/** A property id Neuroglancer's Seg tab can filter by (`name>3`). */
function propertyCandidate(name: string): string {
  let id = name.replace(/[^a-zA-Z0-9_]+/g, "_");
  if (!/^[a-zA-Z]/.test(id)) id = id.startsWith("_") ? `a${id}` : `a_${id}`;
  return id;
}

/** Rows in ascending segment id order, each id once (its first row). */
function rowsBySegmentId(
  segmentIds: BigUint64Array,
  warnings: string[],
): Uint32Array {
  const n = segmentIds.length;
  let ascending = true;
  for (let i = 1; i < n && ascending; ++i) {
    if (segmentIds[i - 1] >= segmentIds[i]) ascending = false;
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; ++i) order[i] = i;
  if (ascending) return order;
  order.sort((a, b) =>
    segmentIds[a] < segmentIds[b]
      ? -1
      : segmentIds[a] > segmentIds[b]
        ? 1
        : a - b,
  );
  let kept = 0;
  for (let i = 0; i < n; ++i) {
    if (kept > 0 && segmentIds[order[kept - 1]] === segmentIds[order[i]]) {
      continue;
    }
    order[kept++] = order[i];
  }
  if (kept < n) {
    warnings.push(
      `${n - kept} objects share a segment id with an earlier one; ` +
        "the properties shown are the first's",
    );
  }
  return order.slice(0, kept);
}

/**
 * Builds the segment-property map of a level from its object attributes and
 * groups.  Numeric columns become numeric properties (so the Seg tab can
 * filter `length>50`), text columns text properties; groups and the
 * categories of dictionary-encoded columns become tags (`#bundle_name`,
 * `#cell_type=L2IT`).  `chainPaths` is level 0's additive chain, level 0
 * first, over which `vertex_count` is summed.
 */
export async function readSegmentProperties(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  table: ZarrVectorsObjectTable,
  warnings: string[],
  signal?: AbortSignal,
  chainPaths: readonly string[] = [levelPath],
): Promise<InlineSegmentPropertyMap | undefined> {
  let names: string[] = [];
  try {
    names = (
      await access.listDirectories(`${levelPath}/object_attributes`, signal)
    ).sort();
  } catch (e) {
    warnings.push(
      "could not list object_attributes/ " +
        `(${e instanceof Error ? e.message : e}); segment properties are ` +
        "unavailable",
    );
  }
  names = names.filter((n) => n !== "segment_id");
  const { numObjects } = table;
  const decoded = new Array<DecodedAttribute | undefined>(names.length);
  await mapConcurrent(names, 8, async (name, index) => {
    try {
      decoded[index] = await decodeAttribute(
        access,
        levelPath,
        name,
        table,
        chainPaths,
        warnings,
        signal,
      );
    } catch (e) {
      signal?.throwIfAborted();
      warnings.push(
        `skipped object attribute ${name}: ${e instanceof Error ? e.message : e}`,
      );
    }
  });
  let groupTags: TagSource[] = [];
  try {
    groupTags = await readGroupTags(access, levelPath, table, warnings, signal);
  } catch (e) {
    signal?.throwIfAborted();
    warnings.push(
      `could not read groups: ${e instanceof Error ? e.message : e}`,
    );
  }
  const channels = decoded.flatMap((d) => d?.channels ?? []);
  const tags = encodeTags(
    [...groupTags, ...decoded.flatMap((d) => d?.tags ?? [])],
    numObjects,
    warnings,
  );
  if (channels.length === 0 && tags === undefined) return undefined;

  // Property ids: filterable, unique without regard to case, never one
  // Neuroglancer reserves; a name the store spells validly keeps it.
  const taken = new Set(["id", "label", TAGS_PROPERTY_ID]);
  const ids = new Array<string>(channels.length);
  channels.forEach((c, i) => {
    const key = c.name.toLowerCase();
    if (c.own && propertyCandidate(c.name) === c.name && !taken.has(key)) {
      taken.add(key);
      ids[i] = c.name;
    }
  });
  channels.forEach((c, i) => {
    if (ids[i] !== undefined) return;
    const base = propertyCandidate(c.name);
    let id = base;
    for (let k = 2; taken.has(id.toLowerCase()); ++k) id = `${base}_${k}`;
    taken.add(id.toLowerCase());
    ids[i] = id;
  });

  const rows = rowsBySegmentId(table.segmentIds, warnings);
  const count = rows.length;
  const segmentIds = new BigUint64Array(count);
  for (let i = 0; i < count; ++i) segmentIds[i] = table.segmentIds[rows[i]];
  const properties: InlineSegmentProperty[] = channels.map((c, i) => {
    const description = c.description === ids[i] ? undefined : c.description;
    if (c.column.type === "string") {
      const source = c.column.values;
      const values = new Array<string>(count);
      for (let k = 0; k < count; ++k) values[k] = source[rows[k]];
      return { id: ids[i], type: "string", description, values };
    }
    const source = c.column.values;
    const values = new (source.constructor as {
      new (n: number): TypedNumberArray<ArrayBuffer>;
    })(count);
    let min = Infinity;
    let max = -Infinity;
    for (let k = 0; k < count; ++k) {
      const v = source[rows[k]];
      values[k] = v;
      // Bounds of the finite values: one Inf must not hide the rest.
      if (Number.isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (min > max) min = max = 0;
    return {
      id: ids[i],
      type: "number",
      dataType: c.column.dataType,
      description,
      values,
      bounds: [min, max],
    };
  });
  if (tags !== undefined) {
    const values = new Array<string>(count);
    for (let k = 0; k < count; ++k) values[k] = tags.values[rows[k]];
    properties.push({
      id: TAGS_PROPERTY_ID,
      type: "tags",
      tags: tags.tags,
      tagDescriptions: tags.tagDescriptions,
      values,
    });
  }
  return { ids: segmentIds, properties };
}

// -------------------------------------------------------------- shaders

/**
 * Object values a shader reads per vertex: `obj_<column>` for a level-0
 * object attribute, `obj_group` for the index of an object's first group.
 */
export interface ShaderObjectValue {
  /** `object_attributes/<name>`, or `undefined` for the group index. */
  column: string | undefined;
  /** Category labels of a dictionary column (the shader sees the code). */
  categories?: string[];
}

/** Object values offered by default when a store has at most this many. */
const DEFAULT_SHADER_COLUMNS = 8;

/**
 * The object values to hand to shaders, within `budget` (shader inputs left
 * after the vertex attributes). `requested` names columns explicitly (from
 * `#attributes=obj:<name>`); otherwise every single-channel numeric or
 * dictionary-coded column of level 0, when there are at most a few. The
 * group index comes first, where the store has groups.
 */
export async function chooseShaderObjectValues(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  requested: readonly string[] | undefined,
  budget: number,
  warnings: string[],
  signal?: AbortSignal,
): Promise<ShaderObjectValue[]> {
  const out: ShaderObjectValue[] = [];
  const groups = await openReader(access, `${levelPath}/groups`, signal);
  if (groups !== undefined) out.push({ column: undefined });
  let names: string[];
  if (requested !== undefined) {
    names = [...requested];
  } else {
    try {
      names = (
        await access.listDirectories(`${levelPath}/object_attributes`, signal)
      ).sort();
    } catch {
      names = [];
    }
  }
  const eligible: ShaderObjectValue[] = [];
  const refused: string[] = [];
  await mapConcurrent(names, 8, async (name) => {
    if (name === "segment_id") return;
    const reader = await openReader(
      access,
      `${levelPath}/object_attributes/${name}`,
      signal,
    ).catch(() => undefined);
    const array = reader?.array;
    const type = array?.elementType;
    const ok =
      array !== undefined &&
      array.stringEncoding === undefined &&
      type !== "vlen" &&
      type !== "utf32" &&
      array.shape.length === 1 &&
      // 64-bit integers are ids, not quantities, unless asked for.
      (requested !== undefined || (type !== "int64" && type !== "uint64"));
    if (!ok) {
      if (requested !== undefined) refused.push(name);
      return;
    }
    if (type === "int64" || type === "uint64") {
      warnings.push(
        `object attribute ${name} is ${type}, read in shaders as float32: ` +
          "values beyond 16,777,216 (such as segment ids) lose precision",
      );
    }
    const categories = array.attributes?.categories;
    eligible.push({
      column: name,
      categories:
        array.attributes?.encoding === "dictionary" && Array.isArray(categories)
          ? categories.map(categoryText)
          : undefined,
    });
  });
  if (refused.length > 0) {
    warnings.push(
      `object attributes ${refused.join(", ")} cannot reach shaders ` +
        "(not one number per object)",
    );
  }
  eligible.sort((a, b) => (a.column! < b.column! ? -1 : 1));
  let chosen = eligible;
  if (requested === undefined && eligible.length > DEFAULT_SHADER_COLUMNS) {
    warnings.push(
      `${eligible.length} object attributes could reach shaders as obj_<name>; ` +
        "none do by default. Name them with #attributes=obj:a,obj:b",
    );
    chosen = [];
  }
  const room = Math.max(0, budget - out.length);
  if (chosen.length > room) {
    warnings.push(
      `only ${room} object attribute(s) fit in the shader alongside the ` +
        `vertex attributes; left out: ${chosen
          .slice(room)
          .map((c) => c.column)
          .join(", ")}`,
    );
    chosen = chosen.slice(0, room);
  }
  if (out.length > budget) out.length = 0;
  return [...out, ...chosen];
}

/** Per-object values for shaders, and the row of an object id. */
export interface ShaderObjectValues {
  rowOf(id: bigint): number | undefined;
  /** One array per requested value, one float per row; NaN where unknown. */
  values: Float32Array[];
}

const shaderValueCache = new WeakMap<
  ZarrVectorsStoreAccess,
  Map<string, Promise<ShaderObjectValues | undefined>>
>();

/**
 * Reads the object values `wanted` for every object of `levelPath` (level
 * 0): numbers as float32, dictionary codes as their code, missing values
 * (the column's absent sentinel) as NaN, and each object's first group
 * index. Cached per store.
 */
export function readShaderObjectValues(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  wanted: readonly ShaderObjectValue[],
): Promise<ShaderObjectValues | undefined> {
  let cache = shaderValueCache.get(access);
  if (cache === undefined) shaderValueCache.set(access, (cache = new Map()));
  const key = `${levelPath}|${wanted.map((w) => w.column ?? "#group").join(",")}`;
  let promise = cache.get(key);
  if (promise === undefined) {
    promise = loadShaderObjectValues(access, levelPath, wanted);
    cache.set(key, promise);
    promise.catch(() => cache!.delete(key));
  }
  return promise;
}

async function loadShaderObjectValues(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  wanted: readonly ShaderObjectValue[],
): Promise<ShaderObjectValues | undefined> {
  const table = await readObjectTable(access, levelPath);
  if (table === undefined) return undefined;
  const n = table.numObjects;
  const index = new SegmentIdIndex(table.segmentIds);
  const values = await Promise.all(
    wanted.map(async ({ column }) => {
      const out = new Float32Array(n).fill(NaN);
      try {
        if (column === undefined) {
          const tags = await readGroupTags(access, levelPath, table, []);
          // Later groups first, so each row ends with its first group.
          for (let g = tags.length - 1; g >= 0; --g) {
            const { rows } = tags[g];
            for (let i = 0; i < rows.length; ++i) out[rows[i]] = g;
          }
          return out;
        }
        const reader = await openReader(
          access,
          `${levelPath}/object_attributes/${column}`,
        );
        if (reader === undefined) return out;
        const { array } = reader;
        const rows = Math.min(n, array.shape[0]);
        const type = array.elementType as ElementType;
        const bytes = await reader.readRows(0, rows);
        const numbers = decodeNumbers(bytes, type, rows);
        const attrs = array.attributes ?? {};
        const absent =
          attrs.fill_sentinel_meaning === "absent"
            ? decodeNumbers(fillBytes(array), type, 1)[0]
            : undefined;
        const fillCode =
          attrs.encoding === "dictionary" ? Number(attrs._FillValue) : NaN;
        for (let r = 0; r < rows; ++r) {
          const v = numbers[r];
          out[r] = v === absent || v === fillCode ? NaN : v;
        }
      } catch (e) {
        warnOnce(
          `object values for shaders: ${column ?? "groups"} unreadable ` +
            `(${e instanceof Error ? e.message : e})`,
        );
      }
      return out;
    }),
  );
  return { rowOf: (id) => index.rowOf(id), values };
}
