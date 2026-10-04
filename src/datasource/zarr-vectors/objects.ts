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
} from "#src/datasource/zarr-vectors/dtype.js";
import type { ZarrVectorsStoreAccess } from "#src/datasource/zarr-vectors/store.js";
import {
  checkObjectIndexLayout,
  readJson,
} from "#src/datasource/zarr-vectors/store.js";
import { mapConcurrent, warnOnce } from "#src/datasource/zarr-vectors/util.js";
import {
  parseZarrArrayMetadata,
  ZarrArrayReader,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import type {
  InlineSegmentNumericalProperty,
  InlineSegmentProperty,
  InlineSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { normalizeInlineSegmentPropertyMap } from "#src/segmentation_display_state/property_map.js";
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

async function openReader(
  access: ZarrVectorsStoreAccess,
  path: string,
  signal?: AbortSignal,
): Promise<ZarrArrayReader | undefined> {
  const json = await readJson(access.read, `${path}/zarr.json`, signal);
  if (json === undefined || json.node_type !== "array") return undefined;
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
    openReader(access, `${levelPath}/object_index/object_ids`, signal),
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

/** Group membership per row, and group names. */
export interface ZarrVectorsGroups {
  names: string[];
  /** First group of each row (`-1` for none). */
  groupByRow: Int32Array;
  /** Every group of the rows that belong to more than one. */
  moreGroups: Map<number, number[]>;
}

/** The groups of `row`, ascending. */
export function groupsOf(groups: ZarrVectorsGroups, row: number): number[] {
  const first = groups.groupByRow[row];
  if (first < 0) return [];
  return groups.moreGroups.get(row) ?? [first];
}

function groupNames(attrs: any, count: number): string[] {
  const raw = attrs?.group_names;
  const names: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < count; ++i) {
    const candidate = Array.isArray(raw) ? raw[i] : raw?.[String(i)];
    let name =
      typeof candidate === "string"
        ? candidate.trim().replace(/\s+/g, "_")
        : "";
    if (name === "") name = `group_${i}`;
    if (seen.has(name)) name = `${name}_${i}`;
    seen.add(name);
    names.push(name);
  }
  return names;
}

/** Reads a level's `groups/` array into a per-row group id. */
export async function readGroups(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  table: ZarrVectorsObjectTable,
  signal?: AbortSignal,
): Promise<ZarrVectorsGroups | undefined> {
  const reader = await openReader(access, `${levelPath}/groups`, signal);
  if (reader === undefined) return undefined;
  const attrs = reader.array.attributes;
  const numGroups = Number(attrs.num_groups ?? reader.array.shape[0]);
  if (!(numGroups > 0)) return undefined;
  const names = groupNames(attrs, numGroups);
  const blobs = await reader.readVlenRows(0, numGroups, signal);
  const groupByRow = new Int32Array(table.numObjects).fill(-1);
  const rowIndex =
    table.objectIds === undefined
      ? undefined
      : new SegmentIdIndex(table.objectIds);
  const moreGroups = new Map<number, number[]>();
  const assign = (objectId: bigint, gid: number) => {
    const row =
      rowIndex === undefined
        ? objectId < BigInt(table.numObjects)
          ? Number(objectId)
          : undefined
        : rowIndex.rowOf(objectId);
    if (row === undefined) return;
    const first = groupByRow[row];
    if (first === -1) {
      groupByRow[row] = gid;
    } else if (first !== gid) {
      // Groups may overlap (a cell in several classes); keep them all.
      let list = moreGroups.get(row);
      if (list === undefined) moreGroups.set(row, (list = [first]));
      if (!list.includes(gid)) list.push(gid);
    }
  };
  const ranges = attrs.group_ranges;
  for (let gid = 0; gid < numGroups; ++gid) {
    const range = ranges?.[String(gid)];
    if (Array.isArray(range) && range.length === 2) {
      for (let id = Number(range[0]); id < Number(range[1]); ++id) {
        assign(BigInt(id), gid);
      }
      continue;
    }
    const blob = blobs[gid];
    if (blob === undefined) continue;
    const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
    for (let i = 0; i + 8 <= blob.byteLength; i += 8) {
      assign(view.getBigUint64(i, true), gid);
    }
  }
  for (const list of moreGroups.values()) list.sort((a, b) => a - b);
  return { names, groupByRow, moreGroups };
}

/**
 * Builds the segment-property map of a level from its object attributes and
 * groups: numeric columns become numeric properties (so the Seg tab can
 * filter `length>50`), groups become tags plus a label (`#bundle_name`).
 */
export async function readSegmentProperties(
  access: ZarrVectorsStoreAccess,
  levelPath: string,
  table: ZarrVectorsObjectTable,
  warnings: string[],
  signal?: AbortSignal,
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
  const properties: InlineSegmentProperty[] = [];
  const columns: (InlineSegmentNumericalProperty[] | undefined)[] = new Array(
    names.length,
  );
  let presentMask: Uint8Array | undefined;
  await mapConcurrent(names, 8, async (name, index) => {
    const path = `${levelPath}/object_attributes/${name}`;
    let reader: ZarrArrayReader | undefined;
    try {
      reader = await openReader(access, path, signal);
    } catch (e) {
      warnings.push(
        `skipped object attribute ${name}: ${e instanceof Error ? e.message : e}`,
      );
      return;
    }
    if (reader === undefined) return;
    const { array } = reader;
    if (array.elementType === "vlen") return;
    if (array.shape[0] !== numObjects) {
      warnings.push(
        `object attribute ${name} has ${array.shape[0]} rows, expected ${numObjects}`,
      );
      return;
    }
    const channels = array.shape.slice(1).reduce((a, b) => a * b, 1);
    if (channels > 4) return;
    const bytes = await reader.readAllRows(signal);
    const values = decodeFloat32(
      bytes,
      array.elementType as ElementType,
      numObjects * channels,
    );
    if (array.attributes?.has_present_mask === true) {
      const maskReader = await openReader(
        access,
        `${path}/present_mask`,
        signal,
      );
      if (maskReader !== undefined) {
        const mask = await maskReader.readAllRows(signal);
        if (presentMask === undefined) presentMask = new Uint8Array(mask);
        else for (let i = 0; i < numObjects; ++i) presentMask[i] &= mask[i];
      }
    }
    const out: InlineSegmentNumericalProperty[] = [];
    for (let c = 0; c < channels; ++c) {
      const column = channels === 1 ? values : new Float32Array(numObjects);
      if (channels !== 1) {
        for (let i = 0; i < numObjects; ++i)
          column[i] = values[i * channels + c];
      }
      out.push({
        id: channels === 1 ? name : `${name}_${c}`,
        type: "number",
        dataType: DataType.FLOAT32,
        description: undefined,
        values: column as Float32Array<ArrayBuffer>,
        bounds: [0, 0],
      });
    }
    columns[index] = out;
  });
  for (const c of columns) if (c !== undefined) properties.push(...c);

  const groups = await readGroups(access, levelPath, table, signal).catch(
    (e) => {
      warnings.push(
        `could not read groups: ${e instanceof Error ? e.message : e}`,
      );
      return undefined;
    },
  );
  if (properties.length === 0 && groups === undefined) return undefined;

  const keep: number[] = [];
  for (let i = 0; i < numObjects; ++i) {
    if (presentMask === undefined || presentMask[i]) keep.push(i);
  }
  const ids = new BigUint64Array(keep.length);
  for (let i = 0; i < keep.length; ++i) ids[i] = table.segmentIds[keep[i]];
  const compacted: InlineSegmentProperty[] = properties.map((p) => {
    const numeric = p as InlineSegmentNumericalProperty;
    const values = new Float32Array(keep.length);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < keep.length; ++i) {
      const v = (numeric.values as Float32Array)[keep[i]];
      values[i] = v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) min = max = 0;
    return { ...numeric, values, bounds: [min, max] };
  });
  if (groups !== undefined) {
    const tagValues = new Array<string>(keep.length);
    const labels = new Array<string>(keep.length);
    for (let i = 0; i < keep.length; ++i) {
      const gids = groupsOf(groups, keep[i]);
      // A tag set is a string of ascending, distinct character codes.
      tagValues[i] = String.fromCharCode(...gids);
      labels[i] = gids.map((g) => groups.names[g]).join(", ");
    }
    compacted.push({
      id: "group",
      type: "tags",
      tags: [...groups.names],
      tagDescriptions: groups.names.map(() => ""),
      values: tagValues,
    });
    compacted.push({
      id: "label",
      type: "label",
      description: undefined,
      values: labels,
    });
  }
  return normalizeInlineSegmentPropertyMap({ ids, properties: compacted });
}
