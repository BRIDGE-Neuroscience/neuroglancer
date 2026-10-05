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
 * Segment properties from object attributes and groups, read from
 * `props_types` (see generate.py's `props_fixture`) and from fixtures with
 * arrays laid over them, and queried through Neuroglancer's own Seg-tab
 * query code.
 */

import { describe, expect, it } from "vitest";
import {
  float16ToNumber,
  numberToFloat16,
} from "#src/datasource/zarr-vectors/dtype.js";
import {
  MAX_TAGS,
  readObjectTable,
  readSegmentProperties,
} from "#src/datasource/zarr-vectors/objects.js";
import type { ZarrVectorsStoreAccess } from "#src/datasource/zarr-vectors/store.js";
import {
  fixtureListDirectories,
  fixtureRead,
} from "#src/datasource/zarr-vectors/test_fixtures.js";
import {
  fillBytes,
  parseZarrArrayMetadata,
  parseZarrJsonText,
  ShardIndexCache,
} from "#src/datasource/zarr-vectors/zarr_array.js";
import type {
  InlineSegmentNumericalProperty,
  InlineSegmentPropertyMap,
  InlineSegmentTagsProperty,
} from "#src/segmentation_display_state/property_map.js";
import {
  executeSegmentQuery,
  parseSegmentQuery,
  PreprocessedSegmentPropertyMap,
  SegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { DataType } from "#src/util/data_type.js";

/** Fixture access, with some files replaced or added. */
function access(
  store: string,
  overlay: Record<string, Uint8Array | string> = {},
): ZarrVectorsStoreAccess {
  const read = fixtureRead(store);
  const list = fixtureListDirectories(store);
  return {
    read: async (path, options) => {
      const replaced = overlay[path];
      if (replaced === undefined) return read(path, options);
      return typeof replaced === "string"
        ? new TextEncoder().encode(replaced)
        : replaced;
    },
    listDirectories: async (path) => {
      const names = new Set(await list(path));
      for (const key of Object.keys(overlay)) {
        if (!key.startsWith(`${path}/`)) continue;
        const child = key.slice(path.length + 1).split("/");
        if (child.length > 1) names.add(child[0]);
      }
      return [...names];
    },
    shardIndexes: new ShardIndexCache(),
  };
}

/** A raw one-chunk uint32 column, as files for an overlay. */
function uint32Column(
  path: string,
  values: number[],
): Record<string, Uint8Array | string> {
  return {
    [`${path}/zarr.json`]: JSON.stringify({
      zarr_format: 3,
      node_type: "array",
      shape: [values.length],
      data_type: "uint32",
      chunk_grid: {
        name: "regular",
        configuration: { chunk_shape: [values.length] },
      },
      chunk_key_encoding: { name: "default" },
      codecs: [{ name: "bytes", configuration: { endian: "little" } }],
      fill_value: 4294967295,
      attributes: { fill_sentinel_meaning: "absent" },
    }),
    [`${path}/c/0`]: new Uint8Array(Uint32Array.from(values).buffer),
  };
}

async function properties(
  store: ZarrVectorsStoreAccess,
  chain?: string[],
): Promise<{ map: InlineSegmentPropertyMap; warnings: string[] }> {
  const table = (await readObjectTable(store, "0"))!;
  const warnings: string[] = [];
  const map = (await readSegmentProperties(
    store,
    "0",
    table,
    warnings,
    undefined,
    chain,
  ))!;
  return { map, warnings };
}

function property(map: InlineSegmentPropertyMap, id: string): any {
  const found = map.properties.find((p) => p.id === id);
  expect(found, id).toBeDefined();
  return found;
}

function query(map: InlineSegmentPropertyMap, text: string) {
  const db = new PreprocessedSegmentPropertyMap(
    new SegmentPropertyMap({ inlineProperties: map }),
  );
  const result = executeSegmentQuery(db, parseSegmentQuery(db, text));
  expect(result.errors).toBeUndefined();
  return Array.from(result.indices!, (i) => Number(map.ids[i])).sort(
    (a, b) => a - b,
  );
}

describe("object attributes as segment properties", async () => {
  const { map, warnings } = await properties(access("props_types"));

  it("reads every column without warnings", () => {
    expect(warnings).toEqual([]);
    expect(Array.from(map.ids, Number)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it("shows missing values as missing", () => {
    const count = property(map, "count") as InlineSegmentNumericalProperty;
    expect(count.dataType).toBe(DataType.FLOAT32);
    expect(Array.from(count.values)).toEqual([5, 0, NaN, -3, 100, 7]);
    expect(count.bounds).toEqual([-3, 100]);
    expect(query(map, "count<1")).toEqual([1, 3]);
    // Rows 4-5 of `half` are a chunk never written: its NaN fill.
    const half = property(map, "half");
    expect(Array.from(half.values)).toEqual([
      1.5,
      -2.25,
      65504,
      float16ToNumber(numberToFloat16(0.1)),
      NaN,
      NaN,
    ]);
  });

  it("keeps 64-bit values exact", () => {
    expect(property(map, "root_id")).toMatchObject({
      type: "string",
      values: [
        "864691134884807418",
        "864691134886037498",
        "",
        "864691135102580256",
        "864691137200012353",
        "864691134886499066",
      ],
    });
    // The missing chunk reads as the exact uint64 sentinel, not 0.
    expect(property(map, "big_u64").values).toEqual([
      "9007199254740993",
      "9223372036854775813",
      "7",
      "0",
      "",
      "",
    ]);
    expect(property(map, "wide").values).toEqual([
      "16777217",
      "1",
      "",
      "3",
      "4",
      "5",
    ]);
    const small = property(map, "small_i64");
    expect(small.dataType).toBe(DataType.INT32);
    expect(Array.from(small.values)).toEqual([-5, 0, 2 ** 31 - 1, 3, 4, 5]);
    expect(property(map, "bad_name").dataType).toBe(DataType.UINT8);
  });

  it("bounds ignore Inf", () => {
    const inf = property(map, "with_inf");
    expect(inf.bounds).toEqual([1, 3]);
    // Inf lies past the bounds a filter is clamped to.
    expect(query(map, "with_inf>2")).toEqual([2]);
  });

  it("reads text columns", () => {
    expect(property(map, "name").values).toEqual([
      "alpha",
      "beta",
      "",
      "gamma",
      "δέλτα",
      "x",
    ]);
    expect(property(map, "note").values).toEqual([
      "n0",
      "",
      "n2",
      "n3",
      "n4",
      "n5",
    ]);
  });

  it("splits every channel of a vector column", () => {
    for (let c = 0; c < 5; ++c) {
      expect(Array.from(property(map, `vec_${c}`).values)).toEqual(
        [0, 1, 2, 3, 4, 5].map((r) => 5 * r + c),
      );
    }
  });

  it("gives every property a distinct, filterable id", () => {
    const ids = map.properties.map((p) => p.id);
    expect(new Set(ids.map((id) => id.toLowerCase())).size).toBe(ids.length);
    for (const id of ids) expect(id).toMatch(/^[a-zA-Z][a-zA-Z0-9_]*$/);
    expect(property(map, "Length").values[0]).toBe(1);
    expect(property(map, "length_2")).toMatchObject({ description: "length" });
    expect(property(map, "label_2")).toMatchObject({ description: "label" });
    expect(property(map, "bad_name")).toMatchObject({
      description: "bad-name",
    });
    expect(query(map, "label_2>3")).toEqual([4, 5]);
  });

  it("turns categories and groups into one set of tags", () => {
    const tags = property(map, "group") as InlineSegmentTagsProperty;
    expect(tags.tags).toEqual([
      "cell_type=L2IT",
      "AC",
      "ac_2",
      "cell_type=PV",
      "cell_type=L5ET",
      "proofread=False",
      "proofread=True",
    ]);
    expect(tags.tagDescriptions[0]).toBe(
      "group 0, 2 objects, from cell_type; object attribute cell_type",
    );
    expect(tags.tagDescriptions[2]).toBe(
      "ac: group 2, 997 objects, from bundle",
    );
    // The group and the category of the same name are one tag.
    expect(query(map, "#cell_type=L2IT")).toEqual([0, 1, 4]);
    expect(query(map, "#ac")).toEqual([2]);
    // A range group stops at the last object.
    expect(query(map, "#ac_2")).toEqual([3, 4, 5]);
    expect(query(map, "#proofread=True")).toEqual([0, 2, 3]);
    expect(property(map, "cell_type").values).toEqual([
      "L2IT",
      "PV",
      "",
      "L5ET",
      "L2IT",
      "PV",
    ]);
    expect(map.properties.some((p) => p.type === "label")).toBe(false);
  });

  it("clamps range groups in a store without object ids", async () => {
    const base = access("props_types");
    const json = JSON.parse(
      new TextDecoder().decode(
        (await base.read("0/object_index/zarr.json", {}))!,
      ),
    );
    json.attributes.layout = "vlen_manifests_v1";
    const v1 = access("props_types", {
      "0/object_index/zarr.json": JSON.stringify(json),
    });
    const { map } = await properties(v1);
    expect(query(map, "#ac_2")).toEqual([3, 4, 5]);
  });
});

describe("groups as tags", () => {
  async function withGroups(count: number, nonEmpty: (gid: number) => boolean) {
    const ranges: Record<string, number[]> = {};
    for (let gid = 0; gid < count; ++gid) {
      ranges[gid] = nonEmpty(gid) ? [gid % 6, (gid % 6) + 1] : [0, 0];
    }
    const base = access("props_types");
    const json = JSON.parse(
      new TextDecoder().decode((await base.read("0/groups/zarr.json", {}))!),
    );
    json.attributes.num_groups = count;
    json.attributes.group_ranges = ranges;
    return properties(
      access("props_types", { "0/groups/zarr.json": JSON.stringify(json) }),
    );
  }

  it("codes only groups that hold objects", async () => {
    const { map, warnings } = await withGroups(70_000, (gid) => gid >= 69_990);
    expect(warnings).toEqual([]);
    const tags = property(map, "group") as InlineSegmentTagsProperty;
    // 10 groups, plus the categories of cell_type and proofread.
    expect(tags.tags.length).toBe(15);
    // Group 0 holds no object, but shares its name with a category that does.
    expect(tags.tags.slice(0, 2)).toEqual(["cell_type=L2IT", "group_69990"]);
    expect(query(map, "#group_69995")).toEqual([69_995 % 6]);
  });

  it("drops tags it cannot encode", async () => {
    const { map, warnings } = await withGroups(MAX_TAGS + 1, () => true);
    expect(warnings.join()).toMatch(/at most 65535/);
    expect(map.properties.some((p) => p.type === "tags")).toBe(false);
  });
});

describe("segment ids", () => {
  it("sorts ids and keeps the first of duplicates", async () => {
    const store = access("props_types");
    const table = (await readObjectTable(store, "0"))!;
    table.segmentIds = BigUint64Array.from([9n, 3n, 7n, 3n, 1n, 12n]);
    const warnings: string[] = [];
    const map = (await readSegmentProperties(store, "0", table, warnings))!;
    expect(Array.from(map.ids, Number)).toEqual([1, 3, 7, 9, 12]);
    // Row 1 (not row 3) is segment 3.
    expect(Array.from(property(map, "count").values)).toEqual([
      100,
      0,
      NaN,
      5,
      7,
    ]);
    expect(warnings.join()).toMatch(/1 objects share a segment id/);
  });
});

describe("additive pyramids", () => {
  it("sums vertex_count over level 0's chain", async () => {
    const own = [
      Array.from({ length: 24 }, (_, i) => (i < 12 ? 16 + 2 * i : 0)),
      Array.from({ length: 24 }, (_, i) =>
        i >= 12 && i < 18 ? 16 + 2 * i : 0,
      ),
      Array.from({ length: 24 }, (_, i) => (i >= 18 ? 16 + 2 * i : 0)),
    ];
    const store = access("add_additive", {
      ...uint32Column("0/object_attributes/vertex_count", own[0]),
      ...uint32Column("1/object_attributes/vertex_count", own[1]),
      ...uint32Column("2/object_attributes/vertex_count", own[2]),
    });
    const levelOnly = await properties(store);
    expect(query(levelOnly.map, "vertex_count>0").length).toBe(12);
    const { map } = await properties(store, ["0", "1", "2"]);
    expect(Array.from(property(map, "vertex_count").values)).toEqual(
      Array.from({ length: 24 }, (_, i) => 16 + 2 * i),
    );
  });
});

describe("fill values", () => {
  const array = (dataType: unknown, fill: unknown, endian = "little") =>
    parseZarrArrayMetadata("a", {
      zarr_format: 3,
      node_type: "array",
      shape: [1],
      data_type: dataType,
      chunk_grid: { name: "regular", configuration: { chunk_shape: [1] } },
      codecs: [{ name: "bytes", configuration: { endian } }],
      fill_value: fill,
    });
  const bits = (dataType: unknown, fill: unknown, endian?: string) =>
    Array.from(fillBytes(array(dataType, fill, endian)));

  it("keeps 64-bit fills exact", () => {
    const json = parseZarrJsonText(
      '{"fill_value": 18446744073709551615, "x": 1}',
    );
    expect(json.fill_value).toBe("18446744073709551615");
    expect(bits("uint64", json.fill_value)).toEqual(new Array(8).fill(255));
    // JSON.parse rounds 18446744073709551615 to 2^64: still the maximum.
    expect(bits("uint64", 2 ** 64)).toEqual(new Array(8).fill(255));
    expect(bits("int64", -9223372036854775808)).toEqual([
      0, 0, 0, 0, 0, 0, 0, 128,
    ]);
  });

  it("encodes float16 and raw-bit fills", () => {
    expect(bits("float16", "NaN")).toEqual([0x00, 0x7e]);
    expect(bits("float16", 1.5)).toEqual([0x00, 0x3e]);
    expect(bits("float16", "-Infinity")).toEqual([0x00, 0xfc]);
    expect(bits("float32", "0x7fc00001")).toEqual([0x01, 0x00, 0xc0, 0x7f]);
    expect(bits("int32", -2, "big")).toEqual([0xff, 0xff, 0xff, 0xfe]);
    expect(
      bits(
        { name: "fixed_length_utf32", configuration: { length_bytes: 8 } },
        "é",
      ),
    ).toEqual([0xe9, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("rounds float16 to nearest even", () => {
    for (const x of [0, 1, -2.5, 65504, 6e-8, 2 ** -14, 0.1, 1 / 3, 1000.7]) {
      const h = numberToFloat16(x);
      const back = float16ToNumber(h);
      // No other half is closer.
      for (const d of [-1, 1]) {
        const other = float16ToNumber((h + d) & 0xffff);
        if (Number.isNaN(other)) continue;
        expect(Math.abs(other - x)).toBeGreaterThanOrEqual(Math.abs(back - x));
      }
    }
    expect(numberToFloat16(65520)).toBe(0x7c00);
    expect(numberToFloat16(-0)).toBe(0x8000);
  });
});
