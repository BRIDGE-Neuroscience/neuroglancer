/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import { describe, expect, it } from "vitest";
import {
  decodeAttributeExactInts,
  decodeAttributeToFloat32,
  isExactIntDtype,
  zeroAttribute,
} from "#src/datasource/zarr-vectors/vertex_attribute_float.js";

/** Bytes of `values` in `dtype`, optionally shifted off alignment. */
function blob(
  values: number[] | bigint[],
  ctor: any,
  leadingPad = 0,
): Uint8Array {
  const arr = ctor.from(values as any);
  const out = new Uint8Array(leadingPad + arr.byteLength);
  out.set(
    new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
    leadingPad,
  );
  return out.subarray(leadingPad);
}

describe("decodeAttributeToFloat32", () => {
  it("views float32 bytes without copying", () => {
    const bytes = blob([1.5, -2.25], Float32Array);
    const out = decodeAttributeToFloat32(bytes, "float32", 2);
    expect(Array.from(out)).toEqual([1.5, -2.25]);
    expect(out.buffer).toBe(bytes.buffer);
  });

  it("copies when the blob is not aligned for its element size", () => {
    // A vlen-bytes payload starts wherever the container put it; a typed-array
    // view over an odd offset is a TypeError, so the decoder must copy.
    const bytes = blob([1.5, -2.25], Float32Array, 2);
    expect(bytes.byteOffset % 4).not.toBe(0);
    const out = decodeAttributeToFloat32(bytes, "float32", 2);
    expect(Array.from(out)).toEqual([1.5, -2.25]);
    expect(out.buffer).not.toBe(bytes.buffer);
  });

  it.each([
    ["uint8", Uint8Array, [0, 255]],
    ["int8", Int8Array, [-128, 127]],
    ["uint16", Uint16Array, [0, 65535]],
    ["int16", Int16Array, [-32768, 32767]],
    ["uint32", Uint32Array, [0, 4000000]],
    ["int32", Int32Array, [-2000000, 2000000]],
  ] as const)("converts %s exactly", (dtype, ctor, values) => {
    const out = decodeAttributeToFloat32(
      blob(values as any, ctor),
      dtype,
      values.length,
    );
    expect(Array.from(out)).toEqual(values);
  });

  it("downcasts float64 columns", () => {
    // The MERFISH obs block stores confidence scores and coordinates as
    // float64; before the downcast they were skipped entirely.
    const out = decodeAttributeToFloat32(
      blob([0.5, 0.875, -3.25], Float64Array),
      "float64",
      3,
    );
    expect(Array.from(out)).toEqual([0.5, 0.875, -3.25]);
  });

  it("downcasts int64 and uint64 columns", () => {
    expect(
      Array.from(
        decodeAttributeToFloat32(
          blob([1n, 670n, -4n], BigInt64Array),
          "int64",
          3,
        ),
      ),
    ).toEqual([1, 670, -4]);
    expect(
      Array.from(
        decodeAttributeToFloat32(
          blob([0n, 5123n], BigUint64Array),
          "uint64",
          2,
        ),
      ),
    ).toEqual([0, 5123]);
  });

  it("loses exactness above 2^24, which is why ids are excluded", () => {
    const id = 864691135000000000n;
    const [decoded] = decodeAttributeToFloat32(
      blob([id], BigInt64Array),
      "int64",
      1,
    );
    expect(decoded).not.toBe(Number(id));
    expect(isExactIntDtype("int64")).toBe(false);
  });

  it("rejects a blob whose length does not match the element count", () => {
    expect(() =>
      decodeAttributeToFloat32(blob([1, 2], Int32Array), "int32", 3),
    ).toThrow(/expected 12 bytes/);
  });
});

describe("decodeAttributeExactInts", () => {
  it("returns native values for exact integer dtypes", () => {
    const out = decodeAttributeExactInts(
      blob([70000, 1], Uint32Array),
      "uint32",
      2,
    );
    expect(Array.from(out!)).toEqual([70000, 1]);
  });

  it("refuses dtypes that cannot carry an exact 32-bit id", () => {
    for (const dtype of ["float32", "float64", "int64", "uint64"] as const) {
      expect(
        decodeAttributeExactInts(blob([1n], BigInt64Array), dtype, 1),
      ).toBeUndefined();
    }
  });
});

describe("zeroAttribute", () => {
  it("stands in for a chunk that lacks the column", () => {
    expect(Array.from(zeroAttribute(3))).toEqual([0, 0, 0]);
  });
});
