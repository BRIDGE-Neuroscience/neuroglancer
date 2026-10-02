/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Decoding of per-vertex attribute blobs into the single representation the
 * render layer uses for all of them: `float32`.
 *
 * One representation is what makes an unbounded number of attributes possible.
 * Attributes used to reach the GPU in their on-disk dtype, one texture (and so
 * one texture unit) each, which capped a layer at ~10 of them and made
 * `prop_<name>()` return an `int32_t` struct for integer columns -- a shader
 * written against a float column would not compile against an integer one.
 * Decoding everything to float32 up front collapses both problems: attributes
 * become one contiguous float block that rides a single texture, and every
 * `prop_<name>()` is a `float`.
 *
 * It also un-drops the 64-bit columns. `float64` scores and `int64` category
 * codes had no 32-bit spelling, so they were skipped silently at load; a
 * MERFISH store was left with nothing but whichever genes sorted first. They
 * are downcast here, exactly as per-OBJECT attributes already are in
 * `object_attribute_bytes.ts`.
 *
 * Lives apart from the chunk downloader so the byte arithmetic -- which fails
 * a whole store when wrong -- is unit-testable without a kvstore.
 */

/** Supported on-disk dtype for a per-vertex attribute. */
export type VertexAttributeDtype =
  | "float32"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float64"
  | "int64"
  | "uint64"
  | "float16"
  | "bool";

/** Bytes per element on disk, per dtype. */
export const ATTRIBUTE_ELEMENT_BYTES: Record<VertexAttributeDtype, number> = {
  float32: 4,
  uint8: 1,
  uint16: 2,
  uint32: 4,
  int8: 1,
  int16: 2,
  int32: 4,
  float64: 8,
  int64: 8,
  uint64: 8,
  float16: 2,
  bool: 1,
};

/**
 * Exact integer dtypes: those whose every value survives a round trip through
 * a JS number, so a column of them can serve as a vertex id. `float32` cannot
 * (it is not an integer type) and the 64-bit dtypes cannot (they exceed 2^53,
 * and the float32 form this module produces loses exactness past 2^24).
 */
const EXACT_INT_CTORS = {
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
} as const;

export type ExactIntDtype = keyof typeof EXACT_INT_CTORS;

/** Whether `dtype` names one of the exact integer dtypes. */
export function isExactIntDtype(
  dtype: VertexAttributeDtype,
): dtype is ExactIntDtype {
  return dtype in EXACT_INT_CTORS;
}

/**
 * Return a copy of `bytes` when its offset is not aligned for `elementSize`,
 * else the original. A typed-array view demands its element alignment, and a
 * vlen-bytes payload starts wherever the container put it.
 */
function alignedFor(
  bytes: Uint8Array,
  elementSize: number,
): { buffer: ArrayBuffer; byteOffset: number } {
  if (bytes.byteOffset % elementSize === 0) {
    return {
      buffer: bytes.buffer as ArrayBuffer,
      byteOffset: bytes.byteOffset,
    };
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return { buffer: copy.buffer, byteOffset: 0 };
}

function checkLength(
  bytes: Uint8Array,
  dtype: VertexAttributeDtype,
  expectedElements: number,
): number {
  const elementSize = ATTRIBUTE_ELEMENT_BYTES[dtype];
  const expectedBytes = expectedElements * elementSize;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(
      `zarr-vectors attribute: dtype=${dtype} expected ${expectedBytes} bytes ` +
        `(${expectedElements} elements), got ${bytes.byteLength}`,
    );
  }
  return elementSize;
}

/**
 * Decode an attribute blob into a `Float32Array` of `expectedElements`.
 *
 * `float32` input is returned as a view over the caller's bytes when alignment
 * allows -- the common case (a gene panel is float32 throughout) stays
 * zero-copy. Everything else is converted element by element; the 64-bit
 * dtypes lose exactness above 2^24, which is why {@link isExactIntDtype}
 * exists to keep them out of the id path.
 */
export function decodeAttributeToFloat32(
  bytes: Uint8Array,
  dtype: VertexAttributeDtype,
  expectedElements: number,
): Float32Array {
  const elementSize = checkLength(bytes, dtype, expectedElements);
  const { buffer, byteOffset } = alignedFor(bytes, elementSize);
  if (dtype === "float32") {
    return new Float32Array(buffer, byteOffset, expectedElements);
  }
  const out = new Float32Array(expectedElements);
  if (dtype === "float64") {
    const wide = new Float64Array(buffer, byteOffset, expectedElements);
    for (let i = 0; i < expectedElements; ++i) out[i] = wide[i];
    return out;
  }
  if (dtype === "float16") {
    const half = new Uint16Array(buffer, byteOffset, expectedElements);
    for (let i = 0; i < expectedElements; ++i) out[i] = halfToFloat(half[i]);
    return out;
  }
  if (dtype === "bool") {
    out.set(new Uint8Array(buffer, byteOffset, expectedElements));
    return out;
  }
  if (dtype === "int64" || dtype === "uint64") {
    const wide =
      dtype === "int64"
        ? new BigInt64Array(buffer, byteOffset, expectedElements)
        : new BigUint64Array(buffer, byteOffset, expectedElements);
    for (let i = 0; i < expectedElements; ++i) out[i] = Number(wide[i]);
    return out;
  }
  const narrow = new EXACT_INT_CTORS[dtype](
    buffer,
    byteOffset,
    expectedElements,
  );
  out.set(narrow);
  return out;
}

/**
 * Decode an attribute blob in its native integer dtype, for the one consumer
 * that needs exactness rather than a uniform type: the vertex-id column, whose
 * values become picking ids. Returns `undefined` for dtypes that cannot carry
 * an exact id (see {@link isExactIntDtype}), which the caller reports.
 */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function decodeAttributeExactInts(
  bytes: Uint8Array,
  dtype: VertexAttributeDtype,
  expectedElements: number,
): ArrayLike<number> | undefined {
  if (!isExactIntDtype(dtype)) return undefined;
  const elementSize = checkLength(bytes, dtype, expectedElements);
  const { buffer, byteOffset } = alignedFor(bytes, elementSize);
  return new EXACT_INT_CTORS[dtype](buffer, byteOffset, expectedElements);
}

/** A zero-filled attribute, for a chunk that lacks the column entirely. */
export function zeroAttribute(numElements: number): Float32Array {
  return new Float32Array(numElements);
}
