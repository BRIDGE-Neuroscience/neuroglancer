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
 * @file Element types of zarr-vectors arrays and their decoding.  Every
 * value the viewer draws becomes float32 (positions, attributes); ids become
 * uint64; vertex indices become uint32.  All data is little-endian.
 */

export type ElementType =
  | "bool"
  | "int8"
  | "uint8"
  | "int16"
  | "uint16"
  | "int32"
  | "uint32"
  | "int64"
  | "uint64"
  | "float16"
  | "float32"
  | "float64";

export const ELEMENT_BYTES: Record<ElementType, number> = {
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

export function isElementType(value: unknown): value is ElementType {
  return typeof value === "string" && value in ELEMENT_BYTES;
}

function view(bytes: Uint8Array) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function checkLength(bytes: Uint8Array, type: ElementType, count: number) {
  const expected = count * ELEMENT_BYTES[type];
  if (bytes.byteLength !== expected) {
    throw new Error(
      `expected ${expected} bytes of ${type} (${count} values), got ${bytes.byteLength}`,
    );
  }
}

export function float16ToNumber(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** The IEEE half-precision bits nearest `x` (ties to even). */
export function numberToFloat16(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  let a = Math.abs(x);
  if (a >= 65520) return sign | 0x7c00; // rounds past the largest half
  if (a < 2 ** -14) {
    // Subnormal: a multiple of 2^-24.
    a = a * 2 ** 24;
    let m = Math.floor(a);
    const r = a - m;
    if (r > 0.5 || (r === 0.5 && m & 1)) ++m;
    return sign | m; // m === 1024 is the smallest normal, as bits
  }
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) --e;
  else if (2 ** (e + 1) <= a) ++e;
  const scaled = (a / 2 ** e - 1) * 1024;
  let m = Math.floor(scaled);
  const r = scaled - m;
  if (r > 0.5 || (r === 0.5 && m & 1)) ++m;
  if (m === 1024) {
    m = 0;
    ++e;
  }
  return sign | ((e + 15) << 10) | m;
}

/** Decodes `count` values of `type` to float32 (zero-copy for aligned float32). */
export function decodeFloat32(
  bytes: Uint8Array,
  type: ElementType,
  count: number,
): Float32Array<ArrayBuffer> {
  checkLength(bytes, type, count);
  if (type === "float32" && bytes.byteOffset % 4 === 0) {
    return new Float32Array(
      bytes.buffer as ArrayBuffer,
      bytes.byteOffset,
      count,
    );
  }
  const v = view(bytes);
  const out = new Float32Array(count);
  const read: (i: number) => number = {
    bool: (i: number) => v.getUint8(i),
    uint8: (i: number) => v.getUint8(i),
    int8: (i: number) => v.getInt8(i),
    uint16: (i: number) => v.getUint16(2 * i, true),
    int16: (i: number) => v.getInt16(2 * i, true),
    float16: (i: number) => float16ToNumber(v.getUint16(2 * i, true)),
    uint32: (i: number) => v.getUint32(4 * i, true),
    int32: (i: number) => v.getInt32(4 * i, true),
    float32: (i: number) => v.getFloat32(4 * i, true),
    uint64: (i: number) => Number(v.getBigUint64(8 * i, true)),
    int64: (i: number) => Number(v.getBigInt64(8 * i, true)),
    float64: (i: number) => v.getFloat64(8 * i, true),
  }[type];
  for (let i = 0; i < count; ++i) out[i] = read(i);
  return out;
}

/** Decodes integer values to uint64 ids (`-1` becomes 2^64 - 1, never a valid id). */
export function decodeUint64(
  bytes: Uint8Array,
  type: ElementType,
  count: number,
): BigUint64Array {
  checkLength(bytes, type, count);
  if ((type === "int64" || type === "uint64") && bytes.byteOffset % 8 === 0) {
    return new BigUint64Array(bytes.buffer, bytes.byteOffset, count).slice();
  }
  const v = view(bytes);
  const out = new BigUint64Array(count);
  const width = ELEMENT_BYTES[type];
  for (let i = 0; i < count; ++i) {
    switch (type) {
      case "int64":
      case "uint64":
        out[i] = v.getBigUint64(8 * i, true);
        break;
      case "int32":
      case "int16":
      case "int8": {
        const x =
          width === 4
            ? v.getInt32(4 * i, true)
            : width === 2
              ? v.getInt16(2 * i, true)
              : v.getInt8(i);
        out[i] = BigInt.asUintN(64, BigInt(x));
        break;
      }
      case "uint32":
        out[i] = BigInt(v.getUint32(4 * i, true));
        break;
      case "uint16":
        out[i] = BigInt(v.getUint16(2 * i, true));
        break;
      case "uint8":
      case "bool":
        out[i] = BigInt(v.getUint8(i));
        break;
      default:
        throw new Error(`${type} is not an integer type`);
    }
  }
  return out;
}

/** Decodes non-negative integer vertex indices to uint32. */
export function decodeIndices(
  bytes: Uint8Array,
  type: ElementType,
  count: number,
): Uint32Array {
  if (type === "int64" || type === "uint64") {
    // Read the low word of each value; the high word must be zero.
    checkLength(bytes, type, count);
    const v = view(bytes);
    const out = new Uint32Array(count);
    for (let i = 0; i < count; ++i) {
      if (v.getUint32(8 * i + 4, true) !== 0) {
        throw new Error("vertex index out of range");
      }
      out[i] = v.getUint32(8 * i, true);
    }
    return out;
  }
  const values = decodeUint64(bytes, type, count);
  const out = new Uint32Array(count);
  for (let i = 0; i < count; ++i) {
    if (values[i] > 0xffffffffn) {
      throw new Error(`vertex index ${values[i]} out of range`);
    }
    out[i] = Number(values[i]);
  }
  return out;
}

/**
 * An attribute array's element type: zarr-vectors' `dtype` stamp, else the
 * zarr data type, else float32 for a byte array with no stamp (as
 * zarr-vectors-py reads it).
 */
export function attributeDtype(json: any): string {
  const stamped = json?.attributes?.dtype;
  if (stamped !== undefined) return String(stamped);
  const dataType = json?.data_type;
  return typeof dataType === "string" && isElementType(dataType)
    ? dataType
    : "float32";
}
