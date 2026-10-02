# @license
# Copyright 2026 Allen Institute for Brain Science
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure, zarr-free ZVF helpers -- importable anywhere, unit-testable offline.

Reused by ``browser_fetch_store`` and :mod:`ngpy.zv`, and testable without
importing zarr. Nothing here does I/O.

The two byte decoders read the ZVF §7.3 layout (the same bytes Neuroglancer's
zarr-vectors datasource decodes for rendering); they are kept for scripts that
want to inspect raw chunks without zarr.
"""

from __future__ import annotations

import struct

import numpy as np

from .index import TractIndex

FRAGMENT_INDEX_MAGIC = 0x5A564647  # 'ZVFG' little-endian
FRAGMENT_INDEX_VERSION = 1
_FRAGMENT_HEADER_BYTES = 16


def decode_vlen_bytes(raw: bytes) -> list[bytes]:
    """Decode a zarr v3 `vlen-bytes` chunk into its list of blobs.

    Layout, all little-endian::

        uint32 num_elements N
        for each i in [0, N):
            uint32 byte_length L_i
            byte[L_i] data
    """
    if len(raw) < 4:
        raise ValueError(f"vlen-bytes chunk too short: {len(raw)} < 4 (header)")
    (n,) = struct.unpack_from("<I", raw, 0)
    out = []
    offset = 4
    for i in range(n):
        if len(raw) < offset + 4:
            raise ValueError(f"vlen-bytes chunk truncated at element {i} header")
        (length,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        if len(raw) < offset + length:
            raise ValueError(
                f"vlen-bytes chunk truncated in element {i} payload "
                f"(declared {length}, have {len(raw) - offset})"
            )
        out.append(raw[offset : offset + length])
        offset += length
    if offset != len(raw):
        raise ValueError(
            f"vlen-bytes chunk has {len(raw) - offset} trailing bytes after {n} elements"
        )
    return out


def decode_fragment_index(raw: bytes) -> list[np.ndarray]:
    """Decode a v1 fragment-index blob into one row-index array per fragment.

    Layout, all little-endian::

        HEADER (16 bytes)
            uint32 magic = 0x5A564647, uint16 version = 1, uint16 flags,
            uint32 num_fragments F, uint32 num_range_fragments R
        RANGE BITMAP    ceil(F/8) bytes padded to an 8-byte boundary;
                        bit f (LSB-first) set iff fragment f is a range
        RANGE TABLE     R x (int64 start, int64 count), in fragment order
        EXPLICIT CSR    uint32 offsets[E+1] then int64 indices[T], E = F - R

    Each fragment is either a contiguous range of row indices into the chunk's
    vertex array, or an explicit list of them.
    """
    if len(raw) < _FRAGMENT_HEADER_BYTES:
        raise ValueError(f"fragment index too short: {len(raw)}")
    magic, version, _flags, num_fragments, num_ranges = struct.unpack_from(
        "<IHHII", raw, 0
    )
    if magic != FRAGMENT_INDEX_MAGIC:
        raise ValueError(f"bad fragment-index magic: 0x{magic:08x}")
    if version != FRAGMENT_INDEX_VERSION:
        raise ValueError(f"unsupported fragment-index version {version}")

    offset = _FRAGMENT_HEADER_BYTES
    bitmap_raw = (num_fragments + 7) >> 3
    bitmap_padded = (bitmap_raw + 7) & ~7
    bitmap = np.frombuffer(raw, dtype=np.uint8, count=bitmap_raw, offset=offset)
    offset += bitmap_padded

    is_range = (
        np.unpackbits(bitmap, bitorder="little")[:num_fragments].astype(bool)
        if num_fragments
        else np.zeros(0, dtype=bool)
    )
    if int(is_range.sum()) != num_ranges:
        raise ValueError(
            f"fragment-index bitmap has {int(is_range.sum())} ranges, "
            f"header declares {num_ranges}"
        )

    range_table = np.frombuffer(
        raw, dtype="<i8", count=num_ranges * 2, offset=offset
    ).reshape(-1, 2)
    offset += num_ranges * 16

    num_explicit = num_fragments - num_ranges
    csr_offsets = np.frombuffer(raw, dtype="<u4", count=num_explicit + 1, offset=offset)
    offset += (num_explicit + 1) * 4
    total = int(csr_offsets[-1]) if num_explicit else 0
    csr_indices = np.frombuffer(raw, dtype="<i8", count=total, offset=offset)

    # Rank within each family, so fragment f maps to its own table row.
    range_rank = np.cumsum(is_range) - is_range
    explicit_rank = np.cumsum(~is_range) - ~is_range

    fragments = []
    for f in range(num_fragments):
        if is_range[f]:
            start, count = range_table[range_rank[f]]
            fragments.append(np.arange(start, start + count, dtype=np.intp))
        else:
            e = explicit_rank[f]
            fragments.append(
                csr_indices[csr_offsets[e] : csr_offsets[e + 1]].astype(np.intp)
            )
    return fragments


def apply_byte_range(data: bytes, byte_range: object) -> bytes:
    """Slice ``data`` per a zarr 3 ``ByteRequest``, or return it whole.

    Duck-typed rather than imported, so this stays usable (and unit-testable)
    where zarr is not installed. The three zarr variants are distinguished by
    their attributes: ``RangeByteRequest`` has ``start``/``end``,
    ``OffsetByteRequest`` has ``offset``, ``SuffixByteRequest`` has ``suffix``.
    """
    if byte_range is None:
        return data
    suffix = getattr(byte_range, "suffix", None)
    if suffix is not None:
        return data[len(data) - int(suffix) :]
    offset = getattr(byte_range, "offset", None)
    if offset is not None:
        return data[int(offset) :]
    start = getattr(byte_range, "start", None)
    if start is not None:
        end = getattr(byte_range, "end", None)
        return data[int(start) : (None if end is None else int(end))]
    return data


def _polylines_to_tract_index(result: dict) -> TractIndex:
    """Turn ``zarr_vectors.read_polylines`` output into a `TractIndex`.

    ``read_polylines`` returns ``{"polylines": [...], "object_ids": [...]}``,
    where ``polylines[i]`` is a list of fragment arrays (each ``(k, 3)``) for
    object ``object_ids[i]``. A `TractIndex` row is one fragment, tagged with
    its object id.
    """
    polylines = result["polylines"]
    object_ids = result["object_ids"]
    position_blocks: list[np.ndarray] = []
    counts: list[int] = []
    row_ids: list[int] = []
    for fragment_list, object_id in zip(polylines, object_ids):
        for fragment in fragment_list:
            frag = np.asarray(fragment, dtype=np.float32).reshape(-1, 3)
            if frag.shape[0] == 0:
                continue
            position_blocks.append(frag)
            counts.append(int(frag.shape[0]))
            row_ids.append(int(object_id))
    if not position_blocks:
        raise ValueError("zarr-vectors read returned no vertices")
    positions = np.concatenate(position_blocks)
    offsets = np.zeros(len(counts) + 1, dtype=np.intp)
    np.cumsum(np.asarray(counts, dtype=np.intp), out=offsets[1:])
    return TractIndex(positions, offsets, np.array(row_ids, dtype=np.uint64))


def byte_range_header(byte_range: object) -> str | None:
    """The HTTP ``Range`` header value for a zarr 3 ``ByteRequest`` (or None).

    Duck-typed like :func:`apply_byte_range`.  ``RangeByteRequest`` is
    end-EXCLUSIVE; HTTP ranges are inclusive.
    """
    if byte_range is None:
        return None
    suffix = getattr(byte_range, "suffix", None)
    if suffix is not None:
        return f"bytes=-{int(suffix)}"
    offset = getattr(byte_range, "offset", None)
    if offset is not None:
        return f"bytes={int(offset)}-"
    start = getattr(byte_range, "start", None)
    if start is not None:
        end = getattr(byte_range, "end", None)
        if end is None:
            return f"bytes={int(start)}-"
        if int(end) <= int(start):
            return None
        return f"bytes={int(start)}-{int(end) - 1}"
    return None


def slice_for_range_header(data: bytes, header: str | None) -> bytes:
    """Apply a ``Range`` header to a WHOLE object (a server that ignored it)."""
    if not header:
        return data
    spec = header.split("=", 1)[1]
    first, _, last = spec.partition("-")
    if first == "":
        return data[len(data) - int(last) :]
    start = int(first)
    if last == "":
        return data[start:]
    return data[start : int(last) + 1]
