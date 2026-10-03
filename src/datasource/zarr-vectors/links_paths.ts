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
 * @file On-disk path grammar for the ZVF 0.9 links layout.
 *
 * Connectivity lives in a single family per resolution level:
 *
 *     <level>/links/<delta>/<offsets>/<chunk_key>
 *
 * `<delta>` is a signed level span (`"0"`, `"+1"`, `"-1"`, …). `<offsets>`
 * says where the *other* endpoints of a record sit relative to its **source**
 * chunk (the array cell holding it): `link_width - 1` offsets, each a
 * `sid_ndim`-tuple, so the endpoint relationship is factored into the path and
 * each array is a plain rank-D vlen grid over the chunk grid. An intra-chunk
 * link is simply one whose offsets are all zero (`0.0.0`), which is why the
 * separate `cross_chunk_links/` family of ZVF ≤0.8 no longer exists.
 *
 * This is a direct port of `zarr_vectors/core/paths.py` + the `links_has_perm`
 * rule from `core/arrays.py`; it is the single definition of the segment
 * grammar, so callers compose paths through these helpers rather than
 * assembling segments inline.
 */

/** Segment used when `linkWidth === 1` (no relative offsets to encode). */
export const SELF_OFFSETS_SEGMENT = "self";

const OFFSET_SEP = "_";
const COMPONENT_SEP = ".";

/** One `sidNdim`-tuple of signed chunk-grid components. */
export type LinkOffset = readonly number[];

/**
 * Format a signed level delta or offset component: `0 -> "0"`, `+N -> "+N"`,
 * `-N -> "-N"`. The leading `+` is preserved so a listing distinguishes a
 * positive delta from the unsigned `0` at a glance.
 */
export function formatDelta(delta: number): string {
  if (!Number.isInteger(delta)) {
    throw new Error(`links: level-delta must be an integer, got ${delta}`);
  }
  if (delta === 0) return "0";
  return delta > 0 ? `+${delta}` : `${delta}`;
}

/** Inverse of {@link formatDelta}; throws on a malformed segment. */
export function parseDelta(segment: string): number {
  if (segment === "0") return 0;
  if (segment.length === 0 || (segment[0] !== "+" && segment[0] !== "-")) {
    throw new Error(
      `links: invalid level-delta segment: ${JSON.stringify(segment)}`,
    );
  }
  const value = Number(segment);
  if (!Number.isInteger(value)) {
    throw new Error(
      `links: invalid level-delta segment: ${JSON.stringify(segment)}`,
    );
  }
  return value;
}

/**
 * Format `linkWidth - 1` relative offsets as a path segment: components joined
 * by `.`, offsets joined by `_`, each component via {@link formatDelta}. An
 * empty list (`linkWidth === 1`) formats as {@link SELF_OFFSETS_SEGMENT}.
 */
export function formatOffsets(offsets: readonly LinkOffset[]): string {
  if (offsets.length === 0) return SELF_OFFSETS_SEGMENT;
  return offsets
    .map((offset) => offset.map((c) => formatDelta(c)).join(COMPONENT_SEP))
    .join(OFFSET_SEP);
}

/**
 * Inverse of {@link formatOffsets}. Returns the `linkWidth - 1` offset tuples,
 * throwing when the segment's arity disagrees with `sidNdim`/`linkWidth` so a
 * malformed listing fails fast rather than decoding to the wrong geometry.
 */
export function parseOffsets(
  segment: string,
  options: { sidNdim: number; linkWidth: number },
): LinkOffset[] {
  const { sidNdim, linkWidth } = options;
  if (segment === SELF_OFFSETS_SEGMENT) {
    if (linkWidth !== 1) {
      throw new Error(
        `links: offsets segment "self" implies linkWidth=1, got linkWidth=${linkWidth}`,
      );
    }
    return [];
  }
  if (linkWidth === 1) {
    throw new Error(
      `links: linkWidth=1 requires the "self" segment, got ${JSON.stringify(segment)}`,
    );
  }
  const parts = segment.split(OFFSET_SEP);
  const expected = linkWidth - 1;
  if (parts.length !== expected) {
    throw new Error(
      `links: offsets segment ${JSON.stringify(segment)} has ${parts.length} ` +
        `offsets; expected ${expected} (linkWidth=${linkWidth} - 1)`,
    );
  }
  return parts.map((part) => {
    const comps = part.split(COMPONENT_SEP);
    if (comps.length !== sidNdim) {
      throw new Error(
        `links: offset ${JSON.stringify(part)} in segment ${JSON.stringify(segment)} ` +
          `has ${comps.length} components; expected sidNdim=${sidNdim}`,
      );
    }
    return comps.map((c) => parseDelta(c));
  });
}

/** The all-zero offsets identifying the intra-chunk array. */
export function intraOffsets(sidNdim: number, linkWidth: number): LinkOffset[] {
  const zero = new Array<number>(sidNdim).fill(0);
  return new Array<LinkOffset>(Math.max(0, linkWidth - 1)).fill(zero);
}

/** Whether every offset is zero (all endpoints in the source chunk). */
export function isIntra(offsets: readonly LinkOffset[]): boolean {
  return offsets.every((offset) => offset.every((c) => c === 0));
}

/** Base directory name of the links family. */
export const LINKS = "links";

/** Path of the `links/<delta>/` group within a resolution level. */
export function linksGroupPath(delta = 0): string {
  return `${LINKS}/${formatDelta(delta)}`;
}

/** Path of a `links/<delta>/<offsets>/` array within a resolution level. */
export function linksPath(
  delta: number,
  offsets: readonly LinkOffset[],
): string {
  return `${linksGroupPath(delta)}/${formatOffsets(offsets)}`;
}

/**
 * Whether `links/<delta>/<offsets>/` rows carry a leading `perm_idx` column
 * (so rows are `1 + linkWidth` wide instead of `linkWidth`).
 *
 * A direct port of `links_has_perm` in `core/arrays.py` — the single rule the
 * writer and reader both consult to agree on record width. A reader should
 * prefer the array's own stamped `has_perm` and fall back to this only for an
 * array written before the stamp existed; the two must agree.
 *
 * `perm_idx` exists only to undo a canonical sort, so it is present exactly
 * when a non-identity endpoint placement is possible: a non-intra, same-level
 * record that was either canonical-sorted (undirected) or duplicated.
 */
export function linksHasPerm(
  offsets: readonly LinkOffset[],
  options: { delta: number; directed: boolean; store: string },
): boolean {
  const { delta, directed, store } = options;
  // Intra-chunk: identity placement, input order preserved.
  if (isIntra(offsets)) return false;
  // Cross-level: the source is always input endpoint 0.
  if (delta !== 0) return false;
  // Duplicate: each copy leads with a different endpoint, so the permutation
  // varies per copy and must be recoverable.
  if (store === "duplicate") return true;
  // Same-level canonical: undirected sorts (perm needed); directed keeps input
  // order (no perm).
  return !directed;
}
