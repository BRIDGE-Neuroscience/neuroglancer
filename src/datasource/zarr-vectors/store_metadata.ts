/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";

/**
 * Pure, WebGL-free helpers for reading a zarr-vectors store's metadata: chunk
 * index bounds, GLSL-legal property identifiers, the `#attributes=` URL
 * fragment, which declared geometry a store's arrays actually hold, and the
 * saved-document provenance a store carries.
 *
 * These were five one-function modules, each with its own spec. They are
 * collected here because they share the property that matters -- no WebGL, no
 * DOM, so they stay testable under Node -- and splitting on that property one
 * function at a time only multiplied files.
 */

// ---------------------------------------------------------------- chunk_bounds

/**
 * Chunk-index bounds for a zarr-vectors resolution level.
 *
 * Split out of `geometry_frontend.ts` so it can be unit-tested under Node
 * without the WebGL-coupled render-layer imports.
 */

/**
 * Convert a level's world-space extent into the half-open chunk-index range
 * `[lower, upper)` the sliceview frustum walk enumerates.
 *
 * A zarr-vectors `chunk_shape` is a physical extent, not a voxel count, so it
 * is routinely fractional (a 0.5 mm MERFISH grid) and the arithmetic has to
 * stay in floats -- truncating the shape to an integer yields 0 and a bound of
 * +/-Infinity, and the frustum walk then binary-splits a box it can never
 * reduce to one chunk until it exhausts the stack.
 *
 * Chunks are indexed around the world origin, so negative indices are normal
 * and floor/ceil handle the sign. A degenerate axis (lower === upper) still
 * yields one chunk: a zero-volume range would make the walk terminate before
 * drawing anything.
 */
export function computeChunkIndexBounds(
  lowerBounds: ArrayLike<number>,
  upperBounds: ArrayLike<number>,
  chunkShape: ArrayLike<number>,
  rank = 3,
): { lowerChunkBound: Float32Array; upperChunkBound: Float32Array } {
  const lowerChunkBound = new Float32Array(rank);
  const upperChunkBound = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    const size = chunkShape[i];
    if (!(size > 0)) {
      throw new Error(
        `zarr-vectors: chunk_shape[${i}] = ${size} is not a positive extent`,
      );
    }
    lowerChunkBound[i] = Math.floor(lowerBounds[i] / size);
    upperChunkBound[i] = Math.max(
      Math.ceil(upperBounds[i] / size),
      lowerChunkBound[i] + 1,
    );
  }
  return { lowerChunkBound, upperChunkBound };
}

// ---------------------------------------------------------------- attributes_fragment

/**
 * The `#attributes=a,b,c` URL fragment, which names the per-vertex attributes a
 * zarr-vectors layer should expose.
 *
 * Parsing and formatting live together, and in their own module, because they
 * have to be exact inverses: the formatted URL is what gets saved into a layer's
 * JSON, so a name that does not survive the round trip turns a saved link into a
 * load error the next time it is opened.
 */

const PREFIX = "attributes=";

/**
 * Parse the fragment's attribute list, or `undefined` when there is no
 * fragment. Throws when the fragment is something else entirely -- silently
 * ignoring it would drop a selection the user asked for.
 */
export function parseAttributesFragment(
  fragment: string | undefined,
): string[] | undefined {
  if (!fragment) return undefined;
  if (!fragment.startsWith(PREFIX)) {
    throw new Error(
      "the only supported fragment is `#attributes=<comma-separated names>`",
    );
  }
  // Split BEFORE decoding: a name containing a percent-encoded comma (`%2C`)
  // is one attribute, not two, and decoding first would split it.
  return fragment
    .slice(PREFIX.length)
    .split(",")
    .map((part) => {
      const trimmed = part.trim();
      try {
        return decodeURIComponent(trimmed);
      } catch {
        // A stray `%` is not an escape; take the name literally rather than
        // failing the whole load over it.
        return trimmed;
      }
    })
    .filter((name) => name.length > 0);
}

/** Format an attribute list back into a fragment `parseAttributesFragment` reads. */
export function formatAttributesFragment(
  attributes: readonly string[] | undefined,
): string {
  if (attributes === undefined) return "";
  return `#${PREFIX}${attributes.map((n) => encodeURIComponent(n)).join(",")}`;
}

// ---------------------------------------------------------------- declared_geometry

/**
 * Choosing which declared geometry a store's arrays actually hold.
 *
 * A ZVF store's root `zarr_vectors.geometry_types` is a LIST, and the writer
 * library has an `add_geometry()` for multi-geometry ("composite") stores that
 * would put each extra geometry in namespaced `vertices_<type>/` arrays. That
 * writer does not work: `add_geometry` needs the same path to be both a Zarr
 * group and a Zarr array, raises before writing any vertex data, and the
 * library marks the module unfinished for exactly that reason
 * (`zarr_vectors/_stability.py`: "add_geometry currently raises on its own
 * documented example"). So no store in the wild has a `vertices_<type>` array.
 *
 * What DOES reach users is the wreckage: `add_geometry` appends the new type to
 * the root metadata BEFORE it fails, so an aborted call leaves a perfectly good
 * single-geometry store whose `geometry_types` claims two. Refusing to open
 * that store -- which is what "exactly one geometry type" did -- loses the data
 * that is actually there over metadata that is merely aspirational.
 *
 * Hence: pick the kind the on-disk arrays are consistent with, render it, and
 * report the rest rather than failing.
 *
 * Lives in its own module so the choice can be unit-tested without the
 * WebGL-coupled datasource.
 */

/** What the level's arrays reveal about the geometry they hold. */
export interface GeometryObservations {
  /** Whether `links/0` exists at all. */
  readonly hasLinks: boolean;
  /** `links/0`'s declared `link_width`, when it has one. */
  readonly linkWidth?: number;
}

export interface DeclaredGeometryResolution {
  /** The kind the primary arrays are read as. */
  readonly kind: ZarrVectorsGeometryKind;
  /** Declared kinds with no readable arrays of their own. */
  readonly skipped: string[];
  /** Declared strings this reader does not know at all. */
  readonly unsupported: string[];
  /** True when the arrays did not single out `kind` and declaration order did. */
  readonly ambiguous: boolean;
}

/**
 * Resolve which declared geometry to read the primary arrays as.
 *
 * With one declared kind this is that kind. With several, the observations
 * decide where they can: a links family of arity >= 3 can only be a surface,
 * and a store with no links at all can only be a point cloud. Where they cannot
 * decide, declaration order wins and `ambiguous` says so, because guessing
 * silently is worse than a caller that can warn.
 */
export function resolveDeclaredGeometry(
  declared: readonly string[],
  observations: GeometryObservations,
): DeclaredGeometryResolution {
  const known = new Set(Object.keys(KIND_CAPABILITIES));
  const unsupported = declared.filter((g) => !known.has(g));
  const candidates = declared.filter((g) =>
    known.has(g),
  ) as ZarrVectorsGeometryKind[];
  if (candidates.length === 0) {
    throw new Error(
      `zarr-vectors: no recognised geometry type in ${JSON.stringify(declared)}; ` +
        `expected one of ${JSON.stringify([...known])}`,
    );
  }
  if (candidates.length === 1) {
    return {
      kind: candidates[0],
      skipped: [],
      unsupported,
      ambiguous: false,
    };
  }

  const { hasLinks, linkWidth } = observations;
  const consistent = candidates.filter((kind) => {
    const caps = KIND_CAPABILITIES[kind];
    // A surface needs face-arity links; a curve needs pair-arity ones.
    if (linkWidth !== undefined) {
      const wantsFaces = caps.primitive === "triangles";
      if (wantsFaces !== linkWidth >= 3) return false;
    }
    // Something with no connectivity cannot own a links family, and something
    // whose every edge is explicit cannot be missing one.
    if (caps.edgeSource === "none" && hasLinks) return false;
    if (caps.edgeSource === "explicit" && !hasLinks) return false;
    return true;
  });

  const kind = consistent.length > 0 ? consistent[0] : candidates[0];
  return {
    kind,
    skipped: candidates.filter((g) => g !== kind),
    unsupported,
    ambiguous: consistent.length !== 1,
  };
}
