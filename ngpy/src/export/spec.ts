/**
 * @license
 * Copyright 2026 The Neuroglancer Authors
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
 * @file The export job spec the Export tab hands to `ngpy.api.export`.
 *
 * Schema v3 of the old `neuroglancer.tract_export` job, unchanged, so a spec
 * downloaded from here still runs on the old native exporter.  Its Python
 * twin is `ngpy/python/ngpy/tract_export/job.py`; the fixture shared by the
 * two test suites (`VALID` there, `tests/ngpy/export_spec.spec.ts` here) keeps
 * them in step.
 */

export const JOB_SCHEMA_VERSION = 3;

export type ExportFormat = "trk" | "zvf";
export type ExportScope = "selected" | "whole";
export type ExportDestination = "download" | "gcs";

export interface ExportGroupInput {
  name: string;
  color: string;
  rois: any[];
  /** Passing zarr-vectors object ids (decimal strings: ids are uint64). */
  objectIds: string[];
}

export interface ExportOptions {
  sourceUrl: string;
  level: number;
  format: ExportFormat;
  scope: ExportScope;
  groups: ExportGroupInput[];
  affine?: number[][];
  fileName: string;
  destination: ExportDestination;
}

export function buildJobSpec(o: ExportOptions): any {
  const spec: any = {
    schemaVersion: JOB_SCHEMA_VERSION,
    source: { url: o.sourceUrl, level: o.level },
    groups:
      o.scope === "whole"
        ? []
        : o.groups.map((g) => ({
            name: g.name,
            color: g.color,
            rois: g.rois,
            objectIds: g.objectIds.map(String),
          })),
    format: o.format,
    scope: o.scope,
    destination: { kind: o.destination, path: exportFileName(o.fileName, o.format) },
  };
  if (o.affine !== undefined) spec.affine = o.affine;
  return spec;
}

/** Voxel->RAS(mm) for a store whose unit is `unitM` metres. */
export function defaultAffine(unitM: number): number[][] {
  const s = unitM * 1000;
  return [
    [s, 0, 0, 0],
    [0, s, 0, 0],
    [0, 0, s, 0],
    [0, 0, 0, 1],
  ];
}

export function formatAffineText(m: number[][]): string {
  return m.map((row) => row.map((v) => String(v)).join(" ")).join("\n");
}

/** Parse 16 whitespace/comma-separated numbers; blank means identity (undefined). */
export function parseAffineText(text: string): number[][] | undefined {
  const values = text
    .split(/[\s,;[\]]+/)
    .filter((s) => s.length > 0)
    .map(Number);
  if (values.length === 0) return undefined;
  if (values.length !== 16 || values.some((v) => !Number.isFinite(v))) {
    throw new Error("The affine must be 16 finite numbers (a 4x4 matrix).");
  }
  return [0, 1, 2, 3].map((r) => values.slice(r * 4, r * 4 + 4));
}

export function exportFileName(base: string, format: ExportFormat): string {
  const stem = (base.trim() || "dissection").replace(/(\.(trk|zvf|zip))+$/i, "");
  return format === "trk" ? `${stem}.trk` : `${stem}.zvf.zip`;
}
