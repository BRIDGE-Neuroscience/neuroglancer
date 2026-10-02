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
 * @file Document format for a saved ROI group (`groups/<id>.json`).
 *
 * Ported unchanged in meaning from `src/roi_store/schema.ts` on the
 * `zarr_vectors_roi_store` branch, so documents saved by the old viewer load
 * here and vice versa.  `group` is the verbatim group persistence JSON
 * (`groupToJson`), with ROI shapes in the store's coordinate frame.
 */

export const ROI_GROUP_SCHEMA_VERSION = 1;
export const ROI_GROUP_PREFIX = "groups/";

export interface RoiGroupSource {
  url: string;
  coordinateSpace?: any;
}

export interface RoiGroupScene {
  url?: string;
  layerName?: string;
}

export interface RoiGroupDocument {
  schemaVersion: number;
  id: string;
  group: any;
  source: RoiGroupSource;
  scene?: RoiGroupScene;
  createdBy?: string;
  createdAt: string;
  updatedAt: string;
}

export interface RoiGroupSummary {
  id: string;
  name: string;
  createdBy?: string;
  sourceUrl?: string;
  updated?: string;
}

export function roiGroupObjectName(id: string): string {
  return `${ROI_GROUP_PREFIX}${id}.json`;
}

export function roiGroupIdFromObjectName(name: string): string | undefined {
  if (!name.startsWith(ROI_GROUP_PREFIX) || !name.endsWith(".json")) {
    return undefined;
  }
  const id = name.slice(ROI_GROUP_PREFIX.length, -".json".length);
  return id.length > 0 && !id.includes("/") ? id : undefined;
}

function randomHex(bits: number): string {
  const bytes = new Uint8Array(bits / 8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export function makeRoiGroupDocument(options: {
  group: any;
  source: RoiGroupSource;
  scene?: RoiGroupScene;
  createdBy?: string;
  id?: string;
  createdAt?: string;
}): RoiGroupDocument {
  const now = new Date().toISOString();
  return {
    schemaVersion: ROI_GROUP_SCHEMA_VERSION,
    id: options.id ?? randomHex(128),
    group: options.group,
    source: options.source,
    ...(options.scene === undefined ? {} : { scene: options.scene }),
    ...(options.createdBy === undefined
      ? {}
      : { createdBy: options.createdBy }),
    createdAt: options.createdAt ?? now,
    updatedAt: now,
  };
}

function req<T>(obj: any, key: string, check: (v: any) => T): T {
  if (obj === null || typeof obj !== "object" || !(key in obj)) {
    throw new Error(`Missing property ${JSON.stringify(key)}`);
  }
  return check(obj[key]);
}

function opt<T>(obj: any, key: string, check: (v: any) => T): T | undefined {
  return obj[key] === undefined ? undefined : check(obj[key]);
}

function str(v: any): string {
  if (typeof v !== "string")
    throw new Error(`Expected string, got ${JSON.stringify(v)}`);
  return v;
}

function obj(v: any): any {
  if (v === null || typeof v !== "object" || Array.isArray(v)) {
    throw new Error(`Expected object, got ${JSON.stringify(v)}`);
  }
  return v;
}

export function parseRoiGroupDocument(json: unknown): RoiGroupDocument {
  const o = obj(json);
  const schemaVersion = req(o, "schemaVersion", (v) => {
    if (typeof v !== "number" || !Number.isInteger(v)) {
      throw new Error(`Expected integer, got ${JSON.stringify(v)}`);
    }
    if (v > ROI_GROUP_SCHEMA_VERSION) {
      throw new Error(
        `Document schema version ${v} is newer than this viewer supports ` +
          `(${ROI_GROUP_SCHEMA_VERSION})`,
      );
    }
    return v;
  });
  return {
    schemaVersion,
    id: req(o, "id", str),
    group: req(o, "group", obj),
    source: req(o, "source", (v) => {
      obj(v);
      return { url: req(v, "url", str), coordinateSpace: v.coordinateSpace };
    }),
    scene: opt(o, "scene", (v) => {
      obj(v);
      return { url: opt(v, "url", str), layerName: opt(v, "layerName", str) };
    }),
    createdBy: opt(o, "createdBy", str),
    createdAt: req(o, "createdAt", str),
    updatedAt: req(o, "updatedAt", str),
  };
}

export function roiGroupCustomMetadata(
  doc: RoiGroupDocument,
): Record<string, string> {
  const metadata: Record<string, string> = {};
  const name = doc.group?.name;
  if (typeof name === "string") metadata.roiGroupName = name;
  if (doc.createdBy !== undefined) metadata.createdBy = doc.createdBy;
  if (doc.source?.url !== undefined) metadata.sourceUrl = doc.source.url;
  return metadata;
}
