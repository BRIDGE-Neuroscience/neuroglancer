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
 * The ROI-store client against an in-memory stand-in for the four GCS JSON-API
 * operations it uses (list with prefix + pagination, read `alt=media`,
 * multipart / media upload, delete), injected as `fetch`.  Ported from the old
 * `tests/roi_store/gcs_client.spec.ts`, which ran against a fake server.
 */

import { describe, expect, it } from "vitest";
import {
  RoiGroupStore,
  RoiStoreListForbiddenError,
  roiGroupStoreChanged,
} from "../../ngpy/src/store/gcs_client.js";
import { makeRoiGroupDocument, parseRoiGroupDocument } from "../../ngpy/src/store/schema.js";

interface StoredObject {
  body: string | Uint8Array;
  contentType: string;
  metadata: Record<string, string>;
}

function fakeGcs(options: { pageSize?: number; anonymousList?: boolean; token?: string } = {}) {
  const objects = new Map<string, StoredObject>();
  const requests: { method: string; url: string; auth: string | null }[] = [];
  const pageSize = options.pageSize ?? 1000;
  const fetchImpl = async (input: string, init: RequestInit = {}): Promise<Response> => {
    const url = new URL(input);
    const method = init.method ?? "GET";
    const auth = new Headers(init.headers).get("Authorization");
    requests.push({ method, url: input, auth });
    const authorised = auth === `Bearer ${options.token ?? "good"}`;
    const m = /^\/(upload\/)?storage\/v1\/b\/([^/]+)\/o(?:\/(.+))?$/.exec(url.pathname);
    if (m === null) return new Response("not found", { status: 404 });
    const [, upload, , encodedName] = m;
    if (upload) {
      if (!authorised) return new Response("unauthorised", { status: 401 });
      if (url.searchParams.get("uploadType") === "multipart") {
        const body = String(init.body);
        const contentType = new Headers(init.headers).get("Content-Type") ?? "";
        const boundary = /boundary=(.+)$/.exec(contentType)![1];
        const parts = body
          .split(`--${boundary}`)
          .filter((p) => p.includes("\r\n\r\n"))
          .map((p) => p.slice(p.indexOf("\r\n\r\n") + 4).trim());
        const meta = JSON.parse(parts[0]);
        const doc = parts[1];
        objects.set(meta.name, { body: doc, contentType: meta.contentType, metadata: meta.metadata });
      } else {
        const name = url.searchParams.get("name")!;
        const body = new Uint8Array(await new Response(init.body as BodyInit).arrayBuffer());
        objects.set(name, {
          body,
          contentType: new Headers(init.headers).get("Content-Type") ?? "",
          metadata: {},
        });
      }
      return new Response("{}", { status: 200 });
    }
    if (encodedName !== undefined) {
      const name = decodeURIComponent(encodedName);
      if (method === "DELETE") {
        if (!authorised) return new Response("unauthorised", { status: 403 });
        objects.delete(name);
        return new Response(null, { status: 204 });
      }
      const obj = objects.get(name);
      if (obj === undefined) return new Response("missing", { status: 404 });
      return new Response(obj.body as BodyInit, { status: 200 });
    }
    if (options.anonymousList === false && !authorised) {
      return new Response("forbidden", { status: 401 });
    }
    const prefix = url.searchParams.get("prefix") ?? "";
    const names = [...objects.keys()].filter((n) => n.startsWith(prefix)).sort();
    const start = Number(url.searchParams.get("pageToken") ?? 0);
    const page = names.slice(start, start + pageSize);
    return Response.json({
      items: page.map((name) => ({ name, updated: "2026-10-02T00:00:00Z", metadata: objects.get(name)!.metadata })),
      ...(start + pageSize < names.length ? { nextPageToken: String(start + pageSize) } : {}),
    });
  };
  return { objects, requests, fetch: fetchImpl };
}

function auth(token = "good") {
  let current: string | undefined = token;
  let invalidations = 0;
  return {
    get cachedAccessToken() {
      return current;
    },
    getAccessToken: async () => {
      current ??= "good";
      return current;
    },
    invalidate: () => {
      ++invalidations;
      current = undefined;
    },
    get invalidations() {
      return invalidations;
    },
  };
}

const GROUP = {
  name: "Arcuate L",
  color: "#ff0000",
  opacity: 0.42,
  rois: [
    { shape: { type: "ellipsoid", center: [1.5, -2.5, 3.5], radii: [4, 5, 6] }, predicate: "any_segment", operator: "and" },
    { shape: { type: "box", lower: [0, 0, 0], upper: [10, 20, 30] }, predicate: "either_endpoint", operator: "andnot" },
  ],
};

function doc(name = "Arcuate L", id?: string) {
  return makeRoiGroupDocument({
    group: { ...GROUP, name },
    source: { url: "gs://b/tracts.zarrvectors/|zarr-vectors:" },
    createdBy: "test@example.com",
    id,
  });
}

describe("RoiGroupStore", () => {
  it("round-trips a group through save and read, under groups/<id>.json", async () => {
    const gcs = fakeGcs();
    const store = new RoiGroupStore({ bucket: "b", endpoint: "http://gcs.test", auth: auth(), fetch: gcs.fetch });
    const d = doc();
    await store.save(d);
    expect(gcs.objects.has(`groups/${d.id}.json`)).toBe(true);
    const loaded = await store.read(d.id);
    expect(loaded.group).toEqual(d.group);
    expect(loaded.createdBy).toBe("test@example.com");
    expect(loaded.schemaVersion).toBe(1);
  });

  it("lists anonymously from custom metadata, across pages", async () => {
    const gcs = fakeGcs({ pageSize: 2 });
    const store = new RoiGroupStore({ bucket: "b", endpoint: "http://gcs.test", auth: auth(), fetch: gcs.fetch });
    for (const n of ["A", "B", "C"]) await store.save(doc(n));
    gcs.objects.set("exports/x.trk", { body: "", contentType: "", metadata: {} });
    const list = await store.list();
    expect(list.map((s) => s.name).sort()).toEqual(["A", "B", "C"]);
    expect(list[0].sourceUrl).toBe("gs://b/tracts.zarrvectors/|zarr-vectors:");
    const listCalls = gcs.requests.filter((r) => r.method === "GET" && r.url.includes("prefix="));
    expect(listCalls.every((r) => r.auth === null)).toBe(true);
    expect(listCalls).toHaveLength(2);
  });

  it("escalates a refused listing to a HELD token, else explains", async () => {
    const gcs = fakeGcs({ anonymousList: false });
    const signedIn = new RoiGroupStore({ bucket: "b", endpoint: "http://gcs.test", auth: auth(), fetch: gcs.fetch });
    await expect(signedIn.list()).resolves.toEqual([]);
    const a = auth();
    a.invalidate();
    const anonymous = new RoiGroupStore({ bucket: "b", endpoint: "http://gcs.test", auth: a, fetch: gcs.fetch });
    await expect(anonymous.list()).rejects.toBeInstanceOf(RoiStoreListForbiddenError);
  });

  it("retries a write ONCE with a fresh token after 401", async () => {
    const gcs = fakeGcs();
    const a = auth("stale");
    const store = new RoiGroupStore({ bucket: "b", endpoint: "http://gcs.test", auth: a, fetch: gcs.fetch });
    await store.save(doc());
    expect(a.invalidations).toBe(1);
    const uploads = gcs.requests.filter((r) => r.url.includes("/upload/"));
    expect(uploads.map((r) => r.auth)).toEqual(["Bearer stale", "Bearer good"]);
  });

  it("uploads export bytes under exports/ and deletes documents", async () => {
    const gcs = fakeGcs();
    const store = new RoiGroupStore({ bucket: "b", endpoint: "http://gcs.test", auth: auth(), fetch: gcs.fetch });
    let changes = 0;
    const off = roiGroupStoreChanged.add(() => ++changes);
    const name = await store.putObject("exports/d.trk", new Uint8Array([1, 2, 3]), "application/octet-stream");
    expect(name).toBe("exports/d.trk");
    expect([...(gcs.objects.get("exports/d.trk")!.body as Uint8Array)]).toEqual([1, 2, 3]);
    const d = doc();
    await store.save(d);
    await store.delete(d.id);
    expect(gcs.objects.has(`groups/${d.id}.json`)).toBe(false);
    expect(changes).toBe(2); // save + delete, not the export upload
    off();
  });

  it("refuses writes without an auth source", async () => {
    const store = new RoiGroupStore({ bucket: "b", fetch: fakeGcs().fetch });
    await expect(store.save(doc())).rejects.toThrow(/sign-in/);
  });
});

describe("parseRoiGroupDocument", () => {
  it("accepts a document and rejects a newer schema", () => {
    const d = doc();
    expect(parseRoiGroupDocument(JSON.parse(JSON.stringify(d)))).toEqual(d);
    expect(() => parseRoiGroupDocument({ ...d, schemaVersion: 2 })).toThrow(/newer/);
    expect(() => parseRoiGroupDocument({ ...d, group: [] })).toThrow();
  });
});
