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
 * @file Saved ROI groups and export artifacts in a Google Cloud Storage bucket.
 *
 * Ported from `src/roi_store/gcs_client.ts` (`zarr_vectors_roi_store`) with
 * Neuroglancer's http helpers replaced by plain `fetch`, and `fetch` injectable
 * so the logic is unit-tested against a stub.  The bucket is public-read:
 * listing and loading are anonymous; only writes carry a token.
 */

import { Signal } from "../util/signal.js";
import type { RoiGroupDocument, RoiGroupSummary } from "./schema.js";
import {
  parseRoiGroupDocument,
  roiGroupCustomMetadata,
  roiGroupIdFromObjectName,
  ROI_GROUP_PREFIX,
  roiGroupObjectName,
} from "./schema.js";

export const DEFAULT_STORAGE_ENDPOINT = "https://storage.googleapis.com";

/** Fires after the store's contents change, so open listings can refresh. */
export const roiGroupStoreChanged = new Signal();

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly url: string,
    message?: string,
  ) {
    super(message ?? `HTTP ${status} from ${url}`);
    this.name = "HttpError";
  }
}

export interface RoiStoreTokenSource {
  getAccessToken(signal?: AbortSignal): Promise<string>;
  invalidate(): void;
  /** A token if one is already held; must NOT prompt for sign-in. */
  readonly cachedAccessToken?: string | undefined;
}

export class RoiStoreListForbiddenError extends Error {
  constructor(readonly signedIn: boolean) {
    super(
      signedIn
        ? "This account is not allowed to list the ROI group bucket."
        : "This bucket does not allow anonymous listing. Sign in, or grant " +
            "allUsers the Storage Object Viewer role on the bucket.",
    );
    this.name = "RoiStoreListForbiddenError";
  }
}

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface RoiGroupStoreOptions {
  bucket: string;
  endpoint?: string;
  auth?: RoiStoreTokenSource;
  fetch?: FetchLike;
}

export class RoiGroupStore {
  readonly bucket: string;
  readonly endpoint: string;
  private auth: RoiStoreTokenSource | undefined;
  private fetchImpl: FetchLike;

  constructor(options: RoiGroupStoreOptions) {
    this.bucket = options.bucket;
    this.endpoint = (options.endpoint ?? DEFAULT_STORAGE_ENDPOINT).replace(/\/+$/, "");
    this.auth = options.auth;
    this.fetchImpl = options.fetch ?? ((input, init) => fetch(input, init));
  }

  private async fetchOk(url: string, init?: RequestInit): Promise<Response> {
    const response = await this.fetchImpl(url, init);
    if (!response.ok) throw new HttpError(response.status, url);
    return response;
  }

  private get objectApiUrl(): string {
    return `${this.endpoint}/storage/v1/b/${encodeURIComponent(this.bucket)}/o`;
  }

  private get uploadApiUrl(): string {
    return `${this.endpoint}/upload/storage/v1/b/${encodeURIComponent(this.bucket)}/o`;
  }

  private objectUrl(name: string): URL {
    const url = new URL(this.objectApiUrl);
    url.pathname += `/${encodeURIComponent(name)}`;
    return url;
  }

  /** Lists saved groups: anonymous first, escalating to a HELD token if refused. */
  async list(signal?: AbortSignal): Promise<RoiGroupSummary[]> {
    const summaries: RoiGroupSummary[] = [];
    let pageToken: string | undefined;
    do {
      const url = new URL(this.objectApiUrl);
      url.searchParams.set("prefix", ROI_GROUP_PREFIX);
      url.searchParams.set("fields", "nextPageToken,items(name,updated,metadata)");
      if (pageToken !== undefined) url.searchParams.set("pageToken", pageToken);
      const response = await this.fetchListPage(url.toString(), signal);
      const page = await response.json();
      for (const item of page?.items ?? []) {
        const id = roiGroupIdFromObjectName(item?.name ?? "");
        if (id === undefined) continue;
        const metadata = item.metadata ?? {};
        summaries.push({
          id,
          name: metadata.roiGroupName ?? id,
          createdBy: metadata.createdBy,
          sourceUrl: metadata.sourceUrl,
          updated: item.updated,
        });
      }
      pageToken = page?.nextPageToken ?? undefined;
    } while (pageToken !== undefined);
    return summaries;
  }

  private async fetchListPage(url: string, signal?: AbortSignal): Promise<Response> {
    try {
      return await this.fetchOk(url, { signal });
    } catch (error) {
      const refused =
        error instanceof HttpError && (error.status === 401 || error.status === 403);
      if (!refused) throw error;
      const token = this.auth?.cachedAccessToken;
      if (token === undefined) throw new RoiStoreListForbiddenError(false);
      try {
        return await this.fetchOk(url, {
          signal,
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch (authedError) {
        if (
          authedError instanceof HttpError &&
          (authedError.status === 401 || authedError.status === 403)
        ) {
          this.auth?.invalidate();
          throw new RoiStoreListForbiddenError(true);
        }
        throw authedError;
      }
    }
  }

  async read(id: string, signal?: AbortSignal): Promise<RoiGroupDocument> {
    const url = this.objectUrl(roiGroupObjectName(id));
    url.searchParams.set("alt", "media");
    const response = await this.fetchOk(url.toString(), { signal });
    return parseRoiGroupDocument(await response.json());
  }

  async save(doc: RoiGroupDocument, signal?: AbortSignal): Promise<void> {
    const url = new URL(this.uploadApiUrl);
    url.searchParams.set("uploadType", "multipart");
    const { body, contentType } = multipartUploadBody(doc);
    await this.fetchWithToken(
      url.toString(),
      { method: "POST", body, headers: { "Content-Type": contentType } },
      signal,
    );
    roiGroupStoreChanged.dispatch();
  }

  /** Upload an export artifact (`.trk`, zipped `.zvf`) under `name`. */
  async putObject(
    name: string,
    data: Blob | ArrayBuffer | Uint8Array,
    contentType: string,
    signal?: AbortSignal,
  ): Promise<string> {
    const url = new URL(this.uploadApiUrl);
    url.searchParams.set("uploadType", "media");
    url.searchParams.set("name", name);
    await this.fetchWithToken(
      url.toString(),
      {
        method: "POST",
        body: data as BodyInit,
        headers: { "Content-Type": contentType },
      },
      signal,
    );
    return name;
  }

  async delete(id: string, signal?: AbortSignal): Promise<void> {
    await this.fetchWithToken(
      this.objectUrl(roiGroupObjectName(id)).toString(),
      { method: "DELETE" },
      signal,
    );
    roiGroupStoreChanged.dispatch();
  }

  /** An authenticated request, retried ONCE with a fresh token on 401/403. */
  private async fetchWithToken(
    url: string,
    init: RequestInit,
    signal?: AbortSignal,
  ): Promise<Response> {
    const { auth } = this;
    if (auth === undefined) throw new Error("Saving to the ROI store requires sign-in");
    for (let attempt = 0; ; ++attempt) {
      const token = await auth.getAccessToken(signal);
      const headers = new Headers(init.headers);
      headers.set("Authorization", `Bearer ${token}`);
      try {
        return await this.fetchOk(url, { ...init, headers, signal });
      } catch (error) {
        const rejected =
          error instanceof HttpError && (error.status === 401 || error.status === 403);
        if (!rejected || attempt > 0) throw error;
        auth.invalidate();
      }
    }
  }
}

export function multipartUploadBody(doc: RoiGroupDocument): {
  body: string;
  contentType: string;
} {
  const boundary = `roi-group-boundary-${doc.id}`;
  const metadata = {
    name: roiGroupObjectName(doc.id),
    contentType: "application/json",
    metadata: roiGroupCustomMetadata(doc),
  };
  const body =
    `--${boundary}\r\n` +
    "Content-Type: application/json; charset=UTF-8\r\n\r\n" +
    `${JSON.stringify(metadata)}\r\n` +
    `--${boundary}\r\n` +
    "Content-Type: application/json; charset=UTF-8\r\n\r\n" +
    `${JSON.stringify(doc)}\r\n` +
    `--${boundary}--`;
  return { body, contentType: `multipart/related; boundary=${boundary}` };
}
