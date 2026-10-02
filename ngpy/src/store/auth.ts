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
 * @file Sign-in for ROI-store writes, without Neuroglancer's credential stack.
 *
 * - `google`: the OAuth2 implicit flow in a popup whose redirect target is
 *   THIS PAGE (ngpy.html).  When the popup lands back here with
 *   `#access_token=...`, `handleOAuthRedirect` (run before the app boots)
 *   hands the fragment to the opener -- `postMessage`, plus a
 *   `BroadcastChannel` in case cross-origin isolation severed `opener` -- and
 *   closes.  The page URL must be registered as an authorised redirect URI of
 *   the OAuth client.
 * - `middleauth`: CAVE's `/api/v1/authorize` popup, which posts
 *   `{token, app_urls}` to its opener (the same protocol Neuroglancer's
 *   middleauth provider uses).  The token is stored under Neuroglancer's own
 *   localStorage key, so a same-origin hosted viewer sees the sign-in too.
 */

import type { RoiStoreTokenSource } from "./gcs_client.js";
import { Signal } from "../util/signal.js";

export interface RoiStoreConfig {
  bucket: string;
  endpoint?: string;
  provider?: "google" | "middleauth";
  clientId?: string;
  scopes?: string[];
  authServer?: string;
}

export const DEFAULT_SCOPES = [
  "openid",
  "email",
  "https://www.googleapis.com/auth/devstorage.read_write",
];

export interface RoiStoreAuth extends RoiStoreTokenSource {
  readonly changed: Signal;
  readonly signedIn: boolean;
  readonly email: string | undefined;
  signIn(): Promise<void>;
  signOut(): void;
}

const OAUTH_CHANNEL = "ngpy-oauth";
const EXPIRY_MARGIN_MS = 5 * 60 * 1000;

interface StoredToken {
  accessToken: string;
  email?: string;
  expiresAt: number;
}

function readJson(key: string): any {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? undefined : JSON.parse(raw);
  } catch {
    return undefined;
  }
}

function writeJson(key: string, value: unknown) {
  try {
    if (value === undefined) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage unavailable (private mode): keep the token in memory only.
  }
}

/** The page URL a popup should be redirected back to (no query, no hash). */
export function redirectUri(location: Location = window.location): string {
  return `${location.origin}${location.pathname}`;
}

export function googleAuthorizeUrl(
  clientId: string,
  scopes: string[],
  state: string,
  redirect: string,
): string {
  const url = new URL("https://accounts.google.com/o/oauth2/v2/auth");
  url.searchParams.set("client_id", clientId);
  url.searchParams.set("redirect_uri", redirect);
  url.searchParams.set("response_type", "token");
  url.searchParams.set("scope", scopes.join(" "));
  url.searchParams.set("state", state);
  url.searchParams.set("prompt", "select_account");
  url.searchParams.set("include_granted_scopes", "true");
  return url.toString();
}

/**
 * If this page load is an OAuth redirect, forward the result to the opener and
 * close.  Returns true when it handled one (the app must then not boot).
 */
export function handleOAuthRedirect(): boolean {
  const hash = window.location.hash.replace(/^#/, "");
  if (!/(^|&)(access_token|error)=/.test(hash) || !/(^|&)state=/.test(hash)) {
    return false;
  }
  const params = Object.fromEntries(new URLSearchParams(hash));
  const message = { ngpyOAuth: params };
  try {
    window.opener?.postMessage(message, window.location.origin);
  } catch {
    // Opener severed; the BroadcastChannel below still reaches it.
  }
  try {
    const channel = new BroadcastChannel(OAUTH_CHANNEL);
    channel.postMessage(message);
    channel.close();
  } catch {
    // No BroadcastChannel.
  }
  document.body.textContent = params.error
    ? `Sign-in failed: ${params.error}. You can close this window.`
    : "Signed in. You can close this window.";
  setTimeout(() => window.close(), 300);
  return true;
}

function openPopup(url: string): Window {
  const w = 480;
  const h = 640;
  const left = Math.max(0, window.screenX + (window.outerWidth - w) / 2);
  const top = Math.max(0, window.screenY + (window.outerHeight - h) / 2);
  const popup = window.open(
    url,
    "_blank",
    `width=${w},height=${h},left=${left},top=${top}`,
  );
  if (popup === null) throw new Error("The sign-in popup was blocked");
  return popup;
}

function randomState(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export class GoogleRoiStoreAuth implements RoiStoreAuth {
  readonly changed = new Signal();
  private token: StoredToken | undefined;
  private pending: Promise<StoredToken> | undefined;
  private static readonly KEY = "ngpy_roi_store_token_v1";

  constructor(private config: RoiStoreConfig) {
    const stored = readJson(GoogleRoiStoreAuth.KEY);
    if (
      stored?.accessToken &&
      stored.expiresAt - EXPIRY_MARGIN_MS > Date.now()
    ) {
      this.token = stored;
    }
  }

  get signedIn(): boolean {
    return this.cachedAccessToken !== undefined;
  }

  get email(): string | undefined {
    return this.signedIn ? this.token?.email : undefined;
  }

  get cachedAccessToken(): string | undefined {
    const t = this.token;
    return t !== undefined && t.expiresAt - EXPIRY_MARGIN_MS > Date.now()
      ? t.accessToken
      : undefined;
  }

  async getAccessToken(): Promise<string> {
    const cached = this.cachedAccessToken;
    if (cached !== undefined) return cached;
    if (this.pending === undefined) {
      this.pending = this.authenticate().finally(() => {
        this.pending = undefined;
      });
    }
    return (await this.pending).accessToken;
  }

  async signIn(): Promise<void> {
    this.signOut();
    await this.getAccessToken();
  }

  signOut(): void {
    const had = this.token !== undefined;
    this.token = undefined;
    writeJson(GoogleRoiStoreAuth.KEY, undefined);
    if (had) this.changed.dispatch();
  }

  invalidate(): void {
    this.signOut();
  }

  private authenticate(): Promise<StoredToken> {
    const { clientId } = this.config;
    if (!clientId) {
      return Promise.reject(
        new Error("No OAuth client id configured for the ROI store"),
      );
    }
    const state = randomState();
    const scopes = this.config.scopes?.length
      ? this.config.scopes
      : DEFAULT_SCOPES;
    const popup = openPopup(
      googleAuthorizeUrl(clientId, scopes, state, redirectUri()),
    );
    return new Promise<StoredToken>((resolve, reject) => {
      let channel: BroadcastChannel | undefined;
      const cleanup = () => {
        window.removeEventListener("message", onMessage);
        channel?.close();
        clearInterval(poll);
      };
      const accept = async (params: Record<string, string>) => {
        if (params.state !== state) return;
        cleanup();
        if (params.error || !params.access_token) {
          reject(new Error(`Sign-in failed: ${params.error ?? "no token"}`));
          return;
        }
        const token: StoredToken = {
          accessToken: params.access_token,
          expiresAt: Date.now() + Number(params.expires_in ?? 3600) * 1000,
        };
        try {
          const info = await fetch(
            "https://openidconnect.googleapis.com/v1/userinfo",
            {
              headers: { Authorization: `Bearer ${token.accessToken}` },
            },
          );
          if (info.ok) token.email = (await info.json()).email;
        } catch {
          // Email is provenance only.
        }
        this.token = token;
        writeJson(GoogleRoiStoreAuth.KEY, token);
        this.changed.dispatch();
        resolve(token);
      };
      const onMessage = (event: MessageEvent) => {
        if (event.origin !== window.location.origin) return;
        const params = event.data?.ngpyOAuth;
        if (params) void accept(params);
      };
      window.addEventListener("message", onMessage);
      try {
        channel = new BroadcastChannel(OAUTH_CHANNEL);
        channel.onmessage = (e) => {
          const params = e.data?.ngpyOAuth;
          if (params) void accept(params);
        };
      } catch {
        channel = undefined;
      }
      const poll = setInterval(() => {
        let closed = false;
        try {
          closed = popup.closed;
        } catch {
          closed = false;
        }
        if (closed) {
          // Give a late BroadcastChannel message a moment before failing.
          setTimeout(() => {
            if (this.token === undefined) {
              cleanup();
              reject(new Error("Sign-in window was closed"));
            }
          }, 1000);
          clearInterval(poll);
        }
      }, 500);
    });
  }
}

export class MiddleAuthRoiStoreAuth implements RoiStoreAuth {
  readonly changed = new Signal();
  private pending: Promise<string> | undefined;

  constructor(private authServer: string) {}

  private get storageKey(): string {
    return `auth_token_v2_${this.authServer}`;
  }

  get cachedAccessToken(): string | undefined {
    const t = readJson(this.storageKey);
    return typeof t?.accessToken === "string" ? t.accessToken : undefined;
  }

  get signedIn(): boolean {
    return this.cachedAccessToken !== undefined;
  }

  get email(): string | undefined {
    return undefined;
  }

  async getAccessToken(): Promise<string> {
    const cached = this.cachedAccessToken;
    if (cached !== undefined) return cached;
    if (this.pending === undefined) {
      this.pending = this.login().finally(() => {
        this.pending = undefined;
      });
    }
    return this.pending;
  }

  async signIn(): Promise<void> {
    this.signOut();
    await this.getAccessToken();
  }

  signOut(): void {
    writeJson(this.storageKey, undefined);
    this.changed.dispatch();
  }

  invalidate(): void {
    this.signOut();
  }

  private login(): Promise<string> {
    const popup = openPopup(`${this.authServer}/api/v1/authorize`);
    return new Promise<string>((resolve, reject) => {
      const onMessage = (event: MessageEvent) => {
        if (event.source !== popup) return;
        const data = event.data;
        if (typeof data?.token !== "string") {
          window.removeEventListener("message", onMessage);
          reject(new Error("Unexpected middleauth response"));
          return;
        }
        window.removeEventListener("message", onMessage);
        writeJson(this.storageKey, {
          tokenType: "Bearer",
          accessToken: data.token,
          url: this.authServer,
          appUrls: Array.isArray(data.app_urls) ? data.app_urls : [],
        });
        this.changed.dispatch();
        resolve(data.token);
      };
      window.addEventListener("message", onMessage);
    });
  }
}

export function makeAuth(config: RoiStoreConfig): RoiStoreAuth {
  if (config.provider === "middleauth") {
    if (!config.authServer) throw new Error("middleauth needs authServer");
    return new MiddleAuthRoiStoreAuth(config.authServer.replace(/\/+$/, ""));
  }
  return new GoogleRoiStoreAuth(config);
}
