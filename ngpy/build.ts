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
 * @file Builds dist/ngpy/ngpy.html: ONE self-contained file.
 *
 *   node ngpy/build.ts [--out dist/ngpy] [--zarr-vectors <wheel|package dir>]
 *                      [--no-minify] [--offline]
 *
 * 1. The Python payload zip: ngpy/python/ngpy -> `ngpy/`, the vendored
 *    upstream subset ngpy/python/vendor/neuroglancer -> `neuroglancer/`, and
 *    zarr-vectors (the released 0.9.2 wheel from PyPI, sha256-pinned and
 *    cached under <out>/.cache, or `--zarr-vectors`) -> `zarr_vectors/`.
 * 2. The Pyodide worker, bundled by esbuild into a string.
 * 3. The page, bundled by esbuild with a virtual `ngpy:embedded` module
 *    carrying (1) as base64, (2), and the demo script.
 * 4. HTML with the CSS and the module script inline.
 *
 * Kept to erasable TypeScript so Node runs it directly.
 */

/// <reference types="node" />

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import zlib from "node:zlib";
import esbuild from "esbuild";

const ROOT = path.resolve(import.meta.dirname, "..");
const NGPY = path.join(ROOT, "ngpy");

const ZARR_VECTORS_WHEEL = {
  name: "zarr_vectors-0.9.2-py3-none-any.whl",
  url:
    "https://files.pythonhosted.org/packages/cf/4b/" +
    "bab59bada24f36ae91a8a1f86bdb68bc9269adfff3527477e93391a134cb/" +
    "zarr_vectors-0.9.2-py3-none-any.whl",
  sha256: "fdf909ee98b5e6928a851332067c6ff6eedb83fdd74325e1aef50d120fb0a2af",
  version: "0.9.2",
};

interface Options {
  out: string;
  zarrVectors: string | undefined;
  minify: boolean;
  offline: boolean;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    out: path.join(ROOT, "dist", "ngpy"),
    zarrVectors: undefined,
    minify: true,
    offline: false,
  };
  for (let i = 0; i < argv.length; ++i) {
    const a = argv[i];
    if (a === "--out") opts.out = path.resolve(argv[++i]);
    else if (a === "--zarr-vectors") opts.zarrVectors = path.resolve(argv[++i]);
    else if (a === "--no-minify") opts.minify = false;
    else if (a === "--offline") opts.offline = true;
    else if (a === "--help" || a === "-h") {
      console.log(
        "node ngpy/build.ts [--out DIR] [--zarr-vectors WHEEL|DIR] [--no-minify] [--offline]",
      );
      process.exit(0);
    } else throw new Error(`unknown argument ${a}`);
  }
  return opts;
}

// -- zip ----------------------------------------------------------------------

interface ZipEntry {
  name: string;
  data: Uint8Array;
}

/** Read every file entry of a zip (stored or deflated). */
function readZip(buf: Buffer): ZipEntry[] {
  let eocd = buf.length - 22;
  while (eocd >= 0 && buf.readUInt32LE(eocd) !== 0x06054b50) --eocd;
  if (eocd < 0) throw new Error("not a zip file");
  const count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  const out: ZipEntry[] = [];
  for (let i = 0; i < count; ++i) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error("bad central directory");
    const method = buf.readUInt16LE(p + 10);
    const compSize = buf.readUInt32LE(p + 20);
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    const local = buf.readUInt32LE(p + 42);
    const name = buf.toString("utf8", p + 46, p + 46 + nameLen);
    p += 46 + nameLen + extraLen + commentLen;
    if (name.endsWith("/")) continue;
    const lNameLen = buf.readUInt16LE(local + 26);
    const lExtraLen = buf.readUInt16LE(local + 28);
    const start = local + 30 + lNameLen + lExtraLen;
    const raw = buf.subarray(start, start + compSize);
    const data =
      method === 0 ? raw : method === 8 ? zlib.inflateRawSync(raw) : undefined;
    if (data === undefined) throw new Error(`unsupported zip method ${method} for ${name}`);
    out.push({ name, data: new Uint8Array(data) });
  }
  return out;
}

/** Write a deflated zip with fixed timestamps (reproducible). */
function writeZip(entries: ZipEntry[]): Buffer {
  const parts: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  const DOS_TIME = 0;
  const DOS_DATE = (0 << 9) | (1 << 5) | 1; // 1980-01-01
  for (const e of [...entries].sort((a, b) => (a.name < b.name ? -1 : 1))) {
    const name = Buffer.from(e.name, "utf8");
    const crc = zlib.crc32(e.data);
    const comp = zlib.deflateRawSync(e.data, { level: 9 });
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6); // UTF-8 names
    local.writeUInt16LE(8, 8);
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(comp.length, 18);
    local.writeUInt32LE(e.data.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);
    parts.push(local, name, comp);
    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(20, 4);
    cd.writeUInt16LE(20, 6);
    cd.writeUInt16LE(0x0800, 8);
    cd.writeUInt16LE(8, 10);
    cd.writeUInt16LE(DOS_TIME, 12);
    cd.writeUInt16LE(DOS_DATE, 14);
    cd.writeUInt32LE(crc, 16);
    cd.writeUInt32LE(comp.length, 20);
    cd.writeUInt32LE(e.data.length, 24);
    cd.writeUInt16LE(name.length, 28);
    cd.writeUInt32LE(0o644 << 16, 38);
    cd.writeUInt32LE(offset, 42);
    central.push(cd, name);
    offset += local.length + name.length + comp.length;
  }
  const cdBuf = Buffer.concat(central);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cdBuf.length, 12);
  eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...parts, cdBuf, eocd]);
}

// -- payload --------------------------------------------------------------------

function walk(dir: string, prefix: string, keep: (rel: string) => boolean): ZipEntry[] {
  const out: ZipEntry[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "__pycache__" || entry.name === "tests") continue;
    const full = path.join(dir, entry.name);
    const rel = `${prefix}${entry.name}`;
    if (entry.isDirectory()) out.push(...walk(full, `${rel}/`, keep));
    else if (keep(rel)) out.push({ name: rel, data: new Uint8Array(fs.readFileSync(full)) });
  }
  return out;
}

async function zarrVectorsEntries(opts: Options): Promise<{ entries: ZipEntry[]; version: string }> {
  if (opts.zarrVectors !== undefined && fs.statSync(opts.zarrVectors).isDirectory()) {
    return {
      entries: walk(opts.zarrVectors, "zarr_vectors/", (rel) => rel.endsWith(".py") || rel.endsWith(".typed")),
      version: `local:${opts.zarrVectors}`,
    };
  }
  let wheelPath = opts.zarrVectors;
  if (wheelPath === undefined) {
    const cache = path.join(opts.out, ".cache");
    wheelPath = path.join(cache, ZARR_VECTORS_WHEEL.name);
    if (!fs.existsSync(wheelPath)) {
      if (opts.offline) throw new Error(`--offline and no cached ${wheelPath}; pass --zarr-vectors`);
      console.log(`downloading ${ZARR_VECTORS_WHEEL.url}`);
      const response = await fetch(ZARR_VECTORS_WHEEL.url);
      if (!response.ok) throw new Error(`HTTP ${response.status} downloading zarr-vectors`);
      fs.mkdirSync(cache, { recursive: true });
      fs.writeFileSync(wheelPath, Buffer.from(await response.arrayBuffer()));
    }
    const digest = createHash("sha256").update(fs.readFileSync(wheelPath)).digest("hex");
    if (digest !== ZARR_VECTORS_WHEEL.sha256) {
      fs.rmSync(wheelPath);
      throw new Error(`zarr-vectors wheel sha256 mismatch (${digest})`);
    }
  }
  const entries = readZip(fs.readFileSync(wheelPath)).filter(
    (e) => !e.name.endsWith(".pyc") && !e.name.includes("__pycache__"),
  );
  const version = /zarr_vectors-([^-]+)-/.exec(path.basename(wheelPath))?.[1] ?? "?";
  return { entries, version };
}

async function buildPayload(opts: Options): Promise<{ zip: Buffer; zarrVectors: string; files: number }> {
  const py = (rel: string) => rel.endsWith(".py") || rel.endsWith("py.typed");
  const entries = [
    ...walk(path.join(NGPY, "python", "ngpy"), "ngpy/", py),
    ...walk(path.join(NGPY, "python", "vendor", "neuroglancer"), "neuroglancer/", py),
  ];
  const zv = await zarrVectorsEntries(opts);
  entries.push(...zv.entries);
  return { zip: writeZip(entries), zarrVectors: zv.version, files: entries.length };
}

// -- bundles --------------------------------------------------------------------

async function bundle(entry: string, opts: Options, plugins: esbuild.Plugin[] = []): Promise<string> {
  const result = await esbuild.build({
    entryPoints: [entry],
    bundle: true,
    write: false,
    format: "esm",
    platform: "browser",
    target: "es2022",
    minify: opts.minify,
    legalComments: "none",
    plugins,
    logLevel: "warning",
  });
  return result.outputFiles[0].text;
}

function embeddedPlugin(values: Record<string, unknown>): esbuild.Plugin {
  return {
    name: "ngpy-embedded",
    setup(build) {
      build.onResolve({ filter: /^ngpy:embedded$/ }, () => ({
        path: "ngpy:embedded",
        namespace: "ngpy-embedded",
      }));
      build.onLoad({ filter: /.*/, namespace: "ngpy-embedded" }, () => ({
        contents: Object.entries(values)
          .map(([k, v]) => `export const ${k} = ${JSON.stringify(v)};`)
          .join("\n"),
        loader: "js",
      }));
    },
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  fs.mkdirSync(opts.out, { recursive: true });
  const payload = await buildPayload(opts);
  const worker = await bundle(path.join(NGPY, "src", "python", "worker.ts"), opts);
  const demo = fs.readFileSync(path.join(NGPY, "examples", "hcp1065_demo.py"), "utf8");
  const buildInfo = {
    version: "0.1.0",
    builtAt: new Date().toISOString(),
    zarrVectors: payload.zarrVectors,
  };
  const page = await bundle(path.join(NGPY, "src", "main.ts"), opts, [
    embeddedPlugin({
      WORKER_SOURCE: worker,
      PAYLOAD_BASE64: payload.zip.toString("base64"),
      DEMO_SCRIPT: demo,
      BUILD_INFO: buildInfo,
    }),
  ]);
  const css = fs.readFileSync(path.join(NGPY, "src", "ui", "style.css"), "utf8");
  const html =
    "<!doctype html>\n" +
    '<html lang="en">\n<head>\n<meta charset="utf-8">\n' +
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n' +
    "<title>ngpy — Neuroglancer + Python</title>\n" +
    `<style>\n${css}</style>\n</head>\n<body>\n<div id="ngpy-root"></div>\n` +
    `<script type="module">${page.replace(/<\/script/gi, "<\\/script")}</script>\n` +
    "</body>\n</html>\n";
  const outFile = path.join(opts.out, "ngpy.html");
  fs.writeFileSync(outFile, html);
  const examplesOut = path.join(opts.out, "examples");
  fs.mkdirSync(examplesOut, { recursive: true });
  for (const f of fs.readdirSync(path.join(NGPY, "examples"))) {
    fs.copyFileSync(path.join(NGPY, "examples", f), path.join(examplesOut, f));
  }
  const mb = (n: number) => `${(n / 1e6).toFixed(2)} MB`;
  console.log(
    `${path.relative(ROOT, outFile)}: ${mb(Buffer.byteLength(html))} ` +
      `(payload zip ${mb(payload.zip.length)}, ${payload.files} files, ` +
      `zarr-vectors ${payload.zarrVectors}; worker ${mb(worker.length)}; page ${mb(page.length - payload.zip.length * 4 / 3)})`,
  );
}

await main();
