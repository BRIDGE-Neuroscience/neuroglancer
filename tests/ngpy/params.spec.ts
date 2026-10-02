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

/** Page parameters, the wrapper's URL state, and iframe URL building. */

import { describe, expect, it } from "vitest";
import {
  isSameOrigin,
  viewerUrlWithState,
} from "../../ngpy/src/host/viewer_host.js";
import {
  buildWrapperHash,
  DEFAULT_NG_URL,
  parseParams,
  parseWrapperHash,
  resolveScriptUrl,
} from "../../ngpy/src/params.js";

describe("parseParams", () => {
  it("defaults to a sibling index.html and the CDN Pyodide", () => {
    const p = parseParams("");
    expect(p.ngUrl).toBe(DEFAULT_NG_URL);
    expect(p.pyodideIndexUrl).toBeUndefined();
    expect(p.startPython).toBe(true);
  });

  it("reads ng / pyodide / script / roiStore / python", () => {
    const p = parseParams(
      "?ng=/client/index.html&pyodide=/py/&script=examples/x.py&roiStore=%7B%22bucket%22%3A%22b%22%7D&python=0",
    );
    expect(p).toEqual({
      ngUrl: "/client/index.html",
      pyodideIndexUrl: "/py/",
      scriptUrl: "examples/x.py",
      roiStore: '{"bucket":"b"}',
      startPython: false,
    });
  });
});

describe("the wrapper hash", () => {
  const viewer = {
    layers: [
      {
        name: "tracts",
        type: "segmentation",
        segments: ["1", "2"],
        segmentColors: { "1": "#f00" },
        source: "s",
      },
      { name: "img", type: "image", source: "t" },
    ],
    layout: "3d",
  };

  it("round-trips viewer + ngpy state and drops the managed layer's segments", () => {
    const hash = buildWrapperHash(viewer, { filter: { groups: [] } }, "tracts");
    const back = parseWrapperHash(hash);
    expect(back.ngpy).toEqual({ filter: { groups: [] } });
    expect(back.viewer.layers[0]).toEqual({
      name: "tracts",
      type: "segmentation",
      source: "s",
    });
    expect(back.viewer.layers[1]).toEqual(viewer.layers[1]);
    expect(back.viewer.layout).toBe("3d");
  });

  it("keeps segments when no layer is filter-managed", () => {
    const back = parseWrapperHash(
      buildWrapperHash(viewer, undefined, undefined),
    );
    expect(back.viewer.layers[0].segments).toEqual(["1", "2"]);
  });

  it("accepts a plain Neuroglancer #! link and ignores junk", () => {
    const plain = `#!${encodeURIComponent(JSON.stringify({ layout: "xy" }))}`;
    expect(parseWrapperHash(plain)).toEqual({
      viewer: { layout: "xy" },
      ngpy: undefined,
    });
    expect(parseWrapperHash("#!not-json")).toEqual({
      viewer: undefined,
      ngpy: undefined,
    });
    expect(parseWrapperHash("")).toEqual({
      viewer: undefined,
      ngpy: undefined,
    });
  });
});

describe("?script=", () => {
  const page = "http://localhost:8000/ngpy.html?script=examples/a.py";

  it("resolves relative to the page (from / as well) and stays same-origin", () => {
    expect(resolveScriptUrl("examples/a.py", page).href).toBe(
      "http://localhost:8000/examples/a.py",
    );
    expect(
      resolveScriptUrl("/s.py", "http://localhost:8000/?script=/s.py").href,
    ).toBe("http://localhost:8000/s.py");
    expect(() => resolveScriptUrl("https://evil.example/x.py", page)).toThrow(
      /origin/,
    );
  });
});

describe("hosting", () => {
  it("knows same-origin from cross-origin builds", () => {
    expect(isSameOrigin("./index.html", "http://a.test/ngpy.html")).toBe(true);
    expect(isSameOrigin("https://b.test/", "http://a.test/ngpy.html")).toBe(
      false,
    );
  });

  it("puts a state in the iframe URL as a #! hash", () => {
    const url = viewerUrlWithState("./index.html#old", { layout: "xy" });
    expect(url).toBe(`./index.html#!${encodeURIComponent('{"layout":"xy"}')}`);
    expect(viewerUrlWithState("./index.html")).toBe("./index.html");
  });
});
