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
 * @file The Python tab: an editor, a Run button and the console.
 *
 * A script gets FULL control of the viewer (and of anything the page can
 * fetch), so only a `?script=` URL on this page's own origin runs without
 * asking; a pasted, opened or dropped script asks first.
 */

import { button, h, setStatus } from "../ui/dom.js";
import type { PythonClient } from "./client.js";

export class PythonPanel {
  readonly element = h("div", { class: "ngpy-panel ngpy-python" });
  readonly editor = h("textarea", {
    class: "ngpy-editor",
    spellcheck: false,
    placeholder:
      "import neuroglancer\nviewer = neuroglancer.Viewer()\nwith viewer.txn() as s:\n    s.layout = '3d'",
  }) as HTMLTextAreaElement;
  private console = h("pre", { class: "ngpy-console" });
  private status = h("div", { class: "ngpy-status" });
  private runButton: HTMLButtonElement;
  private fileInput = h("input", {
    type: "file",
    accept: ".py,text/x-python,text/plain",
    style: { display: "none" },
  });
  private trusted = false;

  constructor(
    private python: PythonClient,
    private beforeRun: () => Promise<void>,
    private demoScript: string,
  ) {
    this.runButton = button("Run ▶", () => void this.run(), {
      class: "ngpy-primary",
      title: "Ctrl+Enter",
    });
    this.editor.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        void this.run();
      }
      if (e.key === "Tab") {
        e.preventDefault();
        this.editor.setRangeText(
          "    ",
          this.editor.selectionStart,
          this.editor.selectionEnd,
          "end",
        );
      }
    });
    this.editor.addEventListener("input", () => (this.trusted = false));
    this.editor.addEventListener("dragover", (e) => e.preventDefault());
    this.editor.addEventListener("drop", (e) => {
      e.preventDefault();
      const file = e.dataTransfer?.files?.[0];
      if (file) void file.text().then((t) => this.load(t, false));
    });
    this.fileInput.addEventListener("change", () => {
      const file = this.fileInput.files?.[0];
      if (file) void file.text().then((t) => this.load(t, false));
      this.fileInput.value = "";
    });
    this.element.append(
      h(
        "div",
        { class: "ngpy-row" },
        this.runButton,
        button("Open .py…", () => this.fileInput.click()),
        button("Load demo", () => this.load(this.demoScript, true)),
        button("Clear console", () => (this.console.textContent = "")),
        this.fileInput,
      ),
      this.editor,
      this.status,
      this.console,
    );
    python.output.add((stream, text) => this.print(text + "\n", stream));
    python.progress.add((message) => setStatus(this.status, message, "busy"));
    python.statusChanged.add(() => this.renderStatus());
    this.renderStatus();
  }

  private renderStatus() {
    const p = this.python;
    this.runButton.disabled = p.status !== "ready";
    if (p.status === "ready") {
      const i = p.info ?? {};
      setStatus(
        this.status,
        `Python ${i.python} · numpy ${i.numpy} · zarr ${i.zarr} · zarr-vectors ${i.zarr_vectors} · JSPI ${p.jspi ? "yes" : "no (ZVF export unavailable)"}`,
        "ok",
      );
    } else if (p.status === "failed") {
      setStatus(this.status, `Python failed to start: ${p.error}`, "error");
    }
  }

  print(text: string, stream: "stdout" | "stderr" | "info" = "stdout") {
    const span = h("span", { class: `ngpy-out-${stream}` }, text);
    this.console.append(span);
    this.console.scrollTop = this.console.scrollHeight;
  }

  /** Put a script in the editor; `trusted` scripts run without a prompt. */
  load(code: string, trusted: boolean) {
    this.editor.value = code;
    this.trusted = trusted;
  }

  async run(): Promise<void> {
    const code = this.editor.value;
    if (!code.trim()) return;
    if (
      !this.trusted &&
      !confirm(
        "Run this Python script? It gets full control of the viewer and can fetch anything this page can.",
      )
    ) {
      return;
    }
    this.trusted = true;
    this.print(">>> running script\n", "info");
    try {
      await this.beforeRun();
      const result = await this.python.run(code, "<ngpy-script>");
      if (result !== undefined && result !== null && result !== "")
        this.print(`${result}\n`);
      this.print(">>> done\n", "info");
    } catch (e) {
      this.print(`${(e as Error).message}\n`, "stderr");
    }
  }
}
