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

/** @file A few DOM helpers; ngpy has no UI framework dependency. */

type Child = Node | string | number | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, any> = {},
  ...children: (Child | Child[])[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null) continue;
    if (key === "class") el.className = String(value);
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key in el && typeof value !== "string") (el as any)[key] = value;
    else el.setAttribute(key, String(value));
  }
  append(el, children);
  return el;
}

export function append(parent: Node, children: (Child | Child[])[]) {
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function clear(el: Element) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function select(
  options: { value: string; label: string }[],
  current: string | undefined,
  onChange: (value: string) => void,
  props: Record<string, any> = {},
): HTMLSelectElement {
  const s = h("select", props);
  for (const o of options) {
    const opt = h("option", { value: o.value }, o.label);
    if (o.value === current) opt.selected = true;
    s.appendChild(opt);
  }
  s.addEventListener("change", () => onChange(s.value));
  return s;
}

export function button(label: string, onClick: () => void, props: Record<string, any> = {}) {
  return h("button", { type: "button", onclick: onClick, ...props }, label);
}

export function field(label: string, control: Node, hint?: string): HTMLElement {
  return h(
    "label",
    { class: "ngpy-field" },
    h("span", { class: "ngpy-field-label" }, label),
    control,
    hint ? h("span", { class: "ngpy-hint" }, hint) : null,
  );
}

export function section(title: string, ...children: (Child | Child[])[]): HTMLElement {
  return h("section", { class: "ngpy-section" }, h("h3", {}, title), ...children);
}

export function setStatus(el: HTMLElement, text: string, kind: "" | "ok" | "error" | "busy" = "") {
  el.textContent = text;
  el.dataset.kind = kind;
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: filename, style: { display: "none" } });
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 1000);
}
