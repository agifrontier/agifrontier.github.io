const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../../assets/js/search-setup.js"), "utf8");

function fixture({ platform = "Linux", failures = [], emptyIndex = false } = {}) {
  const appended = [];
  const listeners = new Map();
  const errors = [];
  const failedModules = new Set();
  const label = { firstChild: { textContent: "ctrl k " } };
  const button = { attributes: {}, setAttribute(key, value) { this.attributes[key] = value; }, querySelector: () => label };
  const status = { hidden: true, textContent: "" };
  let defined = false;
  let opens = 0;
  let empty = emptyIndex;
  const ninja = {
    data: [],
    classList: { add() {}, remove() {}, toggle() {} },
    updateComplete: Promise.resolve(),
    open() { assert.ok(defined, "component must load before opening"); opens += 1; },
  };
  const document = {
    baseURI: "https://example.test/",
    currentScript: { dataset: { searchComponent: "/component.js", searchIndex: "/index.js" } },
    getElementById: (id) => id === "search-toggle" ? button : status,
    querySelector: (selector) => selector === "ninja-keys" ? ninja : label,
    addEventListener: (name, handler) => listeners.set(name, handler),
    createElement: () => ({ remove() { this.removed = true; } }),
    head: { appendChild(script) {
      appended.push(script);
      queueMicrotask(() => {
        const resource = new URL(script.src, document.baseURI).pathname;
        if (failedModules.has(script.src)) {
          script.onerror();
          return;
        }
        if (failures[0] === resource) {
          failures.shift();
          if (script.type === "module") failedModules.add(script.src);
          script.onerror();
          return;
        }
        if (resource === "/component.js") defined = true;
        if (resource === "/index.js") {
          ninja.data = empty ? [] : [{ id: "all-articles", title: "全部文章" }];
          empty = false;
        }
        script.onload();
      });
    } },
  };
  const context = vm.createContext({
    document, navigator: { platform }, setTimeout, clearTimeout, URL,
    customElements: { get: () => defined ? function NinjaKeys() {} : undefined },
    determineComputedTheme: () => "light",
    $: () => ({ hasClass: () => false, collapse() {} }),
    console: { error: (...args) => errors.push(args) },
  });
  context.window = context;
  vm.runInContext(source, context);
  return {
    appended, errors, button, status, label, ninja,
    open: () => vm.runInContext("openSearchModal()", context),
    key(event) { listeners.get("keydown")?.({ preventDefault() {}, stopImmediatePropagation() {}, ...event }); },
    get opens() { return opens; },
  };
}

test("initialization does not request search assets", () => {
  const page = fixture();
  assert.equal(page.appended.length, 0);
  assert.equal(page.opens, 0);
});

test("first activation loads component then complete index", async () => {
  const page = fixture();
  await page.open();
  assert.deepEqual(page.appended.map(item => new URL(item.src, "https://example.test").pathname), ["/component.js", "/index.js"]);
  assert.equal(page.ninja.data[0].id, "all-articles");
  assert.equal(page.opens, 1);
  assert.equal(page.status.hidden, true);
});

test("concurrent activations share loading and open once", async () => {
  const page = fixture();
  await Promise.all([page.open(), page.open(), page.open()]);
  assert.equal(page.appended.length, 2);
  assert.equal(page.opens, 1);
});

test("subsequent activation reuses loaded assets", async () => {
  const page = fixture();
  await page.open();
  await page.open();
  assert.equal(page.appended.length, 2);
  assert.equal(page.opens, 2);
});

for (const failedUrl of ["/component.js", "/index.js"]) {
  test(`failed ${failedUrl} is visible and can be retried`, async () => {
    const page = fixture({ failures: [failedUrl] });
    assert.equal(await page.open(), false);
    assert.equal(page.status.hidden, false);
    assert.match(page.status.textContent, /失败.*重试/);
    assert.equal(page.errors.length, 1);
    await page.open();
    assert.equal(page.opens, 1);
    assert.equal(page.status.hidden, true);
    assert.equal(page.appended.filter(item => new URL(item.src, "https://example.test").pathname === failedUrl).length, 2);
  });
}

test("empty index does not silently open an empty search", async () => {
  const page = fixture({ emptyIndex: true });
  assert.equal(await page.open(), false);
  assert.equal(page.opens, 0);
  assert.equal(page.status.hidden, false);
  await page.open();
  assert.equal(page.opens, 1);
});

test("Escape during loading cancels automatic opening", async () => {
  const page = fixture();
  const ready = page.open();
  page.key({ key: "Escape" });
  await ready;
  assert.equal(page.opens, 0);
  await page.open();
  assert.equal(page.opens, 1);
});

test("Ctrl+K and Command+K trigger lazy search", async () => {
  for (const modifier of ["ctrlKey", "metaKey"]) {
    const page = fixture();
    page.key({ key: "k", [modifier]: true });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(page.opens, 1);
  }
});

test("plain K does not load search and Mac label is preserved", () => {
  const page = fixture({ platform: "MacIntel" });
  page.key({ key: "k" });
  assert.equal(page.appended.length, 0);
  assert.match(page.label.firstChild.textContent, /⌘/);
});
