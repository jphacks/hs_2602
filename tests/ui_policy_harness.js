// 画面(app/static/index.html)のスクリプトを、偽の DOM の上で node の vm で動かし、介入の設定の処理を確かめる。
// 使い方: node tests/ui_policy_harness.js <index.html> <ui_spec.json>  → 各手順の結果を JSON で標準出力に出す
"use strict";
const fs = require("fs");
const vm = require("vm");

const [html, specPath] = process.argv.slice(2);
const script = [...fs.readFileSync(html, "utf8").matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];
const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));

// どのプロパティを読んでも動く偽の要素(書いた値はそのまま読める)
function fakeEl() {
  const store = {
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    style: {}, dataset: {}, value: "", checked: false, textContent: "", innerHTML: "",
  };
  return new Proxy(store, {
    get(t, k) {
      if (k in t) return t[k];
      if (k === "querySelectorAll") return () => [];
      if (typeof k === "symbol") return undefined;
      return () => fakeEl();
    },
    set(t, k, v) { t[k] = v; return true; },
  });
}
const els = {};
const document = {
  getElementById: (id) => (els[id] ||= fakeEl()),
  querySelectorAll: () => [],
  querySelector: () => fakeEl(),
  addEventListener() {},
};
const ctx = {
  document, console,
  window: { scrollTo() {}, addEventListener() {} },
  location: { host: "test" },
  fetch: () => new Promise(() => {}),  // /api/info などは返さない(テストで ui.info を直接入れる)
  WebSocket: class { constructor() {} },
  setTimeout: () => 0, setInterval: () => 0, clearTimeout() {}, alert() {}, confirm: () => true,
  requestAnimationFrame: () => 0, matchMedia: () => ({ matches: false, addEventListener() {} }),
};
vm.createContext(ctx);
vm.runInContext(script + "\n;globalThis.__t = { ui, syncSetup, setPolicy, choosePreset, policyValue, policyPayload, presetName };", ctx);
const t = ctx.__t;

const keys = spec.items.map((it) => it.key);
const values = () => Object.fromEntries(keys.map((k) => [k, t.policyValue(k)]));
const snap = (label) => ({
  label, preset: t.presetName(), values: values(), payload: t.policyPayload(),
  notice: els["policy-notice"]?.textContent || "",
});
const setPurpose = (p) => { t.ui.purpose = p; t.syncSetup(); };  // スタイルのボタンと同じ(値を変えて画面を更新)

const out = [];
t.ui.info = { policy: spec, discussion_types: [], llm: "mock", default_mode: "text" };
setPurpose("job");
out.push(snap("job_default"));
t.setPolicy("stall", "high");
out.push(snap("job_stall_high"));
t.setPolicy("stall", "standard");  // 初期値と同じ値に戻すと「未変更」
out.push(snap("job_stall_back"));
t.setPolicy("content_scope", "hint");
out.push(snap("job_content_hint"));  // 就活GDで内容支援 → 注意を表示
setPurpose("general");  // プリセットを用途に任せている → GW用に切り替わり、変更は初期値に戻る
out.push(snap("general_default"));
t.setPolicy("organize", "low");
out.push(snap("general_organize_low"));
setPurpose("general");  // 同じスタイルを押し直しても変更は残る
out.push(snap("general_same_purpose"));
t.setPolicy("content_scope", "none");
out.push(snap("general_scope_none"));  // ⑥-B は なし に固定
t.setPolicy("content_scope", "hint");
out.push(snap("general_scope_back"));
t.choosePreset("job_gd");  // プリセットを切り替えると変更は初期値に戻る
out.push(snap("choose_job_gd"));
t.setPolicy("turns", "high");
setPurpose("job");  // プリセットを選んだ後は、スタイルを変えてもプリセットと変更を保つ
out.push(snap("pinned_purpose_change"));
t.choosePreset("general_gw");
out.push(snap("job_with_gw_preset"));  // 就活のスタイルで GW用 → ヒントまで → 注意を表示
process.stdout.write(JSON.stringify(out));
