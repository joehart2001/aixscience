/* Checks that fail loudly rather than being eyeballed.
 *
 *   node verify.js
 *
 * 1. every metric in the page is parsed back out of summary_test_metrics.md
 * 2. scene count, rail entries and chips agree, and each chip lands in its scene
 * 3. nothing is still revealing when its scene is cut (the "flashing" bug)
 * 4. the cartoon's secondary structure and bonds are what the chemistry demands
 */
const fs = require("fs");
const path = require("path");
const { assignSS } = require("./src/lib/ss.js");

const page = fs.readFileSync(path.join(__dirname, "dist/levels.html"), "utf8");
const src = page.slice(page.indexOf("<script>") + 8, page.lastIndexOf("</script>"));
let fail = 0;
const check = (ok, msg) => { console.log(`  ${ok ? "ok  " : "FAIL"}  ${msg}`); if (!ok) fail++; };

/* 1 ---------------------------------------------------------------- metrics */
/* read a `const NAME={...}` literal by balancing braces, so inserting another
   declaration after it cannot break this */
function literal(name) {
  const at = src.indexOf(`const ${name}={`);
  let i = src.indexOf("{", at), depth = 0;
  for (let j = i; j < src.length; j++) {
    if (src[j] === "{") depth++;
    else if (src[j] === "}" && --depth === 0) return eval("(" + src.slice(i, j + 1) + ")");
  }
  throw new Error(`unterminated ${name}`);
}
const M = literal("M");
const METRICS = path.join(__dirname, "../mlp-embeds/figs/compare/summary_test_metrics.md");
const md = fs.readFileSync(METRICS, "utf8");   // read in place, never copied
const KEY = { "direct (MLP)": "direct", transformer: "transformer", "SE+MLP": "semlp", "SE+XGB": "sexgb", "boltz2 (frozen)": "boltz2" };
let split = null, n = 0, bad = 0;
for (const line of md.split("\n")) {
  const h = line.match(/^##\s+(\S+)/);
  if (h) { split = h[1]; continue; }
  const c = line.split("|").map(s => s.trim()).filter(Boolean);
  if (c.length === 6 && KEY[c[0]])
    c.slice(1).map(Number).forEach((v, i) => { n++; if (Math.abs(v - M[KEY[c[0]]][split][i]) > 1e-9) bad++; });
}
console.log("\nmetrics");
check(n === 100 && bad === 0, `${n} values read from the .md, ${bad} mismatched`);

/* 2 ------------------------------------------------------------------ scenes */
const cuts = eval(src.match(/const CUT=(\[[^\]]+\])/)[1]);
const rail = eval(src.slice(src.indexOf("const RAIL=") + 11, src.indexOf("];", src.indexOf("const RAIL=")) + 1));
const chips = [...page.matchAll(/data-at="([\d.]+)"/g)].map(m => +m[1]);
const bounds = [0, ...cuts, 1];
console.log("\nscenes");
check(cuts.length + 1 === rail.length,
  `${cuts.length + 1} scenes, ${rail.length} rail entries, ${chips.length} chips`);
const landed = chips.map((a) => bounds.findIndex((_, k) => a >= bounds[k] && a < bounds[k + 1]));
check(landed.every((s, i) => i === 0 || s >= landed[i - 1]), `chips run in order: ${landed.join(",")}`);
check(new Set(landed).size === rail.length, `every scene is reachable from a chip`);

/* 3 ------------------------------------------------------------- dwell times */
const DUR = +src.match(/const DUR=(\d+)/)[1];
console.log("\ndwell (each scene ends on a conclusion, which must be readable)");
["s1", "s2", "s3", "s4", "s5"].forEach((name, i) => {
  const from = src.indexOf(`function ${name}(`);
  const body = src.slice(from, src.indexOf("\n/* ---", from));
  const ends = [...body.matchAll(/sub\(p,[\d.]+,([\d.]+)\)/g)].map(m => +m[1]);
  const strip = body.match(/(?:result|mace)Strip\([^;]*?,\s*(\.\d+)\s*[,)]/);
  if (strip) ends.push(+strip[1] + 0.26);         // strip ends on its conclusion line
  const last = Math.max(...ends, 0);
  const dwell = (1 - last) * (bounds[i + 1] - bounds[i]) * DUR / 1000;
  check(dwell >= 2.0, `${name} last reveal at p=${last.toFixed(2)} -> ${dwell.toFixed(1)}s on screen`);
});

/* 4 ----------------------------------------------------------------- cartoon */
const B = JSON.parse(src.match(/const BOLTZ=(\{.*?\});/s)[1]);
const ss = assignSS(B.tr);
const runs = [];
let s0 = 0;
for (let i = 1; i <= ss.length; i++) if (i === ss.length || ss[i] !== ss[s0]) { runs.push([ss[s0], s0 + 1, i]); s0 = i; }
const helices = runs.filter(r => r[0] === "H");
const strands = runs.filter(r => r[0] === "E");
console.log("\ncartoon");
check(helices.length === 2, `${helices.length} helices (class I platform has 2: ${helices.map(h => h[1] + "-" + h[2]).join(", ")})`);
check(strands.length >= 7, `${strands.length} strands in the sheet floor`);

const at = B.pep.map(([ri, el, x, y, z]) => ({ ri, x, y, z }));
const adj = at.map(() => []);
let bonds = 0, inter = 0;
for (let i = 0; i < at.length; i++) for (let j = i + 1; j < at.length; j++) {
  const d = Math.hypot(at[i].x - at[j].x, at[i].y - at[j].y, at[i].z - at[j].z);
  if (d < 1.95) { bonds++; if (at[i].ri !== at[j].ri) inter++; adj[i].push(j); adj[j].push(i); }
}
const seen = new Set([0]), stack = [0];
while (stack.length) { const v = stack.pop(); for (const w of adj[v]) if (!seen.has(w)) { seen.add(w); stack.push(w); } }
check(inter === 8, `${inter} inter-residue bonds (a 9-mer has 8 peptide bonds)`);
check(seen.size === at.length, `all ${at.length} peptide atoms in one connected chain (${bonds} bonds)`);

/* 5 ------------------------------------------------------------ MACE blocks */
const MACE = literal("MACE");
const ablation = fs.readFileSync(path.join(__dirname, "src/data/mace_feature_blocks.md"), "utf8");
const cell = (blocks, col) => {
  const row = ablation.split("\n").find(l => l.split("|")[1]?.trim() === blocks);
  if (!row) throw new Error(`no row for ${blocks}`);
  return parseFloat(row.split("|")[col].trim().split(" ")[0]);
};
console.log("\nMACE ablation (shown on screen vs src/data/mace_feature_blocks.md)");
[["boltz", "boltz", 5, MACE.boltz], ["boltz+node", "boltzNode", 5, MACE.boltzNode],
 ["node", "node", 5, MACE.node], ["node train", "nodeTrain", 3, MACE.nodeTrain]]
  .forEach(([label, , col, shown]) => {
    const blocks = label.split(" ")[0];
    const want = cell(blocks, col);
    check(Math.abs(want - shown) < 1e-9, `${label} = ${shown.toFixed(3)} (table says ${want.toFixed(3)})`);
  });
check(MACE.boltzNode < MACE.boltz,
  `adding node features lowers the score, ${MACE.boltz.toFixed(3)} -> ${MACE.boltzNode.toFixed(3)}`);
check(Math.abs(MACE.node) < 0.05 && MACE.nodeTrain > 0.7,
  `MACE alone memorises: train ${MACE.nodeTrain.toFixed(3)}, test ${MACE.node.toFixed(3)}`);

/* 6 ------------------------------------------------- does the page even run? */
console.log("\nruntime");
const names = [...src.matchAll(/^function (s\d+)\(/gm)].map(m => m[1]);
check(new Set(names).size === names.length, `scene functions are unique: ${names.join(", ")}`);
const listed = src.match(/const SCENES=\[([^\]]+)\]/)[1].split(",").map(x => x.trim());
check(listed.every(n => names.includes(n)) && listed.length === names.length,
  `SCENES matches the definitions: ${listed.join(", ")}`);
try {
  const ctx = new Proxy({}, { get: (_, k) => k === "canvas"
      ? { parentElement: { clientWidth: 1000 } }
      : (k === "measureText" ? () => ({ width: 10 }) : () => {}), set: () => true });
  const el = () => ({ getContext: () => ctx, parentElement: { clientWidth: 1000 },
    style: {}, setAttribute: () => {}, addEventListener: () => {}, innerHTML: "" });
  global.window = global; global.devicePixelRatio = 1;
  global.document = { getElementById: el, querySelectorAll: () => [], documentElement: {} };
  global.getComputedStyle = () => ({ getPropertyValue: () => "#888888" });
  global.matchMedia = () => ({ matches: false, addEventListener: () => {} });
  global.ResizeObserver = class { observe() {} };
  let n = 0; global.requestAnimationFrame = (f) => { if (n++ < 3) f(n * 16); };
  eval(src);
  check(true, "bundle evaluates and renders without throwing");
} catch (e) {
  check(false, `bundle threw: ${e.message}`);
}

/* 7 ----------------------------------------------------- the Boltz embedding */
const EMB = literal("EMB");
console.log("\nBoltz-2 embedding (the bars in the Boltz scene)");
check(EMB.n === 1547, `${EMB.n} dimensions, as the page claims`);
check(EMB.labels.reduce((a, [, n]) => a + n, 0) + 128 + 2 + 9 === EMB.n ||
      EMB.labels.length >= 3, `${EMB.labels.length} named blocks shown`);
const uniq = new Set(EMB.z.map((x) => Math.round(x * 1000))).size;
check(uniq > EMB.z.length * 0.9, `${uniq}/${EMB.z.length} distinct values (real data, not a waveform)`);
check(!/Math\.sin\(i\*\.41\)/.test(src), "no synthetic sine-wave vector left in the Boltz scene");

/* 8 ------------------------------------------------------------ UMAP panels */
const UMAP = literal("UMAP");
console.log("\nUMAP point clouds");
for (const tag of ["boltz", "node", "edge"]) {
  const d = UMAP[tag];
  const n = d.s.length;
  check(d.xy.length === n * 2, `${tag}: ${n} points, ${d.xy.length} coordinates`);
  const counts = { t: 0, v: 0, e: 0 };
  for (const c of d.s) counts[c]++;
  check(counts.t > 0 && counts.v > 0 && counts.e > 0,
    `${tag}: train ${counts.t}, val ${counts.v}, test ${counts.e}`);
  const uniq = new Set(d.xy).size;
  check(uniq > 400, `${tag}: ${uniq} distinct coordinate values (real embedding, not generated)`);
}
const spread = ["boltz", "node", "edge"].map((t) => {
  const d = UMAP[t], te = [];
  for (let i = 0; i < d.s.length; i++) if (d.s[i] === "e") te.push(d.xy[i * 2]);
  return Math.round(Math.max(...te) - Math.min(...te));
});
check(spread.every((x) => x > 50), `test points are spread, not collapsed: x-range ${spread.join(", ")}`);

console.log(fail ? `\n${fail} FAILED\n` : "\nall checks passed\n");
process.exit(fail ? 1 : 0);
