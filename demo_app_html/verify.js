/* Checks that fail loudly rather than being eyeballed.
 *
 *   node verify.js
 *
 * 1. every metric in the page is parsed back out of summary_all75_test_metrics.md
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
const M = eval("(" + src.slice(src.indexOf("const M={") + 8, src.indexOf("const MODELS=")).trim().replace(/;$/, "") + ")");
const METRICS = path.join(__dirname, "../mlp-embeds/figs/compare/summary_all75_test_metrics.md");
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
check(cuts.length + 1 === rail.length && rail.length === chips.length,
  `${cuts.length + 1} scenes, ${rail.length} rail entries, ${chips.length} chips`);
chips.forEach((a, i) => {
  const s = bounds.findIndex((_, k) => a >= bounds[k] && a < bounds[k + 1]);
  check(s === i, `chip ${i} at t=${a} lands in scene ${s}`);
});

/* 3 ------------------------------------------------------------- dwell times */
const DUR = +src.match(/const DUR=(\d+)/)[1];
console.log("\ndwell (every reveal must finish before its scene is cut)");
["s1", "s2", "s3", "s4", "s5"].forEach((name, i) => {
  const from = src.indexOf(`function ${name}(`);
  const body = src.slice(from, src.indexOf("\n/* ---", from));
  const ends = [...body.matchAll(/sub\(p,[\d.]+,([\d.]+)\)/g)].map(m => +m[1]);
  const strip = body.match(/resultStrip\([^;]*?,\s*(\.\d+)\s*[,)]/);
  if (strip) ends.push(+strip[1] + 0.175);        // strip staggers its four values
  const last = Math.max(...ends, 0);
  const dwell = (1 - last) * (bounds[i + 1] - bounds[i]) * DUR / 1000;
  check(dwell >= 1.0, `${name} last reveal at p=${last.toFixed(2)} -> ${dwell.toFixed(1)}s on screen`);
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

console.log(fail ? `\n${fail} FAILED\n` : "\nall checks passed\n");
process.exit(fail ? 1 : 0);
