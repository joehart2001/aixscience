/* Find overlapping canvas text.
 *
 *   node tools/check_text.js [--width 1000] [--samples 220] [--pad 2]
 *
 * The page draws all its labels with fillText at hand-placed offsets, so
 * nothing stops two of them landing on each other at some point in the
 * timeline. This runs the real bundle against a recording 2D context, steps
 * the animation through the whole loop, and reports any two labels that are
 * visible at the same instant and whose boxes intersect.
 *
 * Glyph widths are approximated per family (no font metrics in node), so treat
 * small overlaps as advisory and large ones as real.
 */
const fs = require("fs");
const path = require("path");

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? Number(process.argv[i + 1]) : d;
};
const WIDTH = arg("width", 1000);     // 1000 = desktop layout, 400 = phone
const SAMPLES = arg("samples", 220);
const PAD = arg("pad", 2);            // ignore intersections thinner than this
const MIN_ALPHA = 0.25;               // a label fading in is not yet "visible"

// rough advance width per family, as a fraction of font size
const ADVANCE = { mono: 0.60, chivo: 0.54, source: 0.50 };
function widthOf(text, font) {
  const size = parseFloat(/(\d+(?:\.\d+)?)px/.exec(font)?.[1] ?? 12);
  const fam = /Plex Mono/.test(font) ? "mono" : /Chivo/.test(font) ? "chivo" : "source";
  return text.length * size * ADVANCE[fam];
}

const draws = [];
let frameIndex = 0;

function makeCtx() {
  const state = { font: "12px sans", textAlign: "left", globalAlpha: 1 };
  const stack = [];
  const noop = () => {};
  const ctx = {
    canvas: { parentElement: { clientWidth: WIDTH } },
    save() { stack.push({ ...state }); },
    restore() { Object.assign(state, stack.pop() ?? state); },
    measureText: (t) => ({ width: widthOf(t, state.font) }),
    fillText(text, x, y) {
      if (!String(text).trim() || state.globalAlpha < MIN_ALPHA) return;
      const size = parseFloat(/(\d+(?:\.\d+)?)px/.exec(state.font)?.[1] ?? 12);
      const w = widthOf(String(text), state.font);
      const x0 = state.textAlign === "right" ? x - w
               : state.textAlign === "center" ? x - w / 2 : x;
      draws.push({ f: frameIndex, text: String(text), size,
                   x0, x1: x0 + w, y0: y - size * 0.78, y1: y + size * 0.24 });
    },
    setTransform: noop, clearRect: noop, fillRect: noop, strokeRect: noop,
    beginPath: noop, closePath: noop, moveTo: noop, lineTo: noop, arc: noop,
    arcTo: noop, quadraticCurveTo: noop, fill: noop, stroke: noop,
    setLineDash: noop, resize: noop, drawImage: noop, polygon: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
  };
  for (const k of ["font", "textAlign", "globalAlpha", "fillStyle", "strokeStyle",
                   "lineWidth", "lineCap", "lineJoin", "textBaseline"]) {
    Object.defineProperty(ctx, k, {
      get: () => state[k], set: (v) => { state[k] = v; }, enumerable: true,
    });
  }
  return ctx;
}

const page = fs.readFileSync(path.join(__dirname, "../dist/levels.html"), "utf8");
const src = page.slice(page.indexOf("<script>") + 8, page.lastIndexOf("</script>"));
const ctx = makeCtx();

let rafCb = null;
global.window = global;
global.devicePixelRatio = 1;
global.getComputedStyle = () => ({ getPropertyValue: () => "#808080" });
global.matchMedia = () => ({ matches: false, addEventListener: () => {} });
global.ResizeObserver = class { observe() {} };
global.requestAnimationFrame = (cb) => { rafCb = cb; };
const chips = [];
global.document = {
  getElementById: (id) => ({
    getContext: () => ctx, parentElement: { clientWidth: WIDTH },
    style: {}, setAttribute: () => {}, addEventListener: () => {},
    innerHTML: "", value: 0, textContent: "",
  }),
  querySelectorAll: () => chips,
  documentElement: {},
};

eval(src);

const DUR = Number(/const DUR=(\d+)/.exec(src)[1]);
const CUT = JSON.parse(/const CUT=(\[[^\]]+\])/.exec(src)[1].replace(/\./g, "0."));
const bounds = [0, ...CUT, 1];
const sceneOf = (u) => bounds.findIndex((_, i) => u >= bounds[i] && u < bounds[i + 1]);

for (let i = 0; i < SAMPLES; i++) {
  frameIndex = i;
  rafCb(i * (DUR / SAMPLES));            // the loop advances t from the timestamp
}

const overlap = (a, b) => {
  const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
  const h = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
  return w > PAD && h > PAD ? { w, h } : null;
};

const worst = new Map();
const byFrame = new Map();
draws.forEach((d) => { (byFrame.get(d.f) ?? byFrame.set(d.f, []).get(d.f)).push(d); });
for (const [f, list] of byFrame) {
  for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
    const o = overlap(list[i], list[j]);
    if (!o) continue;
    if (list[i].text === list[j].text &&
        Math.abs(list[i].x0 - list[j].x0) < 1 && Math.abs(list[i].y0 - list[j].y0) < 1)
      continue;                                    // same label painted twice
    const key = `${list[i].text}\u0000${list[j].text}`;
    const prev = worst.get(key);
    if (!prev || o.w * o.h > prev.area)
      worst.set(key, { a: list[i], b: list[j], area: o.w * o.h, ...o,
                       scene: sceneOf(f / SAMPLES) + 1 });
  }
}

const hits = [...worst.values()].sort((x, y) => y.area - x.area);
const label = WIDTH >= 680 ? "desktop" : "phone";
console.log(`\n${label} (${WIDTH}px): ${draws.length} label draws over ${SAMPLES} frames, ` +
            `${hits.length} overlapping pair${hits.length === 1 ? "" : "s"}`);
const cut = (s) => (s.length > 42 ? s.slice(0, 41) + "…" : s);
for (const h of hits.slice(0, 18)) {
  console.log(`  scene ${h.scene}  ${Math.round(h.w)}x${Math.round(h.h)}px` +
              `   "${cut(h.a.text)}"  /  "${cut(h.b.text)}"`);
}
process.exit(hits.length ? 1 : 0);
