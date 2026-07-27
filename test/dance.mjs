// The dance moves are pure timing logic, so they can be checked without a
// browser: drive the Dancer with a synthetic musical position and count how
// often each part actually changes. This is what proves "Swing double" really
// is the same move at twice the rate.
// Run: node test/dance.mjs   (from the repo root)

import { Dancer, MOVES } from "../web/src/dance.js";

/** Minimal stand-in for the DOM element the Dancer writes to. */
function stubElement() {
  return {
    dataset: {},
    style: { setProperty() {} },
    classList: { add() {}, remove() {}, toggle() {} },
    offsetWidth: 0,
  };
}

globalThis.window = { setTimeout() {}, clearTimeout() {} };

const METER = 4;
const BARS = 8;
const STEPS_PER_BEAT = 64;   // fine enough to catch eighths without aliasing

function run(moveId) {
  const el = stubElement();
  const d = new Dancer(el);
  d.setMove(moveId);
  let arms = 0;
  let hips = 0;
  let lastArms = el.dataset.arms;
  let lastHips = el.dataset.hips;

  const total = BARS * METER * STEPS_PER_BEAT;
  for (let i = 0; i <= total; i++) {
    const beatPos = i / STEPS_PER_BEAT;
    d.update(beatPos, beatPos / METER, METER, 500);
    if (el.dataset.arms !== lastArms) { arms++; lastArms = el.dataset.arms; }
    if (el.dataset.hips !== lastHips) { hips++; lastHips = el.dataset.hips; }
  }
  return { arms, hips, style: el.dataset.style };
}

const beats = BARS * METER;
console.log(`${BARS} bars of ${METER}/4 = ${beats} beats\n`);
console.log("move             | style | arm changes      | hip changes");

let bad = 0;
for (const m of MOVES) {
  const { arms, hips, style } = run(m.id);
  const every = (v) => (v === "bar" ? METER : v === "bar2" ? METER * 2 : v);
  // the first tick is the initial state settling, hence the tolerance of 1
  const wantArms = beats / every(m.arms);
  const wantHips = beats / every(m.hips);
  const okA = Math.abs(arms - wantArms) <= 1;
  const okH = Math.abs(hips - wantHips) <= 1;
  if (!okA || !okH || style !== m.style) bad++;
  console.log(
    `${m.label.padEnd(16)} | ${style.padEnd(5)} | ${String(arms).padStart(3)} (want ${String(wantArms).padStart(3)}) ${okA ? "ok" : "BAD"} | ` +
      `${String(hips).padStart(3)} (want ${String(wantHips).padStart(3)}) ${okH ? "ok" : "BAD"}`
  );
}

// auto mode has to walk through every move and come back around
const el = stubElement();
const d = new Dancer(el);
d.setMove(null);
const seen = [];
for (let i = 0; i <= 64 * METER * STEPS_PER_BEAT; i++) {
  const beatPos = i / STEPS_PER_BEAT;
  d.update(beatPos, beatPos / METER, METER, 500);
  if (seen[seen.length - 1] !== d.move.id) seen.push(d.move.id);
}
const uniq = new Set(seen);
console.log(`\nauto over 64 bars: ${seen.length} changes, ${uniq.size}/${MOVES.length} distinct moves`);
console.log(`  order: ${seen.slice(0, MOVES.length + 1).join(" -> ")}`);
if (uniq.size !== MOVES.length) bad++;

// --- continuous moves: --flow has to sweep the full -1..1 range over exactly
// its declared period, and be smooth (no jump between consecutive frames).
console.log("\ncontinuous | period | range          | max step | zero at downbeat");
for (const m of MOVES.filter((x) => x.flow)) {
  const d = new Dancer(stubElement());
  d.setMove(m.id);
  const period = m.flow === "bar" ? 1 : 2;   // in bars
  const N = 2048;
  const vals = [];
  for (let i = 0; i <= N; i++) vals.push(d.flowValue((i / N) * period));

  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  let maxStep = 0;
  for (let i = 1; i < vals.length; i++) maxStep = Math.max(maxStep, Math.abs(vals[i] - vals[i - 1]));
  // it must return to where it started, or the motion would drift over time
  const closes = Math.abs(vals[0] - vals[vals.length - 1]) < 1e-9;
  const full = lo < -0.999 && hi > 0.999;
  const smooth = maxStep < 0.02;
  if (!full || !smooth || !closes) bad++;

  console.log(
    `${m.label.padEnd(18)} | ${String(period).padStart(2)} bar | ` +
      `${lo.toFixed(3)}..${hi.toFixed(3)} ${full ? "ok " : "BAD"} | ` +
      `${maxStep.toFixed(4)} ${smooth ? "ok " : "BAD"} | ` +
      `${d.flowValue(0).toFixed(3)} ${closes ? "loops" : "DRIFTS"}`
  );
}


console.log(`\n${bad === 0 ? "DANCE OK" : `${bad} MOVES WRONG`}`);
process.exit(bad === 0 ? 0 : 1);
