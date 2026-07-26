// Filling in skipped beats: how many gaps online madmom leaves and how well the
// grid covers them. Compared against offline madmom's beats, which is the
// reference the port is trying to imitate.
// Run: node test/gapfill.mjs   (from the repo root)

import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatTracker } from "../web/src/tracker.js";
import { BeatGrid } from "../web/src/grid.js";

const manifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-beats.bin");
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const HOP = manifest.hop_size;

const ab = readFileSync("test/e2e_audio.f32");
const audio = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));
const ref = JSON.parse(readFileSync("test/e2e_reference.json", "utf8"));
const refBeats = ref.beats.map(Number);

const TOL = 0.07;   // +-70 ms, the standard F-measure tolerance

function score(times) {
  const used = new Set();
  let hits = 0;
  for (const t of times) {
    let best = -1;
    let bestD = TOL;
    for (let i = 0; i < refBeats.length; i++) {
      if (used.has(i)) continue;
      const d = Math.abs(refBeats[i] - t);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) { used.add(best); hits++; }
  }
  const p = times.length ? hits / times.length : 0;
  const r = hits / refBeats.length;
  return { p, r, f: p + r > 0 ? (2 * p * r) / (p + r) : 0, hits };
}

function run(fill) {
  const tracker = new BeatTracker(loadModel(manifest, buf), manifest, buf);
  const grid = new BeatGrid(4);
  const times = [];
  const fills = [];
  for (let i = 0; i < ref.num_hops; i++) {
    const beat = tracker.pushHop(audio.subarray(i * HOP, (i + 1) * HOP));
    const now = i / 100;
    // with fill=false the beat-less hops are hidden from it, so it never fills
    const events = fill || beat !== null ? grid.step(now, beat !== null, tracker.silent) : [];
    for (const e of events) {
      times.push(e.time);
      if (e.predicted) fills.push(e.time);
    }
  }
  return { times, fills };
}

for (const fill of [false, true]) {
  const { times, fills } = run(fill);
  const s = score(times);
  console.log(
    `${fill ? "with fill    " : "without fill "} beats=${String(times.length).padStart(3)}` +
      ` (predicted ${String(fills.length).padStart(2)})  P=${s.p.toFixed(3)} R=${s.r.toFixed(3)} F=${s.f.toFixed(3)}`
  );
  if (fills.length) console.log(`   fills at: ${fills.map((t) => t.toFixed(2)).join(" ")}`);
}
console.log(`offline madmom reference: ${refBeats.length} beats`);

// On this audio the fill never fires: once locked, madmom skips nothing. So the
// logic is tested separately, with a perfect 120 BPM grid that has three beats
// deleted by hand.
const PERIOD = 0.5;
const DROP = new Set([20, 21, 34]);
const grid = new BeatGrid(4);
const got = [];
for (let i = 0; i < 6000; i++) {
  const now = i / 100;
  const n = Math.round(now / PERIOD);
  const isBeat = Math.abs(now - n * PERIOD) < 1e-9 && !DROP.has(n);
  for (const e of grid.step(now, isBeat, false)) got.push({ ...e, n });
}
const filled = got.filter((e) => e.predicted);
console.log(`\nsynthetic 120 BPM grid, deleted beats: ${[...DROP].join(" ")}`);
console.log(`  beats emitted: ${got.length} of ${Math.floor(6000 / 100 / PERIOD) - 1} expected`);
for (const e of filled) {
  const target = e.n * PERIOD;
  console.log(
    `  fill at t=${e.time.toFixed(3)} (expected ${target.toFixed(3)}, ` +
      `error ${((e.time - target) * 1000).toFixed(0)} ms) position ${e.inBar}`
  );
}
// the position must advance by one all the way through: if the fill counted
// wrong, a jump or a repeat would show up here
const seq = got.map((e) => e.inBar);
const ok = seq.every((v, i) => i === 0 || v === (seq[i - 1] % 4) + 1);
console.log(`  positions: ${seq.slice(0, 40).join("")}...  ${ok ? "grid did not shift" : "WARNING: grid shifted"}`);
