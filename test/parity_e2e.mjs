// Full JS pipeline against madmom's full pipeline, on real audio.
// Run: node test/parity_e2e.mjs   (from the repo root)

import { readFileSync, existsSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatTracker } from "../web/src/tracker.js";

if (!existsSync("test/e2e_reference.json")) {
  console.log("missing test/e2e_reference.json (run tools/export_madmom.py)");
  process.exit(0);
}

const manifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-beats.bin");
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const ref = JSON.parse(readFileSync("test/e2e_reference.json", "utf8"));

const audioRaw = readFileSync("test/e2e_audio.f32");
const audio = new Float32Array(
  audioRaw.buffer.slice(audioRaw.byteOffset, audioRaw.byteOffset + audioRaw.byteLength)
);

const model = loadModel(manifest, buffer);
const tracker = new BeatTracker(model, manifest, buffer);
const hop = manifest.hop_size;

console.log(`source: ${ref.source}, ${ref.num_hops} hops (${(ref.num_hops / 100).toFixed(1)} s)`);

const beats = [];
let maxAct = 0;
const t0 = process.hrtime.bigint();
for (let i = 0; i < ref.num_hops; i++) {
  tracker.pushHop(audio.subarray(i * hop, (i + 1) * hop));
  maxAct = Math.max(maxAct, Math.abs(tracker.activation - ref.activations[i]));
  // rawBeat, not the returned value: madmom has no silence gate, and here we
  // compare against madmom, not against what the app ends up showing
  if (tracker.rawBeat !== null) beats.push(tracker.rawBeat);
}
const msPerHop = Number(process.hrtime.bigint() - t0) / 1e6 / ref.num_hops;

console.log(`\nmax |diff| in activations: ${maxAct.toExponential(3)}`);
console.log(`JS beats:     ${beats.length}`);
console.log(`madmom beats: ${ref.beats.length}`);

let maxBeat = 0;
const n = Math.min(beats.length, ref.beats.length);
for (let i = 0; i < n; i++) maxBeat = Math.max(maxBeat, Math.abs(beats[i] - ref.beats[i]));
console.log(`max |diff| in beat times: ${maxBeat.toFixed(6)} s`);

const ibi = [];
for (let i = 1; i < beats.length; i++) ibi.push(beats[i] - beats[i - 1]);
ibi.sort((a, b) => a - b);
const median = ibi[Math.floor(ibi.length / 2)];
console.log(`implied BPM: ${(60 / median).toFixed(1)}`);
console.log(`\nperformance: ${msPerHop.toFixed(3)} ms/hop (10 ms budget) -> ${(msPerHop / 10 * 100).toFixed(1)}% of a core`);

const ok = maxAct < 1e-4 && beats.length === ref.beats.length && maxBeat < 1e-6;
console.log(ok ? "\nPARITY OK" : "\nPARITY FAILED");
process.exit(ok ? 0 : 1);
