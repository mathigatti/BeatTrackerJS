// Instantaneous per-beat intensity on real audio.
import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatTracker } from "../web/src/tracker.js";
import { AccentProfile } from "../web/src/accent.js";

const manifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-beats.bin");
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const ref = JSON.parse(readFileSync("test/e2e_reference.json", "utf8"));
const ab = readFileSync("test/e2e_audio.f32");
const audio = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));

const METER = 4;
const model = loadModel(manifest, buffer);
const tracker = new BeatTracker(model, manifest, buffer);
const accent = new AccentProfile(model, METER);
const hop = manifest.hop_size;

let beatIndex = 0;
const hits = [];
for (let i = 0; i < ref.num_hops; i++) {
  const b = tracker.pushHop(audio.subarray(i * hop, (i + 1) * hop));
  accent.pushFrame(tracker.features.specLinear);
  if (b !== null) {
    const inBar = (beatIndex++ % METER) + 1;
    hits.push({ t: b, pos: inBar, ...accent.onBeat(inBar) });
  }
}

console.log(`${hits.length} beats in ${(ref.num_hops / 100).toFixed(0)} s\n`);
console.log(" t(s) | pos | energy | intensity");
for (const h of hits.slice(20, 40)) {
  const bar = "#".repeat(Math.round(h.intensity * 20)).padEnd(20, "·");
  console.log(
    `${h.t.toFixed(2).padStart(5)} |  ${h.pos}  | ${h.energy.toFixed(2).padStart(7)} | ` +
      `${h.intensity.toFixed(3)} ${bar}`
  );
}
const ints = hits.slice(10).map((h) => h.intensity);
const lo = Math.min(...ints), hi = Math.max(...ints);
const weak = ints.filter((x) => x < 0.35).length;
console.log(`\nintensity range: ${lo.toFixed(3)} - ${hi.toFixed(3)}`);
console.log(`weak beats (<0.35): ${weak}/${ints.length} = ${(weak / ints.length * 100).toFixed(0)}%`);
