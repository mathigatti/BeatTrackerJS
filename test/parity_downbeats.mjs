// Parity of the downbeat model against madmom, and its real cost.
// Run: node test/parity_downbeats.mjs   (from the repo root)

import { readFileSync } from "node:fs";
import { DownbeatFeatures } from "../web/src/downbeat_features.js";
import { DownbeatNetwork } from "../web/src/downbeat_net.js";

const manifest = JSON.parse(readFileSync("web/public/models/madmom-downbeats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-downbeats.bin");
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const ref = JSON.parse(readFileSync("test/reference_downbeats.json", "utf8"));
const ab = readFileSync("test/downbeat_audio.f32");
const audio = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));

const T = ref.num_hops;
const hop = manifest.hop_size;
const F = manifest.feature_size;

console.log(`ensemble: ${manifest.ensemble} bidirectional nets`);
console.log(`features: ${F} dims from ${manifest.resolutions.length} resolutions`);
console.log(`window: ${T} frames (${(T / 100).toFixed(1)} s)\n`);

// --- features
const fx = new DownbeatFeatures(manifest, buffer);
const tf0 = process.hrtime.bigint();
const { feats } = fx.process(audio, T);
const featMs = Number(process.hrtime.bigint() - tf0) / 1e6;

let maxFeat = 0;
for (let t = 0; t < ref.feats_head.length; t++) {
  for (let b = 0; b < F; b++) {
    maxFeat = Math.max(maxFeat, Math.abs(feats[t * F + b] - ref.feats_head[t][b]));
  }
}
console.log(`features: max |diff| = ${maxFeat.toExponential(3)}  (${featMs.toFixed(0)} ms for ${T} frames)`);

// --- network
const net = new DownbeatNetwork(manifest, buffer);
const tn0 = process.hrtime.bigint();
const acts = net.process(feats, T);
const netMs = Number(process.hrtime.bigint() - tn0) / 1e6;

let maxAct = 0;
for (let t = 0; t < T; t++) {
  maxAct = Math.max(maxAct, Math.abs(acts[t * 2] - ref.activations[t][0]));
  maxAct = Math.max(maxAct, Math.abs(acts[t * 2 + 1] - ref.activations[t][1]));
}
console.log(`net:      max |diff| = ${maxAct.toExponential(3)}  (${netMs.toFixed(0)} ms for ${T} frames)`);

console.log("\nframe |   JS beat | madmom beat |    JS down | madmom down");
for (let t = 100; t < 106; t++) {
  console.log(
    `${String(t).padStart(5)} | ${acts[t * 2].toFixed(6).padStart(9)} | ` +
      `${ref.activations[t][0].toFixed(6).padStart(11)} | ` +
      `${acts[t * 2 + 1].toFixed(6).padStart(10)} | ${ref.activations[t][1].toFixed(6).padStart(11)}`
  );
}

const perWindow = (featMs + netMs) / (T / 100);
console.log(`\ncost: ${(featMs + netMs).toFixed(0)} ms for ${(T / 100).toFixed(1)} s of audio`);
console.log(`  -> a 6 s window would cost ~${(perWindow * 6).toFixed(0)} ms`);
console.log(`  -> running every 3 s: ${((perWindow * 6) / 3000 * 100).toFixed(0)}% of a core`);

const ok = maxFeat < 1e-4 && maxAct < 1e-4;
console.log(ok ? "\nPARITY OK" : "\nPARITY FAILED");
process.exit(ok ? 0 : 1);
