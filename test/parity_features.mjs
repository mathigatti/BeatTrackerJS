// Compares JS feature extraction against madmom.
// Run: node test/parity_features.mjs   (from the repo root)

import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { FeatureExtractor } from "../web/src/features.js";

const manifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-beats.bin");
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const ref = JSON.parse(readFileSync("test/reference.json", "utf8"));

const model = loadModel(manifest, buffer);
const fx = new FeatureExtractor(model);
const nb = manifest.num_bands;

let maxSpec = 0;
let maxDiff = 0;
const rows = [];
for (let i = 0; i < ref.frames.length; i++) {
  const feats = fx.process(Float32Array.from(ref.frames[i]));
  const want = ref.feats[i];
  let ms = 0;
  let md = 0;
  for (let b = 0; b < nb; b++) {
    ms = Math.max(ms, Math.abs(feats[b] - want[b]));
    md = Math.max(md, Math.abs(feats[nb + b] - want[nb + b]));
  }
  maxSpec = Math.max(maxSpec, ms);
  maxDiff = Math.max(maxDiff, md);
  rows.push({ i, ms, md });
}

console.log(`frames: ${ref.frames.length}, bands: ${nb}, diff_frames: ${manifest.diff.frames}`);
console.log("\nframe | max|diff| spec | max|diff| diff");
for (const r of rows.slice(0, 6)) {
  console.log(
    `${String(r.i).padStart(5)} | ${r.ms.toExponential(2).padStart(14)} | ${r.md.toExponential(2).padStart(14)}`
  );
}
console.log(`\nmax |diff| log spectrum: ${maxSpec.toExponential(3)}`);
console.log(`max |diff| positive diff: ${maxDiff.toExponential(3)}`);

const ok = maxSpec < 1e-4 && maxDiff < 1e-4;
console.log(ok ? "\nPARITY OK" : "\nPARITY FAILED");
process.exit(ok ? 0 : 1);
