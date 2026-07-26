// Compares the JS forward pass against the activations madmom produced in Python.
// Run: node test/parity.mjs   (from the repo root)

import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatNetwork } from "../web/src/net.js";

const manifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-beats.bin");
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const ref = JSON.parse(readFileSync("test/reference.json", "utf8"));

const model = loadModel(manifest, buffer);
const net = new BeatNetwork(model);

console.log(`ensemble: ${model.nets.length} nets, feature size ${manifest.feature_size}`);
console.log(`reference frames: ${ref.feats.length}`);

// The reference starts at frame 0 with the nets reset, so the port starts from
// `init` and the comparison is exact from the very first frame.
let maxDiffMean = 0;
let maxDiffNet = 0;
const rows = [];
for (let i = 0; i < ref.feats.length; i++) {
  const feats = Float32Array.from(ref.feats[i]);
  const mean = net.step(feats);
  const dMean = Math.abs(mean - ref.acts[i].mean);
  maxDiffMean = Math.max(maxDiffMean, dMean);
  for (let k = 0; k < net.perNet.length; k++) {
    maxDiffNet = Math.max(maxDiffNet, Math.abs(net.perNet[k] - ref.acts[i].per_net[k]));
  }
  rows.push({ i, js: mean, py: ref.acts[i].mean, diff: dMean });
}

console.log("\nframe |        JS |    Python |      diff");
for (const r of rows.slice(0, 8)) {
  console.log(
    `${String(r.i).padStart(5)} | ${r.js.toFixed(6).padStart(9)} | ` +
      `${r.py.toFixed(6).padStart(9)} | ${r.diff.toExponential(2).padStart(9)}`
  );
}
console.log(`\nmax |diff| of the ensemble mean: ${maxDiffMean.toExponential(3)}`);
console.log(`max |diff| per individual net:  ${maxDiffNet.toExponential(3)}`);

const ok = maxDiffMean < 1e-4;
console.log(ok ? "\nPARITY OK" : "\nPARITY FAILED");
process.exit(ok ? 0 : 1);
