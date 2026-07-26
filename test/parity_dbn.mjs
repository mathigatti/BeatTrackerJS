// Compares the JS DBN against madmom. Run: node test/parity_dbn.mjs (from the repo root)

import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatDBN, dbnArrays } from "../web/src/dbn.js";

const manifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-beats.bin");
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const ref = JSON.parse(readFileSync("test/reference_dbn.json", "utf8"));

const model = loadModel(manifest, buffer);
const dbn = new BeatDBN(model, dbnArrays(manifest, buffer));

console.log(`states: ${manifest.dbn.num_states}, transitions: ${manifest.dbn.num_transitions}`);
console.log(`reference activations: ${ref.activations.length}`);

const beats = [];
let stateMatches = 0;
for (let i = 0; i < ref.activations.length; i++) {
  const b = dbn.step(ref.activations[i]);
  if (dbn.lastState === ref.argmax_states[i]) stateMatches++;
  if (b !== null) beats.push(b);
}

const statePct = (stateMatches / ref.activations.length) * 100;
console.log(`\nargmax state matches: ${statePct.toFixed(1)}% of the frames`);
console.log(`JS beats     (${beats.length}): ${beats.slice(0, 8).map((b) => b.toFixed(2)).join(", ")}`);
console.log(`madmom beats (${ref.beats.length}): ${ref.beats.slice(0, 8).map((b) => b.toFixed(2)).join(", ")}`);

let maxDiff = 0;
const n = Math.min(beats.length, ref.beats.length);
for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(beats[i] - ref.beats[i]));
console.log(`max |diff| in beat times: ${maxDiff.toFixed(4)} s`);

const ok = statePct === 100 && beats.length === ref.beats.length && maxDiff < 1e-6;
console.log(ok ? "\nPARITY OK" : "\nPARITY FAILED");
process.exit(ok ? 0 : 1);
