// The silence gate: on noise no beat should come out, on real music all of them
// should. Run: node test/silence.mjs   (from the repo root)

import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatTracker } from "../web/src/tracker.js";

const m = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const raw = readFileSync("web/public/models/madmom-beats.bin");
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const tr = new BeatTracker(loadModel(m, buf), m, buf);
const HOP = m.hop_size;

function run(label, gen, n) {
  tr.reset();
  const beats = [];
  for (let i = 0; i < n; i++) {
    const b = tr.pushHop(gen(i));
    if (b !== null) beats.push(b);
  }
  const ibi = [];
  for (let i = 1; i < beats.length; i++) ibi.push(beats[i] - beats[i - 1]);
  ibi.sort((a, b) => a - b);
  const med = ibi[Math.floor(ibi.length / 2)];
  console.log(
    `${label.padEnd(20)} beats=${String(beats.length).padStart(4)} in ${(n / 100).toFixed(0)}s` +
      (beats.length > 2 ? `  -> ${(60 / med).toFixed(1)} BPM` : "") +
      `   [energy=${tr.energy.toFixed(3)} shape=${tr.shape.toFixed(1)}]`
  );
}

const noise = (amp) => () => {
  const h = new Float32Array(HOP);
  for (let k = 0; k < HOP; k++) h[k] = (Math.random() * 2 - 1) * amp;
  return h;
};

run("digital silence", noise(0), 3000);
run("noise -80 dBFS", noise(1e-4), 3000);
run("noise -60 dBFS", noise(1e-3), 3000);
run("noise -45 dBFS", noise(1e-2), 3000);
run("noise -35 dBFS", noise(3e-2), 3000);
run("noise -25 dBFS", noise(1e-1), 3000);

const ab = readFileSync("test/e2e_audio.f32");
const audio = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));
const ref = JSON.parse(readFileSync("test/e2e_reference.json", "utf8"));
run("real music", (i) => audio.subarray(i * HOP, (i + 1) * HOP), ref.num_hops);
console.log(`(offline madmom reference on that audio: ${ref.beats.length} beats)`);
