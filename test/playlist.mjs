// Playlist behaviour: what happens to the meter when the song changes.
// The estimator has a long memory on purpose (changing meter remaps every bar
// position), and that memory is exactly what can get it stuck. This walks a
// concatenated playlist and prints the meter in effect over time.
// Run: node test/playlist.mjs <file.f32>   (from the repo root)

import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatTracker } from "../web/src/tracker.js";
import { BeatGrid } from "../web/src/grid.js";
import { DownbeatFeatures } from "../web/src/downbeat_features.js";
import { DownbeatNetwork } from "../web/src/downbeat_net.js";
import { estimateBar, MeterTally, CANDIDATES } from "../web/src/downbeat_decode.js";

const bm = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const br = readFileSync("web/public/models/madmom-beats.bin");
const bb = br.buffer.slice(br.byteOffset, br.byteOffset + br.byteLength);
const dm = JSON.parse(readFileSync("web/public/models/madmom-downbeats.json", "utf8"));
const dr = readFileSync("web/public/models/madmom-downbeats.bin");
const db = dr.buffer.slice(dr.byteOffset, dr.byteOffset + dr.byteLength);

const HOP = bm.hop_size, WINDOW = 600, EVERY = 600;
const ab = readFileSync(process.argv[2]);
const audio = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));
const numHops = Math.floor(audio.length / HOP);

const tracker = new BeatTracker(loadModel(bm, bb), bm, bb);
const grid = new BeatGrid(4);
const features = new DownbeatFeatures(dm, db);
const network = new DownbeatNetwork(dm, db);
const tally = new MeterTally(4);

const beatLog = [];
console.log(`${(numHops / 100).toFixed(0)} s of audio\n`);
console.log("t(s) | meter | contrasts            | accumulated          | bpm");
for (let i = 0; i < numHops; i++) {
  const beat = tracker.pushHop(audio.subarray(i * HOP, (i + 1) * HOP));
  const now = i / 100;
  for (const e of grid.step(now, beat !== null, tracker.silent)) {
    beatLog.push({ index: e.index, frame: Math.round(e.time * 100) });
    if (beatLog.length > 200) beatLog.shift();
  }
  if (i >= WINDOW && i % EVERY === 0) {
    const start = i - WINDOW;
    const { feats, numFrames } = features.process(audio.subarray(start * HOP, i * HOP));
    const acts = network.process(feats, numFrames);
    const win = beatLog.filter((x) => x.frame >= start && x.frame < i);
    const est = estimateBar(acts, numFrames, start, win, CANDIDATES);
    if (!est) { console.log(`${String(i / 100).padStart(4)} | (no evidence)`); continue; }
    const m = tally.push(est.ranking);
    grid.meter = m;
    const c = CANDIDATES.map((k) => { const r = est.ranking.find((x) => x.meter === k); return `${k}:${r ? r.contrast.toFixed(2) : "  --"}`; }).join(" ");
    const a = CANDIDATES.map((k) => `${k}:${tally.scores.get(k).toFixed(2)}`).join(" ");
    console.log(`${String(i / 100).padStart(4)} | ${String(m).padStart(5)} | ${c} | ${a} | ${grid.bpm.toFixed(1)}`);
  }
}
