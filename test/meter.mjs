// Meter estimator over songs whose meter is known in advance.
// Run: node test/meter.mjs <file.f32> [...]   (from the repo root)

import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { loadModel } from "../web/src/model.js";
import { BeatTracker } from "../web/src/tracker.js";
import { DownbeatFeatures } from "../web/src/downbeat_features.js";
import { DownbeatNetwork } from "../web/src/downbeat_net.js";
import { estimateBar, MeterTally, CANDIDATES as DEFAULT_CANDIDATES } from "../web/src/downbeat_decode.js";

// CANDIDATES=3,4,6 node test/meter.mjs ...
const CANDIDATES = process.env.CANDIDATES
  ? process.env.CANDIDATES.split(",").map(Number)
  : DEFAULT_CANDIDATES;

const beatManifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const beatRaw = readFileSync("web/public/models/madmom-beats.bin");
const beatBuf = beatRaw.buffer.slice(beatRaw.byteOffset, beatRaw.byteOffset + beatRaw.byteLength);

const dbManifest = JSON.parse(readFileSync("web/public/models/madmom-downbeats.json", "utf8"));
const dbRaw = readFileSync("web/public/models/madmom-downbeats.bin");
const dbBuf = dbRaw.buffer.slice(dbRaw.byteOffset, dbRaw.byteOffset + dbRaw.byteLength);

const HOP = beatManifest.hop_size;
const WINDOW_HOPS = 600;
const EVERY_HOPS = 600;

const features = new DownbeatFeatures(dbManifest, dbBuf);
const network = new DownbeatNetwork(dbManifest, dbBuf);

console.log(`candidates: ${CANDIDATES.join(", ")}   window ${WINDOW_HOPS / 100}s every ${EVERY_HOPS / 100}s\n`);

for (const path of process.argv.slice(2)) {
  const ab = readFileSync(path);
  const audio = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));
  const numHops = Math.floor(audio.length / HOP);
  const tracker = new BeatTracker(loadModel(beatManifest, beatBuf), beatManifest, beatBuf);

  const beatLog = [];
  let beatIndex = 0;
  const rows = [];

  for (let i = 0; i < numHops; i++) {
    if (tracker.pushHop(audio.subarray(i * HOP, (i + 1) * HOP)) !== null) {
      beatLog.push({ index: beatIndex++, frame: i });
    }
    if (i >= WINDOW_HOPS && i % EVERY_HOPS === 0) {
      const startFrame = i - WINDOW_HOPS;
      const { feats, numFrames } = features.process(audio.subarray(startFrame * HOP, i * HOP));
      const acts = network.process(feats, numFrames);
      const inWindow = beatLog.filter((x) => x.frame >= startFrame && x.frame < i);
      rows.push({ at: i / 100, est: estimateBar(acts, numFrames, startFrame, inWindow, CANDIDATES) });
    }
  }

  console.log(`== ${basename(path)}`);
  console.log("t(s) | window | contrasts     | accumulated   | current");
  const tally = new MeterTally(4, CANDIDATES);
  for (const { at, est } of rows) {
    if (!est) {
      console.log(`${String(at).padStart(4)} | (no evidence)`);
      continue;
    }
    const current = tally.push(est.ranking);
    const rank = CANDIDATES.map((m) => {
      const r = est.ranking.find((x) => x.meter === m);
      return `${m}:${r ? r.contrast.toFixed(2) : "  --"}`;
    }).join(" ");
    const acc = CANDIDATES.map((m) => `${m}:${tally.scores.get(m).toFixed(2)}`).join(" ");
    console.log(
      `${String(at).padStart(4)} | ${String(est.meter).padStart(7)} | ${rank} | ${acc} | ${current}/4`
    );
  }
  console.log(`final meter: ${tally.meter}/4\n`);
}
