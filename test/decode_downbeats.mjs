// Full downbeat pipeline on real audio: beat tracker + bidirectional-model
// windows + phase vote. Checks that the vote is stable.
// Run: node test/decode_downbeats.mjs   (from the repo root)

import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatTracker } from "../web/src/tracker.js";
import { DownbeatFeatures } from "../web/src/downbeat_features.js";
import { DownbeatNetwork } from "../web/src/downbeat_net.js";
import { voteBarOrigin } from "../web/src/downbeat_decode.js";

const beatManifest = JSON.parse(readFileSync("web/public/models/madmom-beats.json", "utf8"));
const beatRaw = readFileSync("web/public/models/madmom-beats.bin");
const beatBuf = beatRaw.buffer.slice(beatRaw.byteOffset, beatRaw.byteOffset + beatRaw.byteLength);

const dbManifest = JSON.parse(readFileSync("web/public/models/madmom-downbeats.json", "utf8"));
const dbRaw = readFileSync("web/public/models/madmom-downbeats.bin");
const dbBuf = dbRaw.buffer.slice(dbRaw.byteOffset, dbRaw.byteOffset + dbRaw.byteLength);

const ref = JSON.parse(readFileSync("test/e2e_reference.json", "utf8"));
const ab = readFileSync("test/e2e_audio.f32");
const audio = new Float32Array(ab.buffer.slice(ab.byteOffset, ab.byteOffset + ab.byteLength));

const METER = 4;
const HOP = beatManifest.hop_size;
const WINDOW_HOPS = 600;    // 6 s
const EVERY_HOPS = 600;     // one window every 6 s

const tracker = new BeatTracker(loadModel(beatManifest, beatBuf), beatManifest, beatBuf);
const features = new DownbeatFeatures(dbManifest, dbBuf);
const network = new DownbeatNetwork(dbManifest, dbBuf);

console.log(`${ref.source}: ${(ref.num_hops / 100).toFixed(0)} s`);
console.log(`downbeat ensemble: ${dbManifest.ensemble} nets`);
console.log(`window ${WINDOW_HOPS / 100} s every ${EVERY_HOPS / 100} s\n`);

const beatLog = [];
let beatIndex = 0;
let barOrigin = 0;
const votes = [];
let totalMs = 0;

for (let i = 0; i < ref.num_hops; i++) {
  const b = tracker.pushHop(audio.subarray(i * HOP, (i + 1) * HOP));
  if (b !== null) beatLog.push({ index: beatIndex++, frame: i });

  if (i >= WINDOW_HOPS && i % EVERY_HOPS === 0) {
    const startFrame = i - WINDOW_HOPS;
    const win = audio.subarray(startFrame * HOP, i * HOP);
    const t0 = process.hrtime.bigint();
    const { feats, numFrames } = features.process(win);
    const acts = network.process(feats, numFrames);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    totalMs += ms;
    const inWindow = beatLog.filter((x) => x.frame >= startFrame && x.frame < i);
    const v = voteBarOrigin(acts, numFrames, startFrame, inWindow, METER);
    votes.push({ at: i / 100, v, ms, beats: inWindow.length });
  }
}

console.log("t(s) | beats | vote | margin | scores                        | ms");
for (const { at, v, ms, beats } of votes) {
  if (!v) {
    console.log(`${String(at).padStart(4)} | ${String(beats).padStart(5)} | not enough evidence  ${" ".repeat(21)} | ${ms.toFixed(0)}`);
    continue;
  }
  barOrigin = v.origin;
  const s = v.scores.map((x) => x.toFixed(3)).join(" ");
  console.log(
    `${String(at).padStart(4)} | ${String(beats).padStart(5)} | ${String(v.origin).padStart(4)} | ` +
      `${v.margin.toFixed(3).padStart(6)} | ${s} | ${ms.toFixed(0)}`
  );
}

const decided = votes.filter((x) => x.v).map((x) => x.v.origin);
const stable = decided.length > 1 && new Set(decided).size === 1;
console.log(`\nvotes cast: ${decided.length}/${votes.length} -> ${stable ? "STABLE" : "UNSTABLE"}`);
console.log(`mean cost per window: ${(totalMs / votes.length).toFixed(0)} ms`);
console.log(`running every ${EVERY_HOPS / 100} s -> ${((totalMs / votes.length) / (EVERY_HOPS * 10) * 100).toFixed(0)}% of a core`);
