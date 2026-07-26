import { readFileSync } from "node:fs";
import { loadModel } from "../web/src/model.js";
import { BeatNetwork } from "../web/src/net.js";
const manifest=JSON.parse(readFileSync(`web/public/models/madmom-beats.json`,"utf8"));
const raw=readFileSync(`web/public/models/madmom-beats.bin`);
const buf=raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength);
const ref=JSON.parse(readFileSync("test/reference.json","utf8"));
const net=new BeatNetwork(loadModel(manifest,buf));
const feats=ref.feats.map(f=>Float32Array.from(f));
for(let i=0;i<2000;i++) net.step(feats[i%feats.length]);   // JIT warmup
const N=20000, t0=process.hrtime.bigint();
for(let i=0;i<N;i++) net.step(feats[i%feats.length]);
const ms=Number(process.hrtime.bigint()-t0)/1e6/N;
console.log(`ensemble of 8 nets: ${ms.toFixed(4)} ms/frame`);
console.log(`real-time budget: 10.00 ms/frame (fps=100)`);
console.log(`-> ${(ms/10*100).toFixed(2)}% of a core`);
console.log(`equivalent to ${(1/ms*1000/100).toFixed(0)}x real time`);
