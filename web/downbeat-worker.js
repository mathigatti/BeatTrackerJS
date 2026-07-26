// Web Worker for the downbeat model.
//
// It runs on its own thread because a 6 s window costs ~1.6 s of compute: on the
// main thread it would freeze the UI every time. It takes raw audio and returns
// the activations; the phase vote happens on the main thread, which is where the
// beat grid lives.

import { DownbeatFeatures } from "./src/downbeat_features.js";
import { DownbeatNetwork } from "./src/downbeat_net.js";

let features = null;
let network = null;
let busy = false;

async function init(base) {
  const [manifest, buffer] = await Promise.all([
    fetch(`${base}public/models/madmom-downbeats.json`).then((r) => r.json()),
    fetch(`${base}public/models/madmom-downbeats.bin`).then((r) => r.arrayBuffer()),
  ]);
  features = new DownbeatFeatures(manifest, buffer);
  network = new DownbeatNetwork(manifest, buffer);
  return { ensemble: manifest.ensemble, featureSize: manifest.feature_size };
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type === "init") {
    try {
      const info = await init(msg.base);
      self.postMessage({ type: "ready", ...info });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    }
    return;
  }

  if (msg.type === "window") {
    // if it is still chewing on the previous window, drop this one: arriving
    // late with a stale downbeat is worse than skipping it
    if (busy || !network) {
      self.postMessage({ type: "skipped", startFrame: msg.startFrame });
      return;
    }
    busy = true;
    const t0 = performance.now();
    try {
      const audio = new Float32Array(msg.audio);
      const { feats, numFrames } = features.process(audio);
      const activations = network.process(feats, numFrames);
      self.postMessage(
        {
          type: "activations",
          startFrame: msg.startFrame,
          numFrames,
          activations: activations.buffer,
          ms: performance.now() - t0,
        },
        [activations.buffer]
      );
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    } finally {
      busy = false;
    }
  }
};
