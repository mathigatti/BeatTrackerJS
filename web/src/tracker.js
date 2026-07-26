// Full pipeline: audio -> features -> LSTM ensemble -> DBN -> beats.

import { FeatureExtractor } from "./features.js";
import { BeatNetwork } from "./net.js";
import { BeatDBN, dbnArrays } from "./dbn.js";

// The DBN keeps emitting an evenly spaced grid over silence or noise, and an
// evenly spaced grid is exactly what the confidence metric rewards: measured,
// noise at -35 dBFS invented 29 beats in 30 s and a convincing 102 BPM.
//
// Two gates, because level alone is not enough:
//
//  - level: smoothed rms * 18, cut at 0.02 (~-60 dBFS), same as the Python
//    backend. Covers silence, but loud room noise sails straight through.
//
//  - activation shape. What separates music from noise is not how large the
//    activation is but how spiky. Measured over 45 s of real music vs white
//    noise at four levels, using p97/median over 5 s windows:
//        music  min 3.5   p05 3.8   median 2439
//        noise  max 2.8 at every level (-45 to -10 dBFS)
//    Magnitude does not separate (the decayed peak gives 0.13 on music against
//    0.12 on noise); contrast does, and the bulk of music sits three orders of
//    magnitude above the threshold.
const SILENCE_ENERGY = 0.02;
const ENERGY_ALPHA = 0.08;
const ENERGY_GAIN = 18;
const SHAPE_WINDOW = 500;    // 5 s of activations
const SHAPE_MIN_FILL = 200;  // 2 s is already enough to judge
const SHAPE_EVERY = 25;      // recomputing 4 times a second is plenty
const SHAPE_MIN = 3.0;

export class BeatTracker {
  constructor(model, manifest, buffer) {
    this.model = model;
    this.hopSize = manifest.hop_size;
    this.frameSize = manifest.frame_size;
    this.features = new FeatureExtractor(model);
    this.network = new BeatNetwork(model);
    this.dbn = new BeatDBN(model, dbnArrays(manifest, buffer));
    // sliding window of frameSize samples, advancing by hopSize
    this.frame = new Float32Array(this.frameSize);
    this.activation = 0;
    this.rawBeat = null;
    this.energy = 0;
    this.shape = 0;
    this._smoothed = 0;
    this._acts = new Float32Array(SHAPE_WINDOW);
    this._actPos = 0;
    this._actFill = 0;
    this._scratch = new Float32Array(SHAPE_WINDOW);
  }

  reset() {
    this.features.reset();
    this.network.reset();
    this.dbn.reset();
    this.frame.fill(0);
    this.energy = 0;
    this.shape = 0;
    this._smoothed = 0;
    this._actPos = 0;
    this._actFill = 0;
  }

  get silent() {
    return this.energy < SILENCE_ENERGY || this.shape < SHAPE_MIN;
  }

  _updateShape() {
    const n = this._actFill;
    if (n < SHAPE_MIN_FILL) {
      this.shape = 0;
      return;
    }
    const w = this._scratch.subarray(0, n);
    w.set(this._acts.subarray(0, n));
    w.sort();
    const median = w[Math.floor(0.5 * n)];
    this.shape = w[Math.floor(0.97 * n)] / Math.max(median, 1e-4);
  }

  /**
   * Feeds exactly hopSize samples. Returns the beat time in seconds, or null.
   */
  pushHop(hop) {
    const { frame, frameSize, hopSize } = this;
    let sum = 0;
    for (let i = 0; i < hop.length; i++) sum += hop[i] * hop[i];
    const rms = Math.sqrt(sum / Math.max(1, hop.length));
    this._smoothed = (1 - ENERGY_ALPHA) * this._smoothed + ENERGY_ALPHA * rms;
    this.energy = Math.min(1, this._smoothed * ENERGY_GAIN);

    frame.copyWithin(0, hopSize);
    frame.set(hop, frameSize - hopSize);
    const feats = this.features.process(frame);
    this.activation = this.network.step(feats);
    this._acts[this._actPos] = this.activation;
    this._actPos = (this._actPos + 1) % SHAPE_WINDOW;
    if (this._actFill < SHAPE_WINDOW) this._actFill++;
    if (this._actPos % SHAPE_EVERY === 0) this._updateShape();

    // the model keeps being fed during silence (so it does not start cold when
    // the music returns), but no beats are emitted. `rawBeat` is what the DBN
    // said before the gate: the parity test against madmom uses that.
    this.rawBeat = this.dbn.step(this.activation);
    return this.silent ? null : this.rawBeat;
  }
}
