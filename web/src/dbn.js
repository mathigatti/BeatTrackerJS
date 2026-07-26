// Port of madmom's DBNBeatTrackingProcessor in online mode.
//
// The algorithm was derived by comparing against madmom (diff 5.5e-9): the
// transition model is in CSR *by destination* (for state s, the incoming
// transitions are from_states[pointers[s]..pointers[s+1]]), the densities are
// linear [(1-a)/(lambda-1), a], and the forward pass renormalises every frame.

export class BeatDBN {
  constructor(model, arrays) {
    const cfg = model.config.dbn;
    this.numStates = cfg.num_states;
    this.lambda = cfg.observation_lambda;
    this.maxBpm = cfg.max_bpm;
    this.fps = model.config.fps;

    this.pointers = arrays.pointers;
    this.fromStates = arrays.fromStates;
    this.probabilities = arrays.probabilities;
    this.obsPointers = arrays.obsPointers;
    this.initial = arrays.initial;
    this.statePositions = arrays.statePositions;

    // float64: madmom's forward pass runs in double precision, and the
    // repeated normalisation accumulates error fast in float32.
    this.fwd = new Float64Array(this.numStates);
    this.next = new Float64Array(this.numStates);
    this.reset();
  }

  reset() {
    for (let i = 0; i < this.numStates; i++) this.fwd[i] = this.initial[i];
    this.counter = 0;
    this.lastBeat = 0;
    this.tempo = 0;
  }

  /**
   * Processes one activation. Returns null, or the beat time in seconds.
   */
  step(activation) {
    const a = Math.min(Math.max(activation, 1e-9), 1 - 1e-9);
    const dens0 = (1 - a) / (this.lambda - 1);
    const dens1 = a;

    const { pointers, fromStates, probabilities, obsPointers, fwd, next } = this;
    let total = 0;
    for (let s = 0; s < this.numStates; s++) {
      let acc = 0;
      const end = pointers[s + 1];
      for (let k = pointers[s]; k < end; k++) {
        acc += fwd[fromStates[k]] * probabilities[k];
      }
      const v = acc * (obsPointers[s] === 1 ? dens1 : dens0);
      next[s] = v;
      total += v;
    }
    // normalise and find the most likely state
    let best = 0;
    let bestVal = -1;
    const inv = total > 0 ? 1 / total : 0;
    for (let s = 0; s < this.numStates; s++) {
      const v = next[s] * inv;
      fwd[s] = v;
      if (v > bestVal) {
        bestVal = v;
        best = s;
      }
    }
    this.lastState = best;

    const frame = this.counter;
    this.counter += 1;
    if (obsPointers[best] !== 1) return null;

    // madmom only reports a beat once the minimum interval has elapsed
    const curBeat = frame / this.fps;
    const nextBeat = this.lastBeat + 60 / this.maxBpm;
    if (curBeat < nextBeat) return null;
    this.tempo = 60 / (curBeat - this.lastBeat);
    this.lastBeat = curBeat;
    return curBeat;
  }
}

export function dbnArrays(manifest, buffer) {
  const u32 = (name) => {
    const e = manifest.arrays[name];
    // the blob stores everything as float32; indices are converted back to ints
    const f = new Float32Array(buffer, e.offset, e.count);
    const out = new Uint32Array(e.count);
    for (let i = 0; i < e.count; i++) out[i] = f[i];
    return out;
  };
  const f32 = (name) => {
    const e = manifest.arrays[name];
    return new Float32Array(buffer, e.offset, e.count);
  };
  return {
    pointers: u32("hmm.pointers"),
    fromStates: u32("hmm.from_states"),
    probabilities: f32("hmm.probabilities"),
    obsPointers: u32("hmm.obs_pointers"),
    initial: f32("hmm.initial"),
    statePositions: f32("hmm.state_positions"),
  };
}
