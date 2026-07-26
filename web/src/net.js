// Forward pass of madmom's LSTM ensemble, one frame at a time.
//
// madmom's LSTMs have peepholes and a specific operation order
// (see madmom/ml/nn/layers.py:LSTMLayer.activate): the input and forget gates
// look at the PREVIOUS state, the output gate looks at the ALREADY UPDATED one.

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// out = data @ w + b  (+ state * peep) + prev @ r, with the activation applied
function gateActivate(gate, data, prev, state, out, fn) {
  const { w, r, b, peep } = gate;
  const n = b.length;
  const inSize = w.rows;
  for (let j = 0; j < n; j++) {
    let acc = b[j];
    for (let i = 0; i < inSize; i++) acc += data[i] * w.data[i * n + j];
    if (peep !== null && state !== null) acc += state[j] * peep[j];
    for (let i = 0; i < n; i++) acc += prev[i] * r.data[i * n + j];
    out[j] = fn(acc);
  }
  return out;
}

class LstmState {
  constructor(layer) {
    this.layer = layer;
    this.prev = new Float32Array(layer.size);
    this.state = new Float32Array(layer.size);
    this.ig = new Float32Array(layer.size);
    this.fg = new Float32Array(layer.size);
    this.cellOut = new Float32Array(layer.size);
    this.og = new Float32Array(layer.size);
    this.out = new Float32Array(layer.size);
    this.reset();
  }

  reset() {
    this.prev.set(this.layer.init);
    this.state.set(this.layer.cellInit);
  }

  step(data) {
    const L = this.layer;
    const n = L.size;
    gateActivate(L.inputGate, data, this.prev, this.state, this.ig, sigmoid);
    gateActivate(L.forgetGate, data, this.prev, this.state, this.fg, sigmoid);
    gateActivate(L.cell, data, this.prev, null, this.cellOut, Math.tanh);
    for (let j = 0; j < n; j++) {
      this.state[j] = this.cellOut[j] * this.ig[j] + this.state[j] * this.fg[j];
    }
    // the output gate already sees the updated state
    gateActivate(L.outputGate, data, this.prev, this.state, this.og, sigmoid);
    for (let j = 0; j < n; j++) this.out[j] = Math.tanh(this.state[j]) * this.og[j];
    this.prev.set(this.out);
    return this.out;
  }
}

function denseStep(layer, data) {
  const { w, b, size } = layer;
  const out = new Float32Array(size);
  for (let j = 0; j < size; j++) {
    let acc = b[j];
    for (let i = 0; i < w.rows; i++) acc += data[i] * w.data[i * size + j];
    out[j] = layer.activation === "sigmoid" ? sigmoid(acc) : acc;
  }
  return out;
}

/** Una de las 8 redes del ensemble. Mantiene su estado entre frames. */
class Net {
  constructor(layers) {
    this.layers = layers;
    this.states = layers.map((l) => (l.type === "lstm" ? new LstmState(l) : null));
  }

  reset() {
    for (const s of this.states) if (s) s.reset();
  }

  step(feats) {
    let x = feats;
    for (let i = 0; i < this.layers.length; i++) {
      x = this.layers[i].type === "lstm"
        ? this.states[i].step(x)
        : denseStep(this.layers[i], x);
    }
    return x[0];
  }
}

export class BeatNetwork {
  constructor(model) {
    this.nets = model.nets.map((layers) => new Net(layers));
    this.perNet = new Float32Array(this.nets.length);
  }

  reset() {
    for (const n of this.nets) n.reset();
  }

  /** Returns the ensemble-averaged activation for one frame of features. */
  step(feats) {
    let sum = 0;
    for (let i = 0; i < this.nets.length; i++) {
      const a = this.nets[i].step(feats);
      this.perNet[i] = a;
      sum += a;
    }
    return sum / this.nets.length;
  }
}
