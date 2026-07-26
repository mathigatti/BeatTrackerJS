// madmom's bidirectional ensemble for downbeats.
//
// Each layer runs the sequence forwards and backwards and concatenates
// [fwd, bwd_invertido] (madmom/ml/nn/layers.py:BidirectionalLayer.activate).
// That makes it NON-CAUSAL: the whole window is needed before it produces
// anything. The last layer is dense 50->3 with softmax, and class 0 is dropped,
// leaving [beat, downbeat] just like RNNDownBeatProcessor.

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function loadGate(manifest, buffer, prefix) {
  const view = (name) => {
    const e = manifest.arrays[name];
    return e ? new Float32Array(buffer, e.offset, e.count) : null;
  };
  const meta = (name) => manifest.arrays[name];
  const w = meta(`${prefix}.weights`);
  return {
    w: { data: view(`${prefix}.weights`), rows: w.shape[0], cols: w.shape[1] },
    r: { data: view(`${prefix}.recurrent_weights`) },
    b: view(`${prefix}.bias`),
    peep: view(`${prefix}.peephole_weights`),
  };
}

function loadLstm(manifest, buffer, prefix) {
  const view = (name) => {
    const e = manifest.arrays[name];
    return new Float32Array(buffer, e.offset, e.count);
  };
  return {
    inputGate: loadGate(manifest, buffer, `${prefix}.input_gate`),
    forgetGate: loadGate(manifest, buffer, `${prefix}.forget_gate`),
    cell: loadGate(manifest, buffer, `${prefix}.cell`),
    outputGate: loadGate(manifest, buffer, `${prefix}.output_gate`),
    init: view(`${prefix}.init`),
    cellInit: view(`${prefix}.cell_init`),
    size: view(`${prefix}.cell.bias`).length,
    inputSize: manifest.arrays[`${prefix}.cell.weights`].shape[0],
  };
}

/** Runs an LSTM over a whole sequence; `reverse` walks it backwards. */
function runLstm(L, input, T, inSize, reverse, out, outStride, outOffset) {
  const n = L.size;
  const prev = new Float32Array(L.init);
  const state = new Float32Array(L.cellInit);
  const ig = new Float32Array(n);
  const fg = new Float32Array(n);
  const cl = new Float32Array(n);
  const og = new Float32Array(n);

  const gate = (g, data, dataOff, st, dst, fn) => {
    for (let j = 0; j < n; j++) {
      let acc = g.b[j];
      for (let i = 0; i < inSize; i++) acc += data[dataOff + i] * g.w.data[i * n + j];
      if (g.peep !== null && st !== null) acc += st[j] * g.peep[j];
      for (let i = 0; i < n; i++) acc += prev[i] * g.r.data[i * n + j];
      dst[j] = fn(acc);
    }
  };

  for (let s = 0; s < T; s++) {
    const t = reverse ? T - 1 - s : s;
    const off = t * inSize;
    gate(L.inputGate, input, off, state, ig, sigmoid);
    gate(L.forgetGate, input, off, state, fg, sigmoid);
    gate(L.cell, input, off, null, cl, Math.tanh);
    for (let j = 0; j < n; j++) state[j] = cl[j] * ig[j] + state[j] * fg[j];
    gate(L.outputGate, input, off, state, og, sigmoid);
    const base = t * outStride + outOffset;
    for (let j = 0; j < n; j++) {
      const v = Math.tanh(state[j]) * og[j];
      prev[j] = v;
      out[base + j] = v;
    }
  }
}

class DownbeatNet {
  constructor(manifest, buffer, netIdx) {
    const spec = manifest.nets[netIdx].layers;
    this.layers = spec.map((layer, li) => {
      const p = `net${netIdx}.layer${li}`;
      if (layer.type === "bilstm") {
        return {
          type: "bilstm",
          size: layer.size,
          inputSize: layer.input_size,
          fwd: loadLstm(manifest, buffer, `${p}.fwd`),
          bwd: loadLstm(manifest, buffer, `${p}.bwd`),
        };
      }
      const e = manifest.arrays[`${p}.weights`];
      return {
        type: "dense",
        size: layer.size,
        inputSize: layer.input_size,
        w: new Float32Array(buffer, e.offset, e.count),
        b: new Float32Array(
          buffer,
          manifest.arrays[`${p}.bias`].offset,
          manifest.arrays[`${p}.bias`].count
        ),
        activation: layer.activation,
      };
    });
  }

  /** input: Float32Array(T*inSize). Returns Float32Array(T*3) with softmax. */
  process(input, T) {
    let x = input;
    let inSize = this.layers[0].inputSize;
    for (const layer of this.layers) {
      if (layer.type === "bilstm") {
        const stride = 2 * layer.size;
        const out = new Float32Array(T * stride);
        runLstm(layer.fwd, x, T, inSize, false, out, stride, 0);
        runLstm(layer.bwd, x, T, inSize, true, out, stride, layer.size);
        x = out;
        inSize = stride;
      } else {
        const out = new Float32Array(T * layer.size);
        for (let t = 0; t < T; t++) {
          let max = -Infinity;
          for (let j = 0; j < layer.size; j++) {
            let acc = layer.b[j];
            for (let i = 0; i < inSize; i++) acc += x[t * inSize + i] * layer.w[i * layer.size + j];
            out[t * layer.size + j] = acc;
            if (acc > max) max = acc;
          }
          let sum = 0;
          for (let j = 0; j < layer.size; j++) {
            const v = Math.exp(out[t * layer.size + j] - max);
            out[t * layer.size + j] = v;
            sum += v;
          }
          for (let j = 0; j < layer.size; j++) out[t * layer.size + j] /= sum;
        }
        x = out;
        inSize = layer.size;
      }
    }
    return x;
  }
}

export class DownbeatNetwork {
  constructor(manifest, buffer) {
    this.nets = manifest.nets.map((_, i) => new DownbeatNet(manifest, buffer, i));
    this.featureSize = manifest.feature_size;
  }

  /**
   * feats: Float32Array(T * 314). Returns Float32Array(T * 2) with
   * [beat, downbeat] averaged over the ensemble (class 0 is dropped, exactly as
   * RNNDownBeatProcessor does).
   */
  process(feats, T) {
    const out = new Float32Array(T * 2);
    for (const net of this.nets) {
      const p = net.process(feats, T);
      for (let t = 0; t < T; t++) {
        out[t * 2] += p[t * 3 + 1];
        out[t * 2 + 1] += p[t * 3 + 2];
      }
    }
    const inv = 1 / this.nets.length;
    for (let i = 0; i < out.length; i++) out[i] *= inv;
    return out;
  }
}
