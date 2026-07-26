// Feature extraction equivalent to RNNBeatProcessor's preprocessing:
// Hann window -> |FFT| -> logarithmic filterbank -> log10(1+x) -> positive diff,
// then [spec, diff] are stacked into the 162 values the network expects.

/** Iterative in-place radix-2 FFT over separate real/imaginary arrays. */
export class FFT {
  constructor(n) {
    this.n = n;
    this.levels = Math.log2(n);
    if (this.levels % 1 !== 0) throw new Error("FFT size must be a power of 2");
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let x = i;
      let r = 0;
      for (let j = 0; j < this.levels; j++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
      }
      this.rev[i] = r;
    }
  }

  transform(re, im) {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tre = re[l] * cos[k] + im[l] * sin[k];
          const tim = -re[l] * sin[k] + im[l] * cos[k];
          re[l] = re[j] - tre;
          im[l] = im[j] - tim;
          re[j] += tre;
          im[j] += tim;
        }
      }
    }
  }
}

export class FeatureExtractor {
  constructor(model) {
    const cfg = model.config;
    this.frameSize = cfg.frame_size;
    this.numBands = cfg.num_bands;
    this.numBins = cfg.num_bins;
    this.diffFrames = cfg.diff.frames;
    this.positiveOnly = cfg.diff.positive_only;
    this.window = model.window;
    this.fb = model.filterbank;

    this.fft = new FFT(this.frameSize);
    this.re = new Float64Array(this.frameSize);
    this.im = new Float64Array(this.frameSize);
    this.spec = new Float32Array(this.numBands);
    // filterbank magnitude BEFORE the log: the log compresses the range so
    // hard that it erases the accent differences between beats.
    this.specLinear = new Float32Array(this.numBands);
    this.feats = new Float32Array(2 * this.numBands);
    // circular buffer for the diff
    this.history = [];
    for (let i = 0; i <= this.diffFrames; i++) {
      this.history.push(new Float32Array(this.numBands));
    }
    this.histIdx = 0;
    this.primed = 0;
  }

  reset() {
    for (const h of this.history) h.fill(0);
    this.histIdx = 0;
    this.primed = 0;
  }

  /** frame: Float32Array of frameSize samples. Returns Float32Array(162). */
  process(frame) {
    const { re, im, window, frameSize, numBands } = this;
    for (let i = 0; i < frameSize; i++) {
      re[i] = frame[i] * window[i];
      im[i] = 0;
    }
    this.fft.transform(re, im);

    // filterbank over the magnitude (only the bins each band actually uses)
    const { weights, starts, lengths } = this.fb;
    let w = 0;
    for (let b = 0; b < numBands; b++) {
      const start = starts[b];
      const len = lengths[b];
      let acc = 0;
      for (let k = 0; k < len; k++) {
        const bin = start + k;
        acc += Math.hypot(re[bin], im[bin]) * weights[w + k];
      }
      w += len;
      this.specLinear[b] = acc;
      this.spec[b] = Math.log10(1 + acc);
    }

    // diff against the frame from diffFrames ago
    const prev = this.history[(this.histIdx + this.history.length - this.diffFrames) % this.history.length];
    const out = this.feats;
    for (let b = 0; b < numBands; b++) {
      out[b] = this.spec[b];
      let d = this.primed >= this.diffFrames ? this.spec[b] - prev[b] : 0;
      if (this.positiveOnly && d < 0) d = 0;
      out[numBands + b] = d;
    }

    this.history[this.histIdx].set(this.spec);
    this.histIdx = (this.histIdx + 1) % this.history.length;
    if (this.primed <= this.diffFrames) this.primed++;
    return out;
  }
}
