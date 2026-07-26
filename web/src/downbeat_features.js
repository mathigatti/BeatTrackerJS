// Features for the downbeat model: three resolutions in parallel (windows of
// 1024, 2048 and 4096 samples, all hopping by 441) giving 21, 45 and 91 bands.
// [spec, diff] of each is concatenated: 314 dims.
//
// WATCH the framing: unlike the beat model, here madmom uses its default
// `origin`, which is CENTRED — frame t spans
// `t*hop - frame_size/2` a `+ frame_size/2` (madmom/audio/signal.py:signal_frame).
// Since each resolution has a different size, using end-aligned windows
// (streaming style) shifts them against each other and against madmom.
//
// This underlines that the model is non-causal: frame t needs audio up to 2048
// samples ahead. It is processed over a complete buffer, not hop by hop.

import { FFT } from "./features.js";

class Resolution {
  constructor(cfg, arrays) {
    this.frameSize = cfg.frame_size;
    this.numBands = cfg.num_bands;
    this.diffFrames = cfg.diff_frames;
    this.window = arrays.window;
    this.fb = arrays.fb;
    this.fft = new FFT(this.frameSize);
    this.re = new Float64Array(this.frameSize);
    this.im = new Float64Array(this.frameSize);
  }

  /** Log spectrum of every frame: Float32Array(T * numBands). */
  spectrogram(audio, T, hopSize) {
    const { frameSize, re, im, window, numBands, fb } = this;
    const half = frameSize >> 1;
    const out = new Float32Array(T * numBands);
    for (let t = 0; t < T; t++) {
      const start = t * hopSize - half;
      for (let i = 0; i < frameSize; i++) {
        const s = start + i;
        re[i] = s >= 0 && s < audio.length ? audio[s] * window[i] : 0;
        im[i] = 0;
      }
      this.fft.transform(re, im);
      let w = 0;
      for (let b = 0; b < numBands; b++) {
        const bs = fb.starts[b];
        const len = fb.lengths[b];
        let acc = 0;
        for (let k = 0; k < len; k++) acc += Math.hypot(re[bs + k], im[bs + k]) * fb.weights[w + k];
        w += len;
        out[t * numBands + b] = Math.log10(1 + acc);
      }
    }
    return out;
  }

  /** Writes this resolution's [spec, diff] into `dst`, starting at `off`. */
  write(audio, T, hopSize, dst, stride, off) {
    const spec = this.spectrogram(audio, T, hopSize);
    const nb = this.numBands;
    const d = this.diffFrames;
    for (let t = 0; t < T; t++) {
      const base = t * stride + off;
      for (let b = 0; b < nb; b++) {
        const cur = spec[t * nb + b];
        dst[base + b] = cur;
        let diff = t >= d ? cur - spec[(t - d) * nb + b] : 0;
        if (diff < 0) diff = 0;
        dst[base + nb + b] = diff;
      }
    }
    return 2 * nb;
  }
}

export class DownbeatFeatures {
  constructor(manifest, buffer) {
    this.hopSize = manifest.hop_size;
    this.featureSize = manifest.feature_size;
    const view = (name) => {
      const e = manifest.arrays[name];
      return new Float32Array(buffer, e.offset, e.count);
    };
    this.resolutions = manifest.resolutions.map((cfg) => {
      const tag = `fb${cfg.frame_size}`;
      return new Resolution(cfg, {
        window: view(`window${cfg.frame_size}`),
        fb: {
          weights: view(`${tag}.weights`),
          starts: view(`${tag}.starts`),
          lengths: view(`${tag}.lengths`),
        },
      });
    });
  }

  /**
   * audio: Float32Array of the whole window.
   * Returns Float32Array(T * 314) with T = floor(audio.length / hop).
   */
  process(audio, numFrames) {
    const T = numFrames ?? Math.floor(audio.length / this.hopSize);
    const F = this.featureSize;
    const out = new Float32Array(T * F);
    let off = 0;
    for (const r of this.resolutions) off += r.write(audio, T, this.hopSize, out, F, off);
    return { feats: out, numFrames: T };
  }
}
