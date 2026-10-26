// Detector de ataques graves/agudos en vivo: la CNN de onsets de madmom (Schlüter & Böck) hasta su capa 5
// y la cabeza entrenada en tools/rt_head.py, exportadas por tools/export_rt_web.py. Se le pasan bloques de
// 441 muestras a 44100 Hz (10 ms) y devuelve los ataques a medida que se confirman.
//
// Latencia: cada frame de madmom está centrado en su instante y la ventana más larga es de 4096 muestras,
// así que mira 2048 muestras (46 ms) adelante; la CNN mira 7 frames (70 ms) a cada lado y el detector de
// picos 1 frame más: un ataque se informa ~120 ms después de que suena.

import { FFT } from "/src/features.js";

const CTX = 7;

export class OnsetDetector {
  static async load(base = "rt/") {
    const [meta, bin] = await Promise.all([
      fetch(`${base}onsets.json`).then((r) => r.json()),
      fetch(`${base}onsets.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new OnsetDetector(meta, bin);
  }

  constructor(meta, bin) {
    const all = new Float32Array(bin);
    this.a = {};
    for (const { name, shape, offset } of meta.arrays) {
      this.a[name] = { data: all.subarray(offset, offset + shape.reduce((x, y) => x * y, 1)), shape };
    }
    this.meta = meta;
    this.sizes = meta.frame_sizes;
    this.hop = meta.hop;
    this.maxN = Math.max(...this.sizes);
    this.ffts = Object.fromEntries(this.sizes.map((n) => [n, new FFT(n)]));
    this.classes = meta.clases;
    this.pick = meta.pick;
    this.reset();
  }

  reset() {
    this.samples = new Float32Array(this.maxN * 4);
    this.received = 0;          // muestras recibidas en total
    this.nextFrame = 0;         // próximo frame de espectrograma a calcular
    this.specs = [];            // frames de entrada a la CNN (80 x 3), los últimos 2*CTX+1
    this.cnnFrame = 0;          // próximo frame de salida de la CNN
    this.act = this.classes.map(() => []);
    this.off = 0;               // índice absoluto del primer valor guardado en act
    this.sum = this.classes.map(() => 0);
    this.sum2 = this.classes.map(() => 0);
    this.last = this.classes.map(() => -Infinity);
  }

  sample(i) {
    if (i < 0 || i >= this.received) return 0;
    return this.samples[i % this.samples.length];
  }

  /** Un frame de entrada: los tres espectrogramas log-mel (80 bandas) en el orden de madmom. */
  specFrame(k) {
    const out = new Float32Array(80 * 3);
    const c = k * this.hop;
    this.sizes.forEach((n, ch) => {
      const win = this.a[`window_${n}`].data, fb = this.a[`filterbank_${n}`].data;
      const re = new Float64Array(n), im = new Float64Array(n);
      for (let j = 0; j < n; j++) re[j] = this.sample(c - n / 2 + j) * win[j];
      this.ffts[n].transform(re, im);
      const bins = n / 2;
      for (let b = 0; b < 80; b++) {
        let s = 0;
        for (let f = 0; f < bins; f++) {
          const w = fb[f * 80 + b];
          if (w) s += Math.hypot(re[f], im[f]) * w;
        }
        out[b * 3 + ch] = Math.log(s + this.meta.log_add);
      }
    });
    return out;
  }

  /** Capa 5 de la CNN para el frame central de 2*CTX+1 frames de entrada. */
  cnn(frames) {
    const T = frames.length, mean = this.a.bn_mean.data, inv = this.a.bn_inv_std.data;
    const g = this.meta.bn_gamma, be = this.meta.bn_beta;
    const x = frames.map((f) => f.map((v, i) => (v - mean[i]) * inv[i] * g + be));  // T x (80*3)
    const conv = (inp, Tin, F, C, w, b, O, kt, kf) => {
      // convolución "valid" (madmom invierte el kernel: es convolución, no correlación), tanh, y
      // max-pool de 3 en frecuencia
      const To = Tin - kt + 1, Fo = F - kf + 1, Fp = Math.floor(Fo / 3);
      const out = new Float32Array(To * Fp * O);
      for (let t = 0; t < To; t++) {
        for (let o = 0; o < O; o++) {
          const z = new Float32Array(Fo);
          for (let f = 0; f < Fo; f++) {
            let s = b[o];
            for (let c = 0; c < C; c++) {
              for (let dt = 0; dt < kt; dt++) {
                const row = inp[t + dt];
                for (let df = 0; df < kf; df++) {
                  s += row[(f + df) * C + c] * w[((c * O + o) * kt + (kt - 1 - dt)) * kf + (kf - 1 - df)];
                }
              }
            }
            z[f] = Math.tanh(s);
          }
          for (let p = 0; p < Fp; p++) out[(t * Fp + p) * O + o] = Math.max(z[3 * p], z[3 * p + 1], z[3 * p + 2]);
        }
      }
      return { out, To, Fp };
    };
    const c1 = conv(x, T, 80, 3, this.a.conv1_w.data, this.a.conv1_b.data, 10, 7, 3);
    const rows1 = Array.from({ length: c1.To }, (_, t) => c1.out.subarray(t * c1.Fp * 10, (t + 1) * c1.Fp * 10));
    const c2 = conv(rows1, c1.To, c1.Fp, 10, this.a.conv2_w.data, this.a.conv2_b.data, 20, 3, 3);
    return c2.out;  // 7 frames x 8 x 20 = 1120, en el orden de la capa de stride de madmom
  }

  head(f5) {
    const { head_mu: mu, head_sd: sd, head_W0: W0, head_b0: b0, head_W1: W1, head_b1: b1 } = this.a;
    const H = b0.data.length, K = b1.data.length, D = mu.data.length;
    const h = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      let s = b0.data[j];
      for (let i = 0; i < D; i++) s += ((f5[i] - mu.data[i]) / sd.data[i]) * W0.data[i * H + j];
      h[j] = Math.max(0, s);
    }
    return Array.from({ length: K }, (_, k) => {
      let s = b1.data[k];
      for (let j = 0; j < H; j++) s += h[j] * W1.data[j * K + k];
      return 1 / (1 + Math.exp(-s));
    });
  }

  /** Detector de picos causal, igual que tools/realtime.py pick(): decide el frame k - lookahead. */
  pickAt(ci, k) {
    const la = this.pick.lookahead, cls = this.classes[ci], off = this.off;
    const env = (i) => this.act[ci][i - off];
    const L = Math.max(2, Math.round(this.pick.win_s * 100));
    const kk = k - la;
    if (kk < 3) return null;
    const a = Math.max(off, kk - L);
    let s = 0, s2 = 0;
    for (let i = a; i < kk; i++) { const e = env(i); s += e; s2 += e * e; }
    const n = kk - a, m = s / n, sd = Math.sqrt(Math.max(s2 / n - m * m, 0));
    const v = env(kk);
    if (v <= this.pick.piso[cls] || v < m + this.pick.delta[cls] * sd) return null;
    for (let i = kk - 3; i <= kk + la; i++) if (env(i) > v) return null;
    const t = kk / 100;
    if (t - this.last[ci] < this.pick.min_gap_s) return null;
    this.last[ci] = t;
    return { time: t, reg: cls, strength: v };
  }

  /** Recibe 441 muestras; devuelve los ataques que se pudieron confirmar ([{time, reg, strength}]). */
  pushHop(hop) {
    for (let i = 0; i < hop.length; i++) this.samples[(this.received + i) % this.samples.length] = hop[i];
    this.received += hop.length;
    const found = [];
    // un frame de espectrograma necesita audio hasta su centro + la mitad de la ventana más larga
    while (this.nextFrame * this.hop + this.maxN / 2 <= this.received) {
      const f = this.specFrame(this.nextFrame++);
      if (this.specs.length === 0) for (let i = 0; i < CTX; i++) this.specs.push(f);  // como madmom: repite el primero
      this.specs.push(f);
      if (this.specs.length > 2 * CTX + 1) this.specs.shift();
      if (this.specs.length < 2 * CTX + 1) continue;
      const probs = this.head(this.cnn(this.specs));
      const k = this.cnnFrame++;
      probs.forEach((p, ci) => {
        this.act[ci].push(p);
        const o = this.pickAt(ci, k);
        if (o) found.push(o);
      });
      if (this.act[0].length > 1000) {  // alcanza con 1.5 s de historia para el umbral
        for (const a of this.act) a.splice(0, 500);
        this.off += 500;
      }
    }
    return found;
  }
}
