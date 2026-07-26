// Accent profile per position in the bar.
//
// madmom's beat activation is flat by design (measured: 0.01 contrast between
// the four positions of a 4/4), so it is useless for accents. Band energy is
// not: on real music it gives kick on 1 and 3, snare on 2 and 4.
//
// All of this is causal: on each beat it looks back into a short buffer.

const GROUPS = [
  { name: "low", max: 150 },
  { name: "mid", min: 150, max: 800 },
  { name: "high", min: 3000 },
];

const LOOKBACK = 7;   // frames to look back for the hit (70 ms)

// Instantaneous lighting: ratio between THIS hit's energy and the maximum over
// the last few beats. Below LOW it lights at MIN_ACCENT, above HIGH it lights
// fully, with a smooth transition. Hit-to-hit energy varies far more than an
// average, so the transition band is wide.
const LOW_RATIO = 0.45;
const HIGH_RATIO = 0.92;
const MIN_ACCENT = 0.08;
const RECENT_BEATS = 16;   // beats the normalisation runs over

// Averaged profile per position: there the spread measured on real music is
// only ~11%, so the thresholds have to be much tighter.
const PROFILE_LOW = 0.93;
const PROFILE_HIGH = 0.98;

export class AccentProfile {
  constructor(model, meter) {
    const freqs = model.filterbank.centerFreqs;
    this.groups = GROUPS.map((g) => {
      const idx = [];
      for (let b = 0; b < freqs.length; b++) {
        const f = freqs[b];
        if ((g.min === undefined || f >= g.min) && (g.max === undefined || f < g.max)) {
          idx.push(b);
        }
      }
      return { name: g.name, idx };
    });
    this.numGroups = this.groups.length;
    this.setMeter(meter);

    // circular buffer of per-group energies, to look back on each beat
    this.hist = [];
    for (let i = 0; i < LOOKBACK; i++) this.hist.push(new Float32Array(this.numGroups));
    this.histIdx = 0;
    this.frameEnergy = new Float32Array(this.numGroups);
    this.recent = [];        // energies of the last beats, for normalising
    this.lastBands = {};
  }

  setMeter(meter) {
    this.meter = meter;
    this.profile = Array.from({ length: meter }, () => new Float32Array(this.numGroups));
    this.counts = new Int32Array(meter);
  }

  reset() {
    this.setMeter(this.meter);
    for (const h of this.hist) h.fill(0);
    this.histIdx = 0;
    this.recent = [];
    this.lastBands = {};
  }

  /** Called on every hop with the LINEAR magnitude of the 81 bands. */
  pushFrame(spec) {
    const e = this.hist[this.histIdx];
    for (let g = 0; g < this.numGroups; g++) {
      const idx = this.groups[g].idx;
      let sum = 0;
      for (let k = 0; k < idx.length; k++) sum += spec[idx[k]];
      e[g] = sum / idx.length;
    }
    this.frameEnergy.set(e);
    this.histIdx = (this.histIdx + 1) % LOOKBACK;
  }

  /**
   * Called when a beat is detected, with its position in the bar (1..meter).
   * Returns the instantaneous energy of THAT hit and its intensity from 0 to 1,
   * normalised against recent beats. This is what lights the box: it reflects
   * what actually sounded at that moment, not a historical average.
   */
  onBeat(position) {
    const p = position - 1;
    // peak of each group over the recent window: the hit can land a few frames
    // before the DBN confirms the beat
    const peak = new Float32Array(this.numGroups);
    for (let i = 0; i < LOOKBACK; i++) {
      const e = this.hist[i];
      for (let g = 0; g < this.numGroups; g++) if (e[g] > peak[g]) peak[g] = e[g];
    }
    let energy = 0;
    for (let g = 0; g < this.numGroups; g++) energy += peak[g];
    energy /= this.numGroups;

    // averaged profile per position (still maintained: it tells which band
    // accents each position, even though it no longer drives the lighting)
    if (p >= 0 && p < this.meter) {
      const prof = this.profile[p];
      const n = this.counts[p];
      const alpha = n < 4 ? 1 / (n + 1) : 0.15;
      for (let g = 0; g < this.numGroups; g++) prof[g] = (1 - alpha) * prof[g] + alpha * peak[g];
      this.counts[p] = n + 1;
      this.lastBands[p] = Array.from(peak);
    }

    // moving reference: the maximum over the last beats. It adapts to the
    // material instead of depending on an absolute level.
    this.recent.push(energy);
    if (this.recent.length > RECENT_BEATS) this.recent.shift();
    let ref = 0;
    for (const e of this.recent) if (e > ref) ref = e;

    let intensity = 1;
    if (ref > 0 && this.recent.length >= 3) {
      const ratio = energy / ref;
      const t = Math.max(0, Math.min(1, (ratio - LOW_RATIO) / (HIGH_RATIO - LOW_RATIO)));
      intensity = MIN_ACCENT + (1 - MIN_ACCENT) * (t * t * (3 - 2 * t));
    }
    return { energy, intensity, bands: Array.from(peak) };
  }

  /** Are there enough bars yet for the profile to mean anything? */
  get ready() {
    let min = Infinity;
    for (let p = 0; p < this.meter; p++) min = Math.min(min, this.counts[p]);
    return min >= 3;
  }

  /**
   * For each position returns the per-band profile (normalised within each
   * band, useful to see "kick on 1 and 3, snare on 2 and 4") and an `accent`
   * from 0 to 1 for painting.
   *
   * It no longer drives the lighting (onBeat's instantaneous energy does), but
   * it is useful to know which band accents each position. The `accent` here
   * uses its own thresholds because the spread between averaged positions is
   * only ~11%.
   */
  snapshot() {
    const maxima = new Float32Array(this.numGroups);
    const totals = new Float32Array(this.meter);
    for (let p = 0; p < this.meter; p++) {
      let sum = 0;
      for (let g = 0; g < this.numGroups; g++) {
        if (this.profile[p][g] > maxima[g]) maxima[g] = this.profile[p][g];
        sum += this.profile[p][g];
      }
      totals[p] = sum / this.numGroups;
    }

    let hi = 0;
    for (let p = 0; p < this.meter; p++) hi = Math.max(hi, totals[p]);

    const out = [];
    for (let p = 0; p < this.meter; p++) {
      const bands = {};
      for (let g = 0; g < this.numGroups; g++) {
        bands[this.groups[g].name] = maxima[g] > 0 ? this.profile[p][g] / maxima[g] : 0;
      }
      const ratio = hi > 0 ? totals[p] / hi : 1;
      const t = Math.max(0, Math.min(1, (ratio - PROFILE_LOW) / (PROFILE_HIGH - PROFILE_LOW)));
      const smooth = t * t * (3 - 2 * t);
      out.push({
        position: p + 1,
        bands,
        accent: MIN_ACCENT + (1 - MIN_ACCENT) * smooth,
        ratio,
        total: totals[p],
      });
    }
    return out;
  }
}
