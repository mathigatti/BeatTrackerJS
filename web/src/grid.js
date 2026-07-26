// The beat grid: a single source of truth for the beat index, the period and
// the position inside the bar.
//
// madmom's DBN sometimes skips a beat instead of reporting it late. If the grid
// relied only on what it reports, it would go quiet for half a bar. When a long
// period passes with no news, the grid advances on its own and marks the beat
// as `predicted`.
//
// The fill requires high confidence, and that is not a detail: for the first
// ~20 s madmom's online DBN wanders and leaves gaps of up to 4 periods *on
// purpose* (measured: 9 gaps in the first 20 s of one song, none of them with a
// real beat inside). Filling there recovers nothing and only adds noise —
// without the confidence condition, F against offline madmom drops from 0.970
// to 0.934.

const MIN_BPM = 70;
const MAX_BPM = 180;
const FILL_AFTER = 1.25;      // periods to wait before giving a beat up for lost
const FILL_CONFIDENCE = 0.58; // the same threshold at which the UI says "locked"

export class BeatGrid {
  constructor(meter = 4) {
    this.meter = meter;
    this.barOrigin = 0;
    this.reset();
  }

  reset() {
    this.index = 0;
    this.anchor = null;   // time (s) of the grid's last beat
    this.period = null;
    this.ibis = [];
    this.started = false;
  }

  get bpm() {
    return this.period ? 60 / this.period : 0;
  }

  get confidence() {
    if (this.ibis.length < 3 || !this.period) return Math.min(0.25, 0.05 * this.ibis.length);
    const recent = this.ibis.slice(-8);
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
    const varc = recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length;
    return Math.max(0, Math.min(1, 1 - 4 * (Math.sqrt(varc) / this.period)));
  }

  inBar(index) {
    const m = this.meter;
    return (((index - this.barOrigin) % m) + m) % m + 1;
  }

  /**
   * One hop. `now` in seconds, `beatDetected` if the DBN reported a beat here,
   * `silent` if the gate is closed.
   * Returns the beats that fired: [{ index, inBar, predicted, time }], where
   * `time` is when the beat actually landed (predicted ones arrive a little
   * late: they are only given up for lost FILL_AFTER periods after the previous
   * one).
   */
  step(now, beatDetected, silent) {
    const out = [];
    if (beatDetected) {
      const beat = this._register(now, false);
      if (beat) out.push(beat);
    } else if (!silent && this.period && this.anchor !== null && this.confidence >= FILL_CONFIDENCE) {
      while (now - this.anchor >= FILL_AFTER * this.period) {
        const beat = this._register(this.anchor + this.period, true);
        if (beat) out.push(beat);
      }
    }
    return out;
  }

  _register(now, predicted) {
    if (this.anchor === null) {
      // the first beat only anchors the grid, it is not displayed
      this.anchor = now;
      this.index = 0;
      this.started = true;
      return null;
    }
    const gap = now - this.anchor;
    if (this.period && gap < 0.4 * this.period) {
      // the same beat reported twice: re-anchor without advancing
      this.anchor = now;
      return null;
    }
    const steps = this.period ? Math.max(1, Math.round(gap / this.period)) : 1;
    // predicted beats last exactly one period: counting them towards the tempo
    // would inflate the confidence for free
    if (!predicted && (this.period === null || steps === 1)) this._pushIbi(gap / steps);
    this.index += steps;
    this.anchor = now;
    return { index: this.index, inBar: this.inBar(this.index), predicted, time: now };
  }

  _pushIbi(ibi) {
    if (ibi < 60 / MAX_BPM || ibi > 60 / MIN_BPM) return;
    this.ibis.push(ibi);
    if (this.ibis.length > 12) this.ibis.shift();
    if (this.ibis.length < 3) return;
    const sorted = this.ibis.slice(-8).sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    this.period = this.period === null ? median : 0.7 * this.period + 0.3 * median;
  }
}
