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
// A tempo that moves this much, while we were confidently tracking the old one,
// is a different track rather than the tracker drifting. Measured inside a
// single song the median still wobbles by up to 14% (octave-ish flips while the
// DBN settles), so the threshold sits well above that and only counts once the
// grid is locked — a false reset costs the meter it had already learned.
const NEW_SONG_RATIO = 0.2;
// The fill can lock the grid on an old tempo: if the new song is slower, the
// predicted beat lands before the real one, the real one is then taken as a
// duplicate and never measured (bad guy -> anvil stayed at 136 BPM for good while
// the DBN already said 100). So the gaps between *reported* beats are kept apart,
// and when RAW_AGREE of them agree with each other and not with the period, the
// grid takes them.
const RAW_AGREE = 4;
const RAW_TOL = 0.08;
const RAW_DIFF = 0.15;

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
    this.raw = [];        // gaps between beats the DBN reported, without the fill
    this.lastReal = null;
    this.started = false;
    // set for one step when the tempo jumps enough to mean a new track; the app
    // uses it to forget the meter it had accumulated for the previous song
    this.songChanged = false;
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
    this.songChanged = false;
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
    if (!predicted) this._pushRaw(now);
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

  _pushRaw(now) {
    if (this.lastReal !== null) {
      const gap = now - this.lastReal;
      if (gap >= 60 / MAX_BPM && gap <= 60 / MIN_BPM) this.raw.push(gap);
      if (this.raw.length > RAW_AGREE) this.raw.shift();
    }
    this.lastReal = now;
    if (!this.period || this.raw.length < RAW_AGREE) return;
    const sorted = [...this.raw].sort((a, b) => a - b);
    const med = sorted[sorted.length >> 1];
    if (sorted[0] < med * (1 - RAW_TOL) || sorted[sorted.length - 1] > med * (1 + RAW_TOL)) return;
    if (Math.abs(med - this.period) / this.period <= RAW_DIFF) return;
    this.songChanged = true;
    this.period = med;
    this.ibis = [...this.raw];
    this.raw = [];
  }

  _pushIbi(ibi) {
    if (ibi < 60 / MAX_BPM || ibi > 60 / MIN_BPM) return;
    this.ibis.push(ibi);
    if (this.ibis.length > 12) this.ibis.shift();
    if (this.ibis.length < 3) return;
    const sorted = this.ibis.slice(-8).sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    if (this.period === null) {
      this.period = median;
      return;
    }
    if (
      Math.abs(median - this.period) / this.period > NEW_SONG_RATIO &&
      this.confidence >= FILL_CONFIDENCE
    ) {
      // jump straight there instead of easing in: smoothing across a track
      // change would spend several seconds reporting a tempo neither song has
      this.songChanged = true;
      this.period = median;
      this.ibis = this.ibis.slice(-3);
      return;
    }
    this.period = 0.7 * this.period + 0.3 * median;
  }
}
