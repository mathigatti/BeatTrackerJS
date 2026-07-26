// Decodes the downbeat activations into "the bar is N beats long and the 1
// falls on this position".
//
// Instead of running a second decoder with its own grid (which would then fight
// the beat tracker's), every beat inside the window votes for its position in
// the bar using the downbeat activation it landed on. That is the same approach
// the Python backend uses for the phase; the meter choice here is our own
// (Python uses madmom's DBNDownBeatTrackingProcessor, which is a whole other HMM
// and is not ported).

const PEAK_RADIUS = 2;   // frames on each side to search for the peak

// Careful with this list: a multiple of the true meter scores at least as well
// as the true meter (with 8 over a 4/4, phase class {0} also contains only real
// downbeats). 6 can coexist with 3 only because of the tie-break in MeterTally:
// the algebra says that over a bar of 3 the contrasts of 3 and 6 tie exactly,
// and over a bar of 6 the contrast of 6 is 2.5x that of 3. Breaking ties towards
// the smaller meter keeps a waltz at 3 and lets a chacarera reach 6.
export const CANDIDATES = [3, 4, 6];

/** Peak downbeat activation around each beat of the window. */
function beatPeaks(activations, T, startFrame, beats) {
  const out = [];
  for (const { index, frame } of beats) {
    const t = frame - startFrame;
    if (t < PEAK_RADIUS || t >= T - PEAK_RADIUS) continue;
    let peak = 0;
    for (let k = -PEAK_RADIUS; k <= PEAK_RADIUS; k++) {
      const v = activations[(t + k) * 2 + 1];
      if (v > peak) peak = v;
    }
    out.push({ index, peak });
  }
  return out;
}

/** Mean activation per position in the bar, or null if evidence is missing. */
function phaseScores(peaks, meter) {
  const scores = new Float64Array(meter);
  const counts = new Int32Array(meter);
  for (const { index, peak } of peaks) {
    const phase = ((index % meter) + meter) % meter;
    scores[phase] += peak;
    counts[phase] += 1;
  }
  // at least a couple of bars are needed in every position
  for (let p = 0; p < meter; p++) if (counts[p] < 2) return null;
  for (let p = 0; p < meter; p++) scores[p] /= counts[p];
  return scores;
}

function summarize(scores, meter) {
  let best = 0;
  for (let p = 1; p < meter; p++) if (scores[p] > scores[best]) best = p;
  const sorted = Array.from(scores).sort((a, b) => b - a);
  const margin = sorted[0] > 0 ? (sorted[0] - sorted[1]) / sorted[0] : 0;
  let mean = 0;
  for (let p = 0; p < meter; p++) mean += scores[p];
  mean /= meter;
  // how far the 1 stands out above the window average. Unlike the margin, this
  // IS comparable across different meters: with the wrong meter every phase
  // class mixes downbeats with ordinary beats and all positions end up close to
  // the average.
  const contrast = mean > 0 ? (scores[best] - mean) / mean : 0;
  return { origin: best, margin, contrast, scores: Array.from(scores) };
}

/**
 * activations: Float32Array(T*2) with [beat, downbeat] per frame.
 * startFrame:  global hop index of the window's first frame.
 * beats:       [{ index, frame }] with the beat index and its global hop.
 * Returns { origin, scores, margin, contrast } or null if there is no evidence.
 */
export function voteBarOrigin(activations, T, startFrame, beats, meter, minMargin = 0.25) {
  if (meter < 2) return null;
  const scores = phaseScores(beatPeaks(activations, T, startFrame, beats), meter);
  if (scores === null) return null;
  const result = summarize(scores, meter);
  return result.margin < minMargin ? null : result;
}

/**
 * Scores every candidate meter over the same window. Returns the ranking sorted
 * by contrast (best first), or null if no candidate gathered evidence. It
 * decides nothing on its own: a single 6 s window is wrong often (measured on a
 * waltz: 3 right and 2 wrong out of 9 windows), so the decision is accumulated
 * outside.
 */
export function estimateBar(activations, T, startFrame, beats, candidates = CANDIDATES) {
  const peaks = beatPeaks(activations, T, startFrame, beats);
  const ranking = [];
  for (const meter of candidates) {
    const scores = phaseScores(peaks, meter);
    if (scores === null) continue;
    ranking.push({ meter, ...summarize(scores, meter) });
  }
  if (ranking.length === 0) return null;
  ranking.sort((a, b) => b.contrast - a.contrast);
  return { ...ranking[0], ranking };
}

/**
 * Accumulates the ranking across windows. The meter is decided with memory
 * because changing it remaps every position in the bar at once.
 *
 * The score is a moving AVERAGE of the contrast, not a sum, and that detail is
 * what makes it survive a playlist. Summing looks equivalent but is not: a sum
 * saturates at contrast/(1-decay), so a leader that built up a large total can
 * only be displaced by a challenger whose typical contrast is higher than the
 * leader's ever was. Measured on a chacarera followed by a 4/4 track, the switch
 * back needed 8.46 > 1.3 x 6.48 and only just made it — with a slightly weaker
 * second song it would have stayed on 6 forever. Averaging compares typical
 * contrasts instead of totals, so recovery only ever costs the memory horizon.
 */
export class MeterTally {
  constructor(
    meter,
    candidates = CANDIDATES,
    decay = 0.75,
    switchRatio = 1.3,
    minScore = 0.25,
    tieMargin = 1.3
  ) {
    this.meter = meter;
    // ascending: a multiple of the true meter scores at least as well as it
    // does, so on a tie the smaller one has to win (a waltz is 3, not 6)
    this.candidates = [...candidates].sort((a, b) => a - b);
    this.decay = decay;         // memory of ~4 windows (~24 s)
    this.switchRatio = switchRatio;
    this.tieMargin = tieMargin;
    // one window is not enough to change meter: measured, the first window of a
    // 4/4 song voted 3 (candidate 4 gathered no evidence at all) and only the
    // third one corrected it
    this.minScore = minScore;
    this.scores = new Map(candidates.map((m) => [m, 0]));
  }

  reset(meter = this.meter) {
    this.meter = meter;
    for (const m of this.candidates) this.scores.set(m, 0);
  }

  /** Returns the meter in effect after adding this window. */
  push(ranking) {
    const seen = new Map(ranking.map((r) => [r.meter, r.contrast]));
    for (const m of this.candidates) {
      // a candidate that gathered no evidence this window decays towards zero:
      // not being scorable while the others are is itself evidence against it
      const observed = seen.get(m) ?? 0;
      this.scores.set(m, this.decay * this.scores.get(m) + (1 - this.decay) * observed);
    }
    // the smallest starts out winning; a larger one only displaces it if it
    // beats it by tieMargin, not by noise
    let best = this.candidates[0];
    for (const m of this.candidates.slice(1)) {
      if (this.scores.get(m) > this.scores.get(best) * this.tieMargin) best = m;
    }
    if (
      best !== this.meter &&
      this.scores.get(best) >= this.minScore &&
      this.scores.get(best) > this.switchRatio * this.scores.get(this.meter)
    ) {
      this.meter = best;
    }
    return this.meter;
  }
}
