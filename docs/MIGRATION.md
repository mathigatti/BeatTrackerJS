# Porting madmom to the browser

Technical log of the port of `madmom` (online beat tracking) to plain
JavaScript, running entirely in the browser. No backend, no WASM, no ONNX.

The result produces **exactly the same beats as madmom** — 68 out of 68 over 45 s
of real music, 0.000000 s of difference — at **19% of a core**.

---

## 1. Why madmom and not BeatNet+

BeatNet+ looked like the obvious thing to port: it is PyTorch, it exports to ONNX
in 3.08 MB and runs at 0.32 ms/frame. madmom is neither PyTorch nor TensorFlow —
it implements its own layers in NumPy (`madmom/ml/nn/layers.py`) with pickled
weights, so there is no automatic export path at all.

But measured on the same audio, BeatNet+ clearly lost:

| | madmom online | BeatNet+ |
|---|---|---|
| F-measure (60 s of pop) | **0.794** | 0.526 |
| live BPM deviation | **0.41** | 6.13 |
| mean confidence | **0.80** | 0.18 |

Porting the easy one would have meant shipping the worse engine. And one fact
changed the arithmetic: **madmom's online network is tiny** — 3 LSTM layers
(162→25→25→25) plus a 25→1 dense, about 30k parameters per net. Rewriting that by
hand is a matter of hours, not weeks.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full comparison.

---

## 2. The strategy: export data, don't translate code

The central decision of the port. Anything that is a **fixed structure** gets
dumped from Python as binary data; only **algorithms** get rewritten in JS.

Exported as data:

- The weights of the 8 nets in the ensemble.
- The logarithmic filterbank (sparse 1025×81 matrix).
- The DBN's complete HMM (3213 states, 5009 transitions in CSR).
- The Hann window.

Rewritten in JS:

- The FFT.
- The LSTM forward pass with peepholes.
- The HMM forward pass.
- The spectrogram diff.

The reasoning: **rebuilding the generation of a structure in JS is exactly where
you drift without noticing.** The filterbank, for instance, depends on
`fref=440`, `norm_filters`, `unique_filters` and logarithmic spacing of 12 bands
per octave. Reimplementing that would give a *similar* matrix, and the model
would produce plausible but wrong numbers with no visible error anywhere.
Exporting the matrix eliminates that entire class of bugs.

Everything goes into a single `.bin` of concatenated float32 arrays, plus a
`.json` with offsets and shapes. Total: **1.05 MB**.

---

## 3. The verification method

This is the most important part of the port. **Every stage is compared against
madmom's real output**, not against an expectation.

```
audio → [features] → [LSTM ensemble] → [DBN] → beats
          ↓              ↓                ↓        ↓
        test           test             test     test
```

Besides the weights, the Python exporter dumps per-stage reference vectors: raw
2048-sample frames, their 162-dim features, the activation of each of the 8 nets,
the HMM's argmax state per frame, and the final beat times. Each JS test
reproduces one stage and compares.

Without this the port would be uncheckable: a beat tracker that "kind of works"
is indistinguishable from one with a subtle bug.

### The rule that had to be learned

The first network test gave **4.5e-2** of error and the code was correct. The
error decreased frame by frame — the signature of a different initial state, not
of a broken forward pass. Python came in with 100 frames of accumulated recurrent
state while JS started from `init`.

> **Any parity test on a recurrent model has to pin the initial state**, or the
> numbers mean nothing.

Once the export was changed to start at frame 0 with the nets reset, the
comparison became exact.

---

## 4. How each piece was ported

### 4.1 Features

Chain equivalent to `RNNBeatProcessor(online=True)`:

```
frame(2048) → Hann → |FFT| → filterbank(81) → log10(1+x) → positive diff
            → concat[spec, diff] = 162 dims
```

Parameters extracted from the real pipeline, not from the README:
`frame_size=2048`, `hop=441`, `fps=100`, `num_bands=12`, `fmin=30`, `fmax=17000`,
`fref=440`, `log10(1 + 1·x)`, `diff_ratio=0.5` → `diff_frames=1`,
`positive_diffs=True`.

`diff_frames` was not computed by hand: it is read back from the processor after
one pass.

Own FFT, iterative radix-2 with bit reversal, 2048 points.

### 4.2 The network

8 nets averaged. Each one: 3 `LSTMLayer` + 1 sigmoid `FeedForwardLayer`.

**madmom's LSTMs are not standard.** They have *peephole connections* and a
specific operation order (`madmom/ml/nn/layers.py:LSTMLayer.activate`):

- input gate and forget gate look at the **previous** state
- the state is updated: `state = cell·ig + state·fg`
- the output gate looks at the **already updated** state
- output: `tanh(state)·og`

A port using a generic library LSTM (PyTorch, standard ONNX) would give plausible
but wrong results: the common libraries have no peepholes, and even where they
do, the order differs.

### 4.3 The DBN

This was the highest-risk piece and turned out to be the cleanest.

madmom only ships the compiled `.so` of `madmom/ml/hmm` — no Cython source is
available locally. Instead of guessing the algorithm, it was **derived
empirically**: a candidate forward pass was implemented in NumPy and compared
against `hmm.forward()` until it reached **5.5e-9** of difference and 100% argmax
agreement.

What it turned out to be:

- CSR **by destination**: for state `s`, the incoming transitions are
  `from_states[pointers[s]..pointers[s+1]]` with their probabilities.
- **Linear** densities, not logarithmic: `[(1-a)/(λ-1), a]` with `λ=16`.
- The forward pass is **normalised to sum 1** on every frame.
- Uniform initial distribution (1/3213).

Beat emission: a beat is reported when the most likely state falls in the beat
range (`obs_pointers[s] == 1`) and the minimum interval since the previous one
has elapsed (`60/max_bpm`).

The forward pass runs in **float64**: repeated normalisation accumulates error
fast in float32.

### 4.4 Capture

An `AudioWorklet` collects the browser's 128-sample blocks into hops of 441 and
posts them to the main thread with buffer transfer.

The page can listen to the microphone, to a system monitor source (on Linux,
PipeWire/PulseAudio expose "Monitor of ..." as regular input devices) or to
another tab's audio via `getDisplayMedia`.

---

## 5. The two times the test failed and the code was right

Both were errors in **how the reference was generated in Python**, not in the JS.
Worth recording because the pattern repeats.

**DBN.** Beat times came out exactly twice as large. Cause: the exporter
incremented `counter` by hand without noticing that `process_online` already does
it (`madmom/features/beats.py:1032`).

**Features.** The spectrum matched at 1.19e-7 but the diff was completely wrong.
Cause: the exporter called the preprocessing without `reset=False`, so the diff
buffer restarted on every frame and all 81 columns came out as zeros.

In both cases **the shape of the error is what gave it away, not its magnitude**:
exactly 2×, and zeros where there should not have been any. Loosening tolerances
to "make the test pass" would have buried both bugs.

A third one appeared later with the downbeat model: activations shifted by
exactly 3 frames. Cause: the downbeat pipeline uses **centred** framing, not
`origin='stream'` like the beat model.

---

## 6. Results

### Parity

| stage | max difference |
|---|---|
| features (log spectrum) | 1.19e-7 |
| features (positive diff) | 1.79e-7 |
| LSTM ensemble (mean) | 2.24e-8 |
| LSTM ensemble (individual net) | 3.87e-7 |
| DBN (argmax state) | 100% agreement |
| DBN (beat times) | exact |
| **end-to-end, 45 s of real music** | **68/68 beats, 0.000000 s** |

All within float32 rounding noise.

### Performance

**1.9 ms per 10 ms hop → 19% of a core** (plain JS, Node 20). The original Python
measures ~12% on the same machine, and that gap is the price of the port.

No WASM and no ONNX were needed. The imagined bottleneck — the network — turned
out to be irrelevant: 8 nets of 30k parameters are nothing.

The downbeat model is a different story: one 6 s window costs ~2.7 s of compute,
which is why it runs in a Web Worker every 6 s (~46% of a core) instead of on the
main thread.

### Size

1.05 MB for the beat model (8 nets + filterbank + HMM + window), plus 1.64 MB for
the downbeat ensemble. 2.7 MB total.

---

## 7. How to regenerate

The exporters need `madmom==0.16.1`, which on Python 3.12 does not build with
build isolation because its `setup.py` imports Cython:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install cython numpy librosa
pip install --no-build-isolation madmom==0.16.1
```

Then, from the repo root:

```bash
AUDIO_E2E=/path/to/a/song.mp3 python tools/export_madmom.py
python tools/export_downbeats.py 3     # 3 = ensemble size

node test/parity.mjs            # LSTM ensemble
node test/parity_features.mjs   # features
node test/parity_dbn.mjs        # HMM
node test/parity_e2e.mjs        # full pipeline
node test/parity_downbeats.mjs  # downbeat model

python3 serve.py 8090           # then open localhost:8090
```

`AUDIO_E2E` can be any song; the fixtures it generates weigh ~11 MB, which is why
they are not versioned.

### A NumPy 2 gotcha

`DBNDownBeatTrackingProcessor` crashes under NumPy 2: it calls `np.asarray` on a
ragged list of `(path, log_prob)` pairs (`madmom/features/downbeats.py:287`),
which NumPy 2 refuses. The Python reference backend works around it with a scoped
shim that patches `np.asarray` only for that shape and only while the decoder
runs.

---

## 8. What is missing

- **Not tested with a microphone in a real browser.** The tests run in Node
  against audio files. `AudioWorklet` capture is written but unverified live.
- **The meter does not come from madmom's decoder.**
  `DBNDownBeatTrackingProcessor` is a whole other HMM, with one state space per
  candidate bar length, and it is not ported. Instead the web scores each
  candidate (3, 4 and 6) over the downbeat activations that *are* ported. **And
  it turns out to be better this way**: see section 10.
- Automatic mode only tries 3, 4 and 6. A 5/4 has to be pinned by hand in the UI.
- madmom's online DBN warm-up of ~20-25 s is inherited as-is: it is forward-only
  and needs to accumulate evidence before locking.

---

## 9. The three pieces that did not come from madmom

Porting the model was not enough: the web inherited exactly the same holes the
Python backend had patched by hand, plus one of its own.

### 9.1 Silence gate

The symptom: on noise the DBN emits a perfectly even grid, and an even grid is
exactly what the confidence metric rewards. With noise at -35 dBFS the page
showed 102 BPM at 90% confidence.

Python cuts by level (smoothed rms × 18, threshold 0.02). Copying that was not
enough: it covers digital silence but room noise at -45/-35 dBFS sails straight
through (measured: 49 phantom beats in 30 s with the level cut in place).

What actually separates them is not *how large* the activation is but *how
spiky*. Measured over 45 s of music against white noise at four levels:

| statistic | music | noise |
|---|---|---|
| mean activation | 0.035 | 0.025 |
| decayed peak (τ≈7 s) | 0.13 | 0.12 |
| **p97/median over 5 s** | **min 3.5, median 2439** | **max 2.8** |

The mean and the peak separate nothing. The contrast separates by three orders of
magnitude. Threshold at 3.0: zero phantom beats at every noise level, 65 of 68
real beats kept (the 3 missing ones are the first 2 s, while the statistic fills).

### 9.2 Filling in skipped beats

First attempt, copying Python: fill when 1.5 periods pass with no beat. Result on
real music: **F against offline madmom dropped from 0.970 to 0.934**, with 5
invented beats and zero recovered.

The reason shows up when you look at where the gaps fell: for the first ~20 s the
online DBN wanders and leaves gaps of up to 4 periods, and looking at the
reference, *none* of those gaps has a real beat inside. They are not skipped
beats: it is the decoder not having locked yet. Filling there is inventing.

With the fill conditioned on confidence ≥ 0.58, F goes back to 0.970 and the fill
never fires once on that song — once locked, madmom skips nothing. It stays as a
safety net, verified separately against a synthetic 120 BPM grid with beats
deleted by hand: it recovers them with 0 ms of error and without shifting the
position in the bar.

The wait was also lowered from 1.5 to 1.25 periods: a fill is detected late by
definition, and with high confidence there is no need for that much grace. The
event carries the time *of the beat*, not of the hop where it was noticed, so the
phase ring stays in place even though the flash arrives late.

### 9.3 Meter

See section 8: it is our own criterion over the ported activations, not a port of
madmom's decoder. What took tuning was the hysteresis, not the criterion. A
single 6 s window is wrong often (on a waltz: 3 right and 2 wrong out of 9
windows), so the decision accumulates with a memory of ~7 windows and requires
the new candidate to beat the current one by 1.3×. One more condition was needed:
a minimum score before allowing any change, because the first window of a 4/4
song voted 3 (candidate 4 gathered no evidence at all) and only the third one
corrected it.

---

## 10. The chacarera case: why porting the decoder was not the answer

Testing with chacareras (Argentine folk, 6/8 with a 3/4 hemiola), the page showed
4/4 with a weak phase vote. The obvious hypothesis was that the real decoder was
missing.

That can be checked without porting anything: run madmom's own
`DBNDownBeatTrackingProcessor` on the same file, varying `beats_per_bar`.

```
  candidates [3, 4]     -> meter 4   166.7 BPM  bar every 1.42s
  candidates [3]        -> meter 3
  candidates [6]        -> meter 6   166.7 BPM  bar every 2.14s
  candidates [3, 4, 6]  -> meter 6   <- the correct answer
```

madmom's decoder with `[3,4]` gives **the same wrong 4**. The problem was never
the decoder: it was that 6 was not on the candidate list.

And adding 6 surfaces the opposite problem — madmom with `[3,4,6]` answers **6
for The Blue Danube and for Waltz of the Flowers**, grouping two waltz bars into
one. Its likelihood does not penalise the multiple.

The algebra explains both. With `contrast = (best − mean) / mean`, calling `h` the
downbeat activation on a real downbeat and `l` that of an ordinary beat:

| true meter | contrast(3) | contrast(6) | ratio 6/3 |
|---|---|---|---|
| 3 | 2(h−l)/(h+2l) | 4(h−l)/(2h+4l) | **1.00** |
| 6 | 2(h−l)/(h+5l) | 5(h−l)/(h+5l) | **2.50** |

Over a bar of 3 they tie exactly; over a bar of 6, 6 wins by 2.5×. So it is
enough to **break ties in favour of the smaller meter**: a large candidate only
displaces a smaller one if it beats it by `tieMargin` (1.3). That is the rule
madmom is missing, and it is what allows 3 and 6 on the same list.

Lesson: the answer was not in porting more code, it was in measuring what the
original answered before assuming the original was right.

### And yet, 2 out of 3 chacareras still fail

With 6 on the list and the tie-break in place, over seven songs of known meter:

| song | true meter | result |
|---|---|---|
| The Blue Danube | 3/4 | 3 ✓ |
| Waltz of the Flowers | 3/4 | 3 ✓ |
| Chacarera del rancho | 6/8 | **6** ✓ |
| La Añera | 6/8 | 4 ✗ |
| Chacarera santiagueña | 6/8 | 4 ✗ |
| synthetic 124 BPM | 4/4 | 4 ✓ |
| pop track | 4/4 | 4 ✓ |

Adding 6 broke nothing, but it did not solve the general case either. The
per-window detail says why, and there are two distinct causes:

**La Añera** oscillates (4, 6, 4, 6, 6, 4, 4, 3) with contrasts from 0.4 to 1.5
and no clear winner. Compared to the 2.7 of a clean 4/4, that is a genuine "don't
know": the downbeat evidence exists but does not discriminate.

**Chacarera santiagueña** is worse: candidate 6 **gathers no evidence in 8 of 9
windows**. Scoring a bar of 6 requires at least 2 beats in each of the 6 phase
classes, i.e. 12 beats inside the 6 s window. At that tempo they do not fit. The
candidate never even competes.

The second has a known fix (a longer window for large meters, at the cost of more
latency and more CPU); the first probably does not, at least not with this
downbeat model. That is why the UI has a manual meter selector: with ambiguous
folk material, the honest answer from the automatic mode is to hesitate.
