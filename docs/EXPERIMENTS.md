# What was tried before settling on madmom

Log of the comparison between beat trackers. The goal was a single thing:
**follow the pulse live**, with audio arriving in streaming, not analysing a
whole file offline.

**madmom online** won. This document is why, with the numbers.

---

## 1. The method

### The metric

**Beat F-measure with a ±70 ms tolerance**, the standard tolerance in the beat
tracking literature. An estimated beat counts as a hit if there is an unused
reference beat within 70 ms; each reference can be used only once, so emitting
twice as many beats does not inflate recall.

### The reference

There are no human annotations for these songs, so the reference is **madmom's
offline tracker** (`RNNBeatProcessor` + `DBNBeatTrackingProcessor`, bidirectional
BLSTM + Viterbi over the whole track). It is the reasonable ceiling: it sees the
future, it does not run live, and it is what an online tracker is trying to
approximate.

This has a limit worth stating up front: **it measures similarity to offline
madmom, not musical correctness**. Where offline madmom is wrong, the metric
rewards being wrong in the same way. That is why the conclusions about meter also
lean on songs whose meter was known in advance.

### The material

Chosen to cover different cases, not to produce good numbers:

| song | why |
|---|---|
| synthetic 124 BPM | control floor: if this fails, it is broken |
| pop track with a marked pulse | the easy real case |
| Waltz of the Flowers (Tchaikovsky) | 3/4, orchestral, soft onsets |
| The Blue Danube (Strauss) | 3/4 with rubato: the hard case |
| Take Five (Brubeck) | 5/4, to see what happens outside 3 and 4 |
| 3 chacareras | 6/8 with hemiola, genuinely ambiguous meter |

### How "live" was measured

Offline numbers are not enough: a tracker can have a good F over a file and be
unusable in streaming. It was also measured with real audio coming through the
sound card:

- **BPM deviation** over the last third of the run (once locked)
- **mean confidence** reported by the engine
- **CPU per 10 ms hop**, against a 10 ms budget

---

## 2. The candidates

### Classic tracker, no ML

Spectral-flux onset detection + autocorrelation, written by hand. The baseline
for everything below.

| | F-measure | CPU |
|---|---|---|
| classic tracker | **0.209** | 5% of a core |
| madmom online | **0.855** | 12% of a core |

The interesting part: **both get the tempo right** (113.2 BPM). What the classic
one gets wrong is the **phase** — and it is not a constant offset: correcting for
the best global offset only brings it to 0.466. It finds how fast the music is
going but not where the hits land.

This mattered for the initial diagnosis: the madmom prototype *appeared* worse
than the classic one live. It was not the model, it was an architecture bug —
inference blocked the audio drain and the buffer froze, consuming 10% of the
incoming audio. Measuring it is what exposed it.

### BeatNet 1 (`mjhydri/BeatNet`)

Discarded. It loses to everything on everything measured.

F-measure over 60 s of each song:

| song | madmom online | BeatNet 1 (best of 3 models) | BeatNet+ |
|---|---|---|---|
| synthetic 124 BPM | 0.996 | 0.964 | 0.992 |
| pop track | **0.794** | 0.539 | 0.526 |
| Waltz of the Flowers | **0.535** | 0.330 | 0.354 |
| The Blue Danube | 0.132 | 0.269 | **0.326** |
| Take Five | **0.985** | 0.709 | 0.741 |

Meter, derived by counting beats between downbeats:

| song | true | BeatNet 1 (m1) | BeatNet 1 (m2) | BeatNet+ |
|---|---|---|---|---|
| synthetic | 4 | 2 ✗ | 2 ✗ | 4 ✓ |
| pop track | 4 | 4 ✓ | 4 ✓ | 4 ✓ |
| Waltz of the Flowers | 3 | 3 ✓ | 4 ✗ | 3 ✓ |
| The Blue Danube | 3 | 4 ✗ | 3 ✓ | 3 ✓ |
| Take Five | 5 | 4 ✗ | 4 ✗ | 4 ✗ |

BeatNet+ gets 4 of 5; BeatNet 1 gets 2 of 5 with any of its models. There is not a
single case where BeatNet 1 wins.

**Upstream bugs found along the way:**

- `particle_filtering_cascade.py:269` crashes with a single-element
  `beats_per_bar` (`TypeError: len() of unsized object`).
- The second output column (`:284-289`) is **binary** — `1` = downbeat, `2` =
  ordinary beat — not the position in the bar. This cost a wrong conclusion
  before it was noticed: "meter 2 on every song" was reported by reading that
  column as if it were the beat number.
- The particle filter has `min_beats_per_bar=2`, `max_beats_per_bar=4` hardcoded:
  no version of BeatNet can reach 5/4 or 7/4. That is why Take Five gives 4 in
  the table above.
- The three models (`--model 1/2/3`) were selected with a flag that **never did
  anything**: the wrapper always instantiated `BeatNetPlusInference`, which takes
  weights by path and not a model id.

### BeatNet+ (`mjhydri/BeatNet-Plus`)

The serious rival. Discarded too, but for finer reasons.

Before comparing it, a wrapper bug had to be fixed that made it look much worse
than it was: `_sync_from_classic_tracker` overwrote every state field with the
classic tracker's and only emitted events on downbeats, so **the engine was
actually the classic tracker** (F≈0.21) with downbeat nudges. Once fixed,
BeatNet+ genuinely carries the clock and does well on 4/4 with a marked pulse.

It still loses **live**, which is what matters. Same audio, 40-45 s, with the
capture routing verified on both engines:

| engine | BPM | range | deviation (last third) | mean confidence |
|---|---|---|---|---|
| madmom | 94.3 | 94–106 | **0.41** | **0.80** |
| BeatNet+ (`ig_threshold` 0.45) | 105.1 | 101–131 | **6.13** | 0.18 |

Lowering `--ig-threshold` from 0.55 to 0.45 did not help: the deviation got worse,
from 2.76 to 6.13. The threshold was not the cause. The particle filter is
stochastic and on this material it is genuinely unstable.

Its automatic meter was not reliable live either (it oscillated 4→3→4→2 on a
waltz), though offline over the whole track it got it right.

**Where it does win:** orchestral music with rubato. On The Blue Danube it scores
F=0.326 against madmom's F=0.132, which collapses there. If the goal were
classical repertoire, it would be worth dusting off.

**And it was the comfortable candidate to port to the web:** it exports to ONNX
in 3.08 MB and runs at 0.32 ms per frame (1.6% of a core), with 1e-05 parity
against PyTorch after 200 chained frames. With one trap: the model is *stateful*
(`self.hidden`/`self.cell` are mutated on every forward), and a naive
`torch.onnx.export` freezes the LSTM state as a constant in the graph. The model
then produces garbage **silently** — measured `max |diff| = 0.84` against
PyTorch. You have to wrap it to expose `h0`/`c0` as inputs and `hn`/`cn` as
outputs.

### BEAST

Could not be tested. It is newer (streaming Transformer) and promising, but its
repo does not publish the checkpoint: the README asks you to download
`BEAST_param.pt` from Google Drive by hand, and from this machine Drive returned
a login page instead of the file. Still pending.

### madmom online — the winner

`RNNBeatProcessor(online=True, origin='stream', num_frames=1)` +
`DBNBeatTrackingProcessor(online=True)`.

| | value |
|---|---|
| F-measure against its own offline version | **0.855** |
| CPU (Python) | 12% of a core |
| CPU (ported to JS, in the browser) | 19% of a core |
| warm-up before locking | 20-25 s |
| IBI std once locked | 6-10 ms |

Two warnings that apply to anyone using it:

- **The warm-up is long and it is inherent.** The online DBN is forward-only: it
  cannot revisit the past, so it needs to accumulate evidence. Until ~20 s it
  reports erratic tempos.
- **Rescaling audio to the int16 range makes it worse.** It sounds like the right
  thing because its spectrogram is `log10(1 + spec)`, so input scale matters.
  Measured on real music: **F 0.65 with rescaling against 0.86 without it.** The
  reasoning said it should help; the measurement said otherwise and it stayed
  off.

---

## 3. Meter: more candidates is not better

Using madmom's meter decoder (`DBNDownBeatTrackingProcessor`), which runs one HMM
per candidate bar length and keeps the one with the highest log-probability.
Offline, 90 s per song:

| song | true | candidates `3,4` | candidates `3,4,5,7` |
|---|---|---|---|
| The Blue Danube | 3/4 | 3 ✓ | **7 ✗** |
| Waltz of the Flowers | 3/4 | 3 ✓ | 3 ✓ |
| Take Five | 5/4 | 4 (not a candidate) | 5 ✓ |
| pop track | 4/4 | 4 ✓ | — |

**Adding candidates breaks things that worked.** It is the same phenomenon that
later showed up with 6 on the chacareras: a multiple (or near-multiple) of the
true meter scores as well as the true meter, and madmom's likelihood does not
penalise it. The web solves it with an explicit tie-break in favour of the
smaller meter — see
[MIGRATION.md, section 10](MIGRATION.md#10-the-chacarera-case-why-porting-the-decoder-was-not-the-answer).

Live (6 s windows, not the whole track) the result depends on the beat grid
underneath:

| song | live meter | time to infer it |
|---|---|---|
| Waltz of the Flowers | 3 ✓ | ~28 s |
| pop track | 4 ✓ | immediate (it was the default) |
| The Blue Danube | 4 ✗ | never |

The Blue Danube fails because **the beat tracker collapses before the meter ever
matters**: F=0.132 against its own offline version (70.6 BPM against 83.3). Waltz
of the Flowers gives 0.535 and that is enough. In other words: the bottleneck
there is not meter inference but beat tracking on orchestral music with rubato.

---

## 4. What was left unvalidated

- **The downbeats were never validated against real annotations.** Only that the
  phase vote is stable across a track. Claiming "the 1 is in the right place"
  would need annotated material.
- **The reference is offline madmom, not a human.** See the warning in section 1.
- **BEAST was never run**, because of the checkpoint.
- **None of this was tested with a microphone in a real browser** — the tests run
  in Node against files.

---

## 5. Lessons that cost time

**The shape of an error identifies its cause better than its magnitude.** Three
bugs in the port were found this way, and in all three the reflex of "loosen the
tolerance" would have buried them:

- beat times exactly **2×** the expected ones → the exporter incremented a
  counter the library already incremented itself.
- activations shifted by **exactly 3 frames** → the downbeat pipeline uses
  *centred* framing, not `origin='stream'`.
- a diff that looked like nothing at all → a missing `reset=False` in the call.

**Measure before assuming.** Twice the reasoning pointed the wrong way and the
measurement corrected it: the int16 rescaling (which "should" help and makes
things worse), and the idea that porting madmom's meter decoder would fix the
chacareras (it gives the same error, and with 6 on the list it breaks the
waltzes).

**Verify the capture routing before comparing engines.** One BPM comparison table
had to be thrown away because one engine's capture had silently reverted to the
internal microphone.
