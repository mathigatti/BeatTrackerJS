# Pulso

Live beat tracker that runs **entirely in the browser**: it listens to audio,
locks onto the tempo, shows which beat of the bar you are on and how hard it was
accented — and a little robot dances to it.

No backend, no WASM, no dependencies. [madmom](https://github.com/CPJKU/madmom)'s
neural model is ported to plain JavaScript and produces **exactly the same beats
as the original**, down to float32 rounding noise.

**▶ [Try it](https://pulso-web.netlify.app)** — hit "Listen to input" (microphone
or system monitor) or "Tab audio" to pick up whatever is playing in another tab.

---

## What it does

- **Tempo and phase** from madmom's online tracker (BLSTM + HMM), 1.9 ms per
  10 ms hop: 19% of a core.
- **Automatic meter** among 3, 4 and 6, or pinned by hand from 2 to 6.
- **Position inside the bar**, with the 1 located by the downbeat model.
- **Per-hit accent**: each beat lights up in proportion to the energy that
  actually sounded there, so the accent pattern becomes visible.
- **A dancing robot** whose arms swing on the beat and whose hips shift once per
  bar — in 4/4 you see four arm swings per hip shift.
- **Silence gate**, so it does not invent a tempo when there is no music.

## How it works

```
audio ─▶ AudioWorklet ─▶ features ─▶ 8x LSTM ensemble ─▶ HMM ─▶ beats
   │      (441 hops)     (FFT +       (with peepholes)  (forward)   │
   │                      filterbank                                ▼
   │                      + log + diff)                          grid
   │                                                       (tempo, meter)
   └─▶ rolling 6 s window ─▶ bidirectional BLSTM ─▶ phase vote ─▶ the "1"
             (in a Web Worker)     (downbeat model)
```

The beat tracker is causal and runs in real time. The downbeat model is
bidirectional — it cannot run in streaming — so it lives in a Web Worker over a
6 s window, and every beat votes for its position in the bar.

## Run locally

```bash
python3 serve.py 8090      # serves web/, no caching
```

It has to be over HTTP: ES modules and `AudioWorklet` do not work from `file://`.
In production HTTPS is required (the microphone demands it); Netlify handles that
for you.

## Tests

```bash
npm test                   # parity against madmom, stage by stage
node test/silence.mjs      # the gate: noise vs music
node test/gapfill.mjs      # filling in skipped beats
node test/meter.mjs a.f32  # meter on songs of known meter
```

The parity tests need fixtures that are not versioned (~11 MB). Regenerate them
with the exporters:

```bash
AUDIO_E2E=/path/to/a/song.mp3 python tools/export_madmom.py
python tools/export_downbeats.py 3
```

That needs `madmom==0.16.1` in Python (see
[docs/MIGRATION.md](docs/MIGRATION.md#7-how-to-regenerate)).

## Deploy

Static site, no build. `netlify.toml` already sets `publish = "web"`.

```bash
netlify deploy --prod
```

Or connect the repo from the Netlify dashboard: it picks up `netlify.toml` and
there is nothing to configure.

## Read more

- **[docs/MIGRATION.md](docs/MIGRATION.md)** — how madmom was ported to
  JavaScript: what was exported as data instead of translated, how it was
  verified stage by stage, and the three times the test failed while being right.
- **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** — which other trackers were
  tried before settling on madmom (BeatNet, BeatNet+, BEAST, a classic tracker
  with no ML), what metrics they were compared with, and what came out.

## Layout

```
web/                 what gets published
├── index.html, app.js, styles.css
├── worklet.js       collects 441-sample hops
├── downbeat-worker.js
├── src/             the port: features, network, HMM, grid, accent, meter
└── public/models/   exported weights (2.7 MB)
tools/               exporters from madmom (Python)
test/                parity and behaviour (Node)
docs/                the two long documents
serve.py             development server with caching disabled
```

## Credits

The model is [madmom](https://github.com/CPJKU/madmom)'s (Böck et al.), BSD.
This repo only ports it.

MIT.
