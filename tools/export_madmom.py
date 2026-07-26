#!/usr/bin/env python3
"""Dumps madmom's online beat model into a format the browser can read.

Output in web/public/models/:
  madmom-beats.bin    every float32 array concatenated
  madmom-beats.json   manifest with offsets/shapes + pipeline config
  reference.json      per-stage reference vectors, for the parity tests

The filterbank is exported as a sparse matrix (1.7% density) instead of
rebuilding the triangular filters in JS, which is where it is easy to drift.
"""
from __future__ import annotations

import collections
import collections.abc
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

collections.MutableSequence = collections.abc.MutableSequence  # type: ignore[attr-defined]
for _name, _val in (("float", np.float64), ("int", np.int_), ("complex", np.complex128)):
    if not hasattr(np, _name):
        setattr(np, _name, _val)
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]

from madmom.audio.filters import LogarithmicFilterbank  # noqa: E402
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # noqa: E402
from madmom.ml.nn import NeuralNetwork  # noqa: E402
from madmom.models import BEATS_LSTM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "models"
SAMPLE_RATE = 44100
FPS = 100
FRAME_SIZE = 2048
HOP = SAMPLE_RATE // FPS  # 441
NUM_BANDS, FMIN, FMAX, FREF = 12, 30, 17000, 440.0
MIN_BPM, MAX_BPM = 70, 180


class Blob:
    """Acumula arrays float32 y lleva el registro de offsets."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.entries: dict[str, dict] = {}

    def add(self, name: str, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        self.entries[name] = {
            "offset": len(self.buf),
            "shape": list(a.shape),
            "count": int(a.size),
        }
        self.buf.extend(a.tobytes())


def export_gate(blob: Blob, prefix: str, gate) -> dict:
    """Un gate/cell de madmom: weights, recurrent_weights, bias, peepholes."""
    blob.add(f"{prefix}.weights", gate.weights)
    blob.add(f"{prefix}.recurrent_weights", gate.recurrent_weights)
    blob.add(f"{prefix}.bias", gate.bias)
    has_peep = getattr(gate, "peephole_weights", None) is not None
    if has_peep:
        blob.add(f"{prefix}.peephole_weights", gate.peephole_weights)
    return {"peepholes": has_peep}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blob = Blob()

    # ---------------------------------------------------------------- red
    nets = []
    for idx, path in enumerate(BEATS_LSTM):
        nn = NeuralNetwork.load(path)
        layers = []
        for li, layer in enumerate(nn.layers):
            p = f"net{idx}.layer{li}"
            kind = type(layer).__name__
            if kind == "LSTMLayer":
                for part in ("input_gate", "forget_gate", "cell", "output_gate"):
                    export_gate(blob, f"{p}.{part}", getattr(layer, part))
                blob.add(f"{p}.init", layer.init)
                blob.add(f"{p}.cell_init", layer.cell_init)
                layers.append({
                    "type": "lstm",
                    "size": int(layer.cell.bias.shape[0]),
                    "input_size": int(layer.cell.weights.shape[0]),
                })
            elif kind == "FeedForwardLayer":
                blob.add(f"{p}.weights", layer.weights)
                blob.add(f"{p}.bias", layer.bias)
                layers.append({
                    "type": "dense",
                    "input_size": int(layer.weights.shape[0]),
                    "size": int(layer.weights.shape[1]),
                    "activation": layer.activation_fn.__name__,
                })
            else:
                raise RuntimeError(f"capa no soportada: {kind}")
        nets.append({"layers": layers})

    # --------------------------------------------------------- filterbank
    bin_freqs = np.fft.rfftfreq(FRAME_SIZE, 1.0 / SAMPLE_RATE)
    fb_obj = LogarithmicFilterbank(bin_freqs, num_bands=NUM_BANDS, fmin=FMIN,
                                   fmax=FMAX, fref=FREF, norm_filters=True)
    fb = np.asarray(fb_obj)
    # centre frequencies: used to group bands and measure accent
    blob.add("filterbank.center_freqs", np.asarray(fb_obj.center_frequencies))
    # CSC per band: for each band, its range of bins and their weights
    starts, lengths, weights = [], [], []
    for band in range(fb.shape[1]):
        col = fb[:, band]
        nz = np.nonzero(col)[0]
        starts.append(int(nz[0]))
        lengths.append(int(nz[-1] - nz[0] + 1))
        weights.append(col[nz[0]: nz[-1] + 1])
    blob.add("filterbank.weights", np.concatenate(weights))
    blob.add("filterbank.starts", np.asarray(starts))
    blob.add("filterbank.lengths", np.asarray(lengths))

    # Hann window, exactly as madmom applies it
    blob.add("window", np.hanning(FRAME_SIZE))

    # ----------------------------------------------------------------- DBN
    # The forward pass was verified against madmom (5.5e-9): CSR by destination,
    # linear densities [(1-a)/(lambda-1), a], normalised per frame.
    dbn = DBNBeatTrackingProcessor(min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                                   fps=FPS, online=True)
    blob.add("hmm.pointers", np.asarray(dbn.tm.pointers))
    blob.add("hmm.from_states", np.asarray(dbn.tm.states))
    blob.add("hmm.probabilities", np.asarray(dbn.tm.probabilities))
    blob.add("hmm.obs_pointers", np.asarray(dbn.om.pointers))
    blob.add("hmm.initial", np.asarray(dbn.hmm.initial_distribution))
    blob.add("hmm.state_positions", np.asarray(dbn.st.state_positions))

    # madmom computes diff_frames from the window overlap; we read it back after
    # one pass instead of replicating the formula.
    probe = RNNBeatProcessor(online=True, fps=FPS, origin="stream", num_frames=1)
    _s, _multi, _ = probe[0]
    _diff_p = _multi[0][4]
    probe(np.zeros(FRAME_SIZE, dtype=np.float32), reset=False)
    diff_frames = int(_diff_p.diff_frames)

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "fps": FPS,
        "frame_size": FRAME_SIZE,
        "hop_size": HOP,
        "num_bands": fb.shape[1],
        "num_bins": fb.shape[0],
        "feature_size": 2 * fb.shape[1],
        "log": {"mul": 1, "add": 1},
        "diff": {"ratio": 0.5, "positive_only": True, "frames": diff_frames},
        "ensemble": len(BEATS_LSTM),
        "nets": nets,
        "dbn": {
            "num_states": int(dbn.st.num_states),
            "num_transitions": int(dbn.tm.num_transitions),
            "observation_lambda": int(dbn.om.observation_lambda),
            "min_bpm": MIN_BPM,
            "max_bpm": MAX_BPM,
        },
        "arrays": blob.entries,
    }

    (OUT / "madmom-beats.bin").write_bytes(bytes(blob.buf))
    (OUT / "madmom-beats.json").write_text(json.dumps(manifest, indent=1))

    # ------------------------------------------------------ reference vectors
    proc = RNNBeatProcessor(online=True, fps=FPS, origin="stream", num_frames=1)
    pre = proc[0]
    _sig, multi, _ = pre
    chain = multi[0]
    frames_p, stft_p, filt_p, log_p, diff_p = chain

    rng = np.random.default_rng(20260725)
    audio = (rng.standard_normal(SAMPLE_RATE * 3) * 0.15).astype(np.float32)
    # something with structure, not just noise
    t = np.arange(audio.size) / SAMPLE_RATE
    audio += (0.4 * np.sin(2 * np.pi * 220 * t) * np.exp(-np.mod(t, 0.5) * 25)).astype(np.float32)

    from madmom.processors import BufferProcessor

    buffer = BufferProcessor(FRAME_SIZE)
    ref_frames, ref_feats, ref_acts = [], [], []
    net_objs = [NeuralNetwork.load(p) for p in BEATS_LSTM]
    # Start from frame 0 with the nets in their initial state, so the port can
    # start from `init` and compare with no recurrent-state ambiguity.
    for net in net_objs:
        for layer in net.layers:
            if hasattr(layer, "reset"):
                layer.reset()
    for i in range(60):
        hop = audio[i * HOP:(i + 1) * HOP]
        frame = buffer(hop)
        # reset=False is essential: without it the diff buffer restarts on every
        # call and all 81 diff columns come out as zeros.
        feats = np.asarray(pre(frame, reset=False), dtype=np.float32).ravel()
        acts = [float(np.ravel(n.process(feats.reshape(1, -1), reset=False))[0])
                for n in net_objs]
        ref_frames.append([float(x) for x in frame])
        ref_feats.append([float(x) for x in feats])
        ref_acts.append({"per_net": acts, "mean": float(np.mean(acts))})

    # DBN reference: activations -> forward -> detected beats
    dbn_ref = DBNBeatTrackingProcessor(min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                                       fps=FPS, online=True)
    rng2 = np.random.default_rng(11)
    # activations with a pulse every 50 frames (120 BPM) + noise
    n_act = 400
    dbn_acts = np.clip(rng2.random(n_act) * 0.15, 1e-6, 1.0)
    dbn_acts[np.arange(0, n_act, 50)] = 0.9
    dbn_beats, dbn_argmax = [], []
    for i, a in enumerate(dbn_acts):
        out = dbn_ref.process_online(np.asarray([a], dtype=np.float32), reset=(i == 0))
        fwd_state = int(np.argmax(dbn_ref.hmm._prev))
        dbn_argmax.append(fwd_state)
        for b in np.atleast_1d(out):
            dbn_beats.append(float(b))
        # Nota: process_online ya hace `self.counter += len(activations)`
        # (beats.py:1032), so it must not be incremented here.

    test_dir = ROOT / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "reference.json").write_text(json.dumps({
        "note": "frames de 2048 muestras -> feats de 162 -> activacion por red y promedio",
        "frames": ref_frames,
        "feats": ref_feats,
        "acts": ref_acts,
    }))
    (test_dir / "reference_dbn.json").write_text(json.dumps({
        "note": "activaciones -> estado argmax del forward por frame + beats detectados",
        "activations": [float(a) for a in dbn_acts],
        "argmax_states": dbn_argmax,
        "beats": dbn_beats,
    }))

    # --------------------------------------- end-to-end with real audio
    # Reference audio for the end-to-end test. Any song works; the fixtures are
    # not versioned because they weigh ~11 MB.
    #   AUDIO_E2E=/path/to/song.mp3 python tools/export_madmom.py
    e2e_src = Path(os.environ.get("AUDIO_E2E", ""))
    if e2e_src.name and e2e_src.exists():
        import librosa

        e2e_audio, _ = librosa.load(str(e2e_src), sr=SAMPLE_RATE, mono=True, duration=45.0)
        e2e_audio = e2e_audio.astype(np.float32)
        n_hops = len(e2e_audio) // HOP
        e2e_audio = e2e_audio[: n_hops * HOP]

        rnn = RNNBeatProcessor(online=True, fps=FPS, origin="stream", num_frames=1)
        dbn_e2e = DBNBeatTrackingProcessor(min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                                           fps=FPS, online=True)
        buf2 = BufferProcessor(FRAME_SIZE)
        e2e_acts, e2e_beats = [], []
        for i in range(n_hops):
            frame = buf2(e2e_audio[i * HOP:(i + 1) * HOP])
            act = np.asarray(rnn(frame, reset=False), dtype=np.float32).ravel()
            e2e_acts.append(float(act[0]))
            out = dbn_e2e.process_online(act, reset=(i == 0))
            for b in np.atleast_1d(out):
                e2e_beats.append(float(b))

        (test_dir / "e2e_audio.f32").write_bytes(e2e_audio.tobytes())
        (test_dir / "e2e_reference.json").write_text(json.dumps({
            "source": e2e_src.name,
            "sample_rate": SAMPLE_RATE,
            "num_hops": n_hops,
            "activations": e2e_acts,
            "beats": e2e_beats,
        }))
        print(f"end-to-end: {n_hops} hops, {len(e2e_beats)} beats -> test/e2e_*")
    else:
        print("end-to-end: mp3 not found, skipped")

    total = len(blob.buf) / 1e6
    print(f"weights+filterbank: {total:.2f} MB in madmom-beats.bin")
    print(f"arrays: {len(blob.entries)}")
    print(f"feature size: {manifest['feature_size']} ({fb.shape[1]} bands x2)")
    print(f"reference: {len(ref_feats)} frames -> test/reference.json")


if __name__ == "__main__":
    sys.exit(main())
