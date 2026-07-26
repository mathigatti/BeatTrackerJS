#!/usr/bin/env python3
"""Dumps madmom's downbeat model (RNNDownBeatProcessor) for the browser.

Unlike the beat model, this one is **bidirectional**: every layer runs the
sequence forwards and backwards and concatenates. That makes it non-causal, so
live it can only be run over a rolling window.

Output in web/public/models/: madmom-downbeats.{bin,json}
Reference in test/reference_downbeats.json

Usage: python tools/export_downbeats.py [ensemble size]
"""
from __future__ import annotations

import collections
import collections.abc
import json
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
from madmom.features.downbeats import RNNDownBeatProcessor  # noqa: E402
from madmom.ml.nn import NeuralNetwork  # noqa: E402
from madmom.models import DOWNBEATS_BLSTM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "models"
TEST = ROOT / "test"
SAMPLE_RATE, FPS = 44100, 100
HOP = SAMPLE_RATE // FPS
RESOLUTIONS = [(1024, 3), (2048, 6), (4096, 12)]
# How many nets of the ensemble to export. Measured over three songs: with 2 the
# meter comes out wrong on hard material, and with 3 the downbeats shift (F 0.47
# against the full ensemble); 4 recovers F 0.94 at half the cost. Raise it to 8
# if you want maximum quality and have CPU to spare.
ENSEMBLE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FMIN, FMAX, FREF = 30, 17000, 440.0


class Blob:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.entries: dict[str, dict] = {}

    def add(self, name: str, arr) -> None:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        self.entries[name] = {"offset": len(self.buf), "shape": list(a.shape),
                              "count": int(a.size)}
        self.buf.extend(a.tobytes())


def export_lstm(blob: Blob, prefix: str, layer) -> dict:
    for part in ("input_gate", "forget_gate", "cell", "output_gate"):
        gate = getattr(layer, part)
        blob.add(f"{prefix}.{part}.weights", gate.weights)
        blob.add(f"{prefix}.{part}.recurrent_weights", gate.recurrent_weights)
        blob.add(f"{prefix}.{part}.bias", gate.bias)
        if getattr(gate, "peephole_weights", None) is not None:
            blob.add(f"{prefix}.{part}.peephole_weights", gate.peephole_weights)
    blob.add(f"{prefix}.init", layer.init)
    blob.add(f"{prefix}.cell_init", layer.cell_init)
    return {"size": int(layer.cell.bias.shape[0]),
            "input_size": int(layer.cell.weights.shape[0])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TEST.mkdir(parents=True, exist_ok=True)
    blob = Blob()

    nets = []
    for idx, path in enumerate(DOWNBEATS_BLSTM[:ENSEMBLE]):
        nn = NeuralNetwork.load(path)
        layers = []
        for li, layer in enumerate(nn.layers):
            p = f"net{idx}.layer{li}"
            kind = type(layer).__name__
            if kind == "BidirectionalLayer":
                fwd = export_lstm(blob, f"{p}.fwd", layer.fwd_layer)
                export_lstm(blob, f"{p}.bwd", layer.bwd_layer)
                layers.append({"type": "bilstm", **fwd})
            elif kind == "FeedForwardLayer":
                blob.add(f"{p}.weights", layer.weights)
                blob.add(f"{p}.bias", layer.bias)
                layers.append({"type": "dense",
                               "input_size": int(layer.weights.shape[0]),
                               "size": int(layer.weights.shape[1]),
                               "activation": layer.activation_fn.__name__})
            else:
                raise RuntimeError(f"capa no soportada: {kind}")
        nets.append({"layers": layers})

    # tres resoluciones de features en paralelo
    resolutions = []
    for frame_size, num_bands in RESOLUTIONS:
        bin_freqs = np.fft.rfftfreq(frame_size, 1.0 / SAMPLE_RATE)
        fb = np.asarray(LogarithmicFilterbank(bin_freqs, num_bands=num_bands,
                                              fmin=FMIN, fmax=FMAX, fref=FREF,
                                              norm_filters=True))
        starts, lengths, weights = [], [], []
        for band in range(fb.shape[1]):
            nz = np.nonzero(fb[:, band])[0]
            starts.append(int(nz[0]))
            lengths.append(int(nz[-1] - nz[0] + 1))
            weights.append(fb[nz[0]: nz[-1] + 1, band])
        tag = f"fb{frame_size}"
        blob.add(f"{tag}.weights", np.concatenate(weights))
        blob.add(f"{tag}.starts", np.asarray(starts))
        blob.add(f"{tag}.lengths", np.asarray(lengths))
        blob.add(f"window{frame_size}", np.hanning(frame_size))
        resolutions.append({"frame_size": frame_size, "num_bands": int(fb.shape[1]),
                            "num_bins": int(fb.shape[0])})

    # diff_frames of each resolution, read back from the real pipeline
    proc = RNNDownBeatProcessor(fps=FPS)
    _sig, multi, _ = proc[0]
    proc[0](np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
    for r, chain in zip(resolutions, multi):
        r["diff_frames"] = int(chain[4].diff_frames)

    total_feats = sum(2 * r["num_bands"] for r in resolutions)
    manifest = {
        "sample_rate": SAMPLE_RATE, "fps": FPS, "hop_size": HOP,
        "resolutions": resolutions,
        "feature_size": total_feats,
        "log": {"mul": 1, "add": 1},
        "diff": {"ratio": 0.5, "positive_only": True},
        "ensemble": ENSEMBLE,
        "ensemble_available": len(DOWNBEATS_BLSTM),
        "nets": nets,
        "note": "bidireccional: no causal, correr sobre ventana rodante",
        "arrays": blob.entries,
    }
    (OUT / "madmom-downbeats.bin").write_bytes(bytes(blob.buf))
    (OUT / "madmom-downbeats.json").write_text(json.dumps(manifest, indent=1))

    # ------------------------------------------------------- referencia
    rng = np.random.default_rng(4242)
    dur = 4.0
    t = np.arange(int(SAMPLE_RATE * dur)) / SAMPLE_RATE
    audio = (0.25 * rng.standard_normal(t.size)).astype(np.float32)
    audio += (0.5 * np.sin(2 * np.pi * 110 * t) * np.exp(-np.mod(t, 0.5) * 30)).astype(np.float32)
    n_hops = len(audio) // HOP
    audio = audio[: n_hops * HOP]

    from madmom.ml.nn import NeuralNetworkEnsemble
    subset = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM[:ENSEMBLE])
    feats = np.asarray(proc[0](audio), dtype=np.float32)
    acts = np.delete(np.asarray(subset(feats)), 0, axis=1).astype(np.float32)

    (TEST / "downbeat_audio.f32").write_bytes(audio.tobytes())
    (TEST / "reference_downbeats.json").write_text(json.dumps({
        "note": "audio -> feats(314) -> activaciones [beat, downbeat] por frame",
        "num_hops": int(n_hops),
        "feature_size": int(feats.shape[1]),
        "feats_head": feats[:40].tolist(),
        "activations": acts.tolist(),
    }))

    print(f"downbeats: {len(blob.buf)/1e6:.2f} MB, {len(blob.entries)} arrays, ensemble={ENSEMBLE}/{len(DOWNBEATS_BLSTM)}")
    print(f"features: {total_feats} dims from {len(resolutions)} resolutions "
          f"({[r['num_bands'] for r in resolutions]} bands)")
    print(f"diff_frames per resolution: {[r['diff_frames'] for r in resolutions]}")
    print(f"reference: {feats.shape[0]} frames, activations {acts.shape}")


if __name__ == "__main__":
    sys.exit(main())
