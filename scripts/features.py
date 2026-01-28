# features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import librosa


@dataclass
class SegmentFeatures:
    start: float
    end: float
    pitch_cents: np.ndarray  # (T,) with NaNs for unvoiced
    chord_seq: List[int]  # chord IDs (0..23), run-length reduced
    bpm: float


# -----------------------
# Pitch (melody proxy)
# -----------------------
def hz_to_cents(f_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    f = np.asarray(f_hz, dtype=np.float32)
    out = np.full_like(f, np.nan, dtype=np.float32)
    voiced = np.isfinite(f) & (f > 0)
    out[voiced] = 1200.0 * np.log2(f[voiced] / float(ref_hz))
    return out


def center_by_median(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x) if np.any(np.isfinite(x)) else np.nan
    if not np.isfinite(med):
        return x
    return x - med


def quantize_cents(x: np.ndarray, step_cents: float = 50.0) -> np.ndarray:
    """Quantize to reduce vibrato / ornamentation sensitivity."""
    xq = x.copy()
    m = np.isfinite(xq)
    xq[m] = np.round(xq[m] / step_cents) * step_cents
    return xq


def extract_pitch_cents(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 160,  # 10ms @ 16k
    fmin: float = 65.4,  # C2
    fmax: float = 1046.5,  # C6
    ref_hz: float = 55.0,
    quant_step_cents: float = 50.0,
) -> np.ndarray:
    # Emphasize harmonic content to make pYIN happier
    y_harm = librosa.effects.harmonic(y)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_harm,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )

    cents = hz_to_cents(f0, ref_hz=ref_hz)
    cents = center_by_median(cents)  # transposition invariance (key shift)
    cents = quantize_cents(cents, step_cents=quant_step_cents)
    return cents.astype(np.float32)


# -----------------------
# Very simple chord IDs (major/minor templates)
# -----------------------
_CHORD_NAMES = [
    "C:maj",
    "C#:maj",
    "D:maj",
    "D#:maj",
    "E:maj",
    "F:maj",
    "F#:maj",
    "G:maj",
    "G#:maj",
    "A:maj",
    "A#:maj",
    "B:maj",
    "C:min",
    "C#:min",
    "D:min",
    "D#:min",
    "E:min",
    "F:min",
    "F#:min",
    "G:min",
    "G#:min",
    "A:min",
    "A#:min",
    "B:min",
]


def _build_chord_templates() -> np.ndarray:
    # 12-dim chroma templates for major/minor triads
    tpl = np.zeros((24, 12), dtype=np.float32)
    major = np.zeros(12, dtype=np.float32)
    major[[0, 4, 7]] = 1.0
    minor = np.zeros(12, dtype=np.float32)
    minor[[0, 3, 7]] = 1.0
    for root in range(12):
        tpl[root] = np.roll(major, root)
        tpl[12 + root] = np.roll(minor, root)
    # normalize
    tpl /= np.linalg.norm(tpl, axis=1, keepdims=True) + 1e-8
    return tpl


_CHORD_TPL = _build_chord_templates()


def extract_chord_sequence(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    smooth: int = 5,
) -> List[int]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)

    # score each frame against templates (cosine because normalized)
    scores = _CHORD_TPL @ chroma  # (24, T)
    chord_ids = np.argmax(scores, axis=0).astype(np.int32)

    # simple smoothing by median filter over IDs (works “ok” for coarse chord changes)
    if smooth > 1 and chord_ids.size >= smooth:
        pad = smooth // 2
        x = np.pad(chord_ids, (pad, pad), mode="edge")
        sm = np.empty_like(chord_ids)
        for i in range(chord_ids.size):
            sm[i] = int(np.median(x[i : i + smooth]))
        chord_ids = sm

    # run-length reduce
    out: List[int] = []
    prev = None
    for cid in chord_ids.tolist():
        if prev is None or cid != prev:
            out.append(int(cid))
            prev = cid
    return out


def extract_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def extract_segment_features(
    y_full: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
) -> SegmentFeatures:
    s = int(max(0, np.floor(start_s * sr)))
    e = int(min(y_full.size, np.ceil(end_s * sr)))
    y = y_full[s:e]

    bpm = extract_bpm(y, sr)
    pitch_cents = extract_pitch_cents(y, sr)
    chord_seq = extract_chord_sequence(y, sr)

    return SegmentFeatures(
        start=float(start_s),
        end=float(end_s),
        pitch_cents=pitch_cents,
        chord_seq=chord_seq,
        bpm=bpm,
    )
