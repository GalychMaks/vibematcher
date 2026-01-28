# similarity.py
from __future__ import annotations

from typing import Tuple, Dict, Any, List

import numpy as np
import librosa

from .features import SegmentFeatures


def _strip_nans(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    m = np.isfinite(x)
    return x[m]


def pitch_dtw_similarity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    subseq: bool = True,
    band_rad: float = 0.25,
    sigma_cents: float = 200.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Subsequence DTW on absolute cents distance.
    Returns similarity in [0,1] (roughly), plus debug info.
    """
    xa = _strip_nans(a)
    xb = _strip_nans(b)
    if xa.size < 10 or xb.size < 10:
        return 0.0, {"reason": "too_short"}

    # cost matrix: |xa - xb|
    C = np.abs(xa[:, None] - xb[None, :]).astype(np.float32)

    D, wp = librosa.sequence.dtw(
        C=C,
        subseq=subseq,
        backtrack=True,
        global_constraints=True,
        band_rad=band_rad,
    )

    if subseq:
        end_j = int(np.argmin(D[-1, :]))
        path_cost = float(D[-1, end_j])
        # wp is a full backtrack path for the chosen end; good enough for display/debug
    else:
        path_cost = float(D[-1, -1])

    norm_cost = path_cost / float(xa.size)  # normalize by query length
    sim = float(np.exp(-norm_cost / sigma_cents))

    return sim, {
        "norm_cost": norm_cost,
        "path_cost": path_cost,
        "len_a": int(xa.size),
        "len_b": int(xb.size),
    }


def _bigrams(seq: List[int]) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    for x, y in zip(seq[:-1], seq[1:]):
        out[(x, y)] = out.get((x, y), 0) + 1
    return out


def chord_bigram_cosine(a_seq: List[int], b_seq: List[int]) -> float:
    if len(a_seq) < 2 or len(b_seq) < 2:
        return 0.0
    a = _bigrams(a_seq)
    b = _bigrams(b_seq)
    keys = set(a.keys()) | set(b.keys())
    av = np.array([a.get(k, 0) for k in keys], dtype=np.float32)
    bv = np.array([b.get(k, 0) for k in keys], dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def bpm_similarity(a_bpm: float, b_bpm: float) -> float:
    if a_bpm <= 0 or b_bpm <= 0:
        return 0.0
    r = a_bpm / b_bpm
    r = min(r, 1.0 / r)  # symmetric ratio in (0,1]
    # tolerate tempo doubling/halving a bit (common in trackers)
    r2 = min(r * 2.0, 1.0 / (r * 2.0))
    r = max(r, r2)
    return float(np.clip(r, 0.0, 1.0))


def segment_similarity(
    A: SegmentFeatures,
    B: SegmentFeatures,
    *,
    w_pitch: float = 0.65,
    w_chord: float = 0.25,
    w_bpm: float = 0.10,
) -> Tuple[float, Dict[str, Any]]:
    pitch_sim, pitch_dbg = pitch_dtw_similarity(A.pitch_cents, B.pitch_cents)
    chord_sim = chord_bigram_cosine(A.chord_seq, B.chord_seq)
    tempo_sim = bpm_similarity(A.bpm, B.bpm)

    score = w_pitch * pitch_sim + w_chord * chord_sim + w_bpm * tempo_sim
    dbg = {
        "pitch_sim": pitch_sim,
        "pitch_dbg": pitch_dbg,
        "chord_sim": chord_sim,
        "tempo_sim": tempo_sim,
        "A_bpm": A.bpm,
        "B_bpm": B.bpm,
    }
    return float(score), dbg
