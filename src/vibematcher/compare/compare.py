from dataclasses import dataclass
from typing import Dict, Optional, Literal

import numpy as np

from vibematcher.fingerprint.fingerprint import AudioFingerprint


@dataclass(frozen=True)
class FingerprintComparisonResult:
    """
    Example:
      overall = 0.9
      aspects = {"mert": 0.9, "f0": 0.85}
      correlation_matrices = {
        "mert": (Q, C) cosine-sim matrix,
        "f0_dtw_acc_cost": (Tq, Tc) accumulated DTW cost (lower = better),
        "f0_dtw_path": (L, 2) int32 path indices (i, j),
      }
    """

    overall: float
    aspects: Dict[str, float]
    correlation_matrices: Dict[str, np.ndarray]

    query_chunk_starts_sec: Optional[np.ndarray] = None  # (Q,)
    cand_chunk_starts_sec: Optional[np.ndarray] = None  # (C,)


def cosine_similarity_matrix(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_unit = a / np.maximum(a_norm, eps)
    b_unit = b / np.maximum(b_norm, eps)

    return (a_unit @ b_unit.T).astype(np.float32, copy=False)


AggregationMode = Literal["best", "mean", "topk_bestmean"]


def aggregate_chunk_similarities(
    sim_matrix: np.ndarray, mode: AggregationMode, *, topk: int = 5
) -> float:
    if sim_matrix.size == 0:
        return 0.0

    if mode == "best":
        return float(np.nanmax(sim_matrix))

    if mode == "mean":
        return float(np.nanmean(sim_matrix))

    if mode == "topk_bestmean":
        best_per_q = np.nanmax(sim_matrix, axis=1)  # (Q,)
        best_per_q = best_per_q[np.isfinite(best_per_q)]
        if best_per_q.size == 0:
            return 0.0
        k = max(1, min(int(topk), best_per_q.size))
        return float(np.sort(best_per_q)[-k:].mean())

    raise ValueError(f"Unknown aggregation mode: {mode}")


def _downsample_1d(x: np.ndarray, max_len: Optional[int]) -> np.ndarray:
    x = np.asarray(x)
    if max_len is None or x.size <= max_len:
        return x
    max_len = int(max_len)
    if max_len <= 0:
        return x[:0]
    idx = np.linspace(0, x.size - 1, num=max_len, dtype=np.int64)
    return x[idx]


def _f0_frame_cost_cents(
    q_hz: float, c_hz: float, *, unvoiced_cost_cents: float
) -> float:
    """
    Cost between two f0 frames.
      - both unvoiced (NaN or <=0): 0
      - one voiced, one unvoiced: unvoiced_cost_cents
      - both voiced: abs pitch diff in cents
    """
    q_voiced = np.isfinite(q_hz) and (q_hz > 0.0)
    c_voiced = np.isfinite(c_hz) and (c_hz > 0.0)

    if (not q_voiced) and (not c_voiced):
        return 0.0
    if q_voiced != c_voiced:
        return float(unvoiced_cost_cents)

    # cents distance = 1200*|log2(q/c)|
    return float(abs(np.log2(q_hz / c_hz)) * 1200.0)


def dtw_acc_cost_and_path(
    q_f0: np.ndarray,
    c_f0: np.ndarray,
    *,
    unvoiced_cost_cents: float = 600.0,
    band: Optional[int] = None,  # Sakoe–Chiba band in frames, e.g. 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic DTW with 3-way steps (diag/up/left).
    Returns:
      acc: (Tq, Tc) accumulated cost (float32)
      path: (L, 2) int32 indices (i, j) from start to end
    """
    q = np.asarray(q_f0, dtype=np.float32)
    c = np.asarray(c_f0, dtype=np.float32)
    Tq, Tc = int(q.size), int(c.size)

    if Tq == 0 or Tc == 0:
        acc = np.empty((Tq, Tc), dtype=np.float32)
        path = np.empty((0, 2), dtype=np.int32)
        return acc, path

    inf = np.float32(np.inf)
    acc = np.full((Tq, Tc), inf, dtype=np.float32)

    # DP fill
    for i in range(Tq):
        j0 = 0
        j1 = Tc
        if band is not None:
            b = int(band)
            j0 = max(0, i - b)
            j1 = min(Tc, i + b + 1)

        for j in range(j0, j1):
            cost = np.float32(
                _f0_frame_cost_cents(
                    q[i], c[j], unvoiced_cost_cents=unvoiced_cost_cents
                )
            )

            if i == 0 and j == 0:
                acc[i, j] = cost
            else:
                best_prev = inf
                if i > 0:
                    best_prev = min(best_prev, acc[i - 1, j])
                if j > 0:
                    best_prev = min(best_prev, acc[i, j - 1])
                if i > 0 and j > 0:
                    best_prev = min(best_prev, acc[i - 1, j - 1])

                acc[i, j] = cost + best_prev

    # Backtrack path from (Tq-1, Tc-1)
    i, j = Tq - 1, Tc - 1
    if not np.isfinite(acc[i, j]):
        # No valid path under the band constraint
        return acc, np.empty((0, 2), dtype=np.int32)

    path_rev: list[tuple[int, int]] = [(i, j)]
    while i > 0 or j > 0:
        candidates: list[tuple[float, int, int]] = []

        if i > 0 and np.isfinite(acc[i - 1, j]):
            candidates.append((float(acc[i - 1, j]), i - 1, j))
        if j > 0 and np.isfinite(acc[i, j - 1]):
            candidates.append((float(acc[i, j - 1]), i, j - 1))
        if i > 0 and j > 0 and np.isfinite(acc[i - 1, j - 1]):
            candidates.append((float(acc[i - 1, j - 1]), i - 1, j - 1))

        if not candidates:
            break

        _, i, j = min(candidates, key=lambda t: t[0])
        path_rev.append((i, j))

    path = np.asarray(path_rev[::-1], dtype=np.int32)
    return acc, path


def f0_dtw_similarity(
    q_f0: np.ndarray,
    c_f0: np.ndarray,
    *,
    unvoiced_cost_cents: float = 600.0,
    band: Optional[int] = None,
    scale_cents: float = 150.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
      sim: scalar in (0, 1], higher = more similar
      acc: accumulated DTW cost matrix
      path: DTW path (L, 2)
    """
    acc, path = dtw_acc_cost_and_path(
        q_f0, c_f0, unvoiced_cost_cents=unvoiced_cost_cents, band=band
    )
    if acc.size == 0 or path.size == 0:
        return 0.0, acc, path

    # normalize by path length -> "avg cents per step"
    avg_cost = float(acc[-1, -1]) / max(1, int(path.shape[0]))
    scale = max(1e-6, float(scale_cents))
    sim = float(np.exp(-avg_cost / scale))  # 1.0 when identical, decays with avg cost
    return sim, acc, path


def compare_fingerprints(
    query: AudioFingerprint,
    candidate: AudioFingerprint,
    *,
    aggregation: AggregationMode = "topk_bestmean",
    topk: int = 5,
    include_axes: bool = True,
    # f0 DTW settings
    max_f0_frames: Optional[int] = 2000,
    f0_unvoiced_cost_cents: float = 600.0,
    f0_dtw_band: Optional[int] = None,  # e.g. 150
    f0_scale_cents: float = 150.0,
    # overall weighting
    mert_weight: float = 1.0,
    f0_weight: float = 1.0,
) -> FingerprintComparisonResult:
    aspects: Dict[str, float] = {}
    corr_mats: Dict[str, np.ndarray] = {}

    # --- MERT ---
    q = np.asarray(query.mert_embeddings)
    c = np.asarray(candidate.mert_embeddings)

    if q.ndim != 2 or c.ndim != 2:
        raise ValueError(
            f"Expected 2D mert_embeddings. Got query={q.shape}, candidate={c.shape}"
        )
    if q.size and c.size and q.shape[1] != c.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: query D={q.shape[1]} vs candidate D={c.shape[1]}"
        )

    mert_corr = cosine_similarity_matrix(q, c)  # (Q, C)
    mert_sim = aggregate_chunk_similarities(mert_corr, mode=aggregation, topk=topk)
    aspects["mert"] = mert_sim
    corr_mats["mert"] = mert_corr

    # --- F0 via DTW ---
    qf0 = _downsample_1d(np.asarray(query.f0_hz, dtype=np.float32), max_f0_frames)
    cf0 = _downsample_1d(np.asarray(candidate.f0_hz, dtype=np.float32), max_f0_frames)

    if qf0.ndim != 1 or cf0.ndim != 1:
        raise ValueError(f"Expected 1D f0_hz. Got query={qf0.shape}, cand={cf0.shape}")

    f0_sim, f0_acc, f0_path = f0_dtw_similarity(
        qf0,
        cf0,
        unvoiced_cost_cents=f0_unvoiced_cost_cents,
        band=f0_dtw_band,
        scale_cents=f0_scale_cents,
    )
    aspects["f0"] = f0_sim
    corr_mats["f0_dtw_acc_cost"] = f0_acc  # (Tq, Tc) lower = better
    corr_mats["f0_dtw_path"] = f0_path  # (L, 2) (i, j)

    # --- overall ---
    scores = []
    weights = []
    if mert_weight > 0 and np.isfinite(mert_sim):
        scores.append(mert_sim)
        weights.append(float(mert_weight))
    if f0_weight > 0 and np.isfinite(f0_sim):
        scores.append(f0_sim)
        weights.append(float(f0_weight))

    overall = float(np.average(scores, weights=weights)) if weights else 0.0

    q_axis = getattr(query, "chunk_starts_sec", None) if include_axes else None
    c_axis = getattr(candidate, "chunk_starts_sec", None) if include_axes else None

    return FingerprintComparisonResult(
        overall=overall,
        aspects=aspects,
        correlation_matrices=corr_mats,
        query_chunk_starts_sec=q_axis,
        cand_chunk_starts_sec=c_axis,
    )
