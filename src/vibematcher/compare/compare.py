from typing import Dict, Optional

import numpy as np

from vibematcher.compare.compare_embeddings import (
    AggregationMode,
    aggregate_chunk_similarities,
    cosine_similarity_matrix,
)
from vibematcher.compare.compare_f0 import (
    center_cents_by_median,
    downsample_1d,
    f0_dtw_similarity,
    f0_hz_to_cents,
)
from vibematcher.compare.models import FingerprintComparisonResult
from vibematcher.fingerprint.fingerprint import AudioFingerprint


def compare_fingerprints(
    query: AudioFingerprint,
    candidate: AudioFingerprint,
    *,
    aggregation: AggregationMode = "topk_bestmean",
    topk: int = 5,
    include_axes: bool = True,
    # F0 DTW settings
    max_f0_frames: Optional[int] = 2000,
    f0_ref_hz_for_cents: float = 55.0,
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

    aspects["mert"] = float(mert_sim)
    corr_mats["mert"] = mert_corr

    # --- F0 via DTW (key-invariant: cents + median-centering) ---
    qf0_hz = downsample_1d(np.asarray(query.f0_hz, dtype=np.float32), max_f0_frames)
    cf0_hz = downsample_1d(np.asarray(candidate.f0_hz, dtype=np.float32), max_f0_frames)

    if qf0_hz.ndim != 1 or cf0_hz.ndim != 1:
        raise ValueError(
            f"Expected 1D f0_hz. Got query={qf0_hz.shape}, cand={cf0_hz.shape}"
        )

    q_cents = center_cents_by_median(f0_hz_to_cents(qf0_hz, ref_hz=f0_ref_hz_for_cents))
    c_cents = center_cents_by_median(f0_hz_to_cents(cf0_hz, ref_hz=f0_ref_hz_for_cents))

    f0_sim, f0_acc, f0_path = f0_dtw_similarity(
        q_cents,
        c_cents,
        unvoiced_cost_cents=f0_unvoiced_cost_cents,
        band=f0_dtw_band,
        scale_cents=f0_scale_cents,
    )

    aspects["f0"] = float(f0_sim)
    corr_mats["f0_dtw_acc_cost"] = f0_acc  # (Tq, Tc) lower = better
    corr_mats["f0_dtw_path"] = f0_path  # (L, 2) int32 indices

    # --- overall (weighted average) ---
    scores: list[float] = []
    weights: list[float] = []

    if mert_weight > 0 and np.isfinite(mert_sim):
        scores.append(float(mert_sim))
        weights.append(float(mert_weight))
    if f0_weight > 0 and np.isfinite(f0_sim):
        scores.append(float(f0_sim))
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
