from typing import Dict

import numpy as np

from vibematcher.compare.compare_f0 import compare_f0
from vibematcher.compare.compare_embeddings import (
    AggregationMode,
    aggregate_chunk_similarities,
    cosine_similarity_matrix,
)
from vibematcher.compare.models import FingerprintComparisonResult
from vibematcher.fingerprint.fingerprint import AudioFingerprint


def compare_fingerprints(
    query: AudioFingerprint,
    candidate: AudioFingerprint,
    *,
    aggregation: AggregationMode = "topk_bestmean",
    topk: int = 5,
    # overall weighting
    mert_weight: float = 1.0,
    f0_weight: float = 1.0,
) -> FingerprintComparisonResult:
    aspects: Dict[str, float] = {}
    corr_mats: Dict[str, np.ndarray] = {}

    # --- MERT ---
    q = np.asarray(query.mert_embeddings)
    c = np.asarray(candidate.mert_embeddings)

    mert_corr = cosine_similarity_matrix(q, c)  # (Q, C)
    mert_sim = aggregate_chunk_similarities(mert_corr, mode=aggregation, topk=topk)

    aspects["mert"] = float(mert_sim)
    corr_mats["mert"] = mert_corr

    # --- F0 via DTW (key-invariant: cents + median-centering) ---
    print("Comparing F0...")
    f0_sim = compare_f0(query.f0_hz, candidate.f0_hz)

    aspects["f0"] = float(f0_sim)

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

    return FingerprintComparisonResult(
        overall=overall,
        aspects=aspects,
        correlation_matrices=corr_mats,
    )
