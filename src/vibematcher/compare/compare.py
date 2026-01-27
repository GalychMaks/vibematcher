from dataclasses import dataclass
from typing import Dict, Optional, Literal

import numpy as np

from vibematcher.fingerprint.fingerprint import AudioFingerprint


@dataclass(frozen=True)
class FingerprintComparisonResult:
    """
    Example:
      overall = 0.9
      aspects = {"mert": 0.9}
      correlation_matrices = {"mert": (Q, C) matrix}
    """

    overall: float
    aspects: Dict[str, float]
    correlation_matrices: Dict[str, np.ndarray]  # e.g. {"mert": (Q, C)}

    # optional: for plotting axes in seconds
    query_chunk_starts_sec: Optional[np.ndarray] = None  # (Q,)
    cand_chunk_starts_sec: Optional[np.ndarray] = None  # (C,)


def cosine_similarity_matrix(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """
    Pairwise cosine similarity between rows of a and rows of b.

    a: (Q, D)
    b: (C, D)
    returns: (Q, C)
    """
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
    sim_matrix: np.ndarray,
    mode: AggregationMode,
    *,
    topk: int = 5,
) -> float:
    """
    sim_matrix: (Q, C)

    mode:
      - "best": max over all pairs (very sensitive)
      - "mean": mean over all pairs (strict, penalizes any mismatch)
      - "topk_bestmean": for each query chunk take best candidate match; average top-k
    """
    if sim_matrix.size == 0:
        return 0.0

    if mode == "best":
        return float(sim_matrix.max())

    if mode == "mean":
        return float(sim_matrix.mean())

    if mode == "topk_bestmean":
        # best match per query chunk
        best_per_q = sim_matrix.max(axis=1)  # (Q,)
        k = max(1, min(int(topk), best_per_q.size))
        return float(np.sort(best_per_q)[-k:].mean())

    raise ValueError(f"Unknown aggregation mode: {mode}")


def compare_fingerprints(
    query: AudioFingerprint,
    candidate: AudioFingerprint,
    *,
    aggregation: AggregationMode = "topk_bestmean",
    topk: int = 5,
    include_axes: bool = True,
) -> FingerprintComparisonResult:
    """
    MERT embeddings stored as 2D arrays: (num_chunks, dim).
    Returns:
      - scalar similarity (overall/aspects["mert"])
      - correlation matrix for visualization: correlation_matrices["mert"] (Q, C)
      - optional chunk start axes in seconds (if AudioFingerprint has chunk_starts_sec)
    """
    q = query.mert_embeddings  # (Q, D)
    c = candidate.mert_embeddings  # (C, D)

    if q.ndim != 2 or c.ndim != 2:
        raise ValueError(
            f"Expected 2D mert_embeddings. Got query={q.shape}, candidate={c.shape}"
        )

    if q.size != 0 and c.size != 0 and q.shape[1] != c.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: query D={q.shape[1]} vs candidate D={c.shape[1]}"
        )

    mert_corr = cosine_similarity_matrix(q, c)  # (Q, C)
    mert_sim = aggregate_chunk_similarities(mert_corr, mode=aggregation, topk=topk)

    q_axis = getattr(query, "chunk_starts_sec", None) if include_axes else None
    c_axis = getattr(candidate, "chunk_starts_sec", None) if include_axes else None

    return FingerprintComparisonResult(
        overall=mert_sim,
        aspects={"mert": mert_sim},
        correlation_matrices={"mert": mert_corr},
        query_chunk_starts_sec=q_axis,
        cand_chunk_starts_sec=c_axis,
    )
