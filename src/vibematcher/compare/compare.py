from dataclasses import dataclass
from typing import Dict

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


def aggregate_chunk_similarities(sim_matrix: np.ndarray, mode: str) -> float:
    """
    sim_matrix: (Q, C)

    mode:
      - "best": max over all pairs (good for partial match)
      - "mean": mean over all pairs (stricter, more global)
    """
    if sim_matrix.size == 0:
        return 0.0
    if mode == "best":
        return float(sim_matrix.max())
    if mode == "mean":
        return float(sim_matrix.mean())
    raise ValueError(f"Unknown aggregation mode: {mode}")


def compare_fingerprints(
    query: AudioFingerprint,
    candidate: AudioFingerprint,
    *,
    aggregation: str = "best",
) -> FingerprintComparisonResult:
    """
    MERT embeddings are stored as 2D arrays: (num_chunks, dim).
    Returns:
      - scalar similarity (overall/aspects["mert"])
      - correlation matrix for visualization: correlation_matrices["mert"] (Q, C)
    """
    q = query.mert_embeddings  # (Q, D)
    c = candidate.mert_embeddings  # (C, D)

    if q.ndim != 2 or c.ndim != 2:
        raise ValueError(
            f"Expected 2D mert_embeddings. Got query={q.shape}, candidate={c.shape}"
        )
    if q.shape[1] != c.shape[1] and q.size != 0 and c.size != 0:
        raise ValueError(
            f"Embedding dim mismatch: query D={q.shape[1]} vs candidate D={c.shape[1]}"
        )

    mert_corr = cosine_similarity_matrix(q, c)  # (Q, C)
    mert_sim = aggregate_chunk_similarities(mert_corr, mode=aggregation)

    return FingerprintComparisonResult(
        overall=mert_sim,
        aspects={"mert": mert_sim},
        correlation_matrices={"mert": mert_corr},
    )
