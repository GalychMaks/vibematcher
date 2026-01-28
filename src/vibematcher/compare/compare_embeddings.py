import numpy as np

from vibematcher.compare.models import AggregationMode


def cosine_similarity_matrix(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """
    a: (Q, D), b: (C, D) -> (Q, C) cosine similarity
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_unit = a / np.maximum(a_norm, eps)
    b_unit = b / np.maximum(b_norm, eps)

    return (a_unit @ b_unit.T).astype(np.float32, copy=False)


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
