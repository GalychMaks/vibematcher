from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibematcher.mert.mert_embedder import MertEmbedder
from vibematcher.mert.mert_embedding import MertEmbedding
from vibematcher.muq.muq_embedder import MuQEmbedder
from vibematcher.muq.muq_embedding import MuQEmbedding


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


def aggregate_chunk_similarities(sim_matrix: np.ndarray, mode: str = "best") -> float:
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


@dataclass
class CompareMERT:
    embedder = MertEmbedder()

    @classmethod
    def score(cls, query_path: str | Path, reference_path: str | Path) -> float:
        q_emb = MertEmbedding.from_audio_file(
            query_path,
            embedder=cls.embedder,
        ).m
        r_emb = MertEmbedding.from_audio_file(
            reference_path,
            embedder=cls.embedder,
        ).mert_embeddings

        mert_corr = cosine_similarity_matrix(q_emb, r_emb)  # (Q, C)
        mert_sim = aggregate_chunk_similarities(mert_corr)

        return mert_sim


@dataclass
class CompareMUQ:
    embedder = MuQEmbedder()

    @classmethod
    def score(cls, query_path: str | Path, reference_path: str | Path) -> float:
        q_emb = MuQEmbedding.from_audio_file(
            query_path,
            embedder=cls.embedder,
        ).muq_embeddings
        r_emb = MuQEmbedding.from_audio_file(
            reference_path,
            embedder=cls.embedder,
        ).muq_embeddings

        mert_corr = cosine_similarity_matrix(q_emb, r_emb)  # (Q, C)
        mert_sim = aggregate_chunk_similarities(mert_corr)

        return mert_sim


if __name__ == "__main__":
    # Example usage: compare two full songs by stems
    query = "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav"
    references = [Path(path) for path in Path("data/original").glob("*.wav")]

    # Create once and reuse to avoid reloading demucs model repeatedly
    compare_dwt = CompareMERT()

    for reference in references:
        score = compare_dwt.score(query, reference)

    print(score)
