from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from vibematcher.mert.mert_embedder import MertEmbedder


@dataclass
class MertEmbedding:
    # (num_chunks, embedding_dim) float32
    mert_embeddings: np.ndarray

    MERT_EMBEDDING_KEY = "mert.npy"

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        embedder: Optional[MertEmbedder] = None,
        force_recompute: bool = False,
    ) -> "MertEmbedding":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem
        mert_embeddings_path = cache_dir / cls.MERT_EMBEDDING_KEY

        if not force_recompute and cache_dir.exists() and mert_embeddings_path.exists():
            mert_embeddings = np.asarray(
                np.load(mert_embeddings_path, allow_pickle=False),
                dtype=np.float32,
            )
            return cls(mert_embeddings=mert_embeddings)

        if not embedder:
            embedder = MertEmbedder()

        mert_embeddings = embedder.embed(audio_path)  # (N, D) float32

        # sanity check (optional but helpful)
        if mert_embeddings.ndim != 2:
            raise ValueError(
                f"Expected mert_embeddings to be 2D (num_chunks, dim), "
                f"got shape={mert_embeddings.shape}"
            )
        mert_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mert_embeddings_path, mert_embeddings, allow_pickle=False)

        return cls(mert_embeddings=mert_embeddings)
