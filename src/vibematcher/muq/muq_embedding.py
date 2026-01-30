from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from vibematcher.muq.muq_embedder import MuQEmbedder


@dataclass
class MuQEmbedding:
    # (num_chunks, embedding_dim) float32
    muq_embeddings: np.ndarray

    MUQ_EMBEDDING_KEY = "muq.npy"

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        embedder: Optional[MuQEmbedder] = None,
        force_recompute: bool = False,
    ) -> "MuQEmbedding":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem
        muq_embeddings_path = cache_dir / cls.MUQ_EMBEDDING_KEY

        if not force_recompute and cache_dir.exists() and muq_embeddings_path.exists():
            muq_embeddings = np.asarray(
                np.load(muq_embeddings_path, allow_pickle=False),
                dtype=np.float32,
            )
            return cls(muq_embeddings=muq_embeddings)

        if embedder is None:
            embedder = MuQEmbedder()

        muq_embeddings = embedder.embed(audio_path)

        if muq_embeddings.ndim != 2:
            raise ValueError(
                f"Expected muq_embeddings to be 2D (num_chunks, dim), "
                f"got shape={muq_embeddings.shape}"
            )

        muq_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(muq_embeddings_path, muq_embeddings, allow_pickle=False)

        return cls(muq_embeddings=muq_embeddings)
