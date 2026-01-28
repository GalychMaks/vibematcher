from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from vibematcher.features.mert_embedder import MertEmbedder


@dataclass
class MertEmbedding:
    """
    Stores chunked MERT embeddings: shape (num_chunks, D) float32.
    Cached on disk in: <audio_parent>/<audio_stem>/mert_embedding.npy
    """

    embedding: np.ndarray

    CACHE_FILENAME = "mert_embedding.npy"

    def save_to_dir(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        out = d / self.CACHE_FILENAME
        np.save(out, self.embedding.astype(np.float32, copy=False), allow_pickle=False)

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        mert_embedder: Optional[MertEmbedder] = None,
        force_recompute: bool = False,
    ) -> "MertEmbedding":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem
        emb_path = cache_dir / cls.CACHE_FILENAME
        # If already computed -> load
        if not force_recompute and cache_dir.exists() and emb_path.exists():
            return np.asarray(
                np.load(emb_path, allow_pickle=False),
                dtype=np.float32,
            )

        # Else compute + save
        if mert_embedder is None:
            mert_embedder = MertEmbedder()

        embedding = mert_embedder.embed(audio_path).astype(np.float32, copy=False)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embedding, allow_pickle=False)

        return cls(embedding=embedding)
