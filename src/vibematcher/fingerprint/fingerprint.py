from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibematcher.fingerprint.mert_embedder import MertEmbedder


@dataclass
class AudioFingerprint:
    # (num_chunks, embedding_dim) float32
    mert_embeddings: np.ndarray

    @classmethod
    def from_audio_file(cls, audio_path: str | Path) -> "AudioFingerprint":
        embedder = MertEmbedder()
        mert_embeddings = embedder.embed(audio_path)  # (N, D) float32

        # sanity check (optional but helpful)
        if mert_embeddings.ndim != 2:
            raise ValueError(
                f"Expected mert_embeddings to be 2D (num_chunks, dim), "
                f"got shape={mert_embeddings.shape}"
            )

        return cls(mert_embeddings=mert_embeddings)

    def save_to_dir(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        np.save(
            dir_path / "mert_embeddings.npy",
            self.mert_embeddings.astype(np.float32, copy=False),
        )

    @classmethod
    def load_from_dir(cls, dir_path: str | Path) -> "AudioFingerprint":
        dir_path = Path(dir_path)
        path = dir_path / "mert_embeddings.npy"

        mert_embeddings = np.load(path).astype(np.float32, copy=False)
        if mert_embeddings.ndim != 2:
            raise ValueError(
                f"Expected mert_embeddings.npy to contain a 2D array (num_chunks, dim), "
                f"got shape={mert_embeddings.shape}"
            )

        return cls(mert_embeddings=mert_embeddings)
