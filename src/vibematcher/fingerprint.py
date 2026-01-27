from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibematcher.mert_embedder import MertEmbedder


@dataclass
class AudioFingerprint:
    # Aspects
    mert_embedding: np.ndarray

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
    ) -> "AudioFingerprint":
        # MERT embedding
        embedder = MertEmbedder()
        mert_embedding = embedder.embed(audio_path)

        fp = cls(mert_embedding=mert_embedding)

        return fp

    def save_to_dir(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        np.save(
            dir_path / "mert_embedding.npy",
            self.mert_embedding.astype(np.float32, copy=False),
        )

    @classmethod
    def load_from_dir(cls, dir_path: str | Path) -> "AudioFingerprint":
        dir_path = Path(dir_path)

        mert_embedding_path = dir_path / "mert_embedding.npy"

        fp = cls(
            mert_embedding=np.load(mert_embedding_path).astype(np.float32, copy=False)
        )
        return fp
