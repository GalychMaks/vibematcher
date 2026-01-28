from dataclasses import dataclass
from pathlib import Path
import numpy as np

from vibematcher.mert_embedder import MertEmbedderChunked  # <- your chunked embedder


@dataclass
class AudioFingerprint:
    # Chunked MERT embeddings: shape [num_chunks, feature_dim]
    mert_embeddings: np.ndarray

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        window_len_sec: float = 10.0,
        hop_len_sec: float = 5.0,
    ) -> "AudioFingerprint":
        """
        Compute chunked MERT embeddings from an audio file.
        """
        embedder = MertEmbedderChunked()
        mert_embeddings = embedder.embed_chunks(
            audio_path, window_len_sec=window_len_sec, hop_len_sec=hop_len_sec
        )
        return cls(mert_embeddings=mert_embeddings)

    def save_to_dir(self, dir_path: Path) -> None:
        """
        Save chunked embeddings to directory as a .npy file.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        np.save(
            dir_path / "mert_embeddings.npy",
            self.mert_embeddings.astype(np.float32, copy=False),
        )

    @classmethod
    def load_from_dir(cls, dir_path: str | Path) -> "AudioFingerprint":
        """
        Load previously saved chunked embeddings from directory.
        """
        dir_path = Path(dir_path)
        embeddings_path = dir_path / "mert_embeddings.npy"

        return cls(
            mert_embeddings=np.load(embeddings_path).astype(np.float32, copy=False)
        )