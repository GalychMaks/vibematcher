import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibematcher.features.mert_embedder import MertEmbedder
from vibematcher.features.mert_embedding import MertEmbedding
from vibematcher.fingerprint.f0_extractor import F0Extractor


@dataclass
class AudioFingerprint:
    mert_embedding: MertEmbedding

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        mert_embedder: MertEmbedder | None = None,
        f0_extractor: F0Extractor | None = None,
    ) -> "AudioFingerprint":
        audio_path = Path(audio_path)

        mert_embedder = mert_embedder or MertEmbedder()
        f0_extractor = f0_extractor or F0Extractor()

        mert_embeddings = mert_embedder.embed(audio_path)  # (N, D) float32
        f0_hz = f0_extractor.extract_f0(audio_path)  # (T,) float32 (NaN = unvoiced)

        # sanity checks (optional but helpful)
        if mert_embeddings.ndim != 2:
            raise ValueError(
                f"Expected mert_embeddings to be 2D (num_chunks, dim), "
                f"got shape={mert_embeddings.shape}"
            )
        if f0_hz.ndim != 1:
            raise ValueError(
                f"Expected f0_hz to be 1D (num_frames,), got shape={f0_hz.shape}"
            )

        return cls(
            mert_embeddings=np.asarray(mert_embeddings, dtype=np.float32),
            f0_hz=np.asarray(f0_hz, dtype=np.float32),
        )

    def save_to_dir(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        np.save(
            dir_path / "mert_embeddings.npy",
            self.mert_embeddings.astype(np.float32, copy=False),
        )
        np.save(
            dir_path / "f0_hz.npy",
            self.f0_hz.astype(np.float32, copy=False),
        )
        metadata = {
            "mert_embeddings": {
                "shape": list(self.mert_embeddings.shape),
                "dtype": "float32",
            },
            "f0_hz": {"shape": list(self.f0_hz.shape), "dtype": "float32"},
        }
        (dir_path / "metadata.json").write_text(
            json.dumps(metadata, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def load_from_dir(cls, dir_path: str | Path) -> "AudioFingerprint":
        dir_path = Path(dir_path)

        mert_path = dir_path / "mert_embeddings.npy"
        f0_path = dir_path / "f0_hz.npy"

        mert_embeddings = np.load(mert_path).astype(np.float32, copy=False)
        f0_hz = np.load(f0_path).astype(np.float32, copy=False)

        if mert_embeddings.ndim != 2:
            raise ValueError(
                f"Expected mert_embeddings.npy to contain a 2D array (num_chunks, dim), "
                f"got shape={mert_embeddings.shape}"
            )
        if f0_hz.ndim != 1:
            raise ValueError(
                f"Expected f0_hz.npy to contain a 1D array (num_frames,), "
                f"got shape={f0_hz.shape}"
            )

        return cls(mert_embeddings=mert_embeddings, f0_hz=f0_hz)
