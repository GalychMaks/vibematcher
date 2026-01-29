from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from vibematcher.features.f0_wrapper import RMVPEF0Extractor


@dataclass
class F0Embedding:
    """
    Stores RMVPE f0 features for a chunk: shape (T,) float32, NaN for unvoiced.
    Cached on disk in: <audio_parent>/<audio_stem>/f0_rmvpe.npy
    """

    f0_hz: np.ndarray

    CACHE_FILENAME = "f0_rmvpe.npy"

    def save_to_dir(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        out = d / self.CACHE_FILENAME
        np.save(out, self.f0_hz.astype(np.float32, copy=False), allow_pickle=False)

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        embedder: Optional[RMVPEF0Extractor] = None,
        force_recompute: bool = False,
    ) -> "F0Embedding":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem
        emb_path = cache_dir / cls.CACHE_FILENAME
        if not force_recompute and cache_dir.exists() and emb_path.exists():
            return cls(
                f0_hz=np.asarray(
                    np.load(emb_path, allow_pickle=False), dtype=np.float32
                )
            )

        if embedder is None:
            embedder = RMVPEF0Extractor()

        f0_hz = embedder.extract_f0(audio_path).astype(np.float32, copy=False)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, f0_hz, allow_pickle=False)

        return cls(f0_hz=f0_hz)


if __name__ == "__main__":
    v = F0Embedding.from_audio_file(
        "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav"
    ).f0_hz

    print("feature shape:", v.shape, "dtype:", v.dtype)
    print()
    print(v[:100])
