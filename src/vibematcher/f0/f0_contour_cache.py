from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from vibematcher.f0.f0_wrapper import RMVPEF0Extractor


@dataclass
class F0Contour:
    f0_hz: np.ndarray

    CACHE_FILENAME_HZ = "f0_rmvpe.npy"

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        f0_extractor: Optional[RMVPEF0Extractor] = None,
        force_recompute: bool = False,
    ) -> "F0Contour":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem

        f0_hz_path = cache_dir / cls.CACHE_FILENAME_HZ

        if not force_recompute and cache_dir.exists() and f0_hz_path.exists():
            f0_hz = np.asarray(
                np.load(f0_hz_path, allow_pickle=False), dtype=np.float32
            )

            return cls(f0_hz=f0_hz)

        if f0_extractor is None:
            f0_extractor = RMVPEF0Extractor()

        f0_hz = f0_extractor.extract_f0(audio_path).astype(np.float32, copy=False)

        f0_hz_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(f0_hz_path, f0_hz, allow_pickle=False)

        return cls(f0_hz=f0_hz)


if __name__ == "__main__":
    emb = F0Contour.from_audio_file(
        "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav"
    )

    print("feature shape:", emb.f0_hz.shape, "dtype:", emb.f0_hz.dtype)
    print()
    print(emb.f0_hz[:100])
