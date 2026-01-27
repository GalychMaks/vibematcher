from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class F0FrameExtractor:
    def __init__(
        self,
        *,
        target_sr: Optional[int] = 16000,
        hop_length: int = 160,  # 10ms at 16kHz
        frame_length: int = 1024,
        fmin: float = 32.7,
        fmax: float = 2093.0,
        fill_unvoiced: str = "nan",  # "nan" | "zero"
    ):
        self.target_sr = target_sr
        self.hop_length = int(hop_length)
        self.frame_length = int(frame_length)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.fill_unvoiced = fill_unvoiced

        if self.fill_unvoiced not in {"nan", "zero"}:
            raise ValueError("fill_unvoiced must be 'nan' or 'zero'.")

    def extract(self, path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          times_s: (N,) float32
          f0_hz:   (N,) float32  (NaN or 0 for unvoiced frames)
          voiced:  (N,) bool
        """
        path = Path(path)
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)

        if self.target_sr is not None and sr != self.target_sr:
            audio = resample_poly(audio, self.target_sr, sr).astype(
                np.float32, copy=False
            )
            sr = self.target_sr

        import librosa

        f0, voiced_flag, _voiced_prob = librosa.pyin(
            audio,
            sr=sr,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        voiced = np.asarray(voiced_flag, dtype=bool)
        f0 = np.asarray(f0, dtype=np.float32)

        if self.fill_unvoiced == "zero":
            f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32, copy=False)

        times_s = librosa.times_like(f0, sr=sr, hop_length=self.hop_length).astype(
            np.float32
        )

        return times_s, f0.astype(np.float32, copy=False), voiced
