from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa


class F0Extractor:
    def __init__(
        self,
        *,
        target_sr: Optional[int] = 16000,
        hop_length: int = 160,  # 10ms at 16kHz
        frame_length: int = 1024,
        fmin: float = 32.7,
        fmax: float = 2093.0,
    ):
        self.target_sr = target_sr
        self.hop_length = int(hop_length)
        self.frame_length = int(frame_length)
        self.fmin = float(fmin)
        self.fmax = float(fmax)

    def extract_f0(self, path: str | Path) -> np.ndarray:
        """
        Returns:
          f0_hz: (N,) float32, NaN for unvoiced frames
        """

        path = Path(path)
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)

        if self.target_sr is not None and sr != self.target_sr:
            # librosa expects float, and returns float
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.target_sr,
            ).astype(np.float32, copy=False)
            sr = self.target_sr

        f0, _, _ = librosa.pyin(
            audio,
            sr=sr,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        return np.asarray(f0, dtype=np.float32)
