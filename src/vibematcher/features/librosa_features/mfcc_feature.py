from __future__ import annotations

import librosa
import numpy as np


def extract(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mfcc: int,
) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
