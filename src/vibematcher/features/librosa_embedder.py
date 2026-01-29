from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np

from vibematcher.features.librosa_features import DEFAULT_FEATURES, FeatureSpec


class LibrosaEmbedder:
    """
    Extracts frame-level features from audio using librosa.

    Frame features are stacked as (T, D) where D is the number of features per frame.
    """

    def __init__(
        self,
        *,
        sr: int | None = None,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        features: List[FeatureSpec] | None = None,
    ) -> None:
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.features = features or DEFAULT_FEATURES

    def _align_frames(self, *feats: np.ndarray) -> list[np.ndarray]:
        min_frames = min(f.shape[1] for f in feats)
        if min_frames <= 0:
            return [f[:, :0] for f in feats]
        return [f[:, :min_frames] for f in feats]

    def embed(self, path: str | Path) -> Dict[str, np.ndarray]:
        y, sr = librosa.load(str(path), sr=self.sr, mono=True)
        if not self.features:
            return {}

        feature_mats = []
        feature_names = []
        for spec in self.features:
            feature_names.append(spec.name)
            feature_mats.append(
                spec.extractor(y, sr, self.n_fft, self.hop_length, self.n_mfcc)
            )

        feature_mats = self._align_frames(*feature_mats)

        frames_by_feature: Dict[str, np.ndarray] = {}
        for name, feat in zip(feature_names, feature_mats):
            frame = feat.T.astype(np.float32, copy=False)
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            frames_by_feature[name] = frame
        return frames_by_feature
