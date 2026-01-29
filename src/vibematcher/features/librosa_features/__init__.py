from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from vibematcher.features.librosa_features.chroma_feature import extract as chroma
from vibematcher.features.librosa_features.mfcc_feature import extract as mfcc
from vibematcher.features.librosa_features.rms_feature import extract as rms
from vibematcher.features.librosa_features.spectral_centroid_feature import (
    extract as spectral_centroid,
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    extractor: Callable[[np.ndarray, int, int, int, int], np.ndarray]


DEFAULT_FEATURES: List[FeatureSpec] = [
    FeatureSpec("mfcc", mfcc),
    FeatureSpec("chroma", chroma),
    FeatureSpec("spectral_centroid", spectral_centroid),
    FeatureSpec("rms", rms),
]
