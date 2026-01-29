from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from vibematcher.features.librosa_embedder import LibrosaEmbedder


@dataclass
class LibrosaEmbedding:
    """
    Stores librosa features for a chunk by feature name.
    Each frame feature has shape (T, D_feat).
    Cached on disk in: <audio_parent>/<audio_stem>/librosa_features.npz
    """

    mfcc_frames: np.ndarray
    chroma_frames: np.ndarray
    spectral_centroid_frames: np.ndarray
    rms_frames: np.ndarray

    CACHE_FILENAME = "librosa_features.npz"

    def frames_by_feature(self) -> Dict[str, np.ndarray]:
        return {
            "mfcc": self.mfcc_frames,
            "chroma": self.chroma_frames,
            "spectral_centroid": self.spectral_centroid_frames,
            "rms": self.rms_frames,
        }

    @staticmethod
    def stack_frames(frames_by_feature: Dict[str, np.ndarray]) -> np.ndarray:
        mats = [frames_by_feature[k] for k in frames_by_feature.keys()]
        if not mats:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(mats, axis=1).astype(np.float32, copy=False)

    def save_to_dir(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        out = d / self.CACHE_FILENAME
        np.savez_compressed(
            out,
            mfcc_frames=self.mfcc_frames.astype(np.float32, copy=False),
            chroma_frames=self.chroma_frames.astype(np.float32, copy=False),
            spectral_centroid_frames=self.spectral_centroid_frames.astype(
                np.float32, copy=False
            ),
            rms_frames=self.rms_frames.astype(np.float32, copy=False),
        )

    @classmethod
    def _load_npz(cls, path: Path) -> "LibrosaEmbedding":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                mfcc_frames=np.asarray(data["mfcc_frames"], dtype=np.float32),
                chroma_frames=np.asarray(data["chroma_frames"], dtype=np.float32),
                spectral_centroid_frames=np.asarray(
                    data["spectral_centroid_frames"], dtype=np.float32
                ),
                rms_frames=np.asarray(data["rms_frames"], dtype=np.float32),
            )

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        embedder: Optional[LibrosaEmbedder] = None,
        force_recompute: bool = False,
    ) -> "LibrosaEmbedding":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem
        emb_path = cache_dir / cls.CACHE_FILENAME

        if not force_recompute and cache_dir.exists() and emb_path.exists():
            return cls._load_npz(emb_path)

        if embedder is None:
            embedder = LibrosaEmbedder()

        frames_by_feature = embedder.embed(audio_path)
        empty_frames = np.zeros((0, 0), dtype=np.float32)

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            emb_path,
            mfcc_frames=frames_by_feature.get("mfcc", empty_frames).astype(
                np.float32, copy=False
            ),
            chroma_frames=frames_by_feature.get("chroma", empty_frames).astype(
                np.float32, copy=False
            ),
            spectral_centroid_frames=frames_by_feature.get(
                "spectral_centroid", empty_frames
            ).astype(np.float32, copy=False),
            rms_frames=frames_by_feature.get("rms", empty_frames).astype(
                np.float32, copy=False
            ),
        )
        return cls(
            mfcc_frames=frames_by_feature.get("mfcc", empty_frames),
            chroma_frames=frames_by_feature.get("chroma", empty_frames),
            spectral_centroid_frames=frames_by_feature.get(
                "spectral_centroid", empty_frames
            ),
            rms_frames=frames_by_feature.get("rms", empty_frames),
        )
