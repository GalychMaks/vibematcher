from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
from librosa.feature import (
    chroma_stft,
    mfcc,
    rms,
    spectral_bandwidth,
    spectral_centroid,
    spectral_rolloff,
    zero_crossing_rate,
)
from tqdm import tqdm

from vibematcher.chunking.audio_chunks import AudioChunks
from vibematcher.chunking.song_former import SongFormer


@dataclass(frozen=True)
class LibrosaFeatureChunkResult:
    q_chunk_path: Path
    r_chunk_path: Path
    score: float
    features_query: np.ndarray
    features_reference: np.ndarray


class LibrosaFeatureCompare:
    """
    Compare audio using analytical features from librosa, but at CHUNK level.

    Workflow:
      1) Split query into chunks via SongFormer/AudioChunks
      2) Split each reference into chunks
      3) Extract a single feature vector per chunk
      4) For each reference chunk, find the best matching query chunk (highest cosine sim)
      5) Return results containing q_chunk_path and r_chunk_path (like your MelodySimCompare)

    Similarity score is cosine similarity mapped to [0, 1].
    """

    def extract_features_for_file(self, path: str | Path) -> np.ndarray:
        """
        Extract a feature vector from an audio file using signal processing features.
        """
        y, sr = librosa.load(str(path), sr=None, mono=True)

        mfcc_ft = mfcc(y=y, sr=sr, n_mfcc=13)  # (13, T)
        chroma_ft = chroma_stft(y=y, sr=sr)  # (12, T)
        spectral_centroid_ft = spectral_centroid(y=y, sr=sr)  # (1, T)
        spectral_bandwidth_ft = spectral_bandwidth(y=y, sr=sr)  # (1, T)
        spectral_rolloff_ft = spectral_rolloff(y=y, sr=sr)  # (1, T)
        zero_crossings_rate_ft = zero_crossing_rate(y)  # (1, T)
        rms_ft = rms(y=y)  # (1, T)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # scalar
        tempo_val = float(np.asarray(tempo))

        features = np.concatenate(
            [
                np.mean(mfcc_ft, axis=1),
                np.std(mfcc_ft, axis=1),
                np.mean(chroma_ft, axis=1),
                np.array([np.mean(spectral_centroid_ft)]),
                np.array([np.mean(spectral_bandwidth_ft)]),
                np.array([np.mean(spectral_rolloff_ft)]),
                np.array([np.mean(zero_crossings_rate_ft)]),
                np.array([np.mean(rms_ft)]),
                np.array([tempo_val]),
            ]
        ).astype(np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        cos = float(np.dot(a, b) / denom)
        cos = float(np.clip(cos, -1.0, 1.0))
        return (cos + 1.0) / 2.0

    def compare(
        self,
        query: str | Path,
        references: list[str | Path],
        *,
        force_recompute: bool = False,
    ) -> list[LibrosaFeatureChunkResult]:
        """
        Returns one result per reference chunk:
          - r_chunk_path is that chunk
          - q_chunk_path is the best-matching query chunk
        """
        song_former = SongFormer()

        q_chunks: List[Path] = AudioChunks.from_audio_file(
            query,
            song_former=song_former,
            force_recompute=force_recompute,
        ).chunks
        if not q_chunks:
            raise ValueError(f"No chunks found in query audio: {query}")

        # Precompute query chunk features once
        q_feats: dict[Path, np.ndarray] = {}
        for q_chunk in q_chunks:
            q_feats[q_chunk] = self.extract_features_for_file(q_chunk)

        results: list[LibrosaFeatureChunkResult] = []

        for reference in tqdm(references):
            r_chunks: List[Path] = AudioChunks.from_audio_file(
                reference,
                song_former=song_former,
                force_recompute=force_recompute,
            ).chunks
            if not r_chunks:
                raise ValueError(f"No chunks found in reference audio: {reference}")

            for r_chunk in r_chunks:
                r_feat = self.extract_features_for_file(r_chunk)

                best_q_chunk: Path | None = None
                best_score = float("-inf")
                best_q_feat: np.ndarray | None = None

                for q_chunk, q_feat in q_feats.items():
                    s = self._cosine_similarity(q_feat, r_feat)
                    if s > best_score:
                        best_score = s
                        best_q_chunk = q_chunk
                        best_q_feat = q_feat

                assert best_q_chunk is not None and best_q_feat is not None

                results.append(
                    LibrosaFeatureChunkResult(
                        q_chunk_path=best_q_chunk,
                        r_chunk_path=r_chunk,
                        score=float(best_score),
                        features_query=best_q_feat,
                        features_reference=r_feat,
                    )
                )

        return results

    def scores(
        self, query: str | Path, references: list[str | Path], **kwargs
    ) -> list[float]:
        return [r.score for r in self.compare(query, references, **kwargs)]


if __name__ == "__main__":
    # Example:
    #   python -m vibematcher.features.librosa_sim_chunks -- adjust paths as needed
    comparator = LibrosaFeatureCompare()

    query = Path("data/comparison/Smoke On The Water _2024 Remastered_.wav")
    references = [Path(path) for path in Path("data/original").glob("*.wav")]
    if not references:
        raise SystemExit("No reference files found in data/original")

    results = comparator.compare(query=query, references=references)

    # Rank by best chunk-pair score across ALL returned results
    pairs = list(
        zip(
            [r.q_chunk_path for r in results],
            [r.r_chunk_path for r in results],
            [r.score for r in results],
        )
    )
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Query: {query}")
    print("score     | q_chunk | r_chunk")
    print("-" * 80)
    for q_chunk, r_chunk, score in pairs[:50]:
        print(f"{score:0.6f}  {q_chunk}  {r_chunk}")
