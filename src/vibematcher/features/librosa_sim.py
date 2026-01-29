from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from dtw import dtw
from tqdm import tqdm

from vibematcher.chunking.audio_chunks import AudioChunks
from vibematcher.chunking.song_former import SongFormer
from vibematcher.features.librosa_embedder import LibrosaEmbedder
from vibematcher.features.librosa_embedding import LibrosaEmbedding


@dataclass(frozen=True)
class LibrosaSimChunkResult:
    q_chunk_path: Path
    r_chunk_path: Path
    score: float
    cosine_sim: float
    dtw_distance: float
    dtw_sim: float


class LibrosaSimCompare:
    """
    Compare audio using librosa-based features.

    Workflow:
      1) Split query into chunks via SongFormer/AudioChunks
      2) Split each reference into chunks
      3) Extract cached librosa features per chunk
      4) For each reference chunk, find the best matching query chunk
         using cosine similarity (summary stats) and DTW similarity (frame features)
    """

    def __init__(
        self,
        *,
        embedder: Optional[LibrosaEmbedder] = None,
        cosine_weight: float = 0.5,
        dtw_weight: float = 0.5,
        normalize_frames: bool = True,
    ) -> None:
        self.embedder = embedder or LibrosaEmbedder()
        self.cosine_weight = float(cosine_weight)
        self.dtw_weight = float(dtw_weight)
        self.normalize_frames = bool(normalize_frames)

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

    @staticmethod
    def _summary_from_frames(frames: np.ndarray) -> np.ndarray:
        if frames.size == 0:
            return np.zeros((0,), dtype=np.float32)
        mean = frames.mean(axis=0)
        std = frames.std(axis=0)
        summary = np.concatenate([mean, std]).astype(np.float32, copy=False)
        return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normalize_frames(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (x - mean) / std

    def _dtw_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return float("inf")
        if self.normalize_frames:
            a = self._normalize_frames(a)
            b = self._normalize_frames(b)
        alignment = dtw(a, b, dist_method="euclidean")
        return float(alignment.distance)

    @staticmethod
    def _dtw_similarity(distance: float) -> float:
        if not np.isfinite(distance):
            return 0.0
        return 1.0 / (1.0 + distance)

    def _combined_score(self, cosine_sim: float, dtw_sim: float) -> float:
        weight_sum = self.cosine_weight + self.dtw_weight
        if weight_sum <= 0:
            return 0.0
        return (
            self.cosine_weight * cosine_sim + self.dtw_weight * dtw_sim
        ) / weight_sum

    def compare(
        self,
        query: str | Path,
        references: list[str | Path],
        *,
        force_recompute: bool = False,
    ) -> list[LibrosaSimChunkResult]:
        song_former = SongFormer()

        q_chunks = AudioChunks.from_audio_file(
            query,
            song_former=song_former,
            force_recompute=force_recompute,
        ).chunks
        if not q_chunks:
            raise ValueError(f"No chunks found in query audio: {query}")

        q_embeddings: dict[Path, LibrosaEmbedding] = {}
        for q_chunk in q_chunks:
            q_embeddings[q_chunk] = LibrosaEmbedding.from_audio_file(
                q_chunk,
                embedder=self.embedder,
                force_recompute=force_recompute,
            )

        results: list[LibrosaSimChunkResult] = []

        for reference in tqdm(references):
            r_chunks = AudioChunks.from_audio_file(
                reference,
                song_former=song_former,
                force_recompute=force_recompute,
            ).chunks
            if not r_chunks:
                raise ValueError(f"No chunks found in reference audio: {reference}")

            for r_chunk in r_chunks:
                r_embedding = LibrosaEmbedding.from_audio_file(
                    r_chunk,
                    embedder=self.embedder,
                    force_recompute=force_recompute,
                )

                best_q_chunk: Path | None = None
                best_score = float("-inf")
                best_cos = 0.0
                best_dtw_dist = float("inf")
                best_dtw_sim = 0.0

                for q_chunk, q_embedding in q_embeddings.items():
                    q_frames = LibrosaEmbedding.stack_frames(
                        q_embedding.frames_by_feature()
                    )
                    r_frames = LibrosaEmbedding.stack_frames(
                        r_embedding.frames_by_feature()
                    )

                    q_summary = self._summary_from_frames(q_frames)
                    r_summary = self._summary_from_frames(r_frames)

                    cosine_sim = self._cosine_similarity(q_summary, r_summary)
                    dtw_dist = self._dtw_distance(q_frames, r_frames)
                    dtw_sim = self._dtw_similarity(dtw_dist)
                    score = self._combined_score(cosine_sim, dtw_sim)

                    if score > best_score:
                        best_score = score
                        best_q_chunk = q_chunk
                        best_cos = cosine_sim
                        best_dtw_dist = dtw_dist
                        best_dtw_sim = dtw_sim

                assert best_q_chunk is not None

                results.append(
                    LibrosaSimChunkResult(
                        q_chunk_path=best_q_chunk,
                        r_chunk_path=r_chunk,
                        score=float(best_score),
                        cosine_sim=float(best_cos),
                        dtw_distance=float(best_dtw_dist),
                        dtw_sim=float(best_dtw_sim),
                    )
                )

        return results

    def scores(
        self, query: str | Path, references: list[str | Path], **kwargs
    ) -> list[float]:
        return [r.score for r in self.compare(query, references, **kwargs)]


if __name__ == "__main__":
    comparator = LibrosaSimCompare()

    query = Path("data/comparison/Smoke On The Water _2024 Remastered_.wav")
    references = Path("data/original").glob("*.wav")
    if not references:
        raise SystemExit("No reference files found in data/original")

    results = comparator.compare(
        query=query, references=references, force_recompute=True
    )

    pairs = list(
        zip(
            [r.q_chunk_path for r in results],
            [r.r_chunk_path for r in results],
            [r.score for r in results],
            [r.cosine_sim for r in results],
            [r.dtw_sim for r in results],
        )
    )
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Query: {query}")
    print("score     | cosine  | dtw     | q_chunk | r_chunk")
    print("-" * 100)
    for q_chunk, r_chunk, score, cos, dtw_sim in pairs[:50]:
        print(f"{score:0.6f}  {cos:0.6f}  {dtw_sim:0.6f}  {q_chunk}  {r_chunk}")
