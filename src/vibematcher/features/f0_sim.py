from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dtw import dtw
from tqdm import tqdm

from vibematcher.chunking.audio_chunks import AudioChunks
from vibematcher.chunking.song_former import SongFormer
from vibematcher.features.f0_embedding import F0Embedding
from vibematcher.features.f0_wrapper import RMVPEF0Extractor


@dataclass(frozen=True)
class F0SimChunkResult:
    q_chunk_path: Path
    r_chunk_path: Path
    score: float
    dtw_distance: float
    dtw_sim: float


class F0SimCompare:
    """
    Compare audio using RMVPE-extracted f0 and DTW alignment.

    Workflow:
      1) Split query into chunks via SongFormer/AudioChunks
      2) Split each reference into chunks
      3) Extract cached f0 per chunk
      4) For each reference chunk, find the best matching query chunk using DTW
    """

    def __init__(self) -> None:
        self.embedder = RMVPEF0Extractor()

    @staticmethod
    def _f0_hz_to_cents(f0_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
        f0 = np.asarray(f0_hz, dtype=np.float32)
        cents = np.full_like(f0, np.nan, dtype=np.float32)
        voiced = np.isfinite(f0) & (f0 > 0.0)
        if np.any(voiced):
            cents[voiced] = 1200.0 * np.log2(f0[voiced] / float(ref_hz))
        return cents

    @staticmethod
    def _center_cents_by_median(cents: np.ndarray) -> np.ndarray:
        x = np.asarray(cents, dtype=np.float32)
        voiced = np.isfinite(x)
        if not np.any(voiced):
            return x
        med = float(np.nanmedian(x[voiced]))
        return x - med

    @staticmethod
    def _dtw_distance(q_f0: np.ndarray, r_f0: np.ndarray) -> float:
        q_cents = F0SimCompare._center_cents_by_median(
            F0SimCompare._f0_hz_to_cents(q_f0)
        )
        r_cents = F0SimCompare._center_cents_by_median(
            F0SimCompare._f0_hz_to_cents(r_f0)
        )

        q = q_cents[np.isfinite(q_cents)]
        r = r_cents[np.isfinite(r_cents)]
        if q.size < 2 or r.size < 2:
            return float("inf")

        aln = dtw(q.reshape(-1, 1), r.reshape(-1, 1))
        avg_cost = aln.distance / max(1, len(aln.index1))
        return float(avg_cost)

    @staticmethod
    def _dtw_similarity(distance: float) -> float:
        if not np.isfinite(distance):
            return 0.0
        return 1.0 / (1.0 + distance)

    def compare(
        self,
        query: str | Path,
        references: list[str | Path],
        *,
        force_recompute: bool = False,
    ) -> list[F0SimChunkResult]:
        song_former = SongFormer()

        q_chunks = AudioChunks.from_audio_file(
            query,
            song_former=song_former,
            force_recompute=force_recompute,
        ).chunks
        if not q_chunks:
            raise ValueError(f"No chunks found in query audio: {query}")

        q_f0: dict[Path, np.ndarray] = {}
        for q_chunk in q_chunks:
            q_f0[q_chunk] = F0Embedding.from_audio_file(
                q_chunk,
                embedder=self.embedder,
                force_recompute=force_recompute,
            ).f0_hz

        results: list[F0SimChunkResult] = []

        for reference in tqdm(references):
            r_chunks = AudioChunks.from_audio_file(
                reference,
                song_former=song_former,
                force_recompute=force_recompute,
            ).chunks
            if not r_chunks:
                raise ValueError(f"No chunks found in reference audio: {reference}")

            for r_chunk in r_chunks:
                r_f0 = F0Embedding.from_audio_file(
                    r_chunk,
                    embedder=self.embedder,
                    force_recompute=force_recompute,
                ).f0_hz

                best_q_chunk: Path | None = None
                best_score = float("-inf")
                best_dtw_dist = float("inf")
                best_dtw_sim = 0.0

                for q_chunk, q_seq in q_f0.items():
                    dtw_dist = self._dtw_distance(q_seq, r_f0)
                    dtw_sim = self._dtw_similarity(dtw_dist)

                    if dtw_sim > best_score:
                        best_score = dtw_sim
                        best_q_chunk = q_chunk
                        best_dtw_dist = dtw_dist
                        best_dtw_sim = dtw_sim

                assert best_q_chunk is not None

                results.append(
                    F0SimChunkResult(
                        q_chunk_path=best_q_chunk,
                        r_chunk_path=r_chunk,
                        score=float(best_score),
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
    comparator = F0SimCompare()

    query = Path("data/comparison/Smoke On The Water _2024 Remastered_.wav")
    references = [Path(path) for path in Path("data/original").glob("*.wav")]
    if not references:
        raise SystemExit("No reference files found in data/original")

    results = comparator.compare(query=query, references=references)

    pairs = list(
        zip(
            [r.q_chunk_path for r in results],
            [r.r_chunk_path for r in results],
            [r.score for r in results],
            [r.dtw_sim for r in results],
        )
    )
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Query: {query}")
    print("score     | dtw     | q_chunk | r_chunk")
    print("-" * 100)
    for q_chunk, r_chunk, score, dtw_sim in pairs[:50]:
        print(f"{score:0.6f}  {dtw_sim:0.6f}  {q_chunk}  {r_chunk}")
