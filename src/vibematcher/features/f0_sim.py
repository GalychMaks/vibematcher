from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    voiced_ratio_q: float
    voiced_ratio_r: float


class F0SimCompare:
    """
    Compare audio using RMVPE-extracted f0 and DTW alignment.

    Key points:
      - Convert Hz -> cents for perceptual distance
      - Center by voiced median for key invariance
      - Handle unvoiced frames with an explicit penalty
      - (Optionally) constrain warping with Sakoe–Chiba band
      - Normalize DTW cost by path length
      - Convert normalized cost -> similarity via exp(-d/sigma)
    """

    def __init__(
        self,
        *,
        # DTW / preprocessing knobs
        ref_hz_for_cents: float = 55.0,
        unvoiced_cost_cents: float = 600.0,
        sakoe_chiba_band: Optional[
            int
        ] = 150,  # in frames after downsampling; None = unconstrained
        max_frames: Optional[
            int
        ] = 2000,  # cap sequence length for speed; None = no cap
        downsample_to: Optional[
            int
        ] = None,  # e.g. 1200 to resample each seq to a fixed length
        # similarity transform
        sim_sigma_cents: float = 300.0,
        min_voiced_frames: int = 8,
    ) -> None:
        self.embedder = RMVPEF0Extractor()

        self.ref_hz_for_cents = float(ref_hz_for_cents)
        self.unvoiced_cost_cents = float(unvoiced_cost_cents)
        self.sakoe_chiba_band = sakoe_chiba_band
        self.max_frames = max_frames
        self.downsample_to = downsample_to

        self.sim_sigma_cents = float(sim_sigma_cents)
        self.min_voiced_frames = int(min_voiced_frames)

    # -----------------------
    # Preprocessing utilities
    # -----------------------

    @staticmethod
    def _as_float_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(-1)

    def _f0_hz_to_cents(self, f0_hz: np.ndarray) -> np.ndarray:
        """
        Returns cents array with NaN for unvoiced/invalid frames.
        """
        f0 = self._as_float_1d(f0_hz)
        cents = np.full_like(f0, np.nan, dtype=np.float32)
        voiced = np.isfinite(f0) & (f0 > 0.0)
        if np.any(voiced):
            cents[voiced] = 1200.0 * np.log2(f0[voiced] / self.ref_hz_for_cents)
        return cents

    @staticmethod
    def _center_cents_by_voiced_median(cents: np.ndarray) -> np.ndarray:
        """
        Subtract median of voiced frames -> key invariance.
        """
        x = np.asarray(cents, dtype=np.float32)
        voiced = np.isfinite(x)
        if not np.any(voiced):
            return x
        med = float(np.nanmedian(x[voiced]))
        return x - med

    @staticmethod
    def _voiced_ratio(x: np.ndarray) -> float:
        x = np.asarray(x)
        if x.size == 0:
            return 0.0
        return float(np.isfinite(x).sum() / x.size)

    @staticmethod
    def _downsample_nan_preserving(x: np.ndarray, target_len: int) -> np.ndarray:
        """
        Simple downsample that preserves NaNs by majority/mean rule per bin.
        Not perfect, but stable and fast.

        Strategy:
          - split into target_len bins
          - if a bin has any voiced values, take mean of voiced values
          - else NaN
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = x.size
        if target_len <= 0:
            return x
        if n == target_len:
            return x
        if n < target_len:
            # upsample by linear interpolation on voiced-only; keep unvoiced gaps as NaN
            # (simple, conservative)
            idx = np.linspace(0, n - 1, target_len).astype(np.float32)
            out = np.full((target_len,), np.nan, dtype=np.float32)
            voiced = np.isfinite(x)
            if voiced.sum() < 2:
                return out
            xv = np.where(voiced)[0].astype(np.float32)
            yv = x[voiced].astype(np.float32)
            out = np.interp(idx, xv, yv).astype(np.float32)
            return out

        # n > target_len: binning downsample
        edges = np.linspace(0, n, target_len + 1).astype(np.int32)
        out = np.full((target_len,), np.nan, dtype=np.float32)
        for i in range(target_len):
            a, b = int(edges[i]), int(edges[i + 1])
            if b <= a:
                continue
            seg = x[a:b]
            voiced = np.isfinite(seg)
            if np.any(voiced):
                out[i] = float(np.mean(seg[voiced]))
        return out

    def _maybe_limit_and_downsample(self, cents: np.ndarray) -> np.ndarray:
        x = np.asarray(cents, dtype=np.float32).reshape(-1)

        # cap max frames (for DTW speed)
        if self.max_frames is not None and x.size > self.max_frames:
            # uniform downsample to max_frames
            x = self._downsample_nan_preserving(x, int(self.max_frames))

        # optionally resample to fixed length (useful if you want stable band meaning)
        if self.downsample_to is not None:
            x = self._downsample_nan_preserving(x, int(self.downsample_to))

        return x

    # -----------------------
    # DTW utilities
    # -----------------------

    def _frame_dist(self, a: float, b: float) -> float:
        """
        Distance between two frames (in cents), handling unvoiced as penalty.
        """
        a_f = np.isfinite(a)
        b_f = np.isfinite(b)
        if a_f and b_f:
            return float(abs(a - b))
        if (not a_f) and (not b_f):
            # both unvoiced: treat as equal (0) rather than penalize
            return 0.0
        # mismatch voiced/unvoiced
        return float(self.unvoiced_cost_cents)

    @staticmethod
    def _path_length_from_alignment(aln) -> int:
        """
        Robustly estimate path length across dtw-python variants.
        """
        # dtw-python typically has .index1/.index2 only if keep_internals=True
        idx1 = getattr(aln, "index1", None)
        if idx1 is not None:
            try:
                return int(len(idx1))
            except Exception:
                pass

        # some variants expose .index1s/.index2s
        idx1s = getattr(aln, "index1s", None)
        if idx1s is not None:
            try:
                return int(len(idx1s))
            except Exception:
                pass

        # fallback: assume at least 1 to avoid div-by-zero
        return 1

    def _dtw_distance(
        self, q_f0_hz: np.ndarray, r_f0_hz: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Returns (avg_cost, voiced_ratio_q, voiced_ratio_r)
        avg_cost is normalized by warping path length (smaller is better).
        """
        q_c = self._center_cents_by_voiced_median(self._f0_hz_to_cents(q_f0_hz))
        r_c = self._center_cents_by_voiced_median(self._f0_hz_to_cents(r_f0_hz))

        q_c = self._maybe_limit_and_downsample(q_c)
        r_c = self._maybe_limit_and_downsample(r_c)

        vr_q = self._voiced_ratio(q_c)
        vr_r = self._voiced_ratio(r_c)

        # Require some voiced content (otherwise DTW degenerates)
        if (
            np.isfinite(q_c).sum() < self.min_voiced_frames
            or np.isfinite(r_c).sum() < self.min_voiced_frames
        ):
            return float("inf"), vr_q, vr_r

        # dtw() expects 2D arrays: (T, D)
        q = q_c.reshape(-1, 1)
        r = r_c.reshape(-1, 1)

        # Sakoe–Chiba constraint (optional), but it MUST be wide enough
        window_type = None
        window_args = None
        if self.sakoe_chiba_band is not None:
            n = int(q.shape[0])
            m = int(r.shape[0])
            min_needed = abs(n - m)
            band = max(int(self.sakoe_chiba_band), min_needed)
            window_type = "sakoechiba"
            window_args = {"window_size": band}

        def dist_method(x: np.ndarray, y: np.ndarray) -> float:
            return self._frame_dist(float(x[0]), float(y[0]))

        # Try constrained DTW, fallback if constraints still make it impossible
        try:
            aln = dtw(
                q,
                r,
                dist_method=dist_method,
                window_type=window_type,
                window_args=window_args,
                keep_internals=True,
            )
        except ValueError as e:
            msg = str(e).lower()
            if "no warping path found" not in msg:
                raise
            # fallback: unconstrained
            aln = dtw(
                q,
                r,
                dist_method=dist_method,
                window_type=None,
                window_args=None,
                keep_internals=True,
            )

        path_len = self._path_length_from_alignment(aln)
        avg_cost = float(aln.distance) / float(max(1, path_len))
        return float(avg_cost), vr_q, vr_r

    def _dtw_similarity(self, avg_cost: float) -> float:
        """
        Convert normalized DTW cost (in cents) -> similarity in [0, 1].
        exp(-d/sigma) behaves nicely and is easy to tune via sigma.
        """
        if not np.isfinite(avg_cost):
            return 0.0
        # guard sigma
        sigma = max(1e-6, self.sim_sigma_cents)
        return float(np.exp(-avg_cost / sigma))

    # -----------------------
    # Public API
    # -----------------------

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

        # Cache query f0
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
                best_sim = float("-inf")
                best_avg_cost = float("inf")
                best_vrq = 0.0
                best_vrr = 0.0

                for q_chunk, q_seq in q_f0.items():
                    avg_cost, vrq, vrr = self._dtw_distance(q_seq, r_f0)
                    sim = self._dtw_similarity(avg_cost)

                    if sim > best_sim:
                        best_sim = sim
                        best_q_chunk = q_chunk
                        best_avg_cost = avg_cost
                        best_vrq = vrq
                        best_vrr = vrr

                assert best_q_chunk is not None

                results.append(
                    F0SimChunkResult(
                        q_chunk_path=best_q_chunk,
                        r_chunk_path=r_chunk,
                        score=float(best_sim),
                        dtw_distance=float(best_avg_cost),
                        dtw_sim=float(best_sim),
                        voiced_ratio_q=float(best_vrq),
                        voiced_ratio_r=float(best_vrr),
                    )
                )

        return results

    def scores(
        self, query: str | Path, references: list[str | Path], **kwargs
    ) -> list[float]:
        return [r.score for r in self.compare(query, references, **kwargs)]


if __name__ == "__main__":
    comparator = F0SimCompare(
        # good defaults; tweak if needed:
        unvoiced_cost_cents=600.0,
        sakoe_chiba_band=150,
        max_frames=2000,
        downsample_to=None,  # set e.g. 1200 if you want fixed-length comparison
        sim_sigma_cents=300.0,
    )

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
            [r.dtw_distance for r in results],
            [r.voiced_ratio_q for r in results],
            [r.voiced_ratio_r for r in results],
        )
    )
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Query: {query}")
    print("score    | avg_cost | vq   | vr   | q_chunk | r_chunk")
    print("-" * 140)
    for q_chunk, r_chunk, score, avg_cost, vq, vr in pairs[:50]:
        print(
            f"{score:0.6f}  {avg_cost:8.2f}  {vq:0.2f}  {vr:0.2f}  {q_chunk}  {r_chunk}"
        )
