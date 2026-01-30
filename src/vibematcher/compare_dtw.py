from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from vibematcher.f0.f0_contour_cache import F0Contour
from vibematcher.stems.demucs import DemucsStemsSeparator
from vibematcher.stems.stems_cache import Stems


def f0_to_cents(f0_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
    cents = np.full_like(f0, np.nan, dtype=np.float32)
    m = np.isfinite(f0) & (f0 > 0.0)
    if np.any(m):
        cents[m] = 1200.0 * np.log2(f0[m] / float(ref_hz))
    return cents


def fill_nans(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=np.float32)
    fill = np.nanmedian(x[m]).astype(np.float32)
    return np.where(np.isfinite(x), x, fill).astype(np.float32)


def dtw_aligned_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_fill = fill_nans(x)
    y_fill = fill_nans(y)
    D = np.abs(x_fill[:, None] - y_fill[None, :]).astype(np.float32)
    _, wp = librosa.sequence.dtw(C=D)
    wp = wp[::-1]
    xv = x[wp[:, 0]]
    yv = y[wp[:, 1]]
    m = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[m]
    yv = yv[m]
    if xv.size < 3 or yv.size < 3:
        return float("nan")
    if float(np.std(xv)) == 0.0 or float(np.std(yv)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


@dataclass
class CompareDWT:
    stems_separator = DemucsStemsSeparator()

    @classmethod
    def score(cls, query_path: Path, reference_path: Path) -> float:
        q_stems = Stems.from_audio_file(
            query_path,
            stems_separator=cls.stems_separator,
        ).stems
        r_stems = Stems.from_audio_file(
            reference_path,
            stems_separator=cls.stems_separator,
        ).stems

        best = float("nan")
        for q in q_stems:
            q_cents = f0_to_cents(F0Contour.from_audio_file(q).f0_hz)
            for r in r_stems:
                r_cents = f0_to_cents(F0Contour.from_audio_file(r).f0_hz)

                score = dtw_aligned_corr(q_cents, r_cents)
                if np.isfinite(score) and (not np.isfinite(best) or score > best):
                    best = score

        return best


if __name__ == "__main__":
    # Example usage: compare two full songs by stems
    query = "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav"
    references = [Path(path) for path in Path("data/original").glob("*.wav")]

    # Create once and reuse to avoid reloading demucs model repeatedly
    compare_dwt = CompareDWT()

    for reference in references:
        score = compare_dwt.score(query, reference)

        print(reference)
        print(score)
        print()
