import numpy as np

from dtw import dtw


def f0_hz_to_cents(f0_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    """
    Convert f0 in Hz to cents relative to ref_hz.
    Unvoiced frames (NaN or <=0) become NaN.
    """
    f0 = np.asarray(f0_hz, dtype=np.float32)
    cents = np.full_like(f0, np.nan, dtype=np.float32)

    voiced = np.isfinite(f0) & (f0 > 0.0)
    if np.any(voiced):
        cents[voiced] = 1200.0 * np.log2(f0[voiced] / float(ref_hz))

    return cents


def center_cents_by_median(cents: np.ndarray) -> np.ndarray:
    """
    Subtract median of voiced cents to remove global key offset.
    """
    x = np.asarray(cents, dtype=np.float32)
    voiced = np.isfinite(x)
    if not np.any(voiced):
        return x
    med = float(np.nanmedian(x[voiced]))
    return x - med


def compare_f0(q_f0: np.ndarray, c_f0: np.ndarray) -> float:
    q_cents = center_cents_by_median(f0_hz_to_cents(q_f0))
    c_cents = center_cents_by_median(f0_hz_to_cents(c_f0))

    q = q_cents[np.isfinite(q_cents)]
    c = c_cents[np.isfinite(c_cents)]
    if q.size < 2 or c.size < 2:
        return float("inf")

    # print(f"q: {q}")
    # print(f"c: {c}")

    aln = dtw(q.reshape(-1, 1), c.reshape(-1, 1))
    print(f"DTW alignment distance: {aln.distance}, steps: {len(aln.index1)}")
    avg_cost = aln.distance / max(1, len(aln.index1))  # “cost per step”
    print(f"Average cost per step: {avg_cost}")

    return float(avg_cost)
