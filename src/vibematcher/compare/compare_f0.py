from typing import Optional

import numpy as np


def downsample_1d(x: np.ndarray, max_len: Optional[int]) -> np.ndarray:
    x = np.asarray(x)
    if max_len is None or x.size <= max_len:
        return x
    max_len = int(max_len)
    if max_len <= 0:
        return x[:0]
    idx = np.linspace(0, x.size - 1, num=max_len, dtype=np.int64)
    return x[idx]


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


def _f0_frame_cost_cents(
    q_cents: float, c_cents: float, *, unvoiced_cost_cents: float
) -> float:
    """
    Cost between two *cents* frames:
      - both unvoiced (NaN): 0
      - one voiced, one unvoiced: unvoiced_cost_cents
      - both voiced: abs diff in cents
    """
    q_voiced = np.isfinite(q_cents)
    c_voiced = np.isfinite(c_cents)

    if (not q_voiced) and (not c_voiced):
        return 0.0
    if q_voiced != c_voiced:
        return float(unvoiced_cost_cents)

    return float(abs(q_cents - c_cents))


def dtw_acc_cost_and_path(
    q_f0: np.ndarray,
    c_f0: np.ndarray,
    *,
    unvoiced_cost_cents: float = 600.0,
    band: Optional[int] = None,  # Sakoe–Chiba band in frames
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic DTW with 3-way steps (diag/up/left).
    Inputs are expected to be *cents* arrays (NaN = unvoiced).
    Returns:
      acc: (Tq, Tc) accumulated cost (float32)
      path: (L, 2) int32 indices (i, j) from start to end
    """
    q = np.asarray(q_f0, dtype=np.float32)
    c = np.asarray(c_f0, dtype=np.float32)
    Tq, Tc = int(q.size), int(c.size)

    if Tq == 0 or Tc == 0:
        acc = np.empty((Tq, Tc), dtype=np.float32)
        path = np.empty((0, 2), dtype=np.int32)
        return acc, path

    inf = np.float32(np.inf)
    acc = np.full((Tq, Tc), inf, dtype=np.float32)

    # DP fill
    for i in range(Tq):
        j0, j1 = 0, Tc
        if band is not None:
            b = int(band)
            j0 = max(0, i - b)
            j1 = min(Tc, i + b + 1)

        for j in range(j0, j1):
            cost = np.float32(
                _f0_frame_cost_cents(
                    q[i], c[j], unvoiced_cost_cents=unvoiced_cost_cents
                )
            )

            if i == 0 and j == 0:
                acc[i, j] = cost
            else:
                best_prev = inf
                if i > 0:
                    best_prev = min(best_prev, acc[i - 1, j])
                if j > 0:
                    best_prev = min(best_prev, acc[i, j - 1])
                if i > 0 and j > 0:
                    best_prev = min(best_prev, acc[i - 1, j - 1])

                acc[i, j] = cost + best_prev

    # Backtrack path from (Tq-1, Tc-1)
    i, j = Tq - 1, Tc - 1
    if not np.isfinite(acc[i, j]):
        # No valid path under the band constraint
        return acc, np.empty((0, 2), dtype=np.int32)

    path_rev: list[tuple[int, int]] = [(i, j)]
    while i > 0 or j > 0:
        candidates: list[tuple[float, int, int]] = []

        if i > 0 and np.isfinite(acc[i - 1, j]):
            candidates.append((float(acc[i - 1, j]), i - 1, j))
        if j > 0 and np.isfinite(acc[i, j - 1]):
            candidates.append((float(acc[i, j - 1]), i, j - 1))
        if i > 0 and j > 0 and np.isfinite(acc[i - 1, j - 1]):
            candidates.append((float(acc[i - 1, j - 1]), i - 1, j - 1))

        if not candidates:
            break

        _, i, j = min(candidates, key=lambda t: t[0])
        path_rev.append((i, j))

    path = np.asarray(path_rev[::-1], dtype=np.int32)
    return acc, path


def f0_dtw_similarity(
    q_cents: np.ndarray,
    c_cents: np.ndarray,
    *,
    unvoiced_cost_cents: float = 600.0,
    band: Optional[int] = None,
    scale_cents: float = 150.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Inputs are expected to be *cents* arrays (NaN = unvoiced), ideally median-centered.
    Returns:
      sim: scalar in (0, 1], higher = more similar
      acc: accumulated DTW cost matrix
      path: DTW path (L, 2)
    """
    acc, path = dtw_acc_cost_and_path(
        q_cents, c_cents, unvoiced_cost_cents=unvoiced_cost_cents, band=band
    )
    if acc.size == 0 or path.size == 0:
        return 0.0, acc, path

    # normalize by path length -> "avg cents per step"
    avg_cost = float(acc[-1, -1]) / max(1, int(path.shape[0]))
    scale = max(1e-6, float(scale_cents))
    sim = float(np.exp(-avg_cost / scale))  # 1.0 when identical, decays with avg cost
    return sim, acc, path
