from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np


AggregationMode = Literal["best", "mean", "topk_bestmean"]


@dataclass(frozen=True)
class FingerprintComparisonResult:
    """
    Example:
      overall = 0.9
      aspects = {"mert": 0.9, "f0": 0.85}
      correlation_matrices = {
        "mert": (Q, C) cosine-sim matrix,
        "f0_dtw_acc_cost": (Tq, Tc) accumulated DTW cost (lower = better),
        "f0_dtw_path": (L, 2) int32 path indices (i, j),
      }
    """

    overall: float
    aspects: Dict[str, float]
    correlation_matrices: Dict[str, np.ndarray]

    query_chunk_starts_sec: Optional[np.ndarray] = None  # (Q,)
    cand_chunk_starts_sec: Optional[np.ndarray] = None  # (C,)
