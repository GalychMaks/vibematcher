from dataclasses import dataclass
from typing import Dict

from vibematcher.fingerprint import AudioFingerprint
from vibematcher.utils import cosine_similarity


@dataclass(frozen=True)
class FingerprintComparisonResult:
    """Example
    overall = 0.9
    aspects = {
        "mert": 0.9,
    }
    """

    overall: float
    aspects: Dict[str, float]


def compare_fingerprints(
    query: AudioFingerprint,
    candidate: AudioFingerprint,
) -> FingerprintComparisonResult:
    mert_sim = cosine_similarity(query.mert_embedding, candidate.mert_embedding)

    return FingerprintComparisonResult(
        overall=mert_sim,
        aspects={"mert": mert_sim},
    )
