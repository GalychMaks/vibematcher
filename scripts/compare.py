from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import librosa
import matplotlib.pyplot as plt

from .segment import segment_audio, Segment
from .features import extract_segment_features, SegmentFeatures
from .similarity import segment_similarity


def load_audio(path: str, sr: int, max_seconds: float | None) -> tuple[np.ndarray, int]:
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if max_seconds is not None and max_seconds > 0:
        y = y[: int(max_seconds * sr)]
    return y.astype(np.float32), sr


def compute_features_for_segments(
    y: np.ndarray, sr: int, segs: List[Segment]
) -> List[SegmentFeatures]:
    feats: List[SegmentFeatures] = []
    for seg in segs:
        feats.append(extract_segment_features(y, sr, seg.start, seg.end))
    return feats


def compare_all(
    q_feats: List[SegmentFeatures],
    r_feats: List[SegmentFeatures],
    *,
    topk: int = 5,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    S = np.zeros((len(q_feats), len(r_feats)), dtype=np.float32)
    matches: List[Dict[str, Any]] = []

    for i, A in enumerate(q_feats):
        for j, B in enumerate(r_feats):
            score, dbg = segment_similarity(A, B)
            S[i, j] = score
            matches.append(
                {
                    "i": i,
                    "j": j,
                    "score": float(score),
                    "q_start": A.start,
                    "q_end": A.end,
                    "r_start": B.start,
                    "r_end": B.end,
                    "dbg": dbg,
                }
            )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return S, matches[:topk]


def plagiarism_decision(best_score: float, threshold: float) -> str:
    return "LIKELY_PLAGIARISM" if best_score >= threshold else "unlikely"


def plot_heatmap(S: np.ndarray, out_png: str) -> None:
    plt.figure()
    plt.imshow(S, aspect="auto")
    plt.colorbar()
    plt.xlabel("Reference segments")
    plt.ylabel("Query segments")
    plt.title("Segment similarity heatmap")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query", required=True, help="Query song audio path (wav/mp3/etc.)"
    )
    ap.add_argument("--ref", required=True, help="Reference song audio path")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Analyze only the first N seconds",
    )
    ap.add_argument(
        "--seg",
        type=str,
        default="structure",
        choices=["uniform", "beats", "structure"],
    )
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="Decision threshold on best segment score",
    )
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--out-heatmap", type=str, default=None)
    args = ap.parse_args()

    qy, sr = load_audio(args.query, sr=args.sr, max_seconds=args.max_seconds)
    ry, _ = load_audio(args.ref, sr=args.sr, max_seconds=args.max_seconds)

    q_segs = segment_audio(qy, sr, mode=args.seg)
    r_segs = segment_audio(ry, sr, mode=args.seg)

    q_feats = compute_features_for_segments(qy, sr, q_segs)
    r_feats = compute_features_for_segments(ry, sr, r_segs)

    S, top = compare_all(q_feats, r_feats, topk=args.topk)
    best = top[0] if top else None

    if best is None:
        print("No matches (segments too short / no pitch).")
        return

    decision = plagiarism_decision(best["score"], args.threshold)

    print("\n=== BEST MATCH ===")
    print(
        f"Score: {best['score']:.4f}   Decision: {decision}   (threshold={args.threshold})"
    )
    print(f"Query:     {best['q_start']:.2f}s - {best['q_end']:.2f}s")
    print(f"Reference: {best['r_start']:.2f}s - {best['r_end']:.2f}s")
    print("Details:", json.dumps(best["dbg"], indent=2))

    print("\n=== TOP MATCHES ===")
    for k, m in enumerate(top, start=1):
        print(
            f"{k:02d}) {m['score']:.4f} | Q[{m['q_start']:.2f}-{m['q_end']:.2f}]  vs  R[{m['r_start']:.2f}-{m['r_end']:.2f}]"
        )

    report = {
        "query": args.query,
        "ref": args.ref,
        "sr": sr,
        "seg_mode": args.seg,
        "threshold": args.threshold,
        "decision": decision,
        "best_match": best,
        "top_matches": top,
        "similarity_matrix_shape": list(S.shape),
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report: {args.out_json}")

    if args.out_heatmap:
        plot_heatmap(S, args.out_heatmap)
        print(f"Saved heatmap: {args.out_heatmap}")


if __name__ == "__main__":
    main()
