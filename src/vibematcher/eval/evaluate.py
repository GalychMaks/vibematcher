from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from vibematcher.features.melody_sim import MelodySimCompare


# ==========================
# CONFIG (hardcoded)
# ==========================
GT_CSV = Path("data/song_pairs.csv")  # columns: ori_title, comp_title, relation
ORIGINAL_DIR = Path("data/original")  # originals / references
QUERY_DIR = Path("data/comparison")  # comparisons / queries
CKPT_PATH = Path("models/siamese_net_20250328.ckpt")

OUT_CSV = Path("runs/eval_preds.csv")

# Comparator params
DEVICE = None  # "cuda" / "cpu" / None(auto)
PROPORTION_THRES = 0.2
DECISION_THRES = 0.5
MIN_HITS = 1
FORCE_RECOMPUTE = False

# Label mapping (binary)
POSITIVE_RELATIONS = {"plag", "plag_doubt", "remake", "sample", "cover"}
NEGATIVE_RELATIONS = {"no_plag", "no", "none", "negative", "not_plag", "original"}

# What audio extensions to index (for strict name matching)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def relation_to_label(rel: str) -> int:
    r = (rel or "").strip().lower()
    if r in NEGATIVE_RELATIONS:
        return 0
    if r in POSITIVE_RELATIONS:
        return 1
    # fallback: assume positive unless explicitly negative
    return 1


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "n": float(n),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def list_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out


def build_strict_index(files: List[Path]) -> Dict[str, Path]:
    """
    Strict mapping: filename stem -> path.
    Requires GT titles to match file stems exactly.
    """
    idx: Dict[str, Path] = {}
    for f in files:
        # If duplicates exist, last one wins; tweak if you need collision handling.
        idx[f.stem] = f
    return idx


def main() -> None:
    # Checks
    for p in [GT_CSV, ORIGINAL_DIR, QUERY_DIR, CKPT_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    df = pd.read_csv(GT_CSV)
    required_cols = {"ori_title", "comp_title", "relation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"GT CSV missing columns: {sorted(missing)}")

    original_files = list_audio_files(ORIGINAL_DIR)
    query_files = list_audio_files(QUERY_DIR)
    if not original_files:
        raise ValueError(f"No audio files found under {ORIGINAL_DIR}")
    if not query_files:
        raise ValueError(f"No audio files found under {QUERY_DIR}")

    orig_idx = build_strict_index(original_files)
    query_idx = build_strict_index(query_files)

    comparator = MelodySimCompare(ckpt_path=CKPT_PATH, device=DEVICE)

    y_true: List[int] = []
    y_pred: List[int] = []
    rows_out: List[dict] = []

    # Evaluate all GT pairs
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating pairs"):
        ori_title = str(row["ori_title"])
        comp_title = str(row["comp_title"])
        relation = str(row["relation"])
        gt = relation_to_label(relation)

        ori_path = orig_idx.get(ori_title)
        comp_path = query_idx.get(comp_title)

        if ori_path is None or comp_path is None:
            # unresolved -> conservative negative
            pred_decision = 0
            pred_score = float("nan")
            best_q_chunk = ""
            best_r_chunk = ""
            status = "unresolved_path"

            y_true.append(gt)
            y_pred.append(pred_decision)

            rows_out.append(
                {
                    "ori_title": ori_title,
                    "comp_title": comp_title,
                    "relation": relation,
                    "gt_label": gt,
                    "ori_path": "" if ori_path is None else str(ori_path),
                    "comp_path": "" if comp_path is None else str(comp_path),
                    "pred_decision": pred_decision,
                    "pred_score": pred_score,
                    "best_q_chunk_path": best_q_chunk,
                    "best_r_chunk_path": best_r_chunk,
                    "status": status,
                }
            )
            continue

        # Compare comp(query) vs ori(reference)
        # New impl returns list[MelodySimChunkResult] where each item is the BEST q_chunk for a given r_chunk.
        results = comparator.compare(
            query=comp_path,
            references=[ori_path],
            proportion_thres=PROPORTION_THRES,
            decision_thres=DECISION_THRES,
            min_hits=MIN_HITS,
            force_recompute=FORCE_RECOMPUTE,
        )

        # Find the overall best match (highest score) across all r_chunks
        best = max(results, key=lambda r: r.score) if results else None

        if best is None:
            pred_decision = 0
            pred_score = float("nan")
            best_q_chunk = ""
            best_r_chunk = ""
            status = "no_results"
        else:
            pred_decision = int(best.decision)
            pred_score = float(best.score)
            best_q_chunk = str(best.q_chunk_path)
            best_r_chunk = str(best.r_chunk_path)
            status = "ok"

        y_true.append(gt)
        y_pred.append(pred_decision)

        rows_out.append(
            {
                "ori_title": ori_title,
                "comp_title": comp_title,
                "relation": relation,
                "gt_label": gt,
                "ori_path": str(ori_path),
                "comp_path": str(comp_path),
                "pred_decision": pred_decision,
                "pred_score": pred_score,
                "best_q_chunk_path": best_q_chunk,
                "best_r_chunk_path": best_r_chunk,
                "status": status,
            }
        )

    # Save predictions
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)

    # Metrics
    m = compute_metrics(y_true, y_pred)
    print("\n=== Metrics ===")
    print(f"N           : {int(m['n'])}")
    print(f"TP/TN/FP/FN : {int(m['tp'])}/{int(m['tn'])}/{int(m['fp'])}/{int(m['fn'])}")
    print(f"Accuracy    : {m['accuracy']:.4f}")
    print(f"Precision   : {m['precision']:.4f}")
    print(f"Recall      : {m['recall']:.4f}")
    print(f"F1          : {m['f1']:.4f}")
    print(f"\nSaved predictions to: {OUT_CSV}")


if __name__ == "__main__":
    main()
