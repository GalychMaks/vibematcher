from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# from vibematcher.features.melody_sim import MelodySimCompare as Comparator
from vibematcher.features.librosa_sim import LibrosaFeatureCompare as Comparator

# ==========================
# CONFIG (hardcoded)
# ==========================
GT_CSV = Path("data/song_pairs.csv")  # columns: ori_title, comp_title, relation
ORIGINAL_DIR = Path("data/original")  # originals / references
QUERY_DIR = Path("data/comparison")  # comparisons / queries

OUT_CSV = Path("artifacts/eval_preds_librosa.csv")

# Label mapping (binary)
POSITIVE_RELATIONS = {"plag", "plag_doubt", "remake", "sample", "cover"}
NEGATIVE_RELATIONS = {"no_plag", "no", "none", "negative", "not_plag", "original"}

# What audio extensions to index (for strict name matching)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
SCORE_THRES = 0.60  # tune this


_COMPARATOR: Comparator | None = None


def get_comparator() -> Comparator:
    global _COMPARATOR
    if _COMPARATOR is None:
        _COMPARATOR = Comparator()
    print(f"Loaded Comparator: {_COMPARATOR.__class__.__name__}")
    return _COMPARATOR


def relation_to_label(rel: str) -> int:
    r = (rel or "").strip().lower()
    if r in NEGATIVE_RELATIONS:
        return 0
    if r in POSITIVE_RELATIONS:
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


def pick_best_with_threshold(results, score_thres: float):
    """
    results: list[MelodySimChunkResult]
    Selection rule:
      - if any score >= thres: pick max among those
      - else: pick global max score
    """
    if not results:
        return None, None  # best_any, best_over_thres

    best_any = max(results, key=lambda r: r.score)
    over = [r for r in results if float(r.score) >= score_thres]
    best_over = max(over, key=lambda r: r.score) if over else None
    return best_any, best_over


def main() -> None:
    # Checks
    for p in [GT_CSV, ORIGINAL_DIR, QUERY_DIR]:
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

    comparator = get_comparator()

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
        )

        best_any, best_over = pick_best_with_threshold(results, SCORE_THRES)

        if best_any is None:
            best_score_any = float("nan")
            status = "no_results"
            best_q_chunk = ""
            best_r_chunk = ""
        else:
            best_score_any = float(best_any.score)

            # prediction by threshold on the GLOBAL best score
            pred_decision = 1 if best_score_any >= SCORE_THRES else 0
            pred_score = best_score_any
            status = "ok"

            # selection rule for "best chunk"
            if best_over is not None:
                selected = best_over
            else:
                selected = best_any

            best_q_chunk = str(selected.q_chunk_path)
            best_r_chunk = str(selected.r_chunk_path)

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
