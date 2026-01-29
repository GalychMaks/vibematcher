from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

# from vibematcher.features.melody_sim import MelodySimCompare as Comparator
from vibematcher.features.librosa_sim import LibrosaSimCompare as Comparator

# ==========================
# CONFIG
# ==========================
GT_CSV = Path("data/song_pairs.csv")  # columns: ori_title, comp_title, relation
ORIGINAL_DIR = Path("data/original")  # originals / references
QUERY_DIR = Path("data/comparison")  # comparisons / queries
OUT_CSV = Path("artifacts/retrieval_eval_librosa.csv")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# Label mapping (binary "should match?" according to relation)
POSITIVE_RELATIONS = {"plag", "plag_doubt", "remake", "sample", "cover"}
NEGATIVE_RELATIONS = {"no_plag", "no", "none", "negative", "not_plag", "original"}

# Threshold used only for "match vs no-match" metrics (not for Top-1 retrieval)
SCORE_THRES = 0.60

# Unknown relation handling:
# - "skip": do not use those rows as GT
# - "neg": treat unknown as negative
UNKNOWN_RELATION_POLICY = "skip"


def list_audio_files(root: Path) -> List[Path]:
    return [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_relation(rel) -> str:
    if pd.isna(rel):
        return ""
    return str(rel).strip().lower()


def relation_to_match_label(rel) -> Tuple[Optional[int], bool]:
    """
    Returns (label, is_unknown).
      label=1 => should match (positive)
      label=0 => should NOT match (negative / no-match)
      label=None => skipped (if policy says skip unknown)
    """
    r = normalize_relation(rel)
    if r in POSITIVE_RELATIONS:
        return 1, False
    if r in NEGATIVE_RELATIONS:
        return 0, False

    # unknown
    if UNKNOWN_RELATION_POLICY == "skip":
        return None, True
    if UNKNOWN_RELATION_POLICY == "neg":
        return 0, True
    return None, True


def build_stem_index(
    files: List[Path],
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """
    Map filename stem -> chosen path.
    Returns (index, collisions). Collision stems have >1 file.
    Deterministic choice: prefer .wav, then shorter path, then lexicographic.
    """
    buckets: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        buckets[f.stem].append(f)

    collisions = {stem: paths for stem, paths in buckets.items() if len(paths) > 1}

    def sort_key(p: Path):
        return (0 if p.suffix.lower() == ".wav" else 1, len(str(p)), str(p))

    idx = {stem: sorted(paths, key=sort_key)[0] for stem, paths in buckets.items()}
    return idx, collisions


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
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


def pick_best_original_from_results(
    results,
    originals_by_stem: Dict[str, Path],
) -> Tuple[Optional[str], float]:
    """
    Convert chunk-level results -> best original stem.

    We try to infer which original a result belongs to by checking whether any
    original stem appears in r_chunk_path or its parents.

    Returns (best_original_stem, best_score). If cannot infer, returns (None, nan).
    """
    if not results:
        return None, float("nan")

    # Precompute stems sorted by length descending to avoid partial matches
    stems = sorted(originals_by_stem.keys(), key=len, reverse=True)

    per_stem_best: Dict[str, float] = {}
    for r in results:
        score = float(getattr(r, "score", float("nan")))
        r_chunk = str(getattr(r, "r_chunk_path", ""))

        hit_stem = None
        # heuristic: stem as substring in r_chunk_path
        for s in stems:
            if s and s in r_chunk:
                hit_stem = s
                break

        if hit_stem is None:
            continue

        prev = per_stem_best.get(hit_stem)
        if prev is None or score > prev:
            per_stem_best[hit_stem] = score

    if not per_stem_best:
        # fallback: we can’t map chunks back to originals
        # return the global best score but unknown predicted original
        best_any = max(results, key=lambda x: float(getattr(x, "score", -1e9)))
        return None, float(getattr(best_any, "score", float("nan")))

    best_stem = max(per_stem_best.items(), key=lambda kv: kv[1])[0]
    return best_stem, per_stem_best[best_stem]


def main() -> None:
    # Checks
    for p in [GT_CSV, ORIGINAL_DIR, QUERY_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    # Load GT
    df = pd.read_csv(GT_CSV)
    required_cols = {"ori_title", "comp_title", "relation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"GT CSV missing columns: {sorted(missing)}")

    # Build GT mapping: comp_title -> (ori_title, match_label)
    # If multiple rows per comp_title, keep the first positive if any, else first row.
    gt_rows_by_comp: Dict[str, List[Tuple[str, Optional[int], str]]] = defaultdict(list)

    unknown_rel_counter = Counter()
    for _, row in df.iterrows():
        comp = normalize_text(row["comp_title"])
        ori = normalize_text(row["ori_title"])
        rel = row["relation"]
        match_label, is_unknown = relation_to_match_label(rel)
        rel_norm = normalize_relation(rel)

        if is_unknown:
            unknown_rel_counter[rel_norm] += 1

        gt_rows_by_comp[comp].append((ori, match_label, rel_norm))

    if unknown_rel_counter:
        print("\n=== WARNING: Unknown relation values in CSV ===")
        for k, v in unknown_rel_counter.most_common():
            print(f"  {k!r}: {v} row(s)")
        print("UNKNOWN_RELATION_POLICY =", UNKNOWN_RELATION_POLICY)

    def choose_gt_for_comp(
        comp: str,
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        rows = gt_rows_by_comp.get(comp, [])
        if not rows:
            return None, None, None

        # Prefer a positive row if any exists
        for ori, lbl, reln in rows:
            if lbl == 1:
                return ori, lbl, reln

        # else take first row (could be negative or None)
        ori, lbl, reln = rows[0]
        return ori, lbl, reln

    # Index audio files
    orig_files = list_audio_files(ORIGINAL_DIR)
    qry_files = list_audio_files(QUERY_DIR)
    if not orig_files:
        raise ValueError(f"No audio files found under {ORIGINAL_DIR}")
    if not qry_files:
        raise ValueError(f"No audio files found under {QUERY_DIR}")

    orig_idx, orig_collisions = build_stem_index(orig_files)
    qry_idx, qry_collisions = build_stem_index(qry_files)

    if orig_collisions:
        print("\n=== WARNING: Duplicate original stems (collisions) ===")
        for stem, paths in list(orig_collisions.items())[:20]:
            print(f"  {stem}: {len(paths)} files")
        print("Deterministically choosing one file per stem (prefers .wav).")

    if qry_collisions:
        print("\n=== WARNING: Duplicate query stems (collisions) ===")
        for stem, paths in list(qry_collisions.items())[:20]:
            print(f"  {stem}: {len(paths)} files")
        print("Deterministically choosing one file per stem (prefers .wav).")

    # Prepare comparator
    comparator = Comparator()

    # All originals list for retrieval
    original_stems = list(orig_idx.keys())
    original_paths = [orig_idx[s] for s in original_stems]

    # Evaluate retrieval for each query file
    rows_out: List[dict] = []

    top1_total = 0
    top1_correct = 0

    # Binary match metrics (optional): only counted where GT match_label is 0/1 (not None)
    y_true_match: List[int] = []
    y_pred_match: List[int] = []

    missing_gt = 0
    skipped_unknown = 0
    unmapped_pred = 0

    for q_stem, q_path in tqdm(
        qry_idx.items(), total=len(qry_idx), desc="Retrieval eval"
    ):
        gt_ori_stem, gt_match_label, gt_rel = choose_gt_for_comp(q_stem)

        if gt_ori_stem is None:
            missing_gt += 1
            status = "no_gt_row_for_query"
        elif gt_match_label is None:
            skipped_unknown += 1
            status = "gt_relation_unknown_skipped"
        else:
            status = "ok"

        # Run retrieval: compare query to ALL originals
        results = comparator.compare(
            query=q_path,
            references=original_paths,
        )

        pred_ori_stem, pred_score = pick_best_original_from_results(results, orig_idx)

        if pred_ori_stem is None:
            unmapped_pred += 1

        # Top-1 retrieval correctness (only meaningful if we have a GT original and a “should match” label=1)
        correct_top1 = False
        if gt_ori_stem is not None and gt_match_label == 1:
            top1_total += 1
            if pred_ori_stem == gt_ori_stem:
                top1_correct += 1
                correct_top1 = True

        # Binary “match vs no-match” prediction using threshold
        pred_match = (
            1 if (pd.notna(pred_score) and float(pred_score) >= SCORE_THRES) else 0
        )
        if gt_match_label in (0, 1):
            y_true_match.append(int(gt_match_label))
            y_pred_match.append(int(pred_match))

        rows_out.append(
            {
                "query_title": q_stem,
                "query_path": str(q_path),
                "gt_ori_title": "" if gt_ori_stem is None else gt_ori_stem,
                "gt_match_label": "" if gt_match_label is None else int(gt_match_label),
                "gt_relation_norm": "" if gt_rel is None else gt_rel,
                "pred_ori_title": "" if pred_ori_stem is None else pred_ori_stem,
                "pred_score": float(pred_score),
                "correct_top1": bool(correct_top1),
                "pred_match_by_threshold": int(pred_match),
                "status": status
                if pred_ori_stem is not None
                else (status + "|pred_unmapped"),
            }
        )

    # Save detailed results
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT_CSV, index=False)

    # Report
    print("\n=== Retrieval (Top-1) ===")
    if top1_total:
        print(
            f"Top-1 accuracy on positive GT queries: {top1_correct}/{top1_total} = {top1_correct / top1_total:.4f}"
        )
    else:
        print("Top-1 accuracy: no positive GT queries found (gt_match_label==1).")

    print("\n=== Binary match metrics (thresholded) ===")
    if y_true_match:
        m = compute_binary_metrics(y_true_match, y_pred_match)
        print(f"N           : {int(m['n'])}")
        print(
            f"TP/TN/FP/FN : {int(m['tp'])}/{int(m['tn'])}/{int(m['fp'])}/{int(m['fn'])}"
        )
        print(f"Accuracy    : {m['accuracy']:.4f}")
        print(f"Precision   : {m['precision']:.4f}")
        print(f"Recall      : {m['recall']:.4f}")
        print(f"F1          : {m['f1']:.4f}")
    else:
        print("No rows with gt_match_label in {0,1} (all skipped/unknown).")

    print("\n=== Diagnostics ===")
    print(f"Queries total            : {len(qry_idx)}")
    print(f"Missing GT rows          : {missing_gt}")
    print(f"Skipped unknown GT label : {skipped_unknown}")
    print(f"Pred unmapped to original: {unmapped_pred}")
    print(f"\nSaved retrieval results to: {OUT_CSV}")


if __name__ == "__main__":
    main()
