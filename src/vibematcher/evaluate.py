import csv
from pathlib import Path
from typing import Callable, Iterable


def normalize_title(title: str) -> str:
    return title.strip()


def load_pairs(csv_path: Path) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            comp_title = normalize_title(row.get("comp_title", ""))
            ori_title = normalize_title(row.get("ori_title", ""))
            if not comp_title or not ori_title:
                continue
            pairs.setdefault(comp_title, set()).add(ori_title)
    return pairs


def list_wav_tracks(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.glob("*.wav"))]


def get_comparator(name: str) -> tuple[Callable[[Path, Path], float], bool]:
    name = name.lower()
    if name == "dtw":
        from vibematcher.compare_dtw import CompareDWT

        return CompareDWT.score, True
    if name == "wer":
        try:
            from vibematcher.compare_wer import CompareWER
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "WER comparator is unavailable (missing vibematcher.compare.wer_comparator)."
            ) from exc
        return CompareWER.score, False
    raise SystemExit(f"Unknown comparator '{name}'. Use 'dtw' or 'wer'.")


def score_all(
    query: Path,
    references: Iterable[Path],
    scorer: Callable[[Path, Path], float],
) -> list[tuple[Path, float]]:
    results: list[tuple[Path, float]] = []
    for reference in references:
        score = scorer(query, reference)
        results.append((reference, score))
    return results


def evaluate(
    comparison_dir: Path = Path("data/comparison"),
    original_dir: Path = Path("data/original"),
    pairs_csv_path: Path = Path("data/song_pairs.csv"),
    top_k: int = 1,
    comparator: str = "wer",
) -> int:
    pairs = load_pairs(pairs_csv_path)
    queries = list_wav_tracks(comparison_dir)
    references = list_wav_tracks(original_dir)

    if not queries:
        raise SystemExit(f"No .wav files found in {comparison_dir}")
    if not references:
        raise SystemExit(f"No .wav files found in {original_dir}")

    scorer, higher_is_better = get_comparator(comparator)

    hits = 0
    total = 0
    for query in queries:
        query_title = normalize_title(query.stem)
        results = score_all(query, references, scorer)
        results.sort(key=lambda item: item[1], reverse=higher_is_better)
        selected = results[: max(1, min(top_k, len(results)))]

        gt_set = pairs.get(query_title, set())
        hit = any(
            normalize_title(reference.stem) in gt_set for reference, _ in selected
        )

        total += 1
        hits += int(hit)

        print(f"Query: {query_title}")
        print(f"GT matches: {sorted(gt_set) if gt_set else '[]'}")
        print("Top-k:")
        for rank, (reference, score) in enumerate(selected, start=1):
            print(f"  {rank}. {normalize_title(reference.stem)} -> {score:.6f}")
        print(f"Hit: {hit}")
        print()

    accuracy = hits / total if total else 0.0
    print(f"Overall accuracy: {accuracy:.4f} ({hits}/{total})")
    return 0


if __name__ == "__main__":
    evaluate()
