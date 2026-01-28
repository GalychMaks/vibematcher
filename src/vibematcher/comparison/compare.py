from pathlib import Path

import numpy as np
import pandas as pd  # optional

from vibematcher.chunking.audio_chunks import AudioChunks
from vibematcher.chunking.song_former import SongFormer
from vibematcher.features.mert_embedder import MertEmbedder
from vibematcher.features.mert_embedding import MertEmbedding


def load_chunk_embeddings_matrix(
    audio_path: Path,
    *,
    song_former: SongFormer,
    mert_embedder: MertEmbedder,
    force_recompute: bool = False,
) -> np.ndarray:
    chunks: list[Path] = AudioChunks.from_audio_file(
        audio_path,
        song_former=song_former,
        force_recompute=force_recompute,
    ).chunks

    embeddings: list[np.ndarray] = []
    for chunk_path in chunks:
        emb: np.ndarray = MertEmbedding.from_audio_file(
            chunk_path,
            mert_embedder=mert_embedder,
            force_recompute=force_recompute,
        ).embedding

        embeddings.append(emb.astype(np.float32, copy=False))

    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)

    return np.stack(embeddings, axis=0).astype(np.float32, copy=False)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    a: (Q, D), b: (R, D) -> (Q, R)
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return (a_norm @ b_norm.T).astype(np.float32, copy=False)


def compare_songs(query_path: Path, ref_paths: list[Path]):
    """
    For each reference song:
      1) chunk query and ref using SongFormer (cached on disk)
      2) compute MERT embedding per chunk (cached on disk)
      3) compute cosine similarity for every (query_chunk, ref_chunk)
      4) take the maximum cosine similarity as the final score for that pair
      5) return a table of similarities for all refs
    """
    query_path = Path(query_path)
    ref_paths = [Path(p) for p in ref_paths]

    song_former = SongFormer()
    mert_embedder = MertEmbedder()

    q_embeddings = load_chunk_embeddings_matrix(
        query_path,
        song_former=song_former,
        mert_embedder=mert_embedder,
    )

    rows: list[dict] = []
    for ref_path in ref_paths:
        r_embeddings = load_chunk_embeddings_matrix(
            ref_path,
            song_former=song_former,
            mert_embedder=mert_embedder,
        )

        sim = cosine_sim_matrix(q_embeddings, r_embeddings)

        if sim.size == 0:
            best_sim = float("nan")
        else:
            flat_idx = int(np.argmax(sim))
            qi, ri = np.unravel_index(flat_idx, sim.shape)
            best_sim = float(sim[qi, ri])

        rows.append(
            {
                "reference": str(ref_path),
                "similarity_max_cosine": best_sim,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("similarity_max_cosine", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=Path,
        default=Path("data/comparison/_DANCE_ 싸이 _PSY_ - 챔피언.wav"),
    )
    parser.add_argument(
        "--refs-dir",
        type=Path,
        default=Path("data/original"),
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/similarities.csv"))
    args = parser.parse_args()

    ref_paths = sorted(args.refs_dir.glob("*.wav"))
    df = compare_songs(args.query, ref_paths)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {args.out}")
