from pathlib import Path
import gradio as gr
import pandas as pd
import numpy as np

from vibematcher.fingerprint import AudioFingerprint
from vibematcher.decisionmatrix import compute_similarity_matrix, aggregate_similarity_score, build_decision_matrix

# -----------------------
# Hardcoded settings
# -----------------------
DATASET_DIR = Path("data/original")  # root dir with your dataset audio files
WINDOW_LEN_SEC = 7  # chunk length in seconds
HOP_LEN_SEC = 3      # overlap between chunks in seconds

# -----------------------
# Helper: cache directory for embeddings
# -----------------------
def cache_dir_for_audio(audio_file: Path) -> Path:
    """
    Cache fingerprints/embeddings to: <audio_file.parent>/<audio_file.stem>/
    Example: data/original/Artist/song.wav -> data/original/Artist/song/
    """
    return audio_file.parent / audio_file.stem

# -----------------------
# Gradio callback
# -----------------------
def compare(audio_path: str) -> pd.DataFrame:
    if not audio_path:
        raise gr.Error("Upload an audio file.")

    # -----------------------
    # Load or compute query audio embeddings
    # -----------------------
    query_fp_dir = cache_dir_for_audio(Path(audio_path))
    if query_fp_dir.exists():
        query_fp = AudioFingerprint.load_from_dir(query_fp_dir)
    else:
        query_fp = AudioFingerprint.from_audio_file(
            audio_path, window_len_sec=WINDOW_LEN_SEC, hop_len_sec=HOP_LEN_SEC
        )
    query_fp_dir.mkdir(parents=True, exist_ok=True)
    query_fp.save_to_dir(query_fp_dir)

    items: list[str] = []
    longest_diagonals: list[float] = []
    mean_sims: list[float] = []

    # -----------------------
    # Iterate over dataset files
    # -----------------------
    dataset_files = sorted(DATASET_DIR.rglob("*.wav"))
    for dataset_file in dataset_files:
        dataset_file = Path(dataset_file)
        fp_dir = cache_dir_for_audio(dataset_file)

        # Load or compute dataset embeddings
        if fp_dir.exists():
            dataset_fp = AudioFingerprint.load_from_dir(fp_dir)
        else:
            print(f"Computing embeddings for: {dataset_file}")
            dataset_fp = AudioFingerprint.from_audio_file(
                dataset_file, window_len_sec=WINDOW_LEN_SEC, hop_len_sec=HOP_LEN_SEC
            )
            fp_dir.mkdir(parents=True, exist_ok=True)
            dataset_fp.save_to_dir(fp_dir)

        # Compute similarity matrix (chunk-wise)
        sim_matrix = compute_similarity_matrix(query_fp.mert_embeddings, dataset_fp.mert_embeddings)

        # Aggregate into a continuous similarity score [0-1]
        decision_matrix = build_decision_matrix(sim_matrix)
        longset_diagonal, mean_score = aggregate_similarity_score(decision_matrix, sim_matrix)

        items.append(str(dataset_file))
        longest_diagonals.append(float(longset_diagonal))
        mean_sims.append(float(mean_score))

    # -----------------------
    # Build result DataFrame
    # -----------------------
    df = pd.DataFrame({"item": items, "max_length": longest_diagonals, 'similarity': mean_sims})
    df = df.sort_values(['similarity', "max_length"], ascending=False).reset_index(drop=True)
    return df

# -----------------------
# UI: input audio + button + output
# -----------------------
with gr.Blocks(title="VibeMatcher") as demo:
    gr.Markdown("## VibeMatcher — audio similarity (MERT cosine)")

    inp = gr.Audio(sources=["upload"], type="filepath", label="Input audio")
    run_btn = gr.Button("Compare", variant="primary")

    out = gr.Dataframe(
        headers=["item", 'max_length', "similarity"],
        datatype=["str", 'number', "number"],
        interactive=False,
    )

    run_btn.click(fn=compare, inputs=inp, outputs=out)

# -----------------------
# Launch Gradio app
# -----------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
