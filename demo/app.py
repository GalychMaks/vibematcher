from pathlib import Path

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from vibematcher.compare.compare import compare_fingerprints
from vibematcher.fingerprint.fingerprint import AudioFingerprint


# -----------------------
# Hardcoded settings
# -----------------------
DATASET_DIR = Path("data/original")  # <-- change to your fingerprints root dir


def cache_dir_for_audio(audio_file: Path) -> Path:
    """
    Cache fingerprints to: <audio_file.parent>/<audio_file.stem>/
    Example: data/original/Artist/song.wav -> data/original/Artist/song/
    """
    return audio_file.parent / audio_file.stem


def ensure_fp_for_file(audio_file: Path) -> AudioFingerprint:
    audio_file = Path(audio_file)
    fp_dir = cache_dir_for_audio(audio_file)
    if fp_dir.exists():
        return AudioFingerprint.load_from_dir(fp_dir)

    print(f"Computing fingerprint for: {audio_file}")
    fp = AudioFingerprint.from_audio_file(audio_file)
    fp_dir.mkdir(parents=True, exist_ok=True)
    fp.save_to_dir(fp_dir)
    return fp


def make_heatmap_fig(
    sim: np.ndarray,
    q_starts: np.ndarray | None = None,
    c_starts: np.ndarray | None = None,
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if (
        q_starts is not None
        and c_starts is not None
        and len(q_starts) > 0
        and len(c_starts) > 0
    ):
        extent = [
            float(c_starts[0]),
            float(c_starts[-1]),
            float(q_starts[0]),
            float(q_starts[-1]),
        ]
        im = ax.imshow(
            sim,
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_xlabel("candidate time (sec)")
        ax.set_ylabel("query time (sec)")
    else:
        im = ax.imshow(sim, aspect="auto", origin="lower", vmin=-1.0, vmax=1.0)
        ax.set_xlabel("candidate chunk index")
        ax.set_ylabel("query chunk index")

    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    return fig


# -----------------------
# Gradio callbacks
# -----------------------
def compare(
    audio_path: str,
) -> tuple[pd.DataFrame, AudioFingerprint, str, plt.Figure | None]:
    """
    Returns:
      - results df
      - query fp state
      - details markdown (reset/placeholder)
      - heatmap figure (reset)
    """
    if not audio_path:
        raise gr.Error("Upload an audio file.")

    q = AudioFingerprint.from_audio_file(audio_path)

    items: list[str] = []
    similarities: list[float] = []

    original_files = sorted(DATASET_DIR.rglob("*.wav"))
    for original_file in original_files:
        original_fp = ensure_fp_for_file(Path(original_file))
        r = compare_fingerprints(q, original_fp)
        items.append(str(original_file))
        similarities.append(float(r.overall))

    df = pd.DataFrame({"item": items, "similarity": similarities})
    df = df.sort_values("similarity", ascending=False).reset_index(drop=True)

    details_md = "### Details\nSelect a row in the table above to see details."
    return df, q, details_md, None


def show_details(
    df: pd.DataFrame,
    q: AudioFingerprint,
    evt: gr.SelectData,
) -> tuple[str, plt.Figure]:
    if df is None or len(df) == 0:
        raise gr.Error("No results yet. Click Compare first.")
    if q is None:
        raise gr.Error("No query fingerprint yet. Click Compare first.")

    # evt.index is (row, col) for Dataframe
    row = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)

    item = str(df.iloc[row]["item"])
    cand_fp = ensure_fp_for_file(Path(item))

    r = compare_fingerprints(q, cand_fp)

    mert_mat = r.correlation_matrices.get("mert")
    if mert_mat is None:
        raise gr.Error(
            "compare_fingerprints did not return correlation_matrices['mert']."
        )

    fig = make_heatmap_fig(
        mert_mat,
        r.query_chunk_starts_sec,
        r.cand_chunk_starts_sec,
    )

    md = (
        "### Details\n"
        f"**Item:** `{item}`\n\n"
        f"**Similarity (overall):** `{float(r.overall):.4f}`\n\n"
        f"**Aspects:** `{ {k: float(v) for k, v in r.aspects.items()} }`\n\n"
        f"**Matrix shape (mert):** `{tuple(mert_mat.shape)}`"
    )

    return md, fig


# -----------------------
# UI
# -----------------------
with gr.Blocks(title="VibeMatcher") as demo:
    gr.Markdown("## VibeMatcher — audio similarity (MERT cosine, chunked)")

    q_state = gr.State(None)

    inp = gr.Audio(sources=["upload"], type="filepath", label="Input audio")
    run_btn = gr.Button("Compare", variant="primary")

    out = gr.Dataframe(
        headers=["item", "similarity"],
        datatype=["str", "number"],
        interactive=False,
        type="pandas",
        label="Results",
    )

    gr.Markdown("---")
    details_md = gr.Markdown("### Details\nRun Compare, then click a row.")
    heat = gr.Plot(label="Chunk similarity heatmap")

    # Compare fills table + stores query fp + resets details section
    run_btn.click(fn=compare, inputs=inp, outputs=[out, q_state, details_md, heat])

    # Selecting a row updates details section under the table
    out.select(fn=show_details, inputs=[out, q_state], outputs=[details_md, heat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
