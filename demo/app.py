from pathlib import Path

import gradio as gr
import pandas as pd

# from vibematcher.features.melody_sim import MelodySimCompare as Comparator
from vibematcher.features.librosa_sim import LibrosaFeatureCompare as Comparator


# -----------------------
# Hardcoded settings
# -----------------------
DATASET_DIR = Path("data/original")  # <-- change to your wav root dir

_COMPARATOR: Comparator | None = None


def get_comparator() -> Comparator:
    global _COMPARATOR
    if _COMPARATOR is None:
        _COMPARATOR = Comparator()
    print(f"Loaded Comparator: {_COMPARATOR.__class__.__name__}")
    return _COMPARATOR


# -----------------------
# Gradio callbacks
# -----------------------
def compare(audio_path: str) -> tuple[pd.DataFrame, dict, str, str | None, str | None]:
    """
    Returns:
      - results df (item, melody_score)
      - cache dict (scores only)
      - details markdown (reset)
      - selected q audio (reset)
      - selected r audio (reset)
    """

    if not audio_path:
        raise gr.Warning("Upload an audio file.")

    comp = get_comparator()

    original_files = sorted(DATASET_DIR.glob("*.wav"))

    results = comp.compare(
        query=Path(audio_path),
        references=original_files,
    )

    q_chunks = [str(r.q_chunk_path) for r in results]
    r_chunks = [str(r.r_chunk_path) for r in results]
    scores = [float(r.score) for r in results]

    df = (
        pd.DataFrame(
            {
                "q": q_chunks,
                "r": r_chunks,
                "score": scores,
            }
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    cache = {"scores": {f"{q}|{r}": s for q, r, s in zip(q_chunks, r_chunks, scores)}}

    details_md = "### Details\nClick a row to load/play that audio."
    return df, cache, details_md, None, None


def show_details(
    df: pd.DataFrame,
    cache: dict,
    evt: gr.SelectData,
) -> tuple[str, str, str]:
    if df is None or len(df) == 0:
        raise gr.Warning("No results yet. Click Compare first.")

    row = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)
    q_chunk = str(df.iloc[row]["q"])
    r_chunk = str(df.iloc[row]["r"])
    score = float(df.iloc[row]["score"])

    md = (
        "### Details\n"
        f"**Q chunk:** `{q_chunk}`\n\n"
        f"**R chunk:** `{r_chunk}`\n\n"
        f"**MelodySim score:** `{score:.6f}`\n"
    )

    return md, q_chunk, r_chunk


# -----------------------
# UI
# -----------------------
with gr.Blocks(title="VibeMatcher — MelodySim") as demo:
    gr.Markdown("## VibeMatcher — MelodySim (Siamese over MERT embeddings)")

    cache_state = gr.State({})

    inp = gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Input audio",
    )

    run_btn = gr.Button("Compare", variant="primary")

    out = gr.Dataframe(
        headers=["q", "r", "score"],
        datatype=["str", "str", "number"],
        interactive=False,
        type="pandas",
        label="Results",
    )

    gr.Markdown("---")
    details_md = gr.Markdown("### Details\nRun Compare, then click a row.")
    selected_q_audio = gr.Audio(
        label="Selected q chunk",
        type="filepath",
        interactive=False,
    )
    selected_r_audio = gr.Audio(
        label="Selected r chunk",
        type="filepath",
        interactive=False,
    )

    run_btn.click(
        fn=compare,
        inputs=inp,
        outputs=[out, cache_state, details_md, selected_q_audio, selected_r_audio],
    )

    out.select(
        fn=show_details,
        inputs=[out, cache_state],
        outputs=[details_md, selected_q_audio, selected_r_audio],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
