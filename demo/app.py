from pathlib import Path

import gradio as gr
import pandas as pd

from vibematcher.compare import CompareMUQ


# -----------------------
# Hardcoded settings
# -----------------------
DATASET_DIR = Path("data/original")  # <-- change to your fingerprints root dir
MAX_K = 10


def list_original_wavs() -> list[Path]:
    return sorted(DATASET_DIR.glob("*.wav"))


# -----------------------
# Gradio callback
# -----------------------
def compare(audio_path: str, top_k: int) -> tuple[pd.DataFrame, ...]:
    if not audio_path:
        raise gr.Error("Upload an audio file.")

    original_files = list_original_wavs()
    if not original_files:
        raise gr.Error(f"No .wav files found in {DATASET_DIR}")

    results: list[tuple[Path, float]] = []
    for original_file in original_files:
        score = CompareMUQ.score(audio_path, original_file)
        results.append((original_file, float(score)))

    results.sort(key=lambda item: item[1], reverse=True)
    top_k = max(1, min(int(top_k), len(results)))
    selected = results[:top_k]

    df = pd.DataFrame(
        {
            "rank": list(range(1, top_k + 1)),
            "title": [path.stem for path, _ in selected],
            "path": [str(path) for path, _ in selected],
            "score": [score for _, score in selected],
        }
    )

    audio_updates: list[dict] = []
    for i in range(MAX_K):
        if i < top_k:
            path, score = selected[i]
            label = f"{i + 1}. {path.stem} — {score:.4f}"
            audio_updates.append(gr.update(value=str(path), label=label, visible=True))
        else:
            audio_updates.append(gr.update(value=None, label=None, visible=False))

    return (df, *audio_updates)


# -----------------------
# UI: input audio + button + output
# -----------------------
with gr.Blocks(title="VibeMatcher") as demo:
    gr.Markdown("## VibeMatcher — audio similarity (MuQ cosine)")

    inp = gr.Audio(sources=["upload"], type="filepath", label="Input audio")
    top_k = gr.Slider(
        minimum=1,
        maximum=MAX_K,
        step=1,
        value=min(5, MAX_K),
        label="Top-K",
    )
    run_btn = gr.Button("Find", variant="primary")

    out = gr.Dataframe(
        headers=["rank", "title", "path", "score"],
        datatype=["number", "str", "str", "number"],
        interactive=False,
    )

    audio_outputs: list[gr.Audio] = []
    with gr.Column():
        for i in range(MAX_K):
            audio_outputs.append(
                gr.Audio(
                    label=f"Result {i + 1}",
                    type="filepath",
                    interactive=False,
                    visible=False,
                )
            )

    run_btn.click(fn=compare, inputs=[inp, top_k], outputs=[out, *audio_outputs])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
