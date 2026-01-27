from pathlib import Path

import gradio as gr
import pandas as pd

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


# -----------------------
# Gradio callback
# -----------------------
def compare(audio_path: str) -> pd.DataFrame:
    if not audio_path:
        raise gr.Error("Upload an audio file.")

    q = AudioFingerprint.from_audio_file(audio_path)

    items: list[str] = []
    similarities: list[float] = []

    # NOTE: rglob returns a generator; we want a stable list for reuse + display.
    original_files = sorted(DATASET_DIR.rglob("*.wav"))

    for original_file in original_files:
        original_file = Path(original_file)
        fp_dir = cache_dir_for_audio(original_file)

        if fp_dir.exists():
            original_fp = AudioFingerprint.load_from_dir(fp_dir)
        else:
            print(f"Computing fingerprint for: {original_file}")
            original_fp = AudioFingerprint.from_audio_file(original_file)
            fp_dir.mkdir(parents=True, exist_ok=True)
            original_fp.save_to_dir(fp_dir)

        r = compare_fingerprints(q, original_fp)
        items.append(str(original_file))
        similarities.append(float(r.overall))

    df = pd.DataFrame({"item": items, "similarity": similarities})
    df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
    return df


# -----------------------
# UI: input audio + button + output
# -----------------------
with gr.Blocks(title="VibeMatcher") as demo:
    gr.Markdown("## VibeMatcher — audio similarity (MERT cosine)")

    inp = gr.Audio(sources=["upload"], type="filepath", label="Input audio")
    run_btn = gr.Button("Compare", variant="primary")

    out = gr.Dataframe(
        headers=["item", "similarity"],
        datatype=["str", "number"],
        interactive=False,
    )

    run_btn.click(fn=compare, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
