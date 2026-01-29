from __future__ import annotations

from pathlib import Path
import re

import gradio as gr
import pandas as pd

# from vibematcher.features.melody_sim import MelodySimCompare as Comparator
from vibematcher.features.librosa_sim import LibrosaSimCompare as Comparator


# -----------------------
# Hardcoded settings
# -----------------------
DATASET_DIR = Path("data/original")  # <-- change to your wav root dir
PAIRS_CSV = Path("data/song_pairs.csv")

_COMPARATOR: Comparator | None = None
_PAIRS_INDEX: dict[str, set[str]] | None = (
    None  # norm_title -> set(norm_partner_titles)
)


def get_comparator() -> Comparator:
    global _COMPARATOR
    if _COMPARATOR is None:
        _COMPARATOR = Comparator()
    print(f"Loaded Comparator: {_COMPARATOR.__class__.__name__}")
    return _COMPARATOR


# -----------------------
# Pair / title helpers
# -----------------------
_ws_re = re.compile(r"\s+")
_nonword_re = re.compile(r"[^a-z0-9]+")


def norm_title(s: str) -> str:
    s = s.strip().lower().replace("_", " ")
    s = _ws_re.sub(" ", s)
    s = _nonword_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s


def song_from_chunk_path(p: str) -> str:
    return Path(p).parent.name


def load_pairs_index() -> dict[str, set[str]]:
    """
    Build a lookup:
      norm(title) -> set(norm(other_title))
    Includes both directions (ori <-> comp).
    """
    global _PAIRS_INDEX
    if _PAIRS_INDEX is not None:
        return _PAIRS_INDEX

    if not PAIRS_CSV.exists():
        print(f"[WARN] Pairs CSV not found: {PAIRS_CSV.resolve()}")
        _PAIRS_INDEX = {}
        return _PAIRS_INDEX

    df = pd.read_csv(PAIRS_CSV)

    idx: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        a = str(row.get("ori_title", "")).strip()
        b = str(row.get("comp_title", "")).strip()
        if not a or not b:
            continue

        na = norm_title(a)
        nb = norm_title(b)

        idx.setdefault(na, set()).add(nb)
        idx.setdefault(nb, set()).add(na)  # reverse direction too

    _PAIRS_INDEX = idx
    print(f"Loaded pairs from CSV: {PAIRS_CSV} ({len(idx)} titles indexed)")
    return _PAIRS_INDEX


def expected_partners_norm(title: str) -> set[str]:
    idx = load_pairs_index()
    return idx.get(norm_title(title), set())


# -----------------------
# Gradio callbacks
# -----------------------
def compare(audio_path: str) -> tuple[pd.DataFrame, dict, str, str | None, str | None]:
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

    q_song = song_from_chunk_path(q_chunks[0]) if q_chunks else ""
    expected_norms = expected_partners_norm(q_song)

    r_songs = [song_from_chunk_path(p) for p in r_chunks]
    r_norms = [norm_title(t) for t in r_songs]
    gt = [1 if rn in expected_norms else 0 for rn in r_norms]

    df = (
        pd.DataFrame(
            {
                "gt": gt,
                "q_song": [q_song] * len(scores),
                "r_song": r_songs,
                "score": scores,
                "q": q_chunks,
                "r": r_chunks,
            }
        )
        .sort_values(["gt", "score"], ascending=[False, False])
        .reset_index(drop=True)
    )

    cache = {"q_song": q_song}

    details_md = (
        "### Details\n"
        f"**Query song:** `{q_song}`\n\n"
        "Click a row to load/play that audio."
    )
    return df, cache, details_md, None, None


def show_details(
    df: pd.DataFrame, cache: dict, evt: gr.SelectData
) -> tuple[str, str, str]:
    if df is None or len(df) == 0:
        raise gr.Warning("No results yet. Click Compare first.")

    row = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)

    q_chunk = str(df.iloc[row]["q"])
    r_chunk = str(df.iloc[row]["r"])
    score = float(df.iloc[row]["score"])
    gt = int(df.iloc[row]["gt"])
    q_song = str(df.iloc[row]["q_song"])
    r_song = str(df.iloc[row]["r_song"])

    md = (
        "### Details\n"
        f"**GT:** `{gt}`\n\n"
        f"**Q song:** `{q_song}`\n\n"
        f"**R song:** `{r_song}`\n\n"
        f"**Q chunk:** `{q_chunk}`\n\n"
        f"**R chunk:** `{r_chunk}`\n\n"
        f"**Similarity score:** `{score:.6f}`\n"
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
        headers=["gt", "q_song", "r_song", "score", "q", "r"],
        datatype=["number", "str", "str", "number", "str", "str"],
        interactive=False,
        type="pandas",
        label="Results",
        wrap=True,
    )

    gr.Markdown("---")
    details_md = gr.Markdown("### Details\nRun Compare, then click a row.")
    selected_q_audio = gr.Audio(
        label="Selected q chunk", type="filepath", interactive=False
    )
    selected_r_audio = gr.Audio(
        label="Selected r chunk", type="filepath", interactive=False
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
