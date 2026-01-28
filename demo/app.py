import heapq
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from vibematcher.compare.compare import compare_fingerprints
from vibematcher.fingerprint.fingerprint import AudioFingerprint

# -----------------------
# Hardcoded settings
# -----------------------
DATASET_DIR = Path("data/original")  # <-- change to your fingerprints root dir

# Cache heatmaps only for top-K matches (to avoid storing huge matrices for large datasets)
CACHE_HEATMAP_TOP_K = 100


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


def make_heatmap_fig(sim: np.ndarray, *, p_low=5, p_high=95, gamma=0.5) -> plt.Figure:
    sim = np.asarray(sim, dtype=float)
    vmin, vmax = np.nanpercentile(sim, [p_low, p_high])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
    im = ax.imshow(sim, aspect="auto", origin="lower", norm=norm)

    ax.set_xlabel("candidate chunk index")
    ax.set_ylabel("query chunk index")
    fig.colorbar(
        im,
        ax=ax,
        label=f"cosine similarity (power γ={gamma}, clipped {p_low}–{p_high} pct)",
    )
    fig.tight_layout()
    return fig


# -----------------------
# Gradio callbacks
# -----------------------
def compare(
    audio_path: str,
) -> tuple[pd.DataFrame, dict, str, plt.Figure | None, str | None]:
    """
    Returns:
      - results df
      - cache dict (aspects for all items + heatmaps for top-K)
      - details markdown (reset/placeholder)
      - heatmap figure (reset)
      - selected audio (reset)
    """
    if not audio_path:
        raise gr.Error("Upload an audio file.")

    q = AudioFingerprint.from_audio_file(audio_path)

    items: list[str] = []
    similarities: list[float] = []

    aspects_by_item: dict[str, dict[str, float]] = {}
    query_starts: np.ndarray | None = None

    # Keep only top-K heatmaps in a min-heap: (overall, item, mert_mat, cand_starts)
    top_heap: list[tuple[float, str, np.ndarray, np.ndarray | None]] = []

    original_files = sorted(DATASET_DIR.rglob("*.wav"))
    for original_file in original_files:
        item = str(original_file)

        original_fp = ensure_fp_for_file(Path(original_file))
        r = compare_fingerprints(q, original_fp)

        overall = float(r.overall)
        aspects = {k: float(v) for k, v in r.aspects.items()}

        items.append(item)
        similarities.append(overall)
        aspects_by_item[item] = aspects

        if query_starts is None:
            query_starts = r.query_chunk_starts_sec

        mert_mat = r.correlation_matrices.get("mert")
        if mert_mat is not None:
            entry = (overall, item, mert_mat, r.cand_chunk_starts_sec)
            if len(top_heap) < CACHE_HEATMAP_TOP_K:
                heapq.heappush(top_heap, entry)
            else:
                if overall > top_heap[0][0]:
                    heapq.heapreplace(top_heap, entry)

    df = pd.DataFrame({"item": items, "similarity": similarities})
    df = df.sort_values("similarity", ascending=False).reset_index(drop=True)

    # Build heatmap cache dict from heap
    mert_cache: dict[str, dict[str, object]] = {}
    for overall, item, mat, cand_starts in top_heap:
        mert_cache[item] = {"mat": mat, "cand_starts": cand_starts}

    cache = {
        "query_starts": query_starts,
        "aspects": aspects_by_item,
        "mert": mert_cache,  # only for top-K
        "heatmap_top_k": CACHE_HEATMAP_TOP_K,
    }

    details_md = "### Details\nClick a row to load/play that audio (no recomputation)."
    return df, cache, details_md, None, None


def show_details(
    df: pd.DataFrame,
    cache: dict,
    evt: gr.SelectData,
) -> tuple[str, plt.Figure | None, str]:
    """
    On row click:
      - DO NOT recompute anything
      - load / play selected audio
      - show cached heatmap if available (top-K only)
    """
    if df is None or len(df) == 0:
        raise gr.Error("No results yet. Click Compare first.")

    cache = cache or {}
    aspects_by_item: dict[str, dict[str, float]] = cache.get("aspects", {}) or {}
    mert_cache: dict[str, dict[str, object]] = cache.get("mert", {}) or {}
    top_k = int(cache.get("heatmap_top_k", CACHE_HEATMAP_TOP_K))

    row = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)

    item = str(df.iloc[row]["item"])
    sim = float(df.iloc[row]["similarity"])
    aspects = aspects_by_item.get(item)

    # Load audio path (this is the main thing you asked for)
    selected_audio_path = item

    # Heatmap only if cached from Compare() step
    fig = None
    mert_entry = mert_cache.get(item)
    if mert_entry is not None:
        mat = mert_entry.get("mat")
        if isinstance(mat, np.ndarray):
            fig = make_heatmap_fig(mat)

    md = (
        "### Details\n"
        f"**Item:** `{item}`\n\n"
        f"**Similarity (overall):** `{sim:.4f}`\n\n"
        f"**Aspects:** `{aspects if aspects is not None else 'n/a'}`\n\n"
    )

    if fig is None:
        md += f"**Heatmap:** not cached (only stored for top-{top_k} matches)\n"

    return md, fig, selected_audio_path


# -----------------------
# UI
# -----------------------
with gr.Blocks(title="VibeMatcher") as demo:
    gr.Markdown("## VibeMatcher — audio similarity (MERT cosine, chunked)")

    cache_state = gr.State({})

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
    selected_audio = gr.Audio(
        label="Selected item audio", type="filepath", interactive=False
    )
    heat = gr.Plot(label="Chunk similarity heatmap (cached for top matches)")

    # Compare fills table + stores cache + resets details/plot/audio
    run_btn.click(
        fn=compare,
        inputs=inp,
        outputs=[out, cache_state, details_md, heat, selected_audio],
    )

    # Selecting a row loads audio + shows cached details/heatmap (no recomputation)
    out.select(
        fn=show_details,
        inputs=[out, cache_state],
        outputs=[details_md, heat, selected_audio],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
