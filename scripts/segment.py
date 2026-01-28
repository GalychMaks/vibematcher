# segment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import librosa


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float  # seconds
    label: str = "seg"


SegMode = Literal["uniform", "beats", "structure"]


def uniform_segments(
    duration_s: float, seg_len_s: float = 8.0, hop_s: float = 4.0
) -> List[Segment]:
    segs: List[Segment] = []
    t = 0.0
    while t < duration_s:
        end = min(duration_s, t + seg_len_s)
        if end - t >= 1.0:
            segs.append(Segment(t, end, "uniform"))
        t += hop_s
    return segs


def beat_segments(
    y: np.ndarray,
    sr: int,
    *,
    beats_per_seg: int = 16,
    hop_beats: int = 8,
) -> List[Segment]:
    """
    Beat-synchronous chunks (roughly "phrase-like" if beats_per_seg ~= 8..32).
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if beat_times.size < beats_per_seg + 1:
        # fallback: one big segment
        dur = librosa.get_duration(y=y, sr=sr)
        return [Segment(0.0, dur, "beats_fallback")]

    segs: List[Segment] = []
    i = 0
    while i + beats_per_seg < beat_times.size:
        start = float(beat_times[i])
        end = float(beat_times[i + beats_per_seg])
        if end - start >= 1.0:
            segs.append(Segment(start, end, f"beats_{beats_per_seg}"))
        i += hop_beats

    return segs


def structure_segments_librosa(
    y: np.ndarray,
    sr: int,
    *,
    k: int = 8,
) -> List[Segment]:
    """
    "Structure-ish" segmentation using constrained agglomerative clustering on beat-synchronous chroma.

    NOTE: This is not a full structure analyzer (intro/chorus labels),
    but it often finds change-points that are useful for plagiarism scanning.
    """
    # Beat sync
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    if beat_frames.size < 8:
        dur = librosa.get_duration(y=y, sr=sr)
        return [Segment(0.0, dur, "structure_fallback")]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    # librosa.segment.agglomerative expects time axis = -1 by default
    boundaries = librosa.segment.agglomerative(chroma_sync, k=k, axis=-1)
    # boundaries are indices into beat frames
    boundaries = np.unique(np.clip(boundaries, 0, beat_frames.size - 1))

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    cut_times = beat_times[boundaries]
    cut_times = np.unique(np.concatenate([[0.0], cut_times, [beat_times[-1]]]))

    segs: List[Segment] = []
    for a, b in zip(cut_times[:-1], cut_times[1:]):
        a = float(a)
        b = float(b)
        if b - a >= 1.0:
            segs.append(Segment(a, b, "structure"))
    return segs


def segment_audio(
    y: np.ndarray,
    sr: int,
    mode: SegMode = "structure",
    *,
    uniform_len_s: float = 8.0,
    uniform_hop_s: float = 4.0,
    beats_per_seg: int = 16,
    hop_beats: int = 8,
    structure_k: int = 8,
) -> List[Segment]:
    dur = float(librosa.get_duration(y=y, sr=sr))
    if mode == "uniform":
        return uniform_segments(dur, seg_len_s=uniform_len_s, hop_s=uniform_hop_s)
    if mode == "beats":
        return beat_segments(y, sr, beats_per_seg=beats_per_seg, hop_beats=hop_beats)
    return structure_segments_librosa(y, sr, k=structure_k)
