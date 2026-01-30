import math
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import distance
from vibematcher.pitch.pitch_cache import MelodySequence
from vibematcher.stems.demucs import DemucsStemsSeparator
from vibematcher.stems.stems_cache import Stems

PC_TO_INT = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def _signed_pc_interval(a: int, b: int) -> int:
    """
    Smallest signed pitch-class interval from a->b, in [-6, +6].
    Examples:
      A(9) -> C(0):  +3
      C(0) -> A(9):  -3
      C(0) -> F#(6): +6
      F#(6) -> C(0): -6
    """
    d = (b - a) % 12  # 0..11
    if d > 6:
        d -= 12  # -5..-1 (and -6 if d==6? no, d==6 stays 6)
    return int(d)


def melody_to_signed_intervals(melody: list[str]) -> list[int]:
    """
    Convert pitch-class melody to signed interval sequence.
    Drops tokens not in PC_TO_INT (e.g., <no_pitch>).
    """
    pcs = [PC_TO_INT[n] for n in melody if n in PC_TO_INT]
    if len(pcs) < 2:
        return []
    return [_signed_pc_interval(a, b) for a, b in zip(pcs[:-1], pcs[1:])]


def interval_wer(reference: list[int], hypothesis: list[int]) -> float:
    """
    Word Error Rate (WER) on interval sequences.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    edits = distance.Levenshtein.distance(reference, hypothesis)
    return edits / len(reference)


@dataclass
class CompareWER:
    stems_separator = DemucsStemsSeparator()

    @classmethod
    def score(cls, query_path: str | Path, reference_path: str | Path) -> float:
        q_stems = Stems.from_audio_file(
            query_path,
            stems_separator=cls.stems_separator,
        ).stems
        r_stems = Stems.from_audio_file(
            reference_path,
            stems_separator=cls.stems_separator,
        ).stems

        best = float("nan")
        for q in q_stems:
            q_melody = MelodySequence.from_audio_file(q).melody
            q_intervals = melody_to_signed_intervals(q_melody)

            for r in r_stems:
                r_melody = MelodySequence.from_audio_file(r).melody
                r_intervals = melody_to_signed_intervals(r_melody)

                score = interval_wer(reference=r_intervals, hypothesis=q_intervals)
                if math.isfinite(score) and (not math.isfinite(best) or score < best):
                    best = score

        return best


if __name__ == "__main__":
    # Example usage: compare two full songs by stems
    query = "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav"
    references = [Path(path) for path in Path("data/original").glob("*.wav")]

    # Create once and reuse to avoid reloading demucs model repeatedly
    compare_dwt = CompareWER()

    for reference in references:
        score = compare_dwt.score(query, reference)

    print(score)
