import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import librosa

from vibematcher.f0.f0_contour_cache import F0Contour

NO_PITCH = "<no_pitch>"


PC_NAMES = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


@dataclass
class MelodySequence:
    melody: list[str]

    MELODY_KEY = "melody.txt"

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        force_recompute: bool = False,
    ) -> "MelodySequence":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem
        melody_path = cache_dir / cls.MELODY_KEY
        if not force_recompute and cache_dir.exists() and melody_path.exists():
            melody = melody_path.read_text().split(" ")
            return cls(melody=melody)

        # Load f0
        f0_hz: np.ndarray = F0Contour.from_audio_file(audio_path=audio_path).f0_hz

        # Compute melody
        melody: list[str] = []
        for f in f0_hz:
            if not math.isfinite(f) or f <= 0.0:
                melody.append(NO_PITCH)
                continue

            midi = float(librosa.hz_to_midi(f))
            pc = int(round(midi)) % 12
            melody.append(PC_NAMES[pc])

        melody_path.parent.mkdir(parents=True, exist_ok=True)
        melody_path.write_text(" ".join(melody))

        return cls(melody=melody)


if __name__ == "__main__":
    melody = MelodySequence.from_audio_file(
        "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav",
        force_recompute=True,
    ).melody
    print("Estimated Melody:")
    print(f"len(melody): {len(melody)}")
    print(melody[:100])
