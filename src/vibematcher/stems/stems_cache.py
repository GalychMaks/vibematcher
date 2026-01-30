from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vibematcher.stems.demucs import DemucsStemsSeparator


@dataclass
class Stems:
    stems: list[Path]

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        stems_separator: Optional[DemucsStemsSeparator] = None,
        force_recompute: bool = False,
    ) -> "Stems":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem

        vocals_path = cache_dir / "vocals.wav"
        other_path = cache_dir / "no_vocals.wav"

        if (
            not force_recompute
            and cache_dir.exists()
            and vocals_path.exists()
            and other_path.exists()
        ):
            return cls(stems=[vocals_path, other_path])

        if stems_separator is None:
            stems_separator = DemucsStemsSeparator()

        stems = stems_separator.separate(audio_path)
        return cls(stems=stems)


if __name__ == "__main__":
    stems = Stems.from_audio_file("data/original/Sam Smith - I_m Not The Only One.wav")

    print(f"Stems: {stems.stems}")
