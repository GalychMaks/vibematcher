from dataclasses import dataclass
from pathlib import Path
from typing import Optional


from vibematcher.chunking.song_former import SongFormer


@dataclass
class AudioChunks:
    """
    Cache utility for SongFormer chunking.
    """

    chunks: list[Path]

    @classmethod
    def from_audio_file(
        cls,
        audio_path: str | Path,
        *,
        song_former: Optional[SongFormer] = None,
        force_recompute: bool = False,
    ) -> "AudioChunks":
        audio_path = Path(audio_path)
        cache_dir = audio_path.parent / audio_path.stem

        # If already computed -> load paths from disk
        if not force_recompute and cache_dir.exists():
            return cls(chunks=sorted(cache_dir.glob("*.wav")))

        # Else compute + save
        if song_former is None:
            song_former = SongFormer()

        cache_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = song_former.chunk_to_dir(
            audio_path=audio_path,
            out_dir=cache_dir,
        )

        return cls(chunks=chunk_paths)
