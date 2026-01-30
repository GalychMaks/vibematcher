from pathlib import Path
import subprocess
import shutil


class DemucsStemsSeparator:
    def separate(self, audio_path: str | Path) -> list[Path]:
        audio_path = Path(audio_path)

        # final dir: same name as input, without extension
        final_dir = audio_path.with_suffix("")
        final_dir.mkdir(parents=True, exist_ok=True)

        # run demucs; write into a known root
        out_root = audio_path.parent / "separated"
        model = "htdemucs_ft"

        subprocess.run(
            [
                "demucs",
                "-n",
                "htdemucs_ft",
                "--two-stems",
                "vocals",
                "-o",
                str(out_root),
                str(audio_path),
            ],
            check=True,
        )

        # demucs writes: out_root/<model>/<track_name>/*.wav
        track_dir = out_root / model / audio_path.stem

        stems = []
        for stem in track_dir.glob("*.wav"):
            dest = final_dir / stem.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(stem), str(dest))
            stems.append(dest)

        return stems
