from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import librosa
import numpy as np
import soundfile as sf
from vibematcher.f0.rmvpe import RMVPE


@dataclass
class RMVPEF0Extractor:
    """
    Thin wrapper around RMVPE that:
      - loads audio from `audio_path`
      - converts to mono
      - resamples to 16kHz (RMVPE expects 16k)
      - runs RMVPE inference
      - returns f0 vector as np.ndarray (Hz, with 0 for unvoiced)
    """

    model_path: str = "models/rmvpe.pt"
    is_half: bool = False
    device: Optional[str] = None
    use_jit: bool = False
    target_sr: int = 16000
    normalize_audio: bool = True

    def __post_init__(self) -> None:
        self._rmvpe = RMVPE(
            model_path=self.model_path,
            is_half=self.is_half,
            device=self.device,
            use_jit=self.use_jit,
        )

    def extract_f0(self, audio_path: str | Path) -> np.ndarray:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        # ==================  Load audio ==================
        audio, sr = sf.read(str(path), always_2d=True)

        audio = np.mean(audio.astype(np.float32, copy=False), axis=1)  # mono

        # ================== Resample ==================
        if sr != self.target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.target_sr,
            ).astype(np.float32, copy=False)

        # ================== Peak normalize ==================
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-9:
            audio = (audio / peak).astype(np.float32, copy=False)

        # ================== RMVPE inference ==================
        f0 = self._rmvpe.infer_from_audio(audio)

        # ================== Ensure final type ==================
        f0 = np.asarray(f0, dtype=np.float32)

        return f0


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    extractor = RMVPEF0Extractor()
    v = extractor.extract_f0(
        "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav"
    )
    print("feature shape:", v.shape, "dtype:", v.dtype)
    print()
    print(v[:100])
