from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from vibematcher.features.rmvpe import RMVPE


def _load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    """
    Loads audio as mono float32 in range [-1, 1] (best-effort).
    Returns (audio, sample_rate).
    """
    # Prefer soundfile if available (often simplest / fastest)
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(str(path), always_2d=True)  # (T, C)
        audio = audio.astype(np.float32, copy=False)
        audio = np.mean(audio, axis=1)  # mono
        return audio, int(sr)
    except Exception:
        pass

    # Fallback to torchaudio
    import torchaudio  # type: ignore

    wav, sr = torchaudio.load(str(path))  # (C, T), float32
    if wav.ndim != 2:
        raise ValueError(
            f"Expected torchaudio waveform with shape (C, T), got {tuple(wav.shape)}"
        )
    wav = wav.mean(dim=0)  # mono (T,)
    return wav.cpu().numpy().astype(np.float32, copy=False), int(sr)


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio

    # Prefer torchaudio resampler if available
    try:
        import torchaudio  # type: ignore

        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        wav_rs = torchaudio.functional.resample(wav, sr, target_sr)
        return wav_rs.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass

    # Fallback to librosa resample
    import librosa  # type: ignore

    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(
        np.float32, copy=False
    )


def _peak_normalize(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= eps:
        return audio
    return (audio / peak).astype(np.float32, copy=False)


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

    def extract_f0(self, audio_path: str | Path, thred: float = 0.03) -> np.ndarray:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        audio, sr = _load_audio_mono(path)
        audio = _resample_if_needed(audio, sr, self.target_sr)
        if self.normalize_audio:
            audio = _peak_normalize(audio)

        # RMVPE.infer_from_audio expects a 1D numpy array or torch tensor
        f0 = self._rmvpe.infer_from_audio(audio, thred=thred)

        # Ensure final type is np.ndarray[float32]
        f0 = np.asarray(f0, dtype=np.float32)
        return f0


def extract_f0_rmvpe(
    audio_path: str | Path,
    *,
    model_path: str,
    is_half: bool = False,
    device: Optional[str] = None,
    use_jit: bool = False,
    thred: float = 0.03,
) -> np.ndarray:
    """
    Convenience functional API.
    """
    extractor = RMVPEF0Extractor(
        model_path=model_path,
        is_half=is_half,
        device=device,
        use_jit=use_jit,
    )
    return extractor.extract_f0(audio_path, thred=thred)


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
