from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T


class MuQEmbedder:
    """
    Chunked audio embedder for OpenMuQ/MuQ-MuLan-large that mirrors the MERT embedder API.

    Uses the official `muq` library:
      pip install muq

    The model expects 24 kHz mono audio. (We resample if needed.)
    """

    SUPPORTED_MODELS = {
        "OpenMuQ/MuQ-MuLan-large",
    }

    def __init__(self, model_name: str = "OpenMuQ/MuQ-MuLan-large"):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported MuQ model: '{model_name}'.\n"
                f"Supported models are: {', '.join(sorted(self.SUPPORTED_MODELS))}"
            )

        # Import here so your package can still import without muq installed
        from muq import MuQMuLan  # type: ignore

        print(f"Loading MuQ-MuLan model: {model_name}")
        self.model = MuQMuLan.from_pretrained(model_name)

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Target sample rate per official examples (24k). See HF model card usage.
        self.target_sr = 24000
        self.resampler: Optional[T.Resample] = None  # lazy / per-orig-sr

    def _embed_audio_tensor(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        audio_tensor: 1D mono torch tensor at self.target_sr (24000)
        returns: 1D np.ndarray (D,) float32
        """
        wavs = audio_tensor.unsqueeze(0).to(self.device)  # (1, T)

        with torch.inference_mode():
            # HF model card shows: audio_embeds = mulan(wavs=wavs)
            # This returns a single embedding per input waveform.
            audio_embeds = self.model(wavs=wavs)

        # Make robust to (1, D) or other small variations
        if isinstance(audio_embeds, (list, tuple)):
            audio_embeds = audio_embeds[0]

        if audio_embeds.ndim == 2 and audio_embeds.shape[0] == 1:
            emb = audio_embeds[0]
        elif audio_embeds.ndim == 1:
            emb = audio_embeds
        else:
            # Fall back: average across batch/time dims if any unexpected shape appears
            emb = audio_embeds.mean(dim=0)

        return emb.detach().float().cpu().numpy().astype(np.float32, copy=False)

    def embed(
        self,
        path: str | Path,
        chunk_seconds: float = 5.0,
        overlap_seconds: float = 2.5,
    ) -> np.ndarray:
        """
        Returns a 2D embedding matrix of shape (num_chunks, embedding_dim), float32.

        Chunking:
          - chunk length: 5 seconds (default)
          - overlap: 2.5 seconds (default)
          - step = chunk_seconds - overlap_seconds

        The last chunk is zero-padded to exactly chunk_seconds (if needed).
        """
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds must be >= 0")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")
        if overlap_seconds >= chunk_seconds:
            raise ValueError("overlap_seconds must be < chunk_seconds")

        audio, sr = sf.read(path)  # (T,) or (T, C)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.numel() == 0:
            return np.empty((0, 0), dtype=np.float32)

        # resample whole signal once
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio_tensor = self.resampler(audio_tensor)

        total_len = int(audio_tensor.numel())
        if total_len == 0:
            return np.empty((0, 0), dtype=np.float32)

        chunk_len = int(round(chunk_seconds * self.target_sr))
        overlap_len = int(round(overlap_seconds * self.target_sr))
        step = chunk_len - overlap_len

        if chunk_len <= 0:
            raise ValueError("chunk_seconds too small for the current sampling rate")
        if step <= 0:
            raise ValueError("Invalid chunk/overlap configuration (step <= 0)")

        chunk_embeds: list[np.ndarray] = []
        start = 0

        while start < total_len:
            end = start + chunk_len
            chunk = audio_tensor[start:end]

            # last chunk: pad to full length and stop
            if chunk.numel() < chunk_len:
                chunk = F.pad(chunk, (0, chunk_len - chunk.numel()))
                chunk_embeds.append(self._embed_audio_tensor(chunk))
                break

            chunk_embeds.append(self._embed_audio_tensor(chunk))
            start += step

        if not chunk_embeds:
            return np.empty((0, 0), dtype=np.float32)

        return np.stack(chunk_embeds, axis=0).astype(np.float32, copy=False)
