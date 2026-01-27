from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class MertEmbedder:
    SUPPORTED_MODELS = {
        "MERT-v1-330M",
        "MERT-v1-95M",
        "MERT-v0-public",
        "MERT-v0",
        "music2vec-v1",
    }

    def __init__(self, model_name: str = "MERT-v1-95M"):
        """
        Initializes the MertEmbedder with a specified pretrained MERT model.

        :param model_name: Name of the pretrained model to load. Must be in SUPPORTED_MODELS.
        :raises ValueError: If an unsupported model name is provided.
        """

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported MERT model: '{model_name}'.\n"
                f"Supported models are: {', '.join(sorted(self.SUPPORTED_MODELS))}"
            )

        full_model_name = f"m-a-p/{model_name}"

        print(f"Loading MERT model: {full_model_name}")
        self.model = AutoModel.from_pretrained(full_model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            full_model_name, trust_remote_code=True
        )

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Target sample rate expected by MERT
        self.target_sr = self.processor.sampling_rate
        self.resampler = None  # lazy

    def _embed_audio_tensor(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        audio_tensor: 1D mono torch tensor at self.target_sr
        returns: 1D np.ndarray (D,) float32
        """
        inputs = self.processor(
            audio_tensor.cpu().numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)

        # [num_layers, 1, T, D] -> [num_layers, T, D]
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)

        # time avg -> [num_layers, D]
        time_avg = all_layer_hidden_states.mean(dim=1)

        # layer avg -> [D]
        embedding = time_avg.mean(dim=0)

        return embedding.detach().cpu().numpy().astype(np.float32)

    def embed(
        self,
        path: str | Path,
        chunk_seconds: float = 5.0,
        overlap_seconds: float = 2.5,
    ) -> np.ndarray:
        """
        Returns a 2D embedding matrix of shape (num_chunks, embedding_dim).

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

        # (N, D)
        return np.stack(chunk_embeds, axis=0).astype(np.float32, copy=False)
