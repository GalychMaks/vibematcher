from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import librosa

from transformers import AutoModel, Wav2Vec2FeatureExtractor


class MertEmbedderChunked:
    """
    Chunk-based audio embedder using MERT models.

    Pipeline:
        audio -> silence trimming -> BPM normalization (optional)
              -> resampling -> chunking -> MERT embeddings

    Each chunk is embedded independently and returned as a fixed-size vector.
    """

    SUPPORTED_MODELS = {
        "MERT-v1-330M",
        "MERT-v1-95M",
        "MERT-v0-public",
        "MERT-v0",
        "music2vec-v1",
    }

    def __init__(self, model_name: str = "MERT-v1-95M"):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported MERT model: {model_name}")

        full_model_name = f"m-a-p/{model_name}"
        print(f"Loading MERT model: {full_model_name}")

        self.model = AutoModel.from_pretrained(
            full_model_name, trust_remote_code=True
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            full_model_name, trust_remote_code=True
        )

        self.model.eval()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.target_sr = self.processor.sampling_rate
        self.resampler = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        path: str | Path,
        window_len_sec: float = 10.0,
        hop_len_sec: float = 5.0,
        silence_threshold_db: float = -40.0,
        target_bpm: float | None = 120.0,
    ) -> np.ndarray:
        """
        Split audio into overlapping chunks and compute MERT embeddings.

        Args:
            path: Path to audio file
            window_len_sec: Chunk duration in seconds
            hop_len_sec: Hop size in seconds
            silence_threshold_db: Silence threshold for trimming
            target_bpm: Target BPM for tempo normalization.
                        If None, BPM normalization is disabled.

        Returns:
            np.ndarray of shape (num_chunks, feature_dim)
        """

        # --------------------------------------------------------------
        # Load audio
        # --------------------------------------------------------------
        audio, sr = sf.read(path)

        # Convert stereo to mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # --------------------------------------------------------------
        # Trim leading and trailing silence
        # --------------------------------------------------------------
        audio = self._trim_silence(audio, silence_threshold_db)

        if audio.size == 0:
            raise ValueError(f"Audio contains only silence: {path}")

        # --------------------------------------------------------------
        # BPM normalization (optional)
        # --------------------------------------------------------------
        if target_bpm is not None:
            audio, original_bpm = self._normalize_bpm(
                audio, sr, target_bpm
            )
            print(
                f"BPM normalized: {original_bpm:.2f} -> {target_bpm:.2f}"
            )

        # --------------------------------------------------------------
        # Convert to torch tensor
        # --------------------------------------------------------------
        audio_tensor = torch.from_numpy(audio).float()

        # --------------------------------------------------------------
        # Resample to model sampling rate if needed
        # --------------------------------------------------------------
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(
                    orig_freq=sr, new_freq=self.target_sr
                )
            audio_tensor = self.resampler(audio_tensor)

        # --------------------------------------------------------------
        # Chunking
        # --------------------------------------------------------------
        window_len = int(window_len_sec * self.target_sr)
        hop_len = int(hop_len_sec * self.target_sr)
        audio_len = audio_tensor.shape[0]

        starts = np.arange(0, audio_len, hop_len)
        chunk_embeddings = []

        for start in starts:
            end = start + window_len

            if end > audio_len:
                chunk = torch.nn.functional.pad(
                    audio_tensor[start:], (0, end - audio_len)
                )
            else:
                chunk = audio_tensor[start:end]

            # ----------------------------------------------------------
            # MERT forward pass
            # ----------------------------------------------------------
            inputs = self.processor(
                chunk.numpy(),
                sampling_rate=self.target_sr,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(
                    **inputs, output_hidden_states=True
                )

            # hidden_states: list of (1, T_chunk, D)
            hidden_states = outputs.hidden_states

            # Stack layers -> (num_layers, T_chunk, D)
            all_layers = torch.stack(hidden_states).squeeze(1)

            # Average over layers -> (T_chunk, D)
            layer_avg = all_layers.mean(dim=0)

            # Average over time -> (D,)
            chunk_vector = layer_avg.mean(dim=0)

            chunk_embeddings.append(chunk_vector.cpu().numpy())

        return np.stack(chunk_embeddings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_bpm(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate tempo (BPM) of an audio signal.
        """
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo)

    def _normalize_bpm(
        self,
        audio: np.ndarray,
        sr: int,
        target_bpm: float,
    ) -> tuple[np.ndarray, float]:
        """
        Time-stretch audio to match target BPM without changing pitch.

        Returns:
            stretched_audio, original_bpm
        """
        original_bpm = self._estimate_bpm(audio, sr)

        # Guard against unreliable tempo estimates
        if original_bpm < 30 or original_bpm > 300:
            print(
                f"Warning: unreliable BPM estimate ({original_bpm:.2f}), "
                "skipping BPM normalization."
            )
            return audio, original_bpm

        rate = target_bpm / original_bpm

        stretched = librosa.effects.time_stretch(
            audio.astype(np.float64),
            rate=rate,
        ).astype(audio.dtype)

        return stretched, original_bpm

    def _trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float,
        eps: float = 1e-10,
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of an audio signal.

        Args:
            audio: Mono audio signal
            threshold_db: Silence threshold in decibels
            eps: Numerical stability constant

        Returns:
            Trimmed audio signal
        """
        amplitude = np.abs(audio)
        max_amp = np.max(amplitude) + eps
        amplitude_db = 20 * np.log10(amplitude / max_amp + eps)

        non_silent = np.where(amplitude_db > threshold_db)[0]

        if len(non_silent) == 0:
            return np.array([], dtype=audio.dtype)

        start = non_silent[0]
        end = non_silent[-1] + 1

        return audio[start:end]
