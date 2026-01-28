from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class MertEmbedderChunked:
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

        self.model = AutoModel.from_pretrained(full_model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(full_model_name, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_sr = self.processor.sampling_rate
        self.resampler = None

    def embed_chunks(
        self,
        path: str | Path,
        window_len_sec: float = 10.0,
        hop_len_sec: float = 5.0,
        silence_threshold_db: float = -40.0,
    ) -> np.ndarray:
        """
        Split audio into overlapping chunks, run through MERT,
        and return a matrix [num_chunks, feature_dim],
        averaging over layers but not over time steps.
        """

        # Load audio file
        audio, sr = sf.read(path)

        # Convert stereo to mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Trim silence at beginning and end
        audio = self.trim_silence(audio, threshold_db=silence_threshold_db)

        # Handle fully silent audio
        if audio.size == 0:
            raise ValueError(f"Audio contains only silence: {path}")

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Resample if needed
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio_tensor = self.resampler(audio_tensor)

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

            # Run chunk through MERT
            inputs = self.processor(
                chunk.numpy(),
                sampling_rate=self.target_sr,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # list of (1, T_chunk, D)

            # Stack layers -> (num_layers, T_chunk, D)
            all_layers = torch.stack(hidden_states).squeeze(1)

            # Average over layers -> (T_chunk, D)
            layer_avg = all_layers.mean(dim=0)

            # Average over time to get fixed-size chunk embedding
            chunk_vector = layer_avg.mean(dim=0)  # shape: (feature_dim,)
            chunk_embeddings.append(chunk_vector.cpu().numpy())

        return np.stack(chunk_embeddings)  # shape: (num_chunks, feature_dim)
    
    def trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        eps: float = 1e-10,
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of an audio signal.

        :param audio: 1D numpy array (mono)
        :param threshold_db: silence threshold in decibels
        :param eps: numerical stability constant
        :return: trimmed audio signal
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
    