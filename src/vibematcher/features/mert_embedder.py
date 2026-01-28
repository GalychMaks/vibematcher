from pathlib import Path
import numpy as np
import torch
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torchaudio


class MertEmbedder:
    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M"):
        """
        Initializes the MertEmbedder with a specified pretrained MERT model.

        :param model_name: Name of the pretrained model to load. Must be in SUPPORTED_MODELS.
        :raises ValueError: If an unsupported model name is provided.
        """

        print(f"Loading MERT model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Target sample rate expected by MERT
        self.target_sr = self.processor.sampling_rate

        # Lazy initialization of resampler
        self.resampler = None

    @torch.inference_mode()
    def embed(self, path: str | Path) -> np.ndarray:
        """
        Extracts a embedding from an audio file using a pretrained MERT model.

        The embedding is computed by:
        - Resampling the audio to the model's target sample rate.
        - Feeding it through the model to obtain hidden states from all layers.
        - Averaging over time for each layer.
        - Averaging the results across all layers to obtain a fixed-size embedding.

        :param path: Path to the audio file.
        :return: 1D numpy array representing the extracted feature vector.
        """

        # Load audio file
        audio, sr = torchaudio.load(str(path), normalize=True, channels_first=True)
        audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio = self.resampler(audio)  # shape: (T',)

        inputs = self.processor(
            audio, sampling_rate=self.target_sr, return_tensors="pt"
        )["input_values"].squeeze()

        window_len = int(10 * self.target_sr)
        hop_len = int(10 * self.target_sr)

        if audio.shape[-1] >= window_len:
            audio_windowed = inputs.unfold(-1, window_len, hop_len)
        else:
            audio_windowed = torch.nn.functional.pad(
                inputs, (0, window_len - audio.shape[-1])
            ).unsqueeze(0)

        if isinstance(audio_windowed, list):
            audio_windowed = torch.stack(audio_windowed)

        n, _ = audio_windowed.shape
        print(n)

        time_reduce = torch.nn.AvgPool1d(
            kernel_size=10, stride=10, count_include_pad=False
        ).to(self.device)

        hidden_states = self.model(
            audio_windowed, output_hidden_states=True
        ).hidden_states

        audio_features = torch.stack(
            [
                time_reduce(h.detach()[:, :, :].permute(0, 2, 1)).permute(0, 2, 1)
                for h in hidden_states[2::3]
            ],
            dim=1,
        )

        _, num_layers, _, layer_dim = audio_features.shape
        assert num_layers == 4 and layer_dim == 768

        audio_features = audio_features.permute(0, 1, 3, 2)
        audio_features = audio_features.flatten(start_dim=1, end_dim=2)
        len = audio_features.shape[-1]

        MAX_LEN = 83  # TODO: magic number
        if len < MAX_LEN:
            audio_features = torch.nn.functional.pad(audio_features, (0, MAX_LEN - len))

        return audio_features.cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    # Example usage
    embedder = MertEmbedder()
    embedding = embedder.embed(
        "data/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav"
    )
    print("Embedding shape:", embedding.shape)
    print(embedding[0][0])
