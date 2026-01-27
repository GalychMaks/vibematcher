from pathlib import Path
import numpy as np
import soundfile as sf
import torch
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

        # Lazy initialization of resampler
        self.resampler = None

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
        #!!! Shapes can differ depending on the model, comments are for default model

        # Load audio file
        audio, sr = sf.read(path)  # shape: (T,) or (T, C)

        # Convert stereo to mono
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)  # shape: (T,)

        # Resample if needed
        audio_tensor = torch.from_numpy(audio).float()  # shape: (T,)
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio_tensor = self.resampler(audio_tensor)  # shape: (T',)

        # Tokenize input
        inputs = self.processor(
            audio_tensor.numpy(), sampling_rate=self.target_sr, return_tensors="pt"
        )  # input_values shape: (1, T')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get hidden states from MERT
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            # self.logger.debug(f"Outputs shape: {outputs.hidden_states[0].shape}")
            # each hidden state: (1, T'', 768)

        # Stack hidden states: [13 layers, T'', 768]
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        # shape: (13, T'', 768)

        # self.logger.debug(
        #     f"All layer hidden states shape: {all_layer_hidden_states.shape}"
        # )

        # Average across time -> shape: (13, 768)
        time_avg = all_layer_hidden_states.mean(dim=1)
        # self.logger.debug(f"Time average shape: {time_avg.shape}")

        # Then average across layers -> shape: (768,)
        embedding = time_avg.mean(dim=0)
        # self.logger.debug(f"Embedding shape: {embedding.shape}")

        # Return as NumPy array
        return embedding.cpu().numpy().astype(np.float32)  # shape: (768,)
