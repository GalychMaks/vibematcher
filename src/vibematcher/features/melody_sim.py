from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List, Union

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor


@dataclass(frozen=True)
class MelodySimResult:
    score: float
    decision: int
    decision_matrix: torch.Tensor
    similarity_matrix: torch.Tensor


def _window_slide(seq: torch.Tensor, window_len: int, hop_len: int) -> torch.Tensor:
    if seq.shape[-1] >= window_len:
        return seq.unfold(-1, window_len, hop_len)
    return torch.nn.functional.pad(seq, (0, window_len - seq.shape[-1])).unsqueeze(0)


def _run_mert_model_and_get_features(
    waveforms: torch.Tensor,
    audio_model: AutoModel,
    time_reduce: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    if time_reduce is None:
        time_reduce = torch.nn.AvgPool1d(
            kernel_size=10, stride=10, count_include_pad=False
        )
    hidden_states = audio_model(waveforms, output_hidden_states=True).hidden_states
    audio_features = torch.stack(
        [
            time_reduce(h.detach()[:, :, :].permute(0, 2, 1)).permute(0, 2, 1)
            for h in hidden_states[2::3]
        ],
        dim=1,
    )
    return audio_features


@torch.no_grad()
def _inference_two_window_seqs(
    model: "SiameseWrapper",
    audio_model: AutoModel,
    piece1_windowed: Union[List[torch.Tensor], torch.Tensor],
    piece2_windowed: Union[List[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    model.eval()

    if isinstance(piece1_windowed, list) and isinstance(piece2_windowed, list):
        piece1_windowed = torch.stack(piece1_windowed)
        piece2_windowed = torch.stack(piece2_windowed)

    n1, _ = piece1_windowed.shape
    n2, _ = piece2_windowed.shape

    audio_model.to(model.device)
    audio_model.eval()
    time_reduce = torch.nn.AvgPool1d(
        kernel_size=10, stride=10, count_include_pad=False
    ).to(model.device)

    mert_features1 = _run_mert_model_and_get_features(
        piece1_windowed.to(model.device), audio_model, time_reduce
    )
    mert_features2 = _run_mert_model_and_get_features(
        piece2_windowed.to(model.device), audio_model, time_reduce
    )

    _, num_layers, _, layer_dim = mert_features1.shape
    assert num_layers == 4 and layer_dim == 768

    mert_features1 = mert_features1.permute(0, 1, 3, 2)
    mert_features2 = mert_features2.permute(0, 1, 3, 2)

    mert_features1 = mert_features1.flatten(start_dim=1, end_dim=2)
    mert_features2 = mert_features2.flatten(start_dim=1, end_dim=2)

    l1, l2 = mert_features1.shape[-1], mert_features2.shape[-1]
    max_len = max(83, max(l1, l2))
    if l1 < max_len:
        mert_features1 = torch.nn.functional.pad(mert_features1, (0, max_len - l1))
    if l2 < max_len:
        mert_features2 = torch.nn.functional.pad(mert_features2, (0, max_len - l2))

    similarity_matrix = torch.zeros(n1, n2)
    for i_row in range(n1):
        similarity_matrix[i_row, :] = 1 - model._inference_step(
            mert_features1[i_row : i_row + 1].repeat(n2, 1, 1).to(model.device),
            mert_features2.to(model.device),
        )

    return similarity_matrix


@torch.no_grad()
def _inference_two_pieces(
    model: "SiameseWrapper",
    audio_model: AutoModel,
    audio_processor: Wav2Vec2FeatureExtractor,
    waveform1: torch.Tensor,
    waveform2: torch.Tensor,
    sample_rate: int,
    window_len_sec: float = 10,
    hop_len_sec: float = 10,
) -> torch.Tensor:
    waveform1_input = audio_processor(
        F.resample(waveform1, sample_rate, audio_processor.sampling_rate),
        sampling_rate=audio_processor.sampling_rate,
        return_tensors="pt",
    )["input_values"].squeeze()

    waveform2_input = audio_processor(
        F.resample(waveform2, sample_rate, audio_processor.sampling_rate),
        sampling_rate=audio_processor.sampling_rate,
        return_tensors="pt",
    )["input_values"].squeeze()

    window_len = int(window_len_sec * audio_processor.sampling_rate)
    hop_len = int(hop_len_sec * audio_processor.sampling_rate)

    piece1_windowed = _window_slide(
        torch.as_tensor(waveform1_input), window_len, hop_len
    )
    piece2_windowed = _window_slide(
        torch.as_tensor(waveform2_input), window_len, hop_len
    )

    return _inference_two_window_seqs(
        model,
        audio_model,
        piece1_windowed,
        piece2_windowed,
    )


def _aggregate_decision_matrix(
    similarity_matrix: torch.Tensor,
    proportion_thres: float = 0.2,
    decision_thres: float = 0.5,
    min_hits: int = 1,
) -> tuple[int, torch.Tensor]:
    decision_matrix = similarity_matrix > decision_thres
    n1, n2 = decision_matrix.shape
    row_sum = decision_matrix.sum(dim=0)
    col_sum = decision_matrix.sum(dim=1)
    plagiarized_pieces1 = row_sum >= min_hits
    plagiarized_pieces2 = col_sum >= min_hits
    if (
        plagiarized_pieces1.sum() > proportion_thres * n1
        and plagiarized_pieces2.sum() > proportion_thres * n2
    ):
        return 1, decision_matrix
    return 0, decision_matrix


class _LazyLoader:
    """Lightweight lazy loader to delay heavy model initialization."""

    def __init__(self, factory, **kwargs):
        self._factory = factory
        self._kwargs = kwargs
        self._instance = None

    def get(self):
        if self._instance is None:
            self._instance = self._factory(**self._kwargs)
        return self._instance


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SiameseNet(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.layer1 = ResidualBlock(3072, 512)
        self.layer2 = ResidualBlock(512, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseWrapper(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.siamese_net = SiameseNet(embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _inference_step(
        self, sample1: torch.Tensor, sample2: torch.Tensor
    ) -> torch.Tensor:
        out_embs1 = self.siamese_net(sample1)
        out_embs2 = self.siamese_net(sample2)
        diff = torch.abs(out_embs1 - out_embs2)
        logit = self.classifier(diff).squeeze()
        scores = torch.sigmoid(logit)
        return scores


class MelodySimModel:
    """Loads MelodySim + MERT and runs pairwise similarity inference."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: Optional[str] = None,
        mert_model_name: str = "m-a-p/MERT-v1-95M",
    ) -> None:
        self.ckpt_path = str(ckpt_path)
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mert_model_name = mert_model_name

        self._load_models()

    def _load_models(self) -> None:
        self.siamese = SiameseWrapper(embedding_dim=128)
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.siamese.load_state_dict(state_dict)
        self.siamese.to(self.device)
        self.siamese.device = self.device
        self.siamese.eval()

        # MERT
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.mert_model_name
        )
        self.audio_model = AutoModel.from_pretrained(
            self.mert_model_name, trust_remote_code=True
        ).to(self.device)
        self.audio_model.eval()

    def _load_audio(self, path: str | Path) -> tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(str(path), normalize=True, channels_first=True)
        wav = wav.mean(dim=0, keepdim=True)
        return wav, sr

    @torch.inference_mode()
    def compare(
        self,
        audio_path1: str | Path,
        audio_path2: str | Path,
        window_len_sec: float = 10.0,
        hop_len_sec: float = 10.0,
        proportion_thres: float = 0.2,
        decision_thres: float = 0.5,
        min_hits: int = 1,
    ) -> MelodySimResult:
        wav1, sr1 = self._load_audio(audio_path1)
        wav2, sr2 = self._load_audio(audio_path2)
        if sr1 != sr2:
            # resample wav2 to match wav1 to keep windowing consistent
            wav2 = torchaudio.functional.resample(wav2, sr2, sr1)
            sr2 = sr1

        similarity_matrix = _inference_two_pieces(
            model=self.siamese,
            audio_model=self.audio_model,
            audio_processor=self.audio_processor,
            waveform1=wav1,
            waveform2=wav2,
            sample_rate=sr1,
            window_len_sec=window_len_sec,
            hop_len_sec=hop_len_sec,
        )

        decision, decision_matrix = _aggregate_decision_matrix(
            similarity_matrix,
            proportion_thres=proportion_thres,
            decision_thres=decision_thres,
            min_hits=min_hits,
        )

        score = float(similarity_matrix.mean().item())

        return MelodySimResult(
            score=score,
            decision=int(decision),
            decision_matrix=decision_matrix,
            similarity_matrix=similarity_matrix,
        )

    def score(
        self,
        audio_path1: str | Path,
        audio_path2: str | Path,
        **kwargs: Any,
    ) -> float:
        return self.compare(audio_path1, audio_path2, **kwargs).score


class MelodySimWrapper:
    """Lightweight wrapper that lazily loads the MelodySimModel."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: Optional[str] = None,
        mert_model_name: str = "m-a-p/MERT-v1-95M",
    ) -> None:
        self._lazy = _LazyLoader(
            MelodySimModel,
            ckpt_path=ckpt_path,
            device=device,
            mert_model_name=mert_model_name,
        )

    def load(self) -> MelodySimModel:
        return self._lazy.get()

    def compare(self, *args, **kwargs) -> MelodySimResult:
        return self._lazy.get().compare(*args, **kwargs)

    def score(self, *args, **kwargs) -> float:
        return self._lazy.get().score(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> float:
        return self.score(*args, **kwargs)
