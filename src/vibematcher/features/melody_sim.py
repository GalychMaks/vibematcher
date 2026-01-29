from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from vibematcher.chunking.audio_chunks import AudioChunks
from vibematcher.chunking.song_former import SongFormer
from vibematcher.features.mert_embedder import MertEmbedder
from vibematcher.features.mert_embedding import MertEmbedding


@dataclass(frozen=True)
class MelodySimChunkResult:
    q_chunk_path: Path
    r_chunk_path: Path
    score: float
    decision: int
    decision_matrix: torch.Tensor
    similarity_matrix: torch.Tensor


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
    def __init__(self, embedding_dim: int = 128, device: Optional[str] = None):
        super().__init__()
        self.siamese_net = SiameseNet(embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def inference_step(
        self, sample1: torch.Tensor, sample2: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns similarity scores in [0, 1] (higher = more similar).
        """
        out_embs1 = self.siamese_net(sample1)
        out_embs2 = self.siamese_net(sample2)
        diff = torch.abs(out_embs1 - out_embs2)
        logit = self.classifier(diff).squeeze()
        return torch.sigmoid(logit)


def _aggregate_decision_matrix(
    similarity_matrix: torch.Tensor,
    *,
    proportion_thres: float = 0.2,
    decision_thres: float = 0.5,
    min_hits: int = 1,
) -> tuple[int, torch.Tensor]:
    """
    Matches the logic of your original wrapper (with correct n1/n2 usage).
    """
    decision_matrix = similarity_matrix > decision_thres
    n1, n2 = decision_matrix.shape

    # per your original code:
    row_sum = decision_matrix.sum(dim=0)  # (n2,)
    col_sum = decision_matrix.sum(dim=1)  # (n1,)
    plagiarized_pieces1 = row_sum >= min_hits  # flags columns / reference pieces
    plagiarized_pieces2 = col_sum >= min_hits  # flags rows / query pieces

    # original wrapper condition:
    if (plagiarized_pieces1.sum() > proportion_thres * n2) and (
        plagiarized_pieces2.sum() > proportion_thres * n1
    ):
        return 1, decision_matrix
    return 0, decision_matrix


def _load_checkpoint_into_siamese(
    siamese: SiameseWrapper,
    ckpt_path: str | Path,
    *,
    strict: bool = True,
) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    # common patterns:
    # - {"state_dict": ...}
    # - {"model": ...}
    # - raw state_dict
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "siamese",
            "weights",
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    if state_dict is None:
        state_dict = checkpoint

    # handle "module." prefix from DDP
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v

    # sometimes trainer saves with wrapper prefix, e.g. "siamese.siamese_net..."
    # we support both; try direct load first, then fallback strip common prefix.
    try:
        siamese.load_state_dict(cleaned, strict=strict)
        return
    except RuntimeError:
        pass

    # fallback: strip leading "siamese." if present
    stripped = {}
    for k, v in cleaned.items():
        nk = k
        if (
            nk.startswith("siamese.")
            or nk.startswith("siamese_net.")
            or nk.startswith("classifier.")
        ):
            # leave these; they're already aligned
            stripped[nk] = v
        elif nk.startswith("model.") or nk.startswith("net."):
            stripped[nk.split(".", 1)[1]] = v
        else:
            stripped[nk] = v

    siamese.load_state_dict(stripped, strict=False)


class MelodySimCompare:
    """
    Your original wrapper, but now:
    - loads SiameseWrapper weights from a checkpoint
    - keeps the same MertEmbedding-based workflow
    """

    def __init__(
        self,
        *,
        ckpt_path: str | Path = Path("models/siamese_net_20250328.ckpt"),
        device: Optional[str] = None,
        strict_load: bool = True,
    ):
        self.siamese = SiameseWrapper(device=device)
        _load_checkpoint_into_siamese(self.siamese, ckpt_path, strict=strict_load)

        # ensure correct device + eval after loading
        self.siamese.to(self.siamese.device)
        self.siamese.eval()

        self.mert_embedder = MertEmbedder()

    @torch.no_grad()
    def compare(
        self,
        query: str | Path,
        references: list[str | Path],
        *,
        proportion_thres: float = 0.2,
        decision_thres: float = 0.5,
        min_hits: int = 1,
        force_recompute: bool = False,
    ) -> list[MelodySimChunkResult]:
        song_former = SongFormer()
        mert_embedder = MertEmbedder()

        q_chunks: list[Path] = AudioChunks.from_audio_file(
            query,
            song_former=song_former,
            force_recompute=force_recompute,
        ).chunks
        assert len(q_chunks) > 0, "No chunks found in query audio."
        results: list[MelodySimChunkResult] = []
        device = self.siamese.device

        for r in tqdm(references):
            r_chunks: list[Path] = AudioChunks.from_audio_file(
                r,
                song_former=song_former,
                force_recompute=force_recompute,
            ).chunks
            assert len(r_chunks) > 0, "No chunks found in reference audio."

            for r_chunk_path in r_chunks:
                MertEmbedding.from_audio_file(
                    r_chunk_path,
                    mert_embedder=mert_embedder,
                    force_recompute=force_recompute,
                )
                r_chunk_mert_embedding = torch.as_tensor(
                    MertEmbedding.from_audio_file(
                        r_chunk_path,
                        mert_embedder=self.mert_embedder,
                    ).embedding,
                    dtype=torch.float32,
                )

                best_result: Optional[MelodySimChunkResult] = None
                best_score = float("-inf")
                for q_chunk_path in q_chunks:
                    q_chunk_mert_embedding = torch.as_tensor(
                        MertEmbedding.from_audio_file(
                            q_chunk_path,
                            mert_embedder=self.mert_embedder,
                        ).embedding,
                        dtype=torch.float32,
                    )

                    # ==================  Similarity matrix ========================
                    n1 = q_chunk_mert_embedding.shape[0]
                    n2 = r_chunk_mert_embedding.shape[0]
                    similarity_matrix = torch.zeros(n1, n2, device=device)

                    q_dev = q_chunk_mert_embedding.to(device)
                    r_dev = r_chunk_mert_embedding.to(device)

                    for i_row in range(n1):
                        similarity_matrix[i_row, :] = self.siamese.inference_step(
                            q_dev[i_row : i_row + 1].repeat(n2, 1, 1),
                            r_dev,
                        )

                    # ==================  Decision matrix ========================
                    decision, decision_matrix = _aggregate_decision_matrix(
                        similarity_matrix,
                        proportion_thres=proportion_thres,
                        decision_thres=decision_thres,
                        min_hits=min_hits,
                    )

                    # ================== Score ==================
                    score = float(similarity_matrix.mean().item())

                    # ================== Final (CPU) ==================
                    if score > best_score:
                        best_score = score
                        best_result = MelodySimChunkResult(
                            q_chunk_path=q_chunk_path,
                            r_chunk_path=r_chunk_path,
                            score=score,
                            decision=int(decision),
                            decision_matrix=decision_matrix.detach().cpu(),
                            similarity_matrix=similarity_matrix.detach().cpu(),
                        )

                results.append(best_result)

        return results

    def scores(
        self, query: str | Path, references: list[str | Path], **kwargs
    ) -> list[float]:
        return [r.score for r in self.compare(query, references, **kwargs)]


if __name__ == "__main__":
    # Example:
    #   python -m vibematcher.compare.melody_sim_compare -- adjust paths as needed
    ckpt = Path("models/siamese_net_20250328.ckpt")  # <-- change me
    comparator = MelodySimCompare(ckpt_path=ckpt)
    query = Path("data/comparison/Smoke On The Water _2024 Remastered_.wav")
    print(f"Query: {query}")
    references = [Path(path) for path in Path("data/original").glob("*.wav")]
    results: list[MelodySimChunkResult] = comparator.compare(
        query=query,
        references=references,
    )

    pairs = list(
        zip(
            [r.q_chunk_path for r in results],
            [r.r_chunk_path for r in results],
            [r.score for r in results],
        ),
    )
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("score     | q_chunk | r_chunk")
    print("-" * 60)
    for q_chunk, r_chunk, score in pairs:
        print(f"{score:0.6f}  {q_chunk}  {r_chunk}")

    best_q_chunk, best_r_chunk, best_score = pairs[0]
    print("\nBest option is:")
    print(f"{best_score:0.6f}  {best_q_chunk}  {best_r_chunk}")
