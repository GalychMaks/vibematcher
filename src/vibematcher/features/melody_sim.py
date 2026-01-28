from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from vibematcher.features.mert_embedder import MertEmbedder
from vibematcher.features.mert_embedding import MertEmbedding


@dataclass(frozen=True)
class MelodySimResult:
    score: float
    decision: int
    decision_matrix: torch.Tensor
    similarity_matrix: torch.Tensor


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
    def inference_step(
        self, sample1: torch.Tensor, sample2: torch.Tensor
    ) -> torch.Tensor:
        out_embs1 = self.siamese_net(sample1)
        out_embs2 = self.siamese_net(sample2)
        diff = torch.abs(out_embs1 - out_embs2)
        logit = self.classifier(diff).squeeze()
        scores = torch.sigmoid(logit)
        return scores


class MelodySimCompare:
    def __init__(self):
        self.siamese = SiameseWrapper()
        self.mert_embedder = MertEmbedder()

    def compare(
        self, query: str | Path, references: list[str | Path]
    ) -> list[MelodySimResult]:
        q_mert_embedding = torch.as_tensor(
            MertEmbedding.from_audio_file(
                query,
                mert_embedder=self.mert_embedder,
            ).embedding
        )

        results: list[MelodySimResult] = []
        for r in references:
            r_mert_embedding = torch.as_tensor(
                MertEmbedding.from_audio_file(
                    r,
                    mert_embedder=self.mert_embedder,
                ).embedding
            )

            # ==================  Similarity matrix ========================
            n1 = q_mert_embedding.shape[0]
            n2 = r_mert_embedding.shape[0]
            similarity_matrix = torch.zeros(
                n1,
                n2,
            )
            for i_row in range(n1):
                similarity_matrix[i_row, :] = 1 - self.siamese.inference_step(
                    q_mert_embedding[i_row : i_row + 1]
                    .repeat(n2, 1, 1)
                    .to(self.siamese.device),
                    r_mert_embedding.to(self.siamese.device),
                )

            # ==================  Decision matrix ========================
            proportion_thres = 0.2
            decision_thres = 0.5
            min_hits = 1
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
                decision = 1
            else:
                decision = 0

            # ================== Score ==================
            score = float(similarity_matrix.mean().item())

            # ================== Final ==================
            results.append(
                MelodySimResult(
                    score=score,
                    decision=decision,
                    decision_matrix=decision_matrix,
                    similarity_matrix=similarity_matrix,
                )
            )

        return results

    def scores(self, query: str | Path, references: list[str | Path]) -> list[float]:
        return [r.score for r in self.compare(query, references)]


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


if __name__ == "__main__":
    comparator = MelodySimCompare()
    references = [Path(path) for path in Path("data/original").glob("*.wav")]
    scores = comparator.scores(
        query="data/comparison/_DANCE_ 싸이 _PSY_ - 챔피언.wav",
        references=references,
    )

    pairs = list(zip(references, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    for ref, score in pairs:
        print(f"{score:0.6f}  {ref}")

    best_reference, best_score = pairs[0]
    print("\nBest option is:")
    print(f"{best_reference}: {best_score:0.6f}")
