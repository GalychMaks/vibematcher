import numpy as np
import torch
import torch.nn.functional as F

def compute_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between two sets of embeddings.
    
    :param embeddings1: [num_chunks1, feature_dim] 
    :param embeddings2: [num_chunks2, feature_dim]
    :return: similarity_matrix [num_chunks1, num_chunks2], values in [0,1]
    """
    # Convert embeddings to torch tensors
    emb1 = torch.from_numpy(embeddings1).float()
    emb2 = torch.from_numpy(embeddings2).float()

    # Normalize embeddings along feature dimension
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)

    # Compute cosine similarity matrix: [num_chunks1 x num_chunks2]
    similarity_matrix = torch.matmul(emb1, emb2.T)

    print(similarity_matrix)

    # Convert similarity matrix to numpy
    return similarity_matrix.cpu().numpy()


def build_decision_matrix(similarity_matrix: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    """
    Build a binary decision matrix from the similarity matrix.
    
    :param similarity_matrix: [num_chunks1, num_chunks2], values in [0,1]
    :param threshold: threshold to consider chunks as similar
    :return: decision_matrix [num_chunks1, num_chunks2], 0/1
    """
    # Apply threshold to convert similarities to binary decisions
    decision_matrix = (similarity_matrix >= threshold).astype(int)
    print(decision_matrix)
    return decision_matrix


def aggregate_similarity_score(
    decision_matrix: np.ndarray,
    min_diagonal_length: int = 2,
) -> float:
    """
    Aggregate similarity score based on diagonal matches in the decision matrix.

    A diagonal represents time-aligned matching chunks between two songs.
    The score is based on the longest diagonal of consecutive matches.

    :param decision_matrix: [num_chunks1, num_chunks2], values 0/1
    :param min_diagonal_length: minimum length of a diagonal run to be considered meaningful
    :return: similarity score in range [0, 1]
    """
    num_rows, num_cols = decision_matrix.shape
    # max_possible_length = min(num_rows, num_cols)

    longest_diagonal = 0

    # Iterate over all possible diagonals (top-left to bottom-right)
    # Diagonals are indexed by (row - col)
    for offset in range(-num_cols + 1, num_rows):
        diag = np.diagonal(decision_matrix, offset=offset)

        # Find longest consecutive run of 1s in this diagonal
        current_run = 0
        for value in diag:
            if value == 1:
                current_run += 1
                longest_diagonal = max(longest_diagonal, current_run)
            else:
                current_run = 0

    # If no diagonal of sufficient length exists, similarity is zero
    if longest_diagonal < min_diagonal_length:
        return 0.0

    # Normalize score to [0, 1]
    score = longest_diagonal

    return float(score)
