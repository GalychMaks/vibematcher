import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes differ: {a.shape} vs {b.shape}")

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    # Avoid division by zero (e.g., silent audio producing all-zeros embedding)
    if na < eps or nb < eps:
        return 0.0

    sim = float(np.dot(a, b) / (na * nb))
    # Numerical safety
    return float(np.clip(sim, -1.0, 1.0))
