"""Basic similarity and ranking helpers for retrieval."""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def pairwise_cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and candidate matrix."""
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)

    if query_norm == 0.0:
        return np.zeros(candidates.shape[0], dtype=float)

    dots = candidates @ query
    denom = candidate_norms * query_norm

    scores = np.zeros(candidates.shape[0], dtype=float)
    valid = denom > 0.0
    scores[valid] = dots[valid] / denom[valid]
    return scores


def get_top_k_indices(
    scores: np.ndarray,
    k: int,
    exclude_index: int | None = None,
) -> np.ndarray:
    """Return indices of the top-k highest scores in descending order."""
    if k <= 0 or scores.size == 0:
        return np.array([], dtype=int)

    ranked_indices = np.arange(scores.shape[0])

    if exclude_index is not None:
        ranked_indices = ranked_indices[ranked_indices != exclude_index]

    if ranked_indices.size == 0:
        return np.array([], dtype=int)

    ranked_scores = scores[ranked_indices]
    order = np.argsort(-ranked_scores)
    top_k = min(k, ranked_indices.size)
    return ranked_indices[order[:top_k]]


def get_top_k_results(
    scores: np.ndarray,
    k: int,
    exclude_index: int | None = None,
) -> list[tuple[int, float]]:
    """Return top-k retrieval results as (index, score) pairs."""
    top_indices = get_top_k_indices(scores, k, exclude_index=exclude_index)
    return [(int(idx), float(scores[idx])) for idx in top_indices]


