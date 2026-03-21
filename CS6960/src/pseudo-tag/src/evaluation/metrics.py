"""Basic ranking metrics for retrieval evaluation."""

import numpy as np


def precision_at_k(relevance, k: int) -> float:
    """Compute precision among the top-k items."""
    if k <= 0:
        return 0.0

    rel = np.asarray(relevance)
    top_k = rel[:k]

    if top_k.size == 0:
        return 0.0

    return float(np.sum(top_k > 0) / top_k.size)


def recall_at_k(relevance, k: int) -> float:
    """Compute recall among the top-k items."""
    if k <= 0:
        return 0.0

    rel = np.asarray(relevance)
    total_relevant = np.sum(rel > 0)

    if total_relevant == 0:
        return 0.0

    top_k = rel[:k]
    return float(np.sum(top_k > 0) / total_relevant)


def f1_at_k(relevance, k: int) -> float:
    """Compute F1 score from precision@k and recall@k."""
    precision = precision_at_k(relevance, k)
    recall = recall_at_k(relevance, k)

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def dcg_at_k(relevance, k: int) -> float:
    """Compute discounted cumulative gain at k."""
    if k <= 0:
        return 0.0

    rel = np.asarray(relevance, dtype=float)[:k]

    if rel.size == 0:
        return 0.0

    ranks = np.arange(1, rel.size + 1)
    discounts = np.log2(ranks + 1)
    gains = (2**rel - 1) / discounts
    return float(np.sum(gains))


def ndcg_at_k(relevance, k: int) -> float:
    """Compute normalized discounted cumulative gain at k."""
    if k <= 0:
        return 0.0

    rel = np.asarray(relevance, dtype=float)
    dcg = dcg_at_k(rel, k)
    ideal_dcg = dcg_at_k(np.sort(rel)[::-1], k)

    if ideal_dcg == 0.0:
        return 0.0

    return float(dcg / ideal_dcg)


def mean_metric(values) -> float:
    """Return the mean of metric values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))
