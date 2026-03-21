"""Proxy-task runners:
- run_proxy_task_1
- run_proxy_task_2
- run_all_queries_proxy_task_1
- run_all_queries_proxy_task_2
- summarize_proxy_results
"""

import numpy as np

from src.evaluation.metrics import f1_at_k, mean_metric, ndcg_at_k, precision_at_k, recall_at_k
from src.evaluation.relevance import (
    build_relevance_list,
    composer_relevance_fn,
    tag_jaccard_relevance_fn,
    tag_overlap_relevance_fn,
)
from src.retrieval.similarity import get_top_k_indices, pairwise_cosine_similarity


def run_proxy_task_1(
    embeddings: np.ndarray,
    tracks: list[dict],
    query_index: int,
    k: int,
) -> dict:
    """Run composer-based retrieval for one query."""
    query_embedding = embeddings[query_index]
    scores = pairwise_cosine_similarity(query_embedding, embeddings)
    top_k_indices = get_top_k_indices(scores, k, exclude_index=query_index)

    query_track = tracks[query_index]
    candidate_tracks = [tracks[idx] for idx in top_k_indices]
    relevance = build_relevance_list(query_track, candidate_tracks, composer_relevance_fn)

    return {
        "query_index": query_index,
        "top_k_indices": top_k_indices.tolist(),
        "relevance": relevance.tolist(),
        "precision_at_k": precision_at_k(relevance, k),
        "recall_at_k": recall_at_k(relevance, k),
        "f1_at_k": f1_at_k(relevance, k),
        "ndcg_at_k": ndcg_at_k(relevance, k),
    }


def run_proxy_task_2(
    embeddings: np.ndarray,
    tag_vectors: np.ndarray,
    query_index: int,
    k: int,
    graded: bool = False,
) -> dict:
    """Run tag-based retrieval for one query."""
    query_embedding = embeddings[query_index]
    scores = pairwise_cosine_similarity(query_embedding, embeddings)
    top_k_indices = get_top_k_indices(scores, k, exclude_index=query_index)

    query_tags = tag_vectors[query_index]
    candidate_tags = [tag_vectors[idx] for idx in top_k_indices]

    binary_relevance = build_relevance_list(query_tags, candidate_tags, tag_overlap_relevance_fn)

    if graded:
        ndcg_relevance = build_relevance_list(query_tags, candidate_tags, tag_jaccard_relevance_fn)
    else:
        ndcg_relevance = binary_relevance

    return {
        "query_index": query_index,
        "top_k_indices": top_k_indices.tolist(),
        "relevance": ndcg_relevance.tolist(),
        "precision_at_k": precision_at_k(binary_relevance, k),
        "recall_at_k": recall_at_k(binary_relevance, k),
        "f1_at_k": f1_at_k(binary_relevance, k),
        "ndcg_at_k": ndcg_at_k(ndcg_relevance, k),
    }

def run_all_queries_proxy_task_1(
    embeddings: np.ndarray,
    tracks: list[dict],
    k: int,
) -> list[dict]:
    """Run composer-based retrieval for all queries."""
    return [
        run_proxy_task_1(embeddings=embeddings, tracks=tracks, query_index=i, k=k)
        for i in range(len(tracks))
    ]


def run_all_queries_proxy_task_2(
    embeddings: np.ndarray,
    tag_vectors: np.ndarray,
    k: int,
    graded: bool = False,
) -> list[dict]:
    """Run tag-based retrieval for all queries."""
    return [
        run_proxy_task_2(
            embeddings=embeddings,
            tag_vectors=tag_vectors,
            query_index=i,
            k=k,
            graded=graded,
        )
        for i in range(len(tag_vectors))
    ]


def summarize_proxy_results(results: list[dict]) -> dict:
    """Compute mean retrieval metrics over many query results."""
    return {
        "precision_at_k": mean_metric([result["precision_at_k"] for result in results]),
        "recall_at_k": mean_metric([result["recall_at_k"] for result in results]),
        "f1_at_k": mean_metric([result["f1_at_k"] for result in results]),
        "ndcg_at_k": mean_metric([result["ndcg_at_k"] for result in results]),
    }
