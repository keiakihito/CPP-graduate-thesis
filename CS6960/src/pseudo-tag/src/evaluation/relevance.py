"""Helpers for converting retrieval results into relevance scores."""

import numpy as np

from src.label_utils import compute_tag_overlap, compute_tag_jaccard, extract_composer


def build_relevance_list(query_item, candidate_items, relevance_fn) -> np.ndarray:
    """Apply a relevance function to each candidate item."""
    scores = [relevance_fn(query_item, candidate_item) for candidate_item in candidate_items]
    return np.asarray(scores, dtype=float)


def composer_relevance_fn(query_track: dict, candidate_track: dict) -> int:
    """Return 1 when two tracks share the same composer."""
    return int(extract_composer(query_track) == extract_composer(candidate_track))


def tag_overlap_relevance_fn(query_tags: np.ndarray, candidate_tags: np.ndarray) -> int:
    """Return 1 when two tag vectors overlap on any active tag."""
    return int(compute_tag_overlap(query_tags, candidate_tags) > 0)


def tag_jaccard_relevance_fn(query_tags: np.ndarray, candidate_tags: np.ndarray) -> float:
    """Return Jaccard similarity between two tag vectors."""
    return float(compute_tag_jaccard(query_tags, candidate_tags))
