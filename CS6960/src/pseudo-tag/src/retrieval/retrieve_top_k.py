"""Simple top-k retrieval for thesis MVP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.io_utils import load_json
from src.models.cnn_large import CNNLargeEmbedder
from src.models.cnn_medium import CNNMediumEmbedder
from src.models.cnn_small import CNNSmallEmbedder
from src.models.transformer_large import TransformerLargeEmbedder
from src.models.transformer_medium import TransformerMediumEmbedder
from src.retrieval.similarity import pairwise_cosine_similarity


def create_extractor(model_name: str) -> Any:
    """Create an embedding extractor from a simple model name."""
    if model_name == "cnn_small":
        return CNNSmallEmbedder()
    if model_name == "cnn_medium":
        return CNNMediumEmbedder()
    if model_name == "cnn_large":
        return CNNLargeEmbedder()
    if model_name == "transformer_medium":
        return TransformerMediumEmbedder()
    if model_name == "transformer_large":
        return TransformerLargeEmbedder()
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_corpus_artifacts(
    embeddings_path: str,
    metadata_path: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load saved corpus embeddings and matching metadata."""
    embeddings = np.load(embeddings_path)
    metadata = load_json(Path(metadata_path))

    if not isinstance(metadata, list):
        raise ValueError("Metadata must be a list of dictionaries.")
    if not all(isinstance(item, dict) for item in metadata):
        raise ValueError("Each metadata item must be a dictionary.")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must have shape (N, D), got {embeddings.shape}")
    if embeddings.shape[0] != len(metadata):
        raise ValueError("Embeddings row count does not match metadata length.")

    return np.asarray(embeddings, dtype=np.float32), metadata


def _normalize_path(path_str: str) -> str:
    """Normalize a path string for simple equality checks."""
    return str(Path(path_str).expanduser().resolve(strict=False))


def _get_top_k_indices(
    scores: np.ndarray,
    top_k: int,
    metadata: list[dict[str, Any]],
    query_wav_path: str,
    exclude_self: bool,
) -> np.ndarray:
    """Return deterministically ranked top-k indices."""
    if top_k <= 0 or scores.size == 0:
        return np.array([], dtype=int)

    candidate_indices = np.arange(scores.shape[0], dtype=int)

    if exclude_self:
        query_path = _normalize_path(query_wav_path)
        keep_mask = np.array(
            [
                _normalize_path(str(item.get("path", ""))) != query_path
                for item in metadata
            ],
            dtype=bool,
        )
        candidate_indices = candidate_indices[keep_mask]

    if candidate_indices.size == 0:
        return np.array([], dtype=int)

    candidate_scores = scores[candidate_indices]
    order = np.lexsort((candidate_indices, -candidate_scores))
    return candidate_indices[order[: min(top_k, candidate_indices.size)]]


def retrieve_top_k(
    query_wav_path: str,
    embeddings_path: str,
    metadata_path: str,
    model_name: str,
    top_k: int = 5,
    exclude_self: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve the top-k most similar corpus items for one query WAV file."""
    extractor = create_extractor(model_name)
    corpus_embeddings, metadata = load_corpus_artifacts(embeddings_path, metadata_path)
    query_embedding = np.asarray(extractor.extract(query_wav_path), dtype=np.float32)

    if query_embedding.ndim != 1:
        raise ValueError(f"Query embedding must be 1D, got shape {query_embedding.shape}")
    if corpus_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Query embedding dimension does not match corpus embedding dimension: "
            f"{query_embedding.shape[0]} vs {corpus_embeddings.shape[1]}"
        )

    scores = pairwise_cosine_similarity(query_embedding, corpus_embeddings)
    top_indices = _get_top_k_indices(
        scores=scores,
        top_k=top_k,
        metadata=metadata,
        query_wav_path=query_wav_path,
        exclude_self=exclude_self,
    )

    results: list[dict[str, Any]] = []
    for index in top_indices:
        item = dict(metadata[int(index)])
        item["score"] = float(scores[int(index)])
        results.append(item)

    return results


def main() -> None:
    """Run simple top-k retrieval from the command line."""
    parser = argparse.ArgumentParser(description="Retrieve top-k similar WAV files.")
    parser.add_argument("query_wav_path", type=str, help="Query WAV file.")
    parser.add_argument("embeddings_path", type=str, help="Path to corpus embeddings .npy.")
    parser.add_argument("metadata_path", type=str, help="Path to corpus metadata .json.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn_small",
        choices=[
            "cnn_small",
            "cnn_medium",
            "cnn_large",
            "transformer_medium",
            "transformer_large",
        ],
        help="Embedding backend to use for the query.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include the exact same file path if it appears in the corpus.",
    )
    args = parser.parse_args()

    results = retrieve_top_k(
        query_wav_path=args.query_wav_path,
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        top_k=args.top_k,
        exclude_self=not args.include_self,
    )

    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. score={item['score']:.6f} "
            f"path={item.get('path', '')} "
            f"file_id={item.get('file_id', '')}"
        )


if __name__ == "__main__":
    main()
