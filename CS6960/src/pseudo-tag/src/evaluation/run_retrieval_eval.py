"""Simple retrieval evaluation runner for thesis MVP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.build_relevance import (
    build_relevance,
    extract_track_id,
    load_label_data,
)
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, f1_at_k
from src.retrieval.retrieve_top_k import retrieve_top_k


def run_single_query_eval(
    query_wav_path: str,
    embeddings_path: str,
    metadata_path: str,
    label_data_path: str,
    model_name: str,
    top_k: int = 5,
    relevance_strategy: str = "composer",
) -> dict[str, float]:
    """Run retrieval, build relevance labels, and compute ranking metrics."""
    query_key = extract_track_id(query_wav_path) or Path(query_wav_path).stem

    retrieved_items = retrieve_top_k(
        query_wav_path=query_wav_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        model_name=model_name,
        top_k=top_k,
        exclude_self=True,
    )

    label_data = load_label_data(label_data_path)
    relevance = build_relevance(
        query_key=query_key,
        retrieved_items=retrieved_items,
        label_data=label_data,
        strategy=relevance_strategy,
    )

    if sum(relevance) == 0:
        print("No relevant items for this query. Skipping evaluation.")
        return {
            "precision@k": None,
            "recall@k": None,
            "f1@k": None,
            "ndcg@k": None,
        }

    metrics = {
        f"precision@{top_k}": precision_at_k(relevance, top_k),
        f"recall@{top_k}": recall_at_k(relevance, top_k),
        f"f1@{top_k}": f1_at_k(relevance, top_k),
        f"ndcg@{top_k}": ndcg_at_k(relevance, top_k),
    }

    print("Top-k results:")
    for rank, item in enumerate(retrieved_items, start=1):
        print(
            f"{rank}. score={item['score']:.6f} "
            f"path={item.get('path', '')} "
            f"file_id={item.get('file_id', '')}"
        )

    print(f"Relevance: {relevance}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    return metrics


def main() -> None:
    """Run single-query retrieval evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for one query.")
    parser.add_argument("query_wav_path", type=str, help="Query WAV file.")
    parser.add_argument("embeddings_path", type=str, help="Path to corpus embeddings .npy.")
    parser.add_argument("metadata_path", type=str, help="Path to corpus metadata .json.")
    parser.add_argument("label_data_path", type=str, help="Path to label data .csv or .json.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn_small",
        choices=["cnn_small", "transformer_small"],
        help="Embedding backend to use for retrieval.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to evaluate.")
    parser.add_argument(
        "--relevance-strategy",
        type=str,
        default="composer",
        choices=["composer", "tag_overlap"],
        help="Binary relevance strategy to use.",
    )
    args = parser.parse_args()

    run_single_query_eval(
        query_wav_path=args.query_wav_path,
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        label_data_path=args.label_data_path,
        model_name=args.model_name,
        top_k=args.top_k,
        relevance_strategy=args.relevance_strategy,
    )


if __name__ == "__main__":
    main()
