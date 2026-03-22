"""Batch retrieval evaluation for thesis MVP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.run_retrieval_eval import run_single_query_eval


def collect_wav_paths(input_dir: str) -> list[Path]:
    """Recursively collect WAV files in deterministic order."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not root.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    return sorted(path for path in root.rglob("*.wav") if path.is_file())


def run_batch_eval(
    wav_dir: str,
    embeddings_path: str,
    metadata_path: str,
    label_data_path: str,
    model_name: str,
    top_k: int = 5,
    relevance_strategy: str = "composer",
) -> dict[str, float]:
    """Run retrieval evaluation across all queries and average valid metrics."""
    wav_paths = collect_wav_paths(wav_dir)

    precision_key = f"precision@{top_k}"
    recall_key = f"recall@{top_k}"
    f1_key = f"f1@{top_k}"
    ndcg_key = f"ndcg@{top_k}"

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    ndcg_values: list[float] = []
    skipped_queries = 0

    for wav_path in wav_paths:
        print(f"\n=== Query: {wav_path.name} ===")
        try:
            metrics = run_single_query_eval(
                query_wav_path=str(wav_path),
                embeddings_path=embeddings_path,
                metadata_path=metadata_path,
                label_data_path=label_data_path,
                model_name=model_name,
                top_k=top_k,
                relevance_strategy=relevance_strategy,
            )
        except ValueError as exc:
            print(f"Skipping query due to missing label data: {exc}")
            skipped_queries += 1
            continue

        if metrics is None:
            skipped_queries += 1
            continue

        precision = metrics.get(precision_key)
        recall = metrics.get(recall_key)
        f1 = metrics.get(f1_key)
        ndcg = metrics.get(ndcg_key)

        if precision is None or recall is None or ndcg is None or f1 is None:
            skipped_queries += 1
            continue

        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        ndcg_values.append(float(ndcg))

    valid_queries = len(precision_values)
    average_metrics = {
        precision_key: sum(precision_values) / valid_queries if valid_queries else 0.0,
        recall_key: sum(recall_values) / valid_queries if valid_queries else 0.0,
        f1_key: sum(f1_values) / valid_queries if valid_queries else 0.0,
        ndcg_key: sum(ndcg_values) / valid_queries if valid_queries else 0.0,
    }

    print("\n=== Batch Evaluation Summary ===")
    print(f"valid_queries: {valid_queries}")
    print(f"skipped_queries: {skipped_queries}")
    for name, value in average_metrics.items():
        print(f"{name}: {value:.6f}")

    return average_metrics


def main() -> None:
    """Run batch retrieval evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for all queries.")
    parser.add_argument("wav_dir", type=str, help="Directory containing query WAV files.")
    parser.add_argument("embeddings_path", type=str, help="Path to corpus embeddings .npy.")
    parser.add_argument("metadata_path", type=str, help="Path to corpus metadata .json.")
    parser.add_argument("label_data_path", type=str, help="Path to label data .csv or .json.")
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

    run_batch_eval(
        wav_dir=args.wav_dir,
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        label_data_path=args.label_data_path,
        model_name=args.model_name,
        top_k=args.top_k,
        relevance_strategy=args.relevance_strategy,
    )


if __name__ == "__main__":
    main()
