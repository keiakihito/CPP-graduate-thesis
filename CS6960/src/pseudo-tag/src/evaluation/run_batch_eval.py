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
    debug: bool = False,
) -> dict[str, float]:
    """Run retrieval evaluation across all queries and average valid metrics."""
    wav_paths = collect_wav_paths(wav_dir)

    precision_key = f"precision@{top_k}"
    recall_key = f"recall@{top_k}"
    f1_key = f"f1@{top_k}"
    ndcg_key = f"ndcg@{top_k}"
    hit_key = f"hit@{top_k}"

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    ndcg_values: list[float] = []
    hit_values: list[float] = []
    ndcg_samples: list[tuple[str, float]] = []
    missing_label_queries = 0

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
            missing_label_queries += 1
            continue

        precision = metrics.get(precision_key)
        recall = metrics.get(recall_key)
        f1 = metrics.get(f1_key)
        ndcg = metrics.get(ndcg_key)
        hit = metrics.get(hit_key)

        if precision is None or recall is None or ndcg is None or f1 is None or hit is None:
            missing_label_queries += 1
            continue

        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        ndcg_values.append(float(ndcg))
        hit_values.append(float(hit))
        ndcg_samples.append((wav_path.name, float(ndcg)))

    total_queries = len(wav_paths)
    evaluated_queries = len(precision_values)
    queries_with_hit = int(sum(hit_values))
    average_metrics = {
        precision_key: sum(precision_values) / evaluated_queries if evaluated_queries else 0.0,
        recall_key: sum(recall_values) / evaluated_queries if evaluated_queries else 0.0,
        f1_key: sum(f1_values) / evaluated_queries if evaluated_queries else 0.0,
        ndcg_key: sum(ndcg_values) / evaluated_queries if evaluated_queries else 0.0,
        hit_key: sum(hit_values) / evaluated_queries if evaluated_queries else 0.0,
    }
    raw_sums = {
        precision_key: sum(precision_values),
        recall_key: sum(recall_values),
        f1_key: sum(f1_values),
        ndcg_key: sum(ndcg_values),
        hit_key: sum(hit_values),
    }

    if average_metrics[ndcg_key] > average_metrics[hit_key] + 1e-12:
        raise AssertionError(
            f"Sanity check failed: {ndcg_key}={average_metrics[ndcg_key]:.6f} "
            f"> {hit_key}={average_metrics[hit_key]:.6f}"
        )
    for metric_name, metric_value in average_metrics.items():
        if not 0.0 <= metric_value <= 1.0:
            raise AssertionError(
                f"Range check failed: {metric_name}={metric_value:.6f} is outside [0, 1]"
            )
    if average_metrics[precision_key] == 0.0 and average_metrics[recall_key] == 0.0:
        if average_metrics[f1_key] != 0.0:
            raise AssertionError(
                f"F1 sanity check failed: {f1_key}={average_metrics[f1_key]:.6f} "
                f"while {precision_key}=0 and {recall_key}=0"
            )

    print("\n=== Batch Evaluation Summary ===")
    print(f"total_queries: {total_queries}")
    print(f"evaluated_queries: {evaluated_queries}")
    print(f"missing_label_queries: {missing_label_queries}")
    print(f"queries_with_hit@{top_k}: {queries_with_hit}")
    for name, value in average_metrics.items():
        print(f"{name}: {value:.6f}")

    if debug:
        print("\n=== Debug ===")
        print(f"normalization_denominator: {evaluated_queries}")
        for metric_name in [precision_key, recall_key, f1_key, ndcg_key, hit_key]:
            print(
                f"{metric_name}_raw_sum: {raw_sums[metric_name]:.6f}"
            )
            print(
                f"{metric_name}_normalized_value: {average_metrics[metric_name]:.6f}"
            )
        print(f"sample_ndcg_values: {ndcg_samples[:10]}")

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional query-count and per-query NDCG debug information.",
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
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
