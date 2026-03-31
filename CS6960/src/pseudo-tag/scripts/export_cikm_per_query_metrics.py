#!/usr/bin/env python3
"""Export per-query retrieval metrics for CIKM significance testing.

This script is additive and does not change the existing RecSys/ISMIR batch
evaluation pipeline or its summary outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.build_relevance import (
    build_relevance,
    count_total_relevant_items,
    extract_track_id,
    load_label_data,
)
from src.evaluation.metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.evaluation.run_batch_eval import collect_wav_paths
from src.io_utils import save_csv
from src.retrieval.retrieve_top_k import retrieve_top_k

DEFAULT_MODELS = [
    "transformer_medium",
    "transformer_large",
    "cnn_small",
    "cnn_medium",
    "cnn_large",
]


def compute_query_metrics(
    query_wav_path: str,
    embeddings_path: str,
    metadata_path: str,
    label_data: list[dict[str, Any]],
    model_name: str,
    top_k: int,
    relevance_strategy: str,
) -> dict[str, Any] | None:
    """Compute per-query metrics using the existing retrieval/evaluation logic."""
    query_id = extract_track_id(query_wav_path) or Path(query_wav_path).stem

    retrieved_items = retrieve_top_k(
        query_wav_path=query_wav_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        model_name=model_name,
        top_k=top_k,
        exclude_self=True,
    )

    relevance = build_relevance(
        query_key=query_id,
        retrieved_items=retrieved_items,
        label_data=label_data,
        strategy=relevance_strategy,
    )
    total_relevant = count_total_relevant_items(
        query_key=query_id,
        label_data=label_data,
        strategy=relevance_strategy,
    )

    if total_relevant == 0:
        return None

    relevant_in_top_k = sum(1 for value in relevance[:top_k] if value > 0)
    precision = precision_at_k(relevance, top_k)
    recall = recall_at_k(relevance, top_k, total_relevant=total_relevant)
    f1 = f1_at_k(relevance, top_k, total_relevant=total_relevant)
    ndcg = ndcg_at_k(relevance, top_k)

    return {
        "query_id": str(query_id),
        "query_path": str(query_wav_path),
        "model_name": model_name,
        "proxy_task": relevance_strategy,
        f"precision_at_{top_k}": float(precision),
        f"recall_at_{top_k}": float(recall),
        f"f1_at_{top_k}": float(f1),
        f"ndcg_at_{top_k}": float(ndcg),
        f"hit_at_{top_k}": int(relevant_in_top_k > 0),
        "relevant_in_top_k": int(relevant_in_top_k),
        "total_relevant": int(total_relevant),
    }


def resolve_model_names(model_names: list[str] | None) -> list[str]:
    """Use default model list when no explicit model names are provided."""
    return model_names if model_names else list(DEFAULT_MODELS)


def build_embeddings_and_metadata_paths(
    model_name: str,
    embeddings_root: str,
) -> tuple[str, str]:
    """Build the standard embeddings and metadata paths for one model."""
    root = Path(embeddings_root) / model_name
    embeddings_path = root / f"{model_name}_embeddings.npy"
    metadata_path = root / f"{model_name}_metadata.json"
    return str(embeddings_path), str(metadata_path)


def export_per_query_metrics(
    wav_dir: str,
    labels_path: str,
    output_csv_path: str,
    model_names: list[str],
    relevance_strategies: list[str],
    embeddings_root: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run additive CIKM per-query evaluation and save rows to CSV."""
    wav_paths = collect_wav_paths(wav_dir)
    label_data = load_label_data(labels_path)
    rows: list[dict[str, Any]] = []

    for relevance_strategy in relevance_strategies:
        for model_name in model_names:
            embeddings_path, metadata_path = build_embeddings_and_metadata_paths(
                model_name=model_name,
                embeddings_root=embeddings_root,
            )
            print(
                f"Running per-query export for model={model_name} "
                f"proxy_task={relevance_strategy}"
            )

            for wav_path in wav_paths:
                try:
                    row = compute_query_metrics(
                        query_wav_path=str(wav_path),
                        embeddings_path=embeddings_path,
                        metadata_path=metadata_path,
                        label_data=label_data,
                        model_name=model_name,
                        top_k=top_k,
                        relevance_strategy=relevance_strategy,
                    )
                except ValueError as exc:
                    print(f"Skipping query={wav_path.name} model={model_name}: {exc}")
                    continue

                if row is None:
                    print(
                        f"Skipping query={wav_path.name} model={model_name} "
                        "because total_relevant=0"
                    )
                    continue

                rows.append(row)

    save_csv(rows, Path(output_csv_path))
    print(f"Saved {len(rows)} per-query rows to: {output_csv_path}")
    return rows


def main() -> None:
    """Run per-query export from the command line."""
    parser = argparse.ArgumentParser(
        description="Export per-query retrieval metrics for CIKM significance testing."
    )
    parser.add_argument(
        "--wav-dir",
        type=str,
        default="data/wav",
        help="Directory containing query WAV files.",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="data/output/labels/pseudo_labels.csv",
        help="Path to label data CSV or JSON.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=str,
        default="data/output/embeddings",
        help="Root directory containing per-model embeddings folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/output/evaluation/cikm/per_query_metrics.csv",
        help="Output CSV path for per-query metrics.",
    )
    parser.add_argument(
        "--model-name",
        dest="model_names",
        action="append",
        choices=DEFAULT_MODELS,
        help="Model name to export. Repeat to include multiple models. Defaults to all models.",
    )
    parser.add_argument(
        "--relevance-strategy",
        dest="relevance_strategies",
        action="append",
        choices=["composer", "tag_overlap"],
        help="Proxy task to export. Repeat to include multiple tasks. Defaults to both.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k cutoff for metrics.")
    args = parser.parse_args()

    export_per_query_metrics(
        wav_dir=args.wav_dir,
        labels_path=args.labels_path,
        output_csv_path=args.output_csv,
        model_names=resolve_model_names(args.model_names),
        relevance_strategies=args.relevance_strategies or ["composer", "tag_overlap"],
        embeddings_root=args.embeddings_root,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
