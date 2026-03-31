#!/usr/bin/env python3
"""Run paired t-tests over per-query NDCG@5 rows exported for CIKM."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import ttest_rel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.io_utils import load_csv, save_csv


def _group_rows_by_proxy_and_model(
    rows: list[dict[str, str]],
    ndcg_column: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Group rows as proxy_task -> model_name -> query_id -> ndcg value."""
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for row in rows:
        proxy_task = row["proxy_task"]
        model_name = row["model_name"]
        query_id = row["query_id"]
        grouped[proxy_task][model_name][query_id] = float(row[ndcg_column])

    return grouped


def _build_pairwise_results(
    grouped_rows: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, float | int | str]]:
    """Compute paired t-tests for every model pair within each proxy task."""
    results: list[dict[str, float | int | str]] = []

    for proxy_task, by_model in sorted(grouped_rows.items()):
        model_names = sorted(by_model.keys())

        for index, model_a in enumerate(model_names):
            for model_b in model_names[index + 1 :]:
                query_ids = sorted(set(by_model[model_a]) & set(by_model[model_b]))

                if len(query_ids) < 2:
                    continue

                values_a = np.asarray([by_model[model_a][query_id] for query_id in query_ids])
                values_b = np.asarray([by_model[model_b][query_id] for query_id in query_ids])
                t_stat, p_value = ttest_rel(values_a, values_b)

                results.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "proxy_task": proxy_task,
                        "num_queries": int(len(query_ids)),
                        "mean_a": float(np.mean(values_a)),
                        "mean_b": float(np.mean(values_b)),
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                    }
                )

    return results


def run_paired_ttests(
    input_csv_path: str,
    output_csv_path: str,
    ndcg_column: str,
) -> list[dict[str, float | int | str]]:
    """Load per-query metrics, align rows by query_id, and save paired t-tests."""
    rows = load_csv(Path(input_csv_path))
    grouped_rows = _group_rows_by_proxy_and_model(rows, ndcg_column=ndcg_column)
    results = _build_pairwise_results(grouped_rows)
    save_csv(results, Path(output_csv_path))
    print(f"Saved {len(results)} paired t-test rows to: {output_csv_path}")
    return results


def main() -> None:
    """Run paired t-tests from the command line."""
    parser = argparse.ArgumentParser(
        description="Run paired t-tests on per-query NDCG rows for CIKM."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/output/evaluation/cikm/per_query_metrics.csv",
        help="CSV exported by export_cikm_per_query_metrics.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/output/evaluation/cikm/paired_ttests.csv",
        help="Output CSV path for paired t-test results.",
    )
    parser.add_argument(
        "--ndcg-column",
        type=str,
        default="ndcg_at_5",
        help="Per-query NDCG column to test.",
    )
    args = parser.parse_args()

    run_paired_ttests(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        ndcg_column=args.ndcg_column,
    )


if __name__ == "__main__":
    main()
