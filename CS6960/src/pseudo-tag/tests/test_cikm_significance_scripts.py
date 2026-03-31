"""Focused tests for additive CIKM significance scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(module_name: str, relative_path: str):
    """Load a script module directly from its file path."""
    root_dir = Path(__file__).resolve().parents[1]
    root_dir_str = str(root_dir)
    if root_dir_str not in sys.path:
        sys.path.insert(0, root_dir_str)

    module_path = root_dir / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_cikm_per_query_metrics = _load_module(
    "export_cikm_per_query_metrics",
    "scripts/export_cikm_per_query_metrics.py",
)
run_cikm_paired_ttests = _load_module(
    "run_cikm_paired_ttests",
    "scripts/run_cikm_paired_ttests.py",
)


def test_export_per_query_metrics_writes_required_columns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exporter should save the required per-query columns without touching old runners."""
    output_csv = tmp_path / "per_query_metrics.csv"

    monkeypatch.setattr(
        export_cikm_per_query_metrics,
        "collect_wav_paths",
        lambda wav_dir: [tmp_path / "track_1_a.wav", tmp_path / "track_2_b.wav"],
    )
    monkeypatch.setattr(
        export_cikm_per_query_metrics,
        "load_label_data",
        lambda labels_path: [{"track_id": "1"}, {"track_id": "2"}],
    )
    monkeypatch.setattr(
        export_cikm_per_query_metrics,
        "compute_query_metrics",
        lambda **kwargs: {
            "query_id": "1",
            "query_path": str(tmp_path / "track_1_a.wav"),
            "model_name": kwargs["model_name"],
            "proxy_task": kwargs["relevance_strategy"],
            "precision_at_5": 0.2,
            "recall_at_5": 0.5,
            "f1_at_5": 0.2857142857,
            "ndcg_at_5": 0.75,
            "hit_at_5": 1,
            "relevant_in_top_k": 1,
            "total_relevant": 2,
        },
    )

    rows = export_cikm_per_query_metrics.export_per_query_metrics(
        wav_dir="unused",
        labels_path="unused",
        output_csv_path=str(output_csv),
        model_names=["cnn_small"],
        relevance_strategies=["composer"],
        embeddings_root="unused",
        top_k=5,
    )

    assert rows
    assert output_csv.exists()
    row = rows[0]
    assert row["query_id"] == "1"
    assert row["model_name"] == "cnn_small"
    assert row["proxy_task"] == "composer"
    assert row["ndcg_at_5"] == 0.75
    assert row["hit_at_5"] == 1


def test_run_paired_ttests_aligns_on_query_id_and_writes_output(tmp_path: Path) -> None:
    """Paired t-tests should compare only aligned query rows."""
    input_csv = tmp_path / "per_query_metrics.csv"
    output_csv = tmp_path / "paired_ttests.csv"
    input_csv.write_text(
        "\n".join(
            [
                "query_id,model_name,proxy_task,ndcg_at_5,hit_at_5",
                "q1,cnn_small,composer,0.10,1",
                "q2,cnn_small,composer,0.20,1",
                "q3,cnn_small,composer,0.30,1",
                "q1,cnn_medium,composer,0.15,1",
                "q2,cnn_medium,composer,0.25,1",
                "q3,cnn_medium,composer,0.35,1",
                "q1,cnn_small,tag_overlap,0.50,1",
                "q2,cnn_small,tag_overlap,0.55,1",
                "q1,cnn_medium,tag_overlap,0.45,1",
                "q2,cnn_medium,tag_overlap,0.50,1",
            ]
        ),
        encoding="utf-8",
    )

    rows = run_cikm_paired_ttests.run_paired_ttests(
        input_csv_path=str(input_csv),
        output_csv_path=str(output_csv),
        ndcg_column="ndcg_at_5",
    )

    assert output_csv.exists()
    assert len(rows) == 2
    composer_row = next(row for row in rows if row["proxy_task"] == "composer")
    assert composer_row["model_a"] == "cnn_medium"
    assert composer_row["model_b"] == "cnn_small"
    assert composer_row["num_queries"] == 3
