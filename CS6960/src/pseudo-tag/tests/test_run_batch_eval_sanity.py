"""Regression tests for batch-evaluation metric averaging."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import run_batch_eval


def test_run_batch_eval_counts_zero_relevance_queries_as_zero(monkeypatch) -> None:
    """Queries with no relevant items should contribute zero, not be skipped."""
    monkeypatch.setattr(
        run_batch_eval,
        "collect_wav_paths",
        lambda wav_dir: [
            run_batch_eval.Path("track_1.wav"),
            run_batch_eval.Path("track_2.wav"),
            run_batch_eval.Path("track_3.wav"),
        ],
    )

    per_query = {
        "track_1.wav": {
            "precision@5": 0.2,
            "recall@5": 0.5,
            "f1@5": 0.285714,
            "ndcg@5": 1.0,
            "hit@5": 1.0,
        },
        "track_2.wav": {
            "precision@5": 0.0,
            "recall@5": 0.0,
            "f1@5": 0.0,
            "ndcg@5": 0.0,
            "hit@5": 0.0,
        },
        "track_3.wav": {
            "precision@5": 0.0,
            "recall@5": 0.0,
            "f1@5": 0.0,
            "ndcg@5": 0.0,
            "hit@5": 0.0,
        },
    }

    monkeypatch.setattr(
        run_batch_eval,
        "run_single_query_eval",
        lambda query_wav_path, **kwargs: per_query[run_batch_eval.Path(query_wav_path).name],
    )

    metrics = run_batch_eval.run_batch_eval(
        wav_dir="unused",
        embeddings_path="unused",
        metadata_path="unused",
        label_data_path="unused",
        model_name="cnn_small",
        top_k=5,
        relevance_strategy="composer",
        debug=True,
    )

    assert metrics["ndcg@5"] == 1.0 / 3.0
    assert metrics["hit@5"] == 1.0 / 3.0
