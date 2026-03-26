"""Focused tests for retrieval query embedding behavior."""

from __future__ import annotations

from pathlib import Path

from src.retrieval import retrieve_top_k


def test_retrieve_top_k_reuses_precomputed_query_embedding(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Queries already in the corpus should reuse stored embeddings instead of re-embedding."""
    query_path = tmp_path / "track_1_example.wav"
    query_path.write_bytes(b"")

    corpus_embeddings = retrieve_top_k.np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=retrieve_top_k.np.float32,
    )
    metadata = [
        {"path": str(query_path), "file_id": "track_1_example"},
        {"path": str(tmp_path / "track_2.wav"), "file_id": "track_2"},
        {"path": str(tmp_path / "track_3.wav"), "file_id": "track_3"},
    ]

    monkeypatch.setattr(
        retrieve_top_k,
        "load_corpus_artifacts",
        lambda embeddings_path, metadata_path: (corpus_embeddings, metadata),
    )

    def _fail_create_extractor(model_name: str):
        raise AssertionError("create_extractor should not be called for corpus queries")

    monkeypatch.setattr(retrieve_top_k, "create_extractor", _fail_create_extractor)

    results = retrieve_top_k.retrieve_top_k(
        query_wav_path=str(query_path),
        embeddings_path="unused.npy",
        metadata_path="unused.json",
        model_name="cnn_small",
        top_k=2,
        exclude_self=True,
    )

    assert [item["file_id"] for item in results] == ["track_2", "track_3"]
