"""Focused unit tests for the CNN large embedder wrapper."""

from __future__ import annotations

import numpy as np

from src.models.cnn_large import CNNLargeEmbedder


def test_cnn_large_extract_pools_one_clip_embedding() -> None:
    """extract() should return a 1D pooled embedding for CNN large."""
    extractor = CNNLargeEmbedder.__new__(CNNLargeEmbedder)
    extractor._embedding_dim = 2048

    frame_embeddings = np.ones((1, 2048), dtype=np.float32)
    extractor.extract_frames = lambda wav_path: frame_embeddings

    embedding = extractor.extract("unused.wav")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (2048,)
    assert np.allclose(embedding, 1.0)
