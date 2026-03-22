"""Smoke tests for pretrained embedding extractor interfaces."""

from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np
import pytest


def _write_test_wav(
    path: Path,
    sample_rate: int = 16_000,
    duration_sec: float = 1.0,
) -> None:
    """Write a small mono PCM WAV file for extractor smoke tests."""
    num_samples = int(sample_rate * duration_sec)
    time_axis = np.arange(num_samples, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2.0 * math.pi * 440.0 * time_axis)
    pcm_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())


@pytest.fixture
def wav_path(tmp_path: Path) -> str:
    """Create a temporary valid WAV file for smoke tests."""
    path = tmp_path / "smoke_test.wav"
    _write_test_wav(path)
    return str(path)


@pytest.mark.smoke
@pytest.mark.integration
def test_cnn_small_embedder_smoke(wav_path: str) -> None:
    """CNNSmallEmbedder should return a valid 1D embedding."""
    pytest.importorskip("torch")
    pytest.importorskip("panns_inference")

    from src.models.cnn_small import CNNSmallEmbedder

    extractor = CNNSmallEmbedder()
    embedding = extractor.extract(wav_path)

    # Smoke test the shared extractor contract, not retrieval quality.
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.size > 0
    assert embedding.shape[0] == extractor.get_embedding_dim()
    assert np.all(np.isfinite(embedding))


@pytest.mark.smoke
@pytest.mark.integration
def test_transformer_small_embedder_smoke(wav_path: str) -> None:
    """TransformerSmallEmbedder should return a valid 1D embedding."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from src.models.transformer_small import TransformerSmallEmbedder

    extractor = TransformerSmallEmbedder()
    embedding = extractor.extract(wav_path)

    # Smoke test the shared extractor contract, not model accuracy.
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.size > 0
    assert embedding.shape[0] == extractor.get_embedding_dim()
    assert np.all(np.isfinite(embedding))


@pytest.mark.smoke
@pytest.mark.integration
def test_cnn_base_embedder_smoke(wav_path: str) -> None:
    """CNNBaseEmbedder should return a valid 1D embedding."""
    pytest.importorskip("torch")
    pytest.importorskip("panns_inference")

    from src.models.cnn_base import CNNBaseEmbedder

    extractor = CNNBaseEmbedder()
    embedding = extractor.extract(wav_path)

    # Smoke test the shared extractor contract, not retrieval quality.
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.size > 0
    assert embedding.shape[0] == extractor.get_embedding_dim()
    assert np.all(np.isfinite(embedding))


@pytest.mark.smoke
@pytest.mark.integration
def test_transformer_medium_embedder_smoke(wav_path: str) -> None:
    """TransformerMediumEmbedder should return a valid 1D embedding."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from src.models.transformer_medium import TransformerMediumEmbedder

    extractor = TransformerMediumEmbedder()
    embedding = extractor.extract(wav_path)

    # Smoke test the shared extractor contract, not model accuracy.
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.size > 0
    assert embedding.shape[0] == extractor.get_embedding_dim()
    assert np.all(np.isfinite(embedding))


@pytest.mark.smoke
@pytest.mark.integration
def test_transformer_large_embedder_smoke(wav_path: str) -> None:
    """TransformerLargeEmbedder should return a valid 1D embedding."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from src.models.transformer_large import TransformerLargeEmbedder

    extractor = TransformerLargeEmbedder()
    embedding = extractor.extract(wav_path)

    # Smoke test the shared extractor contract, not model accuracy.
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.size > 0
    assert embedding.shape[0] == extractor.get_embedding_dim()
    assert np.all(np.isfinite(embedding))
