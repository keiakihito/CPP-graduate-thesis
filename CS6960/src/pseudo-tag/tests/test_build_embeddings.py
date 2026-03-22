"""Focused tests for build_embeddings model selection."""

from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np
import pytest

from src.pipeline import build_embeddings


class _DummyCNNSmall:
    pass


class _DummyCNNMedium:
    pass


class _DummyCNNLarge:
    pass


class _DummyTransformerMedium:
    pass


class _DummyTransformerLarge:
    pass


def _write_test_wav(
    path: Path,
    sample_rate: int = 16_000,
    duration_sec: float = 1.0,
) -> None:
    """Write a small mono PCM WAV file for build_embeddings tests."""
    num_samples = int(sample_rate * duration_sec)
    time_axis = np.arange(num_samples, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2.0 * math.pi * 440.0 * time_axis)
    pcm_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())


def test_create_extractor_supported_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_extractor should dispatch all supported model names."""
    monkeypatch.setattr(build_embeddings, "CNNSmallEmbedder", _DummyCNNSmall)
    monkeypatch.setattr(build_embeddings, "CNNMediumEmbedder", _DummyCNNMedium)
    monkeypatch.setattr(build_embeddings, "CNNLargeEmbedder", _DummyCNNLarge)
    monkeypatch.setattr(build_embeddings, "TransformerMediumEmbedder", _DummyTransformerMedium)
    monkeypatch.setattr(build_embeddings, "TransformerLargeEmbedder", _DummyTransformerLarge)

    assert isinstance(build_embeddings.create_extractor("cnn_small"), _DummyCNNSmall)
    assert isinstance(build_embeddings.create_extractor("cnn_medium"), _DummyCNNMedium)
    assert isinstance(build_embeddings.create_extractor("cnn_large"), _DummyCNNLarge)
    assert isinstance(
        build_embeddings.create_extractor("transformer_medium"),
        _DummyTransformerMedium,
    )
    assert isinstance(
        build_embeddings.create_extractor("transformer_large"),
        _DummyTransformerLarge,
    )


def test_create_extractor_unsupported_model() -> None:
    """create_extractor should reject unsupported model names."""
    with pytest.raises(ValueError, match="Unsupported model_name"):
        build_embeddings.create_extractor("not_a_model")


def test_extract_track_embedding_mean_pools_all_segments(tmp_path: Path) -> None:
    """Long tracks should be segmented and pooled across all segment embeddings."""
    wav_path = tmp_path / "long_track.wav"
    _write_test_wav(wav_path, sample_rate=10, duration_sec=6.5)

    class _DurationExtractor:
        def extract(self, segment_path: str) -> np.ndarray:
            with wave.open(segment_path, "rb") as wav_file:
                duration = wav_file.getnframes() / float(wav_file.getframerate())
            return np.array([duration], dtype=np.float32)

    embedding = build_embeddings.extract_track_embedding(
        str(wav_path),
        _DurationExtractor(),
        segment_length_sec=3.0,
    )

    expected = np.array([(3.0 + 3.0 + 0.5) / 3.0], dtype=np.float32)
    assert np.allclose(embedding, expected)
