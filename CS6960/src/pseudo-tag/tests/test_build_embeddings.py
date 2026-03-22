"""Focused tests for build_embeddings model selection."""

from __future__ import annotations

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
