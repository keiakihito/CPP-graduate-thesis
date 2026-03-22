"""Focused tests for build_embeddings model selection."""

from __future__ import annotations

import pytest

from src.pipeline import build_embeddings


class _DummyCNNSmall:
    pass


class _DummyCNNBase:
    pass


class _DummyTransformerSmall:
    pass


class _DummyTransformerBase:
    pass


def test_create_extractor_supported_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_extractor should dispatch all supported model names."""
    monkeypatch.setattr(build_embeddings, "CNNSmallEmbedder", _DummyCNNSmall)
    monkeypatch.setattr(build_embeddings, "CNNBaseEmbedder", _DummyCNNBase)
    monkeypatch.setattr(build_embeddings, "TransformerSmallEmbedder", _DummyTransformerSmall)
    monkeypatch.setattr(build_embeddings, "TransformerBaseEmbedder", _DummyTransformerBase)

    assert isinstance(build_embeddings.create_extractor("cnn_small"), _DummyCNNSmall)
    assert isinstance(build_embeddings.create_extractor("cnn_base"), _DummyCNNBase)
    assert isinstance(
        build_embeddings.create_extractor("transformer_small"),
        _DummyTransformerSmall,
    )
    assert isinstance(
        build_embeddings.create_extractor("transformer_base"),
        _DummyTransformerBase,
    )


def test_create_extractor_unsupported_model() -> None:
    """create_extractor should reject unsupported model names."""
    with pytest.raises(ValueError, match="Unsupported model_name"):
        build_embeddings.create_extractor("not_a_model")
