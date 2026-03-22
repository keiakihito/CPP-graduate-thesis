"""Batch embedding generation for thesis MVP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import wave

import numpy as np

from src.io_utils import ensure_dir, save_json
from src.models.cnn_large import CNNLargeEmbedder
from src.models.cnn_medium import CNNMediumEmbedder
from src.models.cnn_small import CNNSmallEmbedder
from src.models.transformer_large import TransformerLargeEmbedder
from src.models.transformer_medium import TransformerMediumEmbedder


def collect_wav_paths(input_dir: str) -> list[Path]:
    """Recursively collect WAV files in deterministic order."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not root.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")


def create_extractor(model_name: str) -> Any:
    """Create an embedding extractor from a simple model name."""
    if model_name == "cnn_small":
        return CNNSmallEmbedder()
    if model_name == "cnn_medium":
        return CNNMediumEmbedder()
    if model_name == "cnn_large":
        return CNNLargeEmbedder()
    if model_name == "transformer_medium":
        return TransformerMediumEmbedder()
    if model_name == "transformer_large":
        return TransformerLargeEmbedder()
    raise ValueError(f"Unsupported model_name: {model_name}")


def _iter_wav_segments(wav_path: str, segment_length_sec: float = 30.0):
    """Yield PCM WAV byte segments using fixed-length non-overlapping windows."""
    if segment_length_sec <= 0:
        raise ValueError(f"segment_length_sec must be positive, got {segment_length_sec}")

    with wave.open(wav_path, "rb") as wav_file:
        params = wav_file.getparams()
        sample_rate = wav_file.getframerate()
        frames_per_segment = max(1, int(round(sample_rate * segment_length_sec)))

        while True:
            segment_frames = wav_file.readframes(frames_per_segment)
            if not segment_frames:
                break
            yield params, segment_frames


def extract_track_embedding(
    wav_path: str,
    extractor: Any,
    segment_length_sec: float = 30.0,
) -> np.ndarray:
    """Extract one track embedding by mean-pooling fixed-length segment embeddings."""
    segment_embeddings: list[np.ndarray] = []

    with TemporaryDirectory(prefix="build_embeddings_") as temp_dir:
        temp_root = Path(temp_dir)

        for segment_index, (params, segment_frames) in enumerate(
            _iter_wav_segments(wav_path, segment_length_sec=segment_length_sec)
        ):
            segment_path = temp_root / f"segment_{segment_index:04d}.wav"
            with wave.open(str(segment_path), "wb") as segment_file:
                segment_file.setparams(params)
                segment_file.writeframes(segment_frames)

            segment_embedding = np.asarray(
                extractor.extract(str(segment_path)),
                dtype=np.float32,
            )
            if segment_embedding.ndim != 1:
                raise ValueError(
                    f"Expected 1D segment embedding, got shape {segment_embedding.shape}"
                )
            segment_embeddings.append(segment_embedding)

    if not segment_embeddings:
        raise ValueError(f"Audio file contains no segments: {wav_path}")

    return np.mean(np.stack(segment_embeddings, axis=0), axis=0).astype(np.float32)


def build_embeddings(input_dir: str, output_dir: str, model_name: str) -> None:
    """Build and save clip-level embeddings for all WAV files in a directory."""
    wav_paths = collect_wav_paths(input_dir)
    extractor = create_extractor(model_name)
    embedding_dim = extractor.get_embedding_dim()

    output_path = Path(output_dir)
    embeddings_path = output_path / f"{model_name}_embeddings.npy"
    metadata_path = output_path / f"{model_name}_metadata.json"
    failures_path = output_path / f"{model_name}_failures.json"

    output_path.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for i, wav_path in enumerate(wav_paths):
        print(f"[{i+1}/{len(wav_paths)}] Processing: {wav_path}")
        try:
            embedding = extract_track_embedding(str(wav_path), extractor)
        except Exception as exc:
            failures.append(
                {
                    "path": str(wav_path),
                    "model_name": model_name,
                    "error": str(exc),
                }
            )
            continue

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1:
            failures.append(
                {
                    "path": str(wav_path),
                    "model_name": model_name,
                    "error": f"Expected 1D embedding, got shape {embedding.shape}",
                }
            )
            continue

        if embedding.shape[0] != embedding_dim:
            failures.append(
                {
                    "path": str(wav_path),
                    "model_name": model_name,
                    "error": f"Expected embedding_dim={embedding_dim}, got {embedding.shape[0]}",
                }
            )
            continue

        row_index = len(embeddings)
        embeddings.append(embedding)
        metadata.append(
            {
                "index": row_index,
                "path": str(wav_path),
                "file_id": wav_path.stem,
                "model_name": model_name,
            }
        )

    if embeddings:
        matrix = np.stack(embeddings).astype(np.float32)
    else:
        matrix = np.zeros((0, embedding_dim), dtype=np.float32)

    np.save(embeddings_path, matrix)
    save_json(metadata, metadata_path)
    save_json(failures, failures_path)

    print(
        f"build_embeddings complete: model={model_name}, "
        f"processed={len(wav_paths)}, saved={len(metadata)}, failed={len(failures)}"
    )


def main() -> None:
    """Run batch embedding extraction from the command line."""
    parser = argparse.ArgumentParser(description="Build clip embeddings for WAV files.")
    parser.add_argument("input_dir", type=str, help="Directory containing WAV files.")
    parser.add_argument("output_dir", type=str, help="Directory to save embedding outputs.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn_small",
        choices=[
            "cnn_small",
            "cnn_medium",
            "cnn_large",
            "transformer_medium",
            "transformer_large",
        ],
        help="Embedding backend to use.",
    )
    args = parser.parse_args()

    build_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
