"""Inference-only YAMNet embedding extractor for thesis experiments.

This module provides a small wrapper around pretrained YAMNet so the thesis
pipeline can extract one fixed-length clip embedding from a PCM WAV file without
including any training code.
"""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16_000
YAMNET_EMBEDDING_DIM = 1024


def _load_wav_mono(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a PCM WAV file and convert it to mono float32 audio.

    Assumptions:
    - input is a local WAV file readable by the standard-library `wave` module
    - audio is PCM encoded with 8-bit, 16-bit, or 32-bit integer samples
    - 32-bit float WAV is not supported by this loader
    - multi-channel audio is averaged to mono before inference
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {wav_path}")

    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            num_frames = wav_file.getnframes()
            raw_audio = wav_file.readframes(num_frames)
    except wave.Error as exc:
        raise ValueError(f"Failed to read WAV file: {wav_path}") from exc

    if num_frames == 0:
        return np.zeros(0, dtype=np.float32), sample_rate

    if sample_width == 1:
        audio = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    return audio.astype(np.float32), sample_rate


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to the target sample rate using linear interpolation."""
    if orig_sr <= 0:
        raise ValueError(f"Invalid sample rate: {orig_sr}")
    if audio.size == 0 or orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    duration = audio.shape[0] / float(orig_sr)
    target_length = max(1, int(round(duration * target_sr)))

    old_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(new_positions, old_positions, audio)
    return resampled.astype(np.float32)


class CNNSmallEmbedder:
    """Inference-only CNN-style embedder backed by pretrained YAMNet.

    The public interface is intentionally small:
    - `extract_frames()` returns frame-level embeddings from YAMNet
    - `extract()` mean-pools those frames into one clip-level embedding
    """

    def __init__(
        self,
        model_handle: str = YAMNET_MODEL_HANDLE,
        target_sample_rate: int = YAMNET_SAMPLE_RATE,
    ) -> None:
        """Load the pretrained YAMNet model for inference."""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as exc:
            raise ImportError(
                "CNNSmallEmbedder requires 'tensorflow' and 'tensorflow_hub' "
                "to load pretrained YAMNet."
            ) from exc

        self._tf = tf
        self.target_sample_rate = target_sample_rate
        self.model_handle = model_handle

        try:
            self._model = hub.load(model_handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YAMNet model from: {model_handle}") from exc

    def get_embedding_dim(self) -> int:
        """Return the clip embedding dimension produced by YAMNet."""
        return YAMNET_EMBEDDING_DIM

    def load_audio(self, wav_path: str) -> np.ndarray:
        """Load PCM WAV audio and resample it to the YAMNet input rate."""
        audio, sample_rate = _load_wav_mono(wav_path)
        if audio.size == 0:
            raise ValueError(f"Audio file contains no samples: {wav_path}")
        return _resample_audio(audio, sample_rate, self.target_sample_rate)

    def extract_frames(self, wav_path: str) -> np.ndarray:
        """Run YAMNet and return frame-level embeddings."""
        waveform = self.load_audio(wav_path)

        try:
            _, frame_embeddings, _ = self._model(
                self._tf.convert_to_tensor(waveform, dtype=self._tf.float32)
            )
        except Exception as exc:
            raise RuntimeError(f"YAMNet inference failed for: {wav_path}") from exc

        embeddings = np.asarray(frame_embeddings.numpy(), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise RuntimeError(f"YAMNet returned no frame embeddings for: {wav_path}")

        if embeddings.shape[1] != YAMNET_EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected YAMNet embedding dimension: {embeddings.shape[1]}"
            )

        return embeddings

    def _pool_frames(self, frame_embeddings: np.ndarray) -> np.ndarray:
        """Mean-pool frame embeddings into one fixed-length clip embedding."""
        if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
            raise ValueError("Frame embeddings must be a non-empty 2D array.")

        # Mean pooling keeps the interface simple and deterministic for thesis experiments.
        clip_embedding = frame_embeddings.mean(axis=0).astype(np.float32)

        if clip_embedding.shape[0] != YAMNET_EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected YAMNet embedding dimension: {clip_embedding.shape[0]}"
            )

        return clip_embedding

    def extract(self, wav_path: str) -> np.ndarray:
        """Return one mean-pooled clip embedding for a WAV file."""
        frame_embeddings = self.extract_frames(wav_path)
        return self._pool_frames(frame_embeddings)


# Expected setup:
# - Install TensorFlow and TensorFlow Hub in the project environment.
# - The YAMNet model is loaded from TensorFlow Hub via YAMNET_MODEL_HANDLE.
# - Input audio is expected to be PCM WAV; 32-bit float WAV is not supported here.
# - Preprocessing converts input audio to mono 16 kHz before inference.
