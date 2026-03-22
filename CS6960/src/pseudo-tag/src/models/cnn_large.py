"""Inference-only PANNs large embedding extractor for thesis experiments."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from src.models.panns_family import PANNS_SAMPLE_RATE, load_panns_model

PANNS_VARIANT = "cnn14"


def _load_wav_mono(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a PCM WAV file and convert it to mono float32 audio."""
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


def _to_numpy(array_like) -> np.ndarray:
    """Convert tensor-like outputs to a float32 NumPy array."""
    if isinstance(array_like, np.ndarray):
        return array_like.astype(np.float32, copy=False)

    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy"):
        return np.asarray(array_like.numpy(), dtype=np.float32)

    return np.asarray(array_like, dtype=np.float32)


class CNNLargeEmbedder:
    """Inference-only large CNN embedder backed by pretrained PANNs."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        target_sample_rate: int = PANNS_SAMPLE_RATE,
    ) -> None:
        """Load the pretrained PANNs model for inference."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "CNNLargeEmbedder requires 'torch' and vendored PANNs model code "
                "to load a pretrained PANNs model."
            ) from exc

        self._torch = torch
        self.target_sample_rate = target_sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        try:
            self._model, _, self._embedding_dim = load_panns_model(
                PANNS_VARIANT,
                checkpoint_path=checkpoint_path,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError("Failed to load pretrained PANNs model.") from exc

    def get_embedding_dim(self) -> int:
        """Return the clip embedding dimension produced by PANNs."""
        return self._embedding_dim

    def load_audio(self, wav_path: str) -> np.ndarray:
        """Load PCM WAV audio and resample it to the PANNs input rate."""
        audio, sample_rate = _load_wav_mono(wav_path)
        if audio.size == 0:
            raise ValueError(f"Audio file contains no samples: {wav_path}")
        return _resample_audio(audio, sample_rate, self.target_sample_rate)

    def extract_frames(self, wav_path: str) -> np.ndarray:
        """Expose PANNs clip embeddings in a shared 2D interface."""
        waveform = self.load_audio(wav_path)
        batch_audio = self._torch.as_tensor(
            np.expand_dims(waveform, axis=0),
            dtype=self._torch.float32,
            device=self.device,
        )

        try:
            with self._torch.no_grad():
                output_dict = self._model(batch_audio)
        except Exception as exc:
            raise RuntimeError(f"PANNs inference failed for: {wav_path}") from exc

        clip_embeddings = _to_numpy(output_dict["embedding"])
        if clip_embeddings.ndim != 2 or clip_embeddings.shape[0] == 0:
            raise RuntimeError(f"PANNs returned no embeddings for: {wav_path}")

        clip_embedding = clip_embeddings[0]
        if clip_embedding.ndim != 1 or clip_embedding.shape[0] != self._embedding_dim:
            raise RuntimeError(
                f"Unexpected PANNs embedding dimension: {clip_embedding.shape[-1]}"
            )

        return clip_embedding[np.newaxis, :].astype(np.float32)

    def _pool_frames(self, frame_embeddings: np.ndarray) -> np.ndarray:
        """Mean-pool embeddings into one clip representation."""
        if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
            raise ValueError("Frame embeddings must be a non-empty 2D array.")

        clip_embedding = frame_embeddings.mean(axis=0).astype(np.float32)
        if clip_embedding.shape[0] != self._embedding_dim:
            raise RuntimeError(
                f"Unexpected PANNs embedding dimension: {clip_embedding.shape[0]}"
            )

        return clip_embedding

    def extract(self, wav_path: str) -> np.ndarray:
        """Return one pooled clip embedding for a PCM WAV file."""
        frame_embeddings = self.extract_frames(wav_path)
        return self._pool_frames(frame_embeddings)
