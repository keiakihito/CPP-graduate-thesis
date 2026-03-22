"""Inference-only AST embedding extractor for thesis experiments.

This module wraps a pretrained Audio Spectrogram Transformer (AST) model and
returns one fixed-length clip embedding per PCM WAV file. It is intended for
embedding extraction only and does not include any training code.
"""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

# This model is for using in the MVP pipeline.
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16_000


def _load_wav_mono(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a PCM WAV file and convert it to mono float32 audio.

    Assumptions:
    - input is a local PCM WAV file readable by the standard-library `wave` module
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


class TransformerSmallEmbedder:
    """Inference-only AST embedder for fixed-length clip representations.

    This wrapper uses a pretrained Hugging Face AST model. The model may produce
    token-level hidden states rather than literal acoustic frames, so this class
    mean-pools them into one clip-level embedding as a simple baseline strategy
    for the thesis pipeline.
    """

    def __init__(
        self,
        model_name: str = AST_MODEL_NAME,
        target_sample_rate: int = AST_SAMPLE_RATE,
    ) -> None:
        """Load the pretrained AST processor and model for inference."""
        try:
            import torch
            from transformers import AutoFeatureExtractor, ASTModel
        except ImportError as exc:
            raise ImportError(
                "TransformerSmallEmbedder requires 'torch' and 'transformers' "
                "to load a pretrained AST model."
            ) from exc

        self._torch = torch
        self.target_sample_rate = target_sample_rate
        self.model_name = model_name

        try:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self._model = ASTModel.from_pretrained(model_name)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load AST model from: {model_name}") from exc

        self._embedding_dim = int(self._model.config.hidden_size)

    def get_embedding_dim(self) -> int:
        """Return the clip embedding dimension produced by AST."""
        return self._embedding_dim

    def load_audio(self, wav_path: str) -> np.ndarray:
        """Load PCM WAV audio and resample it to the AST input rate."""
        audio, sample_rate = _load_wav_mono(wav_path)
        if audio.size == 0:
            raise ValueError(f"Audio file contains no samples: {wav_path}")
        return _resample_audio(audio, sample_rate, self.target_sample_rate)

    def extract_frames(self, wav_path: str) -> np.ndarray:
        """Run AST and return token-level hidden states.

        Despite the method name, AST outputs transformer token representations
        rather than literal time-domain or spectrogram frame features.
        """
        waveform = self.load_audio(wav_path)

        try:
            inputs = self._feature_extractor(
                waveform,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
            )

            with self._torch.no_grad():
                outputs = self._model(**inputs)
        except Exception as exc:
            raise RuntimeError(f"AST inference failed for: {wav_path}") from exc

        hidden_states = outputs.last_hidden_state
        embeddings = hidden_states.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise RuntimeError(f"AST returned no hidden representations for: {wav_path}")
        if embeddings.shape[1] != self._embedding_dim:
            raise RuntimeError(f"Unexpected AST embedding dimension: {embeddings.shape[1]}")

        return embeddings

    def _pool_frames(self, frame_embeddings: np.ndarray) -> np.ndarray:
        """Mean-pool token-level hidden states into one clip embedding."""
        if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
            raise ValueError("Frame embeddings must be a non-empty 2D array.")

        # Mean pooling is the current baseline pooling strategy for AST outputs.
        clip_embedding = frame_embeddings.mean(axis=0).astype(np.float32)

        if clip_embedding.shape[0] != self._embedding_dim:
            raise RuntimeError(
                f"Unexpected AST embedding dimension: {clip_embedding.shape[0]}"
            )

        return clip_embedding

    def extract(self, wav_path: str) -> np.ndarray:
        """Return one pooled clip embedding for a PCM WAV file."""
        frame_embeddings = self.extract_frames(wav_path)
        return self._pool_frames(frame_embeddings)


# Expected setup:
# - Install PyTorch and Hugging Face Transformers in the project environment.
# - The default checkpoint is loaded from Hugging Face via AST_MODEL_NAME.
# - Input audio is expected to be PCM WAV; 32-bit float WAV is not supported here.
# - Audio is converted to mono and resampled to 16 kHz before AST preprocessing.
