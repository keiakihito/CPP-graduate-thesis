"""Shared PANNs family loader for explicit CNN capacity tiers."""

from __future__ import annotations

from pathlib import Path


PANNS_SAMPLE_RATE = 32_000
PANNS_WINDOW_SIZE = 1024
PANNS_HOP_SIZE = 320
PANNS_MEL_BINS = 64
PANNS_FMIN = 50
PANNS_FMAX = 14_000
PANNS_CLASSES_NUM = 527

PANNS_CHECKPOINTS = {
    "cnn6": Path(__file__).resolve().parents[1] / "checkpoints" / "Cnn6_mAP=0.343.pth",
    "cnn10": Path(__file__).resolve().parents[1] / "checkpoints" / "Cnn10_mAP=0.380.pth",
    "cnn14": Path(__file__).resolve().parents[1] / "checkpoints" / "Cnn14_mAP=0.431.pth",
}

PANNS_EMBEDDING_DIMS = {
    "cnn6": 512,
    "cnn10": 512,
    "cnn14": 2048,
}


def create_panns_model(variant: str):
    """Create a vendored PANNs model for an explicit capacity tier."""
    try:
        from src.models.panns_variants import Cnn10, Cnn14, Cnn6
    except ImportError as exc:
        raise ImportError(
            "PANNs variants require 'torch', 'torchlibrosa', and vendored model code."
        ) from exc

    model_classes = {
        "cnn6": Cnn6,
        "cnn10": Cnn10,
        "cnn14": Cnn14,
    }
    model_class = model_classes.get(variant)
    if model_class is None:
        raise ValueError(f"Unsupported PANNs variant: {variant}")

    return model_class(
        sample_rate=PANNS_SAMPLE_RATE,
        window_size=PANNS_WINDOW_SIZE,
        hop_size=PANNS_HOP_SIZE,
        mel_bins=PANNS_MEL_BINS,
        fmin=PANNS_FMIN,
        fmax=PANNS_FMAX,
        classes_num=PANNS_CLASSES_NUM,
    )


def load_panns_model(
    variant: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
):
    """Load a vendored PANNs model checkpoint for inference."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PANNs loading requires 'torch'.") from exc

    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else PANNS_CHECKPOINTS.get(variant)
    )
    if resolved_checkpoint is None:
        raise ValueError(f"Unsupported PANNs variant: {variant}")
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"PANNs checkpoint not found: {resolved_checkpoint}")

    model = create_panns_model(variant)
    checkpoint = torch.load(str(resolved_checkpoint), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, torch, PANNS_EMBEDDING_DIMS[variant]
