"""Minimal PANNs helper functions required by vendored model variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class Interpolator(nn.Module):
    """Nearest-neighbor interpolation along the time axis."""

    def __init__(self, ratio: int, interpolate_mode: str = "nearest") -> None:
        super().__init__()
        if interpolate_mode != "nearest":
            raise ValueError(f"Unsupported interpolate_mode: {interpolate_mode}")
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, classes_num = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, self.ratio, 1)
        return upsampled.reshape(batch_size, time_steps * self.ratio, classes_num)


def interpolate(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """Interpolate a framewise tensor along the time axis."""
    return Interpolator(ratio=ratio)(x)


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int) -> torch.Tensor:
    """Pad framewise output to the requested frame count."""
    pad = framewise_output[:, -1:, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    return torch.cat((framewise_output, pad), dim=1)


def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor) -> torch.Tensor:
    """Apply pairwise mixup used by the original PANNs models."""
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + x[1::2].transpose(
        0, -1
    ) * mixup_lambda[1::2]
    return out.transpose(0, -1)
