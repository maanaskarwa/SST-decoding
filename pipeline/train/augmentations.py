from __future__ import annotations

import numpy as np
import torch


def add_gaussian_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return x
    return x + torch.randn_like(x) * float(noise_std)


def apply_time_mask(
    x: torch.Tensor,
    time_mask_prob: float,
    rng: np.random.Generator,
    mask_fraction: float = 0.1,
) -> torch.Tensor:
    if time_mask_prob <= 0 or x.shape[-1] < 8:
        return x

    mask_width = max(1, int(round(x.shape[-1] * mask_fraction)))
    mask_width = min(mask_width, x.shape[-1])

    for b in range(x.shape[0]):
        if rng.random() < float(time_mask_prob):
            start = int(rng.integers(0, x.shape[-1] - mask_width + 1))
            x[b, :, start : start + mask_width] = 0.0
    return x


def apply_channel_dropout(x: torch.Tensor, channel_drop_prob: float) -> torch.Tensor:
    if channel_drop_prob <= 0:
        return x

    keep_prob = 1.0 - channel_drop_prob
    channel_mask = (
        torch.rand((x.shape[0], x.shape[1], 1), device=x.device) < keep_prob
    ).to(dtype=x.dtype)
    return x * channel_mask


def apply_train_augmentations(
    x: torch.Tensor,
    noise_std: float,
    time_mask_prob: float,
    channel_drop_prob: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    x = add_gaussian_noise(x=x, noise_std=noise_std)
    x = apply_time_mask(x=x, time_mask_prob=time_mask_prob, rng=rng)
    x = apply_channel_dropout(x=x, channel_drop_prob=channel_drop_prob)
    return x
