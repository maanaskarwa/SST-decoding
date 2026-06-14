from __future__ import annotations

import torch


# used in Captum's explainers (Lime etc)
def build_patch_feature_mask(
    n_channels: int,
    n_times: int,
    channel_group_size: int,
    time_group_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if channel_group_size < 1:
        raise ValueError("channel_group_size must be >= 1")
    if time_group_size < 1:
        raise ValueError("time_group_size must be >= 1")

    mask = torch.zeros((1, n_channels, n_times), dtype=torch.long, device=device)
    group_id = 0
    for c0 in range(0, n_channels, channel_group_size):
        c1 = min(n_channels, c0 + channel_group_size)
        for t0 in range(0, n_times, time_group_size):
            t1 = min(n_times, t0 + time_group_size)
            mask[:, c0:c1, t0:t1] = group_id
            group_id += 1
    return mask


# bruh i have too many small files maybe just merge ts
