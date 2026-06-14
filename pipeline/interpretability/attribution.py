from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pipeline.misc import GO_LABEL, STOP_LABEL


def stop_margin(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    logits = model(x)
    return logits[:, STOP_LABEL] - logits[:, GO_LABEL]


def stop_probability(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    logits = model(x)
    return torch.softmax(logits, dim=1)[:, STOP_LABEL]


def integrated_gradients_stop_margin(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    model.eval()
    x = x.detach()
    baseline = (
        torch.zeros_like(x)
        if baseline is None
        else baseline.to(device=x.device, dtype=x.dtype)
    )

    total_grads = torch.zeros_like(x)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype)[1:]

    for alpha in alphas:
        x_step = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        margin = stop_margin(model=model, x=x_step)
        grads = torch.autograd.grad(
            margin.sum(), x_step, retain_graph=False, create_graph=False
        )[0]
        total_grads += grads.detach()

    avg_grads = total_grads / float(steps)
    return (x - baseline) * avg_grads


def temporal_occlusion_curve(
    model: nn.Module,
    x: torch.Tensor,
    window_size: int = 16,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (n, c, t), got {x.shape}")
    model.eval()
    n_times = int(x.shape[-1])
    if window_size > n_times:
        window_size = n_times

    with torch.no_grad():
        base_p_stop = stop_probability(model=model, x=x)

    centers: list[int] = []
    deltas: list[float] = []
    for start in range(0, n_times - window_size + 1, stride):
        x_occ = x.clone()
        x_occ[:, :, start : start + window_size] = 0.0
        with torch.no_grad():
            occ_p_stop = stop_probability(model=model, x=x_occ)
        centers.append(start + (window_size // 2))
        deltas.append(float((base_p_stop - occ_p_stop).mean().detach().cpu()))

    return np.asarray(centers, dtype=np.int64), np.asarray(deltas, dtype=np.float64)
