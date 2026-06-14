from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pipeline.interpretability.attribution import integrated_gradients_stop_margin


def mean_abs_ig(
    model: torch.nn.Module,
    x_np: np.ndarray,
    *,
    device: torch.device,
    ig_steps: int,
) -> np.ndarray:
    """Compute mean absolute integrated gradients over the input samples."""
    if len(x_np) == 0:
        raise ValueError("mean_abs_ig requires at least one input sample")

    batch = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)

    attr = integrated_gradients_stop_margin(model=model, x=batch, steps=ig_steps)
    attr_abs = attr.detach().abs().cpu().numpy()
    total = attr_abs.sum(axis=0)
    n_total = int(attr_abs.shape[0])

    return total / float(max(n_total, 1))


def time_window_to_num_indices(times_s: np.ndarray, window_ms: float) -> int:
    sample_interval_ms = float(np.median(np.diff(times_s)) * 1000.0)
    return max(1, int(round(window_ms / sample_interval_ms)))


def time_window_to_indices(
    times_s: np.ndarray, window_ms: float, stride_ms: float
) -> tuple[int, int]:
    sample_interval_ms = float(np.median(np.diff(times_s)) * 1000.0)
    if sample_interval_ms <= 0:
        raise RuntimeError(
            f"Invalid time step inferred from times: {sample_interval_ms}"
        )
    return (
        max(1, int(round(window_ms / sample_interval_ms))),
        max(1, int(round(stride_ms / sample_interval_ms))),
    )


def summarize_time_curve(
    curve: np.ndarray, times_s: np.ndarray, p3_mask: np.ndarray
) -> dict[str, Any]:
    peak_index = int(np.argmax(curve))
    total = float(curve.sum())
    return {
        "peak_time_s": float(times_s[peak_index]),
        "peak_value": float(curve[peak_index]),
        "peak_in_p3_window": bool(p3_mask[peak_index]),
        "p3_window_share": float(curve[p3_mask].sum() / max(total, 1e-12)),
        "mean_p3_window": float(curve[p3_mask].mean()),
        "mean_global": float(curve.mean()),
    }
