"""Torch runtime and AMP helpers. Configures performance flags, autocast behavior, gradient scalers, and device resolution."""

from __future__ import annotations

from contextlib import nullcontext

import torch

_AMP_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a device string ('cuda', 'cpu', or 'auto'/other) to a torch.device.

    - 'cuda' -> cuda (errors if unavailable)
    - 'cpu' -> cpu
    - anything else -> cuda if available, else cpu (the common "auto" behavior)
    """
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch_runtime(
    device: torch.device,
    *,
    cudnn_benchmark: bool = False,
    matmul_precision: str | None = None,
    torch_num_threads: int | None = None,
    torch_num_interop_threads: int | None = None,
) -> None:
    if torch_num_threads is not None:
        torch.set_num_threads(max(1, int(torch_num_threads)))
    if torch_num_interop_threads is not None:
        torch.set_num_interop_threads(max(1, int(torch_num_interop_threads)))
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and device.type == "cuda"


def resolve_amp_dtype(
    device: torch.device, amp_dtype: str = "auto"
) -> torch.dtype | None:
    if device.type != "cuda":
        return None

    if amp_dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return _AMP_DTYPE_MAP[amp_dtype]


def autocast_context(
    device: torch.device,
    *,
    enabled: bool,
    amp_dtype: str = "auto",
):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=resolve_amp_dtype(device=device, amp_dtype=amp_dtype),
        enabled=True,
    )


def make_grad_scaler(device: torch.device, *, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device=device.type, enabled=bool(enabled) and device.type == "cuda"
    )
