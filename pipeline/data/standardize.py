from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip: float = 8.0,
) -> np.ndarray:
    out = (X - mean) / std
    if clip > 0:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32, copy=False)


def save_standardizer(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=mean.astype(np.float32), std=std.astype(np.float32))


def load_standardizer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)
