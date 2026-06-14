"""Shared fold standardization and dataloader assembly for campaign training CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pipeline.data import apply_standardizer, fit_standardizer, save_standardizer
from pipeline.train import make_dataloader


@dataclass(frozen=True)
class FoldLoaders:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


def make_standardized_fold_loaders(
    *,
    X: np.ndarray,
    y: np.ndarray,
    sample_id: np.ndarray,
    subject_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    weighted_sampling: bool = True,
) -> FoldLoaders:
    """Fit/save the fold standardizer, apply it, and build train/val/test loaders."""
    mean, std = fit_standardizer(X[train_idx])
    save_standardizer(fold_dir / "standardizer.npz", mean=mean, std=std)

    X_train = apply_standardizer(X[train_idx], mean, std)
    X_val = apply_standardizer(X[val_idx], mean, std)
    X_test = apply_standardizer(X[test_idx], mean, std)

    common_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "prefetch_factor": int(prefetch_factor),
        "persistent_workers": bool(persistent_workers),
    }
    return FoldLoaders(
        train_loader=make_dataloader(
            X=X_train,
            y=y[train_idx],
            sample_ids=sample_id[train_idx],
            subject_ids=subject_ids[train_idx],
            weighted_sampling=bool(weighted_sampling),
            **common_kwargs,
        ),
        val_loader=make_dataloader(
            X=X_val,
            y=y[val_idx],
            sample_ids=sample_id[val_idx],
            subject_ids=subject_ids[val_idx],
            weighted_sampling=False,
            **common_kwargs,
        ),
        test_loader=make_dataloader(
            X=X_test,
            y=y[test_idx],
            sample_ids=sample_id[test_idx],
            subject_ids=subject_ids[test_idx],
            weighted_sampling=False,
            **common_kwargs,
        ),
    )
