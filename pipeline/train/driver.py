from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pipeline.eval.metrics import compute_metrics
from pipeline.misc import STOP_LABEL
from pipeline.perf import autocast_context
from pipeline.train.augmentations import apply_train_augmentations


class EEGDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: np.ndarray,
        subject_ids: np.ndarray,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n, c, t), got {X.shape}")
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))
        self.sample_ids = torch.from_numpy(sample_ids.astype(np.int64, copy=False))
        self.subject_ids = torch.from_numpy(subject_ids.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X[idx],
            self.y[idx],
            self.sample_ids[idx],
            self.subject_ids[idx],
        )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(y_train)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    for cls, count in enumerate(class_counts):
        if count > 0:
            class_weights[cls] = 1.0 / count
    sample_weights = class_weights[y_train]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(y_train),
        replacement=True,
    )


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    subject_ids: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    weighted_sampling: bool,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> DataLoader:
    ds = EEGDataset(X=X, y=y, sample_ids=sample_ids, subject_ids=subject_ids)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)

    if weighted_sampling:
        sampler = make_sampler(y)
        return DataLoader(
            ds,
            sampler=sampler,
            **loader_kwargs,
        )

    return DataLoader(
        ds,
        shuffle=False,
        **loader_kwargs,
    )


def run_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classification_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train_mode: bool,
    noise_std: float,
    time_mask_prob: float,
    channel_drop_prob: float,
    rng: np.random.Generator,
    amp_enabled: bool = False,
    amp_dtype: str = "auto",
    grad_scaler: torch.amp.GradScaler | None = None,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    model.train(mode=train_mode)

    total_loss = 0.0
    total_items = 0

    all_sample_ids: list[np.ndarray] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_y_prob_stop: list[np.ndarray] = []

    for batch in loader:
        x, y, sample_ids, subject_ids = batch
        x, y, sample_ids, subject_ids = (
            x.to(device=device, non_blocking=True),
            y.to(device=device, non_blocking=True),
            sample_ids.to(device=device, non_blocking=True),
            subject_ids.to(device=device, non_blocking=True),
        )

        if train_mode:
            x = apply_train_augmentations(
                x=x,
                noise_std=noise_std,
                time_mask_prob=time_mask_prob,
                channel_drop_prob=channel_drop_prob,
                rng=rng,
            )
        mixup_y: torch.Tensor | None = None
        mixup_lam = 1.0
        if train_mode and mixup_alpha > 0.0 and mixup_prob > 0.0 and x.shape[0] > 1:
            if float(rng.random()) < float(mixup_prob):
                mixup_lam = float(rng.beta(float(mixup_alpha), float(mixup_alpha)))
                perm = torch.randperm(x.shape[0], device=x.device)
                mixup_y = y[perm]
                x = mixup_lam * x + (1.0 - mixup_lam) * x[perm]

        with torch.set_grad_enabled(train_mode):
            with autocast_context(
                device=device, enabled=amp_enabled, amp_dtype=amp_dtype
            ):
                logits = model(x)
                loss = classification_loss(logits, y)
                if mixup_y is not None:
                    loss = mixup_lam * loss + (1.0 - mixup_lam) * classification_loss(
                        logits, mixup_y
                    )

            if train_mode:
                assert optimizer is not None, "optimizer not found in training mode"
                optimizer.zero_grad()
                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        batch_n = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_n
        total_items += batch_n

        go_prob = torch.softmax(logits, dim=1)
        go_pred = torch.argmax(go_prob, dim=1)

        all_sample_ids.append(sample_ids.detach().cpu().numpy())
        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(go_pred.detach().cpu().numpy())
        all_y_prob_stop.append(go_prob[:, STOP_LABEL].detach().float().cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    y_prob_stop = np.concatenate(all_y_prob_stop, axis=0)
    sample_ids_np = np.concatenate(all_sample_ids, axis=0)

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob_stop=y_prob_stop,
    )
    metrics["loss"] = float(total_loss / max(total_items, 1))

    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids_np.astype(int),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "p_stop": y_prob_stop.astype(float),
        }
    )
    return metrics["loss"], metrics, predictions


def predict_probabilities(
    model: torch.nn.Module,
    x_np: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_np).float().to(device))
        p_stop = logits.softmax(dim=1)[:, STOP_LABEL]
        pred = logits.argmax(dim=1)
    return p_stop.detach().cpu().numpy(), pred.detach().cpu().numpy()


def select_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_label: int,
    max_samples: int,
    include_misclassified: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if include_misclassified:
        idx = np.flatnonzero(y_true == target_label)
    else:
        idx = np.flatnonzero((y_true == target_label) & (y_pred == target_label))

    if idx.size <= max_samples:
        return idx.astype(np.int64, copy=False)
    chosen = rng.choice(idx, size=max_samples, replace=False)
    chosen.sort()
    return chosen.astype(np.int64, copy=False)


def map_roi_indices(channel_names: list[str], roi_channels: list[str]) -> list[int]:
    lookup = {channel: idx for idx, channel in enumerate(channel_names)}
    return [lookup[channel] for channel in roi_channels if channel in lookup]
