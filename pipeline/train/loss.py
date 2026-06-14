from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from pipeline.misc import STOP_LABEL


def build_classification_loss(
    y_train: np.ndarray, device: torch.device
) -> nn.CrossEntropyLoss:
    """Build the class-balanced 2-class classification loss."""
    n_classes = 2
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.clip(class_counts, 1.0, None)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.astype(np.float32)
    weight = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight)


def build_loss_from_args(
    args: Any, y_train: np.ndarray, device: torch.device
) -> nn.Module:
    """Build the campaign training loss from common CLI/config arguments."""
    if str(args.class_weight_mode) == "none":
        weight = None
    else:
        loss = build_classification_loss(y_train=y_train, device=device)
        weight = (
            loss.weight.detach().clone()
            if loss.weight is not None
            else torch.ones(2, device=device)
        )
        scale = float(args.stop_loss_scale)
        if abs(scale - 1.0) > 1e-12:
            weight[STOP_LABEL] *= scale
    label_smoothing = float(getattr(args, "label_smoothing", 0.0))
    if weight is None:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
