from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_stop: np.ndarray,
) -> dict[str, float]:
    return {
        "auc": roc_auc_score(y_true, y_prob_stop),
        "average_precision": average_precision_score(y_true, y_prob_stop),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
