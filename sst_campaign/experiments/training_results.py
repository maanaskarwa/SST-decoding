from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pipeline.eval import compute_metrics, plot_confusion
from pipeline.misc import GO_LABEL, STOP_LABEL

CLASSIFICATION_METRICS = ("auc", "average_precision", "accuracy", "balanced_accuracy")


def metric_fields(metrics: dict[str, float], *, prefix: str) -> dict[str, float]:
    # pretty pretty formatting
    return {
        f"{prefix}_{name}": float(metrics[name])
        for name in CLASSIFICATION_METRICS
        if name in metrics
    }


def fold_metric_row(
    *,
    fold: int,
    n_train: int,
    n_val: int,
    n_test: int,
    best_epoch: int,
    best_val_score: float,
    test_loss: float,
    test_metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": int(fold),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "test_loss": float(test_loss),
    }
    row.update(metric_fields(test_metrics, prefix="test"))
    if extra:
        row.update(extra)
    return row


def aggregate_metric_columns(
    fold_df: pd.DataFrame,
    *,
    source_prefix: str = "test",
    output_prefix: str = "test",
    include_std: bool = True,
    std_ddof: int = 0,
) -> dict[str, float]:
    """Aggregate standard metric columns from a fold/results dataframe."""
    out: dict[str, float] = {}
    for metric in CLASSIFICATION_METRICS:
        column = f"{source_prefix}_{metric}"
        if column not in fold_df.columns:
            continue
        values = fold_df[column]
        out[f"mean_{output_prefix}_{metric}"] = float(values.mean())
        if include_std:
            out[f"std_{output_prefix}_{metric}"] = float(values.std(ddof=std_ddof))
    return out


def dataset_count_fields(
    y: np.ndarray, groups: np.ndarray | None = None
) -> dict[str, int]:
    """Return standard dataset-count metadata for a decoded go/stop dataset."""
    out = {
        "n_epochs_total": int(len(y)),
        "n_go": int((y == GO_LABEL).sum()),
        "n_stop": int((y == STOP_LABEL).sum()),
    }
    if groups is not None:
        out["n_subjects"] = int(len(np.unique(groups)))
    return out


def format_metric_triplet(
    metrics: dict[str, float], *, prefix: str = "mean_test"
) -> str:
    """Compact human-readable AUC/AP/accuracy/balanced-accuracy summary."""
    parts: list[str] = []
    labels = {
        "auc": "auc",
        "average_precision": "ap",
        "accuracy": "acc",
        "balanced_accuracy": "bal_acc",
    }
    for metric, label in labels.items():
        key = f"{prefix}_{metric}"
        if key in metrics:
            parts.append(f"{label}={float(metrics[key]):.3f}")
    return " ".join(parts)


def checkpoint_score_from_validation(
    *,
    val_loss: float,
    val_metrics: dict[str, float],
    checkpoint_metric: str,
) -> float:
    """Convert validation outputs to a score where larger is always better."""
    metric = str(checkpoint_metric)
    if metric == "loss":
        return -float(val_loss)
    if metric not in val_metrics:
        raise ValueError(f"Unsupported checkpoint metric: {metric}")
    score = float(val_metrics[metric])
    if not np.isfinite(score):
        raise ValueError(
            f"Checkpoint metric '{metric}' is not finite (got {score}). "
            "Training/validation produced invalid metrics; refusing to select checkpoint silently."
        )
    return score


@dataclass(frozen=True)
class PerSubjectReportPaths:
    metrics_csv: Path
    confusion_dir: Path


def write_per_subject_prediction_report(
    *,
    pred_df: pd.DataFrame,
    out_dir: Path,
    metrics_filename: str,
) -> PerSubjectReportPaths | None:
    """Write per-subject metrics and confusion matrices for prediction rows."""
    required = {"subject", "y_true", "y_pred", "p_stop"}
    if pred_df.empty or not required.issubset(pred_df.columns):
        return None

    confusion_dir = out_dir / "per_subject_confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for subject, subject_df in pred_df.groupby("subject"):
        subject_id = int(cast(Any, subject))
        y_true = subject_df["y_true"].to_numpy(dtype=int)
        y_pred = subject_df["y_pred"].to_numpy(dtype=int)
        p_stop = subject_df["p_stop"].to_numpy(dtype=float)
        n_go = int((y_true == GO_LABEL).sum())
        rows.append(
            {
                "subject": subject_id,
                "folds": _fold_summary(subject_df),
                "n_test": int(len(subject_df)),
                "n_go_true": n_go,
                "n_stop_true": int((y_true == STOP_LABEL).sum()),
                "go_as_go": int(((y_true == GO_LABEL) & (y_pred == GO_LABEL)).sum()),
                "go_as_stop": int(
                    ((y_true == GO_LABEL) & (y_pred == STOP_LABEL)).sum()
                ),
                "stop_as_go": int(
                    ((y_true == STOP_LABEL) & (y_pred == GO_LABEL)).sum()
                ),
                "stop_as_stop": int(
                    ((y_true == STOP_LABEL) & (y_pred == STOP_LABEL)).sum()
                ),
                "go_class_fraction": float(n_go / max(len(subject_df), 1)),
                **metric_fields(
                    compute_metrics(y_true=y_true, y_pred=y_pred, y_prob_stop=p_stop),
                    prefix="test",
                ),
            }
        )
        plot_confusion(
            y_true=y_true,
            y_pred=y_pred,
            out_path=confusion_dir / f"subject_{subject_id:02d}_confusion_matrix.png",
        )

    metrics_csv = out_dir / metrics_filename
    pd.DataFrame(rows).sort_values("subject").reset_index(drop=True).to_csv(
        metrics_csv, index=False
    )
    return PerSubjectReportPaths(metrics_csv=metrics_csv, confusion_dir=confusion_dir)


def _fold_summary(subject_df: pd.DataFrame) -> str:
    if "fold" not in subject_df.columns:
        return ""
    folds = sorted(set(int(x) for x in subject_df["fold"].to_numpy(dtype=int)))
    return ",".join(str(fold) for fold in folds)
