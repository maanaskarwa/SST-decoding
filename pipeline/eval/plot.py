from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from pipeline.misc import GO_LABEL, STOP_LABEL


def plot_training_curves(history_df: pd.DataFrame, out_path: Path) -> None:
    if history_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for fold, fold_df in history_df.groupby("fold"):
        axes[0].plot(
            fold_df["epoch"],
            fold_df["train_loss"],
            alpha=0.8,
            label=f"fold {fold} train",
        )
        axes[0].plot(
            fold_df["epoch"],
            fold_df["val_loss"],
            linestyle="--",
            alpha=0.8,
            label=f"fold {fold} val",
        )
        axes[1].plot(
            fold_df["epoch"], fold_df["val_auc"], alpha=0.85, label=f"fold {fold}"
        )

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].axhline(0.5, color="k", linestyle=":", linewidth=1)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=[GO_LABEL, STOP_LABEL], normalize="true"
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], labels=["go", "stop"])
    ax.set_yticks([0, 1], labels=["go", "stop"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
