from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


def build_cv(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
    cv_mode: str,
) -> tuple[Any, np.ndarray | None]:
    if cv_mode in {"loso", "leave-one-subject-out"}:
        if groups.size < 2 or len(np.unique(groups)) < 2:
            raise RuntimeError("Need at least two unique subjects for LOSO CV.")
        return LeaveOneGroupOut(), groups

    if cv_mode in {"group-kfold", "groupkfold", "group"} and groups.size > 0:
        actual_splits = min(n_splits, len(np.unique(groups)))
        if actual_splits >= 2:
            cv = StratifiedGroupKFold(
                n_splits=actual_splits,
                shuffle=True,
                random_state=random_state,
            )
            return cv, groups

    class_counts = np.bincount(y)
    min_count = int(class_counts[class_counts > 0].min())
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        raise RuntimeError(
            "Not enough samples per class for CV. Need >=2 samples per class."
        )

    cv = StratifiedKFold(
        n_splits=actual_splits, shuffle=True, random_state=random_state
    )
    return cv, None


def split_train_val(
    train_indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    y_train = y[train_indices]
    g_train = groups[train_indices]

    if train_indices.size < 4:
        return train_indices[:-1], train_indices[-1:]

    try:
        if len(np.unique(g_train)) >= 3:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=random_state,
            )
            rel_train, rel_val = next(splitter.split(train_indices, y_train, g_train))
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=random_state,
            )
            rel_train, rel_val = next(splitter.split(train_indices, y_train))
    except ValueError:
        # Stratification can be impossible with scarce labels/groups. Fall back
        # to relative positions, then map back to the original absolute indices.
        rng = np.random.default_rng(random_state)
        rel_perm = rng.permutation(train_indices.size)
        n_val = max(1, int(round(train_indices.size * val_fraction)))
        n_val = min(train_indices.size - 1, n_val)
        rel_val = np.sort(rel_perm[:n_val])
        rel_train = np.sort(rel_perm[n_val:])

    return train_indices[rel_train], train_indices[rel_val]


def save_split(
    out_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    sample_id: np.ndarray,
    subject: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        sample_id=sample_id.astype(np.int64),
        subject=subject.astype(np.int64),
    )


def load_split(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "train_idx": data["train_idx"].astype(np.int64),
        "val_idx": data["val_idx"].astype(np.int64),
        "test_idx": data["test_idx"].astype(np.int64),
        "sample_id": data["sample_id"].astype(np.int64),
        "subject": data["subject"].astype(np.int64),
    }
