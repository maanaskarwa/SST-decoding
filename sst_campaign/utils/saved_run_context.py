from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from pipeline.data import apply_standardizer, load_split, load_standardizer
from sst_campaign.experiments.common import (
    apply_trial_normalization,
    combine_subject_data,
    load_subject_dataset,
    resolve_subjects,
)
from sst_campaign.utils.model_loading import (
    build_model_from_config,
    load_run_cfg_meta,
)
from sst_campaign.utils.model_specs import model_name_from_cfg


@dataclass(frozen=True)
class SavedRunData:
    root: Path
    run_dir: Path
    cfg: dict
    meta: dict
    model_name: str
    subjects: list[int]
    X_raw: np.ndarray
    y: np.ndarray
    subject_col: np.ndarray
    stop_signal_onset_s: np.ndarray
    times_s: np.ndarray
    channel_names: list[str]
    fold_dirs: list[Path]


@dataclass(frozen=True)
class SavedRunFold:
    fold_dir: Path
    fold_idx: int
    model: Any
    test_idx: np.ndarray
    X_te_raw: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray


def load_saved_run_data(
    *,
    root: Path,
    run_dir: Path,
    fold_numbers: Iterable[int] | None = None,
) -> SavedRunData:
    cfg, meta = load_run_cfg_meta(run_dir)
    model_name = model_name_from_cfg(cfg)
    subjects = resolve_subjects(root=root, subject_args=cfg["subjects"])
    subject_data = load_subject_dataset(
        root=root,
        subjects=subjects,
        go_bins=set(int(x) for x in cfg["go_bins"]),
        stop_bins=set(int(x) for x in cfg["stop_bins"]),
        crop_tmin=cfg.get("crop_tmin"),
        crop_tmax=cfg.get("crop_tmax"),
    )
    arrays = combine_subject_data(subject_data)
    trial_normalization = str(cfg.get("trial_normalization", "none"))
    if trial_normalization != "none":
        arrays["X"] = apply_trial_normalization(
            X=arrays["X"],
            times=subject_data[0].times.astype(np.float64, copy=False),
            mode=trial_normalization,
        )
    fold_dirs = sorted([p for p in run_dir.glob("fold_*") if p.is_dir()])
    if fold_numbers is not None:
        wanted = {f"fold_{int(x):02d}" for x in fold_numbers}
        fold_dirs = [p for p in fold_dirs if p.name in wanted]
    if not fold_dirs:
        raise RuntimeError(f"No matching fold directories found in {run_dir}")

    return SavedRunData(
        root=root,
        run_dir=run_dir,
        cfg=cfg,
        meta=meta,
        model_name=model_name,
        subjects=[int(s) for s in subjects],
        X_raw=arrays["X"],
        y=arrays["y"],
        subject_col=arrays["subject_col"],
        stop_signal_onset_s=arrays["stop_signal_onset_s"],
        times_s=subject_data[0].times.astype(np.float64, copy=True),
        channel_names=list(subject_data[0].ch_names),
        fold_dirs=fold_dirs,
    )


def iter_saved_run_folds(
    data: SavedRunData,
    *,
    device: Any,
    attention_enabled: bool = False,
) -> Iterable[SavedRunFold]:
    for fold_dir in data.fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[-1])

        model = build_model_from_config(
            cfg=data.cfg,
            meta=data.meta,
            in_channels=data.X_raw.shape[1],
            n_times=data.X_raw.shape[2],
            attention_enabled=attention_enabled,
        ).to(device)
        ckpt = torch.load(fold_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        split = load_split(fold_dir / "split_indices.npz")
        test_idx = split["test_idx"]
        mean, std = load_standardizer(fold_dir / "standardizer.npz")
        X_te_raw = data.X_raw[test_idx]
        X_te = apply_standardizer(X_te_raw, mean, std)

        yield SavedRunFold(
            fold_dir=fold_dir,
            fold_idx=fold_idx,
            model=model,
            test_idx=test_idx,
            X_te_raw=X_te_raw,
            X_te=X_te,
            y_te=data.y[test_idx],
        )
