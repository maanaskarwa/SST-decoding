"""Core causal-ablation helpers. Defines ablation specs, applies time and channel ablations, evaluates saved models, and summarizes ablation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from pipeline.eval.metrics import compute_metrics
from pipeline.misc import GO_LABEL, STOP_LABEL
from pipeline.train.driver import predict_probabilities

from .manuscript_stats import exact_sign_test_greater

MOTOR_CHANNELS = ["C3", "C4", "FC1", "FC2", "CP1", "CP2", "Cz"]
CENTROPARIETAL_CHANNELS = ["Cz", "CP1", "CP2", "Pz", "P3", "P4"]


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    mode: str  # keep_only | ablate_inside
    time_window: tuple[float, float] | None = None
    channels: list[str] | None = None


def default_ablation_specs(
    p3_window_s: tuple[float, float] = (0.25, 0.45),
) -> list[AblationSpec]:
    p3_window = (float(p3_window_s[0]), float(p3_window_s[1]))
    late_window = (p3_window[1], 0.80)
    return [
        AblationSpec("original", "Unmodified input", mode="keep_only"),
        AblationSpec(
            "keep_prestim_only",
            "Keep prestimulus window only",
            mode="keep_only",
            time_window=(-0.2, 0.0),
        ),
        AblationSpec(
            "keep_early_only",
            "Keep pre-P3 post-stimulus window only",
            mode="keep_only",
            time_window=(0.0, p3_window[0]),
        ),
        AblationSpec(
            "keep_p3_only",
            "Keep stop-relative P3 window only",
            mode="keep_only",
            time_window=p3_window,
        ),
        AblationSpec(
            "keep_late_only",
            "Keep post-P3 late window only",
            mode="keep_only",
            time_window=late_window,
        ),
        AblationSpec(
            "ablate_p3_window",
            "Zero stop-relative P3 window",
            mode="ablate_inside",
            time_window=p3_window,
        ),
        AblationSpec(
            "ablate_late_window",
            "Zero post-P3 late window",
            mode="ablate_inside",
            time_window=late_window,
        ),
        AblationSpec(
            "keep_motor_only",
            "Keep motor ROI channels only",
            mode="keep_only",
            channels=MOTOR_CHANNELS,
        ),
        AblationSpec(
            "ablate_motor_channels",
            "Zero motor ROI channels",
            mode="ablate_inside",
            channels=MOTOR_CHANNELS,
        ),
        AblationSpec(
            "keep_centroparietal_only",
            "Keep centro-parietal ROI only",
            mode="keep_only",
            channels=CENTROPARIETAL_CHANNELS,
        ),
        AblationSpec(
            "ablate_centroparietal",
            "Zero centro-parietal ROI",
            mode="ablate_inside",
            channels=CENTROPARIETAL_CHANNELS,
        ),
        AblationSpec(
            "keep_centroparietal_p3_only",
            "Keep centro-parietal ROI in stop-relative P3 window only",
            mode="keep_only",
            time_window=p3_window,
            channels=CENTROPARIETAL_CHANNELS,
        ),
        AblationSpec(
            "ablate_centroparietal_p3",
            "Zero centro-parietal ROI in stop-relative P3 window",
            mode="ablate_inside",
            time_window=p3_window,
            channels=CENTROPARIETAL_CHANNELS,
        ),
    ]


def channel_indices(channel_names: list[str], requested: list[str] | None) -> list[int]:
    if not requested:
        return []
    lookup = {name: idx for idx, name in enumerate(channel_names)}
    missing = [ch for ch in requested if ch not in lookup]
    if missing:
        raise ValueError(f"Requested ablation channels were not found: {missing}")
    return [lookup[ch] for ch in requested]


def time_mask(times_s: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    if window is None:
        return np.ones(len(times_s), dtype=bool)
    tmin, tmax = window
    return (times_s >= float(tmin)) & (times_s <= float(tmax))


def apply_ablation(
    X: np.ndarray,
    *,
    times_s: np.ndarray,
    channel_names: list[str],
    spec: AblationSpec,
) -> np.ndarray:
    out = X.copy()
    if spec.name == "original":
        return out

    if spec.channels is None and spec.time_window is None:
        raise ValueError(f"Ablation spec {spec.name!r} would be a no-op")

    ch_idx = channel_indices(channel_names=channel_names, requested=spec.channels)
    t_mask = time_mask(times_s=times_s, window=spec.time_window)

    if spec.mode == "keep_only":
        keep = np.zeros_like(out)
        if spec.channels is None:
            keep[:, :, t_mask] = out[:, :, t_mask]
        elif spec.time_window is None:
            keep[:, ch_idx, :] = out[:, ch_idx, :]
        else:
            for ch in ch_idx:
                keep[:, ch, t_mask] = out[:, ch, t_mask]
        return keep

    if spec.mode == "ablate_inside":
        if spec.channels is None:
            out[:, :, t_mask] = 0.0
        elif spec.time_window is None:
            out[:, ch_idx, :] = 0.0
        else:
            for ch in ch_idx:
                out[:, ch, t_mask] = 0.0
        return out

    raise ValueError(f"Unknown ablation mode: {spec.mode}")


def evaluate_ablation_specs(
    model: torch.nn.Module,
    *,
    X: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    channel_names: list[str],
    device: torch.device,
    specs: list[AblationSpec],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    original_metrics: dict[str, float] | None = None
    original_p_stop: np.ndarray | None = None
    stop_mask = y == STOP_LABEL

    for spec in specs:
        X_mod = apply_ablation(
            X=X, times_s=times_s, channel_names=channel_names, spec=spec
        )
        p_stop, y_pred = predict_probabilities(
            model=model,
            x_np=X_mod,
            device=device,
        )
        metrics = compute_metrics(
            y_true=y,
            y_pred=y_pred,
            y_prob_stop=p_stop,
        )
        average_precision = float(metrics["average_precision"])
        if original_metrics is None or original_p_stop is None:
            original_metrics = {**metrics, "average_precision": average_precision}
            original_p_stop = p_stop.copy()
        assert original_metrics is not None
        assert original_p_stop is not None
        mean_p_stop_true_stop = (
            float(p_stop[stop_mask].mean()) if stop_mask.any() else float("nan")
        )
        original_mean_p_stop_true_stop = (
            float(original_p_stop[stop_mask].mean())
            if stop_mask.any()
            else float("nan")
        )
        rows.append(
            {
                **metadata,
                "ablation_name": spec.name,
                "ablation_description": spec.description,
                "ablation_mode": spec.mode,
                "ablation_time_window_s": ""
                if spec.time_window is None
                else f"{float(spec.time_window[0]):.6g},{float(spec.time_window[1]):.6g}",
                "ablation_channels": ""
                if spec.channels is None
                else ",".join(spec.channels),
                "test_auc": float(metrics["auc"]),
                "test_accuracy": float(metrics["accuracy"]),
                "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "test_average_precision": average_precision,
                "delta_auc_vs_original": float(
                    metrics["auc"] - original_metrics["auc"]
                ),
                "delta_accuracy_vs_original": float(
                    metrics["accuracy"] - original_metrics["accuracy"]
                ),
                "delta_balanced_accuracy_vs_original": float(
                    metrics["balanced_accuracy"] - original_metrics["balanced_accuracy"]
                ),
                "delta_average_precision_vs_original": float(
                    average_precision - original_metrics["average_precision"]
                ),
                "mean_p_stop_true_stop": mean_p_stop_true_stop,
                "drop_mean_p_stop_true_stop_vs_original": float(
                    original_mean_p_stop_true_stop - mean_p_stop_true_stop
                ),
                "n_test": int(len(y)),
                "n_go": int((y == GO_LABEL).sum()),
                "n_stop": int((y == STOP_LABEL).sum()),
            }
        )

    return pd.DataFrame(rows)


def summarize_ablation_table(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if df.empty:
        return out

    original_rows = df[df["ablation_name"] == "original"]
    original = {
        "auc": float(original_rows["test_auc"].mean()),
        "accuracy": float(original_rows["test_accuracy"].mean()),
        "balanced_accuracy": float(original_rows["test_balanced_accuracy"].mean()),
        "average_precision": float(original_rows["test_average_precision"].mean()),
    }
    out["original_metrics"] = {
        "auc": float(original["auc"]),
        "accuracy": float(original["accuracy"]),
        "balanced_accuracy": float(original["balanced_accuracy"]),
        "average_precision": float(original["average_precision"]),
    }

    mean_df = df.groupby("ablation_name", as_index=False).agg(
        mean_auc=("test_auc", "mean"),
        mean_accuracy=("test_accuracy", "mean"),
        mean_balanced_accuracy=("test_balanced_accuracy", "mean"),
        mean_average_precision=("test_average_precision", "mean"),
        mean_delta_auc=("delta_auc_vs_original", "mean"),
        mean_delta_accuracy=("delta_accuracy_vs_original", "mean"),
        mean_delta_bal_acc=("delta_balanced_accuracy_vs_original", "mean"),
        mean_delta_average_precision=("delta_average_precision_vs_original", "mean"),
        mean_p_stop_true_stop=("mean_p_stop_true_stop", "mean"),
        mean_drop_p_stop_true_stop=("drop_mean_p_stop_true_stop_vs_original", "mean"),
        n_rows=("ablation_name", "count"),
    )
    mean_df = mean_df.iloc[
        np.argsort(np.asarray(mean_df["mean_delta_bal_acc"], dtype=float))
    ].reset_index(drop=True)
    out["mean_table"] = mean_df.to_dict(orient="records")
    if not mean_df.empty:
        drop_tests = []
        for ablation_name, ablation_df in df.groupby("ablation_name"):
            if ablation_name == "original":
                continue
            drop_bal_acc = -ablation_df["delta_balanced_accuracy_vs_original"].to_numpy(
                dtype=float
            )
            drop_p_stop = ablation_df[
                "drop_mean_p_stop_true_stop_vs_original"
            ].to_numpy(dtype=float)
            drop_tests.append(
                {
                    "ablation_name": str(ablation_name),
                    "n_rows": int(len(ablation_df)),
                    "mean_drop_balanced_accuracy": float(np.nanmean(drop_bal_acc)),
                    "sign_test_p_drop_balanced_accuracy_one_sided": exact_sign_test_greater(
                        drop_bal_acc
                    ),
                    "mean_drop_p_stop_true_stop": float(np.nanmean(drop_p_stop)),
                    "sign_test_p_drop_p_stop_true_stop_one_sided": exact_sign_test_greater(
                        drop_p_stop
                    ),
                }
            )
        out["paired_drop_tests"] = sorted(
            drop_tests,
            key=lambda row: float(row["mean_drop_p_stop_true_stop"]),
            reverse=True,
        )
        worst = mean_df.iloc[0].to_dict()
        retained_df = mean_df[mean_df["ablation_name"] != "original"].copy()
        best_keep = worst
        if not retained_df.empty:
            best_idx = int(
                np.argmax(
                    np.asarray(retained_df["mean_balanced_accuracy"], dtype=float)
                )
            )
            best_keep = retained_df.iloc[best_idx].to_dict()
        out["strongest_drop"] = worst
        out["best_retained"] = best_keep
    return out


def choose_constrained_retrain_pair(
    summary: dict[str, Any],
    p3_window_s: tuple[float, float] = (0.25, 0.45),
) -> dict[str, Any]:
    p3_window = (float(p3_window_s[0]), float(p3_window_s[1]))
    mean_table = summary.get("mean_table", [])
    by_name = {row["ablation_name"]: row for row in mean_table}
    early = by_name.get("keep_early_only")
    late = by_name.get("keep_late_only")
    p3 = by_name.get("keep_p3_only")

    if (
        late
        and early
        and late["mean_balanced_accuracy"] >= early["mean_balanced_accuracy"]
    ):
        return {
            "family": "time_window",
            "pair_name": "early_vs_late",
            "train_jobs": [
                {"label": "early_only", "crop_tmin": 0.0, "crop_tmax": p3_window[0]},
                {"label": "late_only", "crop_tmin": p3_window[1], "crop_tmax": 0.80},
            ],
        }
    if p3 and early:
        return {
            "family": "time_window",
            "pair_name": "early_vs_p3",
            "train_jobs": [
                {"label": "early_only", "crop_tmin": 0.0, "crop_tmax": p3_window[0]},
                {
                    "label": "p3_only",
                    "crop_tmin": p3_window[0],
                    "crop_tmax": p3_window[1],
                },
            ],
        }
    return {
        "family": "time_window",
        "pair_name": "early_vs_late",
        "train_jobs": [
            {"label": "early_only", "crop_tmin": 0.0, "crop_tmax": p3_window[0]},
            {"label": "late_only", "crop_tmin": p3_window[1], "crop_tmax": 0.80},
        ],
    }
