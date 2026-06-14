from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.data.utils import find_subjects, load_subject_decode_data
from pipeline.misc import GO_LABEL, STOP_LABEL
from pipeline.types import SubjectDecodeData


def resolve_subjects(root: Path, subject_args: list[str]) -> list[int]:
    if subject_args == ["all"]:
        subjects = find_subjects(root)
    else:
        subjects = [int(x) for x in subject_args]
    if not subjects:
        raise RuntimeError("No subjects found to decode.")
    return subjects


def load_subject_dataset(
    root: Path,
    subjects: list[int],
    go_bins: set[int],
    stop_bins: set[int],
    crop_tmin: float | None,
    crop_tmax: float | None,
) -> list[SubjectDecodeData]:
    subject_data: list[SubjectDecodeData] = []
    reference_ch_names: list[str] | None = None
    reference_times = np.empty(0, dtype=float)

    for subj in subjects:
        data = load_subject_decode_data(
            root=root,
            subject=subj,
            go_bins=go_bins,
            stop_bins=stop_bins,
            crop_tmin=crop_tmin,
            crop_tmax=crop_tmax,
        )

        if reference_ch_names is None:
            reference_ch_names = list(data.ch_names)
            reference_times = data.times.copy()
        else:
            if data.ch_names != reference_ch_names:
                if set(data.ch_names) != set(reference_ch_names):
                    raise RuntimeError(
                        f"S{subj}: channel mismatch. Expected {reference_ch_names}, got {data.ch_names}"
                    )
                reindex = [data.ch_names.index(ch) for ch in reference_ch_names]
                data.X = data.X[:, reindex, :]
                data.ch_names = list(reference_ch_names)

            if data.times.shape != reference_times.shape or not np.allclose(
                data.times, reference_times, atol=1e-8
            ):
                raise RuntimeError(
                    f"S{subj}: time axis mismatch. Expected {reference_times.shape}, got {data.times.shape}"
                )

        subject_data.append(data)

        n_go = int((data.y == GO_LABEL).sum())
        n_stop = int((data.y == STOP_LABEL).sum())
        print(f"S{subj}: kept={len(data.y)} go={n_go} stop={n_stop}")

    return subject_data


def combine_subject_data(
    subject_data: list[SubjectDecodeData],
) -> dict[str, np.ndarray]:
    X = np.concatenate([d.X for d in subject_data], axis=0)
    y = np.concatenate([d.y for d in subject_data], axis=0)
    groups = np.concatenate(
        [np.full(len(d.y), d.subject, dtype=int) for d in subject_data], axis=0
    )
    epoch_idx = np.concatenate([d.epoch_indices for d in subject_data], axis=0)
    subject_col = groups
    sample_id = np.arange(len(y), dtype=np.int64)
    stop_signal_onset_s = np.concatenate(
        [
            d.stop_signal_onset_s
            if d.stop_signal_onset_s is not None
            else np.full(len(d.y), np.nan, dtype=np.float64)
            for d in subject_data
        ],
        axis=0,
    )

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "epoch_idx": epoch_idx,
        "subject_col": subject_col,
        "sample_id": sample_id,
        "stop_signal_onset_s": stop_signal_onset_s,
    }


def apply_trial_normalization(
    X: np.ndarray,
    times: np.ndarray,
    mode: str,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    if mode == "none":
        return X
    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape (trials, channels, times), got {X.shape}"
        )
    if times.shape[0] != X.shape[2]:
        raise ValueError(
            f"times length {times.shape[0]} does not match X n_times {X.shape[2]}"
        )

    out = X.astype(np.float32, copy=True)
    if mode in {"baseline_subtract", "baseline_zscore"}:
        baseline_mask = times < 0.0
        if not np.any(baseline_mask):
            raise ValueError(
                "Trial baseline normalization requested, but no pre-event samples are available"
            )
        baseline = out[:, :, baseline_mask]
        center = baseline.mean(axis=2, keepdims=True)
        out -= center
        if mode == "baseline_zscore":
            scale = baseline.std(axis=2, keepdims=True)
            out /= np.maximum(scale, eps)
        return out
    if mode == "epoch_zscore":
        center = out.mean(axis=2, keepdims=True)
        scale = out.std(axis=2, keepdims=True)
        return (out - center) / np.maximum(scale, float(eps))
    raise ValueError(f"Unsupported trial normalization mode: {mode}")


def prepare_decoding_dataset(
    *, args: Any, root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load, combine, normalize, and describe the go/stop decoding dataset."""
    subjects = resolve_subjects(root=root, subject_args=args.subjects)

    go_bins = set(args.go_bins)
    stop_bins = set(args.stop_bins)
    if go_bins & stop_bins:
        raise ValueError("go-bins and stop-bins must not overlap")

    subject_data = load_subject_dataset(
        root=root,
        subjects=subjects,
        go_bins=go_bins,
        stop_bins=stop_bins,
        crop_tmin=args.crop_tmin,
        crop_tmax=args.crop_tmax,
    )

    arrays = combine_subject_data(subject_data)
    trial_normalization = str(getattr(args, "trial_normalization", "none"))
    if trial_normalization != "none":
        arrays["X"] = apply_trial_normalization(
            X=arrays["X"],
            times=subject_data[0].times,
            mode=trial_normalization,
        )
        print(f"Applied trial normalization: {trial_normalization}")

    manifest: dict[str, Any] = {
        "subjects": subjects,
        "go_bins": sorted(go_bins),
        "stop_bins": sorted(stop_bins),
        "crop_tmin": args.crop_tmin,
        "crop_tmax": args.crop_tmax,
        "trial_normalization": trial_normalization,
        "n_channels": int(arrays["X"].shape[1]),
        "n_times": int(arrays["X"].shape[2]),
        "times_s": subject_data[0].times.tolist(),
        "channel_names": subject_data[0].ch_names,
    }

    n_go = int((arrays["y"] == GO_LABEL).sum())
    n_stop = int((arrays["y"] == STOP_LABEL).sum())
    if n_go == 0 or n_stop == 0:
        raise RuntimeError(f"Need both classes. got go={n_go}, stop={n_stop}")
    print(
        f"Combined: n_epochs={len(arrays['y'])} n_subjects={len(np.unique(arrays['groups']))} "
        f"go={n_go} stop={n_stop} n_channels={arrays['X'].shape[1]} n_times={arrays['X'].shape[2]}"
    )

    return arrays, manifest


def add_common_training_args(
    parser: argparse.ArgumentParser, *, loso: bool = False
) -> None:
    """Add the shared argparse arguments used by the core training entrypoints.

    loso=True configures a few options that differ for the dedicated LOSO wrapper
    (n-splits is still accepted but documented as ignored; cv-mode is forced later).
    """
    parser.add_argument("--root", default=".", help="Repo root")
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--go-bins", nargs="+", type=int, default=[1])
    parser.add_argument("--stop-bins", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--crop-tmin", type=float, default=None)
    parser.add_argument("--crop-tmax", type=float, default=None)
    parser.add_argument(
        "--trial-normalization",
        choices=["none", "baseline_subtract", "baseline_zscore", "epoch_zscore"],
        default="none",
        help=(
            "Optional per-trial normalization before fold standardization. "
            "Baseline modes use only pre-event samples from the same trial, avoiding held-out-subject aggregate statistics."
        ),
    )
    if loso:
        parser.add_argument(
            "--n-splits", type=int, default=5, help="Ignored in LOSO mode"
        )
        parser.add_argument(
            "--cv-mode",
            default="loso",
            help="Forced to loso for this entrypoint (accepted for compatibility with run_campaign).",
        )
    else:
        parser.add_argument(
            "--cv-mode",
            choices=[
                "groupkfold",
                "group",
                "stratified",
                "random",
                "loso",
                "leave-one-subject-out",
            ],
            default="groupkfold",
        )
        parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--weighted-sampling", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--loss-kind", choices=["cross_entropy", "focal"], default="cross_entropy"
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["auc", "balanced_accuracy", "average_precision", "loss"],
        default="auc",
        help="Validation quantity used for best-checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--class-weight-mode", choices=["balanced", "none"], default="balanced"
    )
    parser.add_argument(
        "--stop-loss-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the stop-class loss weight after class-weight construction.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_transformer",
        choices=[
            "cnn_transformer",
            "transformer_only",
            "pure_cnn",
            "enigma",
        ],
        help="Backbone architecture",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--cnn-width", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cnn-only-blocks", type=int, default=4)
    parser.add_argument("--cnn-only-kernel", type=int, default=5)
    parser.add_argument("--enigma-temporal-filters", type=int, default=40)
    parser.add_argument("--enigma-temporal-kernel", type=int, default=5)
    parser.add_argument("--enigma-temporal-pool-kernel", type=int, default=17)
    parser.add_argument("--enigma-temporal-pool-stride", type=int, default=5)
    parser.add_argument("--enigma-embedding-dim", type=int, default=4)
    parser.add_argument("--enigma-projector-dim", type=int, default=128)
    parser.add_argument("--enigma-projector-dropout", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--time-mask-prob", type=float, default=0.2)
    parser.add_argument("--channel-drop-prob", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--amp-dtype", choices=["auto", "bfloat16", "float16"], default="auto"
    )
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default=None
    )
    parser.add_argument("--random-state", type=int, default=9)
    parser.add_argument("--output-dir", default="sst_campaign_runs")
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional short label to include in the run folder name",
    )
    parser.add_argument(
        "--disable-versioning",
        action="store_true",
        help="Disable auto-increment run folders and write directly to output-dir",
    )
