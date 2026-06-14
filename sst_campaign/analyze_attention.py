"""Attention analysis CLI for saved transformer-family SST campaign runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import torch

from pipeline.interpretability.plot import (
    plot_channel_time_heatmap,
    plot_relevance_curves,
)
from pipeline.misc import GO_LABEL, STOP_LABEL, save_json
from pipeline.perf import configure_torch_runtime, resolve_device
from pipeline.run_versioning import prepare_versioned_output_dir
from pipeline.train.driver import select_indices, set_random_seed
from sst_campaign.utils.common import emit_run_output_dir
from sst_campaign.utils.model_specs import supports_attention
from sst_campaign.utils.saved_run_context import (
    SavedRunFold,
    iter_saved_run_folds,
    load_saved_run_data,
)
from sst_campaign.utils.stop_windows import resolve_p3_window


class _AttentionModel(Protocol):
    def encode_with_attention(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_attention_times(self, original_times_s: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class _AttentionOutputPaths:
    heatmap_png: Path
    layer_curve_png: Path
    rollout_png: Path
    layer_csv: Path
    rollout_csv: Path
    summary_json: Path
    arrays_npz: Path
    selection_csv: Path
    subject_stats_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention analysis for transformer-family SST campaign runs."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default=None
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--target-label", choices=["stop", "go"], default="stop")
    parser.add_argument("--include-misclassified", action="store_true")
    parser.add_argument(
        "--p3-tmin",
        type=float,
        default=0.25,
        help="P3 window start after stop signal, in seconds.",
    )
    parser.add_argument(
        "--p3-tmax",
        type=float,
        default=0.45,
        help="P3 window end after stop signal, in seconds.",
    )
    parser.add_argument("--rollout-permutations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=9)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--disable-versioning", action="store_true")
    return parser.parse_args()


def _record_attention(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
) -> np.ndarray:
    _, attn = cast(_AttentionModel, model).encode_with_attention(x)
    return attn.detach().cpu().numpy()


def _float_tensor(array: np.ndarray, device: Any) -> torch.Tensor:
    return torch.Tensor(array.astype(np.float32, copy=False)).to(device=device)


def _predict_fold_labels(
    *,
    fold: SavedRunFold,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    """Run batched inference for a saved fold and return predicted labels."""
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fold.X_te), batch_size):
            stop = start + batch_size
            batch = _float_tensor(fold.X_te[start:stop], device=device)
            logits = fold.model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _temporal_attention(attn: np.ndarray) -> np.ndarray:
    """Map recorded attention to time-token attention curves."""
    # attn: (layers, batch, heads, tokens, tokens)
    # pull out first token (CLS), return that and the time tokens
    return attn[:, :, :, 0, 1:]


def _sample_rollout(attn_layers: np.ndarray) -> np.ndarray:
    tokens = attn_layers.shape[-1]
    result = np.eye(tokens, dtype=np.float64)
    for layer in attn_layers:
        aug = layer + np.eye(tokens, dtype=np.float64)
        aug = aug / np.clip(aug.sum(axis=-1, keepdims=True), 1e-12, None)
        result = aug @ result
    return result[0, 1:]


def _curve_window_share(curves: np.ndarray, mask: np.ndarray) -> np.ndarray:
    totals = np.clip(curves.sum(axis=1), 1e-12, None)
    return curves[:, mask].sum(axis=1) / totals


def _mean_p3_share_by_layer(
    layer_mean_curves: dict[str, np.ndarray],
    p3_mask: np.ndarray,
    n_layers: int,
) -> dict[str, float]:
    shares: dict[str, float] = {}
    for i in range(n_layers):
        layer_name = f"layer_{i + 1}"
        curve = layer_mean_curves[layer_name]
        shares[layer_name] = float(curve[p3_mask].sum() / max(curve.sum(), 1e-12))
    return shares


def _rollout_p3_statistics(
    *,
    rollout_curves: np.ndarray,
    subjects: np.ndarray,
    p3_mask: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    """Subject-level enrichment tests for rollout mass in the P3 window.

    These tests ask whether the selected P3 window receives more rollout mass
    than expected by either a uniform-over-time baseline or equally wide random
    contiguous time windows. They are descriptive statistics for attention
    concentration, not proof that attention is a causal explanation.
    """
    if len(subjects) != rollout_curves.shape[0]:
        raise ValueError(
            "subjects must contain one entry per rollout curve: "
            f"got {len(subjects)} subjects for {rollout_curves.shape[0]} curves"
        )
    grouped_curves = (
        pd.DataFrame(rollout_curves)
        .assign(subject=subjects)
        .groupby("subject", sort=True)
        .mean()
    )
    unique_subjects = grouped_curves.index.to_numpy(dtype=np.int64)
    subject_curves = grouped_curves.to_numpy()
    subject_shares = _curve_window_share(subject_curves, p3_mask)
    uniform_share = float(p3_mask.mean())
    observed_mean_share = float(subject_shares.mean())
    observed_enrichment = float(observed_mean_share - uniform_share)

    n_permutations = max(0, int(n_permutations))
    uniform_p = None
    random_window_p = None
    if n_permutations > 0 and len(subject_shares) > 1:
        centered = subject_shares - uniform_share
        signs = rng.choice(
            np.asarray([-1.0, 1.0]), size=(n_permutations, len(centered))
        )
        null_enrichment = (signs * centered[None, :]).mean(axis=1)
        uniform_p = float(
            (np.count_nonzero(null_enrichment >= observed_enrichment) + 1)
            / (n_permutations + 1)
        )

        window_size = int(p3_mask.sum())
        n_times = int(p3_mask.size)
        if 0 < window_size < n_times:
            random_enrichment = np.empty(n_permutations, dtype=np.float64)
            for perm_idx in range(n_permutations):
                shares: list[float] = []
                for curve in subject_curves:
                    start = int(rng.integers(0, n_times - window_size + 1))
                    mask = np.zeros(n_times, dtype=bool)
                    mask[start : start + window_size] = True
                    shares.append(float(_curve_window_share(curve[None, :], mask)[0]))
                random_enrichment[perm_idx] = float(np.mean(shares) - uniform_share)
            random_window_p = float(
                (np.count_nonzero(random_enrichment >= observed_enrichment) + 1)
                / (n_permutations + 1)
            )

    subject_df = pd.DataFrame(
        {
            "subject": unique_subjects.astype(int),
            "rollout_p3_share": subject_shares.astype(float),
            "rollout_p3_enrichment_over_uniform": (
                subject_shares - uniform_share
            ).astype(float),
        }
    )
    stats = {
        "n_rollout_samples": int(rollout_curves.shape[0]),
        "n_subjects": int(len(unique_subjects)),
        "uniform_time_share": uniform_share,
        "subject_mean_rollout_p3_share": observed_mean_share,
        "subject_mean_rollout_p3_enrichment_over_uniform": observed_enrichment,
        "subject_permutation_p_uniform_enrichment_one_sided": uniform_p,
        "random_window_p_enrichment_one_sided": random_window_p,
        "n_permutations": int(n_permutations),
        "note": (
            "Tests attention-rollout concentration in the P3 window; pair with ablation/sanity checks "
            "before claiming causal explanation."
        ),
    }
    return stats, subject_df, subject_curves


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_data = load_saved_run_data(root=root, run_dir=run_dir)
    if not supports_attention(run_data.model_name):
        raise RuntimeError(
            f"Attention analysis only supports attention-capable runs, got {run_data.model_name}"
        )
    p3_window = resolve_p3_window(
        y=run_data.y,
        stop_signal_onset_s=run_data.stop_signal_onset_s,
        p3_tmin=float(args.p3_tmin),
        p3_tmax=float(args.p3_tmax),
    )

    set_random_seed(int(args.random_state))
    rng = np.random.default_rng(int(args.random_state))
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        matmul_precision=args.matmul_precision,
    )
    target_label = STOP_LABEL if args.target_label == "stop" else GO_LABEL

    cls_attention_sum = None
    cls_attention_count = 0
    rollout_sum = None
    rollout_count = 0
    selected_rows: list[dict[str, object]] = []
    rollout_sample_curves: list[np.ndarray] = []
    rollout_subjects: list[int] = []
    reduced_times = None
    max_samples_per_fold = max(
        1, int(math.ceil(int(args.max_samples) / len(run_data.fold_dirs)))
    )

    for fold in iter_saved_run_folds(run_data, device=device, attention_enabled=True):
        model = fold.model
        model.eval()

        y_pred = _predict_fold_labels(
            fold=fold, device=device, batch_size=int(args.batch_size)
        )
        selected = select_indices(
            y_true=fold.y_te,
            y_pred=y_pred,
            target_label=target_label,
            max_samples=max_samples_per_fold,
            include_misclassified=bool(args.include_misclassified),
            rng=rng,
        )
        if len(selected) == 0:
            continue
        X_sel = _float_tensor(fold.X_te[selected], device=device)
        with torch.no_grad():
            attn_np = _record_attention(model=model, x=X_sel)
        cls_to_tokens = _temporal_attention(attn_np)
        if reduced_times is None:
            reduced_times = model.get_attention_times(run_data.times_s)[
                : cls_to_tokens.shape[-1]
            ]
        batch_sum = cls_to_tokens.sum(axis=1)
        cls_attention_sum = (
            batch_sum if cls_attention_sum is None else (cls_attention_sum + batch_sum)
        )
        cls_attention_count += int(cls_to_tokens.shape[1])

        selected_subjects = run_data.subject_col[fold.test_idx[selected]]
        attn_mean_heads = attn_np.mean(axis=2)  # (layers,batch,tokens,tokens)
        for sample_idx in range(attn_mean_heads.shape[1]):
            rollout_curve = _sample_rollout(attn_mean_heads[:, sample_idx])
            rollout_sum = (
                rollout_curve if rollout_sum is None else (rollout_sum + rollout_curve)
            )
            rollout_sample_curves.append(rollout_curve)
            rollout_subjects.append(int(selected_subjects[sample_idx]))
            rollout_count += 1
        selected_rows.append(
            {
                "fold": int(fold.fold_idx),
                "n_selected": int(len(selected)),
                "n_subjects": int(len(np.unique(selected_subjects))),
            }
        )

    if cls_attention_sum is None or rollout_sum is None or reduced_times is None:
        raise RuntimeError("No attention-selected samples available")
    mean_cls_attention = cls_attention_sum / float(
        cls_attention_count
    )  # (layers, heads, reduced_times)
    layers, heads, _ = mean_cls_attention.shape
    flat_heatmap = mean_cls_attention.reshape(
        layers * heads, mean_cls_attention.shape[-1]
    )
    heatmap_labels = [
        f"L{layer_idx + 1}H{head_idx + 1}"
        for layer_idx in range(layers)
        for head_idx in range(heads)
    ]
    layer_mean_curves = {
        f"layer_{i + 1}": mean_cls_attention[i].mean(axis=0) for i in range(layers)
    }
    rollout_curve = rollout_sum / float(rollout_count)
    uniform_rollout_curve = np.full_like(
        rollout_curve,
        fill_value=1.0 / float(len(rollout_curve)),
        dtype=np.float64,
    )
    p3_mask = p3_window.mask(reduced_times)
    if not p3_mask.any():
        raise RuntimeError(
            f"P3 window has no samples in attention time axis: {p3_window.window_s}"
        )
    rollout_sample_array = np.stack(rollout_sample_curves, axis=0)
    rollout_subject_array = np.asarray(rollout_subjects, dtype=np.int64)
    rollout_stats, rollout_subject_df, subject_rollout_curves = _rollout_p3_statistics(
        rollout_curves=rollout_sample_array,
        subjects=rollout_subject_array,
        p3_mask=p3_mask,
        rng=np.random.default_rng(int(args.random_state) + 1009),
        n_permutations=int(args.rollout_permutations),
    )

    out_base = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "attention_results")
    )
    out_dir, version_meta = prepare_versioned_output_dir(
        base_output_dir=out_base,
        experiment_name="sst_campaign_analyze_attention",
        config=vars(args),
        disable_versioning=bool(args.disable_versioning),
    )
    print(f"Run output dir: {out_dir}")
    emit_run_output_dir(out_dir)
    print(f"Run version meta: {json.dumps(version_meta)}")
    paths = _AttentionOutputPaths(
        heatmap_png=out_dir / "cls_attention_heatmap.png",
        layer_curve_png=out_dir / "layer_mean_attention_curves.png",
        rollout_png=out_dir / "attention_rollout_curve.png",
        layer_csv=out_dir / "layer_mean_attention_curves.csv",
        rollout_csv=out_dir / "attention_rollout_curve.csv",
        summary_json=out_dir / "attention_summary.json",
        arrays_npz=out_dir / "attention_arrays.npz",
        selection_csv=out_dir / "attention_selection.csv",
        subject_stats_csv=out_dir / "attention_rollout_subject_stats.csv",
    )

    plot_channel_time_heatmap(
        values=flat_heatmap,
        times_s=reduced_times,
        channel_names=heatmap_labels,
        out_path=paths.heatmap_png,
        title="CLS-to-time attention by layer/head",
        p3_window=p3_window.window_s,
        colorbar_label="Attention weight",
    )
    plot_relevance_curves(
        times_s=reduced_times,
        curves=layer_mean_curves,
        out_path=paths.layer_curve_png,
        title="Mean CLS attention by layer",
        ylabel="Attention weight",
        p3_window=p3_window.window_s,
    )
    plot_relevance_curves(
        times_s=reduced_times,
        curves={
            "attention rollout": rollout_curve,
            "uniform time-token baseline": uniform_rollout_curve,
        },
        out_path=paths.rollout_png,
        title="Attention rollout to time tokens",
        ylabel="Rollout weight",
        p3_window=p3_window.window_s,
    )

    pd.DataFrame({"time_s": reduced_times, **layer_mean_curves}).to_csv(
        paths.layer_csv, index=False
    )
    pd.DataFrame(
        {
            "time_s": reduced_times,
            "rollout": rollout_curve,
            "uniform_time_token_baseline": uniform_rollout_curve,
        }
    ).to_csv(paths.rollout_csv, index=False)
    pd.DataFrame(selected_rows).to_csv(paths.selection_csv, index=False)
    rollout_subject_df.to_csv(paths.subject_stats_csv, index=False)
    np.savez_compressed(
        paths.arrays_npz,
        mean_cls_attention=mean_cls_attention,
        reduced_times_s=reduced_times,
        rollout_curve=rollout_curve,
        rollout_sample_curves=rollout_sample_array,
        rollout_sample_subjects=rollout_subject_array,
        subject_rollout_curves=subject_rollout_curves,
        subject_rollout_subjects=rollout_subject_df["subject"].to_numpy(dtype=np.int64),
    )

    summary = {
        "source_run_dir": str(run_dir),
        "model_name": run_data.model_name,
        "device": str(device),
        **p3_window.summary(),
        "n_layers": int(layers),
        "n_heads": int(heads),
        "mean_p3_share_by_layer": _mean_p3_share_by_layer(
            layer_mean_curves, p3_mask, layers
        ),
        "rollout_peak_time_s": float(reduced_times[int(np.argmax(rollout_curve))]),
        "rollout_p3_share": float(
            rollout_curve[p3_mask].sum() / max(rollout_curve.sum(), 1e-12)
        ),
        "uniform_time_token_weight": float(uniform_rollout_curve[0]),
        "rollout_p3_statistics": rollout_stats,
        "outputs": {
            "heatmap_png": str(paths.heatmap_png),
            "layer_curve_png": str(paths.layer_curve_png),
            "rollout_png": str(paths.rollout_png),
            "layer_csv": str(paths.layer_csv),
            "rollout_csv": str(paths.rollout_csv),
            "arrays_npz": str(paths.arrays_npz),
            "selection_csv": str(paths.selection_csv),
            "subject_stats_csv": str(paths.subject_stats_csv),
        },
    }
    save_json(paths.summary_json, summary)
    print(f"Saved: {paths.summary_json}")


if __name__ == "__main__":
    main()
