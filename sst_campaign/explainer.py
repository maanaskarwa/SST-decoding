"""Traditional methods (SHAP, LIME, DeepLift) etc"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, KernelShap, LayerAttribution, LayerGradCam, Lime
from torch import nn

from pipeline.interpretability.explainers import build_patch_feature_mask
from pipeline.interpretability.plot import (
    plot_channel_time_heatmap,
    plot_relevance_curves,
    plot_scalar_history,
)
from pipeline.interpretability_helpers import (
    summarize_time_curve,
    time_window_to_num_indices,
)
from pipeline.misc import GO_LABEL, STOP_LABEL, save_json
from pipeline.perf import configure_torch_runtime, resolve_device
from pipeline.run_versioning import prepare_versioned_output_dir
from pipeline.train.driver import (
    map_roi_indices,
    predict_probabilities,
    select_indices,
    set_random_seed,
)
from sst_campaign.utils.common import emit_run_output_dir
from sst_campaign.utils.saved_run_context import (
    iter_saved_run_folds,
    load_saved_run_data,
)
from sst_campaign.utils.stop_windows import resolve_p3_window

DEFAULT_METHODS = ["deeplift", "cam", "lime", "shap", "am"]
DEFAULT_ROI_CHANNELS = ["Cz", "CP1", "CP2", "Pz", "P3", "P4"]
MAP_METHODS = {"deeplift", "lime", "shap"}
TIME_ONLY_METHODS = {"cam"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic explainer suite for SST campaign runs."
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
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS
    )
    parser.add_argument("--target-label", choices=["stop", "go"], default="stop")
    parser.add_argument("--explainer-batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-surrogate-samples", type=int, default=8)
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
    parser.add_argument("--roi-channels", nargs="+", default=DEFAULT_ROI_CHANNELS)
    parser.add_argument("--feature-window-ms", type=float, default=64.0)
    parser.add_argument("--feature-channel-group", type=int, default=4)
    parser.add_argument("--lime-samples", type=int, default=64)
    parser.add_argument("--shap-samples", type=int, default=64)
    parser.add_argument("--perturbations-per-eval", type=int, default=4)
    parser.add_argument("--am-steps", type=int, default=200)
    parser.add_argument("--am-lr", type=float, default=0.05)
    parser.add_argument("--am-l2", type=float, default=1e-3)
    parser.add_argument("--am-tv", type=float, default=1e-3)
    parser.add_argument("--am-jitter-std", type=float, default=0.0)
    parser.add_argument("--am-clamp-abs", type=float, default=5.0)
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=9)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--disable-versioning", action="store_true")
    return parser.parse_args()



def _grad_cam_values(
    model: nn.Module,
    target_layer: nn.Module,
    batch: torch.Tensor,
    target_label: int,
    input_time_dim: int,
) -> torch.Tensor:
    raw_attr = LayerGradCam(model, target_layer).attribute(
        batch, target=target_label, relu_attributions=True, attr_dim_summation=True
    )
    raw = cast(torch.Tensor, raw_attr[0] if isinstance(raw_attr, tuple) else raw_attr)
    if raw.ndim == 4 and raw.shape[2] == 1:
        raw = raw.squeeze(2)
    if raw.shape[-1] != input_time_dim:
        raw = LayerAttribution.interpolate(
            raw, input_time_dim, interpolate_mode="linear"
        )
    if raw.ndim == 2:
        raw = raw.unsqueeze(1)
    return raw.detach()


def _target_layer(model: torch.nn.Module, model_name: str) -> nn.Module:
    if model_name in {"cnn_transformer"}:
        layers = cast(Any, getattr(model, "temporal_frontend"))
        return cast(nn.Module, layers[9])
    if model_name == "transformer_only":
        return cast(nn.Module, getattr(model, "input_projection"))
    if model_name in {"pure_cnn"}:
        layers = cast(Any, getattr(model, "features"))
        return cast(nn.Module, layers[-2])
    if model_name in {"enigma", "enigma_style"}:
        return cast(nn.Module, getattr(model, "embedding_conv"))
    raise ValueError(
        f"Unsupported model for Grad-CAM target-layer selection: {model_name}"
    )


def _float_tensor(array: np.ndarray, device: Any) -> torch.Tensor:
    return torch.Tensor(array.astype(np.float32, copy=False)).to(device=device)


def _activation_maximization(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    target_label: int,
    steps: int,
    lr: float,
    l2_coeff: float,
    tv_coeff: float,
    jitter_std: float,
    clamp_abs: float | None,
    device: Any,
) -> tuple[torch.Tensor, list[float], list[float]]:
    x = 0.01 * torch.randn(input_shape, device=device)
    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    objective_history: list[float] = []
    probability_history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        other_label = GO_LABEL if target_label == STOP_LABEL else STOP_LABEL
        margin = (logits[:, target_label] - logits[:, other_label]).mean()
        l2_penalty = x.pow(2).mean()
        tv_penalty = (x[..., 1:] - x[..., :-1]).pow(2).mean()
        objective = margin - (l2_coeff * l2_penalty) - (tv_coeff * tv_penalty)
        (-objective).backward()
        optimizer.step()
        with torch.no_grad():
            if jitter_std > 0:
                x.add_(jitter_std * x.new_empty(x.shape).normal_())
            if clamp_abs is not None:
                x.clamp_(-float(clamp_abs), float(clamp_abs))
            probs = nn.functional.softmax(model(x), dim=1)[:, target_label]
            objective_history.append(float(objective.detach().cpu()))
            probability_history.append(float(probs.mean().detach().cpu()))
    return x.detach(), objective_history, probability_history


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_data = load_saved_run_data(root=root, run_dir=run_dir, fold_numbers=args.folds)

    set_random_seed(int(args.random_state))
    rng = np.random.default_rng(int(args.random_state))
    times_s = run_data.times_s
    channel_names = run_data.channel_names
    p3_window = resolve_p3_window(
        y=run_data.y,
        stop_signal_onset_s=run_data.stop_signal_onset_s,
        p3_tmin=float(args.p3_tmin),
        p3_tmax=float(args.p3_tmax),
    )
    p3_mask = p3_window.mask(times_s)
    roi_indices = map_roi_indices(
        channel_names=channel_names, roi_channels=[str(x) for x in args.roi_channels]
    )
    if not roi_indices:
        raise RuntimeError(
            f"None of the ROI channels were found. Requested={args.roi_channels}, available subset={channel_names[:12]}"
        )
    if not p3_mask.any():
        raise RuntimeError(
            f"No samples fall in requested P3 window {p3_window.window_s} for available times"
        )
    pz_index = (
        channel_names.index("Pz")
        if "Pz" in channel_names
        else roi_indices[len(roi_indices) // 2]
    )
    target_label = STOP_LABEL if args.target_label == "stop" else GO_LABEL
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        matmul_precision=args.matmul_precision,
    )
    time_group_size = time_window_to_num_indices(
        times_s=times_s, window_ms=float(args.feature_window_ms)
    )
    feature_mask = build_patch_feature_mask(
        n_channels=run_data.X_raw.shape[1],
        n_times=run_data.X_raw.shape[2],
        channel_group_size=int(args.feature_channel_group),
        time_group_size=time_group_size,
        device=device,
    )

    per_method_abs_sums = {
        method: np.zeros(
            (run_data.X_raw.shape[1], run_data.X_raw.shape[2]), dtype=np.float64
        )
        for method in args.methods
        if method in MAP_METHODS
    }
    per_method_signed_sums = {
        method: np.zeros(
            (run_data.X_raw.shape[1], run_data.X_raw.shape[2]), dtype=np.float64
        )
        for method in args.methods
        if method in MAP_METHODS
    }
    per_method_counts = {method: 0 for method in args.methods if method in MAP_METHODS}
    time_curve_sums = {
        method: np.zeros(run_data.X_raw.shape[2], dtype=np.float64)
        for method in args.methods
        if method in TIME_ONLY_METHODS
    }
    time_curve_counts = {
        method: 0 for method in args.methods if method in TIME_ONLY_METHODS
    }
    am_sum = np.zeros(
        (run_data.X_raw.shape[1], run_data.X_raw.shape[2]), dtype=np.float64
    )
    am_objective_sum = np.zeros(int(args.am_steps), dtype=np.float64)
    am_probability_sum = np.zeros(int(args.am_steps), dtype=np.float64)
    am_count = 0
    selection_rows: list[dict[str, Any]] = []

    for fold in iter_saved_run_folds(run_data, device=device):
        model = fold.model.to(device)
        model.eval()

        p_stop, y_pred = predict_probabilities(
            model=model,
            x_np=fold.X_te,
            device=device,
        )
        per_fold_target_max = max(
            1, int(math.ceil(int(args.max_samples) / len(run_data.fold_dirs)))
        )
        per_fold_surrogate_max = max(
            1,
            int(math.ceil(int(args.max_surrogate_samples) / len(run_data.fold_dirs))),
        )
        selected = select_indices(
            y_true=fold.y_te,
            y_pred=y_pred,
            target_label=target_label,
            max_samples=per_fold_target_max,
            include_misclassified=bool(args.include_misclassified),
            rng=rng,
        )
        surrogate_selected = selected[: min(len(selected), per_fold_surrogate_max)]
        X_sel_std = fold.X_te[selected]
        X_surrogate_std = fold.X_te[surrogate_selected]
        selection_rows.append(
            {
                "fold": int(fold.fold_idx),
                "n_test": int(len(fold.test_idx)),
                "n_target_total": int((fold.y_te == target_label).sum()),
                "n_selected": int(len(selected)),
                "n_surrogate_selected": int(len(surrogate_selected)),
                "accuracy": float((fold.y_te == y_pred).mean()),
                "mean_p_stop": float(p_stop.mean()),
            }
        )
        target_layer = _target_layer(model=model, model_name=run_data.model_name)

        if "deeplift" in args.methods:
            deeplift = DeepLift(model)
            for start in range(0, len(X_sel_std), int(args.explainer_batch_size)):
                batch_np = X_sel_std[start : start + int(args.explainer_batch_size)]
                batch = _float_tensor(batch_np, device=device)
                attr = deeplift.attribute(
                    batch, baselines=batch.new_zeros(batch.shape), target=target_label
                ).detach()
                per_method_abs_sums["deeplift"] += attr.abs().sum(dim=0).cpu().numpy()
                per_method_signed_sums["deeplift"] += attr.sum(dim=0).cpu().numpy()
                per_method_counts["deeplift"] += int(attr.shape[0])

        if "cam" in args.methods:
            for start in range(0, len(X_sel_std), int(args.explainer_batch_size)):
                batch_np = X_sel_std[start : start + int(args.explainer_batch_size)]
                batch = _float_tensor(batch_np, device=device)
                cam = _grad_cam_values(
                    model=model,
                    target_layer=target_layer,
                    batch=batch,
                    target_label=target_label,
                    input_time_dim=run_data.X_raw.shape[2],
                )
                if cam.ndim == 3:
                    cam_curve = cam.abs().mean(dim=1).sum(dim=0).cpu().numpy()
                else:
                    cam_curve = cam.abs().sum(dim=0).cpu().numpy()
                time_curve_sums["cam"] += cam_curve
                time_curve_counts["cam"] += int(cam.shape[0])

        if "lime" in args.methods and len(X_surrogate_std) > 0:
            lime = Lime(model)
            for sample_idx, sample in enumerate(X_surrogate_std):
                x_sample = _float_tensor(sample[None, ...], device=device)
                attr = lime.attribute(
                    x_sample,
                    baselines=x_sample.new_zeros(x_sample.shape),
                    target=target_label,
                    feature_mask=feature_mask,
                    n_samples=int(args.lime_samples),
                    perturbations_per_eval=int(args.perturbations_per_eval),
                    return_input_shape=True,
                    show_progress=False,
                ).detach()
                per_method_abs_sums["lime"] += attr.abs().squeeze(0).cpu().numpy()
                per_method_signed_sums["lime"] += attr.squeeze(0).cpu().numpy()
                per_method_counts["lime"] += 1

        if "shap" in args.methods and len(X_surrogate_std) > 0:
            shap = KernelShap(model)
            for sample_idx, sample in enumerate(X_surrogate_std):
                x_sample = _float_tensor(sample[None, ...], device=device)
                attr = shap.attribute(
                    x_sample,
                    baselines=x_sample.new_zeros(x_sample.shape),
                    target=target_label,
                    feature_mask=feature_mask,
                    n_samples=int(args.shap_samples),
                    perturbations_per_eval=int(args.perturbations_per_eval),
                    return_input_shape=True,
                    show_progress=False,
                ).detach()
                per_method_abs_sums["shap"] += attr.abs().squeeze(0).cpu().numpy()
                per_method_signed_sums["shap"] += attr.squeeze(0).cpu().numpy()
                per_method_counts["shap"] += 1

        if "am" in args.methods:
            optimized_input, objective_history, probability_history = (
                _activation_maximization(
                    model=model,
                    input_shape=(1, run_data.X_raw.shape[1], run_data.X_raw.shape[2]),
                    target_label=target_label,
                    steps=int(args.am_steps),
                    lr=float(args.am_lr),
                    l2_coeff=float(args.am_l2),
                    tv_coeff=float(args.am_tv),
                    jitter_std=float(args.am_jitter_std),
                    clamp_abs=float(args.am_clamp_abs),
                    device=device,
                )
            )
            am_sum += optimized_input.squeeze(0).cpu().numpy()
            am_objective_sum += np.asarray(objective_history, dtype=np.float64)
            am_probability_sum += np.asarray(probability_history, dtype=np.float64)
            am_count += 1

    out_base = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "explainer_suite_results")
    )
    out_dir, version_meta = prepare_versioned_output_dir(
        base_output_dir=out_base,
        experiment_name="sst_campaign_analyze_explainer_suite",
        config=vars(args),
        disable_versioning=bool(args.disable_versioning),
    )
    print(f"Run output dir: {out_dir}")
    emit_run_output_dir(out_dir)
    print(f"Run version meta: {json.dumps(version_meta)}")

    selection_csv = out_dir / "fold_selection_summary.csv"
    pd.DataFrame(selection_rows).to_csv(selection_csv, index=False)
    summary: dict[str, Any] = {
        "source_run_dir": str(run_dir),
        "model_type": run_data.model_name,
        "device": str(device),
        "target_label": str(args.target_label),
        **p3_window.summary(),
        "roi_channels_requested": [str(x) for x in args.roi_channels],
        "roi_channels_resolved": [channel_names[i] for i in roi_indices],
        "folds": [int(p.name.split("_")[-1]) for p in run_data.fold_dirs],
        "selection_csv": str(selection_csv),
        "methods": {},
    }

    for method in args.methods:
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        if method in MAP_METHODS and per_method_counts.get(method, 0) > 0:
            mean_abs = per_method_abs_sums[method] / float(per_method_counts[method])
            mean_signed = per_method_signed_sums[method] / float(
                per_method_counts[method]
            )
            all_curve = mean_abs.mean(axis=0)
            roi_curve = mean_abs[roi_indices].mean(axis=0)
            pz_curve = mean_abs[pz_index]
            per_channel = mean_abs.mean(axis=1)
            top_idx = np.argsort(per_channel)[::-1][:10]
            heatmap_png = method_dir / f"{method}_mean_abs_heatmap.png"
            signed_heatmap_png = method_dir / f"{method}_mean_signed_heatmap.png"
            curve_png = method_dir / f"{method}_temporal_curve.png"
            curve_csv = method_dir / f"{method}_temporal_curve.csv"
            arrays_npz = method_dir / f"{method}_arrays.npz"
            plot_channel_time_heatmap(
                values=mean_abs,
                times_s=times_s,
                channel_names=channel_names,
                out_path=heatmap_png,
                title=f"{method.upper()} mean absolute attribution",
                p3_window=p3_window.window_s,
                colorbar_label="Mean |attribution|",
            )
            plot_channel_time_heatmap(
                values=mean_signed,
                times_s=times_s,
                channel_names=channel_names,
                out_path=signed_heatmap_png,
                title=f"{method.upper()} mean signed attribution",
                p3_window=p3_window.window_s,
                cmap="coolwarm",
                symmetric=True,
                colorbar_label="Mean attribution",
            )
            plot_relevance_curves(
                times_s=times_s,
                curves={
                    f"{method} all channels": all_curve,
                    f"{method} ROI": roi_curve,
                    f"{method} Pz": pz_curve,
                },
                out_path=curve_png,
                title=f"{method.upper()} temporal attribution",
                ylabel="Mean |attribution|",
                p3_window=p3_window.window_s,
            )
            pd.DataFrame(
                {
                    "time_s": times_s,
                    "all_channels": all_curve,
                    "roi": roi_curve,
                    "pz": pz_curve,
                }
            ).to_csv(curve_csv, index=False)
            np.savez_compressed(
                arrays_npz,
                mean_abs=mean_abs,
                mean_signed=mean_signed,
                times_s=times_s,
                roi_indices=np.asarray(roi_indices, dtype=np.int64),
                pz_index=np.asarray([pz_index], dtype=np.int64),
            )
            summary["methods"][method] = {
                "n_samples": int(per_method_counts[method]),
                "temporal_summary": summarize_time_curve(roi_curve, times_s, p3_mask),
                "top_channels": [
                    {
                        "channel": channel_names[int(i)],
                        "score": float(per_channel[int(i)]),
                    }
                    for i in top_idx
                ],
                "outputs": {
                    "heatmap_png": str(heatmap_png),
                    "signed_heatmap_png": str(signed_heatmap_png),
                    "curve_png": str(curve_png),
                    "curve_csv": str(curve_csv),
                    "arrays_npz": str(arrays_npz),
                },
            }
        elif method in TIME_ONLY_METHODS and time_curve_counts.get(method, 0) > 0:
            curve = time_curve_sums[method] / float(time_curve_counts[method])
            curve_png = method_dir / f"{method}_temporal_curve.png"
            curve_csv = method_dir / f"{method}_temporal_curve.csv"
            plot_relevance_curves(
                times_s=times_s,
                curves={f"{method} ROI proxy": curve},
                out_path=curve_png,
                title=f"{method.upper()} temporal relevance",
                ylabel="Mean relevance",
                p3_window=p3_window.window_s,
            )
            pd.DataFrame({"time_s": times_s, "curve": curve}).to_csv(
                curve_csv, index=False
            )
            summary["methods"][method] = {
                "n_samples": int(time_curve_counts[method]),
                "temporal_summary": summarize_time_curve(curve, times_s, p3_mask),
                "outputs": {"curve_png": str(curve_png), "curve_csv": str(curve_csv)},
            }
        elif method == "am" and am_count > 0:
            mean_am = am_sum / float(am_count)
            mean_objective = am_objective_sum / float(am_count)
            mean_probability = am_probability_sum / float(am_count)
            heatmap_png = method_dir / "activation_maximization_heatmap.png"
            curve_png = method_dir / "activation_maximization_curves.png"
            objective_png = method_dir / "activation_maximization_objective.png"
            probability_png = method_dir / "activation_maximization_probability.png"
            history_csv = method_dir / "activation_maximization_history.csv"
            arrays_npz = method_dir / "activation_maximization_arrays.npz"
            plot_channel_time_heatmap(
                values=mean_am,
                times_s=times_s,
                channel_names=channel_names,
                out_path=heatmap_png,
                title="Activation maximization prototype",
                p3_window=p3_window.window_s,
                cmap="coolwarm",
                symmetric=True,
                colorbar_label="Amplitude",
            )
            plot_relevance_curves(
                times_s=times_s,
                curves={
                    "prototype all channels": mean_am.mean(axis=0),
                    "prototype ROI": mean_am[roi_indices].mean(axis=0),
                    "prototype Pz": mean_am[pz_index],
                },
                out_path=curve_png,
                title="Activation maximization temporal curves",
                ylabel="Amplitude",
                p3_window=p3_window.window_s,
            )
            plot_scalar_history(
                steps=np.arange(1, len(mean_objective) + 1),
                values=mean_objective,
                out_path=objective_png,
                title="Activation maximization objective",
                xlabel="Step",
                ylabel="Objective",
            )
            plot_scalar_history(
                steps=np.arange(1, len(mean_probability) + 1),
                values=mean_probability,
                out_path=probability_png,
                title="Activation maximization target probability",
                xlabel="Step",
                ylabel="P(target)",
            )
            pd.DataFrame(
                {
                    "step": np.arange(1, len(mean_objective) + 1),
                    "objective": mean_objective,
                    "target_probability": mean_probability,
                }
            ).to_csv(history_csv, index=False)
            np.savez_compressed(
                arrays_npz,
                prototype=mean_am,
                objective=mean_objective,
                target_probability=mean_probability,
                times_s=times_s,
            )
            summary["methods"][method] = {
                "n_samples": int(am_count),
                "temporal_summary": summarize_time_curve(
                    mean_am[roi_indices].mean(axis=0), times_s, p3_mask
                ),
                "outputs": {
                    "heatmap_png": str(heatmap_png),
                    "curve_png": str(curve_png),
                    "objective_png": str(objective_png),
                    "probability_png": str(probability_png),
                    "history_csv": str(history_csv),
                    "arrays_npz": str(arrays_npz),
                },
            }

    summary_json = out_dir / "explainer_suite_summary.json"
    save_json(summary_json, summary)
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
